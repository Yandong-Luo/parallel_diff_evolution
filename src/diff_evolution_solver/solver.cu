#include "diff_evolution_solver/solver.cuh"
#include "diff_evolution_solver/decoder.cuh"
#include "diff_evolution_solver/debug.cuh"
#include "diff_evolution_solver/evolve.cuh"

namespace cudaprocess{

void CudaDiffEvolveSolver::MallocSetup(){
    CHECK_CUDA(cudaSetDevice(gpu_device_));

    // GPU Device
    CHECK_CUDA(cudaMalloc(&decoder_, sizeof(CudaProblemDecoder)));
    CHECK_CUDA(cudaMalloc(&evolve_data_, sizeof(CudaEvolveData)));
    CHECK_CUDA(cudaMalloc(&new_cluster_data_, sizeof(CudaParamClusterData<64>)));
    CHECK_CUDA(cudaMalloc(&old_cluster_data_, sizeof(CudaParamClusterData<192>)));
    CHECK_CUDA(cudaMalloc(&new_cluster_vec_, sizeof(CudaVector<CudaParamIndividual, CUDA_SOLVER_POP_SIZE>)));
    CHECK_CUDA(cudaMalloc(&device_solver_input_, sizeof(CudaSolverInput)));
    CHECK_CUDA(cudaMalloc(&evaluator_, sizeof(ProblemEvaluator)));
    // CHECK_CUDA(cudaMalloc(&new_param, sizeof(CudaParamClusterData<192>)));

    // CPU Host
    CHECK_CUDA(cudaHostAlloc(&host_decoder_, sizeof(CudaProblemDecoder), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&host_evolve_data_, sizeof(CudaEvolveData), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&host_new_cluster_data_, sizeof(CudaParamClusterData<64>), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&host_old_cluster_data_, sizeof(CudaParamClusterData<192>), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&host_solver_input_, sizeof(CudaSolverInput), cudaHostAllocDefault));

    cuda_utils_ = std::make_shared<CudaUtil>();

    cudamalloc_flag = true;
}

void CudaDiffEvolveSolver::InitDiffEvolveParam(float best, float d_best, float min_best, float diff, float d_diff, float min_diff, float scale_f, float prob_crossover){
    best_ = best;
    d_best_ = d_best;
    min_best_ = min_best;
    diff_ = diff;
    d_diff_ = d_diff;
    min_diff_ = min_diff;
    
    lshade_param_.scale_f = lshade_param_.scale_f1 = scale_f;
    lshade_param_.Cr = prob_crossover;

}

__global__ void InitCudaEvolveData(CudaEvolveData* data, CudaParamClusterData<192>* old_cluster_data, int pop_size){
    int idx = threadIdx.x;
    if (idx == 0) {
        data->best = 0.;
        data->lshade_param.scale_f = data->lshade_param.scale_f1 = 0.6;
        data->lshade_param.Cr = 0.9;
        data->new_cluster_vec->len = pop_size;
        old_cluster_data->len = pop_size;
    }
    if (idx < pop_size){
        // initial the each parameter in old_cluster 
        for (int i = 0; i < CUDA_PARAM_MAX_SIZE; ++i){
            old_cluster_data->all_param[(idx + pop_size) * CUDA_PARAM_MAX_SIZE + i] = 0.f;
        }
        old_cluster_data->fitness[idx + pop_size] = CUDA_MAX_FLOAT;
    }
}

void CudaDiffEvolveSolver::SetBoundary(ProblemEvaluator* evaluator){
    for (int i = 0; i < host_decoder_->con_var_dims_; ++i){
        host_evolve_data_->upper_bound[i] = host_upper_bound_[i] = evaluator->con_upper_bound[i];
        host_evolve_data_->lower_bound[i] = host_lower_bound_[i] = evaluator->con_lower_bound[i];
    }

    for (int i = host_decoder_->con_var_dims_, j = 0; i < host_decoder_->con_var_dims_ + host_decoder_->int_var_dims_; ++i, ++j){
        host_evolve_data_->upper_bound[i] = host_upper_bound_[i] = evaluator->int_upper_bound[j];
        host_evolve_data_->lower_bound[i] = host_lower_bound_[i] = evaluator->int_lower_bound[j];
    }
    // for(int i = 0; i < host_decoder_->dims_; ++i){
    //     printf("index:%d lower bound:%f, upper bound:%f\n",i, host_evolve_data_->lower_bound[i], host_evolve_data_->upper_bound[i]);
    // }
}

/**
 * CudaEvolveData* ----> CudaParamClusterData<T> *
 */
__global__ void UpdateClusterDataBasedEvolve(CudaEvolveData* evolve_data, CudaParamClusterData<64>* new_cluster_data, int num_last_potential_sol){
    int idx = blockIdx.x;
    if (idx >= num_last_potential_sol)   return;
    ConvertCudaParam<64>(new_cluster_data, &evolve_data->last_potential_sol.data[idx], idx, threadIdx.x);
}

/**
 * CudaParamClusterData ----> CudaParamIndividual * as output
 */
template <int T>
__global__ void UpdateVecParamBasedClusterData(CudaParamIndividual *output, CudaParamClusterData<T> *new_cluster_data){
    ConvertCudaParamRevert<T>(new_cluster_data, &output[blockIdx.x], blockIdx.x, threadIdx.x);
}

/**
 * CudaParamClusterData<T> * ---->  CudaEvolveData* 
 */
__global__ void UpdateEvolveWarmStartBasedClusterData(CudaEvolveData *evolve_data, CudaParamClusterData<192> *old_param){
    ConvertCudaParamRevert<192>(old_param, &evolve_data->warm_start, 0, threadIdx.x);
}

__global__ void SaveNewParamAsOldParam(CudaParamClusterData<64> *new_param, CudaParamClusterData<192> *old_param, int left_bound, int right_bound, int bias){
    int sol_id = blockIdx.x;
    if (sol_id < left_bound || sol_id >= right_bound)   return;
    ConvertCudaParamBetweenClusters<64, 192>(new_param, old_param, sol_id, sol_id + bias, threadIdx.x);
}

__global__ void GenerativeRandSolNearBest(CudaEvolveData *evolve, CudaParamClusterData<64> *new_param, float *uniform_data, int rand_idx, float delta_con, float delta_int, int size){
    int sol_id = blockIdx.x;
    int param_id = threadIdx.x;

    if (sol_id == 0 || sol_id >= size)  return;
    float upper_bound = evolve->upper_bound[param_id];
    float lower_bound = evolve->lower_bound[param_id];

    if (param_id < evolve->con_var_dims){
        float rand_range = (upper_bound - lower_bound) * delta_con;

        // based on rand_range update the boundary
        upper_bound = min(upper_bound, new_param->all_param[param_id] + rand_range);
        lower_bound = max(lower_bound, new_param->all_param[param_id] - rand_range);
        
        // based on new boundary near parameter to generate the new parameter
        new_param->all_param[sol_id * CUDA_PARAM_MAX_SIZE + param_id] = lower_bound + uniform_data[64 * 100 * CUDA_PARAM_MAX_SIZE + sol_id * CUDA_SOLVER_POP_SIZE + rand_idx + param_id] * (upper_bound - lower_bound);
    }
    else if(param_id < evolve->int_var_dims){
        float rand_range = (upper_bound - lower_bound) * delta_int;

        // based on rand_range update the boundary
        upper_bound = min(upper_bound, new_param->all_param[param_id] + rand_range);
        lower_bound = max(lower_bound, new_param->all_param[param_id] - rand_range);
        
        // based on new boundary near parameter to generate the new parameter
        new_param->all_param[sol_id * CUDA_PARAM_MAX_SIZE + param_id] = lower_bound + uniform_data[64 * 100 * CUDA_PARAM_MAX_SIZE + sol_id * CUDA_SOLVER_POP_SIZE + rand_idx + param_id] * (upper_bound - lower_bound);
    }
}


void CudaDiffEvolveSolver::WarmStart(ProblemEvaluator* evaluator, CudaParamIndividual* output_sol){
    InitParameter<<<1, default_pop_size_, 0, cuda_utils_->streams_[0]>>>(decoder_, evolve_data_, default_pop_size_, new_cluster_data_, old_cluster_data_, random_center_->uniform_data_);
    CHECK_CUDA(cudaStreamSynchronize(cuda_utils_->streams_[0]));
    if(last_potential_sol_.len > 0){
        // int half_pop_size = default_pop_size_ >> 1;
        int quad_pop_size = default_pop_size_ >> 2;
        // one cluster generate one solution, each cluster works on one block. We need to generate quad_pop_size new solutions based on last potential solution, so init the new cluster in quad_pop_size grid.
        UpdateClusterDataBasedEvolve<<<quad_pop_size, CUDA_PARAM_MAX_SIZE, 0, cuda_utils_->streams_[0]>>>(evolve_data_, new_cluster_data_, last_potential_sol_.len);
    }
    UpdateVecParamBasedClusterData<64><<<default_pop_size_, 16, 0, cuda_utils_->streams_[0]>>>(new_cluster_vec_->data, new_cluster_data_);
    CHECK_CUDA(cudaStreamSynchronize(cuda_utils_->streams_[0]));
    int cet = 10;
    // Update the output param based on warm start.
    CHECK_CUDA(cudaMemcpyAsync(output_sol, &new_cluster_vec_->data[cet], sizeof(CudaParamIndividual), cudaMemcpyDeviceToHost, cuda_utils_->streams_[0]));
    CHECK_CUDA(cudaStreamSynchronize(cuda_utils_->streams_[0]));
}

/**
 * One Block with 64 thread to evaluate different 
 */
// template<int T>
// __global__ void MainEvaluation(CudaEvolveData *evolve, CudaParamClusterData<T> *cluster_data){
//     float result = 0.0;
//     int idx = threadIdx.x;
    
//     if (idx >= T) return;
    
//     for(int i = 0; i < evolve->num_constraint; ++i){
//         float constraint_val = 0.0;
//         for(int j = 0; j < evolve->num_constraint_variable; ++j){
//             if (j == evolve->num_constraint_variable - 1)   constraint_val += evolve->constraint_para[i][j];
//             else{
//                 constraint_val += cluster_data->all_param[idx * CUDA_PARAM_MAX_SIZE + j] * evolve->constraint_para[i][j];
//             }
//         }
//         if (idx == 0) {
//             printf("Individual 0 - Constraint %d value: %f\n", i, constraint_val);
//         }
//         if (constraint_val < 0){
//             cluster_data->fitness[idx] = CUDA_MAX_FLOAT;
//             printf("thread id:%d setting fitness to MAX_FLOAT\n", idx);
//             return;
//         }
//     }

//     // objective function
//     for(int i = 0; i < evolve->num_objective_param; ++i){
//         result += evolve->objective_param[i] * cluster_data->all_param[idx * CUDA_PARAM_MAX_SIZE + i];
//     }

//     cluster_data->fitness[idx] = result;
//     printf("thread id:%d setting fitness to %f\n", idx, result);
// }

template<int T>
__device__ void EvaluateIndividual(CudaEvolveData *evolve, CudaParamClusterData<T> *cluster_data, int sol_id){
    float result = 0.0;
    for(int i = 0; i < evolve->num_constraint; ++i){
        float constraint_val = 0.0;
        for(int j = 0; j < evolve->num_constraint_variable; ++j){
            if (j == evolve->num_constraint_variable - 1)   constraint_val += evolve->constraint_para[i][j];
            else{
                constraint_val += cluster_data->all_param[sol_id * CUDA_PARAM_MAX_SIZE + j] * evolve->constraint_para[i][j];
            }
        }
        // if (sol_id == 0) {
        //     printf("Individual 0 - Constraint %d value: %f\n", i, constraint_val);
        // }
        if (constraint_val < 0){
            cluster_data->fitness[sol_id] = CUDA_MAX_FLOAT;
            printf("thread id:%d setting fitness to MAX_FLOAT\n", sol_id);
            return;
        }
    }

    // objective function
    for(int i = 0; i < evolve->num_objective_param; ++i){
        result += evolve->objective_param[i] * cluster_data->all_param[sol_id * CUDA_PARAM_MAX_SIZE + i];
    }

    cluster_data->fitness[sol_id] = result;
    printf("thread id:%d setting fitness to %f\n", sol_id, result);
}

template <int T = 64, int PARA_SIZE>
__global__ void ParaFindMax2(CudaParamClusterData<PARA_SIZE> *a) {
  __shared__ int idx_list[4];
  __shared__ float value_list[4];
  // if (threadIdx.x > 64) return;
  float value = a->fitness[threadIdx.x];
  int idx = threadIdx.x;

  float tmp_f;
  int tmp_idx;
  tmp_f = __shfl_down_sync(0xffffffff, value, 16);
  tmp_idx = __shfl_down_sync(0xffffffff, idx, 16);
  if (tmp_f < value) {
    value = tmp_f;
    idx = tmp_idx;
  }
  tmp_f = __shfl_down_sync(0xffffffff, value, 8);
  tmp_idx = __shfl_down_sync(0xffffffff, idx, 8);
  if (tmp_f < value) {
    value = tmp_f;
    idx = tmp_idx;
  }
  tmp_f = __shfl_down_sync(0xffffffff, value, 4);
  tmp_idx = __shfl_down_sync(0xffffffff, idx, 4);
  if (tmp_f < value) {
    value = tmp_f;
    idx = tmp_idx;
  }
  tmp_f = __shfl_down_sync(0xffffffff, value, 2);
  tmp_idx = __shfl_down_sync(0xffffffff, idx, 2);
  if (tmp_f < value) {
    value = tmp_f;
    idx = tmp_idx;
  }
  tmp_f = __shfl_down_sync(0xffffffff, value, 1);
  tmp_idx = __shfl_down_sync(0xffffffff, idx, 1);
  if (tmp_f < value) {
    value = tmp_f;
    idx = tmp_idx;
  }

  if ((threadIdx.x & 31) == 0) {
    idx_list[threadIdx.x >> 5] = idx;
    value_list[threadIdx.x >> 5] = value;
  }
  __syncthreads();

  if (T == 128) {
    if (threadIdx.x < 4) {
      value = value_list[threadIdx.x];
      idx = idx_list[threadIdx.x];
      tmp_f = __shfl_down_sync(0x0000000f, value, 2);
      tmp_idx = __shfl_down_sync(0x0000000f, idx, 2);
      if (tmp_f < value) {
        value = tmp_f;
        idx = tmp_idx;
      }
      tmp_f = __shfl_down_sync(0x0000000f, value, 1);
      tmp_idx = __shfl_down_sync(0x0000000f, idx, 1);
      if (tmp_f < value) {
        value = tmp_f;
        idx = tmp_idx;
      }
    }
  } else if (T == 64) {
    if (threadIdx.x < 2) {
      value = value_list[threadIdx.x];
      idx = idx_list[threadIdx.x];
      tmp_f = __shfl_down_sync(0x00000003, value, 1);
      tmp_idx = __shfl_down_sync(0x00000003, idx, 1);
      if (tmp_f < value) {
        value = tmp_f;
        idx = tmp_idx;
      }
    }
  }

  idx = __shfl_sync(0x0000ffff, idx, 0);
  if (threadIdx.x < 16) {
    float para = a->all_param[idx * CUDA_PARAM_MAX_SIZE + threadIdx.x];
    a->all_param[idx * CUDA_PARAM_MAX_SIZE + threadIdx.x] = a->all_param[threadIdx.x];
    a->all_param[threadIdx.x] = para;
    if (threadIdx.x == 0) {
      float f = a->fitness[idx];
      a->fitness[idx] = a->fitness[0];
      a->fitness[0] = f;
    }
  }
}


template<int T>
__global__ void MainEvaluation(CudaEvolveData *evolve, CudaParamClusterData<T> *cluster_data){
    EvaluateIndividual(evolve, cluster_data, blockIdx.x);
}

void CudaDiffEvolveSolver::Evaluation(int size){
    MainEvaluation<64><<<size, 1, 0, cuda_utils_->streams_[0]>>>(evolve_data_, new_cluster_data_);
    // cudaStreamSynchronize(cuda_utils_->streams_[0]); 
}

void CudaDiffEvolveSolver::InitSolver(int gpu_device, CudaRandomCenter *random_center, ProblemEvaluator* host_evaluator, CudaParamIndividual* output_sol, const CudaVector<CudaParamIndividual, CUDA_MAX_POTENTIAL_SOLUTION> *last_potential_sol){
    gpu_device_ = gpu_device;
    random_center_ = random_center;

    CHECK_CUDA(cudaSetDevice(gpu_device_));
    if (DEBUG_PRINT_FLAG) printf("CUDA SET DEVICE\n");

    MallocSetup();

    con_var_dims_ = host_evaluator->num_con_variable;
    int_var_dims_ = host_evaluator->num_int_variable;
    dims_ = host_evaluator->num_con_variable + host_evaluator->num_int_variable;

    InitDiffEvolveParam();
    if (DEBUG_PRINT_FLAG) printf("INIT PARAM FOR DE\n");
    
    // Initial host decoder
    host_decoder_->con_var_dims_ = host_evaluator->num_con_variable;
    host_decoder_->int_var_dims_ = host_evaluator->num_int_variable;
    host_decoder_->dims_ = host_evaluator->num_con_variable + host_evaluator->num_int_variable;

    // Initial evolve data
    host_evolve_data_->best = best_;
    host_evolve_data_->new_cluster_vec = new_cluster_vec_;
    host_evolve_data_->int_var_dims = int_var_dims_;
    host_evolve_data_->con_var_dims = con_var_dims_;
    host_evolve_data_->dims = int_var_dims_ + con_var_dims_;
    
    // constraint
    host_evolve_data_->num_constraint = host_evaluator->num_constraint;
    host_evolve_data_->num_constraint_variable = host_evaluator->num_constraint_variable;
    for (int i = 0; i < host_evaluator->num_constraint; ++i){
        for(int j = 0; j < host_evaluator->num_constraint_variable; ++j){
            host_evolve_data_->constraint_para[i][j] = host_evaluator->constraint_param[i][j];
        }
    }

    // objective
    host_evolve_data_->num_objective_param = host_evaluator->num_objective_param;
    for (int i = 0; i < host_evaluator->num_objective_param; ++i){
        host_evolve_data_->objective_param[i] = host_evaluator->objective_param[i];
    }
    // host_evolve_data_->lower_bound = host_lower_bound_;
    // host_evolve_data_->upper_bound = host_upper_bound_;

    SetBoundary(host_evaluator);

    if (last_potential_sol != nullptr){
        for(int i = 0; i < last_potential_sol->len; ++i){
            memcpy(&last_potential_sol_.data[last_potential_sol_.len], &last_potential_sol->data[i], sizeof(CudaParamIndividual));
            for (int j = 0; j < dims_; j++) {
                last_potential_sol_.data[last_potential_sol_.len].param[j] = std::max(last_potential_sol_.data[last_potential_sol_.len].param[j], host_lower_bound_[j]);
                last_potential_sol_.data[last_potential_sol_.len].param[j] = std::min(last_potential_sol_.data[last_potential_sol_.len].param[j], host_upper_bound_[j]);
            }
            last_potential_sol_.len++;
        }
    }

    host_evolve_data_->last_potential_sol = last_potential_sol_;

    host_solver_input_->evaluator = host_evaluator;
    host_solver_input_->new_param = new_cluster_data_;
    host_solver_input_->old_param = old_cluster_data_;

    if (DEBUG_PRINT_FLAG) printf("START MEMORY ASYNC\n");

    // Host --> GPU device
    CHECK_CUDA(cudaMemcpyAsync(evolve_data_, host_evolve_data_, sizeof(CudaEvolveData), cudaMemcpyHostToDevice, cuda_utils_->streams_[0]));
    CHECK_CUDA(cudaMemcpyAsync(decoder_, host_decoder_, sizeof(CudaProblemDecoder), cudaMemcpyHostToDevice, cuda_utils_->streams_[0]));
    CHECK_CUDA(cudaMemcpyAsync(evaluator_, host_evaluator, sizeof(ProblemEvaluator), cudaMemcpyHostToDevice, cuda_utils_->streams_[0]));
    CHECK_CUDA(cudaMemcpyAsync(device_solver_input_, host_solver_input_, sizeof(CudaSolverInput), cudaMemcpyHostToDevice, cuda_utils_->streams_[0]));
    CHECK_CUDA(cudaStreamSynchronize(cuda_utils_->streams_[0]));
    // if (last_sol == nullptr){
    //     CHECK_CUDA(cudaStreamSynchronize(cuda_utils_->streams_[0]));
    // }

    if (DEBUG_PRINT_FLAG) printf("MEMORY ASYNC SUBMIT\n");

    InitCudaEvolveData<<<1, CUDA_SOLVER_POP_SIZE, 0, cuda_utils_->streams_[0]>>>(evolve_data_, old_cluster_data_, default_pop_size_);

    WarmStart(host_evaluator, output_sol);

    Evaluation(CUDA_SOLVER_POP_SIZE);

    ParaFindMax2<CUDA_SOLVER_POP_SIZE, 64><<<1, default_pop_size_, 0, cuda_utils_->streams_[0]>>>(new_cluster_data_);

    // convert the parameter from warm start to old parameter
    SaveNewParamAsOldParam<<<default_pop_size_, CUDA_PARAM_MAX_SIZE, 0, cuda_utils_->streams_[0]>>>(new_cluster_data_, old_cluster_data_, 0, default_pop_size_, 0);

    // Based on all old parameter to update the warm start of evolve data
    // 将 old_cluster_data_<192> 中索引为0的数据提取出来,填充到evolve data单个CudaParamIndividual结构中,记为warm start。索引为0的解是warm start过程中最优的
    UpdateEvolveWarmStartBasedClusterData<<<1, CUDA_PARAM_MAX_SIZE, 0, cuda_utils_->streams_[0]>>>(evolve_data_, old_cluster_data_);

    if (DEBUG_PRINT_FLAG){
        CHECK_CUDA(cudaStreamSynchronize(cuda_utils_->streams_[0]));
        CHECK_CUDA(cudaMemcpyAsync(host_new_cluster_data_, new_cluster_data_, sizeof(CudaParamClusterData<64>), cudaMemcpyDeviceToHost, cuda_utils_->streams_[0]));
        PrintClusterData(host_new_cluster_data_);
        // CHECK_CUDA(cudaMemcpyAsync(host_old_cluster_data_, old_cluster_data_, sizeof(CudaParamClusterData<192>), cudaMemcpyDeviceToHost, cuda_utils_->streams_[0]));
        // PrintClusterData(host_old_cluster_data_);

        // CHECK_CUDA(cudaMemcpyAsync(host_evolve_data_, evolve_data_, sizeof(CudaEvolveData), cudaMemcpyDeviceToHost, cuda_utils_->streams_[0]));
        // printf("CUDA_MAX_FLOAT %f\n", CUDA_MAX_FLOAT);
    }
}

__global__ void AddWarmStartResultForSolver(CudaEvolveData *evolve, CudaParamClusterData<64> *new_param){
    ConvertCudaParam<64>(new_param, &evolve->warm_start, blockIdx.x, threadIdx.x);
}

void CudaDiffEvolveSolver::Solver(){
    init_pop_size_ = default_pop_size_;
    pop_size_ = default_pop_size_;

    InitCudaEvolveData<<<1, default_pop_size_, 0, cuda_utils_->streams_[0]>>>(evolve_data_, old_cluster_data_, default_pop_size_);

    InitParameter<<<1, default_pop_size_, 0, cuda_utils_->streams_[0]>>>(decoder_, evolve_data_, default_pop_size_, new_cluster_data_, old_cluster_data_, random_center_->uniform_data_);

    AddWarmStartResultForSolver<<<1, CUDA_PARAM_MAX_SIZE, 0, cuda_utils_->streams_[0]>>>(evolve_data_, new_cluster_data_);

    // based on warm start result to generate 
    GenerativeRandSolNearBest<<<default_pop_size_, CUDA_PARAM_MAX_SIZE, 0, cuda_utils_->streams_[0]>>>(evolve_data_, new_cluster_data_, random_center_->uniform_data_, 16, 0.1, 0.1, default_pop_size_);

    Evaluation(CUDA_SOLVER_POP_SIZE);

    ParaFindMax2<CUDA_SOLVER_POP_SIZE, 64><<<1, default_pop_size_, 0, cuda_utils_->streams_[0]>>>(new_cluster_data_);
}

CudaDiffEvolveSolver::~CudaDiffEvolveSolver(){
    if (cudamalloc_flag){
        // GPU device
        CHECK_CUDA(cudaFree(decoder_));
        CHECK_CUDA(cudaFree(evolve_data_));
        CHECK_CUDA(cudaFree(new_cluster_data_));
        CHECK_CUDA(cudaFree(old_cluster_data_));
        CHECK_CUDA(cudaFree(new_cluster_vec_));
        CHECK_CUDA(cudaFree(evaluator_));

        // CPU host
        CHECK_CUDA(cudaFreeHost(host_decoder_));
        CHECK_CUDA(cudaFreeHost(host_evolve_data_));
        CHECK_CUDA(cudaFreeHost(host_new_cluster_data_));
        CHECK_CUDA(cudaFreeHost(host_old_cluster_data_));
    }
    
}

}