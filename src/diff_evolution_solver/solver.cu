#include "diff_evolution_solver/solver.cuh"
#include "diff_evolution_solver/decoder.cuh"
#include "diff_evolution_solver/debug.cuh"

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

    for (int i = host_decoder_->con_var_dims_, j = 0; i < host_decoder_->con_var_dims_ + host_decoder_->bin_var_dims_; ++i, ++j){
        host_evolve_data_->upper_bound[i] = host_upper_bound_[i] = evaluator->binary_upper_bound[j];
        host_evolve_data_->lower_bound[i] = host_lower_bound_[i] = evaluator->binary_lower_bound[j];
    }
    
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


void CudaDiffEvolveSolver::WarmStart(ProblemEvaluator* evaluator, CudaParamIndividual* output_sol){
    InitParameter<<<1, default_pop_size_, 0, cuda_utils_->streams_[0]>>>(decoder_, evolve_data_, default_pop_size_, new_cluster_data_, old_cluster_data_, random_center_->uniform_data_);

    if(last_potential_sol_.len > 0){
        // int half_pop_size = default_pop_size_ >> 1;
        int quad_pop_size = default_pop_size_ >> 2;
        // one cluster generate one solution, each cluster works on one block. We need to generate quad_pop_size new solutions based on last potential solution, so init the new cluster in quad_pop_size grid.
        UpdateClusterDataBasedEvolve<<<quad_pop_size, CUDA_PARAM_MAX_SIZE, 0, cuda_utils_->streams_[0]>>>(evolve_data_, new_cluster_data_, last_potential_sol_.len);
    }
    UpdateVecParamBasedClusterData<64><<<default_pop_size_, 16, 0, cuda_utils_->streams_[0]>>>(new_cluster_vec_->data, new_cluster_data_);

    int cet = 10;
    // Update the output param based on warm start.
    CHECK_CUDA(cudaMemcpyAsync(output_sol, &new_cluster_vec_->data[cet], sizeof(CudaParamIndividual), cudaMemcpyDeviceToHost, cuda_utils_->streams_[0]));
}

void CudaDiffEvolveSolver::InitSolver(int gpu_device, CudaRandomCenter *random_center, ProblemEvaluator* host_evaluator, CudaParamIndividual* output_sol, const CudaVector<CudaParamIndividual, CUDA_MAX_POTENTIAL_SOLUTION> *last_potential_sol){
    gpu_device_ = gpu_device;
    random_center_ = random_center;

    CHECK_CUDA(cudaSetDevice(gpu_device_));
    if (DEBUG_PRINT_FLAG) printf("CUDA SET DEVICE\n");

    MallocSetup();

    con_var_dims_ = host_evaluator->num_con_variable;
    bin_var_dims_ = host_evaluator->num_binary_variable;
    dims_ = host_evaluator->num_con_variable + host_evaluator->num_binary_variable;

    InitDiffEvolveParam();
    if (DEBUG_PRINT_FLAG) printf("INIT PARAM FOR DE\n");
    
    // Initial host decoder
    host_decoder_->con_var_dims_ = host_evaluator->num_con_variable;
    host_decoder_->bin_var_dims_ = host_evaluator->num_binary_variable;
    host_decoder_->dims_ = host_evaluator->num_con_variable + host_evaluator->num_binary_variable;

    // Initial evolve data
    host_evolve_data_->best = best_;
    host_evolve_data_->new_cluster_vec = new_cluster_vec_;
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

    // convert the parameter from warm start to old parameter
    SaveNewParamAsOldParam<<<default_pop_size_, CUDA_PARAM_MAX_SIZE, 0, cuda_utils_->streams_[0]>>>(new_cluster_data_, old_cluster_data_, 0, default_pop_size_, 0);

    // Based on all old parameter to update the warm start of evolve data
    // 将 old_cluster_data_<192> 中索引为0的数据提取出来,填充到evolve data单个CudaParamIndividual结构中,记为warm start。索引为0的解是warm start过程中最优的
    UpdateEvolveWarmStartBasedClusterData<<<1, CUDA_PARAM_MAX_SIZE, 0, cuda_utils_->streams_[0]>>>(evolve_data_, old_cluster_data_);

    if (DEBUG_PRINT_FLAG){
        CHECK_CUDA(cudaStreamSynchronize(cuda_utils_->streams_[0]));
        CHECK_CUDA(cudaMemcpyAsync(host_new_cluster_data_, new_cluster_data_, sizeof(CudaParamClusterData<64>), cudaMemcpyDeviceToHost, cuda_utils_->streams_[0]));
        PrintClusterData(host_new_cluster_data_);
        CHECK_CUDA(cudaMemcpyAsync(host_old_cluster_data_, old_cluster_data_, sizeof(CudaParamClusterData<192>), cudaMemcpyDeviceToHost, cuda_utils_->streams_[0]));
        PrintClusterData(host_old_cluster_data_);

        CHECK_CUDA(cudaMemcpyAsync(host_evolve_data_, evolve_data_, sizeof(CudaEvolveData), cudaMemcpyDeviceToHost, cuda_utils_->streams_[0]));
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