#include "diff_evolution_solver/solver.cuh"
#include "diff_evolution_solver/decoder.cuh"
#include "diff_evolution_solver/debug.cuh"
#include "diff_evolution_solver/evolve.cuh"
#include "diff_evolution_solver/evaluate.cuh"
#include "utils/utils_fun.cuh"

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

void CudaDiffEvolveSolver::InitDiffEvolveParam(float top, float d_top, float min_top, float diff, float d_diff, float min_diff, float scale_f, float prob_crossover){
    top_ = top;
    d_top_ = d_top;
    min_top_ = min_top;
    diff_ = diff;
    d_diff_ = d_diff;
    min_diff_ = min_diff;
    
    lshade_param_.scale_f = lshade_param_.scale_f1 = scale_f;
    lshade_param_.Cr = prob_crossover;

}

__global__ void InitCudaEvolveData(CudaEvolveData* evolve, CudaParamClusterData<192>* old_cluster_data, int pop_size){
    int idx = threadIdx.x;
    if (idx == 0) {
        evolve->top_ratio = 0.;
        evolve->hist_lshade_param.scale_f = evolve->hist_lshade_param.scale_f1 = 0.6;
        evolve->hist_lshade_param.Cr = 0.9;
        evolve->new_cluster_vec->len = pop_size;
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
        // one cluster generate one solution, each cluster works on one block. 
        // We need to generate quad_pop_size new solutions based on last potential solution, so init the new cluster in quad_pop_size grid.
        UpdateClusterDataBasedEvolve<<<quad_pop_size, CUDA_PARAM_MAX_SIZE, 0, cuda_utils_->streams_[0]>>>(evolve_data_, new_cluster_data_, last_potential_sol_.len);
    }
    UpdateVecParamBasedClusterData<64><<<default_pop_size_, 16, 0, cuda_utils_->streams_[0]>>>(new_cluster_vec_->data, new_cluster_data_);
    CHECK_CUDA(cudaStreamSynchronize(cuda_utils_->streams_[0]));
    // int cet = 10;
    // Update the output param based on warm start.
    // CHECK_CUDA(cudaMemcpyAsync(output_sol, &new_cluster_vec_->data[cet], sizeof(CudaParamIndividual), cudaMemcpyDeviceToHost, cuda_utils_->streams_[0]));

    // Evaluate random solutions or potential solutions in warmstart
    Evaluation(CUDA_SOLVER_POP_SIZE);

    // SortParamBasedBitonic<64><<<16, 64, 0, cuda_utils_->streams_[0]>>>(new_cluster_data_->all_param, new_cluster_data_->fitness);

    // Find the best solution among the random solutions or potential solutions in warmstart and put it in the first place
    ParaFindMax2<CUDA_SOLVER_POP_SIZE, 64><<<1, default_pop_size_, 0, cuda_utils_->streams_[0]>>>(new_cluster_data_);

    // based on warm start result to generate random solution. Further improve the quality of the initial population
    GenerativeRandSolNearBest<<<default_pop_size_, CUDA_PARAM_MAX_SIZE, 0, cuda_utils_->streams_[0]>>>(evolve_data_, new_cluster_data_, random_center_->uniform_data_, 16, 0.1, 0.1, default_pop_size_);

    // convert the parameter from warm start to old parameter
    SaveNewParamAsOldParam<<<default_pop_size_, CUDA_PARAM_MAX_SIZE, 0, cuda_utils_->streams_[0]>>>(new_cluster_data_, old_cluster_data_, 0, default_pop_size_, 0);

    // Based on all old parameter to update the warm start of evolve data
    // 将 old_cluster_data_<192> 中索引为0的数据提取出来,填充到evolve data单个CudaParamIndividual结构中,记为warm start。索引为0的解是warm start过程中最优的
    UpdateEvolveWarmStartBasedClusterData<<<1, CUDA_PARAM_MAX_SIZE, 0, cuda_utils_->streams_[0]>>>(evolve_data_, old_cluster_data_);

    CHECK_CUDA(cudaStreamSynchronize(cuda_utils_->streams_[0]));
}


template<int T>
__global__ void MainEvaluation(CudaEvolveData *evolve, CudaParamClusterData<T> *cluster_data){
    DynamicEvaluation2(evolve, cluster_data, evolve->lambda);
    
}

void CudaDiffEvolveSolver::Evaluation(int size){
    ConvertClusterToMatrix<64><<<size, dims_, 0, cuda_utils_->streams_[0]>>>(evolve_data_, new_cluster_data_, param_matrix);
    cudaStreamSynchronize(cuda_utils_->streams_[0]); 
    cudaMemcpy(host_param_matrix, param_matrix, dims_ * size * sizeof(float), cudaMemcpyDeviceToHost);

    printf("check param_matrix:\n");
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < dims_; ++j) {
            printf("row:%d col:%d host_param_matrix:%f ", i, j, host_param_matrix[i * dims_ + j]);
        }
        printf("\n");
    }

    // float alpha = 1;
    // float beta = 1;

    // objective part
    // example:
    // param_matrix: pop_size x dims (64 x 3)
    // objective_matrix: 3 x 1
    // obj_constant_matrix: 1 x 1
    // cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, row_obj, 1, col_obj, &alpha, param_matrix, 64, objective_matrix, 3, &beta, obj_constant_matrix, 1);

    // MainEvaluation<64><<<1, size, 0, cuda_utils_->streams_[0]>>>(evolve_data_, new_cluster_data_);
    // MatrixConstraint<64><<<1, size, threadIdx.x * evolve_data_->dims, cuda_utils_->streams_[0]>>>(handle, evolve_data_, new_cluster_data_, evolve_data_->lambda);
    cudaStreamSynchronize(cuda_utils_->streams_[0]); 
    // printf("================================================\n");
    // DynamicEvaluation<<<1, size, 0, cuda_utils_->streams_[0]>>>(evolve_data_, new_cluster_data_, evolve_data_->lambda);
}

void CudaDiffEvolveSolver::Evolution(int epoch, CudaEvolveType search_type){
    DuplicateBestAndReorganize<<<CUDA_PARAM_MAX_SIZE, 192, 0, cuda_utils_->streams_[0]>>>(epoch, old_cluster_data_, 2);
    CudaEvolveProcess<<<default_pop_size_, CUDA_PARAM_MAX_SIZE, 0, cuda_utils_->streams_[0]>>>(epoch, old_cluster_data_, new_cluster_data_, random_center_->uniform_data_, random_center_->normal_data_, evolve_data_, default_pop_size_, true);
    Evaluation(default_pop_size_);

    // CHECK_CUDA(cudaStreamSynchronize(cuda_utils_->streams_[0]));
    UpdateParameter<64><<<CUDA_PARAM_MAX_SIZE, 128, 0, cuda_utils_->streams_[0]>>>(epoch, evolve_data_, new_cluster_data_, old_cluster_data_);
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

    printf("??????????????????");
    
    // Initial host decoder
    host_decoder_->con_var_dims_ = host_evaluator->num_con_variable;
    host_decoder_->int_var_dims_ = host_evaluator->num_int_variable;
    host_decoder_->dims_ = host_evaluator->num_con_variable + host_evaluator->num_int_variable;

    // Initial evolve data
    host_evolve_data_->top_ratio = top_;
    host_evolve_data_->new_cluster_vec = new_cluster_vec_;
    host_evolve_data_->int_var_dims = int_var_dims_;
    host_evolve_data_->con_var_dims = con_var_dims_;
    host_evolve_data_->dims = int_var_dims_ + con_var_dims_;
    // printf("generation:%d\n",host_evaluator->generation);
    // host_evolve_data_->generation = host_evaluator->generation;
    
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

    host_evolve_data_->constraints = host_evaluator->constraints;
    host_evolve_data_->objective = host_evaluator->objective;

    for(int i = 0; i < host_evaluator->num_constraint; ++i){
        host_evolve_data_->lambda[i] = host_evaluator->lambda[i];
    }

    
    // initial constraint matrix
    row_constraint = host_evaluator->row_constraint_mat;
    col_constraint = host_evaluator->col_constraint_mat;
    row_constraint_constant = host_evaluator->row_constraint_constant_mat;
    col_constraint_constant = host_evaluator->col_constraint_constant_mat;
    row_obj = host_evaluator->row_obj;
    col_obj = host_evaluator->col_obj;
    
    row_obj_constant = host_evaluator->row_obj_constant;
    col_obj_constant = host_evaluator->col_obj_constant;
    
    size_t size_constraint_mat = row_constraint * col_constraint * sizeof(float);
    size_t size_constraint_constant_mat = row_constraint_constant * col_constraint_constant * sizeof(float);
    size_t size_obj = row_obj * col_obj * sizeof(float);
    size_t size_obj_constant = row_obj_constant * col_obj_constant * sizeof(float);

    CHECK_CUDA(cudaHostAlloc(&h_constraint_matrix, size_constraint_mat, cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_constraint_constant_matrix, size_constraint_constant_mat, cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_objective_matrix, size_obj, cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_obj_constant_matrix, size_obj_constant, cudaHostAllocDefault));
    memcpy(h_constraint_matrix, host_evaluator->constraint_mat, size_constraint_mat);
    memcpy(h_constraint_constant_matrix, host_evaluator->constraint_constant_mat, size_constraint_constant_mat);
    memcpy(h_objective_matrix, host_evaluator->obj, size_obj);
    memcpy(h_obj_constant_matrix, host_evaluator->obj_constant, size_obj_constant);


    CHECK_CUDA(cudaMalloc(&constraint_matrix, size_constraint_mat));
    CHECK_CUDA(cudaMalloc(&constraint_constant_matrix, size_constraint_constant_mat));
    CHECK_CUDA(cudaMalloc(&objective_matrix, size_obj));
    CHECK_CUDA(cudaMalloc(&obj_constant_matrix, size_obj_constant));
    // Then copy from host to device memory
    CHECK_CUDA(cudaMemcpy(constraint_matrix, h_constraint_matrix, size_constraint_mat, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(constraint_constant_matrix, h_constraint_constant_matrix, size_constraint_constant_mat, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(objective_matrix, h_objective_matrix, size_obj, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(obj_constant_matrix, h_obj_constant_matrix, size_obj_constant, cudaMemcpyHostToDevice));
    
    CHECK_CUDA(cudaHostAlloc(&host_param_matrix, host_evolve_data_->dims * default_pop_size_ * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaMalloc(&param_matrix, host_evolve_data_->dims * default_pop_size_ * sizeof(float)));

    // Initialize cuBLAS handle
    cublasStatus_t stat = cublasCreate(&cublas_handle_);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("CUBLAS initialization failed");
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

__global__ void LoadWarmStartResultForSolver(CudaEvolveData *evolve, CudaParamClusterData<64> *new_param){
    ConvertCudaParam<64>(new_param, &evolve->warm_start, blockIdx.x, threadIdx.x);
}

void CudaDiffEvolveSolver::Solver(int evolve_round){
    init_pop_size_ = default_pop_size_;
    pop_size_ = default_pop_size_;

    InitCudaEvolveData<<<1, default_pop_size_, 0, cuda_utils_->streams_[0]>>>(evolve_data_, old_cluster_data_, default_pop_size_);

    InitParameter<<<1, default_pop_size_, 0, cuda_utils_->streams_[0]>>>(decoder_, evolve_data_, default_pop_size_, new_cluster_data_, old_cluster_data_, random_center_->uniform_data_);

    LoadWarmStartResultForSolver<<<1, CUDA_PARAM_MAX_SIZE, 0, cuda_utils_->streams_[0]>>>(evolve_data_, new_cluster_data_);

    // based on warm start result to generate 
    GenerativeRandSolNearBest<<<default_pop_size_, CUDA_PARAM_MAX_SIZE, 0, cuda_utils_->streams_[0]>>>(evolve_data_, new_cluster_data_, random_center_->uniform_data_, 16, 0.1, 0.1, default_pop_size_);

    Evaluation(CUDA_SOLVER_POP_SIZE);

    ParaFindMax2<CUDA_SOLVER_POP_SIZE, 64><<<1, default_pop_size_, 0, cuda_utils_->streams_[0]>>>(new_cluster_data_);

    SaveNewParamAsOldParam<<<default_pop_size_, CUDA_PARAM_MAX_SIZE, 0, cuda_utils_->streams_[0]>>>(new_cluster_data_, old_cluster_data_, 0, default_pop_size_, 0);

    for (int i = 0; i < evolve_round; ++i) {
        Evolution(i, CudaEvolveType::GLOBAL);
    }

    // if (DEBUG_PRINT_FLAG){
    //     CHECK_CUDA(cudaStreamSynchronize(cuda_utils_->streams_[0]));
    //     // CHECK_CUDA(cudaMemcpyAsync(host_new_cluster_data_, new_cluster_data_, sizeof(CudaParamClusterData<64>), cudaMemcpyDeviceToHost, cuda_utils_->streams_[0]));
    //     // PrintClusterData(host_new_cluster_data_);
    //     CHECK_CUDA(cudaMemcpyAsync(host_old_cluster_data_, old_cluster_data_, sizeof(CudaParamClusterData<192>), cudaMemcpyDeviceToHost, cuda_utils_->streams_[0]));
    //     PrintClusterData(host_old_cluster_data_);

    //     // CHECK_CUDA(cudaMemcpyAsync(host_evolve_data_, evolve_data_, sizeof(CudaEvolveData), cudaMemcpyDeviceToHost, cuda_utils_->streams_[0]));
    //     // printf("CUDA_MAX_FLOAT %f\n", CUDA_MAX_FLOAT);
    // }
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
        CHECK_CUDA(cudaFree(constraint_matrix));
        CHECK_CUDA(cudaFree(constraint_constant_matrix));
        CHECK_CUDA(cudaFree(objective_matrix));
        CHECK_CUDA(cudaFree(obj_constant_matrix));
        CHECK_CUDA(cudaFree(param_matrix));

        // CPU host
        CHECK_CUDA(cudaFreeHost(host_decoder_));
        CHECK_CUDA(cudaFreeHost(h_constraint_matrix));
        CHECK_CUDA(cudaFreeHost(h_constraint_constant_matrix));
        CHECK_CUDA(cudaFreeHost(h_objective_matrix));
        CHECK_CUDA(cudaFreeHost(h_obj_constant_matrix));
        CHECK_CUDA(cudaFreeHost(host_evolve_data_));
        CHECK_CUDA(cudaFreeHost(host_new_cluster_data_));
        CHECK_CUDA(cudaFreeHost(host_old_cluster_data_));
        
        CHECK_CUDA(cudaFreeHost(host_param_matrix));
    }
    
}

}