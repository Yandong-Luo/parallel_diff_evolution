#include "diff_evolution_solver/solver.cuh"
#include "diff_evolution_solver/decoder.cuh"
#include "diff_evolution_solver/debug.cuh"
#include "diff_evolution_solver/evolve.cuh"
#include "diff_evolution_solver/evaluate.cuh"
#include "utils/utils_fun.cuh"
#include <math.h>

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
    CHECK_CUDA(cudaMalloc(&evaluate_score_, default_pop_size_ * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&last_fitness, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&terminate_flag, sizeof(int)));
    // CHECK_CUDA(cudaMalloc(&new_param, sizeof(CudaParamClusterData<192>)));

    // CPU Host
    CHECK_CUDA(cudaHostAlloc(&host_decoder_, sizeof(CudaProblemDecoder), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&host_evolve_data_, sizeof(CudaEvolveData), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&host_new_cluster_data_, sizeof(CudaParamClusterData<64>), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&host_old_cluster_data_, sizeof(CudaParamClusterData<192>), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&host_solver_input_, sizeof(CudaSolverInput), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&host_evaluate_score_, default_pop_size_ * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&host_result, sizeof(CudaParamIndividual), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_terminate_flag, sizeof(int), cudaHostAllocDefault));
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
__global__ void UpdateVecParamBasedClusterData(CudaParamIndividual *output, CudaParamClusterData<T> *cluster_data){
    ConvertCudaParamRevert<T>(cluster_data, &output[blockIdx.x], blockIdx.x, threadIdx.x);
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
    Evaluation(CUDA_SOLVER_POP_SIZE, 0);

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

void CudaDiffEvolveSolver::Evaluation(int size, int epoch){
    // Row-major arrangement
    ConvertClusterToMatrix<64><<<size, dims_ + 1, 0, cuda_utils_->streams_[0]>>>(evolve_data_, new_cluster_data_, param_matrix);
    // printf("device obj_constant_matrix\n");
    // printMatrix<<<1, row_obj_constant*col_obj_constant, 0, cuda_utils_->streams_[0]>>>(obj_constant_matrix);

    // printf("device objective_matrix\n");
    // printMatrix<<<1, row_obj*col_obj, 0, cuda_utils_->streams_[0]>>>(objective_matrix);
    
    float alpha = 1.;
    float beta = 1.;

    // reset the evaluate score and tmp score
    cudaMemset(evaluate_score_, 0, size * sizeof(float));

    cudaMemset(tmp_score, 0, size * col_constraint * sizeof(float));

    // Based on current epoch and interpolation to update lambda
    // UpdateLambdaBasedInterpolation<<<1, row_lambda * col_lambda, 0, cuda_utils_->streams_[0]>>>(evolve_data_, lambda_matrix, epoch);

    // checking before matrix multiplication
    if(DEBUG_PRINT_FLAG || DEBUG_PRINT_EVALUATE_FLAG){
        
        CHECK_CUDA(cudaMemcpy(host_param_matrix, param_matrix, (dims_+1) * size * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_objective_matrix, objective_matrix, row_obj * col_obj * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_constraint_matrix, constraint_matrix, row_constraint * col_constraint * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_lambda_matrix, lambda_matrix, row_lambda * col_lambda * sizeof(float), cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaStreamSynchronize(cuda_utils_->streams_[0]));

        // check parameter matrix before matrix multiplication
        PrintMatrix(host_param_matrix, size, dims_+1, "CHECK PARAMETER MATRIX BEFORE MATRIX MULTIPLICATION");

        // check obj matrix before matrix multiplication
        PrintMatrix(h_objective_matrix, row_obj, col_obj, "CHECK OBJ MATRIX BEFORE MATRIX MULTIPLICATION");

        // check constraint matrix before matrix multiplication
        PrintMatrix(h_constraint_matrix, row_constraint, col_constraint, "CHECK CONSTRAINT MATRIX BEFORE MATRIX MULTIPLICATION");

        // check lambda matrix before matrix multiplication
        PrintMatrix(h_lambda_matrix, row_lambda, col_lambda, "CHECK LAMBDA MATRIX BEFORE MATRIX MULTIPLICATION");
    }

    // objective part
    // example:
    // param_matrix: pop_size x dims (64 x 3)
    // objective_matrix: 3 x 1
    // obj_constant_matrix: 64 x 1
    //     cublasSgemm(
    //     handle,
    //     CUBLAS_OP_T,  // A is stored row-first, so it is considered transposed, otherwise CUBLAS_OP_N
    //     CUBLAS_OP_T,  // B is stored in row-major order, so it is considered transposed, otherwise CUBLAS_OP_N
    //     m, n, k,     // m: the number of rows in C; n: the number of columns in C; k: the number of columns in A; 
    //     &alpha,
    //     A, lda,  // leading dimension of A. If the matrix is ​​stored row-major, lda should be the number of columns in the matrix. Otherwise, lda should be the number of rows.
    //     B, ldb,  // leading dimension of B
    //     &beta,
    //     C, ldc   // C 的 leading dimension
    // );

    // printf("CHECK THE PARAM OF cublasSgemm: %d %d %d %d %d %d\n", row_obj_constant, col_obj_constant, dims_, dims_, col_obj , row_obj_constant);
    // Strongly recommend reading this blog: https://blog.csdn.net/HaoBBNuanMM/article/details/103054357
    cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, 1, size, dims_ + 1, &alpha, objective_matrix, col_obj, param_matrix, dims_ + 1, &beta, evaluate_score_, col_obj);

    // printf("CHECK THE PARAM OF cublasSgemm: %d %d %d %d %d %d\n", col_constraint, size, dims_+1, dims_+1, col_constraint , size);
    cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, col_constraint, size, dims_ + 1, &alpha, constraint_matrix, col_constraint, param_matrix, dims_ + 1, &beta, tmp_score, col_constraint);

    InequalityMask<<<1, size * col_constraint, 0, cuda_utils_->streams_[0]>>>(tmp_score);

    cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, 1, size, row_lambda, &alpha, lambda_matrix, col_lambda, tmp_score, col_constraint, &beta, evaluate_score_, col_lambda);

    if(DEBUG_PRINT_FLAG || DEBUG_PRINT_EVALUATE_FLAG){
        // GPU DEBUG
        // printf("device obj_constant_matrix\n");
        // printMatrix<<<1, row_obj_constant*col_obj_constant, 0, cuda_utils_->streams_[0]>>>(obj_constant_matrix);

        
        CHECK_CUDA(cudaMemcpy(host_tmp_score, tmp_score, size * col_constraint * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(host_evaluate_score_, evaluate_score_, size * sizeof(float), cudaMemcpyDeviceToHost));
        // CHECK_CUDA(cudaMemcpy(host_param_matrix, param_matrix, (dims_ + 1) * size * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaStreamSynchronize(cuda_utils_->streams_[0]));

        // check tmp score with individual after matrix multiplication
        PrintTmpScoreWithParam(host_tmp_score, host_param_matrix, size, col_constraint, dims_ + 1, "CHECK TMP SCORE");
        // check fitness with individual after matrix multiplication
        PrintFitnesssWithParam(host_evaluate_score_, host_param_matrix, size, 1, dims_ + 1, "CHECK FITNESS WITH PARAM");
    }
    UpdateFitnessBasedMatrix<64><<<1, size, 0, cuda_utils_->streams_[0]>>>(new_cluster_data_, evaluate_score_);


    // MainEvaluation<64><<<1, size, 0, cuda_utils_->streams_[0]>>>(evolve_data_, new_cluster_data_);
}

void CudaDiffEvolveSolver::Evolution(int epoch, CudaEvolveType search_type){
    DuplicateBestAndReorganize<<<CUDA_PARAM_MAX_SIZE, 192, 0, cuda_utils_->streams_[0]>>>(epoch, old_cluster_data_, 2);
    CudaEvolveProcess<<<default_pop_size_, CUDA_PARAM_MAX_SIZE, 0, cuda_utils_->streams_[0]>>>(epoch, old_cluster_data_, new_cluster_data_, random_center_->uniform_data_, random_center_->normal_data_, evolve_data_, default_pop_size_, true);
    Evaluation(default_pop_size_, epoch);

    // CHECK_CUDA(cudaStreamSynchronize(cuda_utils_->streams_[0]));
    UpdateParameter<64><<<CUDA_PARAM_MAX_SIZE, 128, 0, cuda_utils_->streams_[0]>>>(epoch, evolve_data_, new_cluster_data_, old_cluster_data_, terminate_flag, last_fitness);

    CHECK_CUDA(cudaMemcpyAsync(h_terminate_flag, terminate_flag, sizeof(int), cudaMemcpyDeviceToHost, cuda_utils_->streams_[0]));
    CHECK_CUDA(cudaStreamSynchronize(cuda_utils_->streams_[0]));
}

void CudaDiffEvolveSolver::InitSolver(int gpu_device, CudaRandomCenter *random_center, ProblemEvaluator* host_evaluator, CudaParamIndividual *output_sol, const CudaVector<CudaParamIndividual, CUDA_MAX_POTENTIAL_SOLUTION> *last_potential_sol){
    if(DEBUG_ENABLE_NVTX)   init_range = nvtxRangeStart("Init Different Evolvution Solver");

    gpu_device_ = gpu_device;
    random_center_ = random_center;

    CHECK_CUDA(cudaSetDevice(gpu_device_));
    if (DEBUG_PRINT_FLAG || DEBUG_PRINT_INIT_SOLVER_FLAG) printf("CUDA SET DEVICE\n");

    MallocSetup();

    con_var_dims_ = host_evaluator->num_con_variable;
    int_var_dims_ = host_evaluator->num_int_variable;
    dims_ = host_evaluator->num_con_variable + host_evaluator->num_int_variable;

    InitDiffEvolveParam();
    
    if (DEBUG_PRINT_FLAG || DEBUG_PRINT_INIT_SOLVER_FLAG) printf("INIT PARAM FOR DE\n");
    
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
    // row_constraint_constant = host_evaluator->row_constraint_constant_mat;
    // col_constraint_constant = host_evaluator->col_constraint_constant_mat;
    
    row_obj = host_evaluator->row_obj;
    col_obj = host_evaluator->col_obj;
    // row_obj_constant = host_evaluator->row_obj_constant;
    // col_obj_constant = host_evaluator->col_obj_constant;

    row_lambda = host_evaluator->row_lambda;
    col_lambda = host_evaluator->col_lambda;
    
    size_t size_constraint_mat = row_constraint * col_constraint * sizeof(float);
    // size_t size_constraint_constant_mat = row_constraint_constant * col_constraint_constant * sizeof(float);
    size_t size_obj = row_obj * col_obj * sizeof(float);
    // size_t size_obj_constant = row_obj_constant * col_obj_constant * sizeof(float);

    CHECK_CUDA(cudaHostAlloc(&h_constraint_matrix, size_constraint_mat, cudaHostAllocDefault));
    // CHECK_CUDA(cudaHostAlloc(&h_constraint_constant_matrix, size_constraint_constant_mat, cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_objective_matrix, size_obj, cudaHostAllocDefault));
    // CHECK_CUDA(cudaHostAlloc(&h_obj_constant_matrix, row_obj_constant * col_obj_constant * sizeof(float), cudaHostAllocDefault));
    memcpy(h_constraint_matrix, host_evaluator->constraint_mat, size_constraint_mat);
    // memcpy(h_constraint_constant_matrix, host_evaluator->constraint_constant_mat, size_constraint_constant_mat);
    memcpy(h_objective_matrix, host_evaluator->obj, size_obj);
    // memcpy(h_obj_constant_matrix, host_evaluator->obj_constant, row_obj_constant * col_obj_constant * sizeof(float));
    // printf("===============================\n");
    // for(int i = 0; i < row_obj_constant; ++i){
    //     for(int j = 0; j < col_obj_constant; ++j){
    //         // printf("row:%d col:%d objective mat:%f ", i, j, h_objective_matrix[i * col_obj +j]);
    //         printf("obj_constant_matrix[%d,%d]=%f ", i, j, h_obj_constant_matrix[i * col_obj_constant + j]);
    //     }
    //     printf("\n");
    // }
    host_evolve_data_->max_lambda = host_evaluator->max_lambda;
    host_evolve_data_->init_lambda = host_evaluator->init_lambda;
    host_evolve_data_->max_round = host_evaluator->max_round;

    host_evolve_data_->accuracy_rng = host_evaluator->accuracy_rng;
    host_evolve_data_->elite_eval_count = host_evaluator->elite_eval_count;

    CHECK_CUDA(cudaMalloc(&tmp_score, default_pop_size_ * host_evaluator->col_constraint_mat * sizeof(float)));
    CHECK_CUDA(cudaHostAlloc(&host_tmp_score, default_pop_size_ * host_evaluator->col_constraint_mat * sizeof(float), cudaHostAllocDefault));

    CHECK_CUDA(cudaMalloc(&lambda_matrix, row_lambda * col_lambda * sizeof(float)));
    CHECK_CUDA(cudaHostAlloc(&h_lambda_matrix, row_lambda * col_lambda * sizeof(float), cudaHostAllocDefault));

    memcpy(h_lambda_matrix, host_evaluator->lambda, row_lambda * col_lambda * sizeof(float));
    CHECK_CUDA(cudaMemcpy(lambda_matrix, h_lambda_matrix, row_lambda * col_lambda * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMalloc(&constraint_matrix, size_constraint_mat));
    // CHECK_CUDA(cudaMalloc(&constraint_constant_matrix, size_constraint_constant_mat));
    CHECK_CUDA(cudaMalloc(&objective_matrix, size_obj));
    // CHECK_CUDA(cudaMalloc((void**)&obj_constant_matrix, row_obj_constant * col_obj_constant * sizeof(float)));
    // Then copy from host to device memory
    CHECK_CUDA(cudaMemcpy(constraint_matrix, h_constraint_matrix, size_constraint_mat, cudaMemcpyHostToDevice));
    // CHECK_CUDA(cudaMemcpy(constraint_constant_matrix, h_constraint_constant_matrix, size_constraint_constant_mat, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(objective_matrix, h_objective_matrix, size_obj, cudaMemcpyHostToDevice));
    // CHECK_CUDA(cudaMemcpy(obj_constant_matrix, h_obj_constant_matrix, row_obj_constant * col_obj_constant * sizeof(float), cudaMemcpyHostToDevice));
    
    CHECK_CUDA(cudaHostAlloc(&host_param_matrix, (host_evolve_data_->dims + 1) * default_pop_size_ * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaMalloc(&param_matrix, (host_evolve_data_->dims + 1) * default_pop_size_ * sizeof(float)));

    CHECK_CUDA(cudaStreamSynchronize(cuda_utils_->streams_[0]));

    // Initialize cuBLAS handle
    cublasStatus_t status = cublasCreate(&cublas_handle_);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("CUBLAS initialization failed");
    }
    
    *h_terminate_flag = 0;
    cudaMemset(terminate_flag, 0, sizeof(int));
    float init_last_f = CUDA_MAX_FLOAT;
    CHECK_CUDA(cudaMemcpy(last_fitness, &init_last_f, sizeof(float), cudaMemcpyHostToDevice));
    // cudaMemset(last_fitness, CUDA_MAX_FLOAT, sizeof(float));
    
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

    if (DEBUG_PRINT_FLAG || DEBUG_PRINT_INIT_SOLVER_FLAG) printf("START MEMORY ASYNC\n");

    // Host --> GPU device
    CHECK_CUDA(cudaMemcpyAsync(evolve_data_, host_evolve_data_, sizeof(CudaEvolveData), cudaMemcpyHostToDevice, cuda_utils_->streams_[0]));
    CHECK_CUDA(cudaMemcpyAsync(decoder_, host_decoder_, sizeof(CudaProblemDecoder), cudaMemcpyHostToDevice, cuda_utils_->streams_[0]));
    CHECK_CUDA(cudaMemcpyAsync(evaluator_, host_evaluator, sizeof(ProblemEvaluator), cudaMemcpyHostToDevice, cuda_utils_->streams_[0]));
    CHECK_CUDA(cudaMemcpyAsync(device_solver_input_, host_solver_input_, sizeof(CudaSolverInput), cudaMemcpyHostToDevice, cuda_utils_->streams_[0]));
    CHECK_CUDA(cudaStreamSynchronize(cuda_utils_->streams_[0]));
    // if (last_sol == nullptr){
    //     CHECK_CUDA(cudaStreamSynchronize(cuda_utils_->streams_[0]));
    // }

    if (DEBUG_PRINT_FLAG || DEBUG_PRINT_INIT_SOLVER_FLAG) printf("MEMORY ASYNC SUBMIT\n");

    InitCudaEvolveData<<<1, CUDA_SOLVER_POP_SIZE, 0, cuda_utils_->streams_[0]>>>(evolve_data_, old_cluster_data_, default_pop_size_);

    if (DEBUG_ENABLE_NVTX)  nvtxRangeEnd(init_range);

    WarmStart(host_evaluator, output_sol);

    // if (DEBUG_PRINT_FLAG){
    //     CHECK_CUDA(cudaStreamSynchronize(cuda_utils_->streams_[0]));
    //     CHECK_CUDA(cudaMemcpyAsync(host_new_cluster_data_, new_cluster_data_, sizeof(CudaParamClusterData<64>), cudaMemcpyDeviceToHost, cuda_utils_->streams_[0]));
    //     PrintClusterData(host_new_cluster_data_);
    //     // CHECK_CUDA(cudaMemcpyAsync(host_old_cluster_data_, old_cluster_data_, sizeof(CudaParamClusterData<192>), cudaMemcpyDeviceToHost, cuda_utils_->streams_[0]));
    //     // PrintClusterData(host_old_cluster_data_);

    //     // CHECK_CUDA(cudaMemcpyAsync(host_evolve_data_, evolve_data_, sizeof(CudaEvolveData), cudaMemcpyDeviceToHost, cuda_utils_->streams_[0]));
    //     // printf("CUDA_MAX_FLOAT %f\n", CUDA_MAX_FLOAT);
    // }
}

__global__ void LoadWarmStartResultForSolver(CudaEvolveData *evolve, CudaParamClusterData<64> *new_param){
    ConvertCudaParam<64>(new_param, &evolve->warm_start, blockIdx.x, threadIdx.x);
}

template <int T=192>
__global__ void GetSolFromOldParam(CudaParamClusterData<192> *old_param, CudaParamIndividual *solution){
    ConvertCudaParamRevert<192>(old_param, solution, blockIdx.x, threadIdx.x);
}

CudaParamIndividual CudaDiffEvolveSolver::Solver(){
    // nvtx3::mark("Different Evolvution Solver!");
    if(DEBUG_ENABLE_NVTX)   solver_range = nvtxRangeStart("Different Evolvution Solver");

    init_pop_size_ = default_pop_size_;
    pop_size_ = default_pop_size_;

    InitCudaEvolveData<<<1, default_pop_size_, 0, cuda_utils_->streams_[0]>>>(evolve_data_, old_cluster_data_, default_pop_size_);

    InitParameter<<<1, default_pop_size_, 0, cuda_utils_->streams_[0]>>>(decoder_, evolve_data_, default_pop_size_, new_cluster_data_, old_cluster_data_, random_center_->uniform_data_);

    LoadWarmStartResultForSolver<<<1, CUDA_PARAM_MAX_SIZE, 0, cuda_utils_->streams_[0]>>>(evolve_data_, new_cluster_data_);

    // based on warm start result to generate 
    GenerativeRandSolNearBest<<<default_pop_size_, CUDA_PARAM_MAX_SIZE, 0, cuda_utils_->streams_[0]>>>(evolve_data_, new_cluster_data_, random_center_->uniform_data_, 16, 0.1, 0.1, default_pop_size_);

    Evaluation(CUDA_SOLVER_POP_SIZE, 0);

    ParaFindMax2<CUDA_SOLVER_POP_SIZE, 64><<<1, default_pop_size_, 0, cuda_utils_->streams_[0]>>>(new_cluster_data_);

    SaveNewParamAsOldParam<<<default_pop_size_, CUDA_PARAM_MAX_SIZE, 0, cuda_utils_->streams_[0]>>>(new_cluster_data_, old_cluster_data_, 0, default_pop_size_, 0);

    for (int i = 0; i < host_evolve_data_->max_round && !*h_terminate_flag; ++i) {
        // printf("generation i:%d\n", i);
        Evolution(i, CudaEvolveType::GLOBAL);
    }

    if (DEBUG_PRINT_FLAG || DEBUG_PRINT_SOLVER_FLAG){
        // CHECK_CUDA(cudaMemcpyAsync(host_new_cluster_data_, new_cluster_data_, sizeof(CudaParamClusterData<64>), cudaMemcpyDeviceToHost, cuda_utils_->streams_[0]));
        // PrintClusterData(host_new_cluster_data_);
        CHECK_CUDA(cudaMemcpyAsync(host_old_cluster_data_, old_cluster_data_, sizeof(CudaParamClusterData<192>), cudaMemcpyDeviceToHost, cuda_utils_->streams_[0]));
        CHECK_CUDA(cudaStreamSynchronize(cuda_utils_->streams_[0]));
        PrintClusterData<192>(host_old_cluster_data_);

        // CHECK_CUDA(cudaMemcpyAsync(host_evolve_data_, evolve_data_, sizeof(CudaEvolveData), cudaMemcpyDeviceToHost, cuda_utils_->streams_[0]));
        // printf("CUDA_MAX_FLOAT %f\n", CUDA_MAX_FLOAT);
    }
    
    // Get the first individual from old param (after sorting, the first one is the best one)
    GetSolFromOldParam<192><<<1, CUDA_PARAM_MAX_SIZE, 0, cuda_utils_->streams_[0]>>>(old_cluster_data_, new_cluster_vec_->data);
    CHECK_CUDA(cudaMemcpyAsync(host_result, new_cluster_vec_->data, sizeof(CudaParamIndividual), cudaMemcpyDeviceToHost, cuda_utils_->streams_[0]));
    CHECK_CUDA(cudaStreamSynchronize(cuda_utils_->streams_[0]));

    for(int i = con_var_dims_; i < dims_; ++i){
        host_result->param[i] = floor(host_result->param[i]);
    }

    if (DEBUG_PRINT_FLAG || DEBUG_PRINT_SOLVER_FLAG)   printFinalResult(host_result->fitness, host_result->param, dims_);

    if(DEBUG_ENABLE_NVTX)   nvtxRangeEnd(solver_range);

    return *host_result;
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
        // CHECK_CUDA(cudaFree(constraint_constant_matrix));
        CHECK_CUDA(cudaFree(objective_matrix));
        // CHECK_CUDA(cudaFree(obj_constant_matrix));
        CHECK_CUDA(cudaFree(param_matrix));
        CHECK_CUDA(cudaFree(evaluate_score_));
        CHECK_CUDA(cudaFree(tmp_score));
        CHECK_CUDA(cudaFree(lambda_matrix));
        // CHECK_CUDA(cudaFree(result));

        // CPU host
        CHECK_CUDA(cudaFreeHost(host_decoder_));
        CHECK_CUDA(cudaFreeHost(h_constraint_matrix));
        // CHECK_CUDA(cudaFreeHost(h_constraint_constant_matrix));
        CHECK_CUDA(cudaFreeHost(h_objective_matrix));
        // CHECK_CUDA(cudaFreeHost(h_obj_constant_matrix));
        CHECK_CUDA(cudaFreeHost(host_evolve_data_));
        CHECK_CUDA(cudaFreeHost(host_new_cluster_data_));
        CHECK_CUDA(cudaFreeHost(host_old_cluster_data_));
        CHECK_CUDA(cudaFreeHost(host_evaluate_score_));
        CHECK_CUDA(cudaFreeHost(host_param_matrix));
        CHECK_CUDA(cudaFreeHost(host_tmp_score));
        CHECK_CUDA(cudaFreeHost(h_lambda_matrix));
        CHECK_CUDA(cudaFreeHost(host_result));
        
    }
    
}

}