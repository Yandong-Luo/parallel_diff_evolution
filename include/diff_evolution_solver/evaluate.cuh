#ifndef CUDAPROCESS_DIFF_EVALUATE_H
#define CUDAPROCESS_DIFF_EVALUATE_H

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <cublas_v2.h>
// #ifdef __CUDACC__
// #include <cuda.h>
// #include <cuda_runtime_api.h>
// #endif

#include "diff_evolution_solver/data_type.h"

namespace cudaprocess{

template<int T>
__global__ void DynamicEvaluation(CudaEvolveData *evolve, CudaParamClusterData<T> *cluster_data, const float *lambda){
    printf("???????????????????????");
    if (threadIdx.x > 64) return;
    
    int sol_id = threadIdx.x;
    float param[CUDA_PARAM_MAX_SIZE];  // 临时存储当前线程要处理的参数
    float score = 0.f;

    for(int i = 0; i < evolve->dims; ++i) {
        param[i] = cluster_data->all_param[sol_id * CUDA_PARAM_MAX_SIZE + i];
    }
    
    float* constraint_score = evolve->constraints(param);
    score += evolve->objective(param);
    for (int i = 0; i < evolve->num_constraint; ++i){
        score += (constraint_score[i] <= 0) ? 0 : fabs(constraint_score[i]) * lambda[i];
    }
    
    cluster_data->fitness[sol_id] = score;

    printf("threadid.x:%d fitness:%f\n", sol_id, score);
}

/**
 * One Block with 64 thread to evaluate different 
 * In the future we can use matrix to present the constraint
 */
template<int T>
__device__ void DynamicEvaluation2(CudaEvolveData *evolve, CudaParamClusterData<T> *cluster_data, const float *lambda){
    float score = 0.0;
    int idx = threadIdx.x;
    
    if (idx >= T) return;

    float constraint_score[2] = {0.f};
    
    for(int i = 0; i < evolve->num_constraint; ++i){
        for(int j = 0; j < evolve->num_constraint_variable; ++j){
            if (j == evolve->num_constraint_variable - 1)   constraint_score[i] += evolve->constraint_para[i][j];
            else  constraint_score[i] += cluster_data->all_param[idx * CUDA_PARAM_MAX_SIZE + j] * evolve->constraint_para[i][j];
        }
    }

    // objective function
    for(int i = 0; i < evolve->num_objective_param; ++i){
        score += evolve->objective_param[i] * cluster_data->all_param[idx * CUDA_PARAM_MAX_SIZE + i];
    }

    for (int i = 0; i < evolve->num_constraint; ++i){
        score += (constraint_score[i] <= 0) ? 0 : fabs(constraint_score[i]) * lambda[i];
    }

    cluster_data->fitness[idx] = score;
    // printf("thread id:%d setting fitness to %f\n", idx, score);
}

template<int T>
__global__ void ConvertClusterToMatrix(CudaEvolveData *evolve, CudaParamClusterData<T> *cluster_data, float *param_matrix){
    if (threadIdx.x > evolve->dims) return;
    if (blockIdx.x >= T)    return;
    if(threadIdx.x == evolve->dims){
        param_matrix[blockIdx.x * (evolve->dims + 1) + threadIdx.x] = 1.0;
        printf("finish the convert: param[%d] to matrix[%d], value:%f\n", blockIdx.x * CUDA_PARAM_MAX_SIZE + threadIdx.x, blockIdx.x * (evolve->dims + 1) + threadIdx.x, param_matrix[blockIdx.x * (evolve->dims + 1) + threadIdx.x]);
        return;
    }
    // int sol_id = blockIdx.x;
    // int param_id = threadIdx.x;
    param_matrix[blockIdx.x * (evolve->dims + 1) + threadIdx.x] = cluster_data->all_param[blockIdx.x * CUDA_PARAM_MAX_SIZE + threadIdx.x];
    // param_matrix[blockIdx.x * evolve->dims + threadIdx.x] =
    // cluster_data->all_param[blockIdx.x * CUDA_PARAM_MAX_SIZE + threadIdx.x];


    printf("finish the convert: param[%d] to matrix[%d], value:%f\n", blockIdx.x * CUDA_PARAM_MAX_SIZE + threadIdx.x, blockIdx.x * (evolve->dims + 1) + threadIdx.x, param_matrix[blockIdx.x * (evolve->dims + 1) + threadIdx.x]);
}


}


#endif