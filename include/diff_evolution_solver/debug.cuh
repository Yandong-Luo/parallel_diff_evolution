#ifndef CUDAPROCESS_DEBUG_OUTPUT_H
#define CUDAPROCESS_DEBUG_OUTPUT_H

#include "diff_evolution_solver/data_type.h"

// template <int T>
// void PrintClusterData(cudaprocess::CudaParamClusterData<T> *cluster_data){
//     printf("cluster_data len:%d\n", T);
//     for(int i = 0; i < CUDA_PARAM_MAX_SIZE * T; ++i){
        
//         if (i % CUDA_PARAM_MAX_SIZE == 0 || i == 0){
//             printf("\n");
//             // if(cluster_data->fitness[i % CUDA_PARAM_MAX_SIZE] < 10)  printf("=================================\n");
//             printf("Individual fitness: %f, and its param: ", cluster_data->fitness[i % CUDA_PARAM_MAX_SIZE]);
//         }
//         printf("%f ", cluster_data->all_param[i]);
//     }
// }

template <int T>
void PrintClusterData(cudaprocess::CudaParamClusterData<T> *cluster_data){
    printf("cluster_data len:%d\n", T);
    
    // 遍历每个个体
    for(int i = 0; i < T; ++i){
        printf("\nIndividual %d fitness: %f, lshade params(f,f1,cr): %f %f %f\nParams: ", 
            i, 
            cluster_data->fitness[i],
            cluster_data->lshade_param[i * 3],     // scale_f
            cluster_data->lshade_param[i * 3 + 1], // scale_f1
            cluster_data->lshade_param[i * 3 + 2]  // crossover
        );
        
        // 打印该个体的所有参数
        for(int j = 0; j < cluster_data->dims; ++j){  // 只打印实际维度的参数
            printf("%f ", cluster_data->all_param[i * CUDA_PARAM_MAX_SIZE + j]);
        }
    }
    printf("\n\nProblem dimensions: con_var=%d, int_var=%d, total=%d\n", 
        cluster_data->con_var_dims, 
        cluster_data->int_var_dims, 
        cluster_data->dims);
}

void PrintEvolveData(cudaprocess::CudaEvolveData *evolve){
    for(int i=0; i < CUDA_PARAM_MAX_SIZE; ++i){
        printf("warmstart:%f ", evolve->warm_start.param[i]);
    }
}

#endif