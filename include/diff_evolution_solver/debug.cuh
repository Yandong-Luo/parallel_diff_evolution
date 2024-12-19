#ifndef CUDAPROCESS_DEBUG_OUTPUT_H
#define CUDAPROCESS_DEBUG_OUTPUT_H

#include "diff_evolution_solver/data_type.h"

template <int T>
void PrintClusterData(cudaprocess::CudaParamClusterData<T> *cluster_data){
    printf("cluster_data len:%d\n", T);
    for(int i = 0; i < CUDA_PARAM_MAX_SIZE * T; ++i){
        
        if (i % CUDA_PARAM_MAX_SIZE == 0){
            printf("\n");
            printf("Individual: ");
        }
        printf("%f ", cluster_data->all_param[i]);
    }
}

void PrintEvolveData(cudaprocess::CudaEvolveData *evolve){
    
}

#endif