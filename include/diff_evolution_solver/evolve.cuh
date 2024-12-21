#ifndef CUDAPROCESS_DIFF_EVOLVE_H
#define CUDAPROCESS_DIFF_EVOLVE_H

#include <cuda.h>
#include <cuda_runtime_api.h>

// #ifdef __CUDACC__
// #include <cuda.h>
// #include <cuda_runtime_api.h>
// #endif

#include "diff_evolution_solver/data_type.h"

namespace cudaprocess{
    template <int T = 64>
    __global__ void DuplicateBestAndReorganize(int epoch, CudaParamClusterData<64> *old_param, int copy_num){
        // if(epoch > 0)   return;

        int size = 2 * T;

        if (epoch == 0){
            size = T;
        }

        int param_id = blockIdx.x;
        int sol_id = threadIdx.x;

        float param;
        float best_fitness;

        if (sol_id < size + copy_num){
            if (sol_id <= copy_num){
                // For the first copy_num solutions, use the parameters of the best solution (sol_id=0)
                param = old_param->all_param[param_id];
                if (param_id == 0)  best_fitness = old_param->fitness[0];   // copy the best fitness
            }
            else{
                // for the remaining solution, use the original parameters
                param = old_param->all_param[(sol_id - copy_num) * CUDA_PARAM_MAX_SIZE + param_id];
                if(param_id == 0){
                    best_fitness = old_param->fitness[sol_id - copy_num];
                }
            }
        }
        // wait for all thread finish above all copy step
        __syncthreads();
        // reorganize
        if(sol_id < size + copy_num){
            old_param->all_param[sol_id * CUDA_PARAM_MAX_SIZE + param_id] = param;
            if (param_id == 0){
                old_param->fitness[sol_id] = best_fitness;
                if(sol_id == 0){
                    old_param->len = size + copy_num;
                }
            }
        }
    }
}


#endif