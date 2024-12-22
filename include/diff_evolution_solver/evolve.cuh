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

    template <CudaEvolveType SearchType = CudaEvolveType::GLOBAL>
    __global__ void CudaEvolveProcess(int epoch, CudaParamClusterData<192> *old_param, CudaParamClusterData<64> *new_param, float *uniform_data,
                                      float *normal_data, CudaEvolveData *evolve_data, int pop_size, float eliteRatio){
        int dims = evolve_data->dims, con_dims = evolve_data->con_var_dims, int_dims = evolve_data->int_var_dims;
        int sol_idx = blockIdx.x;
        int param_idx = threadIdx.x;
        int UsingEliteStrategy = 0;
        int guideEliteIdx = 0;

        float crossover, scale_f, scale_f1;

        int mutationStartDim;

        float origin_param, best_origin_param, result_param;

        // Avoid using the same random number in the same place as other functions
        int normal_rnd_evolve_pos = epoch * CUDA_SOLVER_POP_SIZE * 3 + sol_idx * 3;
        int uniform_rnd_evolve_pos = epoch * CUDA_SOLVER_POP_SIZE * 38 + sol_idx * 38;

        int selected_parent_idx;
        int parent_param_idx[3], sorted_parent_param_idx[3];
        if (threadIdx.x < 3){
            int total_len = old_param->len;
            // Initialize 3 index that increases with threadIdx.x to ensure that subsequent sorting does not require too frequent operations
            selected_parent_idx = min((int)floor(uniform_data[uniform_rnd_evolve_pos + threadIdx.x] * (total_len - threadIdx.x - 1)), total_len - threadIdx.x - 2);
        }
        // Due to the selected_parent_idx was calculated in thread 0-2, we need let thread 0 know all result of selected_parent_idx for future sorting part
        // Use warp shuffle to share indices between threads
        parent_param_idx[0] = selected_parent_idx;
        parent_param_idx[1] = __shfl_sync(0x0000000f, selected_parent_idx, 1);
        parent_param_idx[2] = __shfl_sync(0x0000000f, selected_parent_idx, 2);

        if(threadIdx.x == 0){
            // sorting part
            // Ensure that the parent individuals selected in the differential evolution algorithm are different and non-repetitive

            for (int i = 0; i < 3; ++i) {
                sorted_parent_param_idx[i] = parent_param_idx[i];
                for (int j = 0; j < i; ++j) {
                    if (parent_param_idx[j] <= sorted_parent_param_idx[i]) sorted_parent_param_idx[i]++;
                }
                parent_param_idx[i] = sorted_parent_param_idx[i];
                // 插入排序
                for (int j = i; j > 0; j--) {
                    if (parent_param_idx[j] < parent_param_idx[j - 1]) {
                        int tmp = parent_param_idx[j];
                        parent_param_idx[j] = parent_param_idx[j - 1];
                        parent_param_idx[j - 1] = tmp;
                    }
                }
                if (sorted_parent_param_idx[i] >= blockIdx.x)   sorted_parent_param_idx[i]++;
            }

            scale_f = min(1.f, max(normal_data[normal_rnd_evolve_pos] * 0.01f + evolve_data->lshade_param.scale_f, 0.1));
            scale_f1 = min(1.f, max(normal_data[normal_rnd_evolve_pos + 1] * 0.01f + evolve_data->lshade_param.scale_f, 0.1));
            crossover = min(1.f, max(normal_data[normal_rnd_evolve_pos + 2] * 0.01f + evolve_data->lshade_param.Cr, 0.1));

            mutationStartDim = min((int)floor(uniform_data[uniform_rnd_evolve_pos + 3] * dims), dims - 1);

            int num_top = int(pop_size * evolve_data->top_ratio) + 1;
            // The index of an individual randomly selected from the top proportion of high-quality individuals in the population
            // Due to uniform_rnd_evolve_pos to uniform_rnd_evolve_pos + 3 have been used for parent_param_idx and mutationStartDim. So, starting from 4
            guideEliteIdx = max(0, min((num_top - 1), (int)floor(uniform_data[uniform_rnd_evolve_pos + 4] * num_top)));

            if (sol_idx * 1.f < pop_size * eliteRatio)  UsingEliteStrategy = 1;
        }
        // make sure all parameter have been calculated
        __syncwarp();

        // 使用 warp shuffle 广播这些值给其他线程
        scale_f = __shfl_sync(0x0000ffff, scale_f, 0);
        scale_f1 = __shfl_sync(0x0000ffff, scale_f1, 0);
        crossover = __shfl_sync(0x0000ffff, crossover, 0);
        UsingEliteStrategy = __shfl_sync(0x0000ffff, UsingEliteStrategy, 0);
        guideEliteIdx = __shfl_sync(0x0000ffff, guideEliteIdx, 0);

        int parent1_idx = sorted_parent_param_idx[0], parent2_idx = sorted_parent_param_idx[1];
        int mutant_idx = sorted_parent_param_idx[2];

        parent1_idx = __shfl_sync(0x0000ffff, parent1_idx, 0);
        parent2_idx = __shfl_sync(0x0000ffff, parent2_idx, 0);
        mutant_idx = __shfl_sync(0x0000ffff, mutant_idx, 0);

        // Other threads need to obtain the broadcasted data
        sorted_parent_param_idx[0] = parent1_idx;
        sorted_parent_param_idx[1] = parent2_idx;
        sorted_parent_param_idx[2] = mutant_idx;
        
        int totalsize = old_param->len;
        // check the random_idx valid or not
        if (sorted_parent_param_idx[0] >= old_param->len || sorted_parent_param_idx[1] >= old_param->len || sorted_parent_param_idx[2] >= old_param->len || old_param->len >= pop_size * 2 + 10) {
            printf("wft: %d, %d, %d, total len: %d\n", sorted_parent_param_idx[0], sorted_parent_param_idx[1], sorted_parent_param_idx[2], old_param->len);
        }

        // record the parameter for mutant
        float mutant_param[3];
        mutant_param[0] = old_param->all_param[sorted_parent_param_idx[0] * CUDA_PARAM_MAX_SIZE + threadIdx.x];
        mutant_param[1] = old_param->all_param[sorted_parent_param_idx[1] * CUDA_PARAM_MAX_SIZE + threadIdx.x];
        mutant_param[2] = old_param->all_param[sorted_parent_param_idx[2] * CUDA_PARAM_MAX_SIZE + threadIdx.x];

        if(UsingEliteStrategy){
            // Use the parameters corresponding to the current individual idx as the basis
            origin_param = old_param->all_param[sol_idx * CUDA_PARAM_MAX_SIZE + threadIdx.x];
        } else{
            // Use the best parameter as the basis
            origin_param = old_param->all_param[threadIdx.x];
        }

        // initial result param
        result_param = origin_param;

        // load the random param from top level as best
        best_origin_param = old_param->all_param[guideEliteIdx * CUDA_PARAM_MAX_SIZE + threadIdx.x];

        float f = (threadIdx.x >= con_dims) ? scale_f1 : scale_f;
        if (SearchType == CudaEvolveType::GLOBAL){
            float mutant_prob = uniform_data[uniform_rnd_evolve_pos + 5 + threadIdx.x];

            // initial the firstMutationDimIdx by last one
            int firstMutationDimIdx = pop_size;

            // crossover select
            if(mutant_prob > (UsingEliteStrategy ? crossover : 0.9f) && threadIdx.x < dims){
                firstMutationDimIdx = threadIdx.x;
            }

            // parallel reduce
            int tmp_idx = __shfl_down_sync(0x0000ffff, firstMutationDimIdx, 8);
            firstMutationDimIdx = min(tmp_idx, firstMutationDimIdx);
            tmp_idx = __shfl_down_sync(0x0000ffff, firstMutationDimIdx, 4);
            firstMutationDimIdx = min(tmp_idx, firstMutationDimIdx);
            tmp_idx = __shfl_down_sync(0x0000ffff, firstMutationDimIdx, 2);
            firstMutationDimIdx = min(tmp_idx, firstMutationDimIdx);
            tmp_idx = __shfl_down_sync(0x0000ffff, firstMutationDimIdx, 1);
            firstMutationDimIdx = min(tmp_idx, firstMutationDimIdx);
            firstMutationDimIdx = __shfl_sync(0x0000ffff, firstMutationDimIdx, 0);

            if (threadIdx.x < dims){
                bool isInMutationWindow = true;
                if (firstMutationDimIdx < dims){
                    int step = threadIdx.x - mutationStartDim;
                    if(step < 0)    step += dims;
                    if(step > firstMutationDimIdx){
                        isInMutationWindow = false;
                    }
                }

                if(isInMutationWindow){
                    if(UsingEliteStrategy){
                        result_param = origin_param + f * (best_origin_param + mutant_param[0] - origin_param - mutant_param[1]);
                    }
                    else{
                        result_param = origin_param + 0.8f * (mutant_param[0] - mutant_param[3]);
                    }
                }
            }
        }

        if (threadIdx.x < dims){
            float lower_bound = evolve_data->lower_bound[threadIdx.x];
            float upper_bound = evolve_data->upper_bound[threadIdx.x];

            if(result_param < lower_bound || result_param > upper_bound){
                result_param = uniform_data[uniform_rnd_evolve_pos + 24 + threadIdx.x] * (upper_bound - lower_bound) + lower_bound;
            }
        }
        new_param->all_param[sol_idx * CUDA_PARAM_MAX_SIZE + threadIdx.x] = result_param;

        if (threadIdx.x == 0) {
            reinterpret_cast<float3 *>(new_param->lshade_param)[sol_idx] = float3{scale_f, scale_f1, crossover};
        }

    }
}


#endif