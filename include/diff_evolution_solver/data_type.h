#ifndef CUDA_DIFF_EVOLUTION_DATA_TYPE_H
#define CUDA_DIFF_EVOLUTION_DATA_TYPE_H

#include "utils.cuh"

namespace cudaprocess{

#define ALIGN(n) __align__(n)


template <typename T, int size>
struct CudaVector {
  T data[size];
  int len{0};
};

struct ProblemEvaluator{
    int num_con_variable = 3;
    int num_binary_variable = 2;
    int binary_upper_bound = 1;
    int binary_lowwer_bound = 0;
    float con_upper_bound = 10.;
    float con_lower_bound = 0.;
};

/*
 * Param:
 * prob_f:  scaling factor for L-Shape 
 * Cr:      Crossover Probability
 * weight:  weight factor
 */
struct CudaLShadePair {
    float scale_f, scale_f1, Cr, weight;
};

struct CudaParamIndividual{
    float param[CUDA_PARAM_MAX_SIZE];
    int con_var_dims, bin_var_dims, dims;
    float cur_scale_f{0.5F}, cur_scale_f1{0.5F}, cur_Cr{0.5F};
    float fitness;
};

/*
 * Store individual variables in a cluster in a more compact way
 */
template <int T>
struct ALIGN(64) CudaParamClusterData{
    float all_param[CUDA_PARAM_MAX_SIZE * T];
    float lshade_param[T * 3];  // current scale_f, current scale_f1, current crossover probability
    int con_var_dims, bin_var_dims, dims;
    float fitness[T];
    int len{0};
};

struct CudaEvolveData{
    float best;
    int con_var_dims, bin_var_dims, dims;
    CudaLShadePair lshade_param;
    CudaVector<CudaParamIndividual, CUDA_SOLVER_POP_SIZE> *new_cluster_vec;
    
};

struct CudaProblemDecoder{
    int dims_{0}, con_var_dims_{0}, bin_var_dims_{0};

};

}


#endif