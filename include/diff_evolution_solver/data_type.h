#ifndef CUDA_DIFF_EVOLUTION_DATA_TYPE_H
#define CUDA_DIFF_EVOLUTION_DATA_TYPE_H

namespace cudaprocess{

#define CUDA_PARA_MAX_SIZE 16

template <typename T, int size>
struct CudaVector {
  T data[size];
  int len{0};
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

struct CudaEvolveData{
    float best;
    int con_var_dims, bin_var_dims;
    CudaLShadePair hyperparam;
};

struct ProblemDecoder{
    int dim_{0}, con_var_dims_{0}, bin_var_dims_{0};
};

}


#endif