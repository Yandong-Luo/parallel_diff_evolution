#ifndef CUDAPROCESS_RANDOM_CENTER_H
#define CUDAPROCESS_RANDOM_CENTER_H


#include <curand.h>
#include <curand_kernel.h>
#include "utils/utils.cuh"

namespace cudaprocess{
    class CudaRandomCenter
    {
    private:
        /* data */
    public:
        CudaRandomCenter(int gpu_device);
        ~CudaRandomCenter();
        void Generate();

        int size_{CUDA_SOLVER_POP_SIZE};
        int grid_size_, block_size_;
        int normal_grid_size_, uniform_grid_size_;
        int normal_size_, uniform_size_;
        curandState_t *states_;
        float *uniform_data_, *normal_data_;
        long long unsigned seed_;
    };
    
    
    
}

#endif