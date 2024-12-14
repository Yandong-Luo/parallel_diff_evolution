#ifndef CUDAPROCESS_DIFF_EVOLUTION_SOLVER_H
#define CUDAPROCESS_DIFF_EVOLUTION_SOLVER_H

#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <sys/time.h>
#include "data_type.h"

namespace cudaprocess{

    // check the ouput of CUDA API function
    #define CHECK_CUDA(call)                                                \
    {                                                                       \
        const cudaError_t error = call;                                     \
        if (error != cudaSuccess) {                                         \
        printf("ERROR: %s:%d,", __FILE__, __LINE__);                        \
        printf("code:%d,reason:%s\n", error, cudaGetErrorString(error));    \
        exit(1);                                                            \
        }                                                                   \
    }

    class CudaDiffEvolveSolver{
        public:
            CudaDiffEvolveSolver(int pop_size = 64, int gpu_device = 0){default_pop_size_ = pop_size; gpu_device_ = gpu_device;};
            ~CudaDiffEvolveSolver();
            void InitDiffEvolvParam(float best = 0.0, float d_top = 0. /*0.002*/, float min_top = 0.0, float diff = 5.0, float d_diff = 0.05, float min_diff = 0.05, float pf = 0.6, float pr = 0.9);
            void Setup(int gpu_device);
            void InitSolver(int con_var_dims, int bin_var_dims);
            void addConstraints();
            void Solver();
        private:
            int gpu_device_;
            int default_pop_size_;
            float best_, d_best_, min_best_;
            float diff_, d_diff_, min_diff_;
            int init_pop_size_, pop_size_;
            int dim_, con_var_dims_, bin_var_dims_;
            float param[CUDA_PARA_MAX_SIZE];
            CudaLShadePair h;
    };
}

#endif