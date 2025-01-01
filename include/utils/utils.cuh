#ifndef CUDA_PROCESS_UTIL_H
#define CUDA_PROCESS_UTIL_H

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "diff_evolution_solver/data_type.h"

namespace cudaprocess {

    
    #define DEBUG_PRINT_FLAG false
    #define DEBUG_PRINT_EVALUATE_FLAG true
    #define DEBUG_PRINT_SOLVER_FLAG false
    #define DEBUG_PRINT_INIT_SOLVER_FLAG true
    #define DEBUG_ENABLE_NVTX true

    #define HOST_DEVICE __device__ __forceinline__ __host__
    #define CUDA_SOLVER_POP_SIZE 64

    #define INT_VARIABLE 0
    #define CONTINUOUS_VARIABLE 1

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

    constexpr int stream_cnt = 7;
    struct CudaUtil {
    cudaStream_t streams_[stream_cnt];
    cudaEvent_t events_[stream_cnt];
    float *tmp_res;
    int *tmp_invalid_flag;

    CudaUtil() {
        for (int i = 0; i < stream_cnt; ++i) {
            CHECK_CUDA(cudaStreamCreate(&streams_[i]));
            CHECK_CUDA(cudaEventCreateWithFlags(&events_[i], cudaEventDisableTiming));
        }
        CHECK_CUDA(cudaMalloc(&tmp_res, sizeof(float) * CUDA_SOLVER_POP_SIZE * stream_cnt));
        CHECK_CUDA(cudaMalloc(&tmp_invalid_flag, sizeof(int) * CUDA_SOLVER_POP_SIZE));
    }

    ~CudaUtil() {
        for (int i = 0; i < stream_cnt; ++i) {
            CHECK_CUDA(cudaStreamDestroy(streams_[i]));
            CHECK_CUDA(cudaEventDestroy(events_[i]));
        }
        CHECK_CUDA(cudaFree(tmp_res));
        CHECK_CUDA(cudaFree(tmp_invalid_flag));
    }
};

}

#endif