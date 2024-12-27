#include "diff_evolution_solver/random_center.cuh"

namespace cudaprocess{
    __global__ void RndInit(curandState_t *states, long long unsigned seed, int total_size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < total_size) {
            curand_init(seed, idx, idx, &states[idx]);
        }
    }

    __global__ void GenUniformRandom(curandState_t *states, float *res, int total_size) {
        int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 10;
        int state_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx + 9 < total_size) {
            curandState_t localState = states[state_idx];  // 本地副本提高效率
        #pragma unroll
            for (int i = 0; i < 10; ++i) {
            res[idx + i] = curand_uniform(&localState);
            if (res[idx + i] >= 1.0) {
                res[idx + i] = 0.;
            }
            }
        }
    }

    __global__ void GenNormalRandom(curandState_t *states, float *res, int total_size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < total_size) {
            res[idx] = curand_normal(&states[idx]);
        }
    }

    // 添加一个辅助kernel来检查states的初始化
    __global__ void CheckStates(curandState_t *states, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            // 生成一个测试随机数
            float test = curand_uniform(&states[idx]);
            printf("State[%d] test random: %f\n", idx, test);
        }
    }

    CudaRandomCenter::CudaRandomCenter(int gpu_device)
    {
        normal_size_ = size_ * 3 * CUDA_MAX_ROUND_NUM;
        uniform_size_ = 512000;
        int state_size_ = 51200;
        CHECK_CUDA(cudaSetDevice(gpu_device));
        CHECK_CUDA(cudaMalloc(&states_, sizeof(curandState_t) * state_size_));
        CHECK_CUDA(cudaMalloc(&uniform_data_, sizeof(float) * uniform_size_));
        CHECK_CUDA(cudaMalloc(&normal_data_, sizeof(float) * normal_size_));
        // auto seed_ = 0;
        // 使用时间作为种子
        auto seed_ = static_cast<long long unsigned>(time(nullptr));
        RndInit<<<50, 1024>>>(states_, seed_, state_size_);

        normal_grid_size_ = (normal_size_ - 1) / 1024 + 1;
        // uniform_grid_size_ = (uniform_size_ - 1) / 1024 + 1;
        uniform_grid_size_ = 50;
        Generate();
        cudaDeviceSynchronize();

        printf("\nChecking random states:\n");
        CheckStates<<<50, 1024>>>(states_, state_size_);
        cudaDeviceSynchronize();
    }
    
    CudaRandomCenter::~CudaRandomCenter()
    {
        CHECK_CUDA(cudaFree(states_));
        CHECK_CUDA(cudaFree(uniform_data_));
        CHECK_CUDA(cudaFree(normal_data_));
    }

    void CudaRandomCenter::Generate() {
        GenUniformRandom<<<uniform_grid_size_, 1024>>>(states_, uniform_data_, uniform_size_);
        GenNormalRandom<<<normal_grid_size_, 1024>>>(states_, normal_data_, normal_size_);
        cudaDeviceSynchronize();
    }

    __device__ __forceinline__ float UniformReal(curandState_t *states, float l, float r) { return l + curand_uniform(states) * (r - l); }

    __device__ __forceinline__ float NormalReal(curandState_t *states, float miu, float sigma) { return curand_normal(states) * sigma + miu; }

    __device__ __forceinline__ int UniformInt(curandState_t *states, int l, int r) {
        int tmp = l + curand_uniform(states) * (r + 1 - l);
        if (tmp == r + 1) {
            tmp = r;
        }
        return tmp;
    }
}