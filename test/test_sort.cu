#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CUDA_PARAM_MAX_SIZE 16
#define T 64  // template parameter for SortParamBasedBitonic

// Error checking macro
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

// Helper function to print arrays
void printArrays(float* fitness, float* params, int size) {
    printf("\nFitness values:\n");
    for(int i = 0; i < size; i++) {
        printf("%.2f ", fitness[i]);
    }
    printf("\n\nParam values (first parameter only):\n");
    for(int i = 0; i < size; i++) {
        printf("%.2f ", params[i * CUDA_PARAM_MAX_SIZE]);
    }
    printf("\n\n");
}

__device__ void BitonicWarpCompare(float &param, float &fitness, int lane_mask){
    float mapping_param = __shfl_xor_sync(0xffffffff, param, lane_mask);
    float mapping_fitness = __shfl_xor_sync(0xffffffff, fitness, lane_mask);
    // determine current sort order is increase (1.0) or decrease (-1.0)
    float sortOrder = (threadIdx.x > (threadIdx.x ^ lane_mask)) ? -1.0 : 1.0;

    if(sortOrder * (mapping_fitness - fitness) < 0.f){
        param = mapping_param;
        fitness = mapping_fitness;
    }
}

// template <int T=64>
__global__ void SortParamBasedBitonic(float *all_param, float *all_fitness){
    if (all_param == nullptr || all_fitness == nullptr) return;
    // each block have a share memory
    __shared__ float sm_sorted_fitness[T];
    __shared__ float sm_sorted_param[T];
    int param_id = blockIdx.x;
    int sol_id = threadIdx.x;
    float current_param = all_param[sol_id * CUDA_PARAM_MAX_SIZE + param_id];
    float current_fitness = all_fitness[sol_id];

    // Sort the contents of 32 threads in a warp based on Bitonic merge sort. Implement detail is the alternative representation of https://en.wikipedia.org/wiki/Bitonic_sorter
    BitonicWarpCompare(current_param, current_fitness, 1);

    BitonicWarpCompare(current_param, current_fitness, 3);
    BitonicWarpCompare(current_param, current_fitness, 1);

    BitonicWarpCompare(current_param, current_fitness, 7);
    BitonicWarpCompare(current_param, current_fitness, 2);
    BitonicWarpCompare(current_param, current_fitness, 1);

    BitonicWarpCompare(current_param, current_fitness, 15);
    BitonicWarpCompare(current_param, current_fitness, 4);
    BitonicWarpCompare(current_param, current_fitness, 2);
    BitonicWarpCompare(current_param, current_fitness, 1);

    // above all finish the sorting 16 threads in Warp, continue to finish 2 group of 16 threads
    BitonicWarpCompare(current_param, current_fitness, 31);
    BitonicWarpCompare(current_param, current_fitness, 8);
    BitonicWarpCompare(current_param, current_fitness, 4);
    BitonicWarpCompare(current_param, current_fitness, 2);
    BitonicWarpCompare(current_param, current_fitness, 1);

    // above all finsh the sort for each warp, continue to finish the sort between different warp by share memory.
    // record the warp sorting result to share memory
    sm_sorted_param[sol_id] = current_param;
    sm_sorted_fitness[sol_id] = current_fitness;

    // Wait for all thread finish above computation
    __syncthreads();

    // if T == 64 (we have 2 warp), we just need to compare these 2 warp by share memory.
    // Otherwise, we need to modify the following code

    int compare_idx = sol_id ^ 63;
    float mapping_param = sm_sorted_param[compare_idx];
    float mapping_fitness = sm_sorted_fitness[compare_idx];

    float sortOrder = (threadIdx.x > (threadIdx.x ^ 63)) ? -1.0 : 1.0;

    if(sortOrder * (mapping_fitness - current_fitness) < 0.f){
        current_param = mapping_param;
        current_fitness = mapping_fitness;
    }
    // Wait for the sort between two warp finish
    __syncthreads();
    // Now, we can come back to the sorting in the warp
    BitonicWarpCompare(current_param, current_fitness, 16);
    BitonicWarpCompare(current_param, current_fitness, 8);
    BitonicWarpCompare(current_param, current_fitness, 4);
    BitonicWarpCompare(current_param, current_fitness, 2);
    BitonicWarpCompare(current_param, current_fitness, 1);

    // above all finish all sorting for fitness and param
    if (blockIdx.x < CUDA_PARAM_MAX_SIZE){
        all_param[sol_id * CUDA_PARAM_MAX_SIZE + param_id] = current_param;
        // printf("======================== Update sorted param for solution id:%d\n", threadIdx.x);
    }
    if (blockIdx.x == 0)    all_fitness[threadIdx.x] = current_fitness;
}

int main() {
    // Host arrays
    float *h_fitness, *h_params;
    // Device arrays
    float *d_fitness, *d_params;

    CHECK_CUDA(cudaSetDevice(0));
    
    // Allocate host memory
    // h_fitness = (float*)malloc(T * sizeof(float));
    // h_params = (float*)malloc(T * CUDA_PARAM_MAX_SIZE * sizeof(float));
    // 主机内存应该使用 cudaHostAlloc 而不是 malloc
    // float *h_fitness, *h_params;
    cudaHostAlloc(&h_fitness, T * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc(&h_params, T * CUDA_PARAM_MAX_SIZE * sizeof(float), cudaHostAllocDefault);
    // Initialize random seed
    srand(time(NULL));
    
    // Initialize fitness with decreasing values
    for(int i = 0; i < T; i++) {
        h_fitness[i] = (float)(T - i);  // Creates values from T down to 1
    }
    
    // Initialize params with random values
    for(int i = 0; i < T * CUDA_PARAM_MAX_SIZE; i++) {
        h_params[i] = (float)rand() / RAND_MAX * 100.0f;  // Random values between 0 and 100
    }
    
    printf("Initial arrays:");
    printArrays(h_fitness, h_params, T);
    
    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_fitness, T * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_params, T * CUDA_PARAM_MAX_SIZE * sizeof(float)));
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_fitness, h_fitness, T * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_params, h_params, T * CUDA_PARAM_MAX_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    // We need CUDA_PARAM_MAX_SIZE blocks because we're sorting each parameter independently
    SortParamBasedBitonic<<<CUDA_PARAM_MAX_SIZE, T>>>(d_params, d_fitness);
    
    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());
    
    // Wait for GPU to finish
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_fitness, d_fitness, T * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_params, d_params, T * CUDA_PARAM_MAX_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("Sorted arrays:");
    printArrays(h_fitness, h_params, T);
    
    // Verify sorting
    bool sorted = true;
    for(int i = 1; i < T; i++) {
        if(h_fitness[i-1] < h_fitness[i]) {
            sorted = false;
            printf("Error: Array not properly sorted at index %d\n", i);
            break;
        }
    }
    if(sorted) {
        printf("Verification: Arrays successfully sorted in descending order!\n");
    }
    
    // Cleanup
    CHECK_CUDA(cudaFreeHost(h_fitness));
    CHECK_CUDA(cudaFreeHost(h_params));
    CHECK_CUDA(cudaFree(d_fitness));
    CHECK_CUDA(cudaFree(d_params));
    
    return 0;
}