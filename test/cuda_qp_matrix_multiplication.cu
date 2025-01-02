#include <stdio.h>
#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <memory>
#include <string.h>
#include <cstdint>

#define N 4    // 向量维度
#define COL 2  // 列数

void printMatrix(float (*matrix)[COL], int row) {
    for(int i = 0; i < row; i++) {
        std::cout << "[ ";
        for(int j = 0; j < COL; j++) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << "]" << std::endl;
    }
}

int main(void) {
    float alpha = 1.0;
    float beta = 0.0;
    
    // 初始化数据
    float h_x[N][COL] = {{1,1},{2,2},{3,3},{4,4}};
    float h_Q[N][N] = {           // 示例对称矩阵
        {1, 0, 0, 0},
        {0, 2, 0, 0},
        {0, 0, 3, 0},
        {0, 0, 0, 4}
    };
    float h_temp[4][2] = {0};   // 临时存储 Qx 的结果
    float h_result[COL][COL] = {0}; // 2x2 结果矩阵

    // 分配设备内存
    float *d_x, *d_Q, *d_temp, *d_result;
    cudaMalloc((void**)&d_x, N * COL * sizeof(float));
    cudaMalloc((void**)&d_Q, N * N * sizeof(float));
    cudaMalloc((void**)&d_temp, N * COL * sizeof(float));
    cudaMalloc((void**)&d_result, COL * COL * sizeof(float));

    // 拷贝数据到设备
    cudaMemcpy(d_x, h_x, N * COL * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q, h_Q, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_temp, 0, N * COL * sizeof(float));

    // 创建 CUBLAS 句柄
    cublasHandle_t handle;
    cublasCreate(&handle);

    // 步骤1: Qx
    cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,  // Q * x
        2, 4, 4,
        &alpha,
        d_x, 2,
        d_Q, 4,   // Q: N×N
        &beta,
        d_temp, 2 // Qx: N×COL
    );

    // 步骤2: x^T * (Qx)
    cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_T,  // 注意这里的转置
        2, 2, 4,
        &alpha,
        d_x, 2,      
        d_temp, 2,   
        &beta,
        d_result, 2 
    );
    // 拷贝结果回主机
    cudaMemcpy(h_temp, d_temp, 4 * 2 * sizeof(float), cudaMemcpyDeviceToHost);
    // 拷贝结果回主机
    cudaMemcpy(h_result, d_result, COL * COL * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果
    std::cout << "X matrix:" << std::endl;
    printMatrix(h_x, N);
    
    std::cout << "\nQ matrix:" << std::endl;
    for(int i = 0; i < N; i++) {
        std::cout << "[ ";
        for(int j = 0; j < N; j++) {
            std::cout << h_Q[i][j] << " ";
        }
        std::cout << "]" << std::endl;
    }
    
    std::cout << "\nResult matrix (2x2):" << std::endl;
    for(int i = 0; i < COL; i++) {
        std::cout << "[ ";
        for(int j = 0; j < COL; j++) {
            std::cout << h_result[i][j] << " ";
        }
        std::cout << "]" << std::endl;
    }

    std::cout << "\nResult matrix (2x2):" << std::endl;
    for(int i = 0; i < 4; i++) {
        std::cout << "[ ";
        for(int j = 0; j < 2; j++) {
            std::cout << h_temp[i][j] << " ";
        }
        std::cout << "]" << std::endl;
    }

    // 清理资源
    cudaFree(d_x);
    cudaFree(d_Q);
    cudaFree(d_temp);
    cudaFree(d_result);
    cublasDestroy(handle);

    return 0;
}