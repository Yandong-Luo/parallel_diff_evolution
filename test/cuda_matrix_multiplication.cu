// #include <stdio.h>
// #include <assert.h>
// #include <fstream>
// #include <sstream>
// #include <iostream>
// #include <cmath>
// #include <sys/stat.h>
// #include <cmath>
// #include <time.h>
// #include <cuda_runtime_api.h>
// #include <cublas_v2.h>
// #include <memory>
// #include <string.h>
// #include <cstdint>
 
// #define M 4
// #define N 1
// #define K 3
 
// void printMatrix(float (*matrix)[N], int row, int col) {
//     for(int i=0;i<row;i++)
//     {
//         std::cout << std::endl;
//         std::cout << " [ ";
//         for (int j=0; j<col; j++) {
//          std::cout << matrix[i][j] << " ";
//         }
//         std::cout << " ] ";
//     }
//     std::cout << std::endl;
// }
 
// int main(void)
// {   
//     // https://claude.ai/chat/487387e8-c06e-42f9-b946-cf6a4846652a
//     float alpha=1.0;
//     float beta=1.0;
//     float h_param[M][K]={ {1,2,3},{1,2,3},{1,2,3},{1,2,3} };
//     float h_obj[K][N]={ {-4}, {-3}, {-5}};
//     float h_obj_constant[M][N] = {0};
//     // float h_obj_constant[4] = {0};
//     // float h_param[K]={ 1,2,3 };
//     // float h_obj[K]={ -4 -3 -5};
//     // float h_obj_constant[M][N] = {0};
//     float *d_param,*d_obj,*d_obj_constant;
//     cudaMalloc((void**)&d_param,M*K*sizeof(float));
//     cudaMalloc((void**)&d_obj,K*N*sizeof(float));
//     cudaMalloc((void**)&d_obj_constant,M*N*sizeof(float));
//     cudaMemcpy(d_param,&h_param,M*K*sizeof(float),cudaMemcpyHostToDevice);
//     cudaMemcpy(d_obj,&h_obj,K*N*sizeof(float),cudaMemcpyHostToDevice);
//     cudaMemcpy(d_obj_constant,&h_obj_constant,M*N*sizeof(float), cudaMemcpyHostToDevice);
//     cublasHandle_t handle;
//     cublasCreate(&handle);

//     printf("CHECK OBJ CONSTANT MATRIX\n");
//     cudaMemcpy(h_obj_constant, d_obj_constant, M * N * sizeof(float), cudaMemcpyDeviceToHost);
//     printMatrix(h_obj_constant, M, N);
//     // for(int i = 0; i < M; ++i){
//     //     for(int j = 0; j < N; ++j){
//     //         // printf("row:%d col:%d objective mat:%f ", i, j, h_objective_matrix[i * col_obj +j]);
//     //         printf("obj_constant_matrix[%d,%d]=%f ", i, j, h_obj_constant[i * N + j]);
//     //     }
//     //     printf("\n");
//     // }

//     printf("%d %d %d %d %d %d\n", M, N, K, K,N,M);
//     // cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_T, M, N, K,&alpha,d_obj, N, d_param, K,&beta, d_obj_constant, N);
//     cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, 
//     M,    // M=64 
//     N,    // N=1
//     K,    // K=3 
//     &alpha,
//     d_param, K,   // A的leading dimension=3
//     d_obj, N,     // B的leading dimension=1  
//     &beta,
//     d_obj_constant, M); // C的leading dimension=64
//     // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
//     // N,    // C的列数 = 1
//     // M,    // C的行数 = 64
//     // K,    // A的列数 = 3
//     // &alpha,
//     // d_obj, N,        // B的主维度 = 3
//     // d_param, K,      // A的主维度 = 64
//     // &beta, 
//     // d_obj_constant, N); // C的主维度 = 1
//     cudaMemcpy(h_obj_constant,d_obj_constant,M*N*sizeof(float),cudaMemcpyDeviceToHost);//此处的h_C是按列存储的CT
//     printMatrix(h_obj_constant, M, N);//按行读取h_C相当于做了CTT=C的结果
//     return 0;
// }


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
 
#define M 2
#define N 2
#define K 4


void printMatrix2(float* matrix, int row, int col) {
    for(int i=0;i<row;i++)
    {
        std::cout << std::endl;
        std::cout << " [ ";
        for (int j=0; j<col; j++) {
         std::cout << matrix[i * col + j] << " ";
        }
        std::cout << " ] ";
    }
    std::cout << std::endl;
}

void printMatrix(float (*matrix)[N], int row, int col) {
    for(int i=0;i<row;i++)
    {
        std::cout << std::endl;
        std::cout << " [ ";
        for (int j=0; j<col; j++) {
         std::cout << matrix[i][j] << " ";
        }
        std::cout << " ] ";
    }
    std::cout << std::endl;
}
 
int main(void)
{
        float alpha=1.0;
        float beta=1.0;
        // float h_A[M][K]={ {1,2,3}, {4,5,6} };
        // float h_B[K][N]={ {1,2,3,4}, {5,6,7,8}, {9,10,11,12} };
        // float h_C[M][N] = {0};

        float h_A[M][K]= {{1.219662, 1.467541, 10.000000, 1.000000},{7.550661, 3.893548, 17.858698, 1.000000}};
        float h_B[K][N]= {{2, 2}, {3, 1}, {1, 3}, {-12, -12}};
        float h_C[M][N] = {0};
        float result[2];
        float *d_a,*d_b,*d_c;
        float host_A[4], host_B[8];
        cudaMalloc((void**)&d_a,M*K*sizeof(float));
        cudaMalloc((void**)&d_b,K*N*sizeof(float));
        cudaMalloc((void**)&d_c,M*N*sizeof(float));
        cudaMemcpy(d_a,&h_A,M*K*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(d_b,&h_B,K*N*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(d_c,h_C,M*N*sizeof(float), cudaMemcpyHostToDevice);

        cudaMemcpy(result, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_A, d_a, M * K * sizeof(float), cudaMemcpyDeviceToHost);
        printMatrix2(host_A, M, K);
        cudaMemcpy(host_B, d_b, K * N * sizeof(float), cudaMemcpyDeviceToHost);
        printMatrix2(host_B, K, N);

        cublasHandle_t handle;
        cublasCreate(&handle);
        printf("CHECK THE PARAM OF cublasSgemm: %d %d %d %d %d %d\n", M, N, K, K, N , M);
        cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N, N, M, K,&alpha, d_b, N, d_a, K,&beta, d_c, N);
        cudaMemcpy(result,d_c,M*N*sizeof(float),cudaMemcpyDeviceToHost);//此处的h_C是按列存储的C
        printMatrix2(result, M, N);//按行优先N行M列的顺序读取h_C相当于做了CT的结果
        return 0;
}