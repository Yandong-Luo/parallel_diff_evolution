#ifndef CUDAPROCESS_CART_POLE_MODEL_H
#define CUDAPROCESS_CART_POLE_MODEL_H

#include "diff_evolution_solver/data_type.h"
#include "cart_pole/cart_pole_utils.cuh"
#include <cublas_v2.h>
#include <math.h>
namespace cart_pole{
    // Based on the control input (u) and model of cart pole calculate the next state
    // cart pole model is from https://courses.ece.ucsb.edu/ECE594/594D_W10Byl/hw/cartpole_eom.pdf
    template <int T>
    __global__ void Compute_NonlinearDynamics(float4 current_state, cudaprocess::CudaParamClusterData<T> *cluster_data, cudaprocess::CartStateList *cluster_state){
        if (threadIdx.x > 0)    return;

        int idx = blockIdx.x * N;
        
        for (int i = 0; i < N; ++i){
            float pos = 0.0f, theta = 0.0f, speed = 0.0f, dtheta = 0.0f;
            if(i == 0)  pos = current_state.x, theta = current_state.y, speed = current_state.z, dtheta = current_state.w;
            else    pos = cluster_state->state[(i - 1) + idx].x, theta = cluster_state->state[(i - 1) + idx].y, speed = cluster_state->state[(i - 1) + idx].z, dtheta = cluster_state->state[(i - 1) + idx].w;

            float force = cluster_data[blockIdx.x * CUDA_PARAM_MAX_SIZE + i];

            float acc = (-mp * ll * __sinf(theta) * dtheta * dtheta + mp * g * __cosf(theta) * __sinf(theta) + force) / mc + mp - mp * __cosf(theta) * __cosf(theta);

            float angular_acc = (-mp * ll * __sinf(theta) * __cosf(theta) * dtheta * dtheta + (mc + mp) * g * __sinf(theta) + force * __cosf(theta)) / (mc + mp * (1 - __cosf(theta) * __cosf(theta))) * ll;
            
            // calculate next position
            cluster_state->state[i + idx].x = pos + acc * runtime_step * runtime_step;

            // calculate next theta
            cluster_state->state[i + idx].y = theta + angular_acc * runtime_step * runtime_step;

            // calculate next velocity
            cluster_state->state[i + idx].z = speed + acc * runtime_step;

            // calculate next dtheta
            cluster_state->state[i + idx].w = dtheta + angular_acc * runtime_step;
        }
    }
}

#endif
