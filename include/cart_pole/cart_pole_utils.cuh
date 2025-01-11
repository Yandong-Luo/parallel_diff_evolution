#ifndef CUDAPROCESS_CART_POLE_UTILS_H
#define CUDAPROCESS_CART_POLE_UTILS_H

namespace cart_pole{

// CONSTANT
const int N = 10;                // prediction step
const float runtime_step = 0.02f;          // Delta t
const float mc = 1.0f;
const float mp = 0.4f;
const float ll = 0.6f;
const float k1 = 50.0f;
const float k2 = 50.0f;

constexpr float PI = 3.14159265358979323846f;

// the boundary of theta
const float cart_lb = -0.6f;
const float cart_ub = 0.6f;

// the boundary of theta
const float theta_lb = -PI/2;
const float theta_ub = PI/2;

// the boundary of speed
const float speed_lb = -2 * (cart_ub - cart_lb) / runtime_step;
const float speed_ub = -2 * (cart_ub - cart_lb) / runtime_step;

// the boundary of \dot theta
const float dtheta_lb = -PI / runtime_step;
const float dtheta_ub = PI / runtime_step;

// the boundary of control input (u)
const float u_lb = -20.0f;
const float u_ub = 20.0f;

const float d_left = 0.40f;
const float d_right = 0.35f;
const float d_max = 0.6f;
const float lam_max = 30.0f;

const float g = 9.81f;
const int num_constraints = 20;     // n_c in paper
const int state_dims = 4;           // n_x in paper
const int control_input_dims = 3;   // n_u in paper
const int u_dims = 1;

const int row_C = N * num_constraints;
const int col_C = N * (state_dims + control_input_dims) + state_dims;       // 

const int row_A = (N + 1) * state_dims;                                     // (N + 1) * nx in paper
const int col_A = N * (state_dims + control_input_dims) + state_dims;       // N * n_{xu} + nx in paper

// E Matrix (4x4), Row priority
const int row_E = 4, col_E = 4;
__constant__ float E[16] = {
    1.0f + 0.0f*runtime_step, 0.0f*runtime_step,        runtime_step,      0.0f*runtime_step,
    0.0f*runtime_step,        1.0f + 0.0f*runtime_step, 0.0f,    runtime_step,
    0.0f*runtime_step,        g*mp/mc*runtime_step,      1.0f,    0.0f*runtime_step,
    0.0f*runtime_step,        g*(mc+mp)/(ll*mc)*runtime_step, 0.0f, 1.0f
};

// Matrix F (4x3), Row priority
const int row_F = 4, col_F = 3;
__constant__ float F[12] = {
    0.0f,         0.0f,         0.0f,
    0.0f,         0.0f,         0.0f,
    runtime_step/mc,        0.0f,         0.0f,
    runtime_step/(ll*mc),   runtime_step/(ll*mp),  -runtime_step/(ll*mp)
};

// Matrix G (4x2), Row priority
const int row_G = 4, col_G = 2;
__constant__ float G[8] = {
    0.0f, 0.0f,
    0.0f, 0.0f,
    0.0f, 0.0f,
    0.0f, 0.0f
};

// Q M (4x4), Row priority
const int row_Q = 4, col_Q = 4;
__constant__ float Q[16] = {
    1.0f,  0.0f,  0.0f,  0.0f,
    0.0f, 50.0f,  0.0f,  0.0f,
    0.0f,  0.0f,  1.0f,  0.0f,
    0.0f,  0.0f,  0.0f, 50.0f
};

// R Matrix (3x3), Row priority
const int row_R = 3, col_R = 3;
__constant__ float R[9] = {
    0.1f, 0.0f, 0.0f,
    0.0f, 0.1f, 0.0f,
    0.0f, 0.0f, 0.1f
};

// H1 Matrix (20x4), Row priority
const int row_H1 = 20, col_H1 = 4;
__constant__ float H1[80] = {
    0.0f,  0.0f,  0.0f,  0.0f,
    0.0f,  0.0f,  0.0f,  0.0f,
    -1.0f,  ll,   0.0f,  0.0f,
    1.0f,  -ll,   0.0f,  0.0f,
    1.0f,  -ll,   0.0f,  0.0f,
    -1.0f,  ll,   0.0f,  0.0f,
    1.0f,  0.0f,  0.0f,  0.0f,
    -1.0f, 0.0f,  0.0f,  0.0f,
    0.0f,  1.0f,  0.0f,  0.0f,
    0.0f, -1.0f,  0.0f,  0.0f,
    0.0f,  0.0f,  1.0f,  0.0f,
    0.0f,  0.0f, -1.0f,  0.0f,
    0.0f,  0.0f,  0.0f,  1.0f,
    0.0f,  0.0f,  0.0f, -1.0f,
    0.0f,  0.0f,  0.0f,  0.0f,
    0.0f,  0.0f,  0.0f,  0.0f,
    0.0f,  0.0f,  0.0f,  0.0f,
    0.0f,  0.0f,  0.0f,  0.0f,
    0.0f,  0.0f,  0.0f,  0.0f,
    0.0f,  0.0f,  0.0f,  0.0f
};

// H2 Matrix (20x3), Row priority
const int row_H2 = 20, col_H2 = 3;
__constant__ float H2[60] = {
    0.0f,     1.0f,     0.0f,
    0.0f,     0.0f,     1.0f,
    0.0f,  1.0f/k1,     0.0f,
    0.0f, -1.0f/k1,     0.0f,
    0.0f,     0.0f,  1.0f/k2,
    0.0f,     0.0f, -1.0f/k2,
    0.0f,     0.0f,     0.0f,
    0.0f,     0.0f,     0.0f,
    0.0f,     0.0f,     0.0f,
    0.0f,     0.0f,     0.0f,
    0.0f,     0.0f,     0.0f,
    0.0f,     0.0f,     0.0f,
    0.0f,     0.0f,     0.0f,
    0.0f,     0.0f,     0.0f,
    1.0f,     0.0f,     0.0f,
    -1.0f,    0.0f,     0.0f,
    0.0f,     1.0f,     0.0f,
    0.0f,    -1.0f,     0.0f,
    0.0f,     0.0f,     1.0f,
    0.0f,     0.0f,    -1.0f
};

// H3 Matrix (20x2), Row priority
const int row_H3 = 20, col_H3 = 2;
__constant__ float H3[40] = {
    -lam_max,      0.0f,
         0.0f, -lam_max,
       d_max,      0.0f,
         0.0f,      0.0f,
         0.0f,    d_max,
         0.0f,      0.0f,
         0.0f,      0.0f,
         0.0f,      0.0f,
         0.0f,      0.0f,
         0.0f,      0.0f,
         0.0f,      0.0f,
         0.0f,      0.0f,
         0.0f,      0.0f,
         0.0f,      0.0f,
         0.0f,      0.0f,
         0.0f,      0.0f,
         0.0f,      0.0f,
         0.0f,      0.0f,
         0.0f,      0.0f,
         0.0f,      0.0f
};

const int row_Inx = state_dims, col_Inx = state_dims;
__constant__ float Inx[16] = {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f
};
}

#endif