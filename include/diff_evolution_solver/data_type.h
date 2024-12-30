#ifndef CUDA_DIFF_EVOLUTION_DATA_TYPE_H
#define CUDA_DIFF_EVOLUTION_DATA_TYPE_H

#include "utils/utils.cuh"
namespace cudaprocess{

#define ALIGN(n) __align__(n)

#define CUDA_PARAM_MAX_SIZE 16
#define CUDA_SOLVER_POP_SIZE 64
#define CUDA_MAX_FLOAT 1e30
#define CUDA_MAX_TASKS 4
#define CUDA_MAX_POTENTIAL_SOLUTION 4
#define CUDA_MAX_ROUND_NUM 100
#define CUDA_MAX_NUM_CONSTRAINT 10

enum CudaEvolveType { CON = 0, INT = 1, GLOBAL = 2 };


template <typename T, int size>
struct CudaVector {
  T data[size];
  int len{0};
};

struct Problem{
    int num_con_variable = 2;
    int num_int_variable = 1;

    float int_upper_bound[1] = {20};
    float int_lower_bound[1] = {0};
    float con_upper_bound[2] = {10., 10.};
    float con_lower_bound[2] = {0, 0};

    // matrix dims
    int row_obj = 4, col_obj = 1;
    int row_constraint_mat = 4, col_constraint_mat = 2;   // row x col should equal to num_constraint x constraint variable + 1 (constant).
    int row_lambda = 2, col_lambda = 1;

    float obj_mat[4][1] = {{-4}, {-3}, {-5}, {0}};
    float constraint_mat[4][2] = {{2, 2}, {3, 1}, {1, 3}, {-12, -12}};

    float lambda[2] = {10, 10};

    float constraint_param[2][4] = {
        {2, 3, 1, -12},
        {2, 1, 3, -12}
    };

    int max_lambda = 100;
    int init_lambda = 1;
    int max_round = 60;

    float accuracy_rng = 0.5;
    int elite_eval_count = 8;

    // // (Abandoned) Use for loop to evaluate 
    // int num_constraint = 2;
    // int num_constraint_variable = 4;
    // // objective function
    // float objective_param[3] = {-4., -3, -5};
    // int num_objective_param = 3;
};

/*
 * Param:
 * prob_f:  scaling factor for L-Shape 
 * Cr:      Crossover Probability
 * weight:  weight factor
 */
struct ALIGN(16) CudaLShadePair {
    float scale_f, scale_f1, Cr, weight;
};

struct CudaParamIndividual{
    float param[CUDA_PARAM_MAX_SIZE];
    int con_var_dims, int_var_dims, dims;
    float cur_scale_f{0.5F}, cur_scale_f1{0.5F}, cur_Cr{0.5F};
    float fitness;
};

/*
 * Store individual variables in a cluster in a more compact way
 */
template <int T>
struct ALIGN(64) CudaParamClusterData{
    float all_param[CUDA_PARAM_MAX_SIZE * T];
    float lshade_param[T * 3];  // current scale_f, current scale_f1, current crossover probability
    int con_var_dims, int_var_dims, dims;
    float fitness[T];
    int len{0};
};

struct ALIGN(64) CudaProblemParam{
    int con_var_dims, int_var_dims, dims;
    int max_lambda;
    int init_lambda;
    int max_round;

    float accuracy_rng;
    int elite_eval_count;

    float top_ratio;
};

struct CudaEvolveData{
    
    CudaLShadePair hist_lshade_param;
    ALIGN(64) float upper_bound[CUDA_PARAM_MAX_SIZE];
    ALIGN(64) float lower_bound[CUDA_PARAM_MAX_SIZE];
    CudaVector<CudaParamIndividual, CUDA_MAX_POTENTIAL_SOLUTION> last_potential_sol;
    CudaParamIndividual warm_start;

    CudaProblemParam problem_param;

    // (Abandoned) Use for loop to evaluate 
    // int num_constraint;
    // float constraint_para[CUDA_MAX_NUM_CONSTRAINT][CUDA_PARAM_MAX_SIZE + 1];
    // int num_constraint_variable;
    // float objective_param[CUDA_PARAM_MAX_SIZE];
    // int num_objective_param;
};

}


#endif