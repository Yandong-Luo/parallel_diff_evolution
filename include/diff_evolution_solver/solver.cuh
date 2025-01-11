#ifndef CUDAPROCESS_DIFF_EVOLUTION_SOLVER_H
#define CUDAPROCESS_DIFF_EVOLUTION_SOLVER_H

#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <sys/time.h>
#include <algorithm>
#include <memory>
#include <nvtx3/nvtx3.hpp>
#include <cublas_v2.h>
#include "diff_evolution_solver/data_type.h"
#include "utils/utils.cuh"
#include "diff_evolution_solver/converter.cuh"
#include "diff_evolution_solver/random_center.cuh"
#include "diff_evolution_solver/random_manager.cuh"

namespace cudaprocess{
    
    class CudaDiffEvolveSolver{
        public:
            CudaDiffEvolveSolver(int pop_size = 64){default_pop_size_ = pop_size;};
            ~CudaDiffEvolveSolver();
            void MallocSetup();
            void MallocReset();
            void InitDiffEvolveParam(float best = 0.0, float d_top = 0. /*0.002*/, float min_top = 0.0, float diff = 5.0, float d_diff = 0.05, float min_diff = 0.05, float pf = 0.6, float pr = 0.9);
            void WarmStart(Problem* problem, CudaParamIndividual* last_sol);
            void InitSolver(int gpu_device);
            void SetBoundary();
            void Evaluation(int size, int epoch);
            void Evolution(int epoch, CudaEvolveType search_type);
            CudaParamIndividual Solver();

            void UpdateCartPoleState(float state[4]);
        private:
            int gpu_device_;
            int default_pop_size_;
            float top_, d_top_, min_top_;
            float diff_, d_diff_, min_diff_;
            int init_pop_size_, pop_size_;
            int dims_, con_var_dims_, int_var_dims_;
            bool cudamalloc_flag{false};
            
            float host_upper_bound_[CUDA_PARAM_MAX_SIZE], host_lower_bound_[CUDA_PARAM_MAX_SIZE];
            CudaVector<CudaParamIndividual, CUDA_SOLVER_POP_SIZE> *new_cluster_vec_;
            CudaVector<CudaParamIndividual, CUDA_MAX_POTENTIAL_SOLUTION> last_potential_sol_;
            CudaLShadePair lshade_param_;

            CudaEvolveData *host_evolve_data_, *evolve_data_;

            std::shared_ptr<CudaUtil> cuda_utils_;
            CudaParamClusterData<64>* new_cluster_data_;
            CudaParamClusterData<192>* old_cluster_data_;       // record current sol, delete sol, replaced sol
            CudaParamClusterData<64>* host_new_cluster_data_;
            CudaParamClusterData<192>* host_old_cluster_data_;

            // CudaRandomManager *random_center_;
            std::shared_ptr<CudaRandomManager> random_center_;

            float *param_matrix, *host_param_matrix;

            float *constraint_matrix;
            float *objective_matrix;
            float *lambda_matrix;
            float *objective_Q_matrix;

            float *h_constraint_matrix;
            float *h_objective_matrix;
            float *h_lambda_matrix;
            float *h_objective_Q_matrix;

            int row_constraint, col_constraint;
            int row_obj, col_obj;
            int row_lambda, col_lambda;
            int row_obj_Q, col_obj_Q;
            float *evaluate_score_, *host_evaluate_score_;
            float *constraint_score, *host_constraint_score;
            float *quad_matrix, *host_quad_matrix;
            float *quad_transform, *h_quad_transform;
            float *quadratic_score, *h_quadratic_score;


            cublasHandle_t cublas_handle_; 
            int max_lambda;
            CudaParamIndividual *result;
            CudaParamIndividual *host_result;

            float accuracy_rng;
            float *last_fitness;
            int *terminate_flag, *h_terminate_flag;

            // NVTX
            nvtxRangeId_t solver_range;
            nvtxRangeId_t init_range;
            nvtxRangeId_t setting_boundary_range;

            nvtxRangeId_t loading_last_sol_range;

            int task_id_ = 0;

            // !--------------- CART POLE ---------------!
            float4 state;
            float *h_state;             // pos, speed, theta, angular velocity from environment (x in paper)
            float *env_constraint, *h_env_constraint;     // h(\theta) in paper

            float *C_matrix, *h_C_matrix;
            float *A_matrix, *h_A_matrix;
    };
}

#endif