#ifndef CUDAPROCESS_DIFF_EVOLUTION_SOLVER_H
#define CUDAPROCESS_DIFF_EVOLUTION_SOLVER_H

#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <sys/time.h>

#include <memory>
#include "data_type.h"
#include "utils.cuh"

namespace cudaprocess{

    class CudaDiffEvolveSolver{
        public:
            CudaDiffEvolveSolver(int pop_size = 64){default_pop_size_ = pop_size;};
            ~CudaDiffEvolveSolver();
            void MallocSetup();
            void InitDiffEvolveParam(float best = 0.0, float d_top = 0. /*0.002*/, float min_top = 0.0, float diff = 5.0, float d_diff = 0.05, float min_diff = 0.05, float pf = 0.6, float pr = 0.9);
            void WarmStart(ProblemEvaluator* evaluator, CudaParamIndividual* last_sol);
            void InitSolver(int gpu_device, ProblemEvaluator* evaluator, CudaParamIndividual* last_sol, const CudaVector<CudaParamIndividual, CUDA_MAX_POTENTIAL_SOLUTION> *last_potential_sol);
            void SetConstraints();
            void Solver();
        private:
            int gpu_device_;
            int default_pop_size_;
            float best_, d_best_, min_best_;
            float diff_, d_diff_, min_diff_;
            int init_pop_size_, pop_size_;
            int dims_, con_var_dims_, bin_var_dims_;
            bool cudamalloc_flag{false};
            
            float upper_bound_[CUDA_PARAM_MAX_SIZE], lower_bound_[CUDA_PARAM_MAX_SIZE];
            CudaVector<CudaParamIndividual, CUDA_SOLVER_POP_SIZE> *new_cluster_vec_;
            CudaLShadePair lshade_param_;

            CudaEvolveData *host_evolve_data_, *evolve_data_;
            CudaProblemDecoder *host_decoder_, *decoder_;

            std::shared_ptr<CudaUtil> cuda_utils_;

            CudaParamClusterData<64>* new_cluster_data_;
            CudaParamClusterData<192>* old_cluster_data_;
    };
}

#endif