#ifndef CUDAPROCESS_DIFF_EVOLUTION_SOLVER_H
#define CUDAPROCESS_DIFF_EVOLUTION_SOLVER_H

#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <sys/time.h>
#include <algorithm>
#include <memory>
#include "diff_evolution_solver/data_type.h"
#include "utils.cuh"
#include "diff_evolution_solver/converter.cuh"
#include "diff_evolution_solver/random_center.cuh"


namespace cudaprocess{
    
    class CudaDiffEvolveSolver{
        public:
            CudaDiffEvolveSolver(int pop_size = 64){default_pop_size_ = pop_size;};
            ~CudaDiffEvolveSolver();
            void MallocSetup();
            void InitDiffEvolveParam(float best = 0.0, float d_top = 0. /*0.002*/, float min_top = 0.0, float diff = 5.0, float d_diff = 0.05, float min_diff = 0.05, float pf = 0.6, float pr = 0.9);
            void WarmStart(ProblemEvaluator* evaluator, CudaParamIndividual* last_sol);
            void InitSolver(int gpu_device, CudaRandomCenter *random_center, ProblemEvaluator* evaluator, CudaParamIndividual* last_sol, const CudaVector<CudaParamIndividual, CUDA_MAX_POTENTIAL_SOLUTION> *last_potential_sol);
            void SetBoundary(ProblemEvaluator* evaluator);
            void Evaluation(int size);
            void Solver();
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
            CudaProblemDecoder *host_decoder_, *decoder_;

            std::shared_ptr<CudaUtil> cuda_utils_;
            CudaParamClusterData<64>* new_cluster_data_;
            CudaParamClusterData<192>* old_cluster_data_;
            CudaParamClusterData<64>* host_new_cluster_data_;
            CudaParamClusterData<192>* host_old_cluster_data_;

            CudaRandomCenter *random_center_;
            ProblemEvaluator *evaluator_;

            CudaSolverInput *host_solver_input_, *device_solver_input_;
    };
}

#endif