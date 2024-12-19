#ifndef CUDAPROCESS_SOLVER_CENTER_H
#define CUDAPROCESS_SOLVER_CENTER_H
#include <cassert>
#include "diff_evolution_solver/data_type.h"
#include "diff_evolution_solver/solver.cuh"
#include "diff_evolution_solver/random_center.cuh"

namespace cudaprocess{

    class CudaSolverCenter
    {
    private:
        int gpu_device_, num_enable_tasks_;
        CudaParamIndividual *tasks_best_sol_;
        CudaVector<CudaParamIndividual, CUDA_MAX_POTENTIAL_SOLUTION> *tasks_potential_sol_;
        CudaDiffEvolveSolver diff_evolve_solvers_[CUDA_MAX_TASKS];
        ProblemEvaluator *tasks_evaluator_;
        std::shared_ptr<CudaRandomCenter> rnd_manager_;

        bool cudamalloc_flag{false};
    public:
        CudaSolverCenter(int gpu_device, int num_enable_tasks){gpu_device_ = gpu_device; num_enable_tasks_ = num_enable_tasks;};
        ~CudaSolverCenter();
        void Init();
        void GenerateSolution(int task_id);
        void ParallelGenerateMultiTaskSol();

        ProblemEvaluator *GetEvaluator(int task_id){
            assert(task_id < task_id);
            return &tasks_evaluator_[task_id];
        };
        
        CudaVector<CudaParamIndividual, CUDA_MAX_POTENTIAL_SOLUTION> *GetPotentialSol(int task_id) {
            assert(task_id < num_enable_tasks_);
            return &tasks_potential_sol_[task_id];
        };
        
        CudaParamIndividual *GetBestSol(int task_id) {
            assert(task_id < num_enable_tasks_);
            return &tasks_best_sol_[task_id];
        };
    };

}

#endif