#include "solver_center/solver_center.h"

namespace cudaprocess{

void CudaSolverCenter::Init(){
    CHECK_CUDA(cudaHostAlloc(&tasks_best_sol_, sizeof(CudaParamIndividual) * CUDA_MAX_TASKS, cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&tasks_potential_sol_, sizeof(CudaVector<CudaParamIndividual, CUDA_MAX_POTENTIAL_SOLUTION>) * CUDA_MAX_TASKS, cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&tasks_evaluator_, sizeof(ProblemEvaluator) * CUDA_MAX_TASKS, cudaHostAllocDefault));
    // for (int i = 0; i < num_enable_tasks; ++i){
    //     diff_evolve_solvers_[i].MallocSetup()
    // }
    // cudamalloc_flag = true;
}

void CudaSolverCenter::ParallelGenerateMultiTaskSol(){
    for (int i = 0; i < num_enable_tasks_; ++i){
        GenerateSolution(i);
    }
}

void CudaSolverCenter::GenerateSolution(int task_id){
    diff_evolve_solvers_[task_id].InitSolver(gpu_device_, &tasks_evaluator_[task_id], &tasks_best_sol_[task_id], &tasks_potential_sol_[task_id]);
}

CudaSolverCenter::~CudaSolverCenter(){
    // if (cudamalloc_flag){
    //     CHECK_CUDA(cudaFree())
    // }
}
}