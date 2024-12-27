#include "solver_center/solver_center.h"

namespace cudaprocess{

void CudaSolverCenter::Init(){
    rnd_manager_ = std::make_shared<CudaRandomCenter>(gpu_device_);
    CHECK_CUDA(cudaHostAlloc(&tasks_best_sol_, sizeof(CudaParamIndividual) * CUDA_MAX_TASKS, cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&tasks_potential_sol_, sizeof(CudaVector<CudaParamIndividual, CUDA_MAX_POTENTIAL_SOLUTION>) * CUDA_MAX_TASKS, cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&tasks_evaluator_, sizeof(ProblemEvaluator) * CUDA_MAX_TASKS, cudaHostAllocDefault));
    // for (int i = 0; i < num_enable_tasks; ++i){
    //     diff_evolve_solvers_[i].MallocSetup()
    // }
    // cudamalloc_flag = true;

    for(int i = 0; i < CUDA_MAX_TASKS; ++i) {
        ProblemEvaluator evaluator;  // 创建一个带默认值的结构体
        tasks_evaluator_[i] = evaluator;  // 复制到分配的内存中
    }
}

void CudaSolverCenter::ParallelGenerateMultiTaskSol(){
    for (int i = 0; i < num_enable_tasks_; ++i){
        GenerateSolution(i);
    }
}

void CudaSolverCenter::GenerateSolution(int task_id){
    printf("tasks_evaluator_[task_id]:%d\n", tasks_evaluator_[task_id].num_int_variable);
    diff_evolve_solvers_[task_id].InitSolver(gpu_device_, rnd_manager_.get(), &tasks_evaluator_[task_id], &tasks_best_sol_[task_id], &tasks_potential_sol_[task_id]);
    // diff_evolve_solvers_[task_id].Solver(1);
}

CudaSolverCenter::~CudaSolverCenter(){
    // if (cudamalloc_flag){
    //     CHECK_CUDA(cudaFree())
    // }
}
}