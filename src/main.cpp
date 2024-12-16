#include <iostream>
#include <cuda_runtime.h>
#include "solver_center/solver_center.h"
#include "diff_evolution_solver/solver.cuh"
// #include "diff_evolution_solver/data_type.h"

int main(int argc,char** argv){
    // Set the number of tasks and GPU device ID
    // each task use a differential evolution
    int num_enable_tasks = 1;
    int gpu_device = 0;

    // Create an instance of the CudaDiffEvolveSolver class
    cudaprocess::CudaSolverCenter solver_center(gpu_device, num_enable_tasks);

    // Initialize the solver
    solver_center.Init();

    solver_center.ParallelGenerateMultiTaskSol();

    // // Set constraints
    // solver.SetConstraints();

    // // Run the solver
    // solver.Solver();

    // Print the result (assume the final result is stored in the class)
    std::cout << "Solver completed successfully!" << std::endl;

    return 0;
}