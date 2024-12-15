#include <iostream>
#include <cuda_runtime.h>
#include "diff_evolution_solver/solver.cuh"
// #include "diff_evolution_solver/data_type.h"

int main(int argc,char** argv){
    // Initialize the problem evaluator (assume ProblemEvaluator is problem-specific)
    cudaprocess::ProblemEvaluator evaluator;

    // Initialize the last solution (assume it's a null pointer or some initial value)
    cudaprocess::CudaParamIndividual* last_solution = nullptr;

    // Set the population size and GPU device ID
    int pop_size = 64;   // Population size
    int gpu_device = 0;  // Default GPU device ID

    // Create an instance of the CudaDiffEvolveSolver class
    cudaprocess::CudaDiffEvolveSolver solver(pop_size, gpu_device);

    // Initialize the solver
    solver.InitSolver(&evaluator, last_solution);

    // // Set constraints
    // solver.SetConstraints();

    // // Run the solver
    // solver.Solver();

    // Print the result (assume the final result is stored in the class)
    std::cout << "Solver completed successfully!" << std::endl;

    return 0;
}