#include "diff_evolution_solver/solver.cuh"

namespace cudaprocess{

void CudaDiffEvolveSolver::Setup(int gpu_device){
    
    
}

void CudaDiffEvolveSolver::InitDiffEvolvParam(float best, float d_best, float min_best, float diff, float d_diff, float min_diff, float scale_f, float prob_crossover){
    best_ = best;
    d_best_ = d_best;
    min_best_ = min_best;
    diff_ = diff;
    d_diff_ = d_diff;
    min_diff_ = min_diff;
    
    h.scale_f = h.scale_f1 = scale_f;
    h.Cr = prob_crossover;

}

__global__ void InitCudaEvolveData(CudaEvolveData* data, int pop_size){
    if (threadIdx.x == 0) {
        data->best = 0.;
        data->hyperparam.scale_f = data->hyperparam.scale_f1 = 0.6;
        data->hyperparam.Cr = 0.9;
    }
}

void CudaDiffEvolveSolver::InitSolver(int con_var_dims, int bin_var_dims){
    CHECK_CUDA(cudaSetDevice(gpu_device_));
    con_var_dims_ = con_var_dims;
    bin_var_dims_ = bin_var_dims;
    
}

void CudaDiffEvolveSolver::Solver(){
    init_pop_size_ = default_pop_size_;
    pop_size_ = default_pop_size_;

    InitDiffEvolvParam();
}

}