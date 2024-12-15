#include "diff_evolution_solver/solver.cuh"
#include "diff_evolution_solver/decoder.cuh"

namespace cudaprocess{

void CudaDiffEvolveSolver::MallocSetup(){
    CHECK_CUDA(cudaSetDevice(gpu_device_));

    // GPU Device
    CHECK_CUDA(cudaMalloc(&decoder_, sizeof(CudaProblemDecoder)));
    CHECK_CUDA(cudaMalloc(&evolve_data_, sizeof(CudaEvolveData)));
    CHECK_CUDA(cudaMalloc(&new_cluster_data_, sizeof(CudaParamClusterData<64>)));
    CHECK_CUDA(cudaMalloc(&old_cluster_data_, sizeof(CudaParamClusterData<192>)));
    CHECK_CUDA(cudaMalloc(&new_cluster_vec_, sizeof(CudaVector<CudaParamIndividual, CUDA_SOLVER_POP_SIZE>)));
    // CHECK_CUDA(cudaMalloc(&new_param, sizeof(CudaParamClusterData<192>)));

    // CPU Host
    CHECK_CUDA(cudaHostAlloc(&host_decoder_, sizeof(CudaProblemDecoder), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&host_evolve_data_, sizeof(CudaEvolveData), cudaHostAllocDefault));

    cuda_utils_ = std::make_shared<CudaUtil>();

    cudamalloc_flag = true;
}

void CudaDiffEvolveSolver::InitDiffEvolveParam(float best, float d_best, float min_best, float diff, float d_diff, float min_diff, float scale_f, float prob_crossover){
    best_ = best;
    d_best_ = d_best;
    min_best_ = min_best;
    diff_ = diff;
    d_diff_ = d_diff;
    min_diff_ = min_diff;
    
    lshade_param_.scale_f = lshade_param_.scale_f1 = scale_f;
    lshade_param_.Cr = prob_crossover;

}

__global__ void InitCudaEvolveData(CudaEvolveData* data, CudaParamClusterData<192>* old_cluster_data, int pop_size){
    int idx = threadIdx.x;
    if (idx == 0) {
        data->best = 0.;
        data->lshade_param.scale_f = data->lshade_param.scale_f1 = 0.6;
        data->lshade_param.Cr = 0.9;
        data->new_cluster_vec->len = pop_size;
        old_cluster_data->len = pop_size;
    }
    if (idx < pop_size){
        // initial the each parameter in old_cluster 
        for (int i = 0; i < CUDA_PARAM_MAX_SIZE; ++i){
            old_cluster_data->all_param[(idx + pop_size) * CUDA_PARAM_MAX_SIZE + i] = 0.f;
        }
        old_cluster_data->fitness[idx + pop_size] = CUDA_MAX_FLOAT;
    }
}

void CudaDiffEvolveSolver::SetConstraints(){

}

void CudaDiffEvolveSolver::WarmStart(ProblemEvaluator* evaluator, CudaParamIndividual* last_sol){
    InitParameter<<<1, default_pop_size_, 0, cuda_utils_->streams_[0]>>>(decoder_, evolve_data_, default_pop_size_, new_cluster_data_, old_cluster_data_);
}

void CudaDiffEvolveSolver::InitSolver(ProblemEvaluator* evaluator, CudaParamIndividual* last_sol){
    CHECK_CUDA(cudaSetDevice(gpu_device_));
    if (DEBUG_PRINT_FLAG) printf("CUDA SET DEVICE\n");

    MallocSetup();

    con_var_dims_ = evaluator->num_con_variable;
    bin_var_dims_ = evaluator->num_binary_variable;
    dims_ = evaluator->num_con_variable + evaluator->num_binary_variable;

    InitDiffEvolveParam();
    if (DEBUG_PRINT_FLAG) printf("INIT PARAM FOR DE\n");
    
    // Initial host decoder
    host_decoder_->con_var_dims_ = evaluator->num_con_variable;
    host_decoder_->bin_var_dims_ = evaluator->num_binary_variable;
    host_decoder_->dims_ = evaluator->num_con_variable + evaluator->num_binary_variable;

    // Initial evolve data
    host_evolve_data_->best = best_;
    host_evolve_data_->new_cluster_vec = new_cluster_vec_;

    if (DEBUG_PRINT_FLAG) printf("START MEMORY ASYNC\n");

    // Host --> GPU device
    CHECK_CUDA(cudaMemcpyAsync(evolve_data_, host_evolve_data_, sizeof(CudaEvolveData), cudaMemcpyHostToDevice, cuda_utils_->streams_[0]));
    CHECK_CUDA(cudaMemcpyAsync(decoder_, host_decoder_, sizeof(CudaProblemDecoder), cudaMemcpyHostToDevice, cuda_utils_->streams_[0]));

    // if (last_sol == nullptr){
    //     CHECK_CUDA(cudaStreamSynchronize(cuda_utils_->streams_[0]));
    // }

    if (DEBUG_PRINT_FLAG) printf("MEMORY ASYNC SUBMIT\n");

    InitCudaEvolveData<<<1, CUDA_SOLVER_POP_SIZE, 0, cuda_utils_->streams_[0]>>>(evolve_data_, old_cluster_data_, default_pop_size_);

    WarmStart(evaluator, last_sol);
}

void CudaDiffEvolveSolver::Solver(){
    init_pop_size_ = default_pop_size_;
    pop_size_ = default_pop_size_;

}

CudaDiffEvolveSolver::~CudaDiffEvolveSolver(){
    if (cudamalloc_flag){
        // GPU device
        CHECK_CUDA(cudaFree(decoder_));
        CHECK_CUDA(cudaFree(evolve_data_));
        CHECK_CUDA(cudaFree(new_cluster_data_));
        CHECK_CUDA(cudaFree(old_cluster_data_));
        CHECK_CUDA(cudaFree(new_cluster_vec_));

        // CPU host
        CHECK_CUDA(cudaFreeHost(host_decoder_));
        CHECK_CUDA(cudaFreeHost(host_evolve_data_));
    }
    
}

}