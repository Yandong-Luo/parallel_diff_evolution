#include "solver_center/solver_center.h"


namespace cudaprocess{

void CudaSolverCenter::Init(std::string filename){
    config = YAML::LoadFile(filename);
    num_tasks_ = config["problems"].size();

    // Initialize cuBLAS handle
    cublasStatus_t status = cublasCreate(&cublas_handle_);

    if (status != CUBLAS_STATUS_SUCCESS) {
        const char* error_msg;
        switch (status) {
            case CUBLAS_STATUS_NOT_INITIALIZED:
                error_msg = "CUBLAS_STATUS_NOT_INITIALIZED: CUBLAS library not initialized";
                break;
            case CUBLAS_STATUS_ALLOC_FAILED:
                error_msg = "CUBLAS_STATUS_ALLOC_FAILED: Resource allocation failed";
                break;
            case CUBLAS_STATUS_ARCH_MISMATCH:
                error_msg = "CUBLAS_STATUS_ARCH_MISMATCH: Device architecture not supported";
                break;
            default:
                error_msg = "Unknown CUBLAS error";
        }
        printf("cuBLAS initialization failed: %s\n", error_msg);
    }

    rnd_manager_ = std::make_shared<CudaRandomCenter>(gpu_device_);

    CHECK_CUDA(cudaHostAlloc(&tasks_best_sol_, sizeof(CudaParamIndividual) * num_tasks_, cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&tasks_potential_sol_, sizeof(CudaVector<CudaParamIndividual, CUDA_MAX_POTENTIAL_SOLUTION>) * num_tasks_, cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&tasks_problem_, sizeof(Problem) * num_tasks_, cudaHostAllocDefault));
    // for (int i = 0; i < num_enable_tasks; ++i){
    //     diff_evolve_solvers_[i].MallocSetup()
    // }

    for(int i = 0; i < num_tasks_; ++i) {
        YAML::Node node = config["problems"][i];
        
        tasks_problem_[i] = LoadProblemFromYaml(node);  // 复制到分配的内存中
    }
    cudamalloc_flag = true;
}

Problem CudaSolverCenter::LoadProblemFromYaml(const YAML::Node& node){
    Problem problem;  // 创建一个带默认值的结构体

    problem.num_continous = node["num_con_variable"].as<int>();
    problem.num_int = node["num_int_variable"].as<int>();

    problem.row_objective_mat = node["objective_dims"]["rows"].as<int>();
    problem.col_objective_mat = node["objective_dims"]["cols"].as<int>();

    problem.row_constraint_mat = node["constraint_dims"]["rows"].as<int>();
    problem.col_constraint_mat = node["constraint_dims"]["cols"].as<int>();

    problem.row_lambda = node["lambda_dims"]["rows"].as<int>();
    problem.col_lambda = node["lambda_dims"]["cols"].as<int>();

    problem.max_evolve_round = node["evolve_params"]["max_evolve_round"].as<int>();
    problem.max_lambda = node["evolve_params"]["max_lambda"].as<int>();
    problem.init_lambda = node["evolve_params"]["init_lambda"].as<int>();
    problem.accuracy_rng = node["evolve_params"]["accuracy_rng"].as<float>();
    problem.elite_eval_count = node["evolve_params"]["elite_eval_count"].as<int>();

    problem.int_upper_bound = new float[problem.num_int];
    problem.int_lower_bound = new float[problem.num_int];
    problem.con_upper_bound = new float[problem.num_continous];
    problem.con_lower_bound = new float[problem.num_continous];

    problem.objective_mat = new float[problem.row_objective_mat * problem.col_objective_mat];
    problem.constraint_mat = new float[problem.row_constraint_mat * problem.col_constraint_mat];
    problem.lambda_mat = new float[problem.row_lambda * problem.col_lambda];

    for (int i = 0; i < problem.row_objective_mat * problem.col_objective_mat; ++i){
        problem.objective_mat[i] = node["objective_matrix"][i].as<float>();
    }
    for (int i = 0; i < problem.row_constraint_mat * problem.col_constraint_mat; ++i){
        problem.constraint_mat[i] = node["constraint_matrix"][i].as<float>();
    }
    for (int i = 0; i < problem.row_lambda * problem.col_lambda; ++i){
        problem.lambda_mat[i] = node["lambda_matrix"][i].as<float>();
    }

    for(int i = 0; i < problem.num_int; i++) {
        problem.int_upper_bound[i] = node["int_bounds"]["upper"][i].as<float>();
        problem.int_lower_bound[i] = node["int_bounds"]["lower"][i].as<float>();
    }
    
    for(int i = 0; i < problem.num_continous; i++) {
        problem.con_upper_bound[i] = node["con_bounds"]["upper"][i].as<float>();
        problem.con_lower_bound[i] = node["con_bounds"]["lower"][i].as<float>();
    }

    return problem;
}

void CudaSolverCenter::ParallelGenerateMultiTaskSol(){
    for (int i = 0; i < num_tasks_; ++i){
        GenerateSolution(i);
    }
}

void CudaSolverCenter::GenerateSolution(int task_id){
    // printf("tasks_problem_[task_id]:%d\n", tasks_problem_[task_id].num_int_variable);
    printf("start the initialization of task %d\n", task_id);
    diff_evolve_solvers_[task_id].InitSolver(gpu_device_, cublas_handle_, task_id, rnd_manager_.get(), &tasks_problem_[task_id], &tasks_best_sol_[task_id], &tasks_potential_sol_[task_id]);
    printf("finish the initialization of task %d\n", task_id);
    tasks_best_sol_[task_id] = diff_evolve_solvers_[task_id].Solver();
}

CudaSolverCenter::~CudaSolverCenter(){
    if (cudamalloc_flag){
        CHECK_CUDA(cudaFreeHost(tasks_best_sol_));
        CHECK_CUDA(cudaFreeHost(tasks_potential_sol_));
        // CHECK_CUDA(cudaFreeHost(tasks_problem_));
        
    }
    cublasDestroy(cublas_handle_);
}
}