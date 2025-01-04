#include "solver_center/solver_center.h"


namespace cudaprocess{

void CudaSolverCenter::Init(std::string filename){
    config = YAML::LoadFile(filename);
    num_tasks_ = config["problems"].size();
    // Initialize cuBLAS handle
    cublasStatus_t status = cublasCreate(&cublas_handle_);

    // rnd_manager_ = std::make_shared<CudaRandomCenter>(gpu_device_);

    rnd_manager_ = std::make_shared<CudaRandomManager>(gpu_device_);

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

    problem.row_objective_Q = node["objective_Q_dims"]["rows"].as<int>();
    problem.col_objective_Q = node["objective_Q_dims"]["cols"].as<int>();

    problem.max_evolve_round = node["evolve_params"]["max_evolve_round"].as<int>();
    problem.max_lambda = node["evolve_params"]["max_lambda"].as<int>();
    problem.init_lambda = node["evolve_params"]["init_lambda"].as<int>();
    problem.accuracy_rng = node["evolve_params"]["accuracy_rng"].as<float>();
    problem.elite_eval_count = node["evolve_params"]["elite_eval_count"].as<int>(); 

        
    if (problem.row_objective_mat * problem.col_objective_mat != 0){
        problem.objective_mat = new float[problem.row_objective_mat * problem.col_objective_mat];

        for (int i = 0; i < problem.row_objective_mat * problem.col_objective_mat; ++i){
            problem.objective_mat[i] = node["objective_matrix"][i].as<float>();
        }
    }
    
    if (problem.row_constraint_mat * problem.col_constraint_mat != 0){
        problem.constraint_mat = new float[problem.row_constraint_mat * problem.col_constraint_mat];

        for (int i = 0; i < problem.row_constraint_mat * problem.col_constraint_mat; ++i){
            problem.constraint_mat[i] = node["constraint_matrix"][i].as<float>();
        }
    }
    
    if (problem.row_lambda * problem.col_lambda != 0){
        problem.lambda_mat = new float[problem.row_lambda * problem.col_lambda];
     
        for (int i = 0; i < problem.row_lambda * problem.col_lambda; ++i){
            problem.lambda_mat[i] = node["lambda_matrix"][i].as<float>();
        }
    }
    
    if (problem.row_objective_Q * problem.col_objective_Q != 0){
        problem.objective_Q_mat = new float[problem.row_objective_Q * problem.col_objective_Q];

        for (int i = 0; i < problem.row_objective_Q * problem.col_objective_Q; ++i){
            problem.objective_Q_mat[i] = node["objective_Q_matrix"][i].as<float>();
        }
    }

    if (problem.num_int != 0){
        problem.int_upper_bound = new float[problem.num_int];
        problem.int_lower_bound = new float[problem.num_int];
        for(int i = 0; i < problem.num_int; i++) {
            problem.int_upper_bound[i] = node["int_bounds"]["upper"][i].as<float>();
            problem.int_lower_bound[i] = node["int_bounds"]["lower"][i].as<float>();
        }
    }
    
    if (problem.num_continous != 0){
        problem.con_upper_bound = new float[problem.num_continous];
        problem.con_lower_bound = new float[problem.num_continous];
        for(int i = 0; i < problem.num_continous; i++) {
            problem.con_upper_bound[i] = node["con_bounds"]["upper"][i].as<float>();
            problem.con_lower_bound[i] = node["con_bounds"]["lower"][i].as<float>();
        }
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
    // printf("start the initialization of task %d\n", task_id);
    printf("???????????????:%d\n",cudamalloc_flag);
    // auto task_rnd_manager = std::make_shared<CudaRandomCenter>(gpu_device_);
    diff_evolve_solvers_[task_id].InitSolver(gpu_device_, cublas_handle_, task_id, rnd_manager_.get(), &tasks_problem_[task_id], &tasks_best_sol_[task_id], &tasks_potential_sol_[task_id]);
    // printf("finish the initialization of task %d\n", task_id);
    printf("???????????????:%d\n",cudamalloc_flag);
    tasks_best_sol_[task_id] = diff_evolve_solvers_[task_id].Solver();
}

// CudaSolverCenter::~CudaSolverCenter(){
//     if (cudamalloc_flag){
        
//         // for(int i = 0; i < num_tasks_; ++i){
//         //     // 先释放每个Problem对象中的动态内存
//         //     if(tasks_problem_[i].int_upper_bound) delete[] tasks_problem_[i].int_upper_bound;
            
//         //     if(tasks_problem_[i].int_lower_bound) delete[] tasks_problem_[i].int_lower_bound;
//         //     if(tasks_problem_[i].con_upper_bound) delete[] tasks_problem_[i].con_upper_bound;
//         //     if(tasks_problem_[i].con_lower_bound) delete[] tasks_problem_[i].con_lower_bound;
//         //     if(tasks_problem_[i].objective_mat) delete[] tasks_problem_[i].objective_mat;
//         //     if(tasks_problem_[i].constraint_mat) delete[] tasks_problem_[i].constraint_mat;
//         //     if(tasks_problem_[i].lambda_mat) delete[] tasks_problem_[i].lambda_mat;
//         //     if(tasks_problem_[i].objective_Q_mat) delete[] tasks_problem_[i].objective_Q_mat;
//         // }
        
//         // CHECK_CUDA(cudaFreeHost(tasks_problem_));
        
//         // CHECK_CUDA(cudaFreeHost(tasks_best_sol_));
        
//         // CHECK_CUDA(cudaFreeHost(tasks_potential_sol_));
//         // 
//     }
//     // cublasDestroy(cublas_handle_);
//     if (cublas_handle_) {
//         cublasStatus_t destroy_status = cublasDestroy(cublas_handle_);
//         if (destroy_status != CUBLAS_STATUS_SUCCESS) {
//             printf("cuBLAS destroy failed: %d\n", destroy_status);
//         }
//     }
// }

CudaSolverCenter::~CudaSolverCenter() {
    // rnd_manager_.reset();
    
    if (cudamalloc_flag) {
        // for (int i = 0; i < num_tasks_; ++i) {
        //     if (tasks_problem_[i].int_upper_bound) delete[] tasks_problem_[i].int_upper_bound;
        //     if (tasks_problem_[i].int_lower_bound) delete[] tasks_problem_[i].int_lower_bound;
        //     if (tasks_problem_[i].con_upper_bound) delete[] tasks_problem_[i].con_upper_bound;
        //     if (tasks_problem_[i].con_lower_bound) delete[] tasks_problem_[i].con_lower_bound;
        //     if (tasks_problem_[i].objective_mat) delete[] tasks_problem_[i].objective_mat;
        //     if (tasks_problem_[i].constraint_mat) delete[] tasks_problem_[i].constraint_mat;
        //     if (tasks_problem_[i].lambda_mat) delete[] tasks_problem_[i].lambda_mat;
        //     if (tasks_problem_[i].objective_Q_mat) delete[] tasks_problem_[i].objective_Q_mat;
        // }
        if (tasks_best_sol_) {
            CHECK_CUDA(cudaFreeHost(tasks_best_sol_));
        }
        if (tasks_potential_sol_) {
            CHECK_CUDA(cudaFreeHost(tasks_potential_sol_));
        }
    }
    
}

}