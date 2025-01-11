// src/python_binding.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "diff_evolution_solver/solver.cuh"

namespace py = pybind11;

PYBIND11_MODULE(DE_cuda_solver, m) {
    // 为模块添加文档
    m.doc() = "CUDA-based Differential Evolution Solver"; 

    // 绑定CudaDiffEvolveSolver类
    py::class_<cudaprocess::CudaDiffEvolveSolver>(m, "Create")
        // 构造函数
        .def(py::init<>())
        
        // 初始化求解器
        .def("init_solver", &cudaprocess::CudaDiffEvolveSolver::InitSolver,
             "Initialize the CUDA solver with specified GPU device",
             py::arg("gpu_device"))
             
        // 更新状态
        .def("update_cart_pole_state", [](cudaprocess::CudaDiffEvolveSolver& self, 
                py::array_t<float> state) {
            // 确保输入是正确的numpy数组
            auto buf = state.request();
            if (buf.ndim != 1 || buf.shape[0] != 4) {
                throw std::runtime_error("State must be a 1D array with 4 elements");
            }
            self.UpdateCartPoleState(static_cast<float*>(buf.ptr));
        }, "Update cart pole state with new state vector")
        
        // 主求解函数
        .def("solve", [](cudaprocess::CudaDiffEvolveSolver& self) {
            auto result = self.Solver();
            
            // 创建返回字典
            py::dict solution;
            solution["fitness"] = result.fitness;
            
            // 转换参数到numpy数组
            auto param_array = py::array_t<float>(result.dims);
            auto buf = param_array.request();
            float* ptr = static_cast<float*>(buf.ptr);
            std::memcpy(ptr, result.param, result.dims * sizeof(float));
            
            solution["param"] = param_array;
            return solution;
        }, "Run the differential evolution solver and return the solution");
}