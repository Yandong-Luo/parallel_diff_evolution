# Set the minimum required CMake version
cmake_minimum_required(VERSION 3.16) # CUDA support requires CMake 3.16 or later

# Set the project name and languages
project(Parallel_DiffEvolutionSolver LANGUAGES CXX CUDA)

# Find the CUDA Toolkit
find_package(CUDA REQUIRED)

# Include CUDA headers and libraries
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARIES})

# Set CUDA compilation options (optional)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

# Include the directory for header files
include_directories(${PROJECT_SOURCE_DIR}/include)

# Set the source files (add main.cpp here)
set(SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/src/diff_evolution_solver/solver.cu
    ${PROJECT_SOURCE_DIR}/src/main.cpp
)

# Add the executable target
add_executable(diff_evolution_solver ${SOURCE_FILES})

# Set the CUDA architecture (adjust based on your hardware)
set_target_properties(diff_evolution_solver PROPERTIES
    CUDA_ARCHITECTURES "86"
)

# Set the C++ standard
set_target_properties(diff_evolution_solver PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)

# Specify the search path for header files
target_include_directories(diff_evolution_solver PRIVATE ${PROJECT_SOURCE_DIR}/include)
