# Set the minimum required CMake version
cmake_minimum_required(VERSION 3.16) # CUDA support requires CMake 3.16 or later

# Set the project name and languages
project(Parallel_DiffEvolutionSolver LANGUAGES CXX CUDA)

# Find the CUDA Toolkit
find_package(CUDA REQUIRED)

# Find CUBLAS
find_library(CUBLAS_LIBRARY cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# Find CURAND
find_library(CURAND_LIBRARY curand HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# # Find nlohmann_json
# find_package(nlohmann_json 3.2.0 REQUIRED)

# Find yaml-cpp
find_package(yaml-cpp REQUIRED)

# Create NVTX interface target
add_library(nvtx3-cpp INTERFACE)
target_include_directories(nvtx3-cpp INTERFACE 
    ${PROJECT_SOURCE_DIR}/thirdparty/nvtx/c/include
)
if(UNIX)
    target_link_libraries(nvtx3-cpp INTERFACE ${NVTOOLSEXT_LIBRARY})
endif()

# Include CUDA headers and libraries
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARIES})

# Enable debug symbols and CUDA debug info
set(CMAKE_BUILD_TYPE Debug)

# IF YOU ENABLE THE FOLLOWING THIS LINE, RANDOM CENTER WILL BECOME WEIRD!!!!!!
# set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

# Set CUDA compilation options (optional)
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_CUDA_FLAGS_DEBUG "-g")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -Xcompiler -fPIC -lineinfo")
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17 -lineinfo")


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

# Include the directory for header files
include_directories(${PROJECT_SOURCE_DIR}/include)

# Set the source files (add main.cpp here)
set(SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/src/diff_evolution_solver/solver.cu
    ${PROJECT_SOURCE_DIR}/src/diff_evolution_solver/random_center.cu
    ${PROJECT_SOURCE_DIR}/src/diff_evolution_solver/random_manager.cu
    ${PROJECT_SOURCE_DIR}/src/solver_center/solver_center.cu
    ${PROJECT_SOURCE_DIR}/src/main.cpp
)

# Add the executable target
cuda_add_executable(diff_evolution_solver ${SOURCE_FILES})

# Set the CUDA architecture (adjust based on your hardware)
set_target_properties(diff_evolution_solver PROPERTIES
    CUDA_ARCHITECTURES "86"
)

set_target_properties(diff_evolution_solver PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Set the C++ standard
set_target_properties(diff_evolution_solver PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)

# Specify the search path for header files
target_include_directories(diff_evolution_solver PRIVATE ${PROJECT_SOURCE_DIR}/include)

# Link the executable with CUBLAS
target_link_libraries(diff_evolution_solver
    # ${CUDA_LIBRARIES}
    ${CUBLAS_LIBRARY}
    ${CURAND_LIBRARY}
    nvtx3-cpp
    # nlohmann_json::nlohmann_json
    yaml-cpp::yaml-cpp
)


# cmake_minimum_required(VERSION 3.16)
# project(Parallel_DiffEvolutionSolver LANGUAGES CXX CUDA)

# find_package(CUDA REQUIRED)
# find_library(CUBLAS_LIBRARY cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
# find_package(yaml-cpp REQUIRED)

# add_library(nvtx3-cpp INTERFACE)
# target_include_directories(nvtx3-cpp INTERFACE ${PROJECT_SOURCE_DIR}/thirdparty/nvtx/c/include)

# include_directories(${CUDA_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)
# link_directories(${CUDA_LIBRARIES})

# set(CMAKE_BUILD_TYPE Debug)
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17")
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

# set(SOURCE_FILES
#     src/diff_evolution_solver/solver.cu
#     src/diff_evolution_solver/random_center.cu
#     src/solver_center/solver_center.cu
#     src/main.cpp)

# cuda_add_executable(diff_evolution_solver ${SOURCE_FILES})

# set_target_properties(diff_evolution_solver PROPERTIES CUDA_ARCHITECTURES "86")
# set_target_properties(diff_evolution_solver PROPERTIES
#     CXX_STANDARD 17
#     CXX_STANDARD_REQUIRED YES)

# target_link_libraries(diff_evolution_solver 
#     ${CUBLAS_LIBRARY}
#     nvtx3-cpp
#     yaml-cpp::yaml-cpp)