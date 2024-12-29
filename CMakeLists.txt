# Set the minimum required CMake version
cmake_minimum_required(VERSION 3.16) # CUDA support requires CMake 3.16 or later

# Set the project name and languages
project(Parallel_DiffEvolutionSolver LANGUAGES CXX CUDA)

# Find the CUDA Toolkit
find_package(CUDA REQUIRED)

# Find CUBLAS
find_library(CUBLAS_LIBRARY cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# Find nvToolsExt
# find_library(NVTOOLSEXT_LIBRARY nvToolsExt HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

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

# Set CUDA compilation options (optional)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

# Include the directory for header files
include_directories(${PROJECT_SOURCE_DIR}/include)

# Set the source files (add main.cpp here)
set(SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/src/diff_evolution_solver/solver.cu
    ${PROJECT_SOURCE_DIR}/src/diff_evolution_solver/random_center.cu
    ${PROJECT_SOURCE_DIR}/src/solver_center/solver_center.cu
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

# Link the executable with CUBLAS
target_link_libraries(diff_evolution_solver PRIVATE
    ${CUDA_LIBRARIES}
    ${CUBLAS_LIBRARY}
    nvtx3-cpp
)
