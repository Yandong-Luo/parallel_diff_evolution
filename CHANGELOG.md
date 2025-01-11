# Changelog
All notable changes to this project will be documented in this file.


## [0.1.1] - 2024-12-13
### Changed
- Yandong Luo: First submit

## [0.1.2] - 2024-12-15
### Changed
- Yandong Luo: The framework from the main function to the warmstart part can run.

## [0.1.3] - 2024-12-16
### Changed
- Yandong Luo: Add the solver_center for paralleling multi-differential evolution solvers.

## [0.1.4] - 2024-12-18
### Changed
- Yandong Luo: Fixed and verified random generation of solutions in constraint space. Completed conversion of data in warm start. But still missing evaluation of warm start.

## [0.1.5] - 2024-12-21
### Changed
- Yandong Luo: Complete the evaluation part, get the best fitness and put it first in the population, and prepare for implementation and evolution

## [0.1.6] - 2024-12-22
### Changed
- Yandong Luo: finish CudaEvolveProcess() function

## [0.1.7] - 2024-12-23
### Changed
- Yandong Luo: finish update parameter part. Still need to sort the parameter when evolve is done.

## [0.1.8] - 2024-12-24
### Changed
- Yandong Luo: finish SortParamBasedBitonic() and BitonicWarpCompare() for sorting the parameter based on fitness.

## [0.1.9] - 2024-12-25
### Changed
- Yandong Luo: Fix blocking issue in update parameter. Add test unit. Fix error in BitonicWarpCompare. Reorganize the whole process and adjust warm start.

## [0.1.10] - 2024-12-25
### Changed
- Yandong Luo: Complete the sort of old param (0-127) and complete the sort test.

## [0.1.11] - 2024-12-25
### Changed
- Yandong Luo: Some issues in random center

## [0.1.12] - 2024-12-25
### Changed
- Yandong Luo: Remove the compilation flag in CMakeList to solve the random center failure problem. Sucessfully verify the parameter matrix.

## [0.1.13] - 2024-12-26
### Changed
- Yandong Luo: The matrix calculation and verification of objective function based on cublas has been completed. It is worth noting that the matrix used to receive the result must be cleared to zero. Otherwise, the result will continue to accumulate.

## [0.1.14] - 2024-12-26
### Changed
- Yandong Luo: Complete and verify all the contents of evaluate calculations, and perform floor() on the integer part.

## [0.1.15] - 2024-12-27
### Changed
- Yandong Luo: Completed a test of a MILP problem. The overall process is correct and the result is correct.

## [0.1.16] - 2024-12-28
### Changed
- Yandong Luo: Early termination is implemented by comparing the fitness values of the top 8 elite individuals with the best fitness from the previous generation.

## [0.1.17] - 2024-12-29
### Analysis
- Yandong Luo: Added nvtx analysis to the solver part and init_solver part.

## [0.1.18] - 2024-12-29
### Changed
- Yandong Luo: Remove all unnecessary implementations and selectively allocate memory space based on debug mode or not. And stop tracking existing qdrep files.

## [0.1.19] - 2024-12-29
### Changed
- Yandong Luo: To optimize the efficiency of host->device, evolve_data was used for memory alignment and multi-stream transmission. However, there was still no significant efficiency improvement. Currently, Nsight shows that the process of host->device is too slow when comparing to the solution of the differential evolution algorithm.

## [0.1.20] - 2024-12-30
### Changed
- Yandong Luo: Configure optimization problem parameters via YAML

## [0.1.21] - 2024-12-30
### Changed
- Yandong Luo: Fixed the error when running multiple tasks. Currently, multiple solving tasks can be automatically generated according to YAML and the solving can be completed.

## [0.1.22] - 2025-1-1
### Changed
- Yandong Luo: Adjust the expression of the matrix for evaluation. Now fill in the yaml in the form of rows.

## [0.1.23] - 2025-1-1
### Changed
- Yandong Luo: Supports solving QP problems

## [0.1.24] - 2025-1-2
### Changed
- Yandong Luo: Based on cuRand to generate the random number. (random_manager)

## [0.1.25] - 2025-1-4
### Changed
- Yandong Luo: Update CMakeList and Add README.md

## [0.1.26] - 2025-1-4
### Changed
- Yandong Luo: Update CMakeList for static CUDA Runtime

## [0.1.27] - 2025-1-7
### Changed
- Yandong Luo: QP and MILP problem solver branch

## [0.2.1] - 2025-1-8
### Changed
- Yandong Luo: Start cart pole system

## [0.2.2] - 2025-1-8
### Changed
- Yandong Luo: Add Cartpole environment and add constant variable/matrix for cuda

## [0.2.3] - 2025-1-9
### Changed
- Yandong Luo: Construct A and C matrix for MPC model. Adjust constant variable and matrix.

## [0.2.4] - 2025-1-11
### Changed
- Yandong Luo: Cart_pole.py file can run cuda based on .so file. CMakeList.txt file has been modified so that can support pybind11