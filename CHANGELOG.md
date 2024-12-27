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