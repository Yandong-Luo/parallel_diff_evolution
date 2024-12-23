# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/chris/parallel_diff_evolution

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/chris/parallel_diff_evolution/build

# Include any dependencies generated for this target.
include CMakeFiles/diff_evolution_solver.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/diff_evolution_solver.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/diff_evolution_solver.dir/flags.make

CMakeFiles/diff_evolution_solver.dir/src/diff_evolution_solver/solver.cu.o: CMakeFiles/diff_evolution_solver.dir/flags.make
CMakeFiles/diff_evolution_solver.dir/src/diff_evolution_solver/solver.cu.o: ../src/diff_evolution_solver/solver.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chris/parallel_diff_evolution/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/diff_evolution_solver.dir/src/diff_evolution_solver/solver.cu.o"
	/usr/local/cuda-11.1/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/chris/parallel_diff_evolution/src/diff_evolution_solver/solver.cu -o CMakeFiles/diff_evolution_solver.dir/src/diff_evolution_solver/solver.cu.o

CMakeFiles/diff_evolution_solver.dir/src/diff_evolution_solver/solver.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/diff_evolution_solver.dir/src/diff_evolution_solver/solver.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/diff_evolution_solver.dir/src/diff_evolution_solver/solver.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/diff_evolution_solver.dir/src/diff_evolution_solver/solver.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/diff_evolution_solver.dir/src/diff_evolution_solver/random_center.cu.o: CMakeFiles/diff_evolution_solver.dir/flags.make
CMakeFiles/diff_evolution_solver.dir/src/diff_evolution_solver/random_center.cu.o: ../src/diff_evolution_solver/random_center.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chris/parallel_diff_evolution/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/diff_evolution_solver.dir/src/diff_evolution_solver/random_center.cu.o"
	/usr/local/cuda-11.1/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/chris/parallel_diff_evolution/src/diff_evolution_solver/random_center.cu -o CMakeFiles/diff_evolution_solver.dir/src/diff_evolution_solver/random_center.cu.o

CMakeFiles/diff_evolution_solver.dir/src/diff_evolution_solver/random_center.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/diff_evolution_solver.dir/src/diff_evolution_solver/random_center.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/diff_evolution_solver.dir/src/diff_evolution_solver/random_center.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/diff_evolution_solver.dir/src/diff_evolution_solver/random_center.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/diff_evolution_solver.dir/src/solver_center/solver_center.cu.o: CMakeFiles/diff_evolution_solver.dir/flags.make
CMakeFiles/diff_evolution_solver.dir/src/solver_center/solver_center.cu.o: ../src/solver_center/solver_center.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chris/parallel_diff_evolution/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CUDA object CMakeFiles/diff_evolution_solver.dir/src/solver_center/solver_center.cu.o"
	/usr/local/cuda-11.1/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/chris/parallel_diff_evolution/src/solver_center/solver_center.cu -o CMakeFiles/diff_evolution_solver.dir/src/solver_center/solver_center.cu.o

CMakeFiles/diff_evolution_solver.dir/src/solver_center/solver_center.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/diff_evolution_solver.dir/src/solver_center/solver_center.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/diff_evolution_solver.dir/src/solver_center/solver_center.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/diff_evolution_solver.dir/src/solver_center/solver_center.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/diff_evolution_solver.dir/src/main.cpp.o: CMakeFiles/diff_evolution_solver.dir/flags.make
CMakeFiles/diff_evolution_solver.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chris/parallel_diff_evolution/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/diff_evolution_solver.dir/src/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/diff_evolution_solver.dir/src/main.cpp.o -c /home/chris/parallel_diff_evolution/src/main.cpp

CMakeFiles/diff_evolution_solver.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/diff_evolution_solver.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/chris/parallel_diff_evolution/src/main.cpp > CMakeFiles/diff_evolution_solver.dir/src/main.cpp.i

CMakeFiles/diff_evolution_solver.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/diff_evolution_solver.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/chris/parallel_diff_evolution/src/main.cpp -o CMakeFiles/diff_evolution_solver.dir/src/main.cpp.s

# Object files for target diff_evolution_solver
diff_evolution_solver_OBJECTS = \
"CMakeFiles/diff_evolution_solver.dir/src/diff_evolution_solver/solver.cu.o" \
"CMakeFiles/diff_evolution_solver.dir/src/diff_evolution_solver/random_center.cu.o" \
"CMakeFiles/diff_evolution_solver.dir/src/solver_center/solver_center.cu.o" \
"CMakeFiles/diff_evolution_solver.dir/src/main.cpp.o"

# External object files for target diff_evolution_solver
diff_evolution_solver_EXTERNAL_OBJECTS =

../diff_evolution_solver: CMakeFiles/diff_evolution_solver.dir/src/diff_evolution_solver/solver.cu.o
../diff_evolution_solver: CMakeFiles/diff_evolution_solver.dir/src/diff_evolution_solver/random_center.cu.o
../diff_evolution_solver: CMakeFiles/diff_evolution_solver.dir/src/solver_center/solver_center.cu.o
../diff_evolution_solver: CMakeFiles/diff_evolution_solver.dir/src/main.cpp.o
../diff_evolution_solver: CMakeFiles/diff_evolution_solver.dir/build.make
../diff_evolution_solver: CMakeFiles/diff_evolution_solver.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/chris/parallel_diff_evolution/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable ../diff_evolution_solver"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/diff_evolution_solver.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/diff_evolution_solver.dir/build: ../diff_evolution_solver

.PHONY : CMakeFiles/diff_evolution_solver.dir/build

CMakeFiles/diff_evolution_solver.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/diff_evolution_solver.dir/cmake_clean.cmake
.PHONY : CMakeFiles/diff_evolution_solver.dir/clean

CMakeFiles/diff_evolution_solver.dir/depend:
	cd /home/chris/parallel_diff_evolution/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/chris/parallel_diff_evolution /home/chris/parallel_diff_evolution /home/chris/parallel_diff_evolution/build /home/chris/parallel_diff_evolution/build /home/chris/parallel_diff_evolution/build/CMakeFiles/diff_evolution_solver.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/diff_evolution_solver.dir/depend

