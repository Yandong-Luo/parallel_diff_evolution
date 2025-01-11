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
include CMakeFiles/Parallel_DiffEvolutionSolver.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Parallel_DiffEvolutionSolver.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Parallel_DiffEvolutionSolver.dir/flags.make

CMakeFiles/Parallel_DiffEvolutionSolver.dir/src/main.cpp.o: CMakeFiles/Parallel_DiffEvolutionSolver.dir/flags.make
CMakeFiles/Parallel_DiffEvolutionSolver.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chris/parallel_diff_evolution/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Parallel_DiffEvolutionSolver.dir/src/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Parallel_DiffEvolutionSolver.dir/src/main.cpp.o -c /home/chris/parallel_diff_evolution/src/main.cpp

CMakeFiles/Parallel_DiffEvolutionSolver.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Parallel_DiffEvolutionSolver.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/chris/parallel_diff_evolution/src/main.cpp > CMakeFiles/Parallel_DiffEvolutionSolver.dir/src/main.cpp.i

CMakeFiles/Parallel_DiffEvolutionSolver.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Parallel_DiffEvolutionSolver.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/chris/parallel_diff_evolution/src/main.cpp -o CMakeFiles/Parallel_DiffEvolutionSolver.dir/src/main.cpp.s

# Object files for target Parallel_DiffEvolutionSolver
Parallel_DiffEvolutionSolver_OBJECTS = \
"CMakeFiles/Parallel_DiffEvolutionSolver.dir/src/main.cpp.o"

# External object files for target Parallel_DiffEvolutionSolver
Parallel_DiffEvolutionSolver_EXTERNAL_OBJECTS =

CMakeFiles/Parallel_DiffEvolutionSolver.dir/cmake_device_link.o: CMakeFiles/Parallel_DiffEvolutionSolver.dir/src/main.cpp.o
CMakeFiles/Parallel_DiffEvolutionSolver.dir/cmake_device_link.o: CMakeFiles/Parallel_DiffEvolutionSolver.dir/build.make
CMakeFiles/Parallel_DiffEvolutionSolver.dir/cmake_device_link.o: libcuda_DE.a
CMakeFiles/Parallel_DiffEvolutionSolver.dir/cmake_device_link.o: /usr/local/cuda-11.1/lib64/libcublas.so
CMakeFiles/Parallel_DiffEvolutionSolver.dir/cmake_device_link.o: /usr/local/cuda-11.1/lib64/libcurand.so
CMakeFiles/Parallel_DiffEvolutionSolver.dir/cmake_device_link.o: /usr/local/lib/libyaml-cpp.a
CMakeFiles/Parallel_DiffEvolutionSolver.dir/cmake_device_link.o: CMakeFiles/Parallel_DiffEvolutionSolver.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/chris/parallel_diff_evolution/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/Parallel_DiffEvolutionSolver.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Parallel_DiffEvolutionSolver.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Parallel_DiffEvolutionSolver.dir/build: CMakeFiles/Parallel_DiffEvolutionSolver.dir/cmake_device_link.o

.PHONY : CMakeFiles/Parallel_DiffEvolutionSolver.dir/build

# Object files for target Parallel_DiffEvolutionSolver
Parallel_DiffEvolutionSolver_OBJECTS = \
"CMakeFiles/Parallel_DiffEvolutionSolver.dir/src/main.cpp.o"

# External object files for target Parallel_DiffEvolutionSolver
Parallel_DiffEvolutionSolver_EXTERNAL_OBJECTS =

../Parallel_DiffEvolutionSolver: CMakeFiles/Parallel_DiffEvolutionSolver.dir/src/main.cpp.o
../Parallel_DiffEvolutionSolver: CMakeFiles/Parallel_DiffEvolutionSolver.dir/build.make
../Parallel_DiffEvolutionSolver: libcuda_DE.a
../Parallel_DiffEvolutionSolver: /usr/local/cuda-11.1/lib64/libcublas.so
../Parallel_DiffEvolutionSolver: /usr/local/cuda-11.1/lib64/libcurand.so
../Parallel_DiffEvolutionSolver: /usr/local/lib/libyaml-cpp.a
../Parallel_DiffEvolutionSolver: CMakeFiles/Parallel_DiffEvolutionSolver.dir/cmake_device_link.o
../Parallel_DiffEvolutionSolver: CMakeFiles/Parallel_DiffEvolutionSolver.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/chris/parallel_diff_evolution/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ../Parallel_DiffEvolutionSolver"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Parallel_DiffEvolutionSolver.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Parallel_DiffEvolutionSolver.dir/build: ../Parallel_DiffEvolutionSolver

.PHONY : CMakeFiles/Parallel_DiffEvolutionSolver.dir/build

CMakeFiles/Parallel_DiffEvolutionSolver.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Parallel_DiffEvolutionSolver.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Parallel_DiffEvolutionSolver.dir/clean

CMakeFiles/Parallel_DiffEvolutionSolver.dir/depend:
	cd /home/chris/parallel_diff_evolution/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/chris/parallel_diff_evolution /home/chris/parallel_diff_evolution /home/chris/parallel_diff_evolution/build /home/chris/parallel_diff_evolution/build /home/chris/parallel_diff_evolution/build/CMakeFiles/Parallel_DiffEvolutionSolver.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Parallel_DiffEvolutionSolver.dir/depend
