# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /storage/icds/swst/deployed/production/20220813/apps/cmake/3.21.4_gcc-8.5.0/bin/cmake

# The command to remove a file.
RM = /storage/icds/swst/deployed/production/20220813/apps/cmake/3.21.4_gcc-8.5.0/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /storage/home/nkm5669/work/athenak

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /storage/home/nkm5669/work/athenak/build

# Utility rule file for NightlyCoverage.

# Include any custom commands dependencies for this target.
include kokkos/CMakeFiles/NightlyCoverage.dir/compiler_depend.make

# Include the progress variables for this target.
include kokkos/CMakeFiles/NightlyCoverage.dir/progress.make

kokkos/CMakeFiles/NightlyCoverage:
	cd /storage/home/nkm5669/work/athenak/build/kokkos && /storage/icds/swst/deployed/production/20220813/apps/cmake/3.21.4_gcc-8.5.0/bin/ctest -D NightlyCoverage

NightlyCoverage: kokkos/CMakeFiles/NightlyCoverage
NightlyCoverage: kokkos/CMakeFiles/NightlyCoverage.dir/build.make
.PHONY : NightlyCoverage

# Rule to build all files generated by this target.
kokkos/CMakeFiles/NightlyCoverage.dir/build: NightlyCoverage
.PHONY : kokkos/CMakeFiles/NightlyCoverage.dir/build

kokkos/CMakeFiles/NightlyCoverage.dir/clean:
	cd /storage/home/nkm5669/work/athenak/build/kokkos && $(CMAKE_COMMAND) -P CMakeFiles/NightlyCoverage.dir/cmake_clean.cmake
.PHONY : kokkos/CMakeFiles/NightlyCoverage.dir/clean

kokkos/CMakeFiles/NightlyCoverage.dir/depend:
	cd /storage/home/nkm5669/work/athenak/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /storage/home/nkm5669/work/athenak /storage/home/nkm5669/work/athenak/kokkos /storage/home/nkm5669/work/athenak/build /storage/home/nkm5669/work/athenak/build/kokkos /storage/home/nkm5669/work/athenak/build/kokkos/CMakeFiles/NightlyCoverage.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : kokkos/CMakeFiles/NightlyCoverage.dir/depend

