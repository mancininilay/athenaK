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

# Include any dependencies generated for this target.
include kokkos/containers/src/CMakeFiles/kokkoscontainers.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include kokkos/containers/src/CMakeFiles/kokkoscontainers.dir/compiler_depend.make

# Include the progress variables for this target.
include kokkos/containers/src/CMakeFiles/kokkoscontainers.dir/progress.make

# Include the compile flags for this target's objects.
include kokkos/containers/src/CMakeFiles/kokkoscontainers.dir/flags.make

kokkos/containers/src/CMakeFiles/kokkoscontainers.dir/impl/Kokkos_UnorderedMap_impl.cpp.o: kokkos/containers/src/CMakeFiles/kokkoscontainers.dir/flags.make
kokkos/containers/src/CMakeFiles/kokkoscontainers.dir/impl/Kokkos_UnorderedMap_impl.cpp.o: ../kokkos/containers/src/impl/Kokkos_UnorderedMap_impl.cpp
kokkos/containers/src/CMakeFiles/kokkoscontainers.dir/impl/Kokkos_UnorderedMap_impl.cpp.o: kokkos/containers/src/CMakeFiles/kokkoscontainers.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/storage/home/nkm5669/work/athenak/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object kokkos/containers/src/CMakeFiles/kokkoscontainers.dir/impl/Kokkos_UnorderedMap_impl.cpp.o"
	cd /storage/home/nkm5669/work/athenak/build/kokkos/containers/src && /storage/home/nkm5669/work/athenak/build/../kokkos/bin/nvcc_wrapper $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT kokkos/containers/src/CMakeFiles/kokkoscontainers.dir/impl/Kokkos_UnorderedMap_impl.cpp.o -MF CMakeFiles/kokkoscontainers.dir/impl/Kokkos_UnorderedMap_impl.cpp.o.d -o CMakeFiles/kokkoscontainers.dir/impl/Kokkos_UnorderedMap_impl.cpp.o -c /storage/home/nkm5669/work/athenak/kokkos/containers/src/impl/Kokkos_UnorderedMap_impl.cpp

kokkos/containers/src/CMakeFiles/kokkoscontainers.dir/impl/Kokkos_UnorderedMap_impl.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kokkoscontainers.dir/impl/Kokkos_UnorderedMap_impl.cpp.i"
	cd /storage/home/nkm5669/work/athenak/build/kokkos/containers/src && /storage/home/nkm5669/work/athenak/build/../kokkos/bin/nvcc_wrapper $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /storage/home/nkm5669/work/athenak/kokkos/containers/src/impl/Kokkos_UnorderedMap_impl.cpp > CMakeFiles/kokkoscontainers.dir/impl/Kokkos_UnorderedMap_impl.cpp.i

kokkos/containers/src/CMakeFiles/kokkoscontainers.dir/impl/Kokkos_UnorderedMap_impl.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kokkoscontainers.dir/impl/Kokkos_UnorderedMap_impl.cpp.s"
	cd /storage/home/nkm5669/work/athenak/build/kokkos/containers/src && /storage/home/nkm5669/work/athenak/build/../kokkos/bin/nvcc_wrapper $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /storage/home/nkm5669/work/athenak/kokkos/containers/src/impl/Kokkos_UnorderedMap_impl.cpp -o CMakeFiles/kokkoscontainers.dir/impl/Kokkos_UnorderedMap_impl.cpp.s

# Object files for target kokkoscontainers
kokkoscontainers_OBJECTS = \
"CMakeFiles/kokkoscontainers.dir/impl/Kokkos_UnorderedMap_impl.cpp.o"

# External object files for target kokkoscontainers
kokkoscontainers_EXTERNAL_OBJECTS =

kokkos/containers/src/libkokkoscontainers.a: kokkos/containers/src/CMakeFiles/kokkoscontainers.dir/impl/Kokkos_UnorderedMap_impl.cpp.o
kokkos/containers/src/libkokkoscontainers.a: kokkos/containers/src/CMakeFiles/kokkoscontainers.dir/build.make
kokkos/containers/src/libkokkoscontainers.a: kokkos/containers/src/CMakeFiles/kokkoscontainers.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/storage/home/nkm5669/work/athenak/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libkokkoscontainers.a"
	cd /storage/home/nkm5669/work/athenak/build/kokkos/containers/src && $(CMAKE_COMMAND) -P CMakeFiles/kokkoscontainers.dir/cmake_clean_target.cmake
	cd /storage/home/nkm5669/work/athenak/build/kokkos/containers/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/kokkoscontainers.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
kokkos/containers/src/CMakeFiles/kokkoscontainers.dir/build: kokkos/containers/src/libkokkoscontainers.a
.PHONY : kokkos/containers/src/CMakeFiles/kokkoscontainers.dir/build

kokkos/containers/src/CMakeFiles/kokkoscontainers.dir/clean:
	cd /storage/home/nkm5669/work/athenak/build/kokkos/containers/src && $(CMAKE_COMMAND) -P CMakeFiles/kokkoscontainers.dir/cmake_clean.cmake
.PHONY : kokkos/containers/src/CMakeFiles/kokkoscontainers.dir/clean

kokkos/containers/src/CMakeFiles/kokkoscontainers.dir/depend:
	cd /storage/home/nkm5669/work/athenak/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /storage/home/nkm5669/work/athenak /storage/home/nkm5669/work/athenak/kokkos/containers/src /storage/home/nkm5669/work/athenak/build /storage/home/nkm5669/work/athenak/build/kokkos/containers/src /storage/home/nkm5669/work/athenak/build/kokkos/containers/src/CMakeFiles/kokkoscontainers.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : kokkos/containers/src/CMakeFiles/kokkoscontainers.dir/depend

