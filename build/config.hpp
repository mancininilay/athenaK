#ifndef CONFIG_HPP_
#define CONFIG_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

// problem generator (string)
#define PROBLEM_GENERATOR "elliptica_rns_dipole_lightbulb"
#define USER_PROBLEM_ENABLED 1

#define TWO_PUNCTURES 0

#define ELLIPTICA 1

#define LORENE 0

// Code versions defined in CMakeLists.txt
#define ATHENA_VERSION_MAJOR 0
#define ATHENA_VERSION_MINOR 1

// use single precision floating-point values (binary32)? default=0 (false; use binary64)
#define SINGLE_PRECISION_ENABLED 0

// use MPI parallelization? default=0 (false)
#define MPI_PARALLEL_ENABLED 0

// use OpenMP parallelization? default=0 (false)
#define OPENMP_PARALLEL_ENABLED 1

// Kokkos tight loop layout
//#define 

// Kokkos nested loop layout (inner)
//#define 

// compiler options
#define COMPILED_WITH ""
#define COMPILER_COMMAND ""
#define COMPILED_WITH_OPTIONS "" // NOLINT

#endif // CONFIG_HPP_
