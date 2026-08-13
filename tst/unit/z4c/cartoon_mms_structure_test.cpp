//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cartoon_mms_structure_test.cpp
//! \brief Lightweight parity and exact provider-reach checks; no convergence grids.

#include <cstdlib>

#include <Kokkos_Core.hpp>

#include "cartoon_derivatives_test_common.hpp"

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  bool passed = false;
  {
    passed = CheckParity() &&
             CheckFullApiAndCartesianDelegation<2>(0.5) &&
             CheckFullApiAndCartesianDelegation<3>(0.5) &&
             CheckFullApiAndCartesianDelegation<4>(0.5) &&
             CheckBlockBoundaryReach<2>() &&
             CheckBlockBoundaryReach<3>() &&
             CheckBlockBoundaryReach<4>();
  }
  Kokkos::finalize();
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
