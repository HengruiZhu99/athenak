//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cartoon_derivatives_test.cpp
//! \brief Driver for the split analytic SO(2) manufactured-field tests.

#include <cstdlib>
#include <iostream>

#include <Kokkos_Core.hpp>

bool RunCartoonDerivativeOrder2();
bool RunCartoonDerivativeOrder4();
bool RunCartoonDerivativeOrder6();

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  bool passed = false;
  {
    const bool order2_passed = RunCartoonDerivativeOrder2();
    const bool order4_passed = RunCartoonDerivativeOrder4();
    const bool order6_passed = RunCartoonDerivativeOrder6();
    passed = order2_passed && order4_passed && order6_passed;
  }
  Kokkos::finalize();
  if (!passed) {
    return EXIT_FAILURE;
  }
  std::cout << "Cartoon derivative manufactured-oracle tests passed\n";
  return EXIT_SUCCESS;
}
