//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file amr_cadence_test.cpp
//! \brief Integer cadence validation for adaptive mesh refinement.

#include <iostream>

#include "mesh/amr_cadence.hpp"

int main() {
  const bool passed = IsValidAMRCadence(1, 1) &&
                      IsValidAMRCadence(5, 17) &&
                      !IsValidAMRCadence(0, 1) &&
                      !IsValidAMRCadence(1, 0) &&
                      !IsValidAMRCadence(-1, 3) &&
                      !IsValidAMRCadence(3, -1);
  if (!passed) {
    std::cerr << "AMR cadence validation regression failed\n";
    return 1;
  }
  std::cout << "AMR cadence validation regression passed\n";
  return 0;
}
