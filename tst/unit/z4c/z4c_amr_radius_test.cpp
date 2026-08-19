//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c_amr_radius_test.cpp
//! \brief Exact AABB distance regression for Z4c radial refinement.

#include <cmath>
#include <iostream>

#include "z4c/z4c_amr.hpp"

namespace {

bool Equal(const Real left, const Real right) {
  return std::abs(left - right) < 1.0e-15;
}

}  // namespace

int main() {
  using z4c::SquaredDistanceToAABB;
  bool passed = true;
  // Origin inside the block.
  passed &= Equal(SquaredDistanceToAABB(0.0, 0.0, 0.0, -1.0, 1.0, -2.0, 2.0,
                                        -3.0, 3.0), 0.0);
  // Closest point on a face, edge, and corner respectively.
  passed &= Equal(SquaredDistanceToAABB(0.0, 0.0, 0.0, 2.0, 3.0, -1.0, 1.0,
                                        -1.0, 1.0), 4.0);
  passed &= Equal(SquaredDistanceToAABB(0.0, 0.0, 0.0, 2.0, 3.0, 4.0, 5.0,
                                        -1.0, 1.0), 20.0);
  passed &= Equal(SquaredDistanceToAABB(0.0, 0.0, 0.0, 2.0, 3.0, 4.0, 5.0,
                                        6.0, 7.0), 56.0);
  // Cartoon half-plane block crosses z=0 but remains rho>0: the radial face,
  // rather than a corner, is nearest.
  passed &= Equal(SquaredDistanceToAABB(0.0, 0.0, 0.0, 1.5, 2.5, -0.25, 0.25,
                                        0.0, 0.0), 2.25);
  if (!passed) {
    std::cerr << "Z4c AMR radius AABB regression failed\n";
    return 1;
  }
  std::cout << "Z4c AMR radius AABB regression passed\n";
  return 0;
}
