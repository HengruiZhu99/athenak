//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c_vertex_history_quadrature_test.cpp
//! \brief Leaf-block nodal quadrature tests for native-VC Z4c history.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <Kokkos_Core.hpp>

#include "coordinates/cell_locations.hpp"
#include "z4c/z4c_history_quadrature.hpp"

namespace {

bool Near(const Real left, const Real right, const Real tolerance = 5.0e-13) {
  return std::abs(left - right) <=
         tolerance * std::max({Real(1.0), std::abs(left), std::abs(right)});
}

Real IntegrateCartoonBlock(const int nr, const int nz,
                           const Real r0, const Real r1,
                           const Real z0, const Real z1,
                           const bool integrate_z = false) {
  const Real dr = (r1 - r0) / nr;
  const Real dz = (z1 - z0) / nz;
  Real integral = 0.0;
  for (int j = 0; j <= nz; ++j) {
    for (int i = 0; i <= nr; ++i) {
      const Real rho = VertexX(i, nr, r0, r1);
      const Real z = VertexX(j, nz, z0, z1);
      const Real value = integrate_z ? z : 1.0;
      integral += value * z4c::Z4cDiagnosticVertexMeasure(
          z4c::Z4cSymmetryMode::cartoon_so2, rho, dr, dz, 1.0, 1.0,
          z4c::Z4cNodalTrapezoidWeight(i, 0, nr, false),
          z4c::Z4cNodalTrapezoidWeight(j, 0, nz, false), 1.0);
    }
  }
  return integral;
}

bool TestSingleBlockRingVolumeAndPolynomial() {
  constexpr Real radius = 2.0;
  constexpr Real z0 = -1.5;
  constexpr Real z1 = 0.5;
  const Real volume = IntegrateCartoonBlock(8, 10, 0.0, radius, z0, z1);
  const Real pi = 0.5 * z4c::kZ4cHistoryTwoPi;
  const Real exact_volume = pi * radius * radius * (z1 - z0);
  if (!Near(volume, exact_volume)) return false;
  const Real linear_z =
      IntegrateCartoonBlock(8, 10, 0.0, radius, z0, z1, true);
  const Real exact_linear =
      pi * radius * radius * 0.5 * (z1 * z1 - z0 * z0);
  return Near(linear_z, exact_linear);
}

bool TestMultipleBlocksAndDecompositionOrder() {
  const std::vector<Real> pieces = {
      IntegrateCartoonBlock(4, 4, 0.0, 1.0, -1.0, 0.0),
      IntegrateCartoonBlock(4, 4, 1.0, 2.0, -1.0, 0.0),
      IntegrateCartoonBlock(4, 4, 0.0, 1.0, 0.0, 1.0),
      IntegrateCartoonBlock(4, 4, 1.0, 2.0, 0.0, 1.0)};
  Real forward = 0.0;
  Real simulated_rank0 = 0.0;
  Real simulated_rank1 = 0.0;
  for (std::size_t index = 0; index < pieces.size(); ++index) {
    forward += pieces[index];
    (index % 2 == 0 ? simulated_rank0 : simulated_rank1) += pieces[index];
  }
  const Real exact = 0.5 * z4c::kZ4cHistoryTwoPi * 4.0 * 2.0;
  return Near(forward, exact) && Near(simulated_rank0 + simulated_rank1, exact);
}

bool TestCoarseFineLeafPartition() {
  Real integral = IntegrateCartoonBlock(4, 4, 0.0, 1.0, -1.0, 1.0);
  for (int rz = 0; rz < 2; ++rz) {
    for (int rr = 0; rr < 2; ++rr) {
      integral += IntegrateCartoonBlock(
          8, 8, 1.0 + 0.5 * rr, 1.0 + 0.5 * (rr + 1),
          -1.0 + rz, -1.0 + (rz + 1));
    }
  }
  return Near(integral, 0.5 * z4c::kZ4cHistoryTwoPi * 4.0 * 2.0);
}

bool TestAxisAndMetricFailure() {
  const Real axis = z4c::Z4cDiagnosticVertexMeasure(
      z4c::Z4cSymmetryMode::cartoon_so2, 0.0, 0.1, 0.1, 1.0,
      1.0, 0.5, 0.5, 1.0);
  const Real invalid = z4c::Z4cDiagnosticVertexMeasure(
      z4c::Z4cSymmetryMode::cartoon_so2, 1.0, 0.1, 0.1, 1.0,
      -1.0, 1.0, 1.0, 1.0);
  return axis == 0.0 && std::isnan(invalid);
}

}  // namespace

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  const bool pass = TestSingleBlockRingVolumeAndPolynomial() &&
                    TestMultipleBlocksAndDecompositionOrder() &&
                    TestCoarseFineLeafPartition() && TestAxisAndMetricFailure();
  Kokkos::finalize();
  if (!pass) {
    std::cerr << "native VC history quadrature regression failed" << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "PASS: native VC leaf quadrature, ring volume, and metric gate"
            << std::endl;
  return EXIT_SUCCESS;
}
