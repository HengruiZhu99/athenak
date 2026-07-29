//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file scalar_field.cpp
//! \brief Device-side unit tests for canonical scalar-field algebra.

#include <cstdlib>
#include <iostream>
#include <limits>

#include <Kokkos_Core.hpp>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "scalar_field/scalar_field_utils.hpp"

namespace {

KOKKOS_INLINE_FUNCTION
bool NearlyEqual(const Real actual, const Real expected) {
  const Real scale = fmax(1.0, fabs(expected));
  const Real tolerance = 64.0*std::numeric_limits<Real>::epsilon()*scale;
  return fabs(actual - expected) <= tolerance;
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::ScalarFieldUnitTests()
//! \brief Check potential, stress-energy, charge, and additive-accumulator conventions.

void ProblemGenerator::ScalarFieldUnitTests(ParameterInput *pin, const bool restart) {
  bool passed = true;
  Kokkos::parallel_reduce(
      "scalar_field_algebra_unit_tests",
      Kokkos::RangePolicy<DevExeSpace>(0, 1),
      KOKKOS_LAMBDA(const int, bool &local_pass) {
        using scalar_field::MatterPoint;
        using scalar_field::PotentialData;
        using scalar_field::PotentialType;

        const Real phi[2] = {2.0, -1.5};
        const Real pi[2] = {0.75, -1.25};
        const Real gradient[2][3] = {
          {1.0, -2.0, 0.5},
          {-0.25, 1.5, -1.0}
        };
        // Packed order is xx, xy, xz, yy, yz, zz.
        const Real metric[6] = {2.0, 0.5, 0.0, 1.0, 0.0, 1.5};

        const PotentialData free_potential(PotentialType::free, 1.2, 0.7);
        const PotentialData quartic_potential(
            PotentialType::mass_quartic, 1.2, 0.7);

        const Real q_real = scalar_field::FieldInvariant(1, phi);
        const Real q_complex = scalar_field::FieldInvariant(2, phi);
        local_pass = local_pass && NearlyEqual(q_real, 2.0);
        local_pass = local_pass && NearlyEqual(q_complex, 3.125);
        local_pass = local_pass &&
                     NearlyEqual(scalar_field::FieldAmplitude(2, phi), 2.5);

        local_pass = local_pass &&
                     NearlyEqual(free_potential.Energy(q_complex), 4.5);
        local_pass = local_pass &&
                     NearlyEqual(quartic_potential.Energy(q_complex),
                                 11.3359375);
        local_pass = local_pass &&
                     NearlyEqual(quartic_potential.Derivative(phi[0], q_complex),
                                 11.63);
        local_pass = local_pass &&
                     NearlyEqual(quartic_potential.Derivative(phi[1], q_complex),
                                 -8.7225);
        local_pass = local_pass &&
                     NearlyEqual(quartic_potential.FrequencySquared(q_complex),
                                 14.565);

        Real inverse_metric[6];
        Real determinant = 0.0;
        scalar_field::InvertMetric(metric, inverse_metric, &determinant);
        local_pass = local_pass && NearlyEqual(determinant, 2.625);
        local_pass = local_pass && NearlyEqual(inverse_metric[0], 4.0/7.0);
        local_pass = local_pass && NearlyEqual(inverse_metric[1], -2.0/7.0);
        local_pass = local_pass && NearlyEqual(inverse_metric[3], 8.0/7.0);
        local_pass = local_pass && NearlyEqual(inverse_metric[5], 2.0/3.0);

        // This assertion fails if the xy term is not doubled in a symmetric contraction.
        local_pass = local_pass &&
                     NearlyEqual(scalar_field::ContractCovector(
                                     inverse_metric, gradient[0]),
                                 271.0/42.0);

        const MatterPoint matter = scalar_field::ComputeMatter(
            2, phi, pi, gradient, metric, quartic_potential);
        local_pass = local_pass && NearlyEqual(matter.energy, 17.368675595238095);
        local_pass = local_pass && NearlyEqual(matter.momentum[0], 1.0625);
        local_pass = local_pass && NearlyEqual(matter.momentum[1], -3.375);
        local_pass = local_pass && NearlyEqual(matter.momentum[2], 1.625);
        local_pass = local_pass && NearlyEqual(matter.stress[0], -29.42485119047619);
        local_pass = local_pass && NearlyEqual(matter.stress[1], -9.996837797619047);
        local_pass = local_pass && NearlyEqual(matter.stress[2], 0.75);
        local_pass = local_pass && NearlyEqual(matter.stress[3], -8.993675595238095);
        local_pass = local_pass && NearlyEqual(matter.stress[4], -2.5);
        local_pass = local_pass && NearlyEqual(matter.stress[5], -21.615513392857142);
        local_pass = local_pass && NearlyEqual(matter.charge, 1.375);

        MatterPoint accumulator;
        scalar_field::ClearMatter(&accumulator);
        local_pass = local_pass && accumulator.energy == 0.0;
        for (int a = 0; a < 3; ++a) {
          local_pass = local_pass && accumulator.momentum[a] == 0.0;
        }
        for (int n = 0; n < 6; ++n) {
          local_pass = local_pass && accumulator.stress[n] == 0.0;
        }
        local_pass = local_pass && accumulator.charge == 0.0;

        scalar_field::AddMatter(matter, &accumulator);
        scalar_field::AddMatter(matter, &accumulator);
        local_pass = local_pass &&
                     NearlyEqual(accumulator.energy, 2.0*matter.energy);
        for (int a = 0; a < 3; ++a) {
          local_pass = local_pass &&
                       NearlyEqual(accumulator.momentum[a],
                                   2.0*matter.momentum[a]);
        }
        for (int n = 0; n < 6; ++n) {
          local_pass = local_pass &&
                       NearlyEqual(accumulator.stress[n], 2.0*matter.stress[n]);
        }
        local_pass = local_pass &&
                     NearlyEqual(accumulator.charge, 2.0*matter.charge);
      },
      Kokkos::LAnd<bool>(passed));

  if (!passed) {
    std::cout << "Scalar-field algebra unit test failed." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "Scalar-field algebra unit test passed." << std::endl;
}
