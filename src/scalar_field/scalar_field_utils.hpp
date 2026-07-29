#ifndef SCALAR_FIELD_SCALAR_FIELD_UTILS_HPP_
#define SCALAR_FIELD_SCALAR_FIELD_UTILS_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file scalar_field_utils.hpp
//! \brief Device-callable algebra for canonical real and complex scalar fields.

#include <cmath>

#include "athena.hpp"

namespace scalar_field {

//! Supported canonical scalar-field potentials.
enum class PotentialType {
  free,
  mass_quartic
};

//! Small device-copyable representation shared by evolution and diagnostics.
struct PotentialData {
  PotentialType type;
  Real mass_squared;
  Real quartic_coupling;

  KOKKOS_INLINE_FUNCTION
  PotentialData() = default;

  KOKKOS_INLINE_FUNCTION
  PotentialData(const PotentialType type_in, const Real mass,
                const Real quartic_coupling_in)
      : type(type_in),
        mass_squared(mass*mass),
        quartic_coupling(quartic_coupling_in) {}

  //! Return V(q), where q = sum_A phi_A^2/2.
  KOKKOS_INLINE_FUNCTION
  Real Energy(const Real q) const {
    Real energy = mass_squared*q;
    if (type == PotentialType::mass_quartic) {
      energy += quartic_coupling*q*q;
    }
    return energy;
  }

  //! Return dV/dq.
  KOKKOS_INLINE_FUNCTION
  Real DerivativeQ(const Real q) const {
    Real derivative = mass_squared;
    if (type == PotentialType::mass_quartic) {
      derivative += 2.0*quartic_coupling*q;
    }
    return derivative;
  }

  //! Return dV/dphi_A = (dV/dq) phi_A.
  KOKKOS_INLINE_FUNCTION
  Real Derivative(const Real phi, const Real q) const {
    return DerivativeQ(q)*phi;
  }

  //! Conservative upper bound on the local potential Hessian eigenvalue.
  KOKKOS_INLINE_FUNCTION
  Real FrequencySquared(const Real q) const {
    Real frequency_squared = mass_squared;
    if (type == PotentialType::mass_quartic) {
      frequency_squared += 6.0*quartic_coupling*q;
    }
    return frequency_squared;
  }
};

//! Canonical invariant q = sum_A phi_A^2/2.
KOKKOS_INLINE_FUNCTION
Real FieldInvariant(const int ncomponents, const Real phi[2]) {
  Real q = 0.0;
  for (int component = 0; component < ncomponents; ++component) {
    q += 0.5*phi[component]*phi[component];
  }
  return q;
}

//! Field-space amplitude sqrt(sum_A phi_A^2).
KOKKOS_INLINE_FUNCTION
Real FieldAmplitude(const int ncomponents, const Real phi[2]) {
  return sqrt(2.0*FieldInvariant(ncomponents, phi));
}

//! Invert a symmetric 3-metric packed as xx, xy, xz, yy, yz, zz.
KOKKOS_INLINE_FUNCTION
void InvertMetric(const Real metric[6], Real inverse[6], Real *determinant) {
  const Real det =
      metric[0]*metric[3]*metric[5] + 2.0*metric[1]*metric[2]*metric[4] -
      metric[0]*metric[4]*metric[4] - metric[3]*metric[2]*metric[2] -
      metric[5]*metric[1]*metric[1];
  const Real inverse_det = 1.0/det;

  inverse[0] = (metric[3]*metric[5] - metric[4]*metric[4])*inverse_det;
  inverse[1] = (metric[2]*metric[4] - metric[1]*metric[5])*inverse_det;
  inverse[2] = (metric[1]*metric[4] - metric[2]*metric[3])*inverse_det;
  inverse[3] = (metric[0]*metric[5] - metric[2]*metric[2])*inverse_det;
  inverse[4] = (metric[1]*metric[2] - metric[0]*metric[4])*inverse_det;
  inverse[5] = (metric[0]*metric[3] - metric[1]*metric[1])*inverse_det;

  if (determinant != nullptr) {
    *determinant = det;
  }
}

//! Contract gamma^{ij} v_i v_j with explicit symmetric off-diagonal factors.
KOKKOS_INLINE_FUNCTION
Real ContractCovector(const Real inverse_metric[6], const Real covector[3]) {
  return inverse_metric[0]*covector[0]*covector[0] +
         inverse_metric[3]*covector[1]*covector[1] +
         inverse_metric[5]*covector[2]*covector[2] +
         2.0*(inverse_metric[1]*covector[0]*covector[1] +
              inverse_metric[2]*covector[0]*covector[2] +
              inverse_metric[4]*covector[1]*covector[2]);
}

//! Undensitized ADM matter variables at one point.
struct MatterPoint {
  Real energy;
  Real momentum[3];
  Real stress[6];
  Real charge;
};

//! Return phi_1 Pi_0 - phi_0 Pi_1 for a complex field, or zero for a real field.
KOKKOS_INLINE_FUNCTION
Real ChargeDensity(const int ncomponents, const Real phi[2], const Real pi[2]) {
  return (ncomponents == 2) ? phi[1]*pi[0] - phi[0]*pi[1] : 0.0;
}

//! Set every matter component to exactly zero.
KOKKOS_INLINE_FUNCTION
void ClearMatter(MatterPoint *matter) {
  matter->energy = 0.0;
  for (int direction = 0; direction < 3; ++direction) {
    matter->momentum[direction] = 0.0;
  }
  for (int component = 0; component < 6; ++component) {
    matter->stress[component] = 0.0;
  }
  matter->charge = 0.0;
}

//! Add one point's matter variables without overwriting earlier producers.
KOKKOS_INLINE_FUNCTION
void AddMatter(const MatterPoint &source, MatterPoint *destination) {
  destination->energy += source.energy;
  for (int direction = 0; direction < 3; ++direction) {
    destination->momentum[direction] += source.momentum[direction];
  }
  for (int component = 0; component < 6; ++component) {
    destination->stress[component] += source.stress[component];
  }
  destination->charge += source.charge;
}

//! Compute E, S_i, S_ij, and charge for canonical scalar components.
KOKKOS_INLINE_FUNCTION
MatterPoint ComputeMatter(const int ncomponents, const Real phi[2],
                          const Real pi[2], const Real gradient[2][3],
                          const Real metric[6], const PotentialData &potential) {
  MatterPoint matter;
  ClearMatter(&matter);

  Real inverse_metric[6];
  InvertMetric(metric, inverse_metric, nullptr);

  Real pi_squared = 0.0;
  Real gradient_squared = 0.0;
  for (int component = 0; component < ncomponents; ++component) {
    pi_squared += pi[component]*pi[component];
    gradient_squared += ContractCovector(inverse_metric, gradient[component]);

    for (int direction = 0; direction < 3; ++direction) {
      matter.momentum[direction] +=
          pi[component]*gradient[component][direction];
    }

    matter.stress[0] += gradient[component][0]*gradient[component][0];
    matter.stress[1] += gradient[component][0]*gradient[component][1];
    matter.stress[2] += gradient[component][0]*gradient[component][2];
    matter.stress[3] += gradient[component][1]*gradient[component][1];
    matter.stress[4] += gradient[component][1]*gradient[component][2];
    matter.stress[5] += gradient[component][2]*gradient[component][2];
  }

  const Real potential_energy =
      potential.Energy(FieldInvariant(ncomponents, phi));
  matter.energy = 0.5*(pi_squared + gradient_squared) + potential_energy;

  const Real isotropic_stress =
      0.5*(pi_squared - gradient_squared) - potential_energy;
  for (int component = 0; component < 6; ++component) {
    matter.stress[component] += metric[component]*isotropic_stress;
  }

  matter.charge = ChargeDensity(ncomponents, phi, pi);
  return matter;
}

}  // namespace scalar_field

#endif  // SCALAR_FIELD_SCALAR_FIELD_UTILS_HPP_
