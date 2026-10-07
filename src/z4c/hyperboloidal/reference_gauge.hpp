// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#ifndef Z4C_HYPERBOLOIDAL_REFERENCE_GAUGE_HPP_
#define Z4C_HYPERBOLOIDAL_REFERENCE_GAUGE_HPP_

#include <initializer_list>
#include "z4c/hyperboloidal/cmc_reference.hpp"

namespace z4c {
namespace hyperboloidal {

// Factored deviations: alpha=alpha_ref+omega*lapse, beta=beta_ref+omega*shift,
// Delta(K_phys-2*Theta_phys)=omega*trace. DO NOT use AthenaK's unscaled Khat
// in place of trace. Input gradients are derivatives of the factored deviations.
template <typename T>
struct GaugeDeviation {
  T lapse, shift[3], trace, chi, lambda[3];
  T dlapse[3], dshift[3][3];  // dshift[component][derivative]
};

template <typename T>
struct GaugeParameters {
  T slicing, shift_driver, lapse_damping, shift_damping;
  void Validate() const {
    for (T v : {slicing, shift_driver, lapse_damping, shift_damping}) {
      if (!std::isfinite(v) || v < 0) {
        throw std::invalid_argument("Reference gauge coefficients must be finite >=0");
      }
    }
  }
};

template <typename T>
struct GaugeRHS {
  T alpha, beta[3];
};

// Algebraic Cartesian extension of equations (9),(10), arXiv:2408.08952v2.
// slicing and shift_driver multiply (S^2-r^2)^2 as in that paper.
// Returns RHS of the UNFACTORED lapse and shift, with no division by omega.
// This is a gauge building block, NOT a closed evolution for GaugeDeviation:
// at scri its RHS must also vanish to preserve the assumed falloff. In particular
// alpha_rhs=0 is a compatibility relation involving trace, lapse and shift.
// No claim about the full coupled Z4c principal symbol is made here.
template <typename T>
KOKKOS_INLINE_FUNCTION
GaugeRHS<T> ReferenceGauge(const CMCReference<T> &ref, const CMCPoint<T> &p,
                          const GaugeDeviation<T> &d, const GaugeParameters<T> &g) {
  const T alpha = p.alpha + p.omega*d.lapse;
  T beta[3];
  for (int i = 0; i < 3; ++i) beta[i] = p.beta[i] + p.omega*d.shift[i];
  const T radial = 2*ref.curvature_radius*ref.scri_radius*p.omega;
  GaugeRHS<T> rhs{};
  rhs.alpha = -(alpha*alpha + g.slicing*radial*radial)*d.trace
      - g.lapse_damping*(2*p.alpha*d.lapse+p.omega*d.lapse*d.lapse);
  for (int j = 0; j < 3; ++j) {
    // Expanded advection minus expanded omega'/omega product. The terms
    // beta*domega*lapse cancel analytically, including at omega=0.
    rhs.alpha += p.omega*(d.shift[j]*p.dalpha[j]+beta[j]*d.dlapse[j])
                 - p.alpha*p.domega[j]*d.shift[j];
  }
  for (int i = 0; i < 3; ++i) {
    rhs.beta[i] = -p.omega*d.shift[i]/ref.curvature_radius
        + (g.shift_driver*radial*radial + T(0.75)*alpha*alpha*d.chi)*d.lambda[i]
        - g.shift_damping*p.omega*d.shift[i];
    for (int j = 0; j < 3; ++j) {
      rhs.beta[i] += beta[j]*(p.domega[j]*d.shift[i]+p.omega*d.dshift[i][j]);
    }
  }
  return rhs;
}

}  // namespace hyperboloidal
}  // namespace z4c
#endif  // Z4C_HYPERBOLOIDAL_REFERENCE_GAUGE_HPP_
