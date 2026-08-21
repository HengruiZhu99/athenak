//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c_history_quadrature.hpp
//! \brief Native-VC leaf quadrature weights for Z4c history diagnostics.

#ifndef Z4C_Z4C_HISTORY_QUADRATURE_HPP_
#define Z4C_Z4C_HISTORY_QUADRATURE_HPP_

#include <limits>

#include "athena.hpp"
#include "z4c/z4c_symmetry.hpp"

namespace z4c {

inline constexpr Real kZ4cHistoryTwoPi =
    6.2831853071795864769252867665590057683943387987502;

KOKKOS_INLINE_FUNCTION
Real Z4cNodalTrapezoidWeight(const int index, const int start, const int end,
                            const bool collapsed) {
  if (collapsed) return 1.0;
  return (index == start || index == end) ? 0.5 : 1.0;
}

//! Proper dual-volume measure for one local leaf-block vertex.  Shared endpoint
//! copies are intentionally retained: their half weights tile the two neighboring
//! leaf dual volumes.  A non-positive/nonfinite determinant remains visible as NaN.
KOKKOS_INLINE_FUNCTION
Real Z4cDiagnosticVertexMeasure(
    const Z4cSymmetryMode mode, const Real rho,
    const Real dx1, const Real dx2, const Real dx3,
    const Real spatial_determinant,
    const Real weight1, const Real weight2, const Real weight3) {
  if (!Kokkos::isfinite(spatial_determinant) || spatial_determinant <= 0.0) {
    return std::numeric_limits<Real>::quiet_NaN();
  }
  const Real proper_factor = Kokkos::sqrt(spatial_determinant);
  if (mode == Z4cSymmetryMode::cartoon_so2) {
    return rho > 0.0 ? kZ4cHistoryTwoPi * rho * dx1 * dx2 *
                           weight1 * weight2 * proper_factor
                     : 0.0;
  }
  return dx1 * dx2 * dx3 * weight1 * weight2 * weight3 * proper_factor;
}

}  // namespace z4c

#endif  // Z4C_Z4C_HISTORY_QUADRATURE_HPP_
