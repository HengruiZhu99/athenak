//========================================================================================
//! \file stationary_gauge_data.hpp
//! \brief Exact ordinary-GH gauge-driver state for the stationary trumpet reference.
//========================================================================================
#ifndef REF_GH_STATIONARY_GAUGE_DATA_HPP_
#define REF_GH_STATIONARY_GAUGE_DATA_HPP_

#include "athena.hpp"
#include "ref_gh/gauge_driver.hpp"
#include "ref_gh/reference_geometry.hpp"
#include "ref_gh/standard_gh_source.hpp"

namespace ref_gh {

struct StationaryGaugeState {
  Real hhat[4];
  Real theta[4];
  Real upsilon[3];
  bool valid;
};

// For the exact stationary reference metric, the physical 1+log/Gamma-driver
// target equals the constraint-satisfying ordinary GH source.  Its frame
// components and the auxiliary theta needed for dt(Hhat_A)=0 are analytic in
// the reference jets.  This is boundary/diagnostic plumbing only; it does not
// alter the evolved gauge equations.
KOKKOS_INLINE_FUNCTION
StationaryGaugeState ComputeStationaryTrumpetGaugeState(
    const DvceArray2D<Real> &table, const Real mass,
    const Real center_x, const Real center_y, const Real center_z,
    const Real x, const Real y, const Real z) {
  StationaryGaugeState result{};
  result.valid = false;
  ReferenceGeometry reference;
  GetReferenceGeometry(1, table, mass, center_x, center_y, center_z, 0.0,
                       x, y, z, reference);
  CoordinateGhGeometry geometry;
  Real determinant = 0.0;
  if (!ComputeCoordinateGhGeometry(
          reference.metric, reference.d_metric, reference, geometry,
          determinant)) return result;

  Real d_base_lower[4][4] = {};  // NOLINT(runtime/arrays)
  ImplicitGaugeSourceDerivative(
      reference.metric, reference.d_metric, reference, geometry,
      d_base_lower);
  for (int A = 0; A < 4; ++A) {
    for (int a = 0; a < 4; ++a) {
      result.hhat[A] +=
          reference.frame[A][a]*geometry.gauge_source[a];
    }
  }
  for (int A = 0; A < 4; ++A) {
    for (int p = 0; p < 3; ++p) {
      Real d_hhat = 0.0;
      for (int a = 0; a < 4; ++a) {
        d_hhat += reference.d_frame[p + 1][A][a]
                      *geometry.gauge_source[a]
                  + reference.frame[A][a]*d_base_lower[p + 1][a];
      }
      result.theta[A] -= geometry.shift[p]*d_hhat;
    }
    for (int B = 0; B < 4; ++B) {
      Real frame_motion = ReferenceFrameMotion(reference, A, 0, B);
      for (int p = 0; p < 3; ++p) {
        frame_motion -= geometry.shift[p]
            *ReferenceFrameMotion(reference, A, p + 1, B);
      }
      result.theta[A] -= frame_motion*result.hhat[B];
    }
  }
  result.valid = true;
  return result;
}

}  // namespace ref_gh

#endif  // REF_GH_STATIONARY_GAUGE_DATA_HPP_
