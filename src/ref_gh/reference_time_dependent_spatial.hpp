//========================================================================================
//! \file reference_time_dependent_spatial.hpp
//! \brief Exact Minkowski reference with a time-dependent Cartesian spatial frame.
//========================================================================================
#ifndef REF_GH_REFERENCE_TIME_DEPENDENT_SPATIAL_HPP_
#define REF_GH_REFERENCE_TIME_DEPENDENT_SPATIAL_HPP_

#include "athena.hpp"
#include "ref_gh/reference_geometry.hpp"
#include "ref_gh/reference_trumpet_schwarzschild.hpp"

namespace ref_gh {

// Pull back inertial Minkowski coordinates by T=t and X^I=a(t)x^I.  The
// orthonormal coframe is theta^0=dt and
// theta^I=a dx^I + adot x^I dt.  Keeping a and adot/a as analytic two-jets
// exercises the time derivative of the spatial frame, reference shift, spin,
// and spin derivatives while the exact reference Riemann tensor remains zero.
struct TimeDependentSpatialReference {
  static constexpr Real amplitude = 0.1;
  static constexpr Real angular_frequency = 0.7;

  KOKKOS_INLINE_FUNCTION
  ReferenceJet ScaleJet(const Real time) const {
    ReferenceJet scale = ConstantJet(
        1.0 + amplitude*Kokkos::sin(angular_frequency*time));
    scale.d[0] = amplitude*angular_frequency
                 *Kokkos::cos(angular_frequency*time);
    scale.dd[0][0] = -amplitude*angular_frequency*angular_frequency
                     *Kokkos::sin(angular_frequency*time);
    return scale;
  }

  KOKKOS_INLINE_FUNCTION
  ReferenceJet ScaleRateJet(const Real time) const {
    ReferenceJet rate = ConstantJet(
        amplitude*angular_frequency*Kokkos::cos(angular_frequency*time));
    rate.d[0] = -amplitude*angular_frequency*angular_frequency
                *Kokkos::sin(angular_frequency*time);
    rate.dd[0][0] = -amplitude*angular_frequency*angular_frequency
                    *angular_frequency*Kokkos::cos(angular_frequency*time);
    return rate;
  }

  KOKKOS_INLINE_FUNCTION
  ReferenceJet ShiftQJet(const Real time) const {
    return ScaleRateJet(time)*Reciprocal(ScaleJet(time));
  }

  KOKKOS_INLINE_FUNCTION
  void PopulatePsiKinematics(const Real time, const Real x, const Real y,
                             const Real z,
                             ReferencePsiKinematics &reference) const {
    ZeroReferencePsiKinematics(reference);
    const Real coordinates[3] = {x, y, z};
    const Real scale = ScaleJet(time).value;
    const Real rate = ScaleRateJet(time).value;
    reference.coframe[0][0] = 1.0;
    for (int I = 0; I < 3; ++I) {
      reference.coframe[I + 1][0] = rate*coordinates[I];
      reference.coframe[I + 1][I + 1] = scale;
      reference.spatial_coframe[I][I] = scale;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void Populate(const Real time, const Real x, const Real y, const Real z,
                ReferenceGeometry &reference) const {
    ZeroReferenceGeometry(reference);
    const ReferenceJet scale = ScaleJet(time);
    const ReferenceJet inverse_scale = Reciprocal(scale);
    const ReferenceJet shift_q = ShiftQJet(time);
    const ReferenceJet coordinates[3] = {
        CoordinateJet(x, 1), CoordinateJet(y, 2), CoordinateJet(z, 3)};

    ReferenceJet coframe[4][4];  // NOLINT(runtime/arrays)
    ReferenceJet frame[4][4];    // NOLINT(runtime/arrays)
    for (int A = 0; A < 4; ++A) {
      for (int a = 0; a < 4; ++a) {
        coframe[A][a] = ConstantJet(0.0);
        frame[A][a] = ConstantJet(0.0);
      }
    }
    coframe[0][0] = ConstantJet(1.0);
    frame[0][0] = ConstantJet(1.0);
    for (int I = 0; I < 3; ++I) {
      coframe[I + 1][0] = scale*shift_q*coordinates[I];
      coframe[I + 1][I + 1] = scale;
      frame[0][I + 1] = -(shift_q*coordinates[I]);
      frame[I + 1][I + 1] = inverse_scale;
    }

    ReferenceJet metric[4][4];          // NOLINT(runtime/arrays)
    ReferenceJet inverse_metric[4][4];  // NOLINT(runtime/arrays)
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) {
        metric[a][b] = -(coframe[0][a]*coframe[0][b]);
        inverse_metric[a][b] = -(frame[0][a]*frame[0][b]);
        for (int I = 1; I < 4; ++I) {
          metric[a][b] = metric[a][b] + coframe[I][a]*coframe[I][b];
          inverse_metric[a][b] =
              inverse_metric[a][b] + frame[I][a]*frame[I][b];
        }
        reference.metric[a][b] = metric[a][b].value;
        reference.inverse_metric[a][b] = inverse_metric[a][b].value;
        reference.coframe[a][b] = coframe[a][b].value;
        reference.frame[a][b] = frame[a][b].value;
        for (int p = 0; p < 4; ++p) {
          reference.d_metric[p][a][b] = metric[a][b].d[p];
          reference.d_frame[p][a][b] = frame[a][b].d[p];
          for (int q = 0; q < 4; ++q) {
            reference.dd_metric[p][q][a][b] = metric[a][b].dd[p][q];
            reference.dd_frame[p][q][a][b] = frame[a][b].dd[p][q];
          }
        }
      }
    }
    for (int I = 0; I < 3; ++I) {
      reference.spatial_frame[I][I] = inverse_scale.value;
      reference.spatial_coframe[I][I] = scale.value;
      reference.dt_spatial_frame[I][I] = inverse_scale.d[0];
    }

    Real first_kind[4][4][4];  // NOLINT(runtime/arrays)
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) {
        for (int c = 0; c < 4; ++c) {
          first_kind[a][b][c] = 0.5*(reference.d_metric[b][a][c]
                                      + reference.d_metric[c][a][b]
                                      - reference.d_metric[a][b][c]);
        }
      }
    }
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) {
        for (int c = 0; c < 4; ++c) {
          for (int ell = 0; ell < 4; ++ell) {
            reference.christoffel[a][b][c] +=
                reference.inverse_metric[a][ell]*first_kind[ell][b][c];
          }
          for (int p = 0; p < 4; ++p) {
            for (int ell = 0; ell < 4; ++ell) {
              const Real d_first = 0.5*(reference.dd_metric[p][b][ell][c]
                                         + reference.dd_metric[p][c][ell][b]
                                         - reference.dd_metric[p][ell][b][c]);
              reference.d_christoffel[p][a][b][c] +=
                  inverse_metric[a][ell].d[p]*first_kind[ell][b][c]
                  + reference.inverse_metric[a][ell]*d_first;
            }
          }
        }
      }
    }
    CompleteReferenceFrameGeometry(reference);
  }
};

}  // namespace ref_gh

#endif  // REF_GH_REFERENCE_TIME_DEPENDENT_SPATIAL_HPP_
