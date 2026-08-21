//========================================================================================
//! \file reference_time_dependent_lapse.hpp
//! \brief Flat reference with a genuinely time-dependent lapse for cache tests.
//========================================================================================
#ifndef REF_GH_REFERENCE_TIME_DEPENDENT_LAPSE_HPP_
#define REF_GH_REFERENCE_TIME_DEPENDENT_LAPSE_HPP_

#include "athena.hpp"
#include "ref_gh/reference_geometry.hpp"
#include "ref_gh/reference_trumpet_schwarzschild.hpp"

namespace ref_gh {

struct TimeDependentLapseReference {
  static constexpr Real amplitude = 0.1;

  KOKKOS_INLINE_FUNCTION
  ReferenceJet AlphaJet(const Real time) const {
    ReferenceJet alpha = ConstantJet(1.0 + amplitude*Kokkos::sin(time));
    alpha.d[0] = amplitude*Kokkos::cos(time);
    alpha.dd[0][0] = -amplitude*Kokkos::sin(time);
    return alpha;
  }

  KOKKOS_INLINE_FUNCTION
  void PopulatePsiKinematics(const Real time, const Real, const Real, const Real,
                             ReferencePsiKinematics &reference) const {
    ZeroReferencePsiKinematics(reference);
    reference.coframe[0][0] = AlphaJet(time).value;
    for (int i = 0; i < 3; ++i) {
      reference.coframe[i + 1][i + 1] = 1.0;
      reference.spatial_coframe[i][i] = 1.0;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void Populate(const Real time, const Real, const Real, const Real,
                ReferenceGeometry &reference) const {
    ZeroReferenceGeometry(reference);
    const ReferenceJet alpha = AlphaJet(time);
    const ReferenceJet inverse_alpha = Reciprocal(alpha);
    ReferenceJet coframe[4][4];  // NOLINT(runtime/arrays)
    ReferenceJet frame[4][4];    // NOLINT(runtime/arrays)
    for (int A = 0; A < 4; ++A) {
      for (int a = 0; a < 4; ++a) {
        coframe[A][a] = ConstantJet(0.0);
        frame[A][a] = ConstantJet(0.0);
      }
    }
    coframe[0][0] = alpha;
    frame[0][0] = inverse_alpha;
    for (int i = 1; i < 4; ++i) {
      coframe[i][i] = ConstantJet(1.0);
      frame[i][i] = ConstantJet(1.0);
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
      reference.spatial_frame[I][I] = 1.0;
      reference.spatial_coframe[I][I] = 1.0;
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
    for (int A = 0; A < 4; ++A) {
      for (int B = 0; B < 4; ++B) {
        reference.ricci_frame[A][B] = 0.0;
        for (int C = 0; C < 4; ++C) {
          for (int D = 0; D < 4; ++D) {
            reference.riemann_frame[A][B][C][D] = 0.0;
          }
        }
      }
    }
  }
};

}  // namespace ref_gh

#endif  // REF_GH_REFERENCE_TIME_DEPENDENT_LAPSE_HPP_
