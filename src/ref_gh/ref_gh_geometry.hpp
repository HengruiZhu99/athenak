//========================================================================================
//! \file ref_gh_geometry.hpp
//! \brief Cell-local reconstruction of coordinate geometry from regular GH fields.
//========================================================================================
#ifndef REF_GH_REF_GH_GEOMETRY_HPP_
#define REF_GH_REF_GH_GEOMETRY_HPP_

#include "athena.hpp"
#include "ref_gh/ref_gh_state.hpp"
#include "ref_gh/reference_cache.hpp"
#include "ref_gh/reference_controlled_schwarzschild.hpp"
#include "ref_gh/reference_geometry.hpp"
#include "ref_gh/reference_time_dependent_lapse.hpp"
#include "ref_gh/reference_time_dependent_spatial.hpp"
#include "ref_gh/reference_trumpet_schwarzschild.hpp"
#include "ref_gh/standard_gh_source.hpp"

namespace ref_gh {

KOKKOS_INLINE_FUNCTION
Real SymmetricConditionNumber3(Real m00, Real m01, Real m02,
                               Real m11, Real m12, Real m22) {
  for (int sweep = 0; sweep < 18; ++sweep) {
    int p = 0;
    int q = 1;
    Real largest = Kokkos::abs(m01);
    if (Kokkos::abs(m02) > largest) {
      largest = Kokkos::abs(m02);
      q = 2;
    }
    if (Kokkos::abs(m12) > largest) {
      largest = Kokkos::abs(m12);
      p = 1;
      q = 2;
    }
    if (largest < 1.0e-14) break;
    const Real app = p == 0 ? m00 : m11;
    const Real aqq = q == 1 ? m11 : m22;
    const Real apq = q == 1 ? m01 : (p == 0 ? m02 : m12);
    const Real angle = 0.5*Kokkos::atan2(2.0*apq, aqq - app);
    const Real cosine = Kokkos::cos(angle);
    const Real sine = Kokkos::sin(angle);
    const Real rotated_p = cosine*cosine*app - 2.0*sine*cosine*apq
                           + sine*sine*aqq;
    const Real rotated_q = sine*sine*app + 2.0*sine*cosine*apq
                           + cosine*cosine*aqq;
    if (p == 0 && q == 1) {
      const Real old02 = m02;
      const Real old12 = m12;
      m00 = rotated_p;
      m11 = rotated_q;
      m01 = 0.0;
      m02 = cosine*old02 - sine*old12;
      m12 = sine*old02 + cosine*old12;
    } else if (p == 0) {
      const Real old01 = m01;
      const Real old12 = m12;
      m00 = rotated_p;
      m22 = rotated_q;
      m02 = 0.0;
      m01 = cosine*old01 - sine*old12;
      m12 = sine*old01 + cosine*old12;
    } else {
      const Real old01 = m01;
      const Real old02 = m02;
      m11 = rotated_p;
      m22 = rotated_q;
      m12 = 0.0;
      m01 = cosine*old01 - sine*old02;
      m02 = sine*old01 + cosine*old02;
    }
  }
  if (m11 < m00) { const Real temporary = m00; m00 = m11; m11 = temporary; }
  if (m22 < m00) { const Real temporary = m00; m00 = m22; m22 = temporary; }
  if (m22 < m11) { const Real temporary = m11; m11 = m22; m22 = temporary; }
  return m00 > 0.0 ? m22/m00 : 0.0;
}

KOKKOS_INLINE_FUNCTION
void LoadSymmetric(const DvceArray5D<Real> &state, const int offset, const int m,
                   const int k, const int j, const int i, Real tensor[4][4]) {
  for (int a = 0; a < 4; ++a) {
    for (int b = a; b < 4; ++b) {
      tensor[a][b] = tensor[b][a] =
          state(m, offset + Symmetric4Index(a, b), k, j, i);
    }
  }
}

KOKKOS_INLINE_FUNCTION
ReferenceGeometry GetReferenceGeometry(const int reference_kind,
                                       const DvceArray2D<Real> &table,
                                       const Real mass, const Real center_x,
                                       const Real center_y, const Real center_z,
                                       const Real time, const Real x,
                                       const Real y, const Real z) {
  if (reference_kind == 0) return MinkowskiReference()(time, x, y, z);
  if (reference_kind == 2) {
    ReferenceGeometry reference;
    TimeDependentLapseReference().Populate(time, x, y, z, reference);
    return reference;
  }
  if (reference_kind == 3) {
    ReferenceGeometry reference;
    TimeDependentSpatialReference().Populate(time, x, y, z, reference);
    return reference;
  }
  if (reference_kind == 4) {
    ReferenceGeometry reference;
    WormholeSchwarzschildReference provider{
        mass, {center_x, center_y, center_z}};
    provider.Populate(time, x, y, z, reference);
    return reference;
  }
  TrumpetSchwarzschildReference provider{table, mass,
                                         {center_x, center_y, center_z}};
  return provider(time, x, y, z);
}

// Fill caller-owned storage in device kernels.  Avoiding a large aggregate return
// keeps the reference geometry out of the device ABI's temporary return storage.
KOKKOS_INLINE_FUNCTION
void GetReferenceGeometry(const int reference_kind,
                          const DvceArray2D<Real> &table,
                          const Real mass, const Real center_x,
                          const Real center_y, const Real center_z,
                          const Real time, const Real x,
                          const Real y, const Real z,
                          ReferenceGeometry &reference) {
  if (reference_kind == 0) {
    MinkowskiReference().Populate(time, x, y, z, reference);
    return;
  }
  if (reference_kind == 2) {
    TimeDependentLapseReference().Populate(time, x, y, z, reference);
    return;
  }
  if (reference_kind == 3) {
    TimeDependentSpatialReference().Populate(time, x, y, z, reference);
    return;
  }
  if (reference_kind == 4) {
    const WormholeSchwarzschildReference provider{
        mass, {center_x, center_y, center_z}};
    provider.Populate(time, x, y, z, reference);
    return;
  }
  const TrumpetSchwarzschildReference provider{
      table, mass, {center_x, center_y, center_z}};
  provider.Populate(time, x, y, z, reference);
}

KOKKOS_INLINE_FUNCTION
void GetReferenceGeometry(const int reference_kind,
                          const DvceArray2D<Real> &table,
                          const Real mass, const Real center_x,
                          const Real center_y, const Real center_z,
                          const Real time, const Real x,
                          const Real y, const Real z,
                          const ControlledReferenceParameters &controlled,
                          ReferenceGeometry &reference) {
  if (reference_kind == 5) {
    const ControlledSchwarzschildReference provider{table, controlled};
    provider.Populate(time, x, y, z, reference);
    return;
  }
  GetReferenceGeometry(reference_kind, table, mass, center_x, center_y,
                       center_z, time, x, y, z, reference);
}

KOKKOS_INLINE_FUNCTION
void GetReferencePsiKinematics(const int reference_kind,
                               const DvceArray2D<Real> &table,
                               const Real mass, const Real center_x,
                               const Real center_y, const Real center_z,
                               const Real time, const Real x,
                               const Real y, const Real z,
                               ReferencePsiKinematics &reference) {
  if (reference_kind == 0) {
    MinkowskiReference().PopulatePsiKinematics(time, x, y, z, reference);
    return;
  }
  if (reference_kind == 2) {
    TimeDependentLapseReference().PopulatePsiKinematics(
        time, x, y, z, reference);
    return;
  }
  if (reference_kind == 3) {
    TimeDependentSpatialReference().PopulatePsiKinematics(
        time, x, y, z, reference);
    return;
  }
  const TrumpetSchwarzschildReference provider{
      table, mass, {center_x, center_y, center_z}};
  provider.PopulatePsiKinematics(time, x, y, z, reference);
}

template <typename Reference>
KOKKOS_INLINE_FUNCTION
Real CoframeDerivative(const Reference &reference, const int p,
                       const int A, const int a) {
  Real derivative = 0.0;
  for (int b = 0; b < 4; ++b) {
    for (int B = 0; B < 4; ++B) {
      derivative -= ReferenceCoframe(reference, A, b)
                    *ReferenceDFrame(reference, p, B, b)
                    *ReferenceCoframe(reference, B, a);
    }
  }
  return derivative;
}

template <typename Reference>
KOKKOS_INLINE_FUNCTION
bool LoadPointGeometry(const DvceArray5D<Real> &state,
                       const Reference &reference, const int m,
                       const int k, const int j, const int i,
                       Real psi[4][4], Real pi[4][4], Real phi[3][4][4],
                       Real d_psi[4][4][4], Real metric[4][4],
                       Real d_metric[4][4][4], CoordinateGhGeometry &geometry,
                       Real &determinant) {
  LoadSymmetric(state, kPsiOffset, m, k, j, i, psi);
  LoadSymmetric(state, kPiOffset, m, k, j, i, pi);
  for (int p = 0; p < 3; ++p) {
    for (int a = 0; a < 4; ++a) {
      for (int b = a; b < 4; ++b) {
        phi[p][a][b] = phi[p][b][a] = state(m, PhiIndex(p, a, b), k, j, i);
      }
    }
  }
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      metric[a][b] = 0.0;
      for (int A = 0; A < 4; ++A) {
        for (int B = 0; B < 4; ++B) {
          metric[a][b] += ReferenceCoframe(reference, A, a)
                          *ReferenceCoframe(reference, B, b)*psi[A][B];
        }
      }
    }
  }
  Real inverse[4][4];  // NOLINT(runtime/arrays)
  if (!Invert4(metric, inverse, determinant) || !(inverse[0][0] < 0.0)) return false;
  const Real alpha = 1.0/Kokkos::sqrt(-inverse[0][0]);
  Real beta[3];  // NOLINT(runtime/arrays)
  for (int p = 0; p < 3; ++p) beta[p] = alpha*alpha*inverse[0][p + 1];
  for (int A = 0; A < 4; ++A) {
    for (int B = 0; B < 4; ++B) {
      for (int p = 0; p < 3; ++p) {
        d_psi[p + 1][A][B] = 0.0;
        for (int I = 0; I < 3; ++I) {
          d_psi[p + 1][A][B] +=
              ReferenceSpatialCoframe(reference, I, p)*phi[I][A][B];
        }
      }
      d_psi[0][A][B] = -alpha*pi[A][B];
      for (int p = 0; p < 3; ++p) d_psi[0][A][B] += beta[p]*d_psi[p + 1][A][B];
    }
  }
  // Differentiate Psi_AB=e_A^a e_B^b g_ab and solve algebraically for dg_ab.
  for (int p = 0; p < 4; ++p) {
    Real frame_corrected[4][4];  // NOLINT(runtime/arrays)
    for (int A = 0; A < 4; ++A) {
      for (int B = 0; B < 4; ++B) {
        frame_corrected[A][B] = d_psi[p][A][B];
        for (int a = 0; a < 4; ++a) {
          for (int b = 0; b < 4; ++b) {
            frame_corrected[A][B] -=
                (ReferenceDFrame(reference, p, A, a)
                   *ReferenceFrame(reference, B, b)
                 + ReferenceFrame(reference, A, a)
                   *ReferenceDFrame(reference, p, B, b))*metric[a][b];
          }
        }
      }
    }
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) {
        d_metric[p][a][b] = 0.0;
        for (int A = 0; A < 4; ++A) {
          for (int B = 0; B < 4; ++B) {
            d_metric[p][a][b] += ReferenceCoframe(reference, A, a)
                                  *ReferenceCoframe(reference, B, b)
                                  *frame_corrected[A][B];
          }
        }
      }
    }
  }
  return ComputeCoordinateGhGeometry(metric, d_metric, reference, geometry, determinant);
}

}  // namespace ref_gh

#endif  // REF_GH_REF_GH_GEOMETRY_HPP_
