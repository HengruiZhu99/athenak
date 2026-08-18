//========================================================================================
//! \file ref_gh_geometry.hpp
//! \brief Cell-local reconstruction of coordinate geometry from regular GH fields.
//========================================================================================
#ifndef REF_GH_REF_GH_GEOMETRY_HPP_
#define REF_GH_REF_GH_GEOMETRY_HPP_

#include "athena.hpp"
#include "ref_gh/ref_gh_state.hpp"
#include "ref_gh/reference_geometry.hpp"
#include "ref_gh/reference_trumpet_schwarzschild.hpp"
#include "ref_gh/standard_gh_source.hpp"

namespace ref_gh {

KOKKOS_INLINE_FUNCTION
void SymmetricEigenvalues3(Real matrix[3][3], Real eigenvalues[3]) {
  for (int sweep = 0; sweep < 18; ++sweep) {
    int p = 0;
    int q = 1;
    Real largest = Kokkos::abs(matrix[0][1]);
    for (int a = 0; a < 3; ++a) {
      for (int b = a + 1; b < 3; ++b) {
        if (Kokkos::abs(matrix[a][b]) > largest) {
          largest = Kokkos::abs(matrix[a][b]);
          p = a;
          q = b;
        }
      }
    }
    if (largest < 1.0e-14) break;
    const Real angle = 0.5*Kokkos::atan2(2.0*matrix[p][q],
                                         matrix[q][q] - matrix[p][p]);
    const Real cosine = Kokkos::cos(angle);
    const Real sine = Kokkos::sin(angle);
    const Real app = matrix[p][p];
    const Real aqq = matrix[q][q];
    const Real apq = matrix[p][q];
    matrix[p][p] = cosine*cosine*app - 2.0*sine*cosine*apq + sine*sine*aqq;
    matrix[q][q] = sine*sine*app + 2.0*sine*cosine*apq + cosine*cosine*aqq;
    matrix[p][q] = matrix[q][p] = 0.0;
    for (int r = 0; r < 3; ++r) {
      if (r == p || r == q) continue;
      const Real arp = matrix[r][p];
      const Real arq = matrix[r][q];
      matrix[r][p] = matrix[p][r] = cosine*arp - sine*arq;
      matrix[r][q] = matrix[q][r] = sine*arp + cosine*arq;
    }
  }
  for (int a = 0; a < 3; ++a) eigenvalues[a] = matrix[a][a];
  for (int a = 0; a < 2; ++a) {
    for (int b = a + 1; b < 3; ++b) {
      if (eigenvalues[b] < eigenvalues[a]) {
        const Real temporary = eigenvalues[a];
        eigenvalues[a] = eigenvalues[b];
        eigenvalues[b] = temporary;
      }
    }
  }
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
  TrumpetSchwarzschildReference provider{table, mass,
                                         {center_x, center_y, center_z}};
  return provider(time, x, y, z);
}

KOKKOS_INLINE_FUNCTION
Real CoframeDerivative(const ReferenceGeometry &reference, const int p,
                       const int A, const int a) {
  Real derivative = 0.0;
  for (int b = 0; b < 4; ++b) {
    for (int B = 0; B < 4; ++B) {
      derivative -= reference.coframe[A][b]*reference.d_frame[p][B][b]
                    *reference.coframe[B][a];
    }
  }
  return derivative;
}

KOKKOS_INLINE_FUNCTION
bool LoadPointGeometry(const DvceArray5D<Real> &state,
                       const ReferenceGeometry &reference, const int m,
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
          metric[a][b] += reference.coframe[A][a]*reference.coframe[B][b]
                          *psi[A][B];
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
          d_psi[p + 1][A][B] += reference.spatial_coframe[I][p]*phi[I][A][B];
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
                (reference.d_frame[p][A][a]*reference.frame[B][b]
                 + reference.frame[A][a]*reference.d_frame[p][B][b])*metric[a][b];
          }
        }
      }
    }
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) {
        d_metric[p][a][b] = 0.0;
        for (int A = 0; A < 4; ++A) {
          for (int B = 0; B < 4; ++B) {
            d_metric[p][a][b] += reference.coframe[A][a]
                                  *reference.coframe[B][b]*frame_corrected[A][B];
          }
        }
      }
    }
  }
  return ComputeCoordinateGhGeometry(metric, d_metric, reference, geometry, determinant);
}

}  // namespace ref_gh

#endif  // REF_GH_REF_GH_GEOMETRY_HPP_
