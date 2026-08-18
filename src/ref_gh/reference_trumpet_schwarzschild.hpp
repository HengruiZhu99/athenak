//========================================================================================
//! \file reference_trumpet_schwarzschild.hpp
//! \brief Device-side stationary n=2 Schwarzschild trumpet reference geometry.
//========================================================================================
#ifndef REF_GH_REFERENCE_TRUMPET_SCHWARZSCHILD_HPP_
#define REF_GH_REFERENCE_TRUMPET_SCHWARZSCHILD_HPP_

#include "athena.hpp"
#include "ref_gh/reference_geometry.hpp"
#include "ref_gh/trumpet_table_generated.hpp"

namespace ref_gh {

enum TrumpetProfileIndex : int {
  kProfileAlpha = 0, kProfileAlphaDy = 1, kProfileAlphaDyy = 2,
  kProfileArealRadius = 3, kProfileArealRadiusDy = 4,
  kProfileArealRadiusDyy = 5, kProfileShiftQ = 6,
  kProfileShiftQDy = 7, kProfileShiftQDyy = 8,
  kTrumpetProfiles = 9
};

struct RadialProfile {
  Real value;
  Real d1;
  Real d2;
};

// Evaluate one quintic Hermite polynomial matching (f,f_y,f_yy) at both
// endpoints.  Returned derivatives are exact derivatives of that same numerical
// polynomial, rather than independent interpolants.
KOKKOS_INLINE_FUNCTION
RadialProfile InterpolateTrumpetProfile(const DvceArray2D<Real> &table,
                                        const int profile, const Real rho) {
  const Real y = Kokkos::log(rho);
  const Real u = (y - kTrumpetLogRMin)/kTrumpetLogRSpacing;
  int index = static_cast<int>(Kokkos::floor(u));
  if (index < 0) index = 0;
  if (index > kTrumpetTableSize - 2) index = kTrumpetTableSize - 2;
  const Real s = u - static_cast<Real>(index);
  const Real h = kTrumpetLogRSpacing;
  const Real a0 = table(profile, index);
  const Real a1 = h*table(profile + 1, index);
  const Real a2 = 0.5*h*h*table(profile + 2, index);
  const Real f = table(profile, index + 1) - (a0 + a1 + a2);
  const Real g = h*table(profile + 1, index + 1) - (a1 + 2.0*a2);
  const Real curvature = h*h*table(profile + 2, index + 1) - 2.0*a2;
  const Real a3 = 10.0*f - 4.0*g + 0.5*curvature;
  const Real a4 = -15.0*f + 7.0*g - curvature;
  const Real a5 = 6.0*f - 3.0*g + 0.5*curvature;
  const Real value = a0 + s*(a1 + s*(a2 + s*(a3 + s*(a4 + s*a5))));
  const Real dy = (a1 + s*(2.0*a2 + s*(3.0*a3
                   + s*(4.0*a4 + s*5.0*a5))))/h;
  const Real dyy = (2.0*a2 + s*(6.0*a3
                    + s*(12.0*a4 + s*20.0*a5)))/(h*h);
  const Real inverse_rho = 1.0/rho;
  return {value, dy*inverse_rho, (dyy - dy)*inverse_rho*inverse_rho};
}

KOKKOS_INLINE_FUNCTION
RadialProfile ArealRadiusToPsi2(const RadialProfile &areal, const Real rho) {
  const Real inverse_rho = 1.0/rho;
  return {areal.value*inverse_rho,
          areal.d1*inverse_rho - areal.value*inverse_rho*inverse_rho,
          areal.d2*inverse_rho - 2.0*areal.d1*inverse_rho*inverse_rho
              + 2.0*areal.value*inverse_rho*inverse_rho*inverse_rho};
}

// A value with coordinate first and second partial derivatives.  This small local
// second-order jet keeps all reference derivatives analytic and internally consistent.
struct ReferenceJet {
  Real value;
  Real d[4];      // NOLINT(runtime/arrays)
  Real dd[4][4];  // NOLINT(runtime/arrays)
};

KOKKOS_INLINE_FUNCTION
ReferenceJet ConstantJet(const Real value) {
  ReferenceJet result;
  result.value = value;
  for (int a = 0; a < 4; ++a) {
    result.d[a] = 0.0;
    for (int b = 0; b < 4; ++b) result.dd[a][b] = 0.0;
  }
  return result;
}

KOKKOS_INLINE_FUNCTION
ReferenceJet CoordinateJet(const Real value, const int direction) {
  ReferenceJet result = ConstantJet(value);
  result.d[direction] = 1.0;
  return result;
}

KOKKOS_INLINE_FUNCTION
ReferenceJet operator+(const ReferenceJet &left, const ReferenceJet &right) {
  ReferenceJet result;
  result.value = left.value + right.value;
  for (int a = 0; a < 4; ++a) {
    result.d[a] = left.d[a] + right.d[a];
    for (int b = 0; b < 4; ++b) result.dd[a][b] = left.dd[a][b] + right.dd[a][b];
  }
  return result;
}

KOKKOS_INLINE_FUNCTION
ReferenceJet operator-(const ReferenceJet &value) {
  ReferenceJet result;
  result.value = -value.value;
  for (int a = 0; a < 4; ++a) {
    result.d[a] = -value.d[a];
    for (int b = 0; b < 4; ++b) result.dd[a][b] = -value.dd[a][b];
  }
  return result;
}

KOKKOS_INLINE_FUNCTION
ReferenceJet operator*(const ReferenceJet &left, const ReferenceJet &right) {
  ReferenceJet result;
  result.value = left.value*right.value;
  for (int a = 0; a < 4; ++a) {
    result.d[a] = left.d[a]*right.value + left.value*right.d[a];
    for (int b = 0; b < 4; ++b) {
      result.dd[a][b] = left.dd[a][b]*right.value + left.value*right.dd[a][b]
                        + left.d[a]*right.d[b] + left.d[b]*right.d[a];
    }
  }
  return result;
}

KOKKOS_INLINE_FUNCTION
ReferenceJet Reciprocal(const ReferenceJet &input) {
  ReferenceJet result;
  const Real inverse = 1.0/input.value;
  result.value = inverse;
  for (int a = 0; a < 4; ++a) {
    result.d[a] = -input.d[a]*inverse*inverse;
    for (int b = 0; b < 4; ++b) {
      result.dd[a][b] = 2.0*input.d[a]*input.d[b]*inverse*inverse*inverse
                        - input.dd[a][b]*inverse*inverse;
    }
  }
  return result;
}

KOKKOS_INLINE_FUNCTION
ReferenceJet RadialJet(const RadialProfile &profile, const Real mass,
                       const Real displacement[3], const Real radius) {
  ReferenceJet result = ConstantJet(profile.value);
  const Real inverse_mass = 1.0/mass;
  const Real d1 = profile.d1*inverse_mass;
  const Real d2 = profile.d2*inverse_mass*inverse_mass;
  for (int i = 0; i < 3; ++i) {
    const Real ni = displacement[i]/radius;
    result.d[i + 1] = d1*ni;
    for (int j = 0; j < 3; ++j) {
      const Real nj = displacement[j]/radius;
      result.dd[i + 1][j + 1] = d2*ni*nj
          + d1*(((i == j) ? 1.0 : 0.0) - ni*nj)/radius;
    }
  }
  return result;
}

KOKKOS_INLINE_FUNCTION
void StoreJet(const ReferenceJet &jet, Real &value, Real derivative[4],
              Real second_derivative[4][4]) {
  value = jet.value;
  for (int a = 0; a < 4; ++a) {
    derivative[a] = jet.d[a];
    for (int b = 0; b < 4; ++b) second_derivative[a][b] = jet.dd[a][b];
  }
}

struct TrumpetSchwarzschildReference {
  DvceArray2D<Real> table;
  Real mass;
  Real center[3];  // NOLINT(runtime/arrays)

  KOKKOS_INLINE_FUNCTION
  ReferenceGeometry operator()(const Real /*time*/, const Real x, const Real y,
                               const Real z) const {
    ReferenceGeometry reference;
    ZeroReferenceGeometry(reference);
    const Real displacement[3] = {x - center[0], y - center[1], z - center[2]};
    const Real radius = Kokkos::sqrt(displacement[0]*displacement[0]
                                     + displacement[1]*displacement[1]
                                     + displacement[2]*displacement[2]);
    const Real rho = radius/mass;
    const ReferenceJet alpha = RadialJet(
        InterpolateTrumpetProfile(table, kProfileAlpha, rho), mass,
        displacement, radius);
    const RadialProfile areal =
        InterpolateTrumpetProfile(table, kProfileArealRadius, rho);
    const ReferenceJet psi2 = RadialJet(
        ArealRadiusToPsi2(areal, rho), mass,
        displacement, radius);
    RadialProfile q_profile = InterpolateTrumpetProfile(
        table, kProfileShiftQ, rho);
    q_profile.value /= mass;
    q_profile.d1 /= mass;
    q_profile.d2 /= mass;
    const ReferenceJet shift_q = RadialJet(q_profile, mass, displacement, radius);
    const ReferenceJet inverse_alpha = Reciprocal(alpha);
    const ReferenceJet inverse_psi2 = Reciprocal(psi2);
    ReferenceJet coordinates[3] = {CoordinateJet(displacement[0], 1),
                                   CoordinateJet(displacement[1], 2),
                                   CoordinateJet(displacement[2], 3)};
    ReferenceJet shift[3];  // NOLINT(runtime/arrays)
    for (int i = 0; i < 3; ++i) shift[i] = shift_q*coordinates[i];

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
    for (int i = 0; i < 3; ++i) {
      coframe[i + 1][0] = psi2*shift[i];
      coframe[i + 1][i + 1] = psi2;
      frame[0][i + 1] = -(shift[i]*inverse_alpha);
      frame[i + 1][i + 1] = inverse_psi2;
    }

    ReferenceJet metric[4][4];          // NOLINT(runtime/arrays)
    ReferenceJet inverse_metric[4][4];  // NOLINT(runtime/arrays)
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) {
        metric[a][b] = -(coframe[0][a]*coframe[0][b]);
        inverse_metric[a][b] = -(frame[0][a]*frame[0][b]);
        for (int I = 1; I < 4; ++I) {
          metric[a][b] = metric[a][b] + coframe[I][a]*coframe[I][b];
          inverse_metric[a][b] = inverse_metric[a][b] + frame[I][a]*frame[I][b];
        }
        reference.metric[a][b] = metric[a][b].value;
        reference.inverse_metric[a][b] = inverse_metric[a][b].value;
        reference.coframe[a][b] = coframe[a][b].value;
        reference.frame[a][b] = frame[a][b].value;
        for (int c = 0; c < 4; ++c) {
          reference.d_metric[c][a][b] = metric[a][b].d[c];
          reference.d_frame[c][a][b] = frame[a][b].d[c];
          for (int d = 0; d < 4; ++d) {
            reference.dd_metric[c][d][a][b] = metric[a][b].dd[c][d];
            reference.dd_frame[c][d][a][b] = frame[a][b].dd[c][d];
          }
        }
      }
    }

    for (int I = 0; I < 3; ++I) {
      reference.spatial_frame[I][I] = inverse_psi2.value;
      reference.spatial_coframe[I][I] = psi2.value;
      for (int J = 0; J < 3; ++J) {
        for (int K = 0; K < 3; ++K) {
          reference.structure[I][J][K] =
              ((J == K) ? inverse_psi2.d[I + 1] : 0.0)
              - ((I == K) ? inverse_psi2.d[J + 1] : 0.0);
        }
      }
    }

    Real first_kind[4][4][4];  // NOLINT(runtime/arrays)
    Real d_inverse[4][4][4];   // NOLINT(runtime/arrays)
    for (int p = 0; p < 4; ++p) {
      for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
          d_inverse[p][a][b] = inverse_metric[a][b].d[p];
          for (int c = 0; c < 4; ++c) {
            first_kind[a][b][c] = 0.5*(reference.d_metric[b][a][c]
                                        + reference.d_metric[c][a][b]
                                        - reference.d_metric[a][b][c]);
          }
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
                  d_inverse[p][a][ell]*first_kind[ell][b][c]
                  + reference.inverse_metric[a][ell]*d_first;
            }
          }
        }
      }
    }
    return reference;
  }
};

}  // namespace ref_gh

#endif  // REF_GH_REFERENCE_TRUMPET_SCHWARZSCHILD_HPP_
