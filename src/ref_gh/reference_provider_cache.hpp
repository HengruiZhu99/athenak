//========================================================================================
//! \file reference_provider_cache.hpp
//! \brief Small provider/profile jets used by the staged Ref-GH cache update.
//========================================================================================
#ifndef REF_GH_REFERENCE_PROVIDER_CACHE_HPP_
#define REF_GH_REFERENCE_PROVIDER_CACHE_HPP_

#include "athena.hpp"
#include "ref_gh/reference_cache.hpp"
#include "ref_gh/reference_time_dependent_lapse.hpp"
#include "ref_gh/reference_trumpet_schwarzschild.hpp"

namespace ref_gh {

struct ReferenceProviderMetadata {
  bool time_dependent;
};

KOKKOS_INLINE_FUNCTION
constexpr ReferenceProviderMetadata GetReferenceProviderMetadata(
    const int reference_kind) {
  return {reference_kind == 2};
}

KOKKOS_INLINE_FUNCTION
void StoreProviderJet(const ReferenceJet &jet, const int offset,
                      const ReferenceProviderPoint &point) {
  point.provider(point.m, offset, point.k, point.j, point.i) = jet.value;
  for (int p = 0; p < 4; ++p) {
    point.provider(point.m, RefJetDerivative(offset, p),
                   point.k, point.j, point.i) = jet.d[p];
    for (int q = 0; q < 4; ++q) {
      point.provider(point.m, RefJetSecondDerivative(offset, p, q),
                     point.k, point.j, point.i) = jet.dd[p][q];
    }
  }
}

KOKKOS_INLINE_FUNCTION
void StoreWorkspaceMetricJet(const ReferenceJet &jet, const int a, const int b,
                             const ReferenceWorkspacePoint &point) {
  const int offset = kRefWorkspaceMetricJet + 21*(4*a + b);
  point.workspace(point.m, offset, point.k, point.j, point.i) = jet.value;
  for (int p = 0; p < 4; ++p) {
    point.workspace(point.m, RefJetDerivative(offset, p),
                    point.k, point.j, point.i) = jet.d[p];
    for (int q = 0; q < 4; ++q) {
      point.workspace(point.m, RefJetSecondDerivative(offset, p, q),
                      point.k, point.j, point.i) = jet.dd[p][q];
    }
  }
}

KOKKOS_INLINE_FUNCTION
ReferenceJet LoadWorkspaceMetricJet(const ReferenceWorkspacePoint &point,
  const int a, const int b) {
  const int offset = kRefWorkspaceMetricJet + 21*(4*a + b);
  ReferenceJet jet;
  jet.value = point.workspace(point.m, offset, point.k, point.j, point.i);
  for (int p = 0; p < 4; ++p) {
    jet.d[p] = point.workspace(point.m, RefJetDerivative(offset, p),
                              point.k, point.j, point.i);
    for (int q = 0; q < 4; ++q) {
      jet.dd[p][q] = point.workspace(
          point.m, RefJetSecondDerivative(offset, p, q),
          point.k, point.j, point.i);
    }
  }
  return jet;
}

KOKKOS_INLINE_FUNCTION
void StoreWorkspaceInverseMetricJet(const ReferenceJet &jet,
                                    const int a, const int b,
                                    const ReferenceWorkspacePoint &point) {
  const int offset =
      kRefWorkspaceInverseMetricJet + 5*RefSymmetricPair4(a, b);
  point.workspace(point.m, offset, point.k, point.j, point.i) = jet.value;
  for (int p = 0; p < 4; ++p) {
    point.workspace(point.m, offset + 1 + p, point.k, point.j, point.i) = jet.d[p];
  }
}

KOKKOS_INLINE_FUNCTION
Real WorkspaceInverseMetric(const ReferenceWorkspacePoint &point,
                            const int a, const int b) {
  return point.workspace(
      point.m, kRefWorkspaceInverseMetricJet + 5*RefSymmetricPair4(a, b),
      point.k, point.j, point.i);
}

KOKKOS_INLINE_FUNCTION
Real WorkspaceDInverseMetric(const ReferenceWorkspacePoint &point,
                             const int p, const int a, const int b) {
  return point.workspace(
      point.m, kRefWorkspaceInverseMetricJet
                 + 5*RefSymmetricPair4(a, b) + 1 + p,
      point.k, point.j, point.i);
}

KOKKOS_INLINE_FUNCTION
ReferenceJet LoadProviderJet(const ReferenceProviderPoint &point,
                             const int offset) {
  ReferenceJet jet;
  jet.value = point.provider(point.m, offset, point.k, point.j, point.i);
  for (int p = 0; p < 4; ++p) {
    jet.d[p] = point.provider(point.m, RefJetDerivative(offset, p),
                             point.k, point.j, point.i);
    for (int q = 0; q < 4; ++q) {
      jet.dd[p][q] = point.provider(
          point.m, RefJetSecondDerivative(offset, p, q),
          point.k, point.j, point.i);
    }
  }
  return jet;
}

// This is the only stage that evaluates the prescribed radial profiles. New
// time-dependent providers extend this dispatch and store their time-dependent
// scalar jets here; downstream tensor stages remain provider-agnostic.
KOKKOS_INLINE_FUNCTION
void PopulateReferenceProviderCache(
    const int reference_kind, const DvceArray2D<Real> &table, const Real mass,
    const Real center_x, const Real center_y, const Real center_z,
    const Real time, const Real x, const Real y, const Real z,
    const ReferenceProviderPoint &point) {
  if (reference_kind == 0) {
    for (int component = 0; component < kReferenceProviderSize; ++component) {
      point.provider(point.m, component, point.k, point.j, point.i) = 0.0;
    }
    return;
  }
  if (reference_kind == 2) {
    const ReferenceJet alpha = TimeDependentLapseReference().AlphaJet(time);
    StoreProviderJet(alpha, kRefProviderAlpha, point);
    StoreProviderJet(ConstantJet(1.0), kRefProviderPsi2, point);
    StoreProviderJet(ConstantJet(0.0), kRefProviderShiftQ, point);
    point.provider(point.m, kRefProviderArealRadius,
                   point.k, point.j, point.i) = 0.0;
    return;
  }

  const Real displacement[3] = {x - center_x, y - center_y, z - center_z};
  const Real radius = Kokkos::sqrt(displacement[0]*displacement[0]
                                   + displacement[1]*displacement[1]
                                   + displacement[2]*displacement[2]);
  const Real rho = radius/mass;
  const ReferenceJet alpha = RadialJet(
      InterpolateTrumpetProfile(table, kCoeffAlpha, rho), mass,
      displacement, radius);
  const RadialProfile areal =
      InterpolateTrumpetProfile(table, kCoeffArealRadius, rho);
  const ReferenceJet psi2 = RadialJet(
      ArealRadiusToPsi2(areal, rho), mass, displacement, radius);
  RadialProfile q_profile =
      InterpolateTrumpetProfile(table, kCoeffShiftQ, rho);
  q_profile.value /= mass;
  q_profile.d1 /= mass;
  q_profile.d2 /= mass;
  const ReferenceJet shift_q =
      RadialJet(q_profile, mass, displacement, radius);
  StoreProviderJet(alpha, kRefProviderAlpha, point);
  StoreProviderJet(psi2, kRefProviderPsi2, point);
  StoreProviderJet(shift_q, kRefProviderShiftQ, point);
  point.provider(point.m, kRefProviderArealRadius,
                 point.k, point.j, point.i) = areal.value;
}

KOKKOS_INLINE_FUNCTION
ReferenceJet ProviderCoordinateJet(const Real x, const Real y, const Real z,
                                   const Real center_x, const Real center_y,
                                   const Real center_z, const int direction) {
  const Real displacement[3] = {x - center_x, y - center_y, z - center_z};
  return CoordinateJet(displacement[direction], direction + 1);
}

KOKKOS_INLINE_FUNCTION
ReferenceJet ProviderCoframeJet(
    const int reference_kind, const ReferenceProviderPoint &point,
    const Real x, const Real y, const Real z, const Real center_x,
    const Real center_y, const Real center_z, const int A, const int a) {
  if (reference_kind == 0) return ConstantJet((A == a) ? 1.0 : 0.0);
  const ReferenceJet alpha = LoadProviderJet(point, kRefProviderAlpha);
  const ReferenceJet psi2 = LoadProviderJet(point, kRefProviderPsi2);
  const ReferenceJet shift_q = LoadProviderJet(point, kRefProviderShiftQ);
  if (A == 0 && a == 0) return alpha;
  if (A > 0 && a == 0) {
    return psi2*(shift_q*ProviderCoordinateJet(
        x, y, z, center_x, center_y, center_z, A - 1));
  }
  if (A > 0 && a == A) return psi2;
  return ConstantJet(0.0);
}

KOKKOS_INLINE_FUNCTION
ReferenceJet ProviderFrameJet(
    const int reference_kind, const ReferenceProviderPoint &point,
    const Real x, const Real y, const Real z, const Real center_x,
    const Real center_y, const Real center_z, const int A, const int a) {
  if (reference_kind == 0) return ConstantJet((A == a) ? 1.0 : 0.0);
  const ReferenceJet inverse_alpha =
      Reciprocal(LoadProviderJet(point, kRefProviderAlpha));
  const ReferenceJet inverse_psi2 =
      Reciprocal(LoadProviderJet(point, kRefProviderPsi2));
  const ReferenceJet shift_q = LoadProviderJet(point, kRefProviderShiftQ);
  if (A == 0 && a == 0) return inverse_alpha;
  if (A == 0 && a > 0) {
    return -(shift_q*ProviderCoordinateJet(
        x, y, z, center_x, center_y, center_z, a - 1)*inverse_alpha);
  }
  if (A > 0 && a == A) return inverse_psi2;
  return ConstantJet(0.0);
}

KOKKOS_INLINE_FUNCTION
ReferenceJet ProviderMetricJet(
    const int reference_kind, const ReferenceProviderPoint &point,
    const Real x, const Real y, const Real z, const Real center_x,
    const Real center_y, const Real center_z, const int a, const int b) {
  ReferenceJet metric = -(
      ProviderCoframeJet(reference_kind, point, x, y, z,
                         center_x, center_y, center_z, 0, a)
      *ProviderCoframeJet(reference_kind, point, x, y, z,
                          center_x, center_y, center_z, 0, b));
  for (int I = 1; I < 4; ++I) {
    metric = metric
        + ProviderCoframeJet(reference_kind, point, x, y, z,
                             center_x, center_y, center_z, I, a)
          * ProviderCoframeJet(reference_kind, point, x, y, z,
                               center_x, center_y, center_z, I, b);
  }
  return metric;
}

KOKKOS_INLINE_FUNCTION
ReferenceJet ProviderInverseMetricJet(
    const int reference_kind, const ReferenceProviderPoint &point,
    const Real x, const Real y, const Real z, const Real center_x,
    const Real center_y, const Real center_z, const int a, const int b) {
  ReferenceJet inverse = -(
      ProviderFrameJet(reference_kind, point, x, y, z,
                       center_x, center_y, center_z, 0, a)
      *ProviderFrameJet(reference_kind, point, x, y, z,
                        center_x, center_y, center_z, 0, b));
  for (int I = 1; I < 4; ++I) {
    inverse = inverse
        + ProviderFrameJet(reference_kind, point, x, y, z,
                           center_x, center_y, center_z, I, a)
          * ProviderFrameJet(reference_kind, point, x, y, z,
                             center_x, center_y, center_z, I, b);
  }
  return inverse;
}

KOKKOS_INLINE_FUNCTION
Real ProviderSpatialFrame(const int reference_kind,
                          const ReferenceProviderPoint &point,
                          const int I, const int i) {
  if (I != i) return 0.0;
  if (reference_kind == 0) return 1.0;
  return Reciprocal(LoadProviderJet(point, kRefProviderPsi2)).value;
}

KOKKOS_INLINE_FUNCTION
Real ProviderSpatialCoframe(const int reference_kind,
                            const ReferenceProviderPoint &point,
                            const int I, const int i) {
  if (I != i) return 0.0;
  if (reference_kind == 0) return 1.0;
  return LoadProviderJet(point, kRefProviderPsi2).value;
}

KOKKOS_INLINE_FUNCTION
Real ProviderDtSpatialFrame(const int reference_kind,
                            const ReferenceProviderPoint &point,
                            const int I, const int i) {
  if (I != i || reference_kind == 0) return 0.0;
  return Reciprocal(LoadProviderJet(point, kRefProviderPsi2)).d[0];
}

KOKKOS_INLINE_FUNCTION
Real ProviderStructure(const int reference_kind,
                       const ReferenceProviderPoint &point,
                       const int I, const int J, const int K) {
  if (reference_kind != 1) return 0.0;
  const ReferenceJet inverse_psi2 =
      Reciprocal(LoadProviderJet(point, kRefProviderPsi2));
  return ((J == K) ? inverse_psi2.d[I + 1] : 0.0)
         - ((I == K) ? inverse_psi2.d[J + 1] : 0.0);
}

KOKKOS_INLINE_FUNCTION
Real ProviderElectricWeyl(const ReferenceProviderPoint &point,
                          const Real mass, const Real displacement[3],
                          const Real radius, const int I, const int J) {
  const Real areal = point.provider(
      point.m, kRefProviderArealRadius, point.k, point.j, point.i);
  const Real inverse_areal = 1.0/areal;
  const Real scale = inverse_areal*inverse_areal*inverse_areal/(mass*mass);
  return scale*((I == J) ? 1.0 : 0.0)
         - 3.0*scale*displacement[I]*displacement[J]/(radius*radius);
}

KOKKOS_INLINE_FUNCTION
Real ProviderRiemann(const int reference_kind,
                     const ReferenceProviderPoint &point,
                     const Real mass, const Real x, const Real y, const Real z,
                     const Real center_x, const Real center_y,
                     const Real center_z, const int A, const int B,
                     const int C, const int D) {
  if (reference_kind != 1) return 0.0;
  const Real displacement[3] = {x - center_x, y - center_y, z - center_z};
  const Real radius = Kokkos::sqrt(displacement[0]*displacement[0]
                                   + displacement[1]*displacement[1]
                                   + displacement[2]*displacement[2]);
  Real lower = 0.0;
  if (A == 0 && B > 0 && C == 0 && D > 0) {
    lower = ProviderElectricWeyl(point, mass, displacement, radius,
                                 B - 1, D - 1);
  } else if (A > 0 && B == 0 && C == 0 && D > 0) {
    lower = -ProviderElectricWeyl(point, mass, displacement, radius,
                                  A - 1, D - 1);
  } else if (A == 0 && B > 0 && C > 0 && D == 0) {
    lower = -ProviderElectricWeyl(point, mass, displacement, radius,
                                  B - 1, C - 1);
  } else if (A > 0 && B == 0 && C > 0 && D == 0) {
    lower = ProviderElectricWeyl(point, mass, displacement, radius,
                                 A - 1, C - 1);
  } else if (A > 0 && B > 0 && C > 0 && D > 0) {
    const int I = A - 1;
    const int J = B - 1;
    const int K = C - 1;
    const int L = D - 1;
    lower = ((I == K) ? ProviderElectricWeyl(
                 point, mass, displacement, radius, J, L) : 0.0)
            + ((J == L) ? ProviderElectricWeyl(
                 point, mass, displacement, radius, I, K) : 0.0)
            - ((I == L) ? ProviderElectricWeyl(
                 point, mass, displacement, radius, J, K) : 0.0)
            - ((J == K) ? ProviderElectricWeyl(
                 point, mass, displacement, radius, I, L) : 0.0);
  }
  return ((A == 0) ? -1.0 : 1.0)*lower;
}

}  // namespace ref_gh

#endif  // REF_GH_REFERENCE_PROVIDER_CACHE_HPP_
