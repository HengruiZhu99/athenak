// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_POINT_STATE_HPP_
#define Z4C_HYPERBOLOIDAL_POINT_STATE_HPP_

#include "z4c_hyperboloidal/fields.hpp"
#include "z4c_hyperboloidal/spherical_policy.hpp"

namespace z4c_hyperboloidal {
// Non-owning native pack readers. Their indices are always (block,field,k,j,i).
// Ghost exchange/refinement is complete before these readers enter a kernel.
struct BlockPointReader {
  DvceArray5D<Real> state;
  int block, k, j, i;
  KOKKOS_INLINE_FUNCTION Real operator()(int f,int di=0,int dj=0,int dk=0) const {
    return state(block,f,k+dk,j+dj,i+di);
  }
};
struct BlockResidualReader {
  DvceArray5D<Real> state, reference;
  DvceArray4D<mapped::Point> geometry;
  int block, k, j, i;
  ResidualWeight weight;
  KOKKOS_INLINE_FUNCTION Real operator()(int f,int di=0,int dj=0,int dk=0) const {
    if (!Fields::Independent(f)) return 0;
    const Real residual=state(block,f,k+dk,j+dj,i+di)
                        -reference(block,f,k+dk,j+dj,i+di);
    if (f!=Fields::chi) return residual;
    const auto &x=geometry(block,k+dk,j+dj,i+di).x;
    return residual/weight.Value(x);
  }
};

// Directional derivatives transform; Cartesian tensor components do not rotate.
KOKKOS_INLINE_FUNCTION mapped::Jet CartesianJet(const mapped::Jet &q,
                                                const mapped::Point &map,
                                                bool need_second=true) {
  mapped::Jet out(q.f);
  for (int a=0;a<3;++a) for (int A=0;A<3;++A) {
    out.d[a]+=map.J[A][a]*q.d[A];
    if (need_second)for (int b=0;b<3;++b) {
      out.h[a][b]+=map.H[A][a][b]*q.d[A];
      for (int B=0;B<3;++B) out.h[a][b]+=map.J[A][a]*map.J[B][b]*q.h[A][B];
    }
  }
  return out;
}

template<class Reader>
KOKKOS_INLINE_FUNCTION Real OffsetValue(const Reader &q,int f,int a,int offset,
                                        int b=0,int offset_b=0) {
  int d[3]{};d[a]+=offset;d[b]+=offset_b;
  return q(f,d[0],d[1],d[2]);
}

// Keep axial samples for the centered and biased derivatives.
struct AxialSamples {
  Real minus_two, minus_one, plus_one, plus_two;

  template <class Reader>
  KOKKOS_INLINE_FUNCTION AxialSamples(const Reader &q, int f, int axis)
      : minus_two(OffsetValue(q, f, axis, -2)),
        minus_one(OffsetValue(q, f, axis, -1)),
        plus_one(OffsetValue(q, f, axis, 1)),
        plus_two(OffsetValue(q, f, axis, 2)) {}

  KOKKOS_INLINE_FUNCTION Real First(Real inv_h) const {
    return inv_h * (minus_two - 8 * minus_one + 8 * plus_one - plus_two) / 12;
  }

  KOKKOS_INLINE_FUNCTION Real Second(Real center, Real inv_h) const {
    return inv_h * inv_h * (-minus_two + 16 * minus_one - 30 * center +
                            16 * plus_one - plus_two) / 12;
  }

  KOKKOS_INLINE_FUNCTION Real Biased(Real center, Real far, Real inv_h, int sign) const {
    const Real vm = (sign > 0 ? minus_one : plus_one) - center;
    const Real v1 = (sign > 0 ? plus_one : minus_one) - center;
    const Real v2 = (sign > 0 ? plus_two : minus_two) - center;
    const Real v3 = far - center;
    return sign * inv_h * (-3 * vm + 18 * v1 - 6 * v2 + v3) / 12;
  }
};

template<class Reader>
KOKKOS_INLINE_FUNCTION Real FirstDerivative(const Reader &q,int f,int a,Real inv_h,
                                            int shift=0) {
  return inv_h*(OffsetValue(q,f,a,shift-2)-8*OffsetValue(q,f,a,shift-1)
       +8*OffsetValue(q,f,a,shift+1)-OffsetValue(q,f,a,shift+2))/12;
}
template<class Reader>
KOKKOS_INLINE_FUNCTION Real SecondDerivative(const Reader &q,int f,int a,int b,
                                             const Real inv_h[3]) {
  if (a==b) return inv_h[a]*inv_h[a]*(-OffsetValue(q,f,a,-2)
      +16*OffsetValue(q,f,a,-1)-30*q(f)+16*OffsetValue(q,f,a,1)
      -OffsetValue(q,f,a,2))/12;
  Real value=0;
  for (int i=-2;i<=2;++i) if (i) for (int j=-2;j<=2;++j) if (j) {
    const Real wi=(i==1||i==-1)?Real(2*i)/3:Real(-i)/24;
    const Real wj=(j==1||j==-1)?Real(2*j)/3:Real(-j)/24;
    value+=wi*wj*OffsetValue(q,f,a,i,b,j);
  }
  return value*inv_h[a]*inv_h[b];
}

struct LogicalShift {
  Real value[3]{};

  KOKKOS_INLINE_FUNCTION static LogicalShift Project(const mapped::Point &map,
                                                     const Real beta[3], bool affine) {
    LogicalShift shift;
    for (int a = 0; a < 3; ++a) {
      if (affine) {
        shift.value[a] = map.J[a][a] * beta[a];
      } else {
        for (int i = 0; i < 3; ++i) shift.value[a] += map.J[a][i] * beta[i];
      }
    }
    return shift;
  }
};

// Fourth-order centered jets plus the existing one-sided transport correction.
// A radial endpoint reconstruction can supply an alternative logical jet to
// CartesianJet without changing the point equations below.
template <class Reader>
KOKKOS_INLINE_FUNCTION mapped::Jet SampleResidualJet(
    const Reader &q, int f, const mapped::Point &map, const Real inv_h[3],
    const LogicalShift &shift, bool need_second, Real &transport, bool affine = false) {
  mapped::Jet logical(q(f));
  transport = 0;
  for (int a = 0; a < 3; ++a) {
    const AxialSamples samples(q, f, a);
    logical.d[a] = samples.First(inv_h[a]);
    if (need_second) {
      logical.h[a][a] = samples.Second(logical.f, inv_h[a]);
      for (int b = 0; b < a; ++b)
        logical.h[b][a] = logical.h[a][b] = SecondDerivative(q, f, a, b, inv_h);
    }
    const Real b = shift.value[a];
    const int sign = b >= 0 ? 1 : -1;
    const Real far = OffsetValue(q, f, a, 3 * sign);
    const Real biased = samples.Biased(logical.f, far, inv_h[a], sign);
    transport += b * (biased - logical.d[a]);
  }
  if (affine) {
    // For an affine Cartesian leaf H=0 and J is constant and diagonal.
    for (int a = 0; a < 3; ++a) {
      logical.d[a] *= map.J[a][a];
      if (need_second) {
        for (int b = 0; b < 3; ++b) logical.h[a][b] *= map.J[a][a] * map.J[b][b];
      }
    }
    return logical;
  }
  return CartesianJet(logical,map,need_second);
}

template <class Reader>
KOKKOS_INLINE_FUNCTION mapped::Jet SampleResidualJet(
    const Reader &q, int f, const mapped::Point &map, const Real inv_h[3],
    const Real beta[3], bool need_second, Real &transport, bool affine = false) {
  return SampleResidualJet(q, f, map, inv_h, LogicalShift::Project(map, beta, affine),
                           need_second, transport, affine);
}

struct PointState {
  CartesianChartJets jets{};
  Real transport[Fields::count]{};
  PointStatus status=PointStatus::success;
};

template <class Reader>
KOKKOS_INLINE_FUNCTION void ApplyCompatibleLapse(
    mapped::Jet &lapse, const Reader &q, const mapped::Point &map, const Real inv_h[3],
    Real r, const HyperbolicShiftParameters<Real> &parameters) {
  const Real w = SphericalPolicy::CompatibleWeight(r, parameters);
  if (w > 0) {
    Real composed = 0;
    for (int k = -2; k <= 2; ++k)
      if (k) {
        const Real c = (k == 1 || k == -1) ? 2. * k / 3 : -k / 24.;
        composed += inv_h[0] * c * FirstDerivative(q, 18, 0, inv_h[0], k);
      }
    Real inverse = 0;
    for (int a = 0; a < 3; ++a) inverse += map.J[0][a] * map.x[a] / r;
    const Real delta =
        w * inverse * inverse * (composed - SecondDerivative(q, 18, 0, 0, inv_h));
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) lapse.h[a][b] += delta * map.x[a] * map.x[b] / (r * r);
  }
}

// Finish either centered or endpoint-reconstructed residual jets with the same
// compatible lapse Hessian, reference restoration and algebraic constraints.
template <class Reader>
KOKKOS_INLINE_FUNCTION void RestorePointState(
    PointState &out, const Reader &q, const mapped::Point &map, const Real inv_h[3],
    const CartesianChartJets &reference,
    const HyperbolicShiftParameters<Real> &parameters, const mapped::Jet *lift = nullptr,
    const mapped::Jet *cached_weight = nullptr) {
  const Real r = SphericalPolicy::Radius(map.x);
  if (!(r >= 0) || (!parameters.residual_weight.Regular() && r == 0) ||
      !Kokkos::isfinite(r)) {
    out.status = PointStatus::invalid_input;
    return;
  }
  ApplyCompatibleLapse(out.jets[18], q, map, inv_h, r, parameters);
  if (!SphericalPolicy::Restore(map.x, out.jets, reference, parameters.residual_weight,
                                cached_weight)) {
    out.status=PointStatus::invalid_geometry;return;
  }
  if (lift)for (int a=0;a<3;++a)out.jets[22+a]=out.jets[22+a]+lift[a];
  out.transport[0] *= cached_weight && parameters.residual_weight.Regular()
                          ? cached_weight->f
                          : parameters.residual_weight.Value(map.x);
  const auto u=SphericalPolicy::Physical(map.x,out.jets);
  if (!(u.chi.value>0 && u.alpha.value>0)) out.status=PointStatus::invalid_input;
}

// Reader returns (u-reference), with chi additionally divided by r^2. Reference
// jets are analytic Cartesian jets; lift is the already interpolated Cartesian
// shift-normal correction, with value and first/second derivatives. Ordinary
// interior stencils need radius four for the compatible lapse Hessian.
template <class Reader>
KOKKOS_INLINE_FUNCTION PointState BuildPointState(
    const Reader &q, const mapped::Point &map, const Real inv_h[3],
    const CartesianChartJets &reference,
    const HyperbolicShiftParameters<Real> &parameters, const mapped::Jet *lift = nullptr,
    const mapped::Jet *cached_weight = nullptr, bool affine = false) {
  PointState out;
  Real beta[3]{};
  for (int a = 0; a < 3; ++a) beta[a] = q(22 + a) + reference[22 + a].f - map.x[a];
  const auto shift = LogicalShift::Project(map, beta, affine);
  for (int f = 0; f < Fields::count; ++f) {
    if (!Fields::Independent(f)) continue;
    out.jets[f] = SampleResidualJet(q, f, map, inv_h, shift, Fields::NeedsBulkHessian(f),
                                    out.transport[f], affine);
  }
  RestorePointState(out, q, map, inv_h, reference, parameters, lift, cached_weight);
  return out;
}
}  // namespace z4c_hyperboloidal
#endif
