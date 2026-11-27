// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_POINT_RHS_HPP_
#define Z4C_HYPERBOLOIDAL_POINT_RHS_HPP_

#include "z4c_hyperboloidal/point_state.hpp"

namespace z4c_hyperboloidal {
// Build the connection and combined Ricci tensor used in the bulk evolution.
template <typename T>
KOKKOS_INLINE_FUNCTION SpatialGeometry<T> BulkRHSGeometry(
    const MetricJet<T> &m, const VectorJet<T> &lambda) {
  SpatialGeometry<T> g{};
  if (!InvertMetric(m.g, g.inverse, g.determinant)) return g;
  T lower[3][3][3];
  for (int k = 0; k < 3; ++k)
    for (int i = 0; i < 3; ++i)
      for (int j = i; j < 3; ++j)
        lower[k][i][j] = lower[k][j][i] =
            T(.5) * (m.dg[i][k][j] + m.dg[j][k][i] - m.dg[k][i][j]);
  for (int k = 0; k < 3; ++k)
    for (int i = 0; i < 3; ++i)
      for (int j = i; j < 3; ++j) {
        T value = 0;
        for (int a = 0; a < 3; ++a) value += g.inverse[k][a] * lower[a][i][j];
        g.connection[k][i][j] = g.connection[k][j][i] = value;
        g.contracted[k] += (i == j ? 1 : 2) * g.inverse[i][j] * value;
      }
  // Metric compatibility gives the derivative of the inverse from the
  // already raised Christoffels. The symmetric Ricci plus the connection
  // constraint then needs just the metric Laplacian and first derivatives.
  T dinv[3][3][3];
  for (int d = 0; d < 3; ++d) {
    T raised[3][3]{};
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k)
          raised[i][j] += g.inverse[i][k] * g.connection[j][d][k];
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) dinv[d][i][j] = -raised[i][j] - raised[j][i];
  }
  for (int i = 0; i < 3; ++i)
    for (int j = i; j < 3; ++j) {
      T ricci = 0;
      for (int a = 0; a < 3; ++a) {
        ricci += T(.5) * (m.g[a][i] * lambda.d[j][a] + m.g[a][j] * lambda.d[i][a] +
                          g.contracted[a] * m.dg[a][i][j]);
        for (int b = 0; b < 3; ++b) {
          ricci -= T(.5) * (g.inverse[a][b] * m.ddg[a][b][i][j] +
                            dinv[j][a][b] * m.dg[a][i][b] +
                            dinv[i][a][b] * m.dg[a][j][b]) +
                   g.connection[a][i][b] * g.connection[b][j][a];
        }
      }
      g.ricci[i][j] = g.ricci[j][i] = ricci;
      g.scalar += (i == j ? 1 : 2) * g.inverse[i][j] * ricci;
    }
  g.valid=Kokkos::isfinite(g.scalar);
  return g;
}

struct PointForcing {
  // Already in physical stored-field units (including the chi r^2 factor).
  Real transport[Fields::count]{};
  Real dissipation[Fields::count]{};
};
struct BulkPointResult {
  Real rate[Fields::count]{};
  PointStatus status=PointStatus::success;
  Real gap=0, speed[3]{};  // speed per unit logical coordinate, no dt/h factor
};

// Complete Cartesian jets after reference restoration, algebraic expansion and
// the endpoint normal lift. Dormant J fields remain registered but do not evolve
// in the N12 formulation. No timestep, array ownership or MPI is hidden here.
KOKKOS_INLINE_FUNCTION BulkPointResult EvaluatePhysicalBulkPoint(
    const mapped::Point &map, const Z4cJet<Real> &u, const VectorJet<Real> &B,
    const VectorJet<Real> &D, const HyperbolicShiftParameters<Real> &p,
    const StationaryGaugeSource<Real> &source, const PointForcing &forcing = {},
    const OmegaJet<Real> *spatial_omega = nullptr) {
  BulkPointResult out;
  const Real *x=map.x,r=SphericalPolicy::Radius(x);
  if (!(u.chi.value>0 && u.alpha.value>0)) {out.status=PointStatus::invalid_input;return out;}
  const auto g=BulkRHSGeometry(u.metric, u.lambda);
  if (!g.valid) {out.status=PointStatus::invalid_geometry;return out;}
  auto omega = spatial_omega ? *spatial_omega : p.profile.At(x);
  SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,omega);
  const bool flat_inner = configured_conformal_frame && inner_telegrapher &&
                          p.profile.flat && r <= p.profile.flat_radius &&
                          r <= p.inner_begin;
  const auto parts = flat_inner ? InnerPlateauRHS(u, omega.omega, p, g)
                                : DampedConformalRHS<Real, true>(x, u, omega, p, &g);
  if (!parts.valid) {
    out.status = PointStatus::invalid_geometry;
    return out;
  }
  Z4cRHS<Real> geometric{};
  if (!AssembleInterior(parts,omega.omega,geometric)) {out.status=PointStatus::invalid_assembly;return out;}
  SphericalPolicy::StoreGeometry(geometric,out.rate);
  auto gauge = HyperbolicGauge(x, u, B, D.value, p, &g, &omega);
  ApplyStationaryGaugeSource(x,u.alpha.value,source,gauge);
  if (!gauge.valid) {out.status=PointStatus::invalid_gauge;return out;}
  out.rate[18]=gauge.lapse_regular+gauge.lapse_pole/omega.omega;
  Real driver[3]{};
  if (!HyperbolicAuxiliaryRate(u,D,geometric.lambda,driver,p.damping)) {
    out.status=PointStatus::invalid_auxiliary;return out;
  }
  for (int a=0;a<3;++a) {
    out.rate[19+a]=gauge.flux_rate[a];out.rate[25+a]=driver[a];
    out.rate[22+a]=gauge.shift_regular[a]+gauge.shift_pole[a]/omega.omega;
  }
  for (int f=0;f<28;++f)out.rate[f]+=forcing.transport[f];
  for (int f=0;f<Fields::count;++f)out.rate[f]+=forcing.dissipation[f];
  // Lambda transport cancels in D_t=Lambda_t-beta.grad Lambda+beta.grad D-eta D;
  // Lambda KO does not cancel, so D receives this additional contribution.
  for (int a=0;a<3;++a)out.rate[25+a]+=forcing.dissipation[15+a];
  const auto coefficient=ShiftCoefficients(r,u.alpha.value,u.chi.value,p);
  out.gap=AllDirectionShiftLapseGap(u.metric.g,u.chi.value,coefficient);
  Real characteristic_chi=u.chi.value;
  if (p.cartesian_chi_guard)
    characteristic_chi+=coefficient.inner*(fmax(characteristic_chi,
        p.inner_chi_guard/(omega.omega*omega.omega))-characteristic_chi);
  for (int A=0;A<3;++A) {
    Real transport=0,Q=0,E=0;
    for (int a=0;a<3;++a) {
      transport+=map.J[A][a]*u.beta.value[a];E+=map.J[A][a]*map.J[A][a];
      for (int b=0;b<3;++b)Q+=g.inverse[a][b]*map.J[A][a]*map.J[A][b];
    }
    Real speed2=fmax(GaugeShiftSpeedSquared(coefficient,Q),
        GaugeLapseSpeedSquared(u.alpha.value,characteristic_chi,coefficient,Q,E,p));
    if (p.enforce_cfl)speed2=fmax(speed2,u.alpha.value*u.alpha.value*characteristic_chi*Q);
    out.speed[A]=fabs(transport)+sqrt(speed2);
  }
  for (int f=0;f<Fields::count;++f)
    if (!Kokkos::isfinite(out.rate[f]))out.status=PointStatus::nonfinite_rhs;
  return out;
}

KOKKOS_INLINE_FUNCTION BulkPointResult EvaluateBulkPoint(
    const mapped::Point &map, const CartesianChartJets &j,
    const HyperbolicShiftParameters<Real> &p, const StationaryGaugeSource<Real> &source,
    const PointForcing &forcing = {}, const OmegaJet<Real> *spatial_omega = nullptr) {
  return EvaluatePhysicalBulkPoint(
      map, SphericalPolicy::Physical(map.x, j), SphericalPolicy::Vector(j, 19),
      SphericalPolicy::Vector(j, 25), p, source, forcing, spatial_omega);
}

// Construct only the tensors consumed by the bulk equations. Curvature,
// Lambda, B, and D use first derivatives only. The residual and reference
// are combined as each independent field is loaded.
template <class Load>
KOKKOS_INLINE_FUNCTION BulkPointResult EvaluateBulkFieldJets(
    const mapped::Point &map, const HyperbolicShiftParameters<Real> &params,
    const StationaryGaugeSource<Real> &source, PointForcing &forcing, Load load,
    const OmegaJet<Real> &spatial_omega, bool evolve_all = false) {
  using J = mapped::Jet;
  Z4cJet<Real> u{};
  VectorJet<Real> B{}, D{};
  CartesianChartMetric<J> chart{
      load(1, true) + J(1), load(2, true), load(3, true),
      load(4, true) + J(1), load(5, true), {}, {},
  };
  // With every component evolved the zz jets are differenced like the other five.
  bool valid_chart;
  if (evolve_all) {
    chart.minor = chart.a * chart.d - chart.b * chart.b;
    valid_chart = chart.a.f > 0 && chart.minor.f > 0;
    chart.f = load(6, true) + J(1);
  } else {
    valid_chart = chart.Complete();
  }
  if (!valid_chart) {
    BulkPointResult bad;
    bad.status = PointStatus::invalid_geometry;
    return bad;
  }
  const auto metric = [&](int i, int j, const J &v) {
    u.metric.g[i][j] = u.metric.g[j][i] = v.f;
    for (int k = 0; k < 3; ++k) {
      u.metric.dg[k][i][j] = u.metric.dg[k][j][i] = v.d[k];
      for (int l = 0; l < 3; ++l)
        u.metric.ddg[k][l][i][j] = u.metric.ddg[k][l][j][i] = v.h[k][l];
    }
  };
  metric(0, 0, chart.a);
  metric(0, 1, chart.b);
  metric(0, 2, chart.c);
  metric(1, 1, chart.d);
  metric(1, 2, chart.e);
  metric(2, 2, chart.f);
  const J xx = load(7, false), xy = load(8, false), xz = load(9, false);
  const J yy = load(10, false), yz = load(11, false);
  const J zz = evolve_all ? load(12, false) : chart.TraceFreeZZ(xx, xy, xz, yy, yz);
  const auto curvature = [&](int i, int j, const J &v) {
    u.a.k[i][j] = u.a.k[j][i] = v.f;
    for (int k = 0; k < 3; ++k) u.a.dk[k][i][j] = u.a.dk[k][j][i] = v.d[k];
  };
  curvature(0, 0, xx);
  curvature(0, 1, xy);
  curvature(0, 2, xz);
  curvature(1, 1, yy);
  curvature(1, 2, yz);
  curvature(2, 2, zz);
  u.chi = SphericalPolicy::Scalar(load(0, true));
  u.alpha = SphericalPolicy::Scalar(load(18, true));
  u.trace = SphericalPolicy::Scalar(load(13, false));
  u.theta = SphericalPolicy::Scalar(load(14, false));
  const Real r = SphericalPolicy::Radius(map.x);
  u.trace.value -= 3;
  u.alpha.value += (1 + r * r) / 2;
  const auto vector = [&](int field, bool hessian, VectorJet<Real> &v) {
    for (int i = 0; i < 3; ++i) {
      const J q = load(field + i, hessian);
      v.value[i] = q.f;
      for (int j = 0; j < 3; ++j) {
        v.d[j][i] = q.d[j];
        if (hessian) {
          for (int k = 0; k < 3; ++k) v.dd[j][k][i] = q.h[j][k];
        }
      }
    }
  };
  vector(15, false, u.lambda);
  vector(19, false, B);
  vector(22, true, u.beta);
  vector(25, false, D);
  for (int i = 0; i < 3; ++i) {
    u.alpha.d[i] += map.x[i];
    u.alpha.dd[i][i] += 1;
    u.beta.value[i] -= map.x[i];
    u.beta.d[i][i] -= 1;
  }
  return EvaluatePhysicalBulkPoint(map, u, B, D, params, source, forcing, &spatial_omega);
}

using ScriPointResult=SphericalPolicy::BoundaryResult;
KOKKOS_INLINE_FUNCTION ScriPointResult EvaluateScriPoint(const mapped::Point &map,
    Real spacing,const CartesianChartJets &j,const HyperbolicShiftParameters<Real> &p,
    const StationaryGaugeSource<Real> &source,Real penalty,
    const SpatialGeometry<Real> *metric_geometry=nullptr,
    const SpatialGeometry<Real> *penrose_geometry=nullptr,
    const SurfacePoleCache *pole_cache=nullptr) {
  // Preserve the existing endpoint validity and pole derivative conventions.
  return SphericalPolicy::Boundary(map, spacing, j, p, source, penalty, metric_geometry,
                                   penrose_geometry, pole_cache);
}
}  // namespace z4c_hyperboloidal
#endif
