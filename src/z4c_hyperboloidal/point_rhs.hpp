// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_POINT_RHS_HPP_
#define Z4C_HYPERBOLOIDAL_POINT_RHS_HPP_

#include "z4c_hyperboloidal/point_state.hpp"

namespace z4c_hyperboloidal {
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
KOKKOS_INLINE_FUNCTION BulkPointResult EvaluateBulkPoint(const mapped::Point &map,
    const CartesianChartJets &j,const HyperbolicShiftParameters<Real> &p,
    const StationaryGaugeSource<Real> &source,const PointForcing &forcing={}) {
  BulkPointResult out;
  const Real *x=map.x,r=SphericalPolicy::Radius(x);
  const auto u=SphericalPolicy::Physical(x,j);
  if (!(u.chi.value>0 && u.alpha.value>0)) {out.status=PointStatus::invalid_input;return out;}
  const auto g=Geometry(u.metric);
  if (!g.valid) {out.status=PointStatus::invalid_geometry;return out;}
  auto omega=p.profile.At(x);
  SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,omega);
  auto parts=ConformalRHS(u,omega,p.kappa1,p.kappa2);
  if (!HyperbolicDamping(x,u,omega,parts,p)) {out.status=PointStatus::invalid_geometry;return out;}
  Z4cRHS<Real> geometric{};
  if (!AssembleInterior(parts,omega.omega,geometric)) {out.status=PointStatus::invalid_assembly;return out;}
  SphericalPolicy::StoreGeometry(geometric,out.rate);
  const auto B=SphericalPolicy::Vector(j,19),D=SphericalPolicy::Vector(j,25);
  auto gauge=HyperbolicGauge(x,u,B,D.value,p);
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

using ScriPointResult=SphericalPolicy::BoundaryResult;
KOKKOS_INLINE_FUNCTION ScriPointResult EvaluateScriPoint(const mapped::Point &map,
    Real spacing,const CartesianChartJets &j,const HyperbolicShiftParameters<Real> &p,
    const StationaryGaugeSource<Real> &source,Real penalty) {
  // Preserve the existing endpoint validity and pole derivative conventions.
  return SphericalPolicy::Boundary(map,spacing,j,p,source,penalty);
}
}  // namespace z4c_hyperboloidal
#endif
