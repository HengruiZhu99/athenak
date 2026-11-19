// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_SCRI_RATE_DIAGNOSTIC_HPP_
#define Z4C_HYPERBOLOIDAL_SCRI_RATE_DIAGNOSTIC_HPP_
#include "z4c_hyperboloidal/spherical_policy.hpp"
namespace z4c_hyperboloidal {
// Passive accepted-state evaluation. Never read a stale RK stage RHS.
// Only metric/lapse/shift rates are admitted here: D's endpoint projection
// depends on adjacent bulk rates and is deliberately not represented.
struct ScriRateDiagnostic {
  std::array<Real,11> metric_rate{}; // chi,gxx,gxy,gxz,gyy,gyz,gzz,alpha,beta[3]
  std::array<Real,12> lapse{};
  std::array<Real,31> unprojected{},penalty_rate{};
};
inline ScriRateDiagnostic DiagnoseScriRate(const mapped::Point &map,Real spacing,
    const CartesianChartJets &j,const HyperbolicShiftParameters<Real> &p,
    const StationaryGaugeSource<Real> &source,Real penalty) {
  ScriRateDiagnostic out;
  const auto rate=SphericalPolicy::Boundary(map,spacing,j,p,source,penalty);
  const auto bare=SphericalPolicy::Boundary(map,spacing,j,p,source,0);
  if(!rate.valid||!bare.valid)throw std::runtime_error("invalid diagnostic rate");
  out.unprojected=rate.rate;
  for(int f=0;f<31;++f)out.penalty_rate[f]=rate.rate[f]-bare.rate[f];
  // Differentiate the implemented five-component determinant-one chart,
  // rather than assuming that the discarded gzz RHS is its tangent.
  CartesianChartJets tangent{};
  for(int f=0;f<31;++f){tangent[f].f=j[f].f;tangent[f].d[0]=rate.rate[f];}
  if(!TryExpandCartesianChart(tangent))throw std::runtime_error("invalid rate chart");
  for(int f=0;f<7;++f)out.metric_rate[f]=tangent[f].d[0];
  out.metric_rate[7]=0; // derivative of the hard endpoint alpha=1 projection
  const auto u=SphericalPolicy::Physical(map.x,j);const auto g=Geometry(u.metric);
  const int ti[6]={0,0,0,1,1,2},tj[6]={0,1,2,1,2,2};
  Real Q=0,Qdot=0,v[3]{},an=0,pn=0,bnn=0;
  for(int a=0;a<3;++a)for(int b=0;b<3;++b)v[a]+=g.inverse[a][b]*map.x[b];
  for(int a=0;a<3;++a){Q+=map.x[a]*v[a];an+=map.x[a]*u.alpha.d[a];
    pn+=map.x[a]*u.trace.d[a];
    for(int b=0;b<3;++b)bnn+=map.x[a]*map.x[b]*u.beta.d[a][b];}
  for(int f=0;f<6;++f)Qdot-=(ti[f]==tj[f]?1:2)*v[ti[f]]*v[tj[f]]*out.metric_rate[1+f];
  const Real s=sqrt(u.chi.value*Q);
  const Real st=(out.metric_rate[0]*Q+u.chi.value*Qdot)/(2*s);
  for(int a=0;a<3;++a)out.metric_rate[8+a]=-st*map.x[a];
  // Exact outer-A identity at alpha=1, beta=-s*n, |n|=1:
  // clock = 2 alpha_n(P+3)+P_n+2 kappa(alpha_n-1)-beta_nn+s-2.
  // Keep signed summands at the SAME point, plus equality residual.
  out.lapse={rate.clock,2*an*(u.trace.value+3),pn,
    2*p.outer_lapse_damping*(an-1),-bnn,s,-2,an,
    rate.speed_t,st,0,rate.rate[6]-tangent[6].d[0]};
  Real sum=0;for(int k=1;k<=6;++k)sum+=out.lapse[k];
  out.lapse[10]=rate.clock-sum;
  return out;
}
} // namespace z4c_hyperboloidal
#endif
