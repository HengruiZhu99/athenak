// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_CURRENT_SPHERICAL_HPP_
#define Z4C_HYPERBOLOIDAL_CURRENT_SPHERICAL_HPP_
#include "z4c/hyperboloidal/spherical_tensor.hpp"
#include "z4c/hyperboloidal/hyperbolic_shift.hpp"
#include "z4c/hyperboloidal/tensor_scri.hpp"
namespace z4c::hyperboloidal::current_spherical {
// TOTAL variables; radial/tangential metric and curvature remain independent.
// No algebraic/geometric constraint is reset by this adapter.
enum Field {chi,gr,gt,ar,at,khat,theta,lambda,alpha,beta,flux,driver,count};
using State=std::array<double,count>;
struct Jets {Z4cJet<double> u;VectorJet<double> B,D;};
inline Jets Lift(double r,const double*q,const double*d,const double*dd) {
  if(!(r>0 && r<=1))throw std::invalid_argument("spherical chart excludes puncture");
  Jets j{};auto&u=j.u;
  u.metric=RadialTensor(q[gr],d[gr],dd[gr],q[gt],d[gt],dd[gt],r);
  const auto A=RadialTensor(q[ar],d[ar],dd[ar],q[at],d[at],dd[at],r);
  for(int a=0;a<3;++a)for(int b=0;b<3;++b){u.a.k[a][b]=A.g[a][b];
    for(int c=0;c<3;++c)u.a.dk[c][a][b]=A.dg[c][a][b];}
  u.chi=RadialScalar(q[chi],d[chi],dd[chi],r);
  u.trace=RadialScalar(q[khat],d[khat],dd[khat],r);
  u.theta=RadialScalar(q[theta],d[theta],dd[theta],r);
  u.alpha=RadialScalar(q[alpha],d[alpha],dd[alpha],r);
  u.lambda=RadialVector(q[lambda],d[lambda],dd[lambda],r);
  u.beta=RadialVector(q[beta],d[beta],dd[beta],r);
  j.B=RadialVector(q[flux],d[flux],dd[flux],r);
  j.D=RadialVector(q[driver],d[driver],dd[driver],r);return j;
}
inline State Pack(const Z4cRHS<double>&v) {
  return {v.chi,v.metric[0][0],v.metric[1][1],v.a[0][0],v.a[1][1],
          v.trace,v.theta,v.lambda[0],0,0,0,0};
}
struct Parts {
  State regular{},pole{},trace{};
  OmegaJet<double> omega{};
  Z4cRHSParts<double> geometry{};
};
inline Parts Evaluate(const double x[3],const Jets&j,
    const HyperbolicShiftParameters<double>&p,const StationaryGaugeSource<double>&source={}) {
  Parts out{};const auto&u=j.u;
  out.omega=p.profile.At(x);
  SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,out.omega);
  out.geometry=ConformalRHS(u,out.omega,p.kappa1,p.kappa2);
  if(!HyperbolicDamping(x,u,out.omega,out.geometry,p))
    throw std::runtime_error("invalid current spherical geometric RHS");
  auto gauge=HyperbolicGauge(x,u,j.B,j.D.value,p);
  if(p.stationary_outer_reference)ApplyStationaryGaugeSource(x,u.alpha.value,source,gauge);
  if(!gauge.valid)throw std::runtime_error("invalid current spherical gauge RHS");
  out.regular=Pack(out.geometry.regular);out.pole=Pack(out.geometry.pole);
  out.regular[alpha]=gauge.lapse_regular;out.pole[alpha]=gauge.lapse_pole;
  out.regular[beta]=gauge.shift_regular[0];out.pole[beta]=gauge.shift_pole[0];
  out.regular[flux]=gauge.flux_rate[0];
  double rate[3];
  if(!HyperbolicAuxiliaryRate(u,j.D,out.geometry.regular.lambda,rate,p.damping))
    throw std::runtime_error("invalid current spherical auxiliary RHS");
  out.regular[driver]=rate[0];out.pole[driver]=out.geometry.pole.lambda[0];
  if(out.omega.omega==0){
    out.trace=Pack(ScriRegularityTraces(u,out.omega));
    const auto g=Geometry(u.metric);
    out.trace[chi]=u.chi.value-1/g.inverse[0][0];
    out.trace[khat]=u.trace.value+3;out.trace[theta]=u.theta.value;
  }
  return out;
}
// Same geometric l'Hopital kernel, penalties, clock and dependent normal D as
// HyperbolicScriDevice. Normal numerator differences are a caller policy.
// Returns the algebraic D trace; its differential compatibility is a diagnostic,
// not an independently enforced boundary datum.
inline State ScriRate(const Jets&j,const Parts&v,const State&normal_pole,
    double h,double penalty_scale,const HyperbolicShiftParameters<double>&p,
    double&dependent_D,double&clock_residual) {
  if(v.omega.omega!=0 || !(h>0 && penalty_scale>0))
    throw std::invalid_argument("invalid scri limit request");
  const auto&u=j.u;
  TensorPole<ScriDual<double>> normal{};
  normal.geometry.chi.derivative=normal_pole[chi];
  normal.geometry.metric[0][0].derivative=normal_pole[gr];
  normal.geometry.metric[1][1].derivative=normal.geometry.metric[2][2].derivative=normal_pole[gt];
  normal.geometry.a[0][0].derivative=normal_pole[ar];
  normal.geometry.a[1][1].derivative=normal.geometry.a[2][2].derivative=normal_pole[at];
  normal.geometry.trace.derivative=normal_pole[khat];
  normal.geometry.theta.derivative=normal_pole[theta];
  normal.geometry.lambda[0].derivative=normal_pole[lambda];
  State rate=Pack(TensorScriLimit(v.geometry.regular,normal,-1.));
  for(int f:{alpha,beta,flux,driver})rate[f]=v.regular[f]-normal_pole[f];
  for(int f:{chi,ar,at,khat,theta,lambda})
    rate[f]+=penalty_scale/h*(4*v.pole[f]-8*v.trace[f]);
  clock_residual=rate[alpha];rate[alpha]=0;
  const double speed=sqrt(u.chi.value/u.metric.g[0][0]);
  rate[beta]=-.5*speed*(rate[chi]/u.chi.value-rate[gr]/u.metric.g[0][0]);
  const double mu=ShiftCoefficients(1.,1.,u.chi.value,p).mu;
  const double regular=-1+u.beta.value[0]*u.beta.d[0][0];
  const double restoring=p.restoring*(u.beta.d[0][0]+1);
  dependent_D=(rate[beta]-regular-restoring)/mu;
  rate[driver]=0;return rate;
}
} // namespace z4c::hyperboloidal::current_spherical
#endif
