// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_SCRI_GEOMETRY_HPP_
#define Z4C_HYPERBOLOIDAL_SCRI_GEOMETRY_HPP_
#include "z4c_hyperboloidal/hyperbolic_shift.hpp"
namespace z4c_hyperboloidal {
struct ScriGeometry {
  Real Q=0,N=0,L=0,D_perp=0,obstruction=0;
  bool valid=false;
};
// n=x/r is held constant along each radial ray when taking partial_n Q.
// D_perp is a vector component, not a derivative. No history reduction enters.
KOKKOS_INLINE_FUNCTION ScriGeometry EvaluateScriGeometry(const Real x[3],
    const Z4cJet<Real> &u,const Real D[3],const OmegaJet<Real> &omega,
    const HyperbolicShiftParameters<Real> &p,
    const SpatialGeometry<Real> *metric=nullptr,
    const SpatialGeometry<Real> *penrose=nullptr) {
  ScriGeometry out;
  const Real r=sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
  if(!(r>0 && u.chi.value>0) || fabs(p.restoring-1)<1e-12)return out;
  const auto g=metric ? *metric : Geometry(u.metric);
  const auto b=penrose ? *penrose : Geometry(PenroseMetric(u.metric,u.chi));
  if(!g.valid || !b.valid)return out;
  Real n[3],v[3]{},chi_n=0,Q_n=0;
  for(int a=0;a<3;++a)n[a]=x[a]/r;
  for(int a=0;a<3;++a) {
    chi_n+=n[a]*u.chi.d[a];out.D_perp+=n[a]*D[a];
    for(int c=0;c<3;++c)v[a]+=g.inverse[a][c]*n[c];
  }
  for(int a=0;a<3;++a) {
    out.Q+=n[a]*v[a];
    for(int c=0;c<3;++c) {
      Real hessian=omega.hessian[a][c];
      for(int k=0;k<3;++k) {
        Q_n-=n[k]*v[a]*v[c]*u.metric.dg[k][a][c];
        hessian-=b.connection[k][a][c]*omega.gradient[k];
      }
      out.L+=b.inverse[a][c]*hessian;
    }
  }
  out.N=chi_n*out.Q+u.chi.value*Q_n;
  const auto coefficient=ShiftCoefficients(r,u.alpha.value,u.chi.value,p);
  out.obstruction=coefficient.mu/(p.restoring-1)*out.D_perp-out.L-3
                  +(p.outer_lapse_damping-1.5)*out.N;
  out.valid=Kokkos::isfinite(out.obstruction)&&Kokkos::isfinite(out.N);
  return out;
}
// Finite continuum source: C4 smoothstep collar, independent of h and reductions.
// Subtraction is a prescribed reference function, not an evolved-field feedback
// or an endpoint projection. At scri the reference obstruction is exactly zero.
KOKKOS_INLINE_FUNCTION Real ScriClockSource(const Real x[3],const Z4cJet<Real> &u,
    const Real D[3],const OmegaJet<Real> &omega,const HyperbolicShiftParameters<Real> &p,
    const StationaryGaugeSource<Real> &source,
    const SpatialGeometry<Real> *metric=nullptr,const SpatialGeometry<Real> *penrose=nullptr) {
  if(p.scri_treatment!=ScriTreatment::clock_source || p.scri_source_epsilon==0)return 0;
  const Real r=sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
  if(r<=p.scri_collar_begin)return 0;
  const auto geometry=EvaluateScriGeometry(x,u,D,omega,p,metric,penrose);
  if(!geometry.valid)return NAN;
  const Real w=Smoothstep9((r-p.scri_collar_begin)/(p.scri_collar_end-p.scri_collar_begin));
  return -p.scri_source_epsilon*w*(geometry.obstruction-source.scri_obstruction_reference);
}
}  // namespace z4c_hyperboloidal
#endif
