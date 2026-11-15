// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_SCRI_RECONSTRUCTION_HPP_
#define Z4C_HYPERBOLOIDAL_SCRI_RECONSTRUCTION_HPP_
#include "z4c_hyperboloidal/spherical_policy.hpp"
namespace z4c_hyperboloidal {
struct ScriReconstruction {
  Real s=0, A=0, d=0, k=0, invariant=0;
  Real lapse=0, shift[3]{};
  bool valid=false;
};
// The input rate must include the actual geometric boundary penalties. This
// reconstruction preserves s, including an erroneous s != 1; it adds no damping.
KOKKOS_INLINE_FUNCTION ScriReconstruction ReconstructRegularScri(
    const Real n[3],const Z4cJet<Real> &u,const SpatialGeometry<Real> &g,
    const SphericalPolicy::BoundaryResult &rate,
    const HyperbolicShiftParameters<Real> &p) {
  ScriReconstruction out;
  Real Q=0, Bn=0;
  for(int a=0;a<3;++a) {
    Bn+=n[a]*rate.defect[a];
    for(int b=0;b<3;++b)Q+=g.inverse[a][b]*n[a]*n[b];
  }
  if(!g.valid || !rate.valid || !(u.chi.value*Q>0))return out;
  out.s=sqrt(u.chi.value*Q);
  out.A=2*(u.trace.value+3+p.outer_lapse_damping);
  out.d=p.restoring-out.s;
  out.k=-out.s*(u.trace.value+2*u.theta.value)/3;
  if(!Kokkos::isfinite(out.A) || !Kokkos::isfinite(out.d) ||
     !Kokkos::isfinite(out.k) || fabs(out.d)<1e-12 || fabs(out.k)<1e-12)return out;
  out.lapse=-rate.speed_t/out.k;
  out.invariant=rate.clock+Bn/out.d-(out.A/out.k+1/out.d)*rate.speed_t;
  out.valid=Kokkos::isfinite(out.lapse)&&Kokkos::isfinite(out.invariant);
  for(int a=0;a<3;++a) {
    out.shift[a]=-(rate.defect[a]-rate.speed_t*n[a])/out.d;
    out.valid=out.valid&&Kokkos::isfinite(out.shift[a]);
  }
  return out;
}
}  // namespace z4c_hyperboloidal
#endif
