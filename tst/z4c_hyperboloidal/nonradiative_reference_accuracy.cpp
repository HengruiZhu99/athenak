// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// A Schwarzschild CMC plateau is stationary and vacuum, independently of how
// its reference jets are constructed. This is not a full evolution test.
#include <cmath>
#include <iostream>
#include <stdexcept>
#include "z4c_hyperboloidal/point_rhs.hpp"
#include "z4c_hyperboloidal/spherical_reference_3d.hpp"
#include "z4c_hyperboloidal/central_z4c_configuration.hpp"
namespace h=z4c_hyperboloidal;
int main() {
  h::CentralZ4cConfiguration cfg;cfg.compact_slicing_end=.97;
  auto p=cfg.Resolve().gauge;p.outer_lapse_damping=.25;
  h::SphericalReference3D reference(p);
  double geometry=0,constraint=0,gauge=0;
  for(double r:{.971,.975,.98,.985,.99,.995,.997})
    for(double angle:{0.,.4,.9}) {
      h::mapped::Point map{};map.x[0]=r*cos(angle);map.x[1]=r*sin(angle);
      for(int a=0;a<3;++a)map.J[a][a]=1;
      const auto j=reference.At(map.x);
      const auto u=h::SphericalPolicy::Physical(map.x,j);
      auto o=p.profile.At(map.x);
      h::SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,o);
      const auto c=h::EvolvedConstraints(u,o);
      const auto rhs=h::EvaluateBulkPoint(map,j,p,reference.source.At(r));
      if(rhs.status!=h::PointStatus::success)throw std::runtime_error("invalid plateau reference");
      for(double v:rhs.rate)if(!std::isfinite(v))
        throw std::runtime_error("nonfinite plateau reference RHS");
      if(!std::isfinite(c.hamiltonian))throw std::runtime_error("nonfinite Hamiltonian");
      for(double v:c.momentum)if(!std::isfinite(v))
        throw std::runtime_error("nonfinite momentum");
      for(int f=0;f<18;++f)geometry=fmax(geometry,fabs(rhs.rate[f]));
      for(int f=18;f<31;++f)gauge=fmax(gauge,fabs(rhs.rate[f]));
      constraint=fmax(constraint,fabs(c.hamiltonian));
      for(double v:c.momentum)constraint=fmax(constraint,fabs(v));
    }
  // Without flat Omega, a user-specified gauge blend may still be inside its
  // unshifted region after the geometric CMC transition. Do not impose beta=-r
  // jets there merely because the geometry has reached the CMC plateau.
  auto inner=p;inner.profile.flat=false;inner.stationary_outer_reference=false;
  inner.profile.flat_radius=.99;inner.inner_begin=.995;inner.inner_end=.999;
  h::SphericalReference3D unshifted(inner);
  const auto &inside=unshifted.RadialJets(.98);
  if(inside[0][9]!=0 || fabs(inside[1][9])>1e-12 || fabs(inside[2][9])>1e-9)
    throw std::runtime_error("analytic plateau jets bypassed the reference gauge blend");
  std::cout<<"CMC plateau geometry RHS="<<geometry<<" gauge RHS="<<gauge
           <<" constraints="<<constraint<<'\n';
  if(!(geometry<1e-9 && gauge<1e-10 && constraint<1e-11))
    throw std::runtime_error("reference differentiation violates stationary vacuum identities");
}
