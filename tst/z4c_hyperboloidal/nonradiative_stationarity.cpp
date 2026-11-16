// Pointwise continuum-gauge inventory only; never a full evolution test.
#include <iostream>
#include <iomanip>
#include "z4c_hyperboloidal/point_rhs.hpp"
#include "z4c_hyperboloidal/spherical_reference_3d.hpp"
#include "z4c_hyperboloidal/central_z4c_configuration.hpp"
namespace h=z4c_hyperboloidal;
int main(){h::CentralZ4cConfiguration c;c.compact_slicing_end=.97;auto p=c.Resolve().gauge;p.outer_lapse_damping=.25;
 h::SphericalReference3D reference(p);std::cout<<std::setprecision(17)<<"r,status,max_geometry,lapse,B_radial,shift_radial,H,M_radial\n";
 for(double r:{.02,.05,.1,.3,.6,.8,.9,.94,.95,.97,.98,.995}){
  h::mapped::Point map{};map.x[0]=r;for(int a=0;a<3;++a)map.J[a][a]=1;
  auto j=reference.At(map.x);auto u=h::SphericalPolicy::Physical(map.x,j);auto o=p.profile.At(map.x);
  h::SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,o);
  auto constraints=h::EvolvedConstraints(u,o);auto rhs=h::EvaluateBulkPoint(map,j,p,reference.source.At(r));double geometry=0;
  for(int f=0;f<18;++f)geometry=fmax(geometry,fabs(rhs.rate[f]));
  std::cout<<r<<','<<int(rhs.status)<<','<<geometry<<','<<rhs.rate[18]<<','<<rhs.rate[19]<<','<<rhs.rate[22]<<','<<constraints.hamiltonian<<','<<constraints.momentum[0]<<'\n';
 }
}
