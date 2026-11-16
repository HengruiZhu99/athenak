// Frozen-background gauge characteristic travel estimate, not an evolution.
#include <iostream>
#include <iomanip>
#include "z4c_hyperboloidal/spherical_reference_3d.hpp"
#include "z4c_hyperboloidal/spherical_policy.hpp"
#include "z4c_hyperboloidal/central_z4c_configuration.hpp"
namespace h=z4c_hyperboloidal;
int main() {
  h::CentralZ4cConfiguration c;c.compact_slicing_end=.97;
  auto p=c.Resolve().gauge;p.outer_lapse_damping=.25;
  h::SphericalReference3D reference(p);
  auto speed=[&](double r){
    double x[3]={r,0,0};auto j=reference.At(x);
    const auto u=h::SphericalPolicy::Physical(x,j);const auto g=h::Geometry(u.metric);
    const auto coeff=h::ShiftCoefficients(r,u.alpha.value,u.chi.value,p);
    return -u.beta.value[0]+sqrt(h::GaugeLapseSpeedSquared(u.alpha.value,u.chi.value,coeff,g.inverse[0][0],h::LargestMetricEigenvalue(u.metric.g),p));
  };
  std::cout<<std::setprecision(17)<<"start_r,speed,travel_code,travel_M\n";
  for(double r:{.9465,.9555,.9645}) {
    constexpr int n=200;double step=(1-r)/n,sum=0;
    for(int k=0;k<n;++k)sum+=step/speed(r+(k+.5)*step);
    std::cout<<r<<','<<speed(r)<<','<<sum<<','<<3*sum<<'\n';
  }
}
