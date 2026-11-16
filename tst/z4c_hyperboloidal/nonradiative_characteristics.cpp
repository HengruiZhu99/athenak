// Frozen-background gauge characteristic travel estimate, not an evolution.
#include <iostream>
#include <iomanip>
#include <vector>
#include "z4c_hyperboloidal/spherical_reference_3d.hpp"
#include "z4c_hyperboloidal/spherical_policy.hpp"
#include "z4c_hyperboloidal/central_z4c_configuration.hpp"
namespace h=z4c_hyperboloidal;
int main(int argc,char **argv) {
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
  std::cerr<<std::setprecision(17)<<"gauge blend="<<p.inner_begin<<","<<p.inner_end
    <<" compact reference transition="<<c.Resolve().compact_radius<<","<<c.Resolve().slicing_end<<'\n';
  std::vector<double> starts;
  for(int i=1;i<argc;++i)starts.push_back(std::stod(argv[i]));
  if(starts.empty())starts={.9465,.9555,.9645};
  for(double r:starts) {
    if(!(r>0 && r<1))throw std::invalid_argument("start radius must lie in (0,1)");
    constexpr int n=200;double step=(1-r)/n,sum=0;
    for(int k=0;k<n;++k)sum+=step/speed(r+(k+.5)*step);
    std::cout<<r<<','<<speed(r)<<','<<sum<<','<<3*sum<<'\n';
  }
}
