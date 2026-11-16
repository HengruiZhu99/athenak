// Evaluate the actual outer lapse equation on independently manufactured 3+1 data.
#include <iostream>
#include <iomanip>
#include "z4c_hyperboloidal/stationary_gauge_reference.hpp"
#include "z4c_hyperboloidal/central_z4c_configuration.hpp"
namespace h=z4c_hyperboloidal;
int main() {
  h::CentralZ4cConfiguration c;c.compact_slicing_end=.97;
  auto p=c.Resolve().gauge;p.outer_lapse_damping=.25;
  h::StationaryGaugeReference reference(p);int id;
  std::cout<<std::setprecision(17)<<"id,r,alpha_dt,A_lapse_rhs,residual,reference_source\n";
  while(std::cin>>id) {
    Real x[3],actual;h::Z4cJet<Real> u{};
    for(auto &v:x)std::cin>>v;
    std::cin>>u.alpha.value>>actual;
    for(auto &v:u.alpha.d)std::cin>>v;
    for(auto &v:u.beta.value)std::cin>>v;
    std::cin>>u.trace.value>>u.chi.value;
    if(!std::cin)return 2;
    const Real r=sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
    if(!(r>.9 && r<1))return 3;
    const h::VectorJet<Real> B{};const Real D[3]{};
    auto gauge=h::HyperbolicGauge(x,u,B,D,p);
    const auto source=reference.At(r);const auto before=gauge.lapse_regular;
    h::ApplyStationaryGaugeSource(x,u.alpha.value,source,gauge);
    const auto rhs=gauge.lapse_regular+gauge.lapse_pole/p.profile.At(x).omega;
    std::cout<<id<<','<<r<<','<<actual<<','<<rhs<<','<<rhs-actual<<','<<gauge.lapse_regular-before<<'\n';
  }
}
