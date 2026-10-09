#include <iomanip>
#include <iostream>
#include <limits>
#include "z4c/hyperboloidal/layer_reference.hpp"
namespace hyp=z4c::hyperboloidal;
int main(int argc,char**argv) {
 Kokkos::ScopeGuard guard(argc,argv);
 hyp::LayerReference<double> ref(1,.5,{true,.05,.95});ref.Validate();
 std::cout<<std::setprecision(17);
 double r;
 while(std::cin>>r){auto p=ref.At(r,0.,0.);auto const&u=p.state;
  std::cout<<"{\"r\":"<<p.radius<<",\"ld_digits\":"<<std::numeric_limits<long double>::digits<<",\"values\":["
   <<p.omega<<','<<p.domega[0]<<','<<p.omega_hessian[0][0]<<','
   <<u.alpha.value<<','<<u.alpha.d[0]<<','<<u.alpha.dd[0][0]<<','
   <<-u.beta.value[0]<<','<<-u.beta.d[0][0]<<','<<-u.beta.dd[0][0][0]<<','
   <<u.trace.value<<','<<u.trace.d[0]<<','<<u.a.k[0][0]<<','<<u.a.dk[0][0][0]<<','
   <<p.k_bar<<','<<p.outgoing<<','<<p.ingoing<<"]}\n";
 }
}
