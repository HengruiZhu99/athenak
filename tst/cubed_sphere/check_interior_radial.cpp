// Host coefficient/accuracy admission, not a PDE stability test.
#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdexcept>
using Real=double;
#define KOKKOS_INLINE_FUNCTION inline
#include "mesh/cubed_sphere/interior_radial.hpp"
using namespace cubed_sphere;
int main() {
 std::cout<<std::setprecision(17)<<"h,D1_max_error,D2_max_error,mixed_max_error\n";
 for(int depth=0;depth<3;++depth)for(int power=0;power<=5;++power) {
  auto sample=[=](int j){return std::pow(j-5.+depth,power);};
  double d1=InteriorRadialFirst(sample,depth,1.),d2=InteriorRadialSecond(sample,depth,1.);
  if(std::abs(d1-(power==1?1.:0.))>1e-10 || std::abs(d2-(power==2?2.:0.))>1e-10)
   throw std::runtime_error("polynomial moment failure");
 }
 for(double h:{.08,.04,.02,.01}) {
  double first=0,second=0,mixed=0;
  for(int depth=0;depth<3;++depth) {
   double x=1-depth*h;
   auto f=[=](int j){return std::exp(1+(j-5)*h);};
   first=std::fmax(first,std::abs(InteriorRadialFirst(f,depth,h)-std::exp(x)));
   second=std::fmax(second,std::abs(InteriorRadialSecond(f,depth,h)-std::exp(x)));
   // Tensor-product radial D1 and centered fourth-order angular D1.
   auto angular=[=](int j) {
    double r=1+(j-5)*h;
    return (std::exp(r-2*h)-8*std::exp(r-h)+8*std::exp(r+h)-std::exp(r+2*h))/(12*h);
   };
   mixed=std::fmax(mixed,std::abs(InteriorRadialFirst(angular,depth,h)-std::exp(x)));
  }
  std::cout<<h<<","<<first<<","<<second<<","<<mixed<<"\n";
 }
}
