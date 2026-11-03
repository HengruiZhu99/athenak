// Point-query driver for the independent embedding audit (not an evolution).
#include <iostream>
#include <iomanip>
#include "src/z4c/hyperboloidal/native_boosted_wormhole.hpp"
int main(int argc,char**argv){
  const bool baseline=argc>1&&std::string(argv[1])=="base";
  z4c::hyperboloidal::NativeBoostedWormhole b(baseline?0:argc>1?std::stod(argv[1]):.1,1./3.,argc>2&&std::string(argv[2])=="free");
  z4c::hyperboloidal::NativeWormhole a(1./3.,2e-13,4,12,z4c::hyperboloidal::NativeWormhole::TransitionProfile::polynomial_c6);
  double x[3];std::cout<<std::setprecision(17);
  while(std::cin>>x[0]>>x[1]>>x[2]){auto q=baseline?a.Fields(x):b.Fields(x);for(auto v:q)std::cout<<v<<' ';std::cout<<'\n';}
}
