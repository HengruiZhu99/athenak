#include "z4c/hyperboloidal/native_wormhole.hpp"
#include <iostream>
#include <iomanip>
int main(int argc,char**argv){try{
 using N=z4c::hyperboloidal::NativeWormhole;
 const double tol=argc>1?std::stod(argv[1]):2e-13,b=argc>2?std::stod(argv[2]):.97;
 N n(1./3.,tol,20,40,N::TransitionProfile::polynomial_c6,b);
 std::cout<<std::setprecision(17)<<n.InnerRadius()<<' '<<n.IsotropicScale()<<' '
          <<n.CompactShape()<<' '<<n.CompactJoinResidual()<<'\n';
 double r;while(std::cin>>r){const auto q=n.Point(r);
  std::cout<<r<<' '<<q.chi<<' '<<q.ar<<' '<<q.trace<<' '<<q.lapse<<' '<<q.shift_ratio<<' '<<q.areal<<'\n';}
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
