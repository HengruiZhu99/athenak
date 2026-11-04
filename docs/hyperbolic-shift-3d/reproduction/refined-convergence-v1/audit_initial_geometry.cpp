#include <iostream>
#include <iomanip>
#include "z4c/hyperboloidal/native_boosted_wormhole.hpp"
int main(int argc,char**argv){
 try{
  using N=z4c::hyperboloidal::NativeWormhole;using B=z4c::hyperboloidal::NativeBoostedWormhole;
  const double v=argc>1?std::stod(argv[1]):.1,start=argc>2?std::stod(argv[2]):4,end=argc>3?std::stod(argv[3]):12;
#ifdef USE_GEOMETRY_AUDIT
  B boosted(v,1./3.,true,start,end);
#else
  B boosted(v,1./3.,true);
#endif
  N stationary(1./3.,2e-13,start,end,N::TransitionProfile::polynomial_c6);
  std::cout<<std::setprecision(17)<<"r,ray,x,y,z";
  for(int f=0;f<31;++f)std::cout<<",q"<<f;
#ifdef USE_GEOMETRY_AUDIT
  std::cout<<",Q,Q0,detJ,map_radial,map_tangential,affine_radius,k,stationary_geometry_error";
#endif
  std::cout<<'\n';
  const double directions[8][3]={{1,0,0},{-1,0,0},{0,1,0},{0,0,1},{.6,.8,0},{-.6,.8,0},{.36,.48,.8},{-.36,.48,.8}};
  for(int j=0;j<121;++j){double r=.02+j*.008;
   for(int ray=0;ray<8;++ray){double x[3];for(int a=0;a<3;++a)x[a]=r*directions[ray][a];
#ifdef USE_GEOMETRY_AUDIT
    B::InitialGeometryAudit audit;const auto q=boosted.Fields(x,&audit);
#else
    const auto q=boosted.Fields(x);
#endif
    std::cout<<r<<','<<ray;for(double a:x)std::cout<<','<<a;for(double a:q)std::cout<<','<<a;
#ifdef USE_GEOMETRY_AUDIT
    double error=0; if(v==0){auto ref=stationary.Fields(x);for(int f=0;f<19;++f)error=std::max(error,std::abs(q[f]-ref[f])/std::max(1.,std::abs(ref[f])));}
    for(double a:{audit.spacelike_margin,audit.unboosted_margin,audit.boost_jacobian,audit.physical_map_radial,audit.physical_map_tangential,audit.affine_radius,audit.map_scale,error})std::cout<<','<<a;
#endif
    std::cout<<'\n';
   }
  }
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
