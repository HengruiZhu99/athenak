#include <iomanip>
#include <iostream>
#include "z4c/hyperboloidal/native_wormhole.hpp"
int main(int argc,char**argv){
 try{
  const double tol=argc>1?std::stod(argv[1]):2e-13;
  z4c::hyperboloidal::NativeWormhole data(1./3.,tol);
  std::cout<<std::setprecision(17)<<"r,chi,Ar,K,alpha,shift_ratio,R,r0,scale\n";
  std::vector<double>radii={.0001,.001,.01,.03,.07,.1,.15,.2,.25,.3,.4,.5,.6,.7,.8,.9,.99,.99999,1-1e-10,1};
  for(int i=0;i<=100;++i)radii.push_back(.005+.995*i/100.);
  for(double r:radii){
   const auto v=data.Point(r);
   std::cout<<r<<','<<v.chi<<','<<v.ar<<','<<v.trace<<','<<v.lapse<<','<<v.shift_ratio<<','<<v.areal<<','<<data.InnerRadius()<<','<<data.IsotropicScale()<<'\n';
   if(r<1){const double x[3]={.6*r,.8*r,0};const auto q=data.Fields(x);
    if(std::abs(q[7]+q[10]+q[12])>1e-12)throw std::runtime_error("nonzero trace of initial A");
    if(q[0]<=0||q[18]+(1+r*r)/2<=0)throw std::runtime_error("nonpositive initial data");}
  }
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
