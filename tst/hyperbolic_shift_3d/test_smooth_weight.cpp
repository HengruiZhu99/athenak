#include <cmath>
#include <iostream>
#include <stdexcept>
#include "z4c/hyperboloidal/hyperbolic_shift.hpp"
namespace h=z4c::hyperboloidal;
int main() {
  auto require=[](bool ok,const char*what){if(!ok)throw std::runtime_error(what);};
  try {
    require(h::Smoothstep9(-.1)==0&&h::Smoothstep9(0.)==0&&
            h::Smoothstep9(1.)==1&&h::Smoothstep9(1.1)==1,"constant endpoints");
    double previous=0,max_error=0,integral=0;
    for(int i=0;i<=10000;++i){
      const double t=i/10000.,s=h::Smoothstep9(t),r=.8-.24*t;
      require(s>=previous-1e-15&&s>=0&&s<=1,"monotonic bounded blend");previous=s;
      require(std::abs(s+h::Smoothstep9(1-t)-1)<2e-14,"symmetric blend");
      const auto c=h::ShiftCoefficients(r,.7,.9);
      require(c.w*c.collar==0,"disjoint w and harmonic collar");
      if(i>0&&i<10000){
        const double eps=1e-5;
        const double fd=(h::ShiftCoefficients(r-2*eps,.7,.9).w-8*h::ShiftCoefficients(r-eps,.7,.9).w+8*h::ShiftCoefficients(r+eps,.7,.9).w-h::ShiftCoefficients(r+2*eps,.7,.9).w)/(12*eps);
        max_error=std::max(max_error,std::abs(fd-c.dw_dr));
      }
      integral+=-c.dw_dr*.24/10000.*(i==0||i==10000?.5:1.);
    }
    require(max_error<1e-8,"analytic radial derivative mismatch");
    require(std::abs(integral-1)<1e-12,"weight derivative normalization");
    for(double r:{0.,.3,.56,.8,.85,1.}){
      const auto c=h::ShiftCoefficients(r,.7,.9);
      if(r<=.56)require(c.w==1&&c.dw_dr==0,"inner constant branch");
      if(r>=.8)require(c.w==0&&c.dw_dr==0,"outer constant branch");
      require(c.collar==h::SmoothCollar((r-.8)/.1),"harmonic collar unchanged");
    }
    std::cout<<"smooth weight derivative error="<<max_error<<" integral="<<integral<<'\n';
  } catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
