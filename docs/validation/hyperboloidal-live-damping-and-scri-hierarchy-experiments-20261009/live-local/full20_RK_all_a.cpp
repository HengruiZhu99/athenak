// Actual dual-kernel continuum Fourier generators at the native span2.1
// outermost active radius. This is a scalar RK root gate, not a native run
// or a finite-difference Nyquist-symbol/semigroup stability assertion.
#include "profile_helpers.hpp"
int main(){std::cout<<std::setprecision(17)<<'[';bool first=true;
 const int ns[3]={24,36,48};const double os[3]={.0142578125,.0038368055555556,.003251953125};
 for(double a:{.5,.75,1.,2.}){hyp::LayerReference<double>ref(1.,a,{true,.05,.95});
 for(int ni=0;ni<3;++ni){const double r=std::sqrt(1-os[ni]);const auto p=ref.At(r,0.,0.);
 for(int mode:{0,1})for(double pert:{0.,.01})for(bool oblique:{false,true})for(int freq:{0,1}){
 const double k=freq?256.:std::acos(-1.)*ns[ni]/2.1;
 const std::array<double,3>n=oblique?std::array<double,3>{.36,-.48,.8}:std::array<double,3>{1,0,0};
 if(!first)std::cout<<',';first=false;
 std::cout<<"{\"a\":"<<a<<",\"N\":"<<ns[ni]<<",\"span\":2.1,\"r\":"<<r<<",\"Omega\":"<<p.omega<<",\"dt\":"<<.03*p.omega<<",\"k\":"<<k<<",\"frequency\":"<<freq<<",\"profile\":"<<mode<<",\"perturb\":"<<pert<<",\"oblique\":"<<oblique<<",\"L\":";
 Print(Matrix(p,a,10,mode,true,pert,k,n));std::cout<<'}';
 }}}std::cout<<"]\n";
}
