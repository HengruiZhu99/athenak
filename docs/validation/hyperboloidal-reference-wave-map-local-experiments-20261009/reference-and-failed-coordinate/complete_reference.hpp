#ifndef PRIVATE_COMPLETE_COORDINATE_REFERENCE_HPP_
#define PRIVATE_COMPLETE_COORDINATE_REFERENCE_HPP_
#include "taylor.hpp"
#include <map>
#include <string>
using T4=Taylor<4>;using T3=Taylor<3>;using T2=Taylor<2>;
struct CompleteRadial {
  T4 weight{},complement{},omega{};
  T3 L{},b{},alpha{},beta{},chi{},g_radial{},A_radial{},A_tangent{},P{};
  T2 lambda{};
};
inline CompleteRadial CompleteReference(double radius){
  if(!(radius>=0&&radius<1))throw std::runtime_error("CompleteReference requires 0<=r<1");
  CompleteRadial p{};const T4 r4=Variable<4>(radius),one4(1);
  const T4 outer=one4-r4*r4;const T3 r=Variable<3>(radius),one(1),B=T3(2)*r,Lout=one+r*r;
  if(radius<=.05){p.complement=1;p.omega=1;p.L=1;p.alpha=1;p.chi=1;p.g_radial=1;return p;}
  if(radius>=.95){p.weight=1;p.omega=outer;p.L=Lout;p.b=B;p.alpha=Lout;p.beta=-B;p.chi=1;p.g_radial=1;p.P=-6;return p;}
  const T4 s=(r4-T4(.05))/T4(.9),t=(T4(.95)-r4)/T4(.9);
  const T4 g=-one4/s+one4/t;
  const bool left=g[0]<=0;
  const T4 e=Exp(left?g:-g),small=e/(one4+e),large=one4/(one4+e);
  p.weight=left?small:large;p.complement=left?large:small;
  // A separate complementary value retains the tiny 1-w tail near r1.
  p.omega=left?one4+p.weight*(outer-one4):outer+p.complement*(one4-outer);
  const T3 w=Truncate<3>(p.weight),v=Truncate<3>(p.complement),wp=Derivative<3>(p.weight);
  const T3 o=Truncate<3>(p.omega),op=Derivative<3>(p.omega);
  p.L=o-r*op;p.b=B*w;
  T3 delta;
  if(left){
    const T3 root=Power(o*o+p.b*p.b,.5),extra=p.b*p.b/(root+o);
    p.alpha=o+extra;delta=extra+r*op;
  }else{
    const T3 O=Truncate<3>(outer),dO=-T3(2)*r;
    const T3 dOmega=v*(one-O),dBeta=-B*v;
    const T3 q=T3(2)*O*dOmega+dOmega*dOmega+T3(2)*B*dBeta+dBeta*dBeta;
    const T3 da=q/(Power(Lout*Lout+q,.5)+Lout);
    const T3 dop=-wp*(one-O)-v*dO;
    const T3 dL=dOmega-r*dop;
    p.L=Lout+dL;p.alpha=Lout+da;delta=da-dL;
  }
  const T3 ratio_delta=delta/p.L,ratio=one+ratio_delta;
  p.chi=Power(ratio,2./3.);p.g_radial=Power(ratio,-4./3.);
  p.beta=left?-B*w*ratio:-B+B*(v-w*ratio_delta);
  // Original production b=2*r*w allows exact factorizations of A and P.
  const T3 difference=-T3(2)*r*wp/p.L;
  p.A_radial=T3(2./3.)*p.g_radial*difference;
  p.A_tangent=-T3(1./3.)*p.chi*difference;
  const T3 trace_correction=T3(2)*r*o*wp/p.L;
  p.P=left?-T3(6)*w-trace_correction:T3(-6)+T3(6)*v-trace_correction;
  const T3 inv_r=Power(ratio,4./3.),inv_t=Power(ratio,-2./3.);
  const T3 inv_difference=inv_t*(T3(2)*ratio_delta+ratio_delta*ratio_delta);
  p.lambda=-Derivative<2>(inv_r)-T2(2)*Truncate<2>(inv_difference)/Variable<2>(radius);
  return p;
}
inline std::map<std::string,std::array<double,5>> RadialDerivatives(const CompleteRadial&p){
  std::map<std::string,std::array<double,5>>out;
  auto add=[&](const std::string&name,const auto&a,int n){auto&b=out[name];b.fill(0);for(int k=0;k<=n;++k)b[k]=a[k]*Factorial(k);};
  add("weight",p.weight,4);add("complement",p.complement,4);add("omega",p.omega,4);
  add("L",p.L,3);add("b",p.b,3);add("alpha",p.alpha,3);add("beta",p.beta,3);add("chi",p.chi,3);add("g_radial",p.g_radial,3);
  add("A_radial",p.A_radial,3);add("A_tangent",p.A_tangent,3);add("P",p.P,3);add("lambda",p.lambda,2);
  return out;
}
#endif
