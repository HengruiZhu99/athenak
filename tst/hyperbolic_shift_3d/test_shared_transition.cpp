#include <cmath>
#include <iostream>
#include <stdexcept>
#include "athena.hpp"
#include "z4c/hyperboloidal/hyperbolic_shift.hpp"
namespace h=z4c::hyperboloidal;
static void Check(bool yes,const char*msg){if(!yes)throw std::runtime_error(msg);}
static void Near(double a,double b,const char*msg){Check(std::abs(a-b)<3e-12*(1+std::abs(b)),msg);}
int main(int argc,char**argv){
 Kokkos::initialize(argc,argv);int code=0;
 try{
  static_assert(h::shared_restoration && h::short_inner_gauge);
  for(double r:{.01,.2,.21,.29,.38,.47,.55,.56,.7,.8,.85,.9,.999,1.}){
   double x[3]={.6*r,0,.8*r},D[3]={.13,-.19,.23};
   h::Z4cJet<double>u{};h::VectorJet<double>B{};
   u.chi.value=.43;u.alpha.value=.71;u.trace.value=.12;
   for(int i=0;i<3;++i){
    u.metric.g[i][i]=1;u.beta.value[i]=.09*(i+1);u.lambda.value[i]=-.07*(i+1);
    u.alpha.d[i]=.03*(i-1);B.value[i]=.04*(i+1);
    for(int j=0;j<3;++j){u.beta.d[j][i]=.02*(i+1)-.013*j;B.d[j][i]=.017*(i-j);}
   }
   const auto a=h::HyperbolicGauge(x,u,B,D);
   Check(a.valid,"shared transition invalid");
   const double W=h::Smoothstep9((r-.2)/.36),o=(1-r*r)/2;
   const double F=.71*.71+4*o*o*.71,mu=1.5*.43*F;
   for(int i=0;i<3;++i){
    double adv=0;for(int j=0;j<3;++j)adv+=u.beta.value[j]*u.beta.d[j][i];
    const double inner=u.lambda.value[i]-6*u.beta.value[i];
    Near(a.shift_regular[i],adv+(1-W)*inner+W*(-x[i]+mu*D[i]),"independent regular shift equation");
    Near(a.shift_pole[i],-4*W*(u.beta.value[i]+x[i]),"single W restoring pole");
    if(r<=.2)Check(a.shift_pole[i]==0,"inner plateau not exact");
   }
   // Changing restoration must affect only the shift pole, not lapse, B or mu.
   h::HyperbolicShiftParameters<double>off;off.restoring=0;
   const auto b=h::HyperbolicGauge(x,u,B,D,off);
   Check(a.lapse_regular==b.lapse_regular&&a.lapse_pole==b.lapse_pole,"lapse changed with restoration");
   for(int i=0;i<3;++i)Check(a.shift_regular[i]==b.shift_regular[i]&&a.flux_rate[i]==b.flux_rate[i],"other sources changed");
   if(r>=.9){
    h::VectorJet<double>zero{};
    const auto old=h::DonorHyperbolicGauge(x,u,zero,D);
    Check(a.lapse_regular==old.lapse_regular&&a.lapse_pole==old.lapse_pole,"near-scri lapse changed");
    for(int i=0;i<3;++i)Check(a.shift_regular[i]==old.shift_regular[i]&&a.shift_pole[i]==old.shift_pole[i],"near-scri shift/pole changed");
   }
  }
  std::cout<<"Single-W independent shift oracle, inner plateau and exact near-scri source identity passed\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';code=1;}
 Kokkos::finalize();return code;
}
