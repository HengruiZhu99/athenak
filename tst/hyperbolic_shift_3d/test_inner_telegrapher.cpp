#include <cmath>
#include <iostream>
#include <stdexcept>
#include "athena.hpp"
#include "z4c/hyperboloidal/hyperbolic_shift.hpp"
namespace h=z4c::hyperboloidal;
static void Check(bool yes,const char*what){if(!yes)throw std::runtime_error(what);}
static void Near(double a,double b,const char*what){Check(std::abs(a-b)<2e-12*(1+std::abs(b)),what);}
int main(int argc,char**argv){
 Kokkos::initialize(argc,argv);int code=0;
 try {
  // Independent physical-variable oracle, including nonradial shift, gradients,
  // a nonflat metric and chi below the branch's coefficient guard.
  for(double r:{.012,.15,.2,.25,.38,.55,.56,.75,.93,1.})
  for(double chi_phys:{2e-7,.12}) {
   double x[3]={.6*r,.8*r,0},o=(1-r*r)/2;
   h::Z4cJet<double> u{};h::VectorJet<double>B{};double D[3]={.3,-.2,.1};
   const double a=.37;
   u.alpha.value=o*a;u.chi.value=o>0?chi_phys/(o*o):1.;u.trace.value=.7;u.theta.value=.023;
   if(o==0)u.alpha.value=.7;
   double g[3]={2.,.52,1.};
   double gi[3][3]={{.52,-.2,0},{-.2,2.,0},{0,0,1.}};
   u.metric.g[0][1]=u.metric.g[1][0]=.2;
   for(int i=0;i<3;++i){
    u.metric.g[i][i]=g[i];u.beta.value[i]=.1*(i+1);u.lambda.value[i]=.2*(i-1);
    u.alpha.d[i]=o*(.13-.07*i)-a*x[i];B.value[i]=.03*(i+1);
    for(int j=0;j<3;++j){u.beta.d[j][i]=.05*(i+1)-.03*j;B.d[j][i]=.11*(j+1)-.08*i;}
   }
   const auto old=h::DonorHyperbolicGauge(x,u,B,D),now=h::HyperbolicGauge(x,u,B,D);
   Check(old.valid&&now.valid,"valid point rejected");
   const double s=h::InnerTelegrapherWeight(r);
   if(r>=.56){
    Check(s==0&&old.lapse_regular==now.lapse_regular&&old.lapse_pole==now.lapse_pole,"outer lapse changed");
    for(int i=0;i<3;++i)Check(old.shift_regular[i]==now.shift_regular[i]&&old.shift_pole[i]==now.shift_pole[i]&&old.flux_rate[i]==now.flux_rate[i],"outer gauge changed");
    continue;
   }
   Check((r>.2)||(s==1),"inner plateau not exact");
   double physical_rate=-2*a*u.trace.value;
   for(int i=0;i<3;++i){physical_rate+=u.beta.value[i]*(.13-.07*i);for(int j=0;j<3;++j)physical_rate+=std::max(chi_phys,1e-5)*gi[i][j]*B.d[i][j];}
   const double expected_lapse=o*physical_rate; // partial_t Omega=0
   Near(now.lapse_regular+now.lapse_pole/o,(1-s)*(old.lapse_regular+old.lapse_pole/o)+s*expected_lapse,"physical lapse conversion");
   for(int i=0;i<3;++i){
    double flux=(.2/3*(.13-.07*i)-B.value[i])/(.1/3);
    double shift=u.lambda.value[i]-6*u.beta.value[i];
    for(int j=0;j<3;++j){flux+=u.beta.value[j]*B.d[j][i];shift+=u.beta.value[j]*u.beta.d[j][i];}
    Near(now.flux_rate[i],(1-s)*old.flux_rate[i]+s*flux,"physical B rate");
    Near(now.shift_regular[i]+now.shift_pole[i]/o,(1-s)*(old.shift_regular[i]+old.shift_pole[i]/o)+s*shift,"physical shift rate");
   }
   // Isolate damping additions from the geometric source. Omega.normal=0.
   h::OmegaJet<double>omega{};omega.omega=o;
   h::Z4cRHSParts<double> parts{};parts.valid=true;
   parts.pole.trace=u.alpha.value*4*1.99*u.theta.value;
   parts.pole.theta=-u.alpha.value*4*1.01*u.theta.value;
   for(int i=0;i<3;++i)parts.pole.lambda[i]=-u.alpha.value*4*u.lambda.value[i];
   Check(h::HyperbolicDamping(x,u,omega,parts),"damping rejected");
   const double A=(1+r*r)/2;
   Near(parts.pole.trace/o,((1-s)*8*A*1.99/o+s*a*.03)*u.theta.value,"trace damping");
   Near(parts.pole.theta/o,-((1-s)*8*A*1.01/o+s*a*.06)*u.theta.value,"theta damping");
   for(int i=0;i<3;++i)Near(parts.pole.lambda[i]/o,-((1-s)*8*A/o+s*a*.06)*u.lambda.value[i],"connection damping");
   auto c=h::ShiftCoefficients(r,u.alpha.value,u.chi.value);
   const double Q=1.7,E=1.2;
   Check(std::isfinite(h::GaugeLapseSpeedSquared(u.alpha.value,u.chi.value,c,Q,E)),"invalid lapse speed estimate");
   if(s==1){Near(h::GaugeLapseSpeedSquared(u.alpha.value,u.chi.value,c,Q,E),(2*a*chi_phys+2*std::max(chi_phys,1e-5))*Q,"inner lapse speed");Near(h::GaugeShiftSpeedSquared(c,Q),4*Q/3,"inner shift speed");}
  }
  double previous=1;
  for(int i=0;i<=1000;++i){double s=h::InnerTelegrapherWeight(.2+.36*i/1000.);Check(s>=0&&s<=previous,"nonmonotone weight");previous=s;}
  std::cout<<"Physical gauge oracle, guarded chi, damping, blend, outer/scri identity and speed estimates passed\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';code=1;}
 Kokkos::finalize();return code;
}
