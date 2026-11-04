#include <cmath>
#include <iostream>
#include <stdexcept>
#include "athena.hpp"
#include "z4c/hyperboloidal/hyperbolic_shift.hpp"
#include "z4c/hyperboloidal/native_wormhole.hpp"
namespace h=z4c::hyperboloidal;
int main(int argc,char**argv){
 Kokkos::initialize(argc,argv);int code=0;
 try {
  h::NativeWormhole initial(1./3.,2e-13,4,12,h::NativeWormhole::TransitionProfile::polynomial_c6);
  const double k=initial.IsotropicScale();
  if(std::abs(k-h::normalized_inner_spatial_scale)>2e-13)throw std::runtime_error("initial spatial map differs from driver normalization");
  Kokkos::View<double*> errors("errors",33);
  Kokkos::parallel_for("nontrivial_normalized_driver",33,KOKKOS_LAMBDA(int a){
   const double r=a/32.,x[3]={.6*r,.8*r,0},o=(1-r*r)/2;
   h::Z4cJet<double>u{};h::VectorJet<double>B{};double D[3]={.3,-.2,.1};
   u.alpha.value=.7;u.chi.value=.12;u.trace.value=.31;
   u.metric.g[0][0]=2.;u.metric.g[1][1]=.52;u.metric.g[2][2]=1.;
   u.metric.g[0][1]=u.metric.g[1][0]=.2;
   for(int i=0;i<3;++i){u.beta.value[i]=.1*(i+1);u.lambda.value[i]=.2*(i-1);B.value[i]=.03*(i+1);u.alpha.d[i]=.13-.07*i;
    for(int j=0;j<3;++j){u.beta.d[j][i]=.05*(i+1)-.03*j;B.d[j][i]=.11*(j+1)-.08*i;}}
   const auto result=h::HyperbolicGauge(x,u,B,D),outer=h::OuterHyperbolicGauge(x,u,B,D);
   const double s=h::InnerTelegrapherWeight(r);double error=result.valid&&outer.valid?0:1;
   for(int i=0;i<3;++i){
    // Pull back a unit-coefficient Cartesian driver, without using shift_gamma.
    const double beta_lab=u.beta.value[i]/k,lambda_lab=k*u.lambda.value[i];
    double shift=k*(lambda_lab-6*beta_lab);
    for(int j=0;j<3;++j)shift+=u.beta.value[j]*u.beta.d[j][i];
    error=fmax(error,fabs(result.shift_regular[i]-((1-s)*outer.shift_regular[i]+s*shift)));
    error=fmax(error,fabs(result.shift_pole[i]-(1-s)*outer.shift_pole[i]));
    if(s==0){error=fmax(error,fabs(result.flux_rate[i]-outer.flux_rate[i]));}
   }
   const auto c=h::ShiftCoefficients(r,u.alpha.value,u.chi.value);
   error=fmax(error,fabs(h::GaugeShiftSpeedSquared(c,1.7)-(4./3.)*((1-s)*c.mu+s*k*k)*1.7));
   if(s==0){error=fmax(error,fabs(result.lapse_regular-outer.lapse_regular));error=fmax(error,fabs(result.lapse_pole-outer.lapse_pole));}
   errors(a)=error;
  });
  auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),errors);
  double maxerr=0;for(int i=0;i<33;++i)maxerr=std::max(maxerr,host(i));
  if(maxerr>2e-12)throw std::runtime_error("coordinate-pulled-back driver, blend or outer closure differs");
  std::cout.precision(17);std::cout<<"{\"passed\":true,\"k\":"<<k<<",\"gamma\":"<<h::InnerTelegrapher::shift_gamma<<",\"max_error\":"<<maxerr<<"}\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';code=1;}
 Kokkos::finalize();return code;
}
