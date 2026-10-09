#include "live_damping_profile.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
namespace hyp=z4c::hyperboloidal;
hyp::ResearchDampingJet<double> Point(const hyp::LayerReference<double>&ref,const double x[3]){
 auto p=ref.At(x[0],x[1],x[2]);auto u=p.state;hyp::OmegaJet<double>o{};o.omega=p.omega;
 for(int i=0;i<3;++i){o.gradient[i]=p.domega[i];for(int j=0;j<3;++j)o.hessian[i][j]=p.omega_hessian[i][j];}
 for(int i=0;i<3;++i){u.beta.value[i]+=.01*std::sin((i+1)*x[0])+.02*x[1]*x[2];u.beta.d[0][i]+=.01*(i+1)*std::cos((i+1)*x[0]);u.beta.d[1][i]+=.02*x[2];u.beta.d[2][i]+=.02*x[1];}
 double n[3]{};if(p.radius>0)for(int i=0;i<3;++i)n[i]=x[i]/p.radius;
 return hyp::ResearchLiveKappa2SpatialJet(u,o,p.radius,n,10.);
}
int main(){double maximum=0,exact_inner=0;int rows=0;
 for(double a:{.5,.75,1.,2.}){hyp::LayerReference<double>ref(1,a,{true,.05,.95});
 for(double r:{0.,.1,.15,.18,.2,.225,.25,.28,.3,.5,.75,.95,.98})for(bool oblique:{false,true}){
 const double n[3]={oblique?.36:1,oblique?-.48:0,oblique?.8:0};double x[3];for(int i=0;i<3;++i)x[i]=r*n[i];const auto base=Point(ref,x);
 for(int i=0;i<3;++i){const double h=1e-5;double f[4];const int offsets[4]={-2,-1,1,2};for(int j=0;j<4;++j){double y[3]={x[0],x[1],x[2]};y[i]+=offsets[j]*h;f[j]=Point(ref,y).value;}
 const double df=(f[0]-8*f[1]+8*f[2]-f[3])/(12*h);maximum=std::max(maximum,std::abs(df-base.d[i])/(1+std::abs(df)+std::abs(base.d[i])));}
 if(r<=.15){exact_inner=std::max(exact_inner,std::abs(base.value));for(int i=0;i<3;++i)exact_inner=std::max(exact_inner,std::abs(base.d[i]));}++rows;
 }}std::cout<<std::setprecision(17)<<"{\"rows\":"<<rows<<",\"full_gradient_relative_FD_error\":"<<maximum<<",\"inner_value_and_gradient_max\":"<<exact_inner<<"}\n";
}
