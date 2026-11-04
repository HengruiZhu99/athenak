// Read-only native initial/reference profiles at supplied radii, no evolution.
#include <fstream>
#include <iomanip>
#include <iostream>
#include "refined_configuration.hpp"
#include "z4c/hyperboloidal/native_boosted_wormhole.hpp"
namespace h=z4c::hyperboloidal;
int main(int argc,char**argv){
 MPI_Init(&argc,&argv);Kokkos::initialize(argc,argv);int code=0;
 try{
  if(argc!=4)throw std::invalid_argument("config radii.txt fresh.csv");
  hyperbolic_reproduction::RefinedConfiguration cfg;cfg.Load(argv[1],hyperbolic_reproduction::ContinuousCoreMesh(16));
  if(cfg.boost_velocity!=0||!cfg.gauge.stationary_outer_reference)throw std::runtime_error("requires stationary reference config");
  h::NativeWormhole base(1./3.,2e-13,cfg.initial_start_M,cfg.initial_end_M,h::NativeWormhole::TransitionProfile::polynomial_c6);
  h::NativeBoostedWormhole native(0,1./3.,true,cfg.initial_start_M,cfg.initial_end_M);
  std::ifstream in(argv[2]);std::ifstream exists(argv[3]);if(!in||exists.good())throw std::runtime_error("missing input/existing output");
  std::ofstream out(argv[3]);out<<std::setprecision(17)<<"r,R_over_M,S,spacelike_margin,dr_dR,chi,alpha,beta_dev_r,K,metric_rr,dr_chi,dr_alpha,dr_beta_dev_r,derivative_step_error\n";
  auto fields=[&](double r){Real x[3]={r,0,0};auto q=native.Fields(x);cfg.InitialFields(x,q);return std::array<double,6>{q[0],q[18]+(1+r*r)/2,q[22],q[13]-3,q[1]+1,q[7]};};
  double r;
  while(in>>r){
   if(!(r>.3&&r<.999))throw std::runtime_error("outside qualified exterior interval");
   auto q=fields(r);auto v=base.Point(r);const double old=(1-r*r)/2,S=-v.trace/3+old*v.ar/2,f=1-2/(3*v.areal),a2=f+v.areal*v.areal*S*S;
   if(!(f>0&&a2>0))throw std::runtime_error("invalid reference exterior slice");
   std::array<double,6> deriv{};double err=0;
   for(double step:{1e-5,5e-6}){auto a=fields(r-2*step),b=fields(r-step),c=fields(r+step),d=fields(r+2*step);
    for(int k=0;k<6;++k){double next=(a[k]-8*b[k]+8*c[k]-d[k])/(12*step);if(step==5e-6)err=std::max(err,std::abs(next-deriv[k])/(1+std::abs(next)));deriv[k]=next;}}
   out<<r<<','<<3*v.areal<<','<<S<<','<<f/a2<<','<<r/(v.areal*std::sqrt(a2));
   for(int k=0;k<5;++k)out<<','<<q[k];for(int k=0;k<3;++k)out<<','<<deriv[k];out<<','<<err<<'\n';
  }
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';code=2;}
 Kokkos::finalize();MPI_Finalize();return code;
}
