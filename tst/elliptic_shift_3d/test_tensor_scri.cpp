// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include "z4c/hyperboloidal/tensor_scri.hpp"
#include "z4c/hyperboloidal/reference_gauge.hpp"
#include "z4c/hyperboloidal/spherical_tensor.hpp"
namespace h = z4c::hyperboloidal;
using U = h::Z4cJet<double>;
h::OmegaJet<double> Omega(const double x[3], const U &u) {
  h::OmegaJet<double> o{};
  o.omega=(1-x[0]*x[0]-x[1]*x[1]-x[2]*x[2])/2;
  for(int i=0;i<3;++i){o.gradient[i]=-x[i];o.hessian[i][i]=-1;}
  h::SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,o);return o;
}
U Shift(const U &a,const double direction[3],double e) {
  U b=a;
  auto scalar=[&](const auto &s,auto &t){
    for(int i=0;i<3;++i){t.value+=e*direction[i]*s.d[i];
      for(int j=0;j<3;++j){t.value+=e*e*direction[i]*direction[j]*s.dd[i][j]/2;t.d[i]+=e*direction[j]*s.dd[j][i];}}
  };
  scalar(a.alpha,b.alpha);scalar(a.chi,b.chi);scalar(a.trace,b.trace);scalar(a.theta,b.theta);
  for(int i=0;i<3;++i)for(int d=0;d<3;++d){
    b.beta.value[i]+=e*direction[d]*a.beta.d[d][i];b.lambda.value[i]+=e*direction[d]*a.lambda.d[d][i];
    for(int j=0;j<3;++j){
      b.a.k[i][j]+=e*direction[d]*a.a.dk[d][i][j];
      b.metric.g[i][j]+=e*direction[d]*a.metric.dg[d][i][j];
      for(int f=0;f<3;++f){b.metric.g[i][j]+=e*e*direction[d]*direction[f]*a.metric.ddg[d][f][i][j]/2;
        b.metric.dg[d][i][j]+=e*direction[f]*a.metric.ddg[f][d][i][j];}
    }
  }return b;
}
h::TensorPole<double> Native(const U &u,const double x[3],bool appendix) {
  const auto o=Omega(x,u);auto parts=h::ConformalRHS(u,o,4.,-.99);
  if(appendix)parts.pole.theta+=3*u.alpha.value*o.normal*u.theta.value;
  const h::CMCReference<double> ref{1,1};
  const auto gauge=h::UnfactoredReferenceGauge(ref,ref.At(x[0],x[1],x[2]),u,h::GaugeParameters<double>{1,0,1,0},false,true);
  if(!parts.valid||!gauge.valid)throw std::runtime_error("invalid native pole test");
  return {parts.pole,gauge.pole.alpha};
}
template<typename T> auto Pack(const h::TensorPole<T> &p){
  std::array<T,19> out{};out[0]=p.geometry.chi;out[13]=p.geometry.trace;out[14]=p.geometry.theta;out[18]=p.lapse;
  int t=1;for(int i=0;i<3;++i)for(int j=i;j<3;++j){out[t]=p.geometry.metric[i][j];out[t+6]=p.geometry.a[i][j];++t;}
  for(int i=0;i<3;++i)out[15+i]=p.geometry.lambda[i];return out;
}
int main(int argc,char **argv){
  Kokkos::initialize(argc,argv);int status=0;
  try{
    std::mt19937 random(74191);std::uniform_real_distribution<double> sample(-.03,.03);
    double value_error=0,derivative_error=0;
    for(int trial=0;trial<32;++trial){
      U u{};u.alpha.value=.9;u.chi.value=1.1;u.trace.value=-3;
      for(int i=0;i<3;++i){u.metric.g[i][i]=1;u.beta.value[i]=sample(random);u.lambda.value[i]=sample(random);
        for(int j=i;j<3;++j){u.metric.g[i][j]+=sample(random);u.metric.g[j][i]=u.metric.g[i][j];
          u.a.k[i][j]=u.a.k[j][i]=sample(random);
          for(int d=0;d<3;++d){u.metric.dg[d][i][j]=u.metric.dg[d][j][i]=sample(random);
            u.a.dk[d][i][j]=u.a.dk[d][j][i]=sample(random);
            for(int f=d;f<3;++f){double a=sample(random);u.metric.ddg[d][f][i][j]=u.metric.ddg[f][d][i][j]=a;
              u.metric.ddg[d][f][j][i]=u.metric.ddg[f][d][j][i]=a;}}}
        for(int d=0;d<3;++d){u.beta.d[d][i]=sample(random);u.lambda.d[d][i]=sample(random);}
      }
      for(auto *s:{&u.chi,&u.alpha,&u.trace,&u.theta}){
        if(s==&u.theta)s->value=sample(random);
        for(int d=0;d<3;++d){s->d[d]=sample(random);for(int f=d;f<3;++f)s->dd[d][f]=s->dd[f][d]=sample(random);}}
      const double x[3]={.6,0,.8},direction[3]={.6,0,.8};const auto o=Omega(x,u);
      for(bool appendix:{false,true}){
        const auto automatic=Pack(h::TensorPoleNormalJet(u,o,x,direction,4.,-.99,1.,1.,appendix));
        const auto native=Pack(Native(u,x,appendix));
        constexpr double e=2e-5;double xp[3],xm[3];for(int i=0;i<3;++i){xp[i]=x[i]+e*direction[i];xm[i]=x[i]-e*direction[i];}
        const auto plus=Pack(Native(Shift(u,direction,e),xp,appendix));
        const auto minus=Pack(Native(Shift(u,direction,-e),xm,appendix));
        for(size_t f=0;f<native.size();++f){value_error=std::max(value_error,std::abs(automatic[f].value-native[f]));
          derivative_error=std::max(derivative_error,std::abs(automatic[f].derivative-(plus[f]-minus[f])/(2*e)));}
      }
    }
    if(value_error>2e-13||derivative_error>2e-8){std::cerr<<std::setprecision(17)<<"value_error="<<value_error<<" derivative_error="<<derivative_error<<"\n";throw std::runtime_error("analytic tensor pole AD audit failed");}
    std::cout<<std::setprecision(17)<<"{\"native_value_error\":"<<value_error<<",\"independent_FD_derivative_error\":"<<derivative_error<<",\"radial_cases\":[";
    const h::CMCReference<double> ref{1,1};
    for(int trial=0;trial<16;++trial){
      double q[8]{},d[8]{},dd[8]{};for(int f:{0,2,3,4,5,6}){q[f]=sample(random);d[f]=sample(random);dd[f]=sample(random);}q[6]=0;
      auto u=h::SphericalJet(1.,q,d,dd,ref);const double beta=-std::sqrt(u.chi.value),v=beta+1;
      u.beta=h::RadialVector(beta,beta+1.5*u.alpha.value*q[2],1.5*((1+d[6])*q[2]+u.alpha.value*d[2]+u.alpha.value*q[2]),1.);
      const double x[3]={1,0,0},normal[3]={1,0,0};auto o=Omega(x,u);o.omega=0;
      auto parts=h::ConformalRHS(u,o,4.,-.99);parts.pole.theta+=3*u.alpha.value*o.normal*u.theta.value;
      auto pole=h::TensorPoleNormalJet(u,o,x,normal,4.,-.99,1.,1.);
      const auto limit=h::TensorScriLimit(parts.regular,pole,-1.);
      const auto gauge=h::UnfactoredReferenceGauge(ref,ref.At(1.,0.,0.),u,h::GaugeParameters<double>{1,0,1,0},false,true);
      double raw[8]={limit.chi,limit.metric[0][0],limit.a[0][0],limit.trace,limit.theta,limit.lambda[0],gauge.regular.alpha-pole.lapse.derivative,0};
      std::cout<<(trial?",":"")<<"{\"q\":[";for(int f=0;f<8;++f)std::cout<<(f?",":"")<<q[f];
      std::cout<<"],\"d\":[";for(int f=0;f<8;++f)std::cout<<(f?",":"")<<d[f];
      std::cout<<"],\"dd\":[";for(int f=0;f<8;++f)std::cout<<(f?",":"")<<dd[f];
      std::cout<<"],\"shift_deviation\":"<<v<<",\"limit\":[";for(int f=0;f<8;++f)std::cout<<(f?",":"")<<raw[f];std::cout<<"]}";
    }std::cout<<"]}\n";
  }catch(const std::exception &e){std::cerr<<e.what()<<"\n";status=1;}
  Kokkos::finalize();return status;
}
