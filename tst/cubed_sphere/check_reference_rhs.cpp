// Diagnostic only: raw analytic stationary-Minkowski RHS before subtraction.
// Samples do not establish an error bound or validate a perturbed evolution.
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include "mesh/cubed_sphere/map.hpp"
#include "z4c/hyperboloidal/layer_wormhole.hpp"
namespace h=z4c::hyperboloidal;

template<class T> void Evaluate(T a,double begin,double end,T damping,const double x[3],T out[5]) {
  h::LayerReference<T> ref(T(1),a,{true,begin,end});
  const auto p=ref.At(T(x[0]),T(x[1]),T(x[2]));
  h::OmegaJet<T> omega{};omega.omega=p.omega;
  for(int i=0;i<3;++i) {
    omega.gradient[i]=p.domega[i];
    for(int j=0;j<3;++j) omega.hessian[i][j]=p.omega_hessian[i][j];
  }
  const auto &u=p.state;
  h::SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,omega);
  h::Z4cRHS<T> rhs{};
  if(!h::AssembleInterior(h::ConformalRHS(u,omega,damping,T(0)),omega.omega,rhs))
    throw std::runtime_error("inadmissible reference RHS");
  out[0]=std::abs(rhs.chi);out[1]=std::abs(rhs.trace);out[2]=std::abs(rhs.theta);
  out[3]=0;out[4]=0;
  for(int i=0;i<3;++i) {
    out[3]=std::max(out[3],std::abs(rhs.lambda[i]));
    for(int j=0;j<3;++j)
      out[4]=std::max({out[4],std::abs(rhs.metric[i][j]),std::abs(rhs.a[i][j])});
  }
}

int main(int argc,char **argv) {
  Kokkos::initialize(argc,argv);
  std::cerr<<"double digits="<<std::numeric_limits<double>::digits
           <<" long double digits="<<std::numeric_limits<long double>::digits<<"\n";
  std::cout<<std::setprecision(21)
      <<"family,point,x,y,z,chi_double,trace_double,theta_double,lambda_double,metric_A_double,chi_long,trace_long,theta_long,lambda_long,metric_A_long\n";
  for(int family=0;family<2;++family) {
    const double a=family ? 1.:.5,begin=family ? .35:.05,end=family ? .75:.95;
    const double damping=family ? 5.:10.;int point=0;
    auto check=[&](const double x[3]) {
      double d[5];long double l[5];
      Evaluate(a,begin,end,damping,x,d);
      Evaluate<long double>(a,begin,end,damping,x,l);
      std::cout<<(family ? "puncture_reference" : "pulse_reference")<<","<<point++;
      for(int i=0;i<3;++i) std::cout<<","<<x[i];
      for(double v:d) std::cout<<","<<v;
      for(long double v:l) std::cout<<","<<v;
      std::cout<<"\n";
    };
    // Predetermined radial/directional sampling through layer joins and collar.
    for(int k=1;k<1000;++k) for(int direction=0;direction<3;++direction) {
      double r=k/1000.,n[3]={1.,0.,0.};
      if(direction==1) for(double &v:n) v=1/std::sqrt(3.);
      if(direction==2) {n[0]=.8;n[1]=-.48;n[2]=.36;}
      double x[3];for(int i=0;i<3;++i) x[i]=r*n[i];check(x);
    }
    // Actual outermost active shell cells for current baseline/fine puncture.
    const int nr=family ? 205:48,na=family ? 30:12;
    cubed_sphere::Map map{family ? .25245594360687235:.25,1.,family ? 0.:2.8592697712828787};
    for(int patch=0;patch<6;++patch) for(int j=0;j<na;++j) for(int k=0;k<na;++k) {
      double q[3]={1.-.5/nr,-1.+(j+.5)*2./na,-1.+(k+.5)*2./na},x[3];
      map.Forward(patch,q,x);check(x);
    }
  }
  Kokkos::finalize();
}
