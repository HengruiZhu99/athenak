// Initial-data consistency diagnostic, not a stability test or RHS modification.
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include "z4c/hyperboloidal/layer_wormhole.hpp"
namespace h=z4c::hyperboloidal;

// Differentiate the metric evolution in ordinary Cartesian coordinates, then
// differentiate g^{jk} Gamma^i_jk. This path does not use the Lambda equation.
void MetricRate(const h::Z4cJet<double>&u,h::MetricJet<double>&rate) {
  double div=0,dd[3]{};
  for(int d=0;d<3;++d) {
    div+=u.beta.d[d][d];
    for(int l=0;l<3;++l) dd[l]+=u.beta.dd[l][d][d];
  }
  for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
    rate.g[i][j]=-2*u.alpha.value*u.a.k[i][j]-2./3*u.metric.g[i][j]*div;
    for(int l=0;l<3;++l)
      rate.dg[l][i][j]=-2*(u.alpha.d[l]*u.a.k[i][j]+u.alpha.value*u.a.dk[l][i][j])
          -2./3*(u.metric.dg[l][i][j]*div+u.metric.g[i][j]*dd[l]);
    for(int d=0;d<3;++d) {
      rate.g[i][j]+=u.beta.value[d]*u.metric.dg[d][i][j]
          +u.metric.g[d][j]*u.beta.d[i][d]+u.metric.g[i][d]*u.beta.d[j][d];
      for(int l=0;l<3;++l)
        rate.dg[l][i][j]+=u.beta.d[l][d]*u.metric.dg[d][i][j]
            +u.beta.value[d]*u.metric.ddg[l][d][i][j]
            +u.metric.dg[l][d][j]*u.beta.d[i][d]+u.metric.g[d][j]*u.beta.dd[l][i][d]
            +u.metric.dg[l][i][d]*u.beta.d[j][d]+u.metric.g[i][d]*u.beta.dd[l][j][d];
    }
  }
}
void ConnectionRate(const h::MetricJet<double>&metric,
                    const h::MetricJet<double>&rate,double result[3]) {
  const auto geometry=h::Geometry(metric);
  double inverse_rate[3][3]{};
  for(int i=0;i<3;++i) for(int j=0;j<3;++j)
  for(int a=0;a<3;++a) for(int b=0;b<3;++b)
    inverse_rate[i][j]-=geometry.inverse[i][a]*rate.g[a][b]*geometry.inverse[b][j];
  for(int i=0;i<3;++i) for(int j=0;j<3;++j) for(int k=0;k<3;++k) {
    result[i]+=inverse_rate[j][k]*geometry.connection[i][j][k];
    for(int a=0;a<3;++a)
      result[i]+=.5*geometry.inverse[j][k]*(
          inverse_rate[i][a]*(metric.dg[j][a][k]+metric.dg[k][a][j]-metric.dg[a][j][k])
          +geometry.inverse[i][a]*(rate.dg[j][a][k]+rate.dg[k][a][j]-rate.dg[a][j][k]));
  }
}

int main(int argc,char **argv) {
  Kokkos::initialize(argc,argv);
  int failures=0;
  {
    h::LayerReference<double> reference(1,1,{true,.35,.75});
    h::LayerWormhole<double> wormhole(reference,.5);
    std::cout<<std::setprecision(17)<<"mass,radius,direction,Omega,H,M,Z,metric_rate_error,connection_rate_error,scale\n";
    for(double mass:{0.,.5})
    for(double radius:{.05,.25,.36,.4,.5,.6,.7,.76,.9,.95,.99,.999,.9999,.99999})
    for(int direction=0;direction<3;++direction) {
      double n[3]={1,0,0};
      if(direction==1) for(double &v:n) v=1/std::sqrt(3.);
      if(direction==2) {n[0]=.8;n[1]=-.48;n[2]=.36;}
      const auto point=reference.At(radius*n[0],radius*n[1],radius*n[2]);
      const auto u=mass ? wormhole.At(radius*n[0],radius*n[1],radius*n[2]) : point.state;
      h::OmegaJet<double> omega{};omega.omega=point.omega;
      for(int a=0;a<3;++a) {
        omega.gradient[a]=point.domega[a];
        for(int b=0;b<3;++b) omega.hessian[a][b]=point.omega_hessian[a][b];
      }
      h::SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,omega);
      h::Z4cRHS<double> rhs{};
      if(!h::AssembleInterior(h::ConformalRHS(u,omega,5/u.alpha.value,0.),omega.omega,rhs))
        throw std::runtime_error("inadmissible initial RHS");
      h::MetricJet<double> rate{};MetricRate(u,rate);
      double gamma_rate[3]{},error=0,scale=1,metric_error=0;
      ConnectionRate(u.metric,rate,gamma_rate);
      for(int i=0;i<3;++i) {
        error=std::max(error,std::abs(gamma_rate[i]-rhs.lambda[i]));
        scale=std::max({scale,std::abs(gamma_rate[i]),std::abs(rhs.lambda[i])});
        for(int j=0;j<3;++j) metric_error=std::max(metric_error,std::abs(rate.g[i][j]-rhs.metric[i][j]));
      }
      const auto constraint=h::EvolvedConstraints(u,omega);
      if(!constraint.valid || error>1e-8*scale || metric_error>1e-12*scale) ++failures;
      std::cout<<mass<<","<<radius<<","<<direction<<","<<omega.omega<<","<<constraint.hamiltonian
          <<","<<std::sqrt(constraint.momentum_conformal_norm2)<<","<<std::sqrt(constraint.z4.z_conformal_norm2)
          <<","<<metric_error<<","<<error<<","<<scale<<"\n";
    }
  }
  Kokkos::finalize();return failures ? 1 : 0;
}
