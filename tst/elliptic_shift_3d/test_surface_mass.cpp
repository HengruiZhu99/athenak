// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <iostream>
#include "z4c/hyperboloidal/surface_mass.hpp"
#include "z4c/hyperboloidal/athenak_bridge.hpp"
namespace h=z4c::hyperboloidal;
void Require(bool yes,const char*reason){if(!yes)throw std::runtime_error(reason);}
int main(int argc,char**argv){Kokkos::initialize(argc,argv);int status=0;try{
  const Real pi=std::acos(-1.);Real mass_error=0,adm_error=0;size_t checks=0;
  for(Real r:{.08,.12,.2,.4,.7,.9})for(auto direction:{std::array<Real,3>{1,0,0},{.6,.8,0},{2./3,1./3,2./3}}){
    Real x[3];for(int i=0;i<3;++i)x[i]=r*direction[i];
    for(bool schwarzschild:{false,true}){h::Z4cJet<Real>u{};h::OmegaJet<Real>o{};
      for(int i=0;i<3;++i)u.metric.g[i][i]=1;u.alpha.value=1;u.chi.value=1;
      if(schwarzschild){const Real m=1./3,psi=1+m/(2*r);u.chi.value=std::pow(psi,-4);o.omega=1;
        for(int i=0;i<3;++i)u.chi.d[i]=2*m*std::pow(psi,-5)*direction[i]/(r*r);
      }else{o.omega=(1-r*r)/2;for(int i=0;i<3;++i)o.gradient[i]=-x[i];u.trace.value=-3;}
      const auto local=h::CoordinateSphereExpansion(u,o,x);Require(local.valid,"exact sphere expansion invalid");
      const Real area=4*pi*local.area_per_solid_angle;const auto mass=h::HawkingSurfaceMass(area,area*local.product);
      Require(mass.valid,"exact sphere mass invalid");mass_error=std::max(mass_error,std::abs(mass.mass-(schwarzschild?Real(1./3):Real(0))));++checks;
    }
  }
  // Independent full-physical ADM connection, including nonflat/off-diagonal
  // conformal metric jets, non-tracefree A and nonzero independent Theta.
  for(int sample=0;sample<40;++sample){h::Z4cJet<Real>u{};h::OmegaJet<Real>o{};const Real x[3]={.2+.001*sample,.13,.17};
    Real radius=0;for(Real v:x)radius+=v*v;radius=std::sqrt(radius);u.alpha.value=1;u.chi.value=.8;o.omega=.6;
    u.trace.value=-2.8;u.theta.value=.003;
    for(int i=0;i<3;++i){u.chi.d[i]=.03*(i+1);o.gradient[i]=-.2*(i+1);
      for(int j=0;j<3;++j){u.metric.g[i][j]=(i==j?1+.1*i: .02);u.a.k[i][j]=.01*(i+j+1);
        for(int k=0;k<3;++k)u.metric.dg[k][i][j]=.001*(k+1)*(i+j+1);}}
    const auto local=h::CoordinateSphereExpansion(u,o,x);Require(local.valid,"nonflat sphere expansion invalid");
    const auto adm=h::ToPhysicalADM(u,o.omega);Require(adm.valid,"independent ADM invalid");
    h::MetricJet<Real>physical{};for(int i=0;i<3;++i)for(int j=0;j<3;++j){physical.g[i][j]=adm.metric[i][j];
      for(int k=0;k<3;++k)physical.dg[k][i][j]=adm.psi4*(u.metric.dg[k][i][j]-u.metric.g[i][j]*(u.chi.d[k]/u.chi.value+2*o.gradient[k]/o.omega));}
    const auto geo=h::Geometry(physical);Require(geo.valid,"physical spatial geometry invalid");
    Real n[3],up[3]{},q=0,k=0,ks=0,trace=0;for(int i=0;i<3;++i)n[i]=x[i]/radius;
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)up[i]+=geo.inverse[i][j]*n[j];
    for(int i=0;i<3;++i)q+=n[i]*up[i];
    for(int i=0;i<3;++i)for(int j=0;j<3;++j){Real dn=((i==j?1.:0)-n[i]*n[j])/radius;
      for(int a=0;a<3;++a)dn-=geo.connection[a][i][j]*n[a];k+=(geo.inverse[i][j]-up[i]*up[j]/q)*dn/std::sqrt(q);
      ks+=adm.curvature[i][j]*up[i]*up[j]/q;trace+=geo.inverse[i][j]*adm.curvature[i][j];}
    adm_error=std::max({adm_error,std::abs(local.mean_curvature-k),std::abs(local.curvature_trace-(ks-trace)),
      std::abs(local.area_per_solid_angle-radius*radius*std::sqrt(geo.determinant*q))});++checks;
  }
  h::Z4cJet<Real>u{};for(int i=0;i<3;++i)u.metric.g[i][i]=1;u.chi.value=u.alpha.value=1;h::OmegaJet<Real>o{};o.omega=1;
  const Real center[3]={0,0,0},point[3]={.2,0,0};Require(!h::CoordinateSphereExpansion(u,o,center).valid,"puncture/center was floored");
  o.omega=0;Require(!h::CoordinateSphereExpansion(u,o,point).valid,"finite-area diagnostic used at scri");
  Require(!h::HawkingSurfaceMass(Real(0),Real(0)).valid,"zero surface area accepted");
  Require(mass_error<1e-10&&adm_error<1e-12,"independent mass/ADM normalization check failed");
  std::cout<<"{\"exact_and_nonflat_checks\":"<<checks<<",\"spherical_mass_error_max\":"<<mass_error<<",\"independent_ADM_error_max\":"<<adm_error
    <<",\"generic_3D_is_Misner_Sharp\":false,\"coupled_mass_validation\":false,\"status\":\"passed\"}\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';status=1;}Kokkos::finalize();return status;}
