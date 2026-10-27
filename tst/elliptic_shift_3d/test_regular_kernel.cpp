// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include "athena.hpp"
#include "legacy_conformal_rhs.hpp"
namespace z4c { namespace hyperboloidal {
// Exact old constraint function from f37942825, retained only as an independent
// arithmetic path for ordinary-range equivalence. Never used by evolution.
template <typename T>
KOKKOS_INLINE_FUNCTION
EvolvedConstraintResult<T> LegacyEvolvedConstraints(const Z4cJet<T> &u, const OmegaJet<T> &o) {
  EvolvedConstraintResult<T> out{};
  out.valid = u.chi.value > 0 && Kokkos::isfinite(u.chi.value);
  if (!out.valid) return out;
  const auto gt = Geometry(u.metric);
  const auto b = PenroseMetric(u.metric, u.chi);
  const auto gb = Geometry(b);
  out.valid = gt.valid && gb.valid;
  if (!out.valid) return out;
  out.z4 = Z4Constraints(u.metric, u.chi.value, u.a.k, u.lambda.value,
                         u.theta.value, o.omega);
  out.physical_trace = u.trace.value+2*u.theta.value;
  T gradient2 = 0, lapomega = 0, norm_a = 0;
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) {
    gradient2 += gb.inverse[i][j]*o.gradient[i]*o.gradient[j];
    T hessian = o.hessian[i][j];
    for (int k = 0; k < 3; ++k) hessian -= gb.connection[k][i][j]*o.gradient[k];
    lapomega += gb.inverse[i][j]*hessian;
    for (int k = 0; k < 3; ++k)
    for (int l = 0; l < 3; ++l) {
      norm_a += gt.inverse[i][k]*gt.inverse[j][l]*u.a.k[i][j]*u.a.k[k][l];
    }
  }
  out.null_residual = gradient2-o.normal*o.normal;
  out.hamiltonian = o.omega*o.omega*(gb.scalar-norm_a)
      +T(2./3.)*out.physical_trace*out.physical_trace+4*o.omega*lapomega-6*gradient2;
  for (int i = 0; i < 3; ++i) {
    T divergence = 0, contraction = 0;
    for (int j = 0; j < 3; ++j)
    for (int k = 0; k < 3; ++k) {
      T derivative = u.a.dk[j][k][i];
      for (int l = 0; l < 3; ++l) {
        derivative -= gt.connection[l][j][k]*u.a.k[l][i]
                     +gt.connection[l][j][i]*u.a.k[k][l];
      }
      divergence += gt.inverse[j][k]*(derivative
          -T(1.5)*u.a.k[k][i]*u.chi.d[j]/u.chi.value);
      contraction += gt.inverse[j][k]*u.a.k[k][i]*o.gradient[j];
    }
    out.momentum[i] = o.omega*divergence-T(2./3.)*(u.trace.d[i]+2*u.theta.d[i])
                     -2*contraction;
    out.valid = out.valid && Kokkos::isfinite(out.momentum[i]);
  }
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) {
    out.momentum_conformal_norm2 += gb.inverse[i][j]*out.momentum[i]*out.momentum[j];
  }
  out.momentum_physical_norm2 = o.omega*o.omega*out.momentum_conformal_norm2;
  out.valid = out.valid && out.z4.valid && Kokkos::isfinite(out.hamiltonian)
      && Kokkos::isfinite(out.null_residual)
      && Kokkos::isfinite(out.momentum_conformal_norm2)
      && Kokkos::isfinite(out.momentum_physical_norm2);
  return out;
}

}}  // namespace z4c::hyperboloidal

namespace h=z4c::hyperboloidal;
void Require(bool test,const char *message){if(!test)throw std::runtime_error(message);}
void Compare(Real a,Real b,Real &worst){Require(std::isfinite(a)&&std::isfinite(b),"nonfinite normal-range comparison");
  worst=std::max(worst,std::abs(a-b)/std::max({Real(1),std::abs(a),std::abs(b)}));}
void CompareRHS(const h::Z4cRHS<Real> &a,const h::Z4cRHS<Real> &b,Real &worst){
  Compare(a.chi,b.chi,worst);Compare(a.trace,b.trace,worst);Compare(a.theta,b.theta,worst);
  for(int i=0;i<3;++i){Compare(a.lambda[i],b.lambda[i],worst);for(int j=0;j<3;++j){Compare(a.metric[i][j],b.metric[i][j],worst);Compare(a.a[i][j],b.a[i][j],worst);}}}
void CompareConstraints(const h::EvolvedConstraintResult<Real> &a,const h::EvolvedConstraintResult<Real> &b,Real &worst){
  Require(a.valid&&b.valid&&a.z4.valid&&b.z4.valid,"normal constraint path rejected");
  Compare(a.hamiltonian,b.hamiltonian,worst);Compare(a.physical_trace,b.physical_trace,worst);Compare(a.null_residual,b.null_residual,worst);
  Compare(a.momentum_conformal_norm2,b.momentum_conformal_norm2,worst);Compare(a.momentum_physical_norm2,b.momentum_physical_norm2,worst);
  Compare(a.z4.determinant_residual,b.z4.determinant_residual,worst);Compare(a.z4.tracefree_residual,b.z4.tracefree_residual,worst);
  Compare(a.z4.theta_physical,b.z4.theta_physical,worst);Compare(a.z4.z_conformal_norm2,b.z4.z_conformal_norm2,worst);Compare(a.z4.z_physical_norm2,b.z4.z_physical_norm2,worst);
  for(int i=0;i<3;++i){Compare(a.momentum[i],b.momentum[i],worst);Compare(a.z4.z_covector[i],b.z4.z_covector[i],worst);}
}

int main(int argc,char **argv){Kokkos::initialize(argc,argv);int result=0;
try{
  std::mt19937_64 generator(319712);std::uniform_real_distribution<Real> random(-1,1);
  Real regular_error=0,pole_error=0,assembled_error=0,constraint_error=0,selected_error=0;
  Real det_error=0,trace_a=0,connection_error=0,ricci_max=0;int cases=0;
  for(int sample=0;sample<128;++sample){
    h::Z4cJet<Real> u{};Real matrix[3][3];for(auto &row:matrix)for(auto &v:row)v=.3*random(generator);
    for(int i=0;i<3;++i)for(int j=0;j<3;++j){u.metric.g[i][j]=i==j?1:0;for(int k=0;k<3;++k)u.metric.g[i][j]+=matrix[k][i]*matrix[k][j];}
    for(int a=0;a<3;++a)for(int i=0;i<3;++i)for(int j=i;j<3;++j){u.metric.dg[a][i][j]=u.metric.dg[a][j][i]=.15*random(generator);
      u.a.dk[a][i][j]=u.a.dk[a][j][i]=.1*random(generator);}
    for(int a=0;a<3;++a)for(int b=a;b<3;++b)for(int i=0;i<3;++i)for(int j=i;j<3;++j){const Real v=.15*random(generator);
      u.metric.ddg[a][b][i][j]=u.metric.ddg[b][a][i][j]=u.metric.ddg[a][b][j][i]=u.metric.ddg[b][a][j][i]=v;}
    for(int i=0;i<3;++i)for(int j=i;j<3;++j)u.a.k[i][j]=u.a.k[j][i]=.1*random(generator);
    auto scalar=[&](h::ScalarJet<Real> &f,Real value){f.value=value;for(int a=0;a<3;++a){f.d[a]=.2*random(generator);
      for(int b=a;b<3;++b)f.dd[a][b]=f.dd[b][a]=.2*random(generator);}};
    scalar(u.chi,.3+.8*(1+random(generator)));scalar(u.alpha,.4+.5*(1+random(generator)));
    scalar(u.trace,-1+.4*random(generator));scalar(u.theta,.1*random(generator));
    for(int v=0;v<3;++v){u.beta.value[v]=.3*random(generator);u.lambda.value[v]=.2*random(generator);
      for(int a=0;a<3;++a){u.beta.d[a][v]=.2*random(generator);u.lambda.d[a][v]=.2*random(generator);
        for(int b=a;b<3;++b){u.beta.dd[a][b][v]=u.beta.dd[b][a][v]=.2*random(generator);u.lambda.dd[a][b][v]=u.lambda.dd[b][a][v]=.2*random(generator);}}}
    const auto geometry=h::Geometry(u.metric);Require(geometry.valid,"random SPD metric rejected");
    det_error=std::max(det_error,std::abs(geometry.determinant-1));
    Real tr=0;for(int i=0;i<3;++i){connection_error=std::max(connection_error,std::abs(u.lambda.value[i]-geometry.contracted[i]));
      for(int j=0;j<3;++j){tr+=geometry.inverse[i][j]*u.a.k[i][j];ricci_max=std::max(ricci_max,std::abs(geometry.ricci[i][j]));}}
    trace_a=std::max(trace_a,std::abs(tr));
    for(int normal_mode=0;normal_mode<3;++normal_mode){
      h::OmegaJet<Real> o{};o.omega=sample%7?.1+.3*(1+random(generator)):0;
      for(int i=0;i<3;++i){o.gradient[i]=.2*random(generator);for(int j=i;j<3;++j)o.hessian[i][j]=o.hessian[j][i]=.2*random(generator);}
      if(normal_mode==0)h::SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,o);
      else{
        // Independent, explicitly nonstationary normal data. In particular do
        // not infer alpha*w from the unrelated beta/Omega coordinate jets.
        o.normal=.4*random(generator);for(int i=0;i<3;++i)o.dnormal[i]=.2*random(generator);
        if(normal_mode==2){o.has_lapse_normal_product=true;o.lapse_normal_product=u.alpha.value*o.normal;
          for(int i=0;i<3;++i)o.d_lapse_normal_product[i]=u.alpha.d[i]*o.normal+u.alpha.value*o.dnormal[i];}
      }
      const Real kappa1=.2+4*(1+random(generator)),kappa2=-.99+.5*random(generator);
      const auto legacy=h::LegacyConformalRHS(u,o,kappa1,kappa2),updated=h::ConformalRHS(u,o,kappa1,kappa2);
      Require(legacy.valid&&updated.valid,"normal-range full kernel rejected");
      CompareRHS(legacy.regular,updated.regular,regular_error);CompareRHS(legacy.pole,updated.pole,pole_error);
      if(o.omega>0){h::Z4cRHS<Real> old_rhs{},new_rhs{};Require(h::AssembleInterior(legacy,o.omega,old_rhs)&&h::AssembleInterior(updated,o.omega,new_rhs),"assembled kernel rejected");CompareRHS(old_rhs,new_rhs,assembled_error);}
      auto selected_legacy=legacy;selected_legacy.pole.theta+=3*h::LapseOmegaNormalProduct(u.alpha.value,o)*u.theta.value;
      const auto selected=h::ConformalRHS(u,o,kappa1,kappa2,true);
      Require(selected.valid,"selected normal-range kernel rejected");
      CompareRHS(selected_legacy.regular,selected.regular,selected_error);CompareRHS(selected_legacy.pole,selected.pole,selected_error);
      if(o.omega>0){h::Z4cRHS<Real> old_rhs{},new_rhs{};Require(h::AssembleInterior(selected_legacy,o.omega,old_rhs)&&h::AssembleInterior(selected,o.omega,new_rhs),"selected assembled kernel rejected");CompareRHS(old_rhs,new_rhs,selected_error);}
      CompareConstraints(h::LegacyEvolvedConstraints(u,o),h::EvolvedConstraints(u,o),constraint_error);++cases;
    }
  }
  Require(det_error>.01&&trace_a>.01&&connection_error>.05&&ricci_max>.001,"random tests accidentally imposed differential/algebraic constraints");
  Require(std::max({regular_error,pole_error,assembled_error,constraint_error,selected_error})<3e-11,"full normal-range equation equivalence failed");

  int regular_extreme_cases=0,old_extreme_failures=0;Real scaled_chi_source_error=0;
  for(int sample=0;sample<2;++sample){
    h::Z4cJet<Real> u{};for(int i=0;i<3;++i)u.metric.g[i][i]=1;
    u.chi.value=1e-200;u.alpha.value=sample?1e-200:1;
    if(sample==0)u.chi.d[0]=1;
    u.trace.value=.1;u.theta.value=.02;
    h::OmegaJet<Real> o{};o.omega=.5;for(int i=0;i<3;++i)o.hessian[i][i]=-1;
    if(sample==1){u.beta.value[0]=1;o.gradient[0]=1;h::SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,o);}
    const auto updated=h::ConformalRHS(u,o,4.,-.99),old=h::LegacyConformalRHS(u,o,4.,-.99);
    Require(updated.valid,"finite chi-extreme full kernel rejected");if(!old.valid)++old_extreme_failures;
    if(sample==1){const Real expected=Real(2./3.)*u.chi.value*(u.alpha.value*(u.trace.value+2*u.theta.value)-3*o.lapse_normal_product);
      scaled_chi_source_error=std::max(scaled_chi_source_error,std::abs(updated.pole.chi/expected-1));
      Require(updated.pole.chi!=0,"alpha*chi underflow erased a representable chi pole");}
    const auto constraints=h::EvolvedConstraints(u,o);
    Require(std::isfinite(constraints.hamiltonian),"regular extreme Hamiltonian not finite");
    for(int i=0;i<3;++i)Require(std::isfinite(constraints.momentum[i]),"regular extreme momentum not finite");
    if(sample==0)Require(constraints.valid,"finite extreme null/constraint diagnostics rejected");
    else Require(!constraints.valid&&!std::isfinite(constraints.null_residual),"unrepresentable null diagnostic was manufactured finite");
    ++regular_extreme_cases;
  }
  h::Z4cJet<Real> tiny{};for(int i=0;i<3;++i){tiny.metric.g[i][i]=1;tiny.a.k[i][i]=.01*(i+1);}
  tiny.chi.value=1;tiny.alpha.value=1e-300;tiny.theta.value=.02;tiny.beta.value[0]=1e100;
  h::OmegaJet<Real> o{};o.omega=.5;o.gradient[0]=1;for(int i=0;i<3;++i)o.hessian[i][i]=-1;
  h::SetStationaryOmegaNormal(tiny.alpha.value,tiny.beta.value,tiny.alpha.d,tiny.beta.d,o);
  Require(o.has_lapse_normal_product&&std::isfinite(o.lapse_normal_product)&&!std::isfinite(o.normal),"extreme normal/product setup not exercised");
  const auto stable=h::ConformalRHS(tiny,o,4.,-.99),old=h::LegacyConformalRHS(tiny,o,4.,-.99);
  Require(stable.valid&&!old.valid,"finite alpha*w product did not repair range failure");++regular_extreme_cases;++old_extreme_failures;
  const auto invalid_null=h::EvolvedConstraints(tiny,o);
  Require(!invalid_null.valid&&!std::isfinite(invalid_null.null_residual)&&std::isfinite(invalid_null.hamiltonian),"null validity was concealed by alpha*w product");
  for(int i=0;i<3;++i)Require(std::isfinite(invalid_null.momentum[i]),"independent extreme momentum unavailable");
  o.has_lapse_normal_product=false;Require(!h::ConformalRHS(tiny,o,4.,-.99).valid,"explicit normal fallback silently inferred a product");
  tiny.alpha.value=0;o.has_lapse_normal_product=true;Require(!h::ConformalRHS(tiny,o,4.,-.99).valid,"positive-lapse active guard was relaxed");
  Require(scaled_chi_source_error<3e-13,"stable alpha*w chi pole mismatch");

  // The selected Appendix-C equation omits the known 3*N*Theta term. Removing
  // it algebraically before evaluation is equivalent to the disclosed old
  // post-addition, but preserves a collapsed-alpha remainder lost by canceling
  // two O(1) floating-point values after evaluating the full tensor equation.
  h::Z4cJet<Real> collapsed{};for(int i=0;i<3;++i)collapsed.metric.g[i][i]=1;
  collapsed.chi.value=1;collapsed.alpha.value=1e-200;collapsed.trace.value=.1;collapsed.theta.value=.02;collapsed.beta.value[0]=1;
  h::OmegaJet<Real> co{};co.omega=.5;co.gradient[0]=1;for(int i=0;i<3;++i)co.hessian[i][i]=-1;
  h::SetStationaryOmegaNormal(collapsed.alpha.value,collapsed.beta.value,collapsed.alpha.d,collapsed.beta.d,co);
  const auto collapsed_old=h::LegacyConformalRHS(collapsed,co,4.,-.99),collapsed_selected=h::ConformalRHS(collapsed,co,4.,-.99,true);
  Require(collapsed_old.valid&&collapsed_selected.valid,"collapsed selected cancellation setup rejected");
  const Real physical_k=collapsed.trace.value+2*collapsed.theta.value;
  const Real expected_theta=collapsed.alpha.value*(physical_k*physical_k/3-3-4*(2-.99)*collapsed.theta.value);
  const Real post_add=collapsed_old.pole.theta+3*h::LapseOmegaNormalProduct(collapsed.alpha.value,co)*collapsed.theta.value;
  const Real selected_tiny_error=std::abs(collapsed_selected.pole.theta/expected_theta-1);
  const Real post_add_relative_error=std::abs(post_add/expected_theta-1);
  Require(collapsed_selected.pole.theta!=0&&selected_tiny_error<3e-13,"selected tiny-alpha remainder was lost");
  Require(post_add_relative_error>.5,"old post-add control did not expose cancellation");
  std::cout<<std::setprecision(17)<<"{\"random_offconstraint_cases\":"<<cases<<",\"normal_data_modes\":3,\"regular_error\":"<<regular_error
    <<",\"pole_error\":"<<pole_error<<",\"assembled_error\":"<<assembled_error<<",\"constraint_error\":"<<constraint_error
    <<",\"selected_AppendixC_error\":"<<selected_error<<",\"selected_tiny_alpha_relative_error\":"<<selected_tiny_error
    <<",\"selected_tiny_alpha_theta_pole\":"<<collapsed_selected.pole.theta<<",\"legacy_post_add_value\":"<<post_add
    <<",\"legacy_post_add_relative_error\":"<<post_add_relative_error
    <<",\"maximum_determinant_error_sampled\":"<<det_error<<",\"maximum_trace_A_sampled\":"<<trace_a<<",\"maximum_Lambda_minus_V_sampled\":"<<connection_error
    <<",\"finite_full_RHS_extreme_cases\":"<<regular_extreme_cases<<",\"expected_legacy_extreme_failures\":"<<old_extreme_failures
    <<",\"scaled_chi_source_error\":"<<scaled_chi_source_error<<",\"null_nonfinite_validity_preserved\":true,\"positive_lapse_guard_preserved\":true}\n";
}catch(const std::exception &e){std::cerr<<e.what()<<"\n";result=1;}
Kokkos::finalize();return result;}
