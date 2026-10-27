// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include "athena.hpp"
// Deliberately include the new helper before ScalarJet is defined.
#include "z4c/hyperboloidal/regular_penrose.hpp"
#include "z4c/hyperboloidal/conformal_rhs.hpp"

namespace h=z4c::hyperboloidal;
void Require(bool test,const char *message){if(!test)throw std::runtime_error(message);}
void Compare(Real a,Real b,Real scale,Real &worst){Require(std::isfinite(a)&&std::isfinite(b),"nonfinite comparison");
  worst=std::max(worst,std::abs(a-b)/std::max({scale,std::abs(a),std::abs(b)}));}
h::MetricJet<Real> Flat(){h::MetricJet<Real> g{};for(int i=0;i<3;++i)g.g[i][i]=1;return g;}

int main(int argc,char **argv){Kokkos::initialize(argc,argv);int result=0;
try{
  const bool expect_regular_rhs=argc>1&&std::string(argv[1])=="--expect-regular-rhs";
  std::mt19937_64 generator(827431);std::uniform_real_distribution<Real> random(-1,1);
  Real normal_error=0,nonflat_ricci_max=0,min_det=std::numeric_limits<Real>::max(),max_det=0;
  constexpr int random_cases=128;
  for(int test=0;test<random_cases;++test){
    h::MetricJet<Real> g{};Real matrix[3][3];for(auto &row:matrix)for(auto &v:row)v=.4*random(generator);
    for(int i=0;i<3;++i)for(int j=0;j<3;++j){g.g[i][j]=i==j?1:0;
      for(int k=0;k<3;++k)g.g[i][j]+=matrix[k][i]*matrix[k][j];}
    for(int a=0;a<3;++a)for(int i=0;i<3;++i)for(int j=i;j<3;++j)g.dg[a][i][j]=g.dg[a][j][i]=.3*random(generator);
    for(int a=0;a<3;++a)for(int b=a;b<3;++b)for(int i=0;i<3;++i)for(int j=i;j<3;++j){const Real v=.3*random(generator);
      g.ddg[a][b][i][j]=g.ddg[b][a][i][j]=g.ddg[a][b][j][i]=g.ddg[b][a][j][i]=v;}
    h::ScalarJet<Real> chi{},f{};chi.value=.15+1.3*(1+random(generator));f.value=test%3?1+.2*random(generator):0;
    Real z[3],dz[3][3];
    for(int i=0;i<3;++i){chi.d[i]=.2*random(generator);f.d[i]=.3*random(generator);z[i]=.3*random(generator);
      for(int j=0;j<3;++j)dz[i][j]=.2*random(generator);
      for(int j=i;j<3;++j){chi.dd[i][j]=chi.dd[j][i]=.2*random(generator);f.dd[i][j]=f.dd[j][i]=.3*random(generator);}}
    const auto gt=h::Geometry(g),gb=h::Geometry(h::PenroseMetric(g,chi));
    Require(gt.valid&&gb.valid,"normal-range geometry rejected");min_det=std::min(min_det,gt.determinant);max_det=std::max(max_det,gt.determinant);
    const auto curvature=h::RegularPenroseGeometry(g,gt,chi);
    const auto scalar=h::RegularPenroseHessian(g,gt,chi,f);
    const auto covector=h::RegularPenroseCovectorDerivative(g,gt,chi,z,dz);
    Require(curvature.valid&&scalar.valid&&covector.valid,"regular normal-range helper rejected");
    Real old_h[3][3]{},old_z[3][3]{},old_lap=0,old_div=0,new_trace=0,ricci_trace=0;
    for(int i=0;i<3;++i)for(int j=0;j<3;++j){
      nonflat_ricci_max=std::max(nonflat_ricci_max,std::abs(gt.ricci[i][j]));
      old_h[i][j]=f.dd[i][j];old_z[i][j]=dz[i][j];
      for(int k=0;k<3;++k){old_h[i][j]-=gb.connection[k][i][j]*f.d[k];old_z[i][j]-=gb.connection[k][i][j]*z[k];}
      old_lap+=gb.inverse[i][j]*old_h[i][j];old_div+=gb.inverse[i][j]*old_z[i][j];
      new_trace+=gt.inverse[i][j]*scalar.chi_hessian[i][j];ricci_trace+=gt.inverse[i][j]*curvature.chi_ricci[i][j];
      Compare(curvature.chi_ricci[i][j],chi.value*gb.ricci[i][j],1.,normal_error);
      Compare(scalar.chi_hessian[i][j],chi.value*old_h[i][j],1.,normal_error);
      Compare(covector.chi_derivative[i][j],chi.value*old_z[i][j],1.,normal_error);
    }
    Compare(curvature.scalar,gb.scalar,1.,normal_error);Compare(scalar.laplacian,old_lap,1.,normal_error);
    Compare(covector.divergence,old_div,1.,normal_error);Compare(new_trace,scalar.laplacian,1.,normal_error);
    Compare(ricci_trace,curvature.scalar,1.,normal_error);
    // This is the full scaled geometric combination entering the curvature
    // regular source, including arbitrary independent covector derivatives.
    Real old_tensor[3][3],new_tensor[3][3],old_tensor_trace=0,new_tensor_trace=0;
    for(int i=0;i<3;++i)for(int j=0;j<3;++j){
      old_tensor[i][j]=chi.value*(-old_h[i][j]+f.value*gb.ricci[i][j]+old_z[i][j]+old_z[j][i]);
      new_tensor[i][j]=-scalar.chi_hessian[i][j]+f.value*curvature.chi_ricci[i][j]
          +covector.chi_derivative[i][j]+covector.chi_derivative[j][i];
      old_tensor_trace+=gt.inverse[i][j]*old_tensor[i][j];new_tensor_trace+=gt.inverse[i][j]*new_tensor[i][j];
      Compare(old_tensor[i][j],new_tensor[i][j],1.,normal_error);
    }
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)
      Compare(old_tensor[i][j]-g.g[i][j]*old_tensor_trace/3,new_tensor[i][j]-g.g[i][j]*new_tensor_trace/3,1.,normal_error);
  }
  Require(nonflat_ricci_max>.001,"normal tests accidentally flat");Require(normal_error<3e-11,"normal-range conformal signs or contractions disagree");

  const auto flat=Flat();const auto gt=h::Geometry(flat);Real extreme_error=0;
  int finite_extreme_cases=0,old_geometry_failures=0,old_rhs_failures=0;
  for(int test=0;test<4;++test){
    h::ScalarJet<Real> chi{},f{};chi.value=1e-200;f.value=1;
    Real z[3]{},dz[3][3]{},expected_ricci[3][3]{},expected_h[3][3]{},expected_z[3][3]{};
    Real expected_scalar=0,expected_lap=0,expected_div=0,curvature_scale=1,derivative_scale=1;
    if(test==0){
      f.d[0]=.2;f.d[1]=-.1;f.d[2]=.3;
      for(int i=0;i<3;++i){f.dd[i][i]=i+2;expected_h[i][i]=chi.value*(i+2);expected_lap+=expected_h[i][i];}
      derivative_scale=chi.value;
    }else if(test==1){
      // chi=chi0 exp(1e100 v.x), f=exp(1e100 b.x) at x=0.
      // All requested scaled curvature/Hessian combinations are O(1), despite
      // the unscaled bar metric/curvature jet being unrepresentable.
      const Real v[3]={1,-.7,.4},b[3]={.3,.2,-.1},c[3]={.1,-.2,.4};
      Real norm=0,cross=0,cross_z=0,norm_b=0;
      for(int i=0;i<3;++i){chi.d[i]=1e-100*v[i];f.d[i]=1e100*b[i];z[i]=1e100*c[i];
        norm+=v[i]*v[i];norm_b+=b[i]*b[i];cross+=v[i]*b[i];cross_z+=v[i]*c[i];
        for(int j=0;j<3;++j){chi.dd[i][j]=v[i]*v[j];f.dd[i][j]=1e200*b[i]*b[j];dz[i][j]=1e200*.02*(1+i-2*j);}}
      for(int i=0;i<3;++i)for(int j=0;j<3;++j){
        expected_ricci[i][j]=.25*(v[i]*v[j]-(i==j?norm:0));
        expected_h[i][j]=b[i]*b[j]+.5*(v[i]*b[j]+v[j]*b[i]-(i==j?cross:0));
        expected_z[i][j]=.02*(1+i-2*j)+.5*(v[i]*c[j]+v[j]*c[i]-(i==j?cross_z:0));}
      expected_scalar=-.5*norm;expected_lap=norm_b-.5*cross;
      expected_div=-.5*cross_z;for(int i=0;i<3;++i)expected_div+=.02*(1-i);
    }else if(test==2){
      // Positive local linear chi jet: ratios reach 1e200, but their products
      // in chi*Ricci and scalar curvature still fit in double precision.
      chi.d[0]=1;f.d[0]=.2;f.d[1]=-.1;f.d[2]=.3;
      expected_scalar=-2.5e200;curvature_scale=1e200;
      for(int i=0;i<3;++i)for(int j=0;j<3;++j){
        f.dd[i][j]=i==j?i+1:0;
        expected_ricci[i][j]=-.25e200*((i==0&&j==0?1:0)+(i==j?3:0));
        expected_h[i][j]=chi.value*f.dd[i][j]+.5*((i==0?f.d[j]:0)+(j==0?f.d[i]:0)-(i==j?f.d[0]:0));}
      expected_lap=6e-200-.1;
    }else{
      // Forming an unscaled trace first would overflow even though its scaled
      // Laplacian is representable. Sum already-scaled covariant components.
      for(int i=0;i<3;++i){f.dd[i][i]=1e308;expected_h[i][i]=1e108;}expected_lap=3e108;derivative_scale=1e108;
    }
    const auto curvature=h::RegularPenroseGeometry(flat,gt,chi);
    const auto scalar=h::RegularPenroseHessian(flat,gt,chi,f);
    const auto covector=h::RegularPenroseCovectorDerivative(flat,gt,chi,z,dz);
    Require(curvature.valid&&scalar.valid&&covector.valid,"finite analytic extreme combination rejected");
    Compare(curvature.scalar,expected_scalar,curvature_scale,extreme_error);
    Compare(scalar.laplacian,expected_lap,derivative_scale,extreme_error);Compare(covector.divergence,expected_div,1.,extreme_error);
    for(int i=0;i<3;++i)for(int j=0;j<3;++j){
      Compare(curvature.chi_ricci[i][j],expected_ricci[i][j],curvature_scale,extreme_error);
      Compare(scalar.chi_hessian[i][j],expected_h[i][j],derivative_scale,extreme_error);
      Compare(covector.chi_derivative[i][j],expected_z[i][j],1.,extreme_error);
      const Real combination=-scalar.chi_hessian[i][j]+f.value*curvature.chi_ricci[i][j]
          +covector.chi_derivative[i][j]+covector.chi_derivative[j][i];
      Require(std::isfinite(combination),"combined scaled curvature source is nonfinite");
    }
    const auto old_geometry=h::Geometry(h::PenroseMetric(flat,chi));
    if(!old_geometry.valid)++old_geometry_failures;
    h::Z4cJet<Real> u{};u.metric=flat;u.chi=chi;u.alpha=f;
    h::OmegaJet<Real> o{};o.omega=.5;for(int i=0;i<3;++i)o.hessian[i][i]=-1;
    if(!h::ConformalRHS(u,o,4.,-.99).valid)++old_rhs_failures;
    ++finite_extreme_cases;
  }
  Require(extreme_error<3e-12,"analytic extreme combined formula mismatch");
  Require(old_geometry_failures==4,"expected old Penrose geometry range failures missing");
  Require(old_rhs_failures==(expect_regular_rhs?0:4),"current full RHS extreme range behavior differs from requested baseline/integrated mode");
  h::ScalarJet<Real> invalid{};invalid.value=1e-300;invalid.d[0]=1e100;
  const auto impossible=h::RegularPenroseGeometry(flat,gt,invalid);
  Require(!impossible.valid&&impossible.status==h::RegularPenroseStatus::nonfinite_result,"unrepresentable physical combination was hidden");
  invalid={};Require(h::RegularPenroseGeometry(flat,gt,invalid).status==h::RegularPenroseStatus::nonpositive_chi,"zero chi was floored");
  invalid.value=1;invalid.d[0]=std::numeric_limits<Real>::quiet_NaN();
  Require(h::RegularPenroseGeometry(flat,gt,invalid).status==h::RegularPenroseStatus::nonfinite_input,"nonfinite input accepted");
  std::cout<<std::setprecision(17)<<"{\"random_nonflat_cases\":"<<random_cases<<",\"maximum_normal_combination_error\":"<<normal_error
    <<",\"nonflat_Ricci_maximum\":"<<nonflat_ricci_max<<",\"metric_determinant_range\":["<<min_det<<","<<max_det<<"]"
    <<",\"determinant_one_assumed\":false,\"finite_analytic_extreme_cases\":"<<finite_extreme_cases
    <<",\"maximum_extreme_analytic_error\":"<<extreme_error<<",\"expected_old_geometry_failures\":"<<old_geometry_failures
    <<",\"expect_regular_full_RHS\":"<<expect_regular_rhs<<",\"current_full_RHS_range_failures\":"<<old_rhs_failures
    <<",\"unrepresentable_physical_result_reported\":true,\"zero_chi_rejected_without_floor\":true}\n";
}catch(const std::exception &e){std::cerr<<e.what()<<"\n";result=1;}
Kokkos::finalize();return result;}
