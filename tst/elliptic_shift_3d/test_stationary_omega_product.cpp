// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include "athena.hpp"
#include "z4c/hyperboloidal/stationary_omega_product.hpp"
#include "z4c/hyperboloidal/conformal_constraints.hpp"
#include "z4c/hyperboloidal/tensor_scri.hpp"
namespace h=z4c::hyperboloidal;
// Frozen pre-product arithmetic is retained only as a comparison/failure oracle.
void LegacyStationaryNormal(Real alpha,const Real beta[3],const Real da[3],
    const Real db[3][3],h::OmegaJet<Real> &o){
 o.normal=0;for(int i=0;i<3;++i)o.normal-=beta[i]*o.gradient[i]/alpha;
 for(int d=0;d<3;++d){o.dnormal[d]=-o.normal*da[d]/alpha;
  for(int i=0;i<3;++i)o.dnormal[d]-=(db[d][i]*o.gradient[i]+beta[i]*o.hessian[d][i])/alpha;}
}
int main(int argc,char **argv){Kokkos::initialize(argc,argv);int status=0;
try{
 const Real grad[3]={-.2,.1,.4},hess[3][3]={{-.7,.03,.02},{.03,-.9,-.01},{.02,-.01,-1.1}};
 const Real beta[3]={.3,-.2,.1},dbeta[3][3]={{.11,.02,-.03},{-.04,.07,.01},{.02,.03,-.06}};
 const Real direction[3]={.3,-.4,.5};
 const auto product=h::StationaryOmegaProduct(beta,dbeta,grad,hess);
 using D=h::ScriDual<Real>;
 Real legacy_error=0,derivative_error=0,pole_error=0,pole_derivative_error=0,cache_error=0;
 int finite_cases=0,nonfinite_legacy=0,normal_range_failures=0;
 for(Real alpha:{1.,1e-2,1e-50,1e-150,1e-250,1e-300}){
  const Real da[3]={.2*alpha,-.1*alpha,.05*alpha};h::OmegaJet<Real> omega{};
  for(int i=0;i<3;++i){omega.gradient[i]=grad[i];for(int j=0;j<3;++j)omega.hessian[i][j]=hess[i][j];}
  auto legacy=omega;LegacyStationaryNormal(alpha,beta,da,dbeta,legacy);
  h::SetStationaryOmegaNormal(alpha,beta,da,dbeta,omega);
  if(!omega.has_lapse_normal_product)throw std::runtime_error("stationary product not marked explicit");
  cache_error=std::max(cache_error,std::abs(omega.lapse_normal_product-product.value));
  for(int d=0;d<3;++d)cache_error=std::max(cache_error,std::abs(omega.d_lapse_normal_product[d]-product.d[d]));
  const Real old_value=alpha*legacy.normal;
  if(!std::isfinite(old_value))throw std::runtime_error("finite legacy comparison fixture");
  legacy_error=std::max(legacy_error,std::abs(old_value-product.value));
  for(int d=0;d<3;++d){const Real old_gradient=da[d]*legacy.normal+alpha*legacy.dnormal[d];
   if(!std::isfinite(old_gradient))throw std::runtime_error("finite legacy gradient fixture");
   legacy_error=std::max(legacy_error,std::abs(old_gradient-product.d[d]));}
  const Real chi=.8,K=-2.9,theta=.02,A=.03;
  const Real old_poles[3]={(2./3)*alpha*chi*(K-3*legacy.normal),-3*alpha*legacy.normal*theta,alpha*A*(K-legacy.normal)};
  const Real new_poles[3]={(2./3)*chi*(alpha*K-3*product.value),-3*product.value*theta,A*(alpha*K-product.value)};
  for(int i=0;i<3;++i)pole_error=std::max(pole_error,std::abs(old_poles[i]-new_poles[i]));
  Real dalpha=0;for(int d=0;d<3;++d)dalpha+=direction[d]*da[d];
  const Real dn=product.Directional(direction),dchi=.013,dK=-.017,dtheta=.005,dA=.009;
  const D ad(alpha,dalpha),cd(chi,dchi),kd(K,dK),td(theta,dtheta),Ad(A,dA),nd(product.value,dn);
  const D dual_poles[3]={D(2./3.)*cd*(ad*kd-D(3)*nd),D(-3)*nd*td,Ad*(ad*kd-nd)};
  const Real differentiated[3]={(2./3.)*(dchi*(alpha*K-3*product.value)+chi*(dalpha*K+alpha*dK-3*dn)),
    -3*(dn*theta+product.value*dtheta),dA*(alpha*K-product.value)+A*(dalpha*K+alpha*dK-dn)};
  for(int i=0;i<3;++i){if(!std::isfinite(dual_poles[i].derivative))throw std::runtime_error("expanded pole AD not finite");
    pole_derivative_error=std::max(pole_derivative_error,std::abs(dual_poles[i].derivative-differentiated[i]));}
  ++finite_cases;
 }
 // Differentiate the independent polynomial coordinate data, not w=N/alpha.
 // beta(x)=beta+dbeta*x, Omega_i(x)=grad_i+hess_di*x_d.
 const Real eps=1e-5;Real bp[3],bm[3],gp[3],gm[3];
 for(int i=0;i<3;++i){Real db=0,dg=0;for(int d=0;d<3;++d){db+=direction[d]*dbeta[d][i];dg+=direction[d]*hess[d][i];}
  bp[i]=beta[i]+eps*db;bm[i]=beta[i]-eps*db;gp[i]=grad[i]+eps*dg;gm[i]=grad[i]-eps*dg;}
 const Real fd=(h::StationaryOmegaProductValue(bp,gp)-h::StationaryOmegaProductValue(bm,gm))/(2*eps);
 derivative_error=std::abs(fd-product.Directional(direction));
 D bd[3],gd[3];
 for(int i=0;i<3;++i){Real db=0,dg=0;for(int d=0;d<3;++d){db+=direction[d]*dbeta[d][i];dg+=direction[d]*hess[d][i];}
  bd[i]=D(beta[i],db);gd[i]=D(grad[i],dg);}
 const auto dual=h::StationaryOmegaProductValue(bd,gd);
 derivative_error=std::max(derivative_error,std::abs(dual.derivative-product.Directional(direction)));
 const Real huge_beta[2][3]={{1e12,.25,-.5},{1e12,-1e12,1.}};
 const Real huge_gradient[2][3]={{1.,.4,-.3},{1.,1.,-.25}};
 Real largest_finite_product=0;
 for(int fixture=0;fixture<2;++fixture){
  const Real alpha=1e-300,da[3]={.2*alpha,-.1*alpha,.05*alpha};h::OmegaJet<Real> omega{};
  for(int i=0;i<3;++i){omega.gradient[i]=huge_gradient[fixture][i];for(int j=0;j<3;++j)omega.hessian[i][j]=hess[i][j];}
  auto legacy=omega;LegacyStationaryNormal(alpha,huge_beta[fixture],da,dbeta,legacy);
  h::SetStationaryOmegaNormal(alpha,huge_beta[fixture],da,dbeta,omega);
  const auto safe=h::StationaryOmegaProduct(huge_beta[fixture],dbeta,huge_gradient[fixture],hess);
  if(std::isfinite(alpha*legacy.normal))throw std::runtime_error("legacy overflow/NaN fixture not exercised");
  ++nonfinite_legacy;
  if(!std::isfinite(safe.value))throw std::runtime_error("stationary contraction not finite");
  for(int d=0;d<3;++d)if(!std::isfinite(safe.d[d]))throw std::runtime_error("stationary product derivative not finite");
  if(!omega.has_lapse_normal_product||omega.lapse_normal_product!=safe.value)
    throw std::runtime_error("cached product lost finite extreme-lapse contraction");
  for(int d=0;d<3;++d)if(omega.d_lapse_normal_product[d]!=safe.d[d])
    throw std::runtime_error("cached product gradient changed");
  if(!std::isfinite(omega.normal))++normal_range_failures;
  const Real normal_terms[3]={(2./3)*.8*(alpha*(-2.9)-3*safe.value),-3*safe.value*.02,.03*(alpha*(-2.9)-safe.value)};
  for(Real term:normal_terms)if(!std::isfinite(term))throw std::runtime_error("expanded pole product not finite");
  largest_finite_product=std::max(largest_finite_product,std::abs(safe.value));
  if(fixture==1&&safe.value!=.25)throw std::runtime_error("unscaled contraction cancellation changed");
 }
 // Nonstationary coordinate data must remain explicit; independent normal data
 // are never silently reinterpreted as stationary-Omega kinematics.
 const Real omega_t=.7,domega_t[3]={.03,-.04,.02};
 const auto general=h::CoordinateOmegaProduct(beta,dbeta,grad,hess,omega_t,domega_t);
 if(std::abs(general.value-product.value-omega_t)>1e-15)throw std::runtime_error("nonstationary Omega_t lost");
 for(int d=0;d<3;++d)if(std::abs(general.d[d]-product.d[d]-domega_t[d])>1e-15)
   throw std::runtime_error("nonstationary Omega_t spatial jet lost");
 h::OmegaJet<Real> independent{};independent.normal=general.value/.9;
 for(int d=0;d<3;++d)independent.dnormal[d]=general.d[d]/.9;
 if(independent.has_lapse_normal_product)throw std::runtime_error("independent normal inferred stationary cache");
 independent.has_lapse_normal_product=true;independent.lapse_normal_product=general.value;
 for(int d=0;d<3;++d)independent.d_lapse_normal_product[d]=general.d[d];
 // An explicitly supplied nonstationary product is distinct from the stationary
 // contraction, and must not be recomputed from beta by a general kernel.
 if(independent.lapse_normal_product==product.value)
   throw std::runtime_error("explicit nonstationary product changed to stationary data");
 const D tiny_ratio=D(2e-200,2e-150)/D(1e-200,1e-150);
 if(tiny_ratio.value!=2||tiny_ratio.derivative!=0)
   throw std::runtime_error("AD quotient denominator squared underflow");
 h::Z4cJet<Real> tiny{};const Real matrix[3][3]={{1.2,.1,.02},{.1,.9,-.03},{.02,-.03,1.1}};
 tiny.alpha.value=1e-300;tiny.chi.value=1e-200;tiny.trace.value=-2.9;tiny.theta.value=.02;
 const Real logchi[3]={.1,-.2,.03},x[3]={.2,-.1,.3};h::OmegaJet<Real> compact{};compact.omega=.4;
 for(int i=0;i<3;++i){tiny.beta.value[i]=huge_beta[0][i];tiny.lambda.value[i]=.01*(i+1);
  tiny.chi.d[i]=tiny.chi.value*logchi[i];compact.gradient[i]=grad[i];
  for(int j=0;j<3;++j){tiny.metric.g[i][j]=matrix[i][j];tiny.a.k[i][j]=.003*(i+j+1);
   tiny.chi.dd[i][j]=tiny.chi.value*logchi[i]*logchi[j];compact.hessian[i][j]=hess[i][j];}}
 const Real reference_beta[3]={-.2,.1,-.3};int tiny_pole_components=0;
 for(bool appendix:{false,true}){
  const auto poles=h::TensorPoleSources(tiny,compact,Real(.7),reference_beta,Real(4),Real(-.99),Real(1),Real(1),appendix,true);
  const auto dual_poles=h::TensorPoleNormalJet(tiny,compact,x,direction,Real(4),Real(-.99),Real(1),Real(1),appendix,true);
  auto check=[&](Real value,const D &dual){if(!std::isfinite(value)||!std::isfinite(dual.value)||!std::isfinite(dual.derivative))
      throw std::runtime_error("tiny-lapse/chi stationary pole or directional AD not finite");++tiny_pole_components;};
  check(poles.geometry.chi,dual_poles.geometry.chi);check(poles.geometry.trace,dual_poles.geometry.trace);check(poles.geometry.theta,dual_poles.geometry.theta);
  for(int i=0;i<3;++i){check(poles.geometry.lambda[i],dual_poles.geometry.lambda[i]);for(int j=0;j<3;++j)check(poles.geometry.a[i][j],dual_poles.geometry.a[i][j]);}
 }
 if(legacy_error>3e-15||pole_error>3e-15||pole_derivative_error>3e-15||derivative_error>3e-10||cache_error!=0||normal_range_failures!=1)
   throw std::runtime_error("stationary normal-product identity/derivative audit");
 std::cout<<std::setprecision(17)<<"{\"finite_legacy_cases\":"<<finite_cases
   <<",\"legacy_product_error\":"<<legacy_error<<",\"pole_product_error\":"<<pole_error
   <<",\"pole_directional_derivative_error\":"<<pole_derivative_error
   <<",\"directional_derivative_error\":"<<derivative_error<<",\"nonfinite_legacy_cases\":"<<nonfinite_legacy
   <<",\"product_cache_error\":"<<cache_error<<",\"unit_normal_out_of_range_cases\":"<<normal_range_failures
   <<",\"tiny_stationary_pole_components\":"<<tiny_pole_components
   <<",\"largest_finite_new_product\":"<<largest_finite_product
   <<",\"nonstationary_time_data_preserved\":true,\"alpha_zero_normal_not_defined\":true}\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';status=1;}
Kokkos::finalize();return status;}
