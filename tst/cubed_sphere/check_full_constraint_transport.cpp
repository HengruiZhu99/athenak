// Read-only analytic-data compatibility audit. No native evolution or boundary update.
#include <iostream>
#include <iomanip>
#include <cmath>
#include "z4c/hyperboloidal/layer_wormhole.hpp"
#include "mesh/cubed_sphere/reference_wave_map.hpp"
#include "tst/cubed_sphere/native_constraint_transport.hpp"
namespace h=z4c::hyperboloidal;
void Audit(const h::Z4cJet<double>&u,const h::LayerPoint<double>&p,const double n[3],double mass,double radius,int direction,int variant){
 h::OmegaJet<double> o{};o.omega=p.omega;
 for(int a=0;a<3;++a){o.gradient[a]=p.domega[a];for(int b=0;b<3;++b)o.hessian[a][b]=p.omega_hessian[a][b];}
 h::SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,o);
 auto constraints=h::EvolvedConstraints(u,o);h::Z4cRHS<double> f{};
 if(!constraints.valid||!h::AssembleInterior(h::ConformalRHS(u,o,5.,0.),p.omega,f))throw std::runtime_error("invalid analytic data");
 auto geo=h::Geometry(u.metric);double gradf[3][3][3]{};double div=0,ddiv[3]{};
 for(int a=0;a<3;++a){div+=u.beta.d[a][a];for(int e=0;e<3;++e)ddiv[e]+=u.beta.dd[e][a][a];}
 for(int e=0;e<3;++e)for(int i=0;i<3;++i)for(int j=0;j<3;++j){
  gradf[e][i][j]=-2*(u.alpha.d[e]*u.a.k[i][j]+u.alpha.value*u.a.dk[e][i][j])-2./3.*(u.metric.dg[e][i][j]*div+u.metric.g[i][j]*ddiv[e]);
  for(int d=0;d<3;++d)gradf[e][i][j]+=u.beta.d[e][d]*u.metric.dg[d][i][j]+u.beta.value[d]*u.metric.ddg[e][d][i][j]+u.metric.dg[e][d][j]*u.beta.d[i][d]+u.metric.g[d][j]*u.beta.dd[e][i][d]+u.metric.dg[e][i][d]*u.beta.d[j][d]+u.metric.g[i][d]*u.beta.dd[e][j][d];
 }
 double invdot[3][3]{},gammadot[3]{},connDot[3][3][3]{};
 for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int a=0;a<3;++a)for(int b=0;b<3;++b)invdot[i][j]-=geo.inverse[i][a]*f.metric[a][b]*geo.inverse[b][j];
 for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int k=0;k<3;++k){
  for(int a=0;a<3;++a)connDot[i][j][k]+=.5*(invdot[i][a]*(u.metric.dg[j][a][k]+u.metric.dg[k][a][j]-u.metric.dg[a][j][k])+geo.inverse[i][a]*(gradf[j][a][k]+gradf[k][a][j]-gradf[a][j][k]));
  gammadot[i]+=invdot[j][k]*geo.connection[i][j][k]+geo.inverse[j][k]*connDot[i][j][k];
 }
 double z[3]{},zdot[3]{},beta=0,norm=0,derivative=0,variationError[2]{};
 for(int i=0;i<3;++i){beta+=u.beta.value[i]*n[i];derivative+=n[i]*(u.chi.d[i]-p.state.chi.d[i]);for(int j=0;j<3;++j){norm+=u.chi.value*geo.inverse[i][j]*n[i]*n[j];z[i]+=.5*u.metric.g[i][j]*(u.lambda.value[j]-geo.contracted[j]);zdot[i]+=.5*(f.metric[i][j]*(u.lambda.value[j]-geo.contracted[j])+u.metric.g[i][j]*(f.lambda[j]-gammadot[j]));}}
 for(int step=0;step<2;++step){double eps=step?5e-6:1e-5;auto plus=u.metric,minus=u.metric;
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){plus.g[i][j]+=eps*f.metric[i][j];minus.g[i][j]-=eps*f.metric[i][j];for(int d=0;d<3;++d){plus.dg[d][i][j]+=eps*gradf[d][i][j];minus.dg[d][i][j]-=eps*gradf[d][i][j];}}
  auto gp=h::Geometry(plus),gm=h::Geometry(minus);for(int i=0;i<3;++i)variationError[step]=std::max(variationError[step],std::abs((gp.contracted[i]-gm.contracted[i])/(2*eps)-gammadot[i])/std::max(1.,std::abs(gammadot[i])));
 }
 const double c=u.alpha.value*std::sqrt(norm),residual=f.chi+(c-beta)*derivative;auto transport=nct::Evaluate(u,o,n,5.,0.);double zerr=0,sommerfeld=std::abs(transport.theta_residual),omit=std::abs(transport.tangential_omission[0]);for(int i=0;i<3;++i){zerr=std::max(zerr,std::abs(transport.z[i]-zdot[i]));sommerfeld=std::max(sommerfeld,std::abs(transport.z_residual[i]));omit=std::max(omit,std::abs(transport.tangential_omission[i+1]));}
 std::cout<<mass<<","<<radius<<","<<direction<<","<<variant<<","<<p.omega<<","<<constraints.hamiltonian<<","<<std::sqrt(constraints.momentum_conformal_norm2)<<","<<std::sqrt(constraints.z4.z_conformal_norm2)<<","<<u.theta.value<<","<<f.theta;
 for(double v:zdot)std::cout<<","<<v;std::cout<<","<<f.chi<<","<<u.chi.value-p.state.chi.value<<","<<residual<<","<<variationError[0]<<","<<variationError[1]<<","<<std::max(std::abs(gammadot[0]),std::max(std::abs(gammadot[1]),std::abs(gammadot[2])))<<","<<f.theta-u.alpha.value*constraints.hamiltonian/(2*p.omega)<<","<<std::max(std::abs(zdot[0]-u.alpha.value*constraints.momentum[0]/p.omega),std::max(std::abs(zdot[1]-u.alpha.value*constraints.momentum[1]/p.omega),std::abs(zdot[2]-u.alpha.value*constraints.momentum[2]/p.omega)))<<","<<transport.theta-f.theta<<","<<zerr<<","<<sommerfeld<<","<<omit<<"\n";
}
int main(int argc,char**argv){Kokkos::initialize(argc,argv);std::cout<<std::setprecision(17)<<"mass,radius,direction,gauge_variant,Omega,H,M_conformal,Z_conformal,Theta,Theta_dot,Z_dot_x,Z_dot_y,Z_dot_z,chi_dot,chi_deviation,naive_chi_Sommerfeld,connection_variation_error1,connection_variation_error2,Gamma_dot_max,Theta_identity_error,Z_identity_error_max,Theta_transport_error,Z_transport_error_max,constraint_Sommerfeld_max,tangential_omission_max\n";
 {h::LayerReference<double> ref(1.,1.,{true,.35,.75});h::LayerWormhole<double> hole(ref,.5);double directions[3][3]={{1,0,0},{.8,-.48,.36},{1/std::sqrt(3.),1/std::sqrt(3.),1/std::sqrt(3.)}};
 for(double r:{.90,.95,.98,.99})for(int k=0;k<3;++k){auto n=directions[k];auto p=ref.At(r*n[0],r*n[1],r*n[2]);auto u=hole.At(r*n[0],r*n[1],r*n[2]);Audit(p.state,p,n,0.,r,k,0);Audit(u,p,n,.5,r,k,0);
 for(int kind=0;kind<2;++kind){auto angular=kind?u:p.state;double x[3]={r*n[0],r*n[1],r*n[2]};angular.alpha.value+=.03*(x[0]*x[0]-x[1]*x[1]);angular.alpha.d[0]+=.06*x[0];angular.alpha.d[1]-=.06*x[1];angular.alpha.dd[0][0]+=.06;angular.alpha.dd[1][1]-=.06;
 const int aa[3]={0,1,2},bb[3]={1,2,0};const double amp[3]={.01,-.02,.015};for(int i=0;i<3;++i){int a=aa[i],b=bb[i];angular.beta.value[i]+=amp[i]*x[a]*x[b];angular.beta.d[a][i]+=amp[i]*x[b];angular.beta.d[b][i]+=amp[i]*x[a];angular.beta.dd[a][b][i]+=amp[i];angular.beta.dd[b][a][i]+=amp[i];}Audit(angular,p,n,kind?.5:0.,r,k,1);angular.chi.value+=.02*(x[0]*x[0]+.4*x[0]*x[1]-x[1]*x[1]);angular.chi.d[0]+=.04*x[0]+.008*x[1];angular.chi.d[1]+=.008*x[0]-.04*x[1];angular.chi.dd[0][0]+=.04;angular.chi.dd[1][1]-=.04;angular.chi.dd[0][1]+=.008;angular.chi.dd[1][0]+=.008;Audit(angular,p,n,kind?.5:0.,r,k,2);
 auto full=angular;full.metric={};for(int i=0;i<3;++i){double rate=i==0?.07:(i==1?-.07:0.);full.metric.g[i][i]=std::exp(rate*x[0]);full.metric.dg[0][i][i]=rate*full.metric.g[i][i];full.metric.ddg[0][0][i][i]=rate*rate*full.metric.g[i][i];}
 auto gf=h::Geometry(full.metric);double dgi[3][3][3]{},tr=0,dtr[3]{};for(int e=0;e<3;++e)for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int a=0;a<3;++a)for(int b=0;b<3;++b)dgi[e][i][j]-=gf.inverse[i][a]*full.metric.dg[e][a][b]*gf.inverse[b][j];
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){tr+=gf.inverse[i][j]*angular.a.k[i][j];for(int e=0;e<3;++e)dtr[e]+=dgi[e][i][j]*angular.a.k[i][j]+gf.inverse[i][j]*angular.a.dk[e][i][j];}
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){full.a.k[i][j]-=full.metric.g[i][j]*tr/3;for(int e=0;e<3;++e)full.a.dk[e][i][j]-=(full.metric.dg[e][i][j]*tr+full.metric.g[i][j]*dtr[e])/3;}
 full.theta.value=.003*(1+x[0]*x[0]+.2*x[1]*x[2]);full.theta.d[0]=.006*x[0];full.theta.d[1]=.0006*x[2];full.theta.d[2]=.0006*x[1];full.theta.dd[0][0]=.006;full.theta.dd[1][2]=full.theta.dd[2][1]=.0006;
 double zz[3],dz[3][3]{};for(int i=0;i<3;++i){int j=(i+1)%3,k=(i+2)%3;zz[i]=.002*(i+1)*(1+x[i]+.2*x[j]*x[k]);dz[i][i]=.002*(i+1);dz[j][i]=.0004*(i+1)*x[k];dz[k][i]=.0004*(i+1)*x[j];}
 for(int i=0;i<3;++i){full.lambda.value[i]=gf.contracted[i];for(int e=0;e<3;++e)full.lambda.d[e][i]=gf.dcontracted[e][i];for(int j=0;j<3;++j){full.lambda.value[i]+=2*gf.inverse[i][j]*zz[j];for(int e=0;e<3;++e)full.lambda.d[e][i]+=2*(dgi[e][i][j]*zz[j]+gf.inverse[i][j]*dz[e][j]);}}
 Audit(full,p,n,kind?.5:0.,r,k,3);}}}
 Kokkos::finalize();}
