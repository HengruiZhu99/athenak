// Diagnostic derivation only; not included by the production solver.
#ifndef RESEARCH_NATIVE_CONSTRAINT_TRANSPORT_HPP_
#define RESEARCH_NATIVE_CONSTRAINT_TRANSPORT_HPP_
#include "z4c/hyperboloidal/conformal_rhs.hpp"
namespace nct {
namespace h=z4c::hyperboloidal;
struct Result {double theta=0,z[3]{},dz[3][3]{},outgoing[3]{},theta_residual=0,z_residual[3]{},tangential_omission[4]{};};
inline Result Evaluate(const h::Z4cJet<double>&u,const h::OmegaJet<double>&o,const double n[3],double k1,double k2){
 Result r;auto g=h::Geometry(u.metric);auto con=h::EvolvedConstraints(u,o);double Z[3]{},Zu[3]{},div=0,normal=0,beta_n=0;
 for(int i=0;i<3;++i){Zu[i]=.5*(u.lambda.value[i]-g.contracted[i]);for(int j=0;j<3;++j)Z[i]+=.5*u.metric.g[i][j]*(u.lambda.value[j]-g.contracted[j]);}
 for(int d=0;d<3;++d)for(int i=0;i<3;++i)for(int j=0;j<3;++j)r.dz[d][i]+=.5*(u.metric.dg[d][i][j]*(u.lambda.value[j]-g.contracted[j])+u.metric.g[i][j]*(u.lambda.d[d][j]-g.dcontracted[d][j]));
 for(int i=0;i<3;++i){beta_n+=u.beta.value[i]*n[i];for(int j=0;j<3;++j){normal+=g.inverse[i][j]*n[i]*n[j];double cov=r.dz[i][j];for(int k=0;k<3;++k)cov-=g.connection[k][i][j]*Z[k];div+=g.inverse[i][j]*cov;}}
 const double a=u.alpha.value,K=u.trace.value+2*u.theta.value,c=a*std::sqrt(u.chi.value*normal);
 r.theta=a*con.hamiltonian/(2*o.omega)+o.omega*a*u.chi.value*div-a*(3*o.normal+k1*(2+k2))*u.theta.value/o.omega;
 for(int j=0;j<3;++j)r.theta+=u.beta.value[j]*u.theta.d[j];
 for(int i=0;i<3;++i){r.z[i]=a*(con.momentum[i]+u.theta.d[i])/o.omega-a*(2*K/3+k1)*Z[i]/o.omega;
  for(int j=0;j<3;++j){r.z[i]+=u.beta.value[j]*r.dz[j][i]-2*a*u.a.k[i][j]*Zu[j]+Z[j]*u.beta.d[i][j];for(int k=0;k<3;++k)r.z[i]+=u.metric.g[i][k]*Zu[j]*u.beta.d[j][k];}
  r.outgoing[i]=-u.beta.value[i];for(int j=0;j<3;++j)r.outgoing[i]+=a*std::sqrt(u.chi.value)*g.inverse[i][j]*n[j]/std::sqrt(normal);
 }
 r.theta_residual=r.theta;for(int i=0;i<3;++i){r.theta_residual+=r.outgoing[i]*u.theta.d[i];r.tangential_omission[0]+=(r.outgoing[i]-(c-beta_n)*n[i])*u.theta.d[i];}
 for(int i=0;i<3;++i){r.z_residual[i]=r.z[i];for(int j=0;j<3;++j){r.z_residual[i]+=r.outgoing[j]*r.dz[j][i];r.tangential_omission[i+1]+=(r.outgoing[j]-(c-beta_n)*n[j])*r.dz[j][i];}}
 return r;
}
}
#endif
