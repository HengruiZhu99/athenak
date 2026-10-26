// Analysis-only nonlinear leading-pole manifold of the actual native22/RWM.
// No evolved fields or numerical endpoint resets are introduced here.
#pragma once
#include "goal1_native_endpoint.hpp"
namespace cubed_sphere {namespace native_scri_leading {
namespace h=z4c::hyperboloidal;
struct Result {bool valid=false;h::Z4cJet<Real>u{};Real c=0,T=0,w=0,tangential=0,normal=0,scalar_gap=0;};
inline Result Solve(h::Z4cJet<Real>u,const Real n[3]){
 Result out;auto gt=h::Geometry(u.metric),bar=h::Geometry(h::PenroseMetric(u.metric,u.chi));if(!gt.valid||!bar.valid)return out;
 Real v[3]{},q[3]{},nu[3],c=0,T=0,v2=0;for(int i=0;i<3;++i){nu[i]=-2*n[i];for(int j=0;j<3;++j)v[i]+=u.chi.value*gt.inverse[i][j]*n[j];c+=n[i]*v[i];v2+=v[i]*v[i];T+=u.chi.value*gt.inverse[i][i];}T-=v2/c;if(!(c>0&&T>0))return out;
 Real root=std::sqrt(c);u.alpha.value=4*root/T;u.trace.value=-6*root;u.theta.value=0;for(int i=0;i<3;++i)u.beta.value[i]=-4*v[i]/T;
 Real w=-2*root,kappa=10/u.alpha.value,aa=2*kappa,bb=-4*u.chi.value/(3*w),qn=0;for(int i=0;i<3;++i){for(int j=0;j<3;++j)q[i]+=gt.inverse[i][j]*nu[j];qn+=nu[i]*q[i];}Real normal=aa+bb*qn;if(aa==0||normal==0||w+kappa==0)return out;
 Real H[3][3]{},tr=0;for(int i=0;i<3;++i)for(int j=0;j<3;++j){H[i][j]=i==j?-2:0;for(int k=0;k<3;++k)H[i][j]-=bar.connection[k][i][j]*nu[k];tr+=gt.inverse[i][j]*H[i][j];}for(int i=0;i<3;++i)for(int j=0;j<3;++j)H[i][j]-=u.metric.g[i][j]*tr/3;
 // Actual native Lambda uses -= (gradient term + 4 alpha Aup*dOmega).
 Real J[3]{},rhs[3]{},nrhs=0;for(int i=0;i<3;++i){for(int j=0;j<3;++j){rhs[i]-=2./3*gt.inverse[i][j]*(2*u.trace.d[j]+u.theta.d[j]);for(int k=0;k<3;++k)J[i]+=gt.inverse[i][j]*H[j][k]*q[k];}rhs[i]+=4*u.chi.value/w*J[i];nrhs+=nu[i]*rhs[i];}
 Real zup[3]{},zcov[3]{},zn=0;for(int i=0;i<3;++i){zup[i]=rhs[i]/aa-bb*q[i]*nrhs/(aa*normal);zn+=nu[i]*zup[i];u.lambda.value[i]=gt.contracted[i]+2*zup[i];}for(int i=0;i<3;++i)for(int j=0;j<3;++j)zcov[i]+=u.metric.g[i][j]*zup[j];
 for(int i=0;i<3;++i)for(int j=0;j<3;++j)u.a.k[i][j]=-u.chi.value/w*(H[i][j]+zcov[i]*nu[j]+zcov[j]*nu[i]-2./3*u.metric.g[i][j]*zn);
 out.u=u;out.c=c;out.T=T;out.w=w;out.tangential=aa;out.normal=normal;out.scalar_gap=w+kappa;out.valid=true;return out;
}
}}
