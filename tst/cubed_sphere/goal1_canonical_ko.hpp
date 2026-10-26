// Versioned research dissipation. Does not provide incoming or scri equations.
#ifndef GOAL1_CANONICAL_KO_HPP_
#define GOAL1_CANONICAL_KO_HPP_
#include "mesh/cubed_sphere/research_radial_sbp.hpp"
namespace cubed_sphere {
namespace canonical_ko {
namespace h=z4c::hyperboloidal;
struct Frame {Real L[3][3]{},inverse[3][3]{};};
KOKKOS_INLINE_FUNCTION Frame MetricFrame(const h::Z4cJet<Real>&u){
  Frame f;
  for(int i=0;i<3;++i)for(int j=0;j<=i;++j){Real v=u.metric.g[i][j];for(int k=0;k<j;++k)v-=f.L[i][k]*f.L[j][k];
    f.L[i][j]=i==j?sqrt(v):v/f.L[j][j];}
  for(int col=0;col<3;++col)for(int i=0;i<3;++i){Real v=i==col?1:0;for(int j=0;j<i;++j)v-=f.L[i][j]*f.inverse[j][col];f.inverse[i][col]=v/f.L[i][i];}
  return f;
}
KOKKOS_INLINE_FUNCTION void MetricChart(const Frame&f,const Real s[3][3],Real v[5]){
  Real x[3][3]{};
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int k=0;k<3;++k)for(int l=0;l<3;++l)x[i][j]+=f.inverse[i][k]*s[k][l]*f.inverse[j][l];
  v[0]=(x[0][0]-x[1][1])/sqrt(Real(2));v[1]=(x[0][0]+x[1][1]-2*x[2][2])/sqrt(Real(6));
  v[2]=sqrt(Real(2))*x[0][1];v[3]=sqrt(Real(2))*x[0][2];v[4]=sqrt(Real(2))*x[1][2];
}
KOKKOS_INLINE_FUNCTION void InverseMetricChart(const Frame&f,const Real v[5],Real s[3][3]){
  Real x[3][3]{};x[0][0]=v[0]/sqrt(Real(2))+v[1]/sqrt(Real(6));x[1][1]=-v[0]/sqrt(Real(2))+v[1]/sqrt(Real(6));x[2][2]=-2*v[1]/sqrt(Real(6));
  x[0][1]=x[1][0]=v[2]/sqrt(Real(2));x[0][2]=x[2][0]=v[3]/sqrt(Real(2));x[1][2]=x[2][1]=v[4]/sqrt(Real(2));
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){s[i][j]=0;for(int k=0;k<3;++k)for(int l=0;l<3;++l)s[i][j]+=f.L[i][k]*x[k][l]*f.L[j][l];}
}
// Field order chi, five metric-frame TF components, lapse, three Cartesian shift.
KOKKOS_INLINE_FUNCTION void Momentum(const h::Z4cJet<Real>&u,const h::Z4cJet<Real>&ref,
    const Real n[3],Real omega,Real pi[10]){
  const Real alpha=u.alpha.value,chi=u.chi.value;const auto g=h::Geometry(u.metric);const auto frame=MetricFrame(u);
  Real beta_normal[3]{},chi_normal=0,alpha_normal=0,div_normal=0;
  for(int i=0;i<3;++i){chi_normal+=n[i]*(u.chi.d[i]-ref.chi.d[i]);alpha_normal+=n[i]*(u.alpha.d[i]-ref.alpha.d[i]);
    for(int a=0;a<3;++a)beta_normal[i]+=n[a]*(u.beta.d[a][i]-ref.beta.d[a][i]);div_normal+=n[i]*beta_normal[i];}
  const Real dP=u.trace.value-ref.trace.value,dTheta=u.theta.value-ref.theta.value;
  pi[0]=-Real(2./3.)*chi*div_normal+Real(2./3.)*alpha*chi*(dP+2*dTheta)/omega;
  Real metric[3][3]{};
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){metric[i][j]=-2*alpha*(u.a.k[i][j]-ref.a.k[i][j])-Real(2./3.)*u.metric.g[i][j]*div_normal;
    for(int k=0;k<3;++k)metric[i][j]+=u.metric.g[k][i]*n[j]*beta_normal[k]+u.metric.g[k][j]*n[i]*beta_normal[k];}
  MetricChart(frame,metric,pi+1);pi[6]=-alpha*alpha*dP/omega;
  for(int i=0;i<3;++i){pi[7+i]=alpha*alpha*chi*(u.lambda.value[i]-ref.lambda.value[i]);
    for(int j=0;j<3;++j)pi[7+i]+=g.inverse[i][j]*n[j]*(alpha*alpha*chi_normal/2-alpha*chi*alpha_normal);}
}
KOKKOS_INLINE_FUNCTION void Lift(const h::Z4cJet<Real>&u,Real omega,const Real desired[10],h::Z4cRHS<Real>&p){
  const Real a=u.alpha.value,chi=u.chi.value;const auto frame=MetricFrame(u);
  p.trace=-omega*desired[6]/(a*a);p.theta=(3*omega*desired[0]/(2*a*chi)-p.trace)/2;
  Real metric[3][3]{};InverseMetricChart(frame,desired+1,metric);
  for(int i=0;i<3;++i){p.lambda[i]=desired[7+i]/(a*a*chi);for(int j=0;j<3;++j)p.a[i][j]=-metric[i][j]/(2*a);}
}
template<class Sample,class Geometry>KOKKOS_INLINE_FUNCTION
Real TaperedRadial(const Sample&q,const Geometry&geometry,int row,const ResearchRadialSBP&op,Real ds,Real epsilon){
  const Real c[4]={-1,3,-3,1};Real out=0;const int N=op.d1.extent_int(0);const Real volume=ResearchLogicalVolume(geometry(row));
  for(int l=0;l<4;++l){const int start=row-l;if(start<0||start+3>=N)continue;Real third=0,line=0,anchor=q(start,0,0);
    for(int j=0;j<4;++j){third+=c[j]*(q(start+j,0,0)-anchor);const auto p=geometry(start+j);Real norm=0;
      for(int a=0;a<3;++a)norm+=p.J[0][a]*p.J[0][a];line+=ResearchLogicalVolume(p)*sqrt(norm)/4;}
    const Real z=(start+Real(1.5))/(N-1),taper=16*z*z*(1-z)*(1-z);
    out-=epsilon*c[l]*line*taper*third/(64*ds*op.norm(row)*volume);
  }
  return out;
}
} // namespace canonical_ko
} // namespace cubed_sphere
#endif
