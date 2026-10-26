// Versioned finite-radius hierarchy. Not a regular exact-scri prescription.
#ifndef GOAL1_BOUNDARY_FULL_HPP_
#define GOAL1_BOUNDARY_FULL_HPP_
#include "mesh/cubed_sphere/research_boundary_hierarchy.hpp"
#include "mesh/cubed_sphere/research_canonical_full.hpp"
namespace cubed_sphere { namespace boundary_full {
namespace h=z4c::hyperboloidal;namespace ck=canonical_ko;
struct Hierarchy:boundary_hierarchy::Hierarchy {
 KOKKOS_INLINE_FUNCTION void Gradient(const h::Z4cJet<Real>&u,const h::Z4cJet<Real>&ref,Real d[3][10])const {
  const auto frame=ck::MetricFrame(u);
  for(int a=0;a<3;++a){Real m[3][3]{};d[a][0]=u.chi.d[a]-ref.chi.d[a];d[a][6]=u.alpha.d[a]-ref.alpha.d[a];
   for(int i=0;i<3;++i){d[a][7+i]=u.beta.d[a][i]-ref.beta.d[a][i];for(int j=0;j<3;++j)m[i][j]=u.metric.dg[a][i][j]-ref.metric.dg[a][i][j];}ck::MetricChart(frame,m,d[a]+1);}
 }
 KOKKOS_INLINE_FUNCTION void Conormal(const h::Z4cJet<Real>&u,Real v[3])const {
  auto g=h::Geometry(u.metric);Real norm=0;for(int i=0;i<3;++i){v[i]=0;for(int j=0;j<3;++j)v[i]+=g.inverse[i][j]*n[j];norm+=n[i]*v[i];}for(int i=0;i<3;++i)v[i]/=norm;
 }
 KOKKOS_INLINE_FUNCTION void ConfigurationNormal(const h::Z4cJet<Real>&u,const h::Z4cJet<Real>&ref,Real z[10])const {
  Real gradient[3][10]{},v[3];Gradient(u,ref,gradient);Conormal(u,v);for(int j=0;j<10;++j){z[j]=0;for(int a=0;a<3;++a)z[j]+=v[a]*gradient[a][j];}
 }
 KOKKOS_INLINE_FUNCTION void Target(const h::Z4cJet<Real>&u,const h::Z4cJet<Real>&ref,const Real free_data[6],Real incoming[10],Real outgoing[10],Real target[10],Real forcing[4])const {
  Real z[10]{},pi[10],U[4],b[10]{};ConfigurationNormal(u,ref,z);full_canonical::Momentum(u,ref,omega,pi);ConstraintValues(u,U);
  for(int j=0;j<10;++j){incoming[j]=(z[j]+pi[j]/c)/2;outgoing[j]=(z[j]-pi[j]/c)/2;}
  for(int i=0;i<4;++i){forcing[i]=U[i];for(int j=0;j<10;++j)forcing[i]-=Jin[i][j]*incoming[j]+Jout[i][j]*outgoing[j];
   b[i]=-forcing[i];for(int j=0;j<10;++j)b[i]-=Jout[i][j]*outgoing[j];}
  for(int i=0;i<6;++i)b[4+i]=free_data[i];solve.Solve(b,target);
 }
 // Full principal constraint rows in w=(Pi_full,q_x,q_y,q_z), all frame fields.
 KOKKOS_INLINE_FUNCTION void PrincipalRows(const h::Z4cJet<Real>&u,Real U[4][40])const {
  const Real alpha=u.alpha.value,chi=u.chi.value;auto g=h::Geometry(u.metric);auto frame=ck::MetricFrame(u);
  for(int i=0;i<4;++i)for(int j=0;j<40;++j)U[i][j]=0;
  U[0][0]=3/(4*alpha*chi);U[0][6]=1/(2*alpha*alpha);for(int i=0;i<3;++i)U[1+i][7+i]=1/(alpha*alpha*chi);
  for(int a=0;a<3;++a)for(int col=0;col<10;++col){Real q[10]{},m[3][3]{};q[col]=1;ck::InverseMetricChart(frame,q+1,m);U[0][10+10*a+col]=q[7+a]/(2*alpha);
   for(int i=0;i<3;++i){Real gamma=0;for(int j=0;j<3;++j)for(int k=0;k<3;++k)gamma+=g.inverse[i][j]*g.inverse[a][k]*m[j][k];
    U[1+i][10+10*a+col]=-g.inverse[i][a]*(q[0]/(2*chi)-q[6]/alpha)-gamma;}}
 }
 // Physical normal principal flux plus the homogeneous full tangential SAT.
 KOKKOS_INLINE_FUNCTION void PrincipalPort(const h::Z4cJet<Real>&u,Real flux[40][40],Real weight[40][40],Real lift[10][40])const {
  auto g=h::Geometry(u.metric);Real a[3][3],v[3],U[4][40]{};Conormal(u,v);PrincipalRows(u,U);
  for(int i=0;i<40;++i)for(int j=0;j<40;++j){weight[i][j]=0;flux[i][j]=0;}
  for(int j=0;j<10;++j)weight[j][j]=1;
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){a[i][j]=u.alpha.value*u.alpha.value*u.chi.value*g.inverse[i][j];for(int col=0;col<10;++col)weight[10+10*i+col][10+10*j+col]=a[i][j];}
  for(int col=0;col<40;++col){Real b[10]{},x[10];for(int i=0;i<4;++i)b[i]=U[i][col];
   int field=col%10;const Real factor=col<10?1/(2*c):v[(col-10)/10]/2;
   for(int i=0;i<6;++i)b[4+i]=F[i][field]*factor;solve.Solve(b,x);for(int j=0;j<10;++j)lift[j][col]=2*c*x[j];}
  for(int i=0;i<40;++i)for(int j=0;j<40;++j)flux[i][j]=beta*weight[i][j];
  for(int axis=0;axis<3;++axis){Real an=0;for(int j=0;j<3;++j)an+=n[j]*a[j][axis];for(int col=0;col<10;++col){flux[col][10+10*axis+col]+=an;flux[10+10*axis+col][col]+=an;}}
  for(int i=0;i<10;++i)for(int j=0;j<40;++j){flux[i][j]-=(c+beta)*lift[i][j];flux[j][i]-=(c+beta)*lift[i][j];}
 }
 KOKKOS_INLINE_FUNCTION bool PortCertificate(const h::Z4cJet<Real>&u,Real &minimum_pivot)const {
  if(!(beta<0) || !(c+beta>0))return false;
  Real flux[40][40],weight[40][40],lift[10][40];PrincipalPort(u,flux,weight,lift);Real L[40][40]{};minimum_pivot=std::numeric_limits<Real>::max();
  const Real speed=fmax(Real(1),fmax(fabs(beta),c));
  for(int i=0;i<40;++i)for(int j=0;j<=i;++j){Real value=-flux[i][j]/(speed*sqrt(weight[i][i]*weight[j][j]));for(int k=0;k<j;++k)value-=L[i][k]*L[j][k];
   if(i==j){if(!(value>0) || !Kokkos::isfinite(value))return false;minimum_pivot=fmin(minimum_pivot,value);L[i][j]=sqrt(value);}else L[i][j]=value/L[j][j];}
  return true;
 }
 KOKKOS_INLINE_FUNCTION void Residual(const h::Z4cJet<Real>&u,const h::Z4cJet<Real>&ref,const Real free_data[6],Real e[10])const {
  Real z[10]{},pi[10],U[4],b[10]{};ConfigurationNormal(u,ref,z);full_canonical::Momentum(u,ref,omega,pi);ConstraintValues(u,U);
  for(int i=0;i<4;++i)b[i]=U[i];for(int i=0;i<6;++i){b[4+i]=-free_data[i];for(int j=0;j<10;++j)b[4+i]+=F[i][j]*(z[j]+pi[j]/c)/2;}
  solve.Solve(b,e);for(int j=0;j<10;++j)e[j]*=2*c;
 }
};
// Point-fixture check only; this must never be used to reset native derivatives.
KOKKOS_INLINE_FUNCTION void IncomingFixtureCorrection(h::Z4cJet<Real>&u,const h::Z4cJet<Real>&ref,const Hierarchy&b,const Real incoming[10],const Real target[10]) {
 Real delta[10],m[3][3]{},pi[10],after[10],desired[10];for(int j=0;j<10;++j)delta[j]=target[j]-incoming[j];full_canonical::Momentum(u,ref,b.omega,pi);ck::InverseMetricChart(ck::MetricFrame(u),delta+1,m);
 for(int a=0;a<3;++a){u.chi.d[a]+=b.n[a]*delta[0];u.alpha.d[a]+=b.n[a]*delta[6];for(int i=0;i<3;++i){u.beta.d[a][i]+=b.n[a]*delta[7+i];for(int j=0;j<3;++j)u.metric.dg[a][i][j]+=b.n[a]*m[i][j];}}
 full_canonical::Momentum(u,ref,b.omega,after);for(int j=0;j<10;++j)desired[j]=pi[j]+b.c*delta[j]-after[j];h::Z4cRHS<Real> p{};ck::Lift(u,b.omega,desired,p);
 u.trace.value+=p.trace;u.theta.value+=p.theta;for(int i=0;i<3;++i){u.lambda.value[i]+=p.lambda[i];for(int j=0;j<3;++j)u.a.k[i][j]+=p.a[i][j];}
}
} // boundary_full
// Interface for the unchanged archived point-driver signatures.
namespace boundary_full_chart {
using canonical_ko::MetricFrame;using canonical_ko::MetricChart;using canonical_ko::InverseMetricChart;using canonical_ko::Lift;
KOKKOS_INLINE_FUNCTION void Momentum(const z4c::hyperboloidal::Z4cJet<Real>&u,const z4c::hyperboloidal::Z4cJet<Real>&ref,const Real[3],Real omega,Real pi[10]){full_canonical::Momentum(u,ref,omega,pi);}
} }
#endif
