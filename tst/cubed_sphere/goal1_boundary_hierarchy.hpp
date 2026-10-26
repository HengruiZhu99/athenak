// Finite-radius diagnostic hierarchy only. No native SAT or scri equations.
#ifndef GOAL1_BOUNDARY_HIERARCHY_HPP_
#define GOAL1_BOUNDARY_HIERARCHY_HPP_
#include <limits>
#include "mesh/cubed_sphere/research_canonical_ko.hpp"
namespace cubed_sphere { namespace boundary_hierarchy {
namespace h=z4c::hyperboloidal; namespace ck=canonical_ko;
struct Square10 {
  Real a[10][10]{},scale[10]{};int pivot[10]{};bool valid=false;
  KOKKOS_INLINE_FUNCTION bool Factor(const Real rows[10][10]) {
    valid=false;
    for(int i=0;i<10;++i){Real norm=0;for(int j=0;j<10;++j)norm+=rows[i][j]*rows[i][j];scale[i]=sqrt(norm);
      if(!(scale[i]>0))return false;for(int j=0;j<10;++j)a[i][j]=rows[i][j]/scale[i];}
    for(int k=0;k<10;++k){int p=k;for(int i=k+1;i<10;++i)if(fabs(a[i][k])>fabs(a[p][k]))p=i;pivot[k]=p;
      if(!(fabs(a[p][k])>64*std::numeric_limits<Real>::epsilon()))return false;
      if(p!=k)for(int j=0;j<10;++j){Real t=a[k][j];a[k][j]=a[p][j];a[p][j]=t;}
      for(int i=k+1;i<10;++i){a[i][k]/=a[k][k];for(int j=k+1;j<10;++j)a[i][j]-=a[i][k]*a[k][j];}}
    valid=true;return true;
  }
  KOKKOS_INLINE_FUNCTION void Solve(const Real b[10],Real x[10])const {
    for(int i=0;i<10;++i)x[i]=b[i]/scale[i];
    for(int k=0;k<10;++k){int p=pivot[k];if(p!=k){Real t=x[k];x[k]=x[p];x[p]=t;}}
    for(int k=0;k<10;++k)for(int i=k+1;i<10;++i)x[i]-=a[i][k]*x[k];
    for(int i=9;i>=0;--i){for(int j=i+1;j<10;++j)x[i]-=a[i][j]*x[j];x[i]/=a[i][i];}
  }
};
struct Hierarchy {
 Real n[3]{},beta=0,c=0,omega=0,Jin[4][10]{},Jout[4][10]{},F[6][10]{},K[10][10]{},R[10][10]{};
 Square10 solve;
 KOKKOS_INLINE_FUNCTION bool Build(const h::Z4cJet<Real>&u,const Real direction[3],Real Omega) {
  if(!(Omega>0) || !(u.alpha.value>0) || !(u.chi.value>0))return false;
  beta=0;for(int i=0;i<6;++i)for(int j=0;j<10;++j)F[i][j]=0;
  omega=Omega;auto g=h::Geometry(u.metric);if(!g.valid || !(u.metric.g[0][0]>0) || !(u.metric.g[0][0]*u.metric.g[1][1]-u.metric.g[0][1]*u.metric.g[0][1]>0))return false;auto frame=ck::MetricFrame(u);
  Real nup[3]{},length=0;
  for(int i=0;i<3;++i){n[i]=direction[i];beta+=n[i]*u.beta.value[i];for(int j=0;j<3;++j)nup[i]+=g.inverse[i][j]*direction[j];length+=n[i]*nup[i];}
  if(!(length>0))return false;c=u.alpha.value*sqrt(u.chi.value*length);
  // Metric-orthonormal tangents to the covector, using the best conditioned axis.
  Real unit[3],e[3]{},f[3]{};for(int i=0;i<3;++i)unit[i]=nup[i]/sqrt(length);
  for(int pass=0;pass<2;++pass){Real best=-1;
   for(int axis=0;axis<3;++axis){Real v[3],normal_dot=0,tangent_dot=0;
    for(int i=0;i<3;++i){normal_dot+=unit[i]*u.metric.g[i][axis];tangent_dot+=e[i]*u.metric.g[i][axis];}
    for(int i=0;i<3;++i)v[i]=(i==axis?1:0)-unit[i]*normal_dot-(pass?e[i]*tangent_dot:0);
    Real norm=0;for(int i=0;i<3;++i)for(int j=0;j<3;++j)norm+=v[i]*u.metric.g[i][j]*v[j];
    if(norm>best && norm>0){best=norm;for(int i=0;i<3;++i)(pass?f[i]:e[i])=v[i]/sqrt(norm);}}
   if(!(best>0))return false;
  }
  const Real alpha=u.alpha.value,chi=u.chi.value;
  for(int col=0;col<10;++col){Real q[10]{},m[3][3]{};q[col]=1;ck::InverseMetricChart(frame,q+1,m);
   Real div=0;for(int i=0;i<3;++i)div+=n[i]*q[7+i];
   const Real scalar=3*q[0]/(4*alpha*chi)+q[6]/(2*alpha*alpha);
   Jin[0][col]=c*scalar+div/(2*alpha);Jout[0][col]=-c*scalar+div/(2*alpha);
   for(int i=0;i<3;++i){Real gamma=0;for(int j=0;j<3;++j)for(int k=0;k<3;++k)gamma+=g.inverse[i][j]*nup[k]*m[j][k];
    Real common=-nup[i]*(q[0]/(2*chi)-q[6]/alpha)-gamma;
    Jin[1+i][col]=common+c*q[7+i]/(alpha*alpha*chi);Jout[1+i][col]=common-c*q[7+i]/(alpha*alpha*chi);}
   F[0][col]=q[6];for(int i=0;i<3;++i)F[1+i][col]=q[7+i];
   for(int i=0;i<3;++i)for(int j=0;j<3;++j){F[4][col]+=(e[i]*e[j]-f[i]*f[j])*m[i][j]/sqrt(Real(2));F[5][col]+=(e[i]*f[j]+f[i]*e[j])*m[i][j]/sqrt(Real(2));}
   for(int i=0;i<4;++i)K[i][col]=Jin[i][col];for(int i=0;i<6;++i)K[4+i][col]=F[i][col];
  }
  if(!solve.Factor(K))return false;
  for(int col=0;col<10;++col){Real b[10]{},x[10];for(int i=0;i<4;++i)b[i]=-Jout[i][col];solve.Solve(b,x);for(int i=0;i<10;++i)R[i][col]=x[i];}
  return true;
 }
 KOKKOS_INLINE_FUNCTION void ConfigurationNormal(const h::Z4cJet<Real>&u,const h::Z4cJet<Real>&ref,Real z[10])const {
  Real m[3][3]{};for(int a=0;a<3;++a){z[0]+=n[a]*(u.chi.d[a]-ref.chi.d[a]);z[6]+=n[a]*(u.alpha.d[a]-ref.alpha.d[a]);
   for(int i=0;i<3;++i){z[7+i]+=n[a]*(u.beta.d[a][i]-ref.beta.d[a][i]);for(int j=0;j<3;++j)m[i][j]+=n[a]*(u.metric.dg[a][i][j]-ref.metric.dg[a][i][j]);}}
  ck::MetricChart(ck::MetricFrame(u),m,z+1);
 }
 KOKKOS_INLINE_FUNCTION void ConstraintValues(const h::Z4cJet<Real>&u,Real U[4])const {
  U[0]=u.theta.value/omega;auto g=h::Geometry(u.metric);for(int i=0;i<3;++i)U[1+i]=u.lambda.value[i]-g.contracted[i];
 }
 KOKKOS_INLINE_FUNCTION void Target(const h::Z4cJet<Real>&u,const h::Z4cJet<Real>&ref,const Real free_data[6],Real incoming[10],Real outgoing[10],Real target[10],Real forcing[4])const {
  Real z[10]{},pi[10],U[4],b[10]{};ConfigurationNormal(u,ref,z);ck::Momentum(u,ref,n,omega,pi);ConstraintValues(u,U);
  for(int j=0;j<10;++j){incoming[j]=(z[j]+pi[j]/c)/2;outgoing[j]=(z[j]-pi[j]/c)/2;}
  for(int i=0;i<4;++i){forcing[i]=U[i];for(int j=0;j<10;++j)forcing[i]-=Jin[i][j]*incoming[j]+Jout[i][j]*outgoing[j];
   b[i]=-forcing[i];for(int j=0;j<10;++j)b[i]-=Jout[i][j]*outgoing[j];}
  for(int i=0;i<6;++i)b[4+i]=free_data[i];solve.Solve(b,target);
 }
};
// Point-fixture operation only: modifies derivative jets, never an evolution.
KOKKOS_INLINE_FUNCTION void IncomingFixtureCorrection(h::Z4cJet<Real>&u,const h::Z4cJet<Real>&ref,const Hierarchy&b,const Real incoming[10],const Real target[10]) {
 Real delta[10],m[3][3]{},pi[10],after[10],desired[10];for(int j=0;j<10;++j)delta[j]=target[j]-incoming[j];ck::Momentum(u,ref,b.n,b.omega,pi);
 ck::InverseMetricChart(ck::MetricFrame(u),delta+1,m);
 for(int a=0;a<3;++a){u.chi.d[a]+=b.n[a]*delta[0];u.alpha.d[a]+=b.n[a]*delta[6];
  for(int i=0;i<3;++i){u.beta.d[a][i]+=b.n[a]*delta[7+i];for(int j=0;j<3;++j)u.metric.dg[a][i][j]+=b.n[a]*m[i][j];}}
 ck::Momentum(u,ref,b.n,b.omega,after);for(int j=0;j<10;++j)desired[j]=pi[j]+b.c*delta[j]-after[j];h::Z4cRHS<Real> p{};ck::Lift(u,b.omega,desired,p);
 u.trace.value+=p.trace;u.theta.value+=p.theta;for(int i=0;i<3;++i){u.lambda.value[i]+=p.lambda[i];for(int j=0;j<3;++j)u.a.k[i][j]+=p.a[i][j];}
}
} } // cubed_sphere::boundary_hierarchy
#endif
