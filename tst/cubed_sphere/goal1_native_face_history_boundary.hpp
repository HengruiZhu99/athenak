// Native second-order finite-cut characteristic boundary selection only.
// Native A/P/Theta/Lambda RHS penalties, no new evolved fields/gradient equations.
// Reuses verified native frame/K rows from pre-reduction finite-cut diagnostics.
#pragma once
#include <limits>
#include "z4c/hyperboloidal/cartesian_patch.hpp"
namespace cubed_sphere { namespace native_face_boundary {
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
struct Boundary {
 Real n[3]{},beta=0,c=0,omega=0,Jin[4][10]{},Jout[4][10]{},F[6][10]{},K[10][10]{};
 Square10 solve;
 KOKKOS_INLINE_FUNCTION bool Build(const h::Z4cJet<Real>&u,const Real direction[3],Real Omega,bool regular_basis=false) {
  if(!(Omega>0) || !(u.alpha.value>0) || !(u.chi.value>0))return false;
  beta=0;for(int i=0;i<6;++i)for(int j=0;j<10;++j)F[i][j]=0;
  omega=Omega;auto g=h::Geometry(u.metric);if(!g.valid || !(u.metric.g[0][0]>0) || !(u.metric.g[0][0]*u.metric.g[1][1]-u.metric.g[0][1]*u.metric.g[0][1]>0))return false;auto frame=MetricFrame(u);
  Real nup[3]{},length=0;
  for(int i=0;i<3;++i){n[i]=direction[i];beta+=n[i]*u.beta.value[i];for(int j=0;j<3;++j)nup[i]+=g.inverse[i][j]*direction[j];length+=n[i]*nup[i];}
  if(!(length>0))return false;c=u.alpha.value*sqrt(u.chi.value*length);
  // Metric-orthonormal tangents to the covector, using the best conditioned axis.
  Real unit[3],e[3]{},f[3]{};for(int i=0;i<3;++i)unit[i]=nup[i]/sqrt(length);
  if(regular_basis){
    // Tangent seed selection depends on the fixed geometric covector only.
    // Both seeds lie in ker(n); Gram-Schmidt uses the live positive metric.
    int axis=0;Real nn=0;for(int i=0;i<3;++i){nn+=n[i]*n[i];if(fabs(n[i])<fabs(n[axis]))axis=i;}
    Real seed[3],other[3];for(int i=0;i<3;++i)seed[i]=(i==axis?1:0)-n[i]*n[axis]/nn;
    for(int i=0;i<3;++i)other[i]=n[(i+1)%3]*seed[(i+2)%3]-n[(i+2)%3]*seed[(i+1)%3];
    Real norm=0;for(int i=0;i<3;++i)for(int j=0;j<3;++j)norm+=seed[i]*u.metric.g[i][j]*seed[j];
    if(!(norm>0))return false;for(int i=0;i<3;++i)e[i]=seed[i]/sqrt(norm);
    Real dot=0;for(int i=0;i<3;++i)for(int j=0;j<3;++j)dot+=e[i]*u.metric.g[i][j]*other[j];
    for(int i=0;i<3;++i)other[i]-=e[i]*dot;
    norm=0;for(int i=0;i<3;++i)for(int j=0;j<3;++j)norm+=other[i]*u.metric.g[i][j]*other[j];
    if(!(norm>0))return false;for(int i=0;i<3;++i)f[i]=other[i]/sqrt(norm);
  }else{
  for(int pass=0;pass<2;++pass){Real best=-1;
   for(int axis=0;axis<3;++axis){Real v[3],normal_dot=0,tangent_dot=0;
    for(int i=0;i<3;++i){normal_dot+=unit[i]*u.metric.g[i][axis];tangent_dot+=e[i]*u.metric.g[i][axis];}
    for(int i=0;i<3;++i)v[i]=(i==axis?1:0)-unit[i]*normal_dot-(pass?e[i]*tangent_dot:0);
    Real norm=0;for(int i=0;i<3;++i)for(int j=0;j<3;++j)norm+=v[i]*u.metric.g[i][j]*v[j];
    if(norm>best && norm>0){best=norm;for(int i=0;i<3;++i)(pass?f[i]:e[i])=v[i]/sqrt(norm);}}
   if(!(best>0))return false;
  }
  }
  const Real alpha=u.alpha.value,chi=u.chi.value;
  for(int col=0;col<10;++col){Real q[10]{},m[3][3]{};q[col]=1;InverseMetricChart(frame,q+1,m);
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
  return true;
 }

 // Evaluate four actual native constraints and six free incoming projections.
 // Arrays are local characteristic analysis/BC coefficients, never state fields.
 KOKKOS_INLINE_FUNCTION void Rows(const h::Z4cJet<Real>&u,const h::Z4cJet<Real>&ref,Real rows[10])const {
  auto g=h::Geometry(u.metric);const auto frame=MetricFrame(u);
  Real normal[10]{},strain[3][3]{},normal_metric[3][3]{},v[3]{},length=0,div=0;
  for(int i=0;i<3;++i){for(int j=0;j<3;++j)v[i]+=g.inverse[i][j]*n[j];length+=n[i]*v[i];}
  for(int i=0;i<3;++i)v[i]/=length;
  for(int a=0;a<3;++a){normal[6]+=v[a]*(u.alpha.d[a]-ref.alpha.d[a]);div+=u.beta.d[a][a]-ref.beta.d[a][a];
   for(int i=0;i<3;++i){normal[7+i]+=v[a]*(u.beta.d[a][i]-ref.beta.d[a][i]);for(int j=0;j<3;++j)normal_metric[i][j]+=v[a]*(u.metric.dg[a][i][j]-ref.metric.dg[a][i][j]);}}
  MetricChart(frame,normal_metric,normal+1);
  const Real alpha=u.alpha.value,chi=u.chi.value;
  rows[0]=u.theta.value/omega;for(int i=0;i<3;++i)rows[1+i]=u.lambda.value[i]-g.contracted[i];
  rows[4]=(normal[6]-alpha*alpha*(u.trace.value-ref.trace.value)/(omega*c))/2;
  for(int i=0;i<3;++i){Real rate=alpha*alpha*chi*(u.lambda.value[i]-ref.lambda.value[i]);
   for(int j=0;j<3;++j)rate+=g.inverse[i][j]*(alpha*alpha*(u.chi.d[j]-ref.chi.d[j])/2-alpha*chi*(u.alpha.d[j]-ref.alpha.d[j]));
   rows[5+i]=(normal[7+i]+rate/c)/2;}
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){strain[i][j]=-2*alpha*(u.a.k[i][j]-ref.a.k[i][j])-Real(2./3.)*u.metric.g[i][j]*div;
   for(int k=0;k<3;++k)strain[i][j]+=u.metric.g[k][i]*(u.beta.d[j][k]-ref.beta.d[j][k])+u.metric.g[k][j]*(u.beta.d[i][k]-ref.beta.d[i][k]);}
  Real rate[5];MetricChart(frame,strain,rate);
  for(int i=0;i<2;++i){rows[8+i]=0;for(int j=0;j<5;++j)rows[8+i]+=F[4+i][1+j]*(normal[1+j]+rate[j]/c)/2;}
 }
 // Convert the boundary residual to a native RHS penalty. This is a dynamic
 // penalty, never an instantaneous field/derivative reset. No exact-scri use.
 KOKKOS_INLINE_FUNCTION void Penalty(const h::Z4cJet<Real>&u,const Real rows[10],Real factor,h::Z4cRHS<Real>&rhs)const {
  Real a[10];solve.Solve(rows,a);for(int i=0;i<10;++i)a[i]*=-factor*(c+beta)*2*c;
  NativeLift(u,omega,a,rhs);
 }
 KOKKOS_INLINE_FUNCTION static void NativeLift(const h::Z4cJet<Real>&u,Real Omega,const Real a[10],h::Z4cRHS<Real>&rhs){
  const Real alpha=u.alpha.value,chi=u.chi.value;
  rhs.trace=-Omega*a[6]/(alpha*alpha);rhs.theta=(3*Omega*a[0]/(2*alpha*chi)-rhs.trace)/2;
  Real tensor[3][3]{};InverseMetricChart(MetricFrame(u),a+1,tensor);
  for(int i=0;i<3;++i){rhs.lambda[i]=a[7+i]/(alpha*alpha*chi);for(int j=0;j<3;++j)rhs.a[i][j]=-tensor[i][j]/(2*alpha);}
 }
};
}}
