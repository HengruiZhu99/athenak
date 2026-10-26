// Independent Cartesian scalar jets for isolated native donor/operator tests.
#ifndef GOAL1_OPERATOR_ORACLE_HPP_
#define GOAL1_OPERATOR_ORACLE_HPP_
#include "mesh/cubed_sphere/conformal_bridge.hpp"
namespace cubed_sphere {namespace operator_oracle {
namespace h=z4c::hyperboloidal;
KOKKOS_INLINE_FUNCTION h::ScalarJet<Real> Function(const Real x[3],int kind) {
 h::ScalarJet<Real> f{};const Real k[3]={1.1,-.9,.7},p[3]={.3,-.2,.4};
 if(kind==0){f.value=1;return f;}
 if(kind==1){f.value=.2;for(int i=0;i<3;++i){f.value+=k[i]*x[i];f.d[i]=k[i];}return f;}
 if(kind==2){f.value=.3+x[0]*x[0]+.4*x[0]*x[1]-.7*x[1]*x[2]+.2*x[0]*x[2]+.1*x[2]*x[2];
  f.d[0]=2*x[0]+.4*x[1]+.2*x[2];f.d[1]=.4*x[0]-.7*x[2];f.d[2]=-.7*x[1]+.2*x[0]+.2*x[2];
  f.dd[0][0]=2;f.dd[2][2]=.2;f.dd[0][1]=f.dd[1][0]=.4;f.dd[0][2]=f.dd[2][0]=.2;f.dd[1][2]=f.dd[2][1]=-.7;return f;}
 Real phase=0,power=0;for(int i=0;i<3;++i){phase+=k[i]*x[i];power+=p[i]*x[i];}const Real s=sin(phase),c=cos(phase),e=exp(power);
 f.value=s+.2*e+.1*x[0]*x[1]*x[2]+.07*(x[0]*x[0]-x[1]*x[1]);
 for(int i=0;i<3;++i){f.d[i]=k[i]*c+.2*p[i]*e+.1*x[(i+1)%3]*x[(i+2)%3]+(i==0?.14*x[0]:(i==1?-.14*x[1]:0));
  for(int j=0;j<3;++j)f.dd[i][j]=-k[i]*k[j]*s+.2*p[i]*p[j]*e+(i!=j?.1*x[3-i-j]:0)+(i==j?(i==0?.14:(i==1?-.14:0)):0);}
 return f;
}
KOKKOS_INLINE_FUNCTION Real Coefficient(int n){return (n%2?-1:1)*(1+Real(n)/32);}
KOKKOS_INLINE_FUNCTION h::ScalarJet<Real> Component(const h::Z4cJet<Real>&u,int n) {
 if(n==0)return u.chi;if(n==7)return u.trace;if(n==17)return u.theta;if(n==18)return u.alpha;
 h::ScalarJet<Real> s{};const int a[6]={0,0,0,1,1,2},b[6]={0,1,2,1,2,2};
 if(n>=1&&n<=6){const int i=a[n-1],j=b[n-1];s.value=u.metric.g[i][j];for(int d=0;d<3;++d){s.d[d]=u.metric.dg[d][i][j];for(int e=0;e<3;++e)s.dd[d][e]=u.metric.ddg[d][e][i][j];}}
 else if(n>=8&&n<=13){const int i=a[n-8],j=b[n-8];s.value=u.a.k[i][j];for(int d=0;d<3;++d)s.d[d]=u.a.dk[d][i][j];}
 else {const bool shift=n>=19;const auto &v=shift?u.beta:u.lambda;int component=n-(shift?19:14);s.value=v.value[component];for(int d=0;d<3;++d){s.d[d]=v.d[d][component];for(int e=0;e<3;++e)s.dd[d][e]=v.dd[d][e][component];}}
 return s;
}
} }
#endif
