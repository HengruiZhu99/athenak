// Numerical candidate v3: full-gradient canonical momentum, same gauge/reference.
#ifndef GOAL1_CANONICAL_FULL_HPP_
#define GOAL1_CANONICAL_FULL_HPP_
#include "mesh/cubed_sphere/research_canonical_ko.hpp"
namespace cubed_sphere { namespace full_canonical {
namespace h=z4c::hyperboloidal;
using canonical_ko::MetricFrame;using canonical_ko::MetricChart;using canonical_ko::InverseMetricChart;using canonical_ko::Lift;
KOKKOS_INLINE_FUNCTION void Momentum(const h::Z4cJet<Real>&u,const h::Z4cJet<Real>&ref,Real omega,Real pi[10]) {
 const Real alpha=u.alpha.value,chi=u.chi.value;const auto g=h::Geometry(u.metric);Real div=0;
 for(int i=0;i<3;++i)div+=u.beta.d[i][i]-ref.beta.d[i][i];
 pi[0]=-Real(2./3.)*chi*div+Real(2./3.)*alpha*chi*((u.trace.value-ref.trace.value)+2*(u.theta.value-ref.theta.value))/omega;
 Real metric[3][3]{};
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){metric[i][j]=-2*alpha*(u.a.k[i][j]-ref.a.k[i][j])-Real(2./3.)*u.metric.g[i][j]*div;
  for(int k=0;k<3;++k)metric[i][j]+=u.metric.g[k][i]*(u.beta.d[j][k]-ref.beta.d[j][k])+u.metric.g[k][j]*(u.beta.d[i][k]-ref.beta.d[i][k]);}
 MetricChart(MetricFrame(u),metric,pi+1);pi[6]=-alpha*alpha*(u.trace.value-ref.trace.value)/omega;
 for(int i=0;i<3;++i){pi[7+i]=alpha*alpha*chi*(u.lambda.value[i]-ref.lambda.value[i]);
  for(int j=0;j<3;++j)pi[7+i]+=g.inverse[i][j]*(alpha*alpha*(u.chi.d[j]-ref.chi.d[j])/2-alpha*chi*(u.alpha.d[j]-ref.alpha.d[j]));}
}
// Active non-SBP coordinate lines have unit weight in that direction. The fixed
// radial SBP weight cancels between line and cell energy factors.
template<class Sample,class Geometry>KOKKOS_INLINE_FUNCTION
Real ActiveLineTapered(const Sample&q,const Geometry&geometry,int row,int nodes,int axis,Real spacing,Real epsilon) {
 const Real coeff[4]={-1,3,-3,1};Real out=0;const Real volume=ResearchLogicalVolume(geometry(row));
 for(int l=0;l<4;++l){const int start=row-l;if(start<0||start+3>=nodes)continue;Real third=0,line=0,anchor=q(start);
  for(int j=0;j<4;++j){third+=coeff[j]*(q(start+j)-anchor);auto p=geometry(start+j);Real norm=0;for(int a=0;a<3;++a)norm+=p.J[axis][a]*p.J[axis][a];line+=ResearchLogicalVolume(p)*sqrt(norm)/4;}
  const Real z=(start+Real(1.5))/(nodes-1),taper=16*z*z*(1-z)*(1-z);
  out-=epsilon*coeff[l]*line*taper*third/(64*spacing*volume);
 }
 return out;
}
} }
#endif
