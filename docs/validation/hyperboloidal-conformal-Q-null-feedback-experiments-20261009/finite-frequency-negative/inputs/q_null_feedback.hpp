#ifndef SCRATCH_CONFORMAL_Q_NULL_FEEDBACK_HPP_
#define SCRATCH_CONFORMAL_Q_NULL_FEEDBACK_HPP_
// Exploratory value-only off-constraint gauge extension. Geometric storage is P.
#include "z4c/hyperboloidal/layer_gauge.hpp"
#include "factored_base.hpp"
namespace qnf {
namespace hyp=z4c::hyperboloidal;
struct Parameters { double r0=.85,r1=.95,sigma=5; bool physical_inner=false; };
template<class T> T NullDifference(const hyp::LayerPoint<T>&p,const hyp::Z4cJet<T>&u) {
  const auto g=hyp::Geometry(u.metric),h=hyp::Geometry(p.state.metric);
  T db{},bref{},dg{};
  for(int i=0;i<3;++i){db+=(u.beta.value[i]-p.beta[i])*p.domega[i];bref+=p.beta[i]*p.domega[i];
    for(int j=0;j<3;++j)dg+=((u.chi.value-p.state.chi.value)*h.inverse[i][j]
      +u.chi.value*(g.inverse[i][j]-h.inverse[i][j]))*p.domega[i]*p.domega[j];}
  const T wnref=-bref/p.alpha, dwn=(-db-wnref*(u.alpha.value-p.alpha))/u.alpha.value;
  return dg-dwn*(2*wnref+dwn);
}
// alpha^2*(Nraw-Nraw_ref), avoiding a live-lapse quotient.
template<class T> T WeightedNullDifference(const hyp::LayerPoint<T>&p,const hyp::Z4cJet<T>&u){
  const auto g=hyp::Geometry(u.metric),h=hyp::Geometry(p.state.metric);
  T B{},Bh{},db{},dg{};for(int i=0;i<3;++i){B+=u.beta.value[i]*p.domega[i];Bh+=p.beta[i]*p.domega[i];db+=(u.beta.value[i]-p.beta[i])*p.domega[i];
    for(int j=0;j<3;++j)dg+=((u.chi.value-p.state.chi.value)*h.inverse[i][j]+u.chi.value*(g.inverse[i][j]-h.inverse[i][j]))*p.domega[i]*p.domega[j];}
  const T alpha=u.alpha.value,hB=alpha*Bh/p.alpha;
  const T Dmatch=alpha/p.alpha>T(.5)?Bh*(alpha-p.alpha)/p.alpha-db:hB-B;
  return alpha*alpha*dg+Dmatch*(B+hB);
}
template<class T> hyp::GaugeRHSParts<T> Gauge(const hyp::LayerPoint<T>&p,
    const hyp::Z4cJet<T>&u,const hyp::LayerGaugeParameters&g,const Parameters&par={}) {
  const T W=hyp::LayerCoefficients(p.radius,u.alpha.value,g).weight;
  auto inner=g;inner.physical_trace_lapse=true;inner.preferred_source=false;
  // Do not evaluate an unused Q source in the exact physical-P inner branch.
  auto out=par.physical_inner&&W==T(0)?FactoredBaseGauge(p,u,inner):FactoredBaseGauge(p,u,g);
  if(g.physical_trace_lapse||!g.preferred_source||par.r0<g.r1||!(par.r1>par.r0)||par.sigma<0){out.valid=false;return out;}
  if(par.physical_inner){
    const auto phys=FactoredBaseGauge(p,u,inner);
    out.regular.alpha=(1-W)*phys.regular.alpha+W*out.regular.alpha;
    out.pole.alpha=(1-W)*phys.pole.alpha+W*out.pole.alpha;
    out.valid=out.valid&&phys.valid;
  }
  const T v=hyp::SmoothCutoff(p.radius,T(par.r0),T(par.r1)).value;
  if(v==T(0)||par.sigma==0)return out;
  T norm{};for(int i=0;i<3;++i)norm+=p.domega[i]*p.domega[i];
  if(!(norm>T(0))){out.valid=false;return out;}
  const T delta=WeightedNullDifference(p,u);
  for(int i=0;i<3;++i)out.pole.beta[i]+=v*T(par.sigma)*p.domega[i]*delta/norm;
  for(int i=0;i<3;++i)out.valid=out.valid&&Kokkos::isfinite(out.pole.beta[i]);
  return out;
}
template<class T> bool Assemble(const hyp::GaugeRHSParts<T>&q,T O,hyp::GaugeRHS<T>&f){
  if(!hyp::AssembleGaugeInterior(q,O,f))return false;
  for(int i=0;i<3;++i)f.beta[i]+=q.pole.beta[i]/O;
  return true;
}
}
#endif
