#ifndef SCRATCH_Q_NULL_EARLY_FEEDBACK_HPP_
#define SCRATCH_Q_NULL_EARLY_FEEDBACK_HPP_
// Distinct exploratory continuation into 0<W<1. No preferred Box claim there.
#include "inputs/q_null_feedback.hpp"
namespace earlynf {
namespace hyp=z4c::hyperboloidal;
struct Parameters {double sigma=5; int weight=0;}; //0:Wgauge;1:Smooth(.65,.85)
template<class T> T Weight(const hyp::LayerPoint<T>&p,const hyp::Z4cJet<T>&u,const hyp::LayerGaugeParameters&g,const Parameters&par){
 return par.weight==0?hyp::LayerCoefficients(p.radius,u.alpha.value,g).weight:hyp::SmoothCutoff(p.radius,T(.65),T(.85)).value;
}
template<class T> hyp::GaugeRHSParts<T> Gauge(const hyp::LayerPoint<T>&p,const hyp::Z4cJet<T>&u,const hyp::LayerGaugeParameters&g,const Parameters&par={}){
 auto out=qnf::Gauge(p,u,g,{.85,.95,0,true});
 if(par.weight<0||par.weight>1||par.sigma<0){out.valid=false;return out;}
 const T weight=Weight(p,u,g,par);if(weight==T(0)||par.sigma==0)return out;
 T norm{};for(int i=0;i<3;++i)norm+=p.domega[i]*p.domega[i];
 if(!(norm>T(0))){out.valid=false;return out;} // strict admitted support, no floor
 const T delta=qnf::WeightedNullDifference(p,u);
 for(int i=0;i<3;++i){out.pole.beta[i]+=weight*T(par.sigma)*p.domega[i]*delta/norm;out.valid=out.valid&&Kokkos::isfinite(out.pole.beta[i]);}
 return out;
}
}
#endif
