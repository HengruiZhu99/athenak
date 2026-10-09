#ifndef SCRATCH_NULL_FEEDBACK_HPP_
#define SCRATCH_NULL_FEEDBACK_HPP_
#include <cmath>
#include <stdexcept>
#include "z4c/hyperboloidal/cartesian_patch.hpp"
namespace research {
namespace hyp=z4c::hyperboloidal;
struct Parameters { double source_r0=.85, source_r1=.95, sigma=5; };

// Scratch only. The source weight starts inside the W=1 harmonic collar.
inline hyp::GaugeRHSParts<double> Gauge(
    const hyp::LayerPoint<double>& p,const hyp::Z4cJet<double>& u,
    const hyp::LayerGaugeParameters& g,const Parameters& par={}) {
  if(!g.physical_trace_lapse||g.preferred_source||par.source_r0<g.r1||
      !(par.source_r0<par.source_r1)||par.sigma<0)
    throw std::runtime_error("invalid scratch source parameters");
  auto out=hyp::InteriorLayerGauge(p,u,g);
  if(!out.valid)return out;
  const double sourceweight=hyp::SmoothCutoff(p.radius,par.source_r0,par.source_r1).value;
  if(sourceweight==0)return out;
  const auto c=hyp::LayerCoefficients(p.radius,u.alpha.value,g);
  if(c.weight!=1)throw std::runtime_error("source must stay in harmonic collar");
  const auto geo=hyp::Geometry(u.metric),refgeo=hyp::Geometry(p.state.metric);
  const double alpha=u.alpha.value, chi=u.chi.value, da=alpha-p.alpha;
  double dbomega=0,brefomega=0,refadvalpha=0,norm=0,bdomega=0;
  for(int i=0;i<3;++i){
    dbomega+=(u.beta.value[i]-p.beta[i])*p.domega[i];
    brefomega+=p.beta[i]*p.domega[i];
    refadvalpha+=p.beta[i]*p.dalpha[i];
    norm+=p.domega[i]*p.domega[i];
    bdomega+=u.beta.value[i]*p.domega[i];
  }
  if(!(norm>0))throw std::runtime_error("nonzero Omega gradient required");
  const double wnref=-brefomega/p.alpha;
  const double dwn=(-dbomega-wnref*da)/alpha;
  const double f0refpole=-p.omega*p.k_bar/p.alpha;
  const double matching=alpha*dbomega+brefomega*da;
  // In W=1, the live-P contribution cancels exactly in F0's pole.
  // Factor the reference and quotient difference before evaluating it.
  const double f0pole=f0refpole+(3*dwn-f0refpole*da)/alpha+
    (matching+g.scri_lapse_damping*(alpha+p.alpha)*da)/(alpha*alpha*alpha);
  const double f0reg=(refadvalpha+alpha*c.nu*std::log1p(da/p.alpha))
                     /(alpha*alpha*alpha);
  double deltaG=0,hessian4=0,contraction=0;
  for(int i=0;i<3;++i){
    double refadv=0;for(int j=0;j<3;++j)refadv+=p.beta[j]*p.state.beta.d[j][i];
    double source=chi*p.state.lambda.value[i]+(refadv+c.eta*(u.beta.value[i]-p.beta[i]))
                    /(alpha*alpha)-u.beta.value[i]*f0reg;
    for(int j=0;j<3;++j){
      deltaG+=((chi-p.state.chi.value)*refgeo.inverse[i][j]
              +chi*(geo.inverse[i][j]-refgeo.inverse[i][j]))*p.domega[i]*p.domega[j];
      source+=chi*geo.inverse[i][j]*(p.state.chi.d[j]/(2*p.state.chi.value)-p.dalpha[j]/p.alpha);
      hessian4+=(chi*geo.inverse[i][j]-u.beta.value[i]*u.beta.value[j]/(alpha*alpha))
                 *p.omega_hessian[i][j];
    }
    contraction+=p.domega[i]*source;
  }
  // Nraw-Nraw_ref = deltaG-dwn*(2wnref+dwn). Exact zero on the reference.
  const double deltaNull=deltaG-dwn*(2*wnref+dwn);
  const double deltaReg=hessian4-p.omega*p.w_omega-contraction;
  const double deltaPole=bdomega*f0pole-par.sigma*deltaNull;
  for(int i=0;i<3;++i){
    out.regular.beta[i]-=sourceweight*alpha*alpha*p.domega[i]*deltaReg/norm;
    out.pole.beta[i]-=sourceweight*alpha*alpha*p.domega[i]*deltaPole/norm;
  }
  return out;
}
inline hyp::GaugeRHS<double> Assemble(const hyp::GaugeRHSParts<double>& parts,double O) {
  hyp::GaugeRHS<double> rhs{};
  if(!hyp::AssembleGaugeInterior(parts,O,rhs))throw std::runtime_error("invalid scratch assembly");
  for(int i=0;i<3;++i)rhs.beta[i]+=parts.pole.beta[i]/O;
  return rhs;
}
}
#endif
