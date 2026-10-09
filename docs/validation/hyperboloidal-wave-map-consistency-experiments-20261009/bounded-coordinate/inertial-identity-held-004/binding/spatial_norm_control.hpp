#ifndef SCRATCH_SPATIAL_NORM_SHIFT_CONTROL_HPP_
#define SCRATCH_SPATIAL_NORM_SHIFT_CONTROL_HPP_
#include <cmath>
#include <stdexcept>
#include "z4c/hyperboloidal/layer_gauge.hpp"
namespace spatial_norm {
namespace hyp=z4c::hyperboloidal;
struct Parameters {double xi=2,eta=6,C=2./3.;};
inline double NormDeviation(const hyp::LayerPoint<double>&p,const hyp::Z4cJet<double>&u) {
  const auto geo=hyp::Geometry(u.metric),ref=hyp::Geometry(p.state.metric);
  if(!geo.valid||!ref.valid)throw std::runtime_error("invalid norm-feedback metric");
  double Ghat=0,deltaG=0;
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){
    const double gradient=p.domega[i]*p.domega[j];
    Ghat+=p.state.chi.value*ref.inverse[i][j]*gradient;
    // Exact subtraction on the analytic reference, including nonflat layers.
    deltaG+=((u.chi.value-p.state.chi.value)*ref.inverse[i][j]
       +u.chi.value*(geo.inverse[i][j]-ref.inverse[i][j]))*gradient;
  }
  if(!(Ghat>0))throw std::runtime_error("norm feedback requires positive spatial reference norm");
  return deltaG/Ghat;
}
inline hyp::GaugeRHSParts<double> Gauge(const hyp::LayerPoint<double>&p,
    const hyp::Z4cJet<double>&u,const hyp::LayerGaugeParameters&g,const Parameters&par) {
  if(!g.physical_trace_lapse||g.preferred_source||par.xi<0||!(par.eta>0))
    throw std::runtime_error("invalid exploratory spatial-norm control");
  auto changed=g;changed.scri_lapse_damping=par.xi;
  auto out=hyp::InteriorLayerGauge(p,u,changed);
  const double W=hyp::LayerCoefficients(p.radius,u.alpha.value,g).weight;
  if(W==0)return out;
  double q=0;for(double d:p.domega)q+=d*d;
  if(!(q>0))throw std::runtime_error("norm feedback requires outer radial gradient");
  const double relative_norm=NormDeviation(p,u);
  for(int i=0;i<3;++i){const double n=-p.domega[i]/std::sqrt(q);
    out.pole.beta[i]-=par.eta*W*((u.beta.value[i]-p.beta[i])+par.C*n*relative_norm);
  }
  return out;
}
inline hyp::GaugeRHS<double> Assemble(const hyp::GaugeRHSParts<double>&p,double omega) {
  hyp::GaugeRHS<double> r{};
  if(!hyp::AssembleGaugeInterior(p,omega,r))throw std::runtime_error("invalid spatial-norm gauge assembly");
  for(int i=0;i<3;++i)r.beta[i]+=p.pole.beta[i]/omega;return r;
}
}
#endif
