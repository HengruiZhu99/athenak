// Exact Reference/Gauge extraction; only norm=true C0 is used.
#include "/Users/hz0693/research/hyperboloidal/build-layer-research/continuum/discrete-bianchi/immutable-discrete-bulk-20261009/dual_helpers.hpp"
#include "z4c/hyperboloidal/layer_gauge.hpp"
hyp::LayerPoint<D> Reference(const hyp::LayerPoint<double>&p) {
  hyp::LayerPoint<D> q{};q.state=Lift(p.state);q.alpha=p.alpha;q.radius=p.radius;
  q.omega=p.omega;q.k_bar=p.k_bar;q.k_physical=p.k_physical;q.w_omega=p.w_omega;
  for(int i=0;i<3;++i){q.beta[i]=p.beta[i];q.domega[i]=p.domega[i];q.dalpha[i]=p.dalpha[i];
    for(int j=0;j<3;++j)q.omega_hessian[i][j]=p.omega_hessian[i][j];}
  return q;
}
hyp::GaugeRHS<D> Gauge(const hyp::LayerPoint<D>&p,const Jet&u,double a,bool norm) {
  hyp::LayerGaugeParameters g;g.physical_trace_lapse=true;g.preferred_source=false;
  g.scri_lapse_damping=norm?1/a:1.5;
  auto parts=hyp::InteriorLayerGauge(p,u,g);
  if(norm) {
    const D W=hyp::LayerCoefficients(p.radius,u.alpha.value,g).weight;
    if(W>0) {
      const auto live=hyp::Geometry(u.metric),ref=hyp::Geometry(p.state.metric);
      D q{},ghat{},delta{};
      for(int i=0;i<3;++i){q+=p.domega[i]*p.domega[i];for(int j=0;j<3;++j){
        const D weight=p.domega[i]*p.domega[j];
        ghat+=p.state.chi.value*ref.inverse[i][j]*weight;
        delta+=((u.chi.value-p.state.chi.value)*ref.inverse[i][j]
                 +u.chi.value*(live.inverse[i][j]-ref.inverse[i][j]))*weight;
      }}
      const double eta=1.5/(a*a),C=(1-1/1.5)/a;
      for(int i=0;i<3;++i)parts.pole.beta[i]-=eta*W*((u.beta.value[i]-p.beta[i])-C*p.domega[i]*delta/(Kokkos::sqrt(q)*ghat));
    }
  }
  hyp::GaugeRHS<D> f;if(!hyp::AssembleGaugeInterior(parts,p.omega,f))throw std::runtime_error("gauge invalid");
  for(int i=0;i<3;++i)f.beta[i]+=parts.pole.beta[i]/p.omega;
  return f;
}
