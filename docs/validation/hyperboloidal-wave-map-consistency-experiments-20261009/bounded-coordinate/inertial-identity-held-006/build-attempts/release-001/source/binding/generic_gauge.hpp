// Exact frozen baseline-dual norm=true formula, templated for spatial jets.
template<class T> hyp::GaugeRHS<T> GenericGauge(const hyp::LayerPoint<T>&p,const hyp::Z4cJet<T>&u,double a,bool norm) {
  hyp::LayerGaugeParameters g;g.physical_trace_lapse=true;g.preferred_source=false;
  g.scri_lapse_damping=norm?1/a:1.5;
  auto parts=hyp::InteriorLayerGauge(p,u,g);
  if(norm) {
    const T W=hyp::LayerCoefficients(p.radius,u.alpha.value,g).weight;
    if(W>0) {
      const auto live=hyp::Geometry(u.metric),ref=hyp::Geometry(p.state.metric);
      T q{},ghat{},delta{};
      for(int i=0;i<3;++i){q+=p.domega[i]*p.domega[i];for(int j=0;j<3;++j){
        const T weight=p.domega[i]*p.domega[j];
        ghat+=p.state.chi.value*ref.inverse[i][j]*weight;
        delta+=((u.chi.value-p.state.chi.value)*ref.inverse[i][j]
                 +u.chi.value*(live.inverse[i][j]-ref.inverse[i][j]))*weight;
      }}
      const double eta=1.5/(a*a),C=(1-1/1.5)/a;
      for(int i=0;i<3;++i)parts.pole.beta[i]-=eta*W*((u.beta.value[i]-p.beta[i])-C*p.domega[i]*delta/(Kokkos::sqrt(q)*ghat));
    }
  }
  hyp::GaugeRHS<T> f;if(!hyp::AssembleGaugeInterior(parts,p.omega,f))throw std::runtime_error("gauge invalid");
  for(int i=0;i<3;++i)f.beta[i]+=parts.pole.beta[i]/p.omega;
  return f;
}
