// Exact frozen actual_bridge.cpp binding extraction: Values through OutputNormals.
Raw Values(const Jet&u,bool derivative){Raw a{};auto take=[&](D x){return derivative?x.d:x.v;};a[0]=take(u.chi.value);a[7]=take(u.trace.value);a[17]=take(u.theta.value);a[18]=take(u.alpha.value);for(int t=0;t<6;++t){a[1+t]=take(u.metric.g[ti[t]][tj[t]]);a[8+t]=take(u.a.k[ti[t]][tj[t]]);}for(int i=0;i<3;++i){a[14+i]=take(u.lambda.value[i]);a[19+i]=take(u.beta.value[i]);}return a;}
template<class T> std::array<T,22> Pack(const hyp::Z4cRHS<T>&f,const hyp::GaugeRHS<T>&g){std::array<T,22>a{};a[0]=f.chi;a[7]=f.trace;a[17]=f.theta;a[18]=g.alpha;for(int t=0;t<6;++t){a[1+t]=f.metric[ti[t]][tj[t]];a[8+t]=f.a[ti[t]][tj[t]];}for(int i=0;i<3;++i){a[14+i]=f.lambda[i];a[19+i]=g.beta[i];}return a;}
std::array<D,22> ActualDual(const hyp::LayerPoint<double>&p,const Jet&u){
  auto o=Omega(u,p);hyp::Z4cRHS<D>f{},f0{};const auto b=Lift(p.state);
  if(!hyp::AssembleInterior(hyp::ConformalRHS(u,o,D(10)/u.alpha.value,D(0)),o.omega,f)
      ||!hyp::AssembleInterior(hyp::ConformalRHS(b,Omega(b,p),D(10),D(0)),o.omega,f0))throw std::runtime_error("invalid C0");
  f.chi-=f0.chi;f.trace-=f0.trace;f.theta-=f0.theta;
  for(int i=0;i<3;++i){f.lambda[i]-=f0.lambda[i];for(int j=0;j<3;++j){f.metric[i][j]-=f0.metric[i][j];f.a[i][j]-=f0.a[i][j];}}
  return Pack(f,Gauge(Reference(p),u,.5,true));
}
hyp::Z4cJet<double> DoubleState(const Jet&u){hyp::Z4cJet<double>b{};auto scalar=[](auto&a,const auto&c){a.value=c.value.v;for(int i=0;i<3;++i){a.d[i]=c.d[i].v;for(int j=0;j<3;++j)a.dd[i][j]=c.dd[i][j].v;}};scalar(b.chi,u.chi);scalar(b.alpha,u.alpha);scalar(b.trace,u.trace);scalar(b.theta,u.theta);for(int i=0;i<3;++i){b.beta.value[i]=u.beta.value[i].v;b.lambda.value[i]=u.lambda.value[i].v;for(int d=0;d<3;++d){b.beta.d[d][i]=u.beta.d[d][i].v;b.lambda.d[d][i]=u.lambda.d[d][i].v;for(int e=0;e<3;++e){b.beta.dd[d][e][i]=u.beta.dd[d][e][i].v;b.lambda.dd[d][e][i]=u.lambda.dd[d][e][i].v;}}for(int j=0;j<3;++j){b.metric.g[i][j]=u.metric.g[i][j].v;b.a.k[i][j]=u.a.k[i][j].v;for(int d=0;d<3;++d){b.metric.dg[d][i][j]=u.metric.dg[d][i][j].v;b.a.dk[d][i][j]=u.a.dk[d][i][j].v;for(int e=0;e<3;++e)b.metric.ddg[d][e][i][j]=u.metric.ddg[d][e][i][j].v;}}}return b;}
Raw ActualDouble(const hyp::LayerPoint<double>&p,const Jet&d){
  const auto u=DoubleState(d),b=p.state;
  auto omega=[&](const auto&v){hyp::OmegaJet<double>o{};o.omega=p.omega;for(int i=0;i<3;++i){o.gradient[i]=p.domega[i];for(int j=0;j<3;++j)o.hessian[i][j]=p.omega_hessian[i][j];}hyp::SetStationaryOmegaNormal(v.alpha.value,v.beta.value,v.alpha.d,v.beta.d,o);return o;};
  const auto o=omega(u),o0=omega(b);hyp::Z4cRHS<double>f{},f0{};
  if(!hyp::AssembleInterior(hyp::ConformalRHS(u,o,10/u.alpha.value,0.),o.omega,f)||!hyp::AssembleInterior(hyp::ConformalRHS(b,o0,10.,0.),o.omega,f0))throw std::runtime_error("invalid double C0");
  f.chi-=f0.chi;f.trace-=f0.trace;f.theta-=f0.theta;for(int i=0;i<3;++i){f.lambda[i]-=f0.lambda[i];for(int j=0;j<3;++j){f.metric[i][j]-=f0.metric[i][j];f.a[i][j]-=f0.a[i][j];}}
  hyp::LayerGaugeParameters g;g.physical_trace_lapse=true;g.preferred_source=false;g.scri_lapse_damping=2;
  // Actual double private wrapper, not a newly substituted formula.
  const auto parts=hyp::ResearchNativeSpatialGauge(p,u,g);hyp::GaugeRHS<double>gr{};
  if(!hyp::ResearchNativeSpatialAssemble(parts,p.omega,gr))throw std::runtime_error("invalid double gauge");
  return Pack(f,gr);
}
Raw Derivative(const std::array<D,22>&a){Raw r{};for(int i=0;i<22;++i)r[i]=a[i].d;return r;}
Raw Plain(const std::array<D,22>&a){Raw r{};for(int i=0;i<22;++i)r[i]=a[i].v;return r;}
std::array<double,2> InputNormals(const hyp::LayerPoint<double>&p,const Jet&u){const auto g=hyp::Geometry(p.state.metric);double t=0,a=0;for(int i=0;i<3;++i)for(int j=0;j<3;++j){t+=g.inverse[i][j]*u.metric.g[i][j].d;a+=g.inverse[i][j]*u.a.k[i][j].d;for(int k=0;k<3;++k)for(int l=0;l<3;++l)a-=g.inverse[i][k]*g.inverse[j][l]*p.state.a.k[k][l]*u.metric.g[i][j].d;}return {t,a};}
std::array<double,2> OutputNormals(const hyp::LayerPoint<double>&p,const Raw&f){const auto g=hyp::Geometry(p.state.metric);double t=0,a=0;for(int q=0;q<6;++q){const int i=ti[q],j=tj[q];const double w=i==j?1:2;t+=w*g.inverse[i][j]*f[1+q];a+=w*g.inverse[i][j]*f[8+q];for(int k=0;k<3;++k)for(int l=0;l<3;++l)a-=w*g.inverse[i][k]*g.inverse[j][l]*p.state.a.k[k][l]*f[1+q];}return {t,a};}

