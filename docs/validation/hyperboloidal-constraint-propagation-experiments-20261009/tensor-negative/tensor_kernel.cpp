// Exploratory transfer of the 1111.2177 tensor prescription to the actual gauge.
#define main immutable_discrete_main
#include "../discrete-bianchi/immutable-discrete-bulk-20261009/discrete_kernel.cpp"
#undef main
Kernel ExtractNovel(const Symbols&s,bool compatible,double alpha,double chi,
                       bool spd,double radius,const std::array<double,3>&velocity) {
  hyp::LayerPoint<double> pd{}; pd.omega=1;pd.alpha=alpha;pd.radius=radius;
  pd.state.alpha.value=alpha;pd.state.chi.value=chi;
  const double b[3][3]={{1.3,.2,-.1},{0,.8,.12},{0,0,1/1.04}};
  for(int i=0;i<3;++i) {
    pd.beta[i]=pd.state.beta.value[i]=velocity[i];
    for(int j=0;j<3;++j)
      for(int k=0;k<3;++k)
        pd.state.metric.g[i][j]+=(spd?b[k][i]*b[k][j]:double(i==k&&j==k));
  }
  hyp::LayerPoint<D> p{};p.omega=1;p.alpha=alpha;p.radius=radius;
  p.state=Lift(pd.state);for(int i=0;i<3;++i)p.beta[i]=velocity[i];
  hyp::LayerGaugeParameters gauge;gauge.preferred_source=false;
  gauge.physical_trace_lapse=true;gauge.lapse_outer=0;gauge.shift_outer=0;
  gauge.scri_lapse_damping=0;
  Kernel result;
  for(int col=0;col<20;++col)for(int phase=0;phase<2;++phase) {
    const auto part=[phase](C z){return phase?z.imag():z.real();};
    Jet u=Lift(pd.state);J seed(D(0,phase?0:1));
    for(int i=0;i<3;++i) {
      seed.d[i]=D(0,part(s.d[i]));
      for(int j=0;j<3;++j)
        seed.dd[i][j]=D(0,part(compatible?s.compatible[i][j]:s.native[i][j]));
    }
    Seed(u,col,seed);Consistent(u);
    Jet uc=Lift(pd.state);J seedc=seed;
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)
      seedc.dd[i][j]=D(0,part(s.compatible[i][j]));
    Seed(uc,col,seedc);Consistent(uc);
    auto o=Omega(u,pd);hyp::Z4cRHS<D> f;
    if(!hyp::AssembleInterior(hyp::ConformalRHS(u,o,D(0),D(0)),o.omega,f))
      throw std::runtime_error("invalid geometric extraction");
    // Linear constant-background realization of Eqs27--29 of 1111.2177.
    // Keep every Laplacian; correct only scalar TF Hessian and vector grad-div.
    const auto gt=hyp::Geometry(u.metric);
    D da[3][3]{},tr{};
    for(int i=0;i<3;++i)for(int j=0;j<3;++j) {
      da[i][j]=-u.chi.value*(uc.alpha.dd[i][j]-u.alpha.dd[i][j])
              +u.alpha.value*(uc.chi.dd[i][j]-u.chi.dd[i][j])/2;
      tr+=gt.inverse[i][j]*da[i][j];
    }
    for(int i=0;i<3;++i)for(int j=0;j<3;++j) {
      f.a[i][j]+=da[i][j]-u.metric.g[i][j]*tr/3;
      for(int k=0;k<3;++k)
        f.lambda[i]+=gt.inverse[i][j]*(uc.beta.dd[j][k][k]-u.beta.dd[j][k][k])/3;
    }
    hyp::GaugeRHS<D> g;
    if(!hyp::AssembleGaugeInterior(hyp::InteriorLayerGauge(p,u,gauge),D(1),g))
      throw std::runtime_error("invalid gauge extraction");
    double v[20]{};v[0]=g.alpha.d;v[1]=f.chi.d;v[2]=f.trace.d;v[3]=f.theta.d;
    const int ti[5]={0,0,0,1,1},tj[5]={0,1,2,1,2};
    for(int j=0;j<3;++j){v[4+j]=g.beta[j].d;v[17+j]=f.lambda[j].d;}
    for(int j=0;j<5;++j){v[7+j]=f.metric[ti[j]][tj[j]].d;v[12+j]=f.a[ti[j]][tj[j]].d;}
    const C factor=phase?I:C(1);
    for(int row=0;row<20;++row)result.l[row][col]+=factor*v[row];
    const auto q=Constraints(u,pd);
    for(int row=0;row<8;++row)result.q[row][col]+=factor*q[row];
  }
  return result;
}

int main(){
  const double desired=7./17;double lo=0,hi=std::acos(-1.);
  for(int it=0;it<80;++it){const double t=(lo+hi)/2,c=std::cos(t);
    const double ratio=std::pow(std::sin(t)*(4-c)/3,2)/((30-32*c+2*std::cos(2*t))/12);
    if(ratio>desired)lo=t;else hi=t;
  }
  const double theta=(lo+hi)/2;
  lo=.45;hi=.85;hyp::LayerGaugeParameters gauge;
  for(int it=0;it<80;++it){double r=(lo+hi)/2;
    if(hyp::LayerCoefficients(r,1.,gauge).weight<.9)lo=r;else hi=r;
  }
  const double r=(lo+hi)/2;const std::array<double,3> vel={.3,-.2,.1};
  const auto s=Stencils({theta,0,0},.125,vel);
  const auto k=ExtractNovel(s,false,1.,1.,false,r,vel);
  const auto c=hyp::LayerCoefficients(r,1.,gauge);
  std::cout<<std::setprecision(17)<<"{\"theta\":"<<theta<<",\"r\":"<<r
    <<",\"W\":"<<c.weight<<",\"f\":"<<c.f<<",\"mu\":"<<c.mu
    <<",\"D\":[";
  for(int i=0;i<3;++i){if(i)std::cout<<',';Complex(s.d[i]);}
  std::cout<<"],\"S\":[";
  for(int i=0;i<3;++i){if(i)std::cout<<',';std::cout<<'[';
    for(int j=0;j<3;++j){if(j)std::cout<<',';Complex(s.native[i][j]);}std::cout<<']';}
  std::cout<<"],\"L\":";Matrix(k.l);std::cout<<",\"Q\":";Matrix(k.q,8);
  std::cout<<"}\n";
}
