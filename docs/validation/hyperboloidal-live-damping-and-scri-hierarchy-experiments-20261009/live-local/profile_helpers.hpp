// Fresh actual full20 audit; generic dual gauge copied from frozen full20.
#include "../discrete-bianchi/immutable-discrete-bulk-20261009/dual_helpers.hpp"
#include "z4c/hyperboloidal/layer_gauge.hpp"
#include "live_damping_profile.hpp"
hyp::LayerPoint<D> Reference(const hyp::LayerPoint<double>&p) {
  hyp::LayerPoint<D> q{};q.state=Lift(p.state);q.alpha=p.alpha;q.radius=p.radius;
  q.omega=p.omega;q.k_bar=p.k_bar;q.k_physical=p.k_physical;q.w_omega=p.w_omega;
  for(int i=0;i<3;++i){q.beta[i]=p.beta[i];q.domega[i]=p.domega[i];q.dalpha[i]=p.dalpha[i];
    for(int j=0;j<3;++j)q.omega_hessian[i][j]=p.omega_hessian[i][j];}
  return q;
}
hyp::GaugeRHSParts<D> GaugeParts(const hyp::LayerPoint<D>&p,const Jet&u,double a,bool norm) {
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
  return parts;
}
hyp::GaugeRHS<D> Gauge(const hyp::LayerPoint<D>&p,const Jet&u,double a,bool norm) {
  const auto parts=GaugeParts(p,u,a,norm);hyp::GaugeRHS<D> f;
  if(!hyp::AssembleGaugeInterior(parts,p.omega,f))throw std::runtime_error("gauge invalid");
  for(int i=0;i<3;++i)f.beta[i]+=parts.pole.beta[i]/p.omega;
  return f;
}
std::array<D,20> Evaluate(const hyp::LayerPoint<double>&pd,const Jet&u,double a,
                        double kappa,int form,bool norm) {
  const auto o=Omega(u,pd);hyp::Z4cRHS<D> f;
  if(!hyp::AssembleInterior(hyp::ConformalRHS(u,o,D(kappa)/u.alpha.value,form?hyp::ResearchLiveKappa2Profile(u,o,D(pd.radius),D(kappa)):D(0)),o.omega,f))throw std::runtime_error("geometry invalid");
  const auto g=Gauge(Reference(pd),u,a,norm);
  std::array<D,20> out{};out[0]=g.alpha;out[1]=f.chi;out[2]=f.trace;out[3]=f.theta;
  const int ti[5]={0,0,0,1,1},tj[5]={0,1,2,1,2};
  for(int i=0;i<3;++i){out[4+i]=g.beta[i];out[17+i]=f.lambda[i];}
  for(int i=0;i<5;++i){out[7+i]=f.metric[ti[i]][tj[i]];out[12+i]=f.a[ti[i]][tj[i]];}
  return out;
}
Jet Background(const hyp::LayerPoint<double>&p,double perturb) {
  Jet u=Lift(p.state);
  for(int col=0;col<20;++col){J seed(D(perturb*std::sin(col+1.)));
    for(int i=0;i<3;++i){seed.d[i]=perturb*std::cos((col+1.)*(i+1.))/3;
      for(int j=0;j<3;++j)seed.dd[i][j]=perturb*std::sin((col+1.)*(i+j+2.))/5;}
    Seed(u,col,seed);}
  Consistent(u);return u;
}
M Matrix(const hyp::LayerPoint<double>&p,double a,double kappa,int form,bool norm,
         double perturb,double k,const std::array<double,3>&n) {
  const auto base=Background(p,perturb);M m{};
  for(int col=0;col<20;++col)for(int phase=0;phase<2;++phase) {
    auto u=base;J seed(D(0,phase?0:1));
    for(int i=0;i<3;++i){seed.d[i]=D(0,phase?k*n[i]:0);for(int j=0;j<3;++j)seed.dd[i][j]=D(0,phase?0:-k*k*n[i]*n[j]);}
    Seed(u,col,seed);Consistent(u);const auto f=Evaluate(p,u,a,kappa,form,norm);
    for(int row=0;row<20;++row)m[row][col]+=(phase?I:C(1))*f[row].d;
  }
  return m;
}
void Print(const M&m) {std::cout<<'[';for(int i=0;i<20;++i){if(i)std::cout<<',';std::cout<<'[';
  for(int j=0;j<20;++j){if(j)std::cout<<',';std::cout<<'['<<m[i][j].real()<<','<<m[i][j].imag()<<']';}std::cout<<']';}std::cout<<']';}
std::array<D,20> Fields(const hyp::Z4cRHS<D>&f,const hyp::GaugeRHS<D>&g){
 std::array<D,20> out{};out[0]=g.alpha;out[1]=f.chi;out[2]=f.trace;out[3]=f.theta;
 const int ti[5]={0,0,0,1,1},tj[5]={0,1,2,1,2};
 for(int i=0;i<3;++i){out[4+i]=g.beta[i];out[17+i]=f.lambda[i];}
 for(int i=0;i<5;++i){out[7+i]=f.metric[ti[i]][tj[i]];out[12+i]=f.a[ti[i]][tj[i]];}return out;
}
