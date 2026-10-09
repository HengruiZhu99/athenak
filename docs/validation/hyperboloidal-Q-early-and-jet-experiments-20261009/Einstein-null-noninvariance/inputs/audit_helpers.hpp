#include "inputs/dual_helpers.hpp"
#include "q_null_feedback.hpp"
hyp::LayerPoint<D> Reference(const hyp::LayerPoint<double>&p){hyp::LayerPoint<D>q{};
 q.state=Lift(p.state);q.alpha=p.alpha;q.radius=p.radius;q.omega=p.omega;q.k_bar=p.k_bar;q.k_physical=p.k_physical;q.w_omega=p.w_omega;
 for(int i=0;i<3;++i){q.beta[i]=p.beta[i];q.domega[i]=p.domega[i];q.dalpha[i]=p.dalpha[i];for(int j=0;j<3;++j)q.omega_hessian[i][j]=p.omega_hessian[i][j];}return q;}
std::array<D,20> Fields(const hyp::Z4cRHS<D>&f,const hyp::GaugeRHS<D>&g){std::array<D,20>o{};
 o[0]=g.alpha;o[1]=f.chi;o[2]=f.trace;o[3]=f.theta;const int ti[5]={0,0,0,1,1},tj[5]={0,1,2,1,2};
 for(int i=0;i<3;++i){o[4+i]=g.beta[i];o[17+i]=f.lambda[i];}for(int i=0;i<5;++i){o[7+i]=f.metric[ti[i]][tj[i]];o[12+i]=f.a[ti[i]][tj[i]];}return o;}
struct RS {std::array<D,20>r,s;};
RS Parts(const hyp::LayerPoint<double>&p,const Jet&u,double k,double sigma,bool inner=false){
 auto f=hyp::ConformalRHS(u,Omega(u,p),D(k)/u.alpha.value,D(0));
 hyp::LayerGaugeParameters g;g.preferred_source=true;g.physical_trace_lapse=false;
 auto q=qnf::Gauge(Reference(p),u,g,{.85,.95,sigma,inner});
 hyp::GaugeRHS<D>r{},s{};r.alpha=q.regular.alpha;s.alpha=q.pole.alpha;
 for(int i=0;i<3;++i){r.beta[i]=q.regular.beta[i];s.beta[i]=q.pole.beta[i];}
 return {Fields(f.regular,r),Fields(f.pole,s)};
}
J OmegaJ(const hyp::LayerPoint<double>&p){J o(D(p.omega));for(int i=0;i<3;++i){o.d[i]=p.domega[i];for(int j=0;j<3;++j)o.dd[i][j]=p.omega_hessian[i][j];}return o;}
J Unit(const hyp::LayerPoint<double>&p,int i){const double r=p.radius,n[3]={1.,0.,0.};J y{D(n[i])};
 for(int d=0;d<3;++d){y.d[d]=((d==i?1.:0.)-n[d]*n[i])/r;for(int e=0;e<3;++e)y.dd[d][e]=(-(d==i?1.:0.)*n[e]-(e==i?1.:0.)*n[d]-(d==e?1.:0.)*n[i]+3*n[i]*n[d]*n[e])/(r*r);}return y;}
J Linear(const J&x){J y(D(0,x.v.v));for(int i=0;i<3;++i){y.d[i]=D(0,x.d[i].v);for(int j=0;j<3;++j)y.dd[i][j]=D(0,x.dd[i][j].v);}return y;}
Jet Witness(const hyp::LayerPoint<double>&p){auto u=Lift(p.state);const auto o=OmegaJ(p);Seed(u,0,Linear(o));for(int i=0;i<3;++i)Seed(u,4+i,Linear(J(D(-1))*o*Unit(p,i)));Consistent(u);return u;}
void Print(const std::array<D,20>&v,bool d=true){std::cout<<'[';for(int i=0;i<20;++i)std::cout<<(i?",":"")<<(d?v[i].d:v[i].v);std::cout<<']';}
