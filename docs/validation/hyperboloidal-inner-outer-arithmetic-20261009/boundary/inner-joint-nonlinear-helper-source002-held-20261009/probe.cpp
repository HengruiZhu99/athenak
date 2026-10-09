// SOURCE ONLY. Fixed local gauge/kernel queries; no grid/evolution/operator.
#include "dual_helpers.hpp"
#include "nonlinear_values.hpp"
#include "test_support.hpp"
#include "inner_gauge.hpp"
#include <cstring>
#include <string>
#include <vector>
namespace inner { template<> struct Number<D>{
  static double Value(D x){return x.v;}static double Derivative(D x){return x.d;}
  static D Make(double x,double d){return D(x,d);}
}; }
namespace {
const double dirs[3][3]={{1,0,0},{.36,-.48,.8},{-.48,.64,.6}};
const double radii[14]={0,.025,.05,.1,.3,.45,.5,.65,.84,.85,.9,.95,.98,.995};
const char*families[14]={"reference","mild-lapse","mild-chi","shift","physical-P","Lambda",
 "chi-gradient","SPD-tensor","off-constraint-Theta","mixed","collapsed",
 "small-alpha-large-chi","large-alpha-small-chi","chi-gradient-contrast"};
const double e[3]={1,-.5,.25};
template<class T>void N(T x){double v=inner::Number<T>::Value(x);
 if(std::isfinite(v))std::cout<<v;else std::cout<<(std::isnan(v)?"\"NaN\"":v>0?"\"+Infinity\"":"\"-Infinity\"");}
template<class T>void S(T x){std::cout<<'[';N(x);std::cout<<',';N(inner::Number<T>::Derivative(x));std::cout<<']';}
template<class T>void V(const T *x,int n){std::cout<<'[';for(int i=0;i<n;++i){if(i)std::cout<<',';S(x[i]);}std::cout<<']';}
template<class T>void Pack(const hyp::Z4cJet<T>&u){
 std::cout<<"{\"alpha\":";S(u.alpha.value);std::cout<<",\"alpha_d\":";V(u.alpha.d,3);
 std::cout<<",\"chi\":";S(u.chi.value);std::cout<<",\"chi_d\":";V(u.chi.d,3);
 std::cout<<",\"P\":";S(u.trace.value);std::cout<<",\"Theta\":";S(u.theta.value);
 std::cout<<",\"beta\":";V(u.beta.value,3);std::cout<<",\"Lambda\":";V(u.lambda.value,3);
 std::cout<<",\"beta_d\":[";for(int j=0;j<3;++j){if(j)std::cout<<',';V(u.beta.d[j],3);}std::cout<<']';
 std::cout<<",\"g\":[";for(int i=0;i<3;++i){if(i)std::cout<<',';V(u.metric.g[i],3);}std::cout<<']';
 std::cout<<",\"g_d\":[";for(int d=0;d<3;++d){if(d)std::cout<<',';std::cout<<'[';for(int i=0;i<3;++i){if(i)std::cout<<',';V(u.metric.dg[d][i],3);}std::cout<<']';}std::cout<<']';
 std::cout<<",\"A\":[";for(int i=0;i<3;++i){if(i)std::cout<<',';V(u.a.k[i],3);}std::cout<<"]}";
}
template<class T>std::array<T,8> Parts(const hyp::GaugeRHSParts<T>&q){return {
 q.regular.alpha,q.regular.beta[0],q.regular.beta[1],q.regular.beta[2],
 q.pole.alpha,q.pole.beta[0],q.pole.beta[1],q.pole.beta[2]};}
template<class T>std::array<T,4> Fields(const hyp::GaugeRHS<T>&q){return {q.alpha,q.beta[0],q.beta[1],q.beta[2]};}
void Audit(const inner::Audit&a){std::cout<<"{\"products\":"<<a.products<<",\"coefficient_calls\":"<<a.coefficient_calls
 <<",\"coefficient_scaled_away\":"<<a.coefficient_scaled_away<<",\"max_product_exponent\":"<<a.maximum_product_exponent<<'}';}
template<class T>void Context(const hyp::LayerPoint<T>&p,const T*x,double curvature,double G0,const std::string&kind,int direction){
 std::cout<<"{\"kind\":\""<<kind<<"\",\"curvature\":"<<curvature<<",\"G0\":"<<G0<<",\"direction\":"<<direction<<",\"xyz\":";V(x,3);
 std::cout<<",\"radius\":";S(p.radius);std::cout<<",\"Omega\":";S(p.omega);std::cout<<",\"Omega_d\":";V(p.domega,3);
 std::cout<<",\"Phat\":";S(p.k_physical);std::cout<<",\"W\":";N(hyp::SmoothCutoff(p.radius,T(.45),T(.85)).value);
 std::cout<<",\"reference\":";Pack(p.state);
 const auto c=rwm::ReferenceConnection(p,x);std::cout<<",\"connection\":[";
 for(int a=0;a<4;++a){if(a)std::cout<<',';std::cout<<'[';for(int i=0;i<3;++i){if(i)std::cout<<',';V(c.scaled[a][i],3);}std::cout<<']';}std::cout<<']';
}
hyp::Z4cJet<double> State(const hyp::LayerPoint<double>&p,int family){
 auto u=p.state;auto apply=[&](int f){
 if(f==1){u.alpha.value=1.1*p.alpha;for(int j=0;j<3;++j)u.alpha.d[j]=1.1*p.dalpha[j]+.002*e[j];}
 if(f==2){u.chi.value=.8*p.state.chi.value;for(int j=0;j<3;++j)u.chi.d[j]=.8*p.state.chi.d[j]+.003*e[j];}
 if(f==3)for(int i=0;i<3;++i){u.beta.value[i]=p.beta[i]+.01*e[i];for(int j=0;j<3;++j)u.beta.d[j][i]=p.state.beta.d[j][i]+.002*e[i]*e[j];}
 if(f==4)u.trace.value=p.k_physical+.004;
 if(f==5)for(int i=0;i<3;++i)u.lambda.value[i]=p.state.lambda.value[i]+.004*e[i];
 if(f==6)for(int j=0;j<3;++j)u.chi.d[j]=p.state.chi.d[j]+.02*e[j];
 if(f==7){const double d[3]={1.25,.8,1};for(int i=0;i<3;++i)for(int j=0;j<3;++j){u.metric.g[i][j]=d[i]*d[j]*p.state.metric.g[i][j];u.a.k[i][j]=d[i]*d[j]*p.state.a.k[i][j];for(int k=0;k<3;++k){u.metric.dg[k][i][j]=d[i]*d[j]*p.state.metric.dg[k][i][j];u.a.dk[k][i][j]=d[i]*d[j]*p.state.a.dk[k][i][j];for(int l=0;l<3;++l)u.metric.ddg[k][l][i][j]=d[i]*d[j]*p.state.metric.ddg[k][l][i][j];}}}
 if(f==8){u.theta.value=.07;u.trace.value=p.k_physical+.004;}
 };
 if(family==9){for(int f=1;f<=8;++f)apply(f);}else apply(family);
 if(family>=10){const double av[4]={1e-150,1e-150,1e60,1e100},cv[4]={1e-150,1e60,1e-150,1e-150};u.alpha.value=av[family-10];u.chi.value=cv[family-10];
  const double ad[3]={.001,-.0015,.0005},cd[3]={.002,-.001,.0005};
  for(int j=0;j<3;++j){u.alpha.d[j]=u.alpha.value*ad[j];u.chi.d[j]=u.chi.value*cd[j];u.beta.value[j]=p.beta[j]+.003*e[j];u.lambda.value[j]=p.state.lambda.value[j]+.004*e[j];}u.trace.value=p.k_physical+.004;u.theta.value=.07;}
 return u;
}
template<class T>void EmitGauge(const hyp::LayerPoint<T>&p,const hyp::Z4cJet<T>&u,const T*x,double G0,bool baseline=true){
 inner::Audit a{};const auto q=inner::Gauge(p,u,x,{G0},&a);hyp::GaugeRHS<T>f{};
 const bool assembled=rwm::Assemble(q,p.omega,f);const auto qp=Parts(q);const auto fp=Fields(f);
 std::cout<<",\"input\":";Pack(u);std::cout<<",\"valid\":"<<(q.valid?"true":"false")<<",\"assembled\":"<<(assembled?"true":"false")<<",\"parts\":";V(qp.data(),8);std::cout<<",\"rhs\":";V(fp.data(),4);std::cout<<",\"arithmetic\":";Audit(a);
 if(baseline){const auto qb=rwm::Gauge(p,u,rwm::ReferenceConnection(p,x));const auto bp=Parts(qb);std::cout<<",\"baseline_valid\":"<<(qb.valid?"true":"false")<<",\"baseline_parts\":";V(bp.data(),8);bool equal=q.valid==qb.valid;for(int i=0;i<8;++i){const double v=inner::Number<T>::Value(qp[i]),b=inner::Number<T>::Value(bp[i]);equal=equal&&std::memcmp(&v,&b,sizeof(double))==0;}std::cout<<",\"outer_value_bitwise\":"<<(equal?"true":"false");}
}
hyp::LayerPoint<D> CastPoint(const hyp::LayerPoint<double>&p){hyp::LayerPoint<D>q{};
 q.state=Lift(p.state);q.alpha=p.alpha;q.omega=p.omega;q.radius=p.radius;q.L=p.L;q.b=p.b;q.k_bar=p.k_bar;q.k_physical=p.k_physical;
 for(int i=0;i<3;++i){q.beta[i]=p.beta[i];q.domega[i]=p.domega[i];q.dalpha[i]=p.dalpha[i];for(int j=0;j<3;++j)q.omega_hessian[i][j]=p.omega_hessian[i][j];}return q;}
Jet Direction(const hyp::Z4cJet<double>&s,int col,double eps){auto u=Lift(s);J d(D(eps,1));for(int i=0;i<3;++i){d.d[i]=D(eps*.17*(i+1),.17*(i+1));for(int j=0;j<3;++j)d.dd[i][j]=D(eps*.07*(i+j+1),.07*(i+j+1));}Seed(u,col,d);Consistent(u);return u;}
template<class T>std::array<T,22> Actual22(const hyp::LayerPoint<T>&p,const hyp::Z4cJet<T>&u,const T*x,double G0,bool candidate){
 hyp::OmegaJet<T>o{},o0{};o.omega=o0.omega=p.omega;
 for(int i=0;i<3;++i){o.gradient[i]=o0.gradient[i]=p.domega[i];for(int j=0;j<3;++j)o.hessian[i][j]=o0.hessian[i][j]=p.omega_hessian[i][j];}
 hyp::SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,o);
 hyp::SetStationaryOmegaNormal(p.state.alpha.value,p.state.beta.value,p.state.alpha.d,p.state.beta.d,o0);
 hyp::Z4cRHS<T>f{},f0{};hyp::GaugeRHS<T>g{};
 if(!hyp::AssembleInterior(hyp::ConformalRHS(u,o,T(10)/u.alpha.value,T(0)),p.omega,f)||
    !hyp::AssembleInterior(hyp::ConformalRHS(p.state,o0,T(10),T(0)),p.omega,f0))throw std::runtime_error("actual local geometry invalid");
 const auto q=candidate?inner::Gauge(p,u,x,{G0}):rwm::Gauge(p,u,rwm::ReferenceConnection(p,x));
 if(!rwm::Assemble(q,p.omega,g))throw std::runtime_error("actual local gauge invalid");
 return {g.alpha,f.chi-f0.chi,f.trace-f0.trace,f.theta-f0.theta,g.beta[0],g.beta[1],g.beta[2],
 f.metric[0][0]-f0.metric[0][0],f.metric[0][1]-f0.metric[0][1],f.metric[0][2]-f0.metric[0][2],f.metric[1][1]-f0.metric[1][1],f.metric[1][2]-f0.metric[1][2],f.metric[2][2]-f0.metric[2][2],
 f.a[0][0]-f0.a[0][0],f.a[0][1]-f0.a[0][1],f.a[0][2]-f0.a[0][2],f.a[1][1]-f0.a[1][1],f.a[1][2]-f0.a[1][2],f.a[2][2]-f0.a[2][2],f.lambda[0]-f0.lambda[0],f.lambda[1]-f0.lambda[1],f.lambda[2]-f0.lambda[2]};
}
void MainGrid(const std::string&mode){for(double a:{.5,1.,2.,4.}){
 hyp::LayerReference<double>ref(1,a,{true,.05,.95});ref.Validate();
 for(double r:radii)for(int d=0;d<3;++d)for(double G0:{.375,.75}){const double x[3]={r*dirs[d][0],r*dirs[d][1],r*dirs[d][2]};const auto p=ref.At(x[0],x[1],x[2]);
  if(mode=="sources"){for(int family=0;family<14;++family){Context(p,x,a,G0,"source",d);std::cout<<",\"family\":\""<<families[family]<<'"';EmitGauge(p,State(p,family),x,G0);std::cout<<"}\n";}}
  if(mode=="reference"){Context(p,x,a,G0,"reference",d);EmitGauge(p,p.state,x,G0);const auto adm=audit::Metric4(p.state,p,nullptr,nullptr,true),emb=audit::Embedding(p,x,1,a,{true,.05,.95});
   for(const auto&entry:{std::make_pair("ADM",adm),std::make_pair("embedding",emb)}){std::cout<<",\""<<entry.first<<"\":[";for(int aa=0;aa<4;++aa){if(aa)std::cout<<',';std::cout<<'[';for(int i=0;i<4;++i){if(i)std::cout<<',';std::cout<<'[';for(int j=0;j<4;++j){if(j)std::cout<<',';S(p.omega*entry.second.Gamma[aa][i][j]);}std::cout<<']';}std::cout<<']';}std::cout<<']';}std::cout<<"}\n";}
  if(mode=="principal"){auto pd=CastPoint(p);D xd[3]={x[0],x[1],x[2]};for(int col=0;col<19;++col){auto u=Lift(p.state);
    if(col==0)u.trace.value.d=1;if(col>=1&&col<=3)u.lambda.value[col-1].d=1;
    if(col>=4&&col<=6)u.alpha.d[col-4].d=1;if(col>=7&&col<=9)u.chi.d[col-7].d=1;
    if(col>=10)u.beta.d[(col-10)/3][(col-10)%3].d=1;
    Context(pd,xd,a,G0,"principal",d);std::cout<<",\"column\":"<<col;EmitGauge(pd,u,xd,G0);std::cout<<"}\n";}}
 }} }
void Duals(){for(double a:{.5,1.,2.})for(double r:{.025,.3,.5,.7,.84,.9,.98})for(int d=0;d<3;++d)for(double G0:{.375,.75}){
 hyp::LayerReference<double>ref(1,a,{true,.05,.95});const double x[3]={r*dirs[d][0],r*dirs[d][1],r*dirs[d][2]};const auto p=ref.At(x[0],x[1],x[2]);const auto pd=CastPoint(p);const D xd[3]={x[0],x[1],x[2]};const auto base=State(p,9);
 for(int col=0;col<20;++col){const auto u=Direction(base,col,0);Context(pd,xd,a,G0,"dual",d);std::cout<<",\"column\":"<<col;EmitGauge(pd,u,xd,G0);const auto full=Actual22(pd,u,xd,G0,true),baseline=Actual22(pd,u,xd,G0,false);
  std::cout<<",\"actual22\":";V(full.data(),22);std::cout<<",\"baseline22\":";V(baseline.data(),22);std::cout<<",\"FD\":[";int level=0;
  for(double eps:{1e-4,5e-5,2.5e-5}){if(level++)std::cout<<',';const auto up=Values(Direction(base,col,eps)),um=Values(Direction(base,col,-eps));const auto fp=Actual22(p,up,x,G0,true),fm=Actual22(p,um,x,G0,true);std::cout<<"{\"epsilon\":"<<eps<<",\"plus22\":";V(fp.data(),22);std::cout<<",\"minus22\":";V(fm.data(),22);std::cout<<'}';}std::cout<<"]}\n";
 }} }
void Coefficients(){const std::vector<double>v={std::numeric_limits<double>::denorm_min(),std::numeric_limits<double>::min(),1e-300,1e-150,1e-18,.05,1,3,1e150,1e300,std::numeric_limits<double>::max()};
 for(double a:v)for(double ch:v)for(double W:{0.,std::ldexp(1.,-52),.125,.5,1-std::ldexp(1.,-52),1.})for(double G0:{.375,.75}){bool ok=false;inner::Audit audit{};const auto k=inner::Coefficient(a,ch,W,G0,ok,&audit);std::cout<<"{\"kind\":\"coefficient\",\"alpha\":";S(a);std::cout<<",\"chi\":";S(ch);std::cout<<",\"W\":"<<W<<",\"G0\":"<<G0<<",\"valid\":"<<(ok?"true":"false")<<",\"k\":";S(k);std::cout<<",\"arithmetic\":";Audit(audit);std::cout<<"}\n";}
 for(double a:{1e-150,1e-18,1.,1e60})for(double ch:{1e-150,1e-18,1.,1e60})for(double W:{0.,std::ldexp(1.,-52),.125,.5,1-std::ldexp(1.,-52)})for(double G0:{.375,.75}){bool ok=false,badok=false;inner::Audit audit{};const D al(a,a),chi(ch,-.3*ch);const auto k=inner::Coefficient(al,chi,W,G0,ok,&audit),bad=inner::Coefficient(al,chi,W,G0,badok,nullptr,true);std::cout<<"{\"kind\":\"coefficient-dual\",\"alpha\":";S(al);std::cout<<",\"chi\":";S(chi);std::cout<<",\"W\":"<<W<<",\"G0\":"<<G0<<",\"valid\":"<<(ok?"true":"false")<<",\"k\":";S(k);std::cout<<",\"value_only_control\":";S(bad);std::cout<<",\"arithmetic\":";Audit(audit);std::cout<<"}\n";}
}
void CoreWitness(){for(double a:{.5,1.,2.,4.})for(double r:{0.,.025,.05})for(int d=0;d<3;++d)for(double G0:{.375,.75})for(int type=0;type<2;++type){hyp::LayerReference<double>ref(1,a,{true,.05,.95});const double x[3]={r*dirs[d][0],r*dirs[d][1],r*dirs[d][2]};const auto p=ref.At(x[0],x[1],x[2]);auto u=p.state;u.alpha.value=1e100;u.chi.value=type?1e-150:1e100;u.trace.value=u.theta.value=0;for(int i=0;i<3;++i){u.alpha.d[i]=0;u.chi.d[i]=type?u.chi.value*.002*e[i]:0;u.beta.value[i]=0;u.lambda.value[i]=type?0:.004*e[i];for(int j=0;j<3;++j)u.beta.d[j][i]=0;}
 Context(p,x,a,G0,"core-witness",d);std::cout<<",\"witness\":"<<type;EmitGauge(p,u,x,G0,false);std::cout<<"}\n";}}
void Invalid(){const double bad[4]={0,-1,std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity()};
 for(int field=0;field<4;++field)for(int j=0;j<4;++j){double a=1,ch=1,G=.375,W=.5;if(field==0)a=bad[j];if(field==1)ch=bad[j];if(field==2)G=bad[j];if(field==3)W=j==0?-.01:j==1?1.01:bad[j];bool valid=false;inner::Coefficient(a,ch,W,G,valid);std::cout<<"{\"kind\":\"invalid-coefficient\",\"field\":"<<field<<",\"case\":"<<j<<",\"rejected\":"<<(!valid?"true":"false")<<"}\n";}
 hyp::LayerReference<double>ref(1,.5,{true,.05,.95});const double x[3]={.5,0,0};const auto p=ref.At(.5,0,0);
 for(int j=0;j<6;++j){auto u=p.state;auto pp=p;if(j==0)u.alpha.value=0;if(j==1)u.chi.value=-1;if(j==2)u.alpha.value=std::numeric_limits<double>::infinity();if(j==3)u.chi.value=std::numeric_limits<double>::quiet_NaN();if(j==4)for(int i=0;i<3;++i)for(int k=0;k<3;++k)u.metric.g[i][k]=0;if(j==5)pp.omega=0;const auto q=inner::Gauge(pp,u,x);std::cout<<"{\"kind\":\"invalid-source\",\"case\":"<<j<<",\"rejected\":"<<(!q.valid?"true":"false")<<"}\n";}}
void Nonrepresentable(){hyp::LayerReference<double>ref(1,.5,{true,.05,.95});for(double r:{.3,.5,.84})for(int d=0;d<3;++d)for(double G0:{.375,.75}){const double x[3]={r*dirs[d][0],r*dirs[d][1],r*dirs[d][2]};const auto p=ref.At(x[0],x[1],x[2]);auto u=State(p,13);u.chi.value=1e100;const double cd[3]={.002,-.001,.0005};for(int j=0;j<3;++j)u.chi.d[j]=u.chi.value*cd[j];Context(p,x,.5,G0,"nonrepresentable",d);EmitGauge(p,u,x,G0,false);std::cout<<"}\n";}}
}
int main(int argc,char**argv){try{if(argc!=2)throw std::runtime_error("exactly one fixed mode required");std::cout<<std::setprecision(17);const std::string mode=argv[1];
 if(mode=="sources"||mode=="reference"||mode=="principal")MainGrid(mode);
 else if(mode=="duals")Duals();else if(mode=="coefficients")Coefficients();else if(mode=="core-witnesses")CoreWitness();else if(mode=="invalid")Invalid();else if(mode=="nonrepresentable")Nonrepresentable();else throw std::runtime_error("unknown fixed mode");
 return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
