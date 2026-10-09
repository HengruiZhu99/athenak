// SOURCE ONLY. Fixed direct RWM field-dual arithmetic supplement; no PDE/kernel.
#include "dual_helpers.hpp"
#include "nonlinear_values.hpp"
#include "reference_wave_map.hpp"
#include <cstring>
#include <limits>
#include <string>
namespace inner { template<> struct Number<D>{
  static double Value(D x){return x.v;}static double Derivative(D x){return x.d;}
  static D Make(double x,double d){return D(x,d);}
}; }
namespace {
const double e[3]={1,-.5,.25};
const char*seed_names[17]={"zero","alpha-relative","chi-relative","joint-relative",
 "A0-balanced","alpha-chi-balanced","alpha-gradient-only","chi-gradient-only",
 "A0-balanced-plus-gradients","beta-value","beta-derivative","Lambda-value",
 "physical-P","metric-STF","Theta-only","all-used-mixed","unconsumed-jet-only"};
const char*family_names[4]={"collapsed","small-alpha-large-chi","large-alpha-small-chi","chi-gradient-contrast"};
const char*closed_names[3]={"tensor-pole","chi-flux","alpha-flux"};
const double directions[2][3]={{1,0,0},{.36,-.48,.8}};
const double radii[7]={.025,.1,.5,.65,.84,.95,.995};
unsigned long helper_calls=0;
#include "original_state.hpp"
#include "original_cast.hpp"

template<class T>void Scalar(T x){const double v=inner::Number<T>::Value(x);
 if(std::isfinite(v))std::cout<<v;else std::cout<<(std::isnan(v)?"\"NaN\"":v>0?"\"+Infinity\"":"\"-Infinity\"");}
template<class T>void Atom(T x){std::cout<<'[';Scalar(x);std::cout<<',';Scalar(inner::Number<T>::Derivative(x));std::cout<<']';}
template<class T>void Vector(const T*x,int n){std::cout<<'[';for(int i=0;i<n;++i){if(i)std::cout<<',';Atom(x[i]);}std::cout<<']';}
template<class T>void Matrix(const T x[3][3]){std::cout<<'[';for(int i=0;i<3;++i){if(i)std::cout<<',';Vector(x[i],3);}std::cout<<']';}
template<class T>void Pack(const hyp::Z4cJet<T>&u){
 std::cout<<"{\"alpha\":";Atom(u.alpha.value);std::cout<<",\"alpha_d\":";Vector(u.alpha.d,3);
 std::cout<<",\"chi\":";Atom(u.chi.value);std::cout<<",\"chi_d\":";Vector(u.chi.d,3);
 std::cout<<",\"P\":";Atom(u.trace.value);std::cout<<",\"Theta\":";Atom(u.theta.value);
 std::cout<<",\"beta\":";Vector(u.beta.value,3);std::cout<<",\"beta_d\":";Matrix(u.beta.d);
 std::cout<<",\"Lambda\":";Vector(u.lambda.value,3);std::cout<<",\"g\":";Matrix(u.metric.g);
 std::cout<<",\"g_d\":[";for(int k=0;k<3;++k){if(k)std::cout<<',';Matrix(u.metric.dg[k]);}std::cout<<']';
 std::cout<<",\"g_dd\":[";for(int k=0;k<3;++k){if(k)std::cout<<',';std::cout<<'[';for(int l=0;l<3;++l){if(l)std::cout<<',';Matrix(u.metric.ddg[k][l]);}std::cout<<']';}std::cout<<']';
 std::cout<<",\"A\":";Matrix(u.a.k);std::cout<<'}';
}
template<class T>std::array<T,8> Parts(const hyp::GaugeRHSParts<T>&q){return {q.regular.alpha,q.regular.beta[0],q.regular.beta[1],q.regular.beta[2],q.pole.alpha,q.pole.beta[0],q.pole.beta[1],q.pole.beta[2]};}
template<class T>std::array<T,4> Fields(const hyp::GaugeRHS<T>&f){return {f.alpha,f.beta[0],f.beta[1],f.beta[2]};}
template<class F>void Visit(Jet&u,F operation){
 for(auto*s:{&u.alpha,&u.chi,&u.trace,&u.theta}){operation(s->value);for(int i=0;i<3;++i){operation(s->d[i]);for(int j=0;j<3;++j)operation(s->dd[i][j]);}}
 for(auto*v:{&u.beta,&u.lambda})for(int i=0;i<3;++i){operation(v->value[i]);for(int j=0;j<3;++j){operation(v->d[j][i]);for(int k=0;k<3;++k)operation(v->dd[k][j][i]);}}
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){operation(u.metric.g[i][j]);operation(u.a.k[i][j]);for(int k=0;k<3;++k){operation(u.metric.dg[k][i][j]);operation(u.a.dk[k][i][j]);for(int l=0;l<3;++l)operation(u.metric.ddg[k][l][i][j]);}}
}
bool AllFinite(Jet u){bool good=true;Visit(u,[&](D&x){good=good&&std::isfinite(x.v)&&std::isfinite(x.d);});return good;}
bool AllZeroTangent(Jet u){bool good=true;Visit(u,[&](D&x){good=good&&x.d==0;});return good;}
bool SPD(const Jet&u){const auto&g=u.metric.g;const double d2=g[0][0].v*g[1][1].v-g[0][1].v*g[1][0].v;
 const double d3=g[0][0].v*(g[1][1].v*g[2][2].v-g[1][2].v*g[2][1].v)-g[0][1].v*(g[1][0].v*g[2][2].v-g[1][2].v*g[2][0].v)+g[0][2].v*(g[1][0].v*g[2][1].v-g[1][1].v*g[2][0].v);
 return std::isfinite(d2)&&std::isfinite(d3)&&g[0][0].v>0&&d2>0&&d3>0;}
Jet SeedField(hyp::Z4cJet<double>base,int seed,bool zero_gradients){
 if(zero_gradients)for(int j=0;j<3;++j)base.alpha.d[j]=base.chi.d[j]=0;
 Jet u=Lift(base);double xa=0,xc=0;double za[3]{},zc[3]{};
 if(seed==1||seed==3||seed==4||seed==5||seed==8||seed==15)xa=1;
 if(seed==2||seed==3)xc=1;if(seed==4||seed==8||seed==15)xc=-2;if(seed==5)xc=-1;
 for(int j=0;j<3;++j){if(seed==6||seed==8||seed==15)za[j]=e[j];if(seed==7||seed==15)zc[j]=e[j];if(seed==8)zc[j]=-2*e[j];}
 u.alpha.value.d=xa*base.alpha.value;u.chi.value.d=xc*base.chi.value;
 for(int j=0;j<3;++j){u.alpha.d[j].d=xa*base.alpha.d[j]+base.alpha.value*za[j];u.chi.d[j].d=xc*base.chi.d[j]+base.chi.value*zc[j];}
 if(seed==9||seed==15)for(int i=0;i<3;++i)u.beta.value[i].d=e[i]/16;
 if(seed==10||seed==15)for(int j=0;j<3;++j)for(int i=0;i<3;++i)u.beta.d[j][i].d=e[j]*e[i]/32;
 if(seed==11||seed==15)for(int i=0;i<3;++i)u.lambda.value[i].d=e[i]/64;
 if(seed==12||seed==15)u.trace.value.d=1./128;
 if(seed==13||seed==15){const double diagonal[3]={1./32,-1./32,0};for(int i=0;i<3;++i)for(int j=0;j<3;++j)u.metric.g[i][j].d=base.metric.g[i][j]*diagonal[j]+diagonal[i]*base.metric.g[i][j];}
 if(seed==14||seed==15)u.theta.value.d=1;
 if(seed==16)for(int i=0;i<3;++i)for(int j=0;j<3;++j){u.a.k[i][j].d=e[i]*e[j]/64;for(int k=0;k<3;++k){u.metric.dg[k][i][j].d=e[k]*e[i]*e[j]/128;for(int l=0;l<3;++l)u.metric.ddg[k][l][i][j].d=e[k]*e[l]*e[i]*e[j]/256;}}
 return u; // No Consistent() call: these are independently registered local field jets.
}
template<class T>struct Output{hyp::GaugeRHSParts<T>parts{};hyp::GaugeRHS<T>rhs{};bool assembled=false;};
template<class T>Output<T> Call(const hyp::LayerPoint<T>&p,const hyp::Z4cJet<T>&u,const rwm::Connection<T>&c,bool legacy){
 ++helper_calls;Output<T>out;out.parts=legacy?rwm::LegacyGauge(p,u,c):rwm::Gauge(p,u,c);out.assembled=rwm::Assemble(out.parts,p.omega,out.rhs);return out;
}
template<class T>void EmitOutput(const Output<T>&out){const auto ps=Parts(out.parts);const auto fs=Fields(out.rhs);
 std::cout<<"{\"valid\":"<<(out.parts.valid?"true":"false")<<",\"assembled\":"<<(out.assembled?"true":"false")<<",\"parts\":";Vector(ps.data(),8);std::cout<<",\"rhs\":";Vector(fs.data(),4);std::cout<<'}';}
void Context(const hyp::LayerPoint<D>&p,const D*x,const rwm::Connection<D>&c,const Jet&u){
 std::cout<<",\"xyz\":";Vector(x,3);std::cout<<",\"Omega\":";Atom(p.omega);std::cout<<",\"Omega_d\":";Vector(p.domega,3);
 std::cout<<",\"reference\":";Pack(p.state);std::cout<<",\"input\":";Pack(u);std::cout<<",\"connection\":[";
 bool zero=AllZeroTangent(p.state)&&p.alpha.d==0&&p.omega.d==0&&p.radius.d==0&&p.L.d==0&&p.b.d==0&&p.k_bar.d==0&&p.k_physical.d==0;
 bool coefficients_finite=true;
 for(int i=0;i<3;++i){zero=zero&&x[i].d==0&&p.beta[i].d==0&&p.domega[i].d==0&&p.dalpha[i].d==0;coefficients_finite=coefficients_finite&&Kokkos::isfinite(x[i])&&Kokkos::isfinite(p.beta[i])&&Kokkos::isfinite(p.domega[i])&&Kokkos::isfinite(p.dalpha[i]);for(int j=0;j<3;++j){zero=zero&&p.omega_hessian[i][j].d==0;coefficients_finite=coefficients_finite&&Kokkos::isfinite(p.omega_hessian[i][j]);}}
 for(int a=0;a<4;++a){if(a)std::cout<<',';Matrix(c.scaled[a]);for(int i=0;i<3;++i)for(int j=0;j<3;++j){zero=zero&&c.scaled[a][i][j].d==0;coefficients_finite=coefficients_finite&&Kokkos::isfinite(c.scaled[a][i][j]);}}
 std::cout<<"],\"reference_and_coefficients_zero_tangent\":"<<(zero?"true":"false")
 <<",\"all_input_finite\":"<<(AllFinite(u)&&AllFinite(p.state)&&coefficients_finite&&Kokkos::isfinite(p.alpha)&&Kokkos::isfinite(p.omega)&&Kokkos::isfinite(p.radius)&&Kokkos::isfinite(p.L)&&Kokkos::isfinite(p.b)&&Kokkos::isfinite(p.k_bar)&&Kokkos::isfinite(p.k_physical)?"true":"false")
 <<",\"geometry_valid\":"<<(hyp::Geometry(u.metric).valid&&hyp::Geometry(p.state.metric).valid?"true":"false")
 <<",\"positive_lapse_chi\":"<<(u.alpha.value.v>0&&u.chi.value.v>0?"true":"false")
 <<",\"SPD\":"<<(SPD(u)?"true":"false")<<",\"uses_legacy_near\":"<<(rwm::UsesLegacyNear(p,u)?"true":"false");
}
hyp::Z4cJet<double> Along(Jet u,double s){Visit(u,[&](D&x){x.v+=s*x.d;x.d=0;});return Values(u);}
void EmitFD(const hyp::LayerPoint<double>&p,const Jet&u,const double*x){
 const auto c=rwm::ReferenceConnection(p,x);std::cout<<",\"FD\":[";int index=0;
 for(double h:{1e-3,5e-4,2.5e-4,1.25e-4,6.25e-5}){if(index++)std::cout<<',';const auto plus=Along(u,h);const auto minus=Along(u,-h);
  std::cout<<"{\"h\":"<<h<<",\"plus\":";const auto fp=Call(p,plus,c,false);EmitOutput(fp);std::cout<<",\"minus\":";const auto fm=Call(p,minus,c,false);EmitOutput(fm);
  const auto pd=Lift(plus);const auto md=Lift(minus);
  std::cout<<",\"side_fields_finite\":"<<(AllFinite(pd)&&AllFinite(md)?"true":"false")<<",\"side_positive_SPD\":"<<(plus.alpha.value>0&&minus.alpha.value>0&&plus.chi.value>0&&minus.chi.value>0&&SPD(pd)&&SPD(md)?"true":"false")<<'}';}
 std::cout<<']';
}
void Direct(){int base_id=0;
 for(double a:{.5,2.})for(double r:radii)for(int direction=0;direction<2;++direction)for(int family=0;family<4;++family){
  hyp::LayerReference<double>ref(1,a,{true,.05,.95});ref.Validate();const double x[3]={r*directions[direction][0],r*directions[direction][1],r*directions[direction][2]};
  const auto p=ref.At(x[0],x[1],x[2]);const auto pd=CastPoint(p);const D xd[3]={x[0],x[1],x[2]};const auto c=rwm::ReferenceConnection(pd,xd);const auto base=State(p,10+family);
  for(int variant=0;variant<2;++variant)for(int seed=0;seed<17;++seed){if(variant&&seed!=6&&seed!=7&&seed!=8&&seed!=15)continue;
   const auto u=SeedField(base,seed,variant);const auto out=Call(pd,u,c,false);const auto old=Call(pd,u,c,true);
   std::cout<<"{\"kind\":\"direct-dual\",\"base_index\":"<<base_id<<",\"a\":"<<a<<",\"nominal_radius\":"<<r<<",\"direction\":"<<direction<<",\"family\":\""<<family_names[family]<<"\",\"seed_index\":"<<seed<<",\"seed_id\":\""<<seed_names[seed]<<"\",\"zero_primal_gradients\":"<<(variant?"true":"false")<<",\"stored_radius\":"<<p.radius<<",\"W_context\":"<<hyp::SmoothCutoff(p.radius,.45,.85).value;
   Context(pd,xd,c,u);std::cout<<",\"new\":";EmitOutput(out);std::cout<<",\"legacy\":";EmitOutput(old);
   if(variant==0&&seed==1&&a==.5&&direction==1&&(r==.025||r==.5||r==.84||r==.995))EmitFD(p,u,x);
   std::cout<<",\"helper_calls_cumulative\":"<<helper_calls<<"}\n";
  }++base_id;
 }
}
void Closed(){const int seeds[6][3]={{0,0,0},{1,0,0},{0,1,0},{1,1,0},{1,-2,0},{0,0,1}};
 Output<D>zero_old[3]{};
 for(int which=0;which<3;++which)for(int seed=0;seed<6;++seed){hyp::LayerPoint<double>p{};p.alpha=p.omega=p.L=1;p.state.alpha.value=p.state.chi.value=1;
  for(int i=0;i<3;++i)p.state.metric.g[i][i]=1;
  if(which==1)p.state.chi.d[0]=1;if(which==2)p.dalpha[0]=p.state.alpha.d[0]=2;
  auto base=p.state;base.alpha.value=std::ldexp(1.,which==1?300:-300);base.chi.value=std::ldexp(1.,which==0?601:which==1?-600:600);
  for(int j=0;j<3;++j)base.alpha.d[j]=base.chi.d[j]=0;
  if(which==0)p.domega[0]=.5;if(which==1)base.chi.d[0]=-base.chi.value;if(which==2)base.alpha.d[0]=base.alpha.value;
  auto u=Lift(base);const int xa=seeds[seed][0],xc=seeds[seed][1],xg=seeds[seed][2];u.alpha.value.d=xa*base.alpha.value;u.chi.value.d=xc*base.chi.value;
  if(which==1)u.chi.d[0].d=-xc*base.chi.value+xg*base.chi.value;if(which==2)u.alpha.d[0].d=(xa+xg)*base.alpha.value;
  const auto pd=CastPoint(p);const D xd[3]{};rwm::Connection<D>c{};c.valid=true;
  const auto out=Call(pd,u,c,false);const auto old=Call(pd,u,c,true);if(seed==0)zero_old[which]=old;
  std::cout<<"{\"kind\":\"closed-dual\",\"case_index\":"<<which<<",\"case_id\":\""<<closed_names[which]<<"\",\"seed_index\":"<<seed<<",\"xi\":["<<xa<<','<<xc<<','<<xg<<']';
  Context(pd,xd,c,u);std::cout<<",\"new\":";EmitOutput(out);std::cout<<",\"legacy\":";EmitOutput(old);std::cout<<",\"helper_calls_cumulative\":"<<helper_calls<<"}\n";
 }
 for(int which=0;which<3;++which){std::cout<<"{\"kind\":\"legacy-negative\",\"case_index\":"<<which<<",\"case_id\":\""<<closed_names[which]<<"\",\"reused_seed_index\":0,\"legacy\":";EmitOutput(zero_old[which]);std::cout<<",\"helper_calls_cumulative\":"<<helper_calls<<"}\n";}
}
}
int main(int argc,char**argv){try{if(argc!=2)throw std::runtime_error("exact fixed direct/closed mode required");std::cout<<std::setprecision(17);const std::string mode=argv[1];
 if(mode=="direct")Direct();else if(mode=="closed")Closed();else throw std::runtime_error("unknown fixed mode");return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
