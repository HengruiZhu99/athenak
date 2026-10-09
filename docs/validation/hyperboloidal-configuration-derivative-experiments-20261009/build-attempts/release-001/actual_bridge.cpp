// Private actual C0/spatial-norm continuum bridge; no discrete radial operator.
#include "baseline_dual_spatial.hpp"
#include "inputs/basis/reference_conversion.hpp"
#include "all_m_data.hpp"
#include "../../continuum/preferred/native-overlay/spatial-norm-family/native_injection.hpp"
#undef InteriorLayerGauge
#undef AssembleGaugeInterior
#include <algorithm>
#include <string>
#include <vector>
using X3=std::array<double,3>;
using Raw=std::array<double,22>;
using TJ=totalj::Jet<double>;
using TM=totalj::MatrixJet<double>;
const hyp::LayerReference<double> reference(1.,.5,{true,.05,.95});
constexpr int ti[6]={0,0,0,1,1,2},tj[6]={0,1,2,1,2,2};

double Norm(const Raw&a){double s=0;for(auto x:a)s+=x*x;return std::sqrt(s);}
double Max(const Raw&a){double v=0;for(auto x:a)v=std::max(v,std::abs(x));return v;}
Raw Difference(const Raw&a,const Raw&b){Raw x{};for(int i=0;i<22;++i)x[i]=a[i]-b[i];return x;}
TJ Plus(TJ a,const TJ&b){return totalj::Add(a,b);}
TJ Times(TJ a,double b){return totalj::Scale(a,b);}
J Native(const TJ&a){J b(D(0,a.value));for(int i=0;i<3;++i){b.d[i]=D(0,a.d[i]);for(int j=0;j<3;++j)b.dd[i][j]=D(0,a.dd[i][j]);}return b;}
template<class T> totalj::Jet<D> Direction(const totalj::Jet<T>&a){totalj::Jet<D>b;b.value=D(0,a.value);for(int i=0;i<3;++i){b.d[i]=D(0,a.d[i]);for(int j=0;j<3;++j)b.dd[i][j]=D(0,a.dd[i][j]);}return b;}
J Native(const totalj::Jet<D>&a){J b(a.value);for(int i=0;i<3;++i){b.d[i]=a.d[i];for(int j=0;j<3;++j)b.dd[i][j]=a.dd[i][j];}return b;}
template<class T> totalj::Jet<D> ReferenceScalar(const hyp::ScalarJet<T>&a){totalj::Jet<D>b;b.value=a.value;for(int i=0;i<3;++i){b.d[i]=a.d[i];for(int j=0;j<3;++j)b.dd[i][j]=a.dd[i][j];}return b;}
totalj::MatrixJet<D> ReferenceMatrix(const hyp::MetricJet<double>&a){totalj::MatrixJet<D>b{};for(int i=0;i<3;++i)for(int j=0;j<3;++j){b[i][j].value=a.g[i][j];for(int d=0;d<3;++d){b[i][j].d[d]=a.dg[d][i][j];for(int e=0;e<3;++e)b[i][j].dd[d][e]=a.ddg[d][e][i][j];}}return b;}
struct Physical {TJ alpha{},P{},theta{};std::array<TJ,3> beta{},lambda{};TM h{},s{};};
Physical SeedPhysical(int j,int m,int channel,int phase,const X3&x,const totalj::WJet<double>&w){
  const auto c=allm::ChannelAt(j,channel);
  const auto f=(m==0&&phase==0)?totalj::EvaluateBasis(j,c.spin,c.L,x,w):allm::Evaluate(j,m,c.spin,c.L,phase==1,x,w);
  Physical p{};
  if(c.kind==0)p.alpha=f.component[0];if(c.kind==2)p.P=f.component[0];if(c.kind==3)p.theta=f.component[0];
  if(c.kind==1)for(int i=0;i<3;++i)p.h[i][i]=Times(f.component[0],1/std::sqrt(3.));
  if(c.kind==4||c.kind==5)for(int i=0;i<3;++i)(c.kind==4?p.beta:p.lambda)[i]=f.component[i];
  if(c.kind==6||c.kind==7)for(int i=0;i<3;++i)for(int k=0;k<3;++k)(c.kind==6?p.h:p.s)[i][k]=f.component[3*i+k];
  return p;
}
Jet LiftPhysical(const hyp::LayerPoint<double>&p,const Physical&v){
  Jet u=Lift(p.state);const auto bar=hyp::PenroseMetric(p.state.metric,p.state.chi);
  totalj::MatrixJet<D> h{},s{},Aref{};
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){h[i][j]=Direction(v.h[i][j]);s[i][j]=Direction(v.s[i][j]);Aref[i][j].value=p.state.a.k[i][j];for(int d=0;d<3;++d)Aref[i][j].d[d]=p.state.a.dk[d][i][j];}
  const auto metric=totalj::ConvertMetric(ReferenceMatrix(bar),ReferenceScalar(p.state.chi),h);
  const auto a=totalj::ConvertA(metric,Aref,s);
  Scalar(u.chi,Native(metric.chi));Scalar(u.alpha,Native(v.alpha));Scalar(u.trace,Native(v.P));Scalar(u.theta,Native(v.theta));
  for(int i=0;i<3;++i){Vector(u.beta,i,Native(v.beta[i]));Vector(u.lambda,i,Native(v.lambda[i]));for(int j=i;j<3;++j){SetMetric(u,i,j,Metric(u,i,j)+Native(metric.metric[i][j]));SetA(u,i,j,A(u,i,j)+Native(a[i][j]));}}
  return u;
}
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

using R3=std::array<std::array<double,3>,3>;
R3 Rotation(){const X3 n={.36,-.48,.8};const double c=std::cos(.7),s=std::sin(.7);R3 r{};for(int i=0;i<3;++i)for(int j=0;j<3;++j){r[i][j]=(i==j?c:0)+(1-c)*n[i]*n[j];for(int k=0;k<3;++k)r[i][j]-=s*static_cast<double>((i-j)*(j-k)*(k-i)/2)*n[k];}return r;}
X3 RotatePoint(const X3&x,const R3&r){X3 y{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)y[i]+=r[i][j]*x[j];return y;}
TJ RotateScalar(const TJ&a,const R3&r){TJ b{};b.value=a.value;for(int i=0;i<3;++i)for(int j=0;j<3;++j){b.d[i]+=r[i][j]*a.d[j];for(int k=0;k<3;++k)for(int l=0;l<3;++l)b.dd[i][k]+=r[i][j]*r[k][l]*a.dd[j][l];}return b;}
Physical RotatePhysical(const Physical&a,const R3&r){Physical b{};b.alpha=RotateScalar(a.alpha,r);b.P=RotateScalar(a.P,r);b.theta=RotateScalar(a.theta,r);for(int i=0;i<3;++i)for(int j=0;j<3;++j){b.beta[i]=Plus(b.beta[i],Times(RotateScalar(a.beta[j],r),r[i][j]));b.lambda[i]=Plus(b.lambda[i],Times(RotateScalar(a.lambda[j],r),r[i][j]));for(int k=0;k<3;++k)for(int l=0;l<3;++l){b.h[i][k]=Plus(b.h[i][k],Times(RotateScalar(a.h[j][l],r),r[i][j]*r[k][l]));b.s[i][k]=Plus(b.s[i][k],Times(RotateScalar(a.s[j][l],r),r[i][j]*r[k][l]));}}return b;}

Jet ChartState(const hyp::LayerPoint<double>&p,double finite,int column,double scale,bool dual){Jet u=Lift(p.state);for(int c=0;c<20;++c){J z(D(finite*std::sin(c+1.)));for(int i=0;i<3;++i){z.d[i]=finite*std::cos((c+1.)*(i+1.))/3;for(int j=0;j<3;++j)z.dd[i][j]=finite*std::sin((c+1.)*(i+j+2.))/5;}Seed(u,c,z);}if(column>=0){J z(dual?D(0,1):D(scale));for(int i=0;i<3;++i){z.d[i]=dual?D(0,.13*(i+1)):D(scale*.13*(i+1));for(int j=0;j<3;++j)z.dd[i][j]=dual?D(0,.07*(i+j+1)):D(scale*.07*(i+j+1));}Seed(u,column,z);}Consistent(u);return u;}
TJ Polynomial(double a,const X3&x){TJ j{};const double rho=x[0]*x[0]+x[1]*x[1]+x[2]*x[2];j.value=a*(1+rho+rho*rho);for(int i=0;i<3;++i){j.d[i]=2*a*x[i]*(1+2*rho);for(int k=0;k<3;++k)j.dd[i][k]=a*((i==k?2:0)*(1+2*rho)+8*x[i]*x[k]);}return j;}
Physical PureGauge(const X3&x,int kind){Physical p{};if(kind==0)p.alpha=Polynomial(.17,x);else for(int i=0;i<3;++i)p.beta[i]=Polynomial(.11*(i+1),x);return p;}
Raw GaugeRate(const X3&x,int kind){const auto p=reference.At(x[0],x[1],x[2]);return Derivative(ActualDual(p,LiftPhysical(p,PureGauge(x,kind))));}
Jet RawTangent(const hyp::LayerPoint<double>&p,const std::array<TJ,22>&a){Jet u=Lift(p.state);Scalar(u.chi,Native(a[0]));Scalar(u.trace,Native(a[7]));Scalar(u.theta,Native(a[17]));Scalar(u.alpha,Native(a[18]));for(int q=0;q<6;++q){SetMetric(u,ti[q],tj[q],Metric(u,ti[q],tj[q])+Native(a[1+q]));SetA(u,ti[q],tj[q],A(u,ti[q],tj[q])+Native(a[8+q]));}for(int i=0;i<3;++i){Vector(u.lambda,i,Native(a[14+i]));Vector(u.beta,i,Native(a[19+i]));}return u;}
std::array<TJ,22> VariationJets(const Jet&u){std::array<TJ,22>a{};auto scalar=[](const auto&b){TJ v{};v.value=b.value.d;for(int i=0;i<3;++i){v.d[i]=b.d[i].d;for(int j=0;j<3;++j)v.dd[i][j]=b.dd[i][j].d;}return v;};a[0]=scalar(u.chi);a[7]=scalar(u.trace);a[17]=scalar(u.theta);a[18]=scalar(u.alpha);for(int q=0;q<6;++q){const int i=ti[q],j=tj[q];a[1+q].value=u.metric.g[i][j].d;a[8+q].value=u.a.k[i][j].d;for(int d=0;d<3;++d){a[1+q].d[d]=u.metric.dg[d][i][j].d;a[8+q].d[d]=u.a.dk[d][i][j].d;for(int e=0;e<3;++e)a[1+q].dd[d][e]=u.metric.ddg[d][e][i][j].d;}}for(int i=0;i<3;++i){a[14+i].value=u.lambda.value[i].d;a[19+i].value=u.beta.value[i].d;for(int d=0;d<3;++d){a[14+i].d[d]=u.lambda.d[d][i].d;a[19+i].d[d]=u.beta.d[d][i].d;for(int e=0;e<3;++e){a[14+i].dd[d][e]=u.lambda.dd[d][e][i].d;a[19+i].dd[d][e]=u.beta.dd[d][e][i].d;}}}return a;}
Jet MapAt(const X3&x,int c){const auto p=reference.At(x[0],x[1],x[2]);const double rho=x[0]*x[0]+x[1]*x[1]+x[2]*x[2];return LiftPhysical(p,SeedPhysical(2,0,c,0,x,{1+rho+rho*rho,1+2*rho,2}));}
double MapJetError(const X3&x,int c,double h){const auto exact=VariationJets(MapAt(x,c));const auto center=Values(MapAt(x,c),true);std::array<TJ,22>fd{};for(int f=0;f<22;++f)fd[f].value=center[f];const int s[4]={-2,-1,1,2};const double d[4]={1,-8,8,-1},dd[4]={-1,16,16,-1};for(int i=0;i<3;++i){for(int q=0;q<4;++q){auto y=x;y[i]+=s[q]*h;const auto v=Values(MapAt(y,c),true);for(int f=0;f<22;++f){fd[f].d[i]+=d[q]*v[f]/(12*h);fd[f].dd[i][i]+=dd[q]*v[f]/(12*h*h);}}for(int f=0;f<22;++f)fd[f].dd[i][i]-=30*center[f]/(12*h*h);for(int k=0;k<i;++k){for(int q=0;q<4;++q)for(int t=0;t<4;++t){auto y=x;y[i]+=s[q]*h;y[k]+=s[t]*h;const auto v=Values(MapAt(y,c),true);for(int f=0;f<22;++f)fd[f].dd[i][k]+=d[q]*d[t]*v[f]/(144*h*h);}for(int f=0;f<22;++f)fd[f].dd[k][i]=fd[f].dd[i][k];}}
  double error=0;for(int f=0;f<22;++f)for(int i=0;i<3;++i){error=std::max(error,std::abs(fd[f].d[i]-exact[f].d[i])/std::max(1.,std::abs(exact[f].d[i])));if(f==0||(f>=1&&f<=6))for(int j=0;j<3;++j)error=std::max(error,std::abs(fd[f].dd[i][j]-exact[f].dd[i][j])/std::max(1.,std::abs(exact[f].dd[i][j])));}return error;
}
Raw ProjectPoint(Raw q){hyp::MetricJet<double>g{};for(int t=0;t<6;++t)g.g[ti[t]][tj[t]]=g.g[tj[t]][ti[t]]=q[1+t];const auto geo=hyp::Geometry(g);if(!geo.valid)throw std::runtime_error("invalid point projector metric");const double scale=1/std::cbrt(geo.determinant);double trace=0;for(int t=0;t<6;++t)trace+=(ti[t]==tj[t]?1:2)*geo.inverse[ti[t]][tj[t]]*q[8+t];for(int t=0;t<6;++t){q[1+t]=scale*g.g[ti[t]][tj[t]];q[8+t]-=g.g[ti[t]][tj[t]]*trace/3;}return q;}
double ProjectorError(const hyp::LayerPoint<double>&p,const Jet&u){const Raw exact=Values(u,true),base=Values(Lift(p.state),false);double best=1e99;for(double e:{1e-4,1e-5,1e-6}){auto a=base,b=base;for(int i=0;i<22;++i){a[i]+=e*exact[i];b[i]-=e*exact[i];}a=ProjectPoint(a);b=ProjectPoint(b);Raw fd{};for(int i=0;i<22;++i)fd[i]=(a[i]-b[i])/(2*e);best=std::min(best,Norm(Difference(fd,exact))/std::max(1.,Norm(exact)));}return best;}
double SpatialNormals(const hyp::LayerPoint<double>&p,const Jet&u){auto inv=totalj::MatrixInverse(ReferenceMatrix(p.state.metric));totalj::MatrixJet<D>dg{},da{},aref{},raised{};for(int i=0;i<3;++i)for(int j=0;j<3;++j){const J g=Metric(u,i,j),a=A(u,i,j);dg[i][j].value=g.v.d;da[i][j].value=a.v.d;aref[i][j].value=p.state.a.k[i][j];for(int d=0;d<3;++d){dg[i][j].d[d]=g.d[d].d;da[i][j].d[d]=a.d[d].d;aref[i][j].d[d]=p.state.a.dk[d][i][j];for(int e=0;e<3;++e)dg[i][j].dd[d][e]=g.dd[d][e].d;}}
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int k=0;k<3;++k)for(int l=0;l<3;++l)raised[i][j]=totalj::Add(raised[i][j],totalj::Multiply(totalj::Multiply(inv[i][k],inv[j][l]),aref[k][l]));
  const auto gtrace=totalj::Contract(inv,dg),atrace=totalj::Sub(totalj::Contract(inv,da),totalj::Contract(raised,dg));double e=std::max(std::abs(gtrace.value.v),std::abs(atrace.value.v));for(int i=0;i<3;++i){e=std::max({e,std::abs(gtrace.d[i].v),std::abs(atrace.d[i].v)});for(int j=0;j<3;++j)e=std::max(e,std::abs(gtrace.dd[i][j].v));}double scale=1;for(auto x:Values(u,true))scale=std::max(scale,std::abs(x));for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int d=0;d<3;++d){scale=std::max({scale,std::abs(u.metric.dg[d][i][j].d),std::abs(u.a.dk[d][i][j].d)});for(int k=0;k<3;++k)scale=std::max(scale,std::abs(u.metric.ddg[d][k][i][j].d));}return e/scale;
}
std::array<double,8> GaugeCdot(const X3&x,int kind,double h){const auto center=GaugeRate(x,kind);std::array<TJ,22>jet{};for(int f=0;f<22;++f)jet[f].value=center[f];const int s[4]={-2,-1,1,2};const double d[4]={1,-8,8,-1},dd[4]={-1,16,16,-1};for(int i=0;i<3;++i){for(int q=0;q<4;++q){auto y=x;y[i]+=s[q]*h;const auto v=GaugeRate(y,kind);for(int f=0;f<22;++f){jet[f].d[i]+=d[q]*v[f]/(12*h);jet[f].dd[i][i]+=dd[q]*v[f]/(12*h*h);}}for(int f=0;f<22;++f)jet[f].dd[i][i]-=30*center[f]/(12*h*h);for(int k=0;k<i;++k){for(int q=0;q<4;++q)for(int t=0;t<4;++t){auto y=x;y[i]+=s[q]*h;y[k]+=s[t]*h;const auto v=GaugeRate(y,kind);for(int f=0;f<22;++f)jet[f].dd[i][k]+=d[q]*d[t]*v[f]/(144*h*h);}for(int f=0;f<22;++f)jet[f].dd[k][i]=jet[f].dd[i][k];}}
  const auto p=reference.At(x[0],x[1],x[2]);return Constraints(RawTangent(p,jet),p);
}

void PrintRaw(const Raw&r){for(auto x:r)std::cout<<x<<' ';}
void Batch(){int j,m,c,phase,rot;X3 x;totalj::WJet<double>w;while(std::cin>>j>>m>>c>>phase>>rot>>x[0]>>x[1]>>x[2]>>w.value>>w.rho_d>>w.rho_dd){auto v=SeedPhysical(j,m,c,phase,x,w);if(rot){const auto r=Rotation();v=RotatePhysical(v,r);x=RotatePoint(x,r);}const auto p=reference.At(x[0],x[1],x[2]);if(!(p.omega>0))throw std::runtime_error("noninterior batch point");const auto u=LiftPhysical(p,v);const auto f=Derivative(ActualDual(p,u));PrintRaw(Values(u,true));PrintRaw(f);const auto n=InputNormals(p,u),o=OutputNormals(p,f);std::cout<<n[0]<<' '<<n[1]<<' '<<o[0]<<' '<<o[1]<<' '<<p.omega<<'\n';}}
void Local(){
  double ref_rhs=0,ref_c=0,input=0,output=0,fd_best=0,fd_worst=0,negative=0,tt=0,initial_q=0,origin=0,projector=0,chart=0,spatial_normals=0,laplacian=0;
  std::vector<double> fdrows;const X3 n={.36,-.48,.8};
  for(double r:{0.,.025,.05,.1,.2,.3,.45,.6,.8,.85,.9,.95,.98,std::sqrt(.9973046875)}){X3 x{r*n[0],r*n[1],r*n[2]};const auto p=reference.At(x[0],x[1],x[2]);ref_rhs=std::max(ref_rhs,Max(Plain(ActualDual(p,Lift(p.state)))));
    auto u0=Lift(p.state);const auto qc=hyp::EvolvedConstraints(u0,Omega(u0,p));ref_c=std::max(ref_c,std::abs(qc.hamiltonian.v));ref_c=std::max(ref_c,std::abs(qc.z4.theta_physical.v));for(int i=0;i<3;++i){ref_c=std::max(ref_c,std::abs(qc.momentum[i].v));ref_c=std::max(ref_c,std::abs(qc.z4.z_covector[i].v));}
    for(int c=0;c<20;++c){const auto u=LiftPhysical(p,SeedPhysical(2,0,c,0,x,{1,.3,.2}));const auto f=Derivative(ActualDual(p,u));const auto a=InputNormals(p,u),b=OutputNormals(p,f);for(int i=0;i<2;++i){input=std::max(input,std::abs(a[i])/std::max(1.,Norm(Values(u,true))));output=std::max(output,std::abs(b[i])/std::max(1.,Norm(f)));}projector=std::max(projector,ProjectorError(p,u));auto v=u;Consistent(v);chart=std::max(chart,Norm(Difference(Values(v,true),Values(u,true)))/std::max(1.,Norm(Values(u,true))));spatial_normals=std::max(spatial_normals,SpatialNormals(p,u));}
  }
  for(double r:{.025,.3,.6,.85,.98})for(double finite:{0.,.003}){X3 x{r*n[0],r*n[1],r*n[2]};const auto p=reference.At(x[0],x[1],x[2]);for(int c=0;c<20;++c){const auto u=ChartState(p,finite,c,0,true);const auto exact=Derivative(ActualDual(p,u));double best=1e99;for(double e:{1e-4,3e-5,1e-5,3e-6,1e-6}){const auto plus=ActualDouble(p,ChartState(p,finite,c,e,false)),minus=ActualDouble(p,ChartState(p,finite,c,-e,false));Raw fd{};for(int f=0;f<22;++f)fd[f]=(plus[f]-minus[f])/(2*e);const double err=Norm(Difference(fd,exact))/std::max(1.,Norm(exact));best=std::min(best,err);fd_worst=std::max(fd_worst,err);fdrows.push_back(err);}fd_best=std::max(fd_best,best);}}
  {const auto p=reference.At(.9,0.,0.);const auto u=ChartState(p,0.,4,0,true);hyp::LayerGaugeParameters g;g.physical_trace_lapse=true;g.preferred_source=false;g.scri_lapse_damping=2;const auto good=Gauge(Reference(p),u,.5,true);const auto badparts=hyp::ResearchNativeSpatialGauge(Reference(p),u,g);hyp::GaugeRHS<D>bad{};hyp::ResearchNativeSpatialAssemble(badparts,D(p.omega),bad);for(int i=0;i<3;++i)negative=std::max(negative,std::abs(good.beta[i].d-bad.beta[i].d));}
  for(const X3&x:{X3{0,0,0},X3{.01,-.02,.03}}){const auto p=reference.At(x[0],x[1],x[2]);auto u=Lift(p.state);J h(D(0,x[2]*x[2]));h.d[2]=D(0,2*x[2]);h.dd[2][2]=D(0,2);SetMetric(u,0,1,Metric(u,0,1)+h);J a(D(0,x[2]));a.d[2]=D(0,1);SetA(u,0,1,A(u,0,1)+a);const auto f=Derivative(ActualDual(p,u));Raw expected{};expected[2]=-2*x[2];expected[9]=-1;tt=std::max(tt,Max(Difference(f,expected)));for(auto q:Constraints(u,p))tt=std::max(tt,std::abs(q));}
  for(int j=0;j<3;++j)for(int s=0;s<3;++s)for(int L=std::abs(j-s);L<=j+s;++L){const auto a=totalj::EvaluateBasis(j,s,L,X3{0,0,0},{1,1,2});const auto b=allm::Evaluate(j,0,s,L,0,X3{0,0,0},{1,1,2});for(int k=0;k<a.components;++k){origin=std::max(origin,std::abs(a.component[k].value-b.component[k].value));for(int d=0;d<3;++d){origin=std::max(origin,std::abs(a.component[k].d[d]-b.component[k].d[d]));for(int e=0;e<3;++e)origin=std::max(origin,std::abs(a.component[k].dd[d][e]-b.component[k].dd[d][e]));}}}
  for(int j=0;j<3;++j)for(int s=0;s<3;++s)for(int L=std::abs(j-s);L<=j+s;++L)for(const X3&x:{X3{0,0,0},X3{.36,-.48,.8}}){const double rho=x[0]*x[0]+x[1]*x[1]+x[2]*x[2];const auto p=totalj::EvaluateBasis(j,s,L,x,{1,0,0}),v=totalj::EvaluateBasis(j,s,L,x,{1+rho+rho*rho,1+2*rho,2});for(int c=0;c<p.components;++c){double trace=0;for(int d=0;d<3;++d)trace+=v.component[c].dd[d][d];const double exact=(8*rho+(4*L+6)*(1+2*rho))*p.component[c].value;laplacian=std::max(laplacian,std::abs(trace-exact)/std::max(1.,std::abs(exact)));}}
  std::cout<<"{\"reference_rhs_max\":"<<ref_rhs<<",\"reference_constraints_max\":"<<ref_c<<",\"input_normal_scaled\":"<<input<<",\"output_normal_scaled\":"<<output<<",\"spatial_tangent_normals_scaled\":"<<spatial_normals<<",\"native_point_projector_best_eps_error\":"<<projector<<",\"coordinate_zz_chart_tangent_error\":"<<chart<<",\"solid_laplacian_scaled_error\":"<<laplacian<<",\"double_dual_best_eps_max\":"<<fd_best<<",\"double_dual_all_eps_max\":"<<fd_worst<<",\"double_only_wrapper_difference\":"<<negative<<",\"TT_core_max\":"<<tt<<",\"origin_primary_allm_max\":"<<origin<<",\"fd_cases\":200,\"actual_reference_map_jet_fd\":[";bool first=true;
  for(double r:{.025,.3,.6,.9,.98})for(int c:{1,10,15}){const X3 x{r*n[0],r*n[1],r*n[2]};if(!first)std::cout<<',';first=false;std::cout<<"{\"r\":"<<r<<",\"channel\":"<<c<<",\"errors\":[";bool f=true;for(double h:{.001,.0005,.00025}){if(!f)std::cout<<',';f=false;std::cout<<MapJetError(x,c,h);}std::cout<<"]}";}
  std::cout<<"],\"pure_gauge\":[";first=true;
  for(double r:{0.,.025,.3,.6,.9,.98})for(int kind=0;kind<2;++kind){const X3 x{r*n[0],r*n[1],r*n[2]};const auto p=reference.At(x[0],x[1],x[2]);double q0=0;for(auto q:Constraints(LiftPhysical(p,PureGauge(x,kind)),p))q0=std::max(q0,std::abs(q));initial_q=std::max(initial_q,q0);if(!first)std::cout<<',';first=false;std::cout<<"{\"r\":"<<r<<",\"kind\":"<<kind<<",\"initial_constraint\":"<<q0<<",\"rate_norm\":"<<Norm(GaugeRate(x,kind))<<",\"Cdot\":[";bool f=true;for(double h:{.001,.0005,.00025}){const auto q=GaugeCdot(x,kind,h);if(!f)std::cout<<',';f=false;std::cout<<'[';for(int i=0;i<8;++i){if(i)std::cout<<',';std::cout<<q[i];}std::cout<<']';}std::cout<<"]}";}
  std::cout<<"],\"initial_gauge_constraint_max\":"<<initial_q<<",\"fd_errors\":[";for(std::size_t i=0;i<fdrows.size();++i){if(i)std::cout<<',';std::cout<<fdrows[i];}std::cout<<"]}\n";
}
int FrozenBridgeMain(int argc,char**argv){try{std::cout<<std::setprecision(17);if(argc>1&&std::string(argv[1])=="--batch")Batch();else Local();}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
