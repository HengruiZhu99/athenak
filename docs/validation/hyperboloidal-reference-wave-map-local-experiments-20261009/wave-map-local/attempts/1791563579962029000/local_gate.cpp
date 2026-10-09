#include "dual_helpers.hpp"
#include "nonlinear_values.hpp"
#include "test_support.hpp"
namespace {
namespace hyp=z4c::hyperboloidal;
template<class T> hyp::LayerPoint<T> CastPoint(const hyp::LayerPoint<double>&p){hyp::LayerPoint<T>q{};
 // Only fields consumed by this gauge/source are lifted; none are absent jets.
 q.state=Lift(p.state);q.alpha=p.alpha;q.omega=p.omega;q.radius=p.radius;q.L=p.L;q.b=p.b;q.k_bar=p.k_bar;q.k_physical=p.k_physical;
 for(int i=0;i<3;++i){q.beta[i]=p.beta[i];q.domega[i]=p.domega[i];q.dalpha[i]=p.dalpha[i];for(int j=0;j<3;++j)q.omega_hessian[i][j]=p.omega_hessian[i][j];}return q;}
std::array<double,4> Fields(const hyp::GaugeRHS<double>&f){return {f.alpha,f.beta[0],f.beta[1],f.beta[2]};}
std::array<D,4> Fields(const hyp::GaugeRHS<D>&f){return {f.alpha,f.beta[0],f.beta[1],f.beta[2]};}
void Maximum(double&v,double a,double b){if(!std::isfinite(a)||!std::isfinite(b))throw std::runtime_error("nonfinite comparison operand");v=std::max(v,audit::ScaleError(a,b));}
Jet Direction(const hyp::Z4cJet<double>&s,int col,double eps){auto u=Lift(s);J d(D(eps,1));for(int i=0;i<3;++i){d.d[i]=D(eps*.17*(i+1),.17*(i+1));for(int j=0;j<3;++j)d.dd[i][j]=D(eps*.07*(i+j+1),.07*(i+j+1));}Seed(u,col,d);Consistent(u);return u;}
}
int main(){
 double wide_error=0;const wide::DD small(1e-20);wide_error=std::abs(static_cast<double>((wide::DD(1)+small)-wide::DD(1))-1e-20);
 for(double v:{1e-8,.5,2.,100.}){const wide::DD x(v),z=wide::Sqrt(x),y(1.23456789);wide_error=std::max(wide_error,std::abs(static_cast<double>(z*z-x))/std::max(1.,v));wide_error=std::max(wide_error,std::abs(static_cast<double>((x/y)*y-x))/std::max(1.,v));}
 if(wide_error>1e-28)throw std::runtime_error("double-double sanity failure");
 double embedding=0,adm=0,zero_time=0,transform=0,sourceforms=0,rawref=0,factoredref=0,rawgeom=0,rawfactor=0,Bidentity=0,fullZsource=0,physicalsource=0,core=0,tinymax=0,dualgauge=0,dualsource=0;
 double fdg[3]{},fds[3]{};int rows=0,offrows=0,dualrows=0,core_rows=0,tinyrows=0;
 const std::array<std::array<double,3>,3> directions={{{1,0,0},{.36,-.48,.8},{-.48,.64,.6}}};
 for(double a:{.5,.75,1.,2.})for(const auto&geo:{hyp::LayerParameters{true,.05,.95},hyp::LayerParameters{true,.2,.8},hyp::LayerParameters{false,.05,.95}}){
 hyp::LayerReference<double>reference(1,a,geo);reference.Validate();
 for(double r:{0.,.01,.049,.05,.050001,.1,.2,.200001,.44,.45,.5,.65,.799999,.8,.85,.9,.949999,.95,.98,.999,.99999})for(const auto&n:directions){
  const double x[3]={r*n[0],r*n[1],r*n[2]};const auto p=reference.At(x[0],x[1],x[2]);const auto c=rwm::ReferenceConnection(p,x);if(!c.valid)throw std::runtime_error("reference connection invalid");
  const auto pa=audit::Metric4(p.state,p,nullptr,nullptr,true),ba=audit::Metric4(p.state,p),pe=audit::Embedding(p,x,1,a,geo);
  for(int aa=0;aa<4;++aa)for(int b=0;b<4;++b)for(int d=0;d<4;++d){const double cs=(b&&d)?c.scaled[aa][b-1][d-1]:0;
   Maximum(embedding,cs,p.omega*pe.Gamma[aa][b][d]);Maximum(adm,cs,p.omega*pa.Gamma[aa][b][d]);
   if(b==0||d==0)zero_time=std::max(zero_time,std::abs(p.omega*pa.Gamma[aa][b][d]));
   double expected=p.omega*ba.Gamma[aa][b][d];for(int i=0;i<3;++i)expected-=
      ((aa==b&&d==i+1?1.:0)+(aa==d&&b==i+1?1.:0)-ba.g[b][d]*ba.inv[aa][i+1])*p.domega[i];
   Maximum(transform,p.omega*pa.Gamma[aa][b][d],expected);
  }
  hyp::GaugeRHS<double>rf{};const auto qr=rwm::Gauge(p,p.state,c);if(!rwm::Assemble(qr,p.omega,rf))throw std::runtime_error("reference gauge invalid");
  for(double v:Fields(rf))factoredref=std::max(factoredref,std::abs(v));
  const auto rr=audit::RawGauge(p,p.state,c);for(double v:Fields(rr)){if(!std::isfinite(v))throw std::runtime_error("nonfinite raw reference row");rawref=std::max(rawref,std::abs(v));}
  double Bh=0,Rh=0,Lg=0;const auto gref=hyp::Geometry(p.state.metric);for(int i=0;i<3;++i){Bh+=p.beta[i]*p.domega[i];Rh+=p.beta[i]*p.dalpha[i];for(int j=0;j<3;++j)Lg+=(p.alpha*p.alpha*p.state.chi.value*gref.inverse[i][j]-p.beta[i]*p.beta[j])*c.scaled[0][i][j];}
  Maximum(Bidentity,p.alpha*p.k_physical+Bh+Lg,p.omega*Rh/p.alpha);
  hyp::Z4cRHS<double>gr{};const auto pr=hyp::ConformalRHS(p.state,audit::Omega(p.state,p),10/p.alpha,0.);if(!hyp::AssembleInterior(pr,p.omega,gr))throw std::runtime_error("raw reference geometry invalid");
  rawgeom=std::max({rawgeom,std::abs(gr.chi),std::abs(gr.trace),std::abs(gr.theta)});for(int i=0;i<3;++i){rawgeom=std::max(rawgeom,std::abs(gr.lambda[i]));for(int j=0;j<3;++j)rawgeom=std::max({rawgeom,std::abs(gr.metric[i][j]),std::abs(gr.a[i][j])});}
  auto u=Off(p);const auto gu=hyp::Geometry(u.metric);const auto live=audit::Metric4(u,p);double fs[4]{};rwm::ScaledSource(p,u,c,fs);
  double s=0;for(int b=0;b<4;++b)for(int d=0;d<4;++d)s+=live.inv[b][d]*ba.g[b][d];
  for(int aa=0;aa<4;++aa){double fc=0;for(int b=0;b<4;++b)for(int d=0;d<4;++d)fc+=p.omega*live.inv[b][d]*ba.Gamma[aa][b][d];for(int i=0;i<3;++i)fc-=(4*live.inv[aa][i+1]-s*ba.inv[aa][i+1])*p.domega[i];Maximum(sourceforms,fs[aa],fc);}
  hyp::GaugeRHS<double>f{};if(!rwm::Assemble(rwm::Gauge(p,u,c),p.omega,f))throw std::runtime_error("offconstraint gauge invalid");
  const auto raw=audit::RawGauge(p,u,c);const auto vf=Fields(f),vr=Fields(raw);for(int i=0;i<4;++i)Maximum(rawfactor,vf[i],vr[i]);
  hyp::Z4cRHS<double>gf{};if(!hyp::AssembleInterior(hyp::ConformalRHS(u,audit::Omega(u,p),10/u.alpha.value,0.),p.omega,gf))throw std::runtime_error("live geometry invalid");
  const auto fbar=audit::Metric4(u,p,&gf,&f),fphys=audit::Metric4(u,p,&gf,&f,true);const auto gamma=audit::Contract(fbar),gammaphys=audit::Contract(fphys);
  const double z0=u.theta.value/(p.omega*u.alpha.value);double zbar[4]={z0,0,0,0};for(int i=0;i<3;++i)zbar[i+1]=.5*u.chi.value*(u.lambda.value[i]-gu.contracted[i])-u.beta.value[i]*z0;
  for(int aa=0;aa<4;++aa){Maximum(fullZsource,p.omega*(gamma[aa]+2*zbar[aa]),fs[aa]);double hc=0;for(int b=0;b<4;++b)for(int d=0;d<4;++d)hc+=fphys.inv[b][d]*pa.Gamma[aa][b][d];Maximum(physicalsource,gammaphys[aa]+2*p.omega*p.omega*zbar[aa],hc);}
  ++rows;++offrows;
  if(geo.enabled&&r<=geo.r0){hyp::GaugeRHS<double>target{};const double al=u.alpha.value;
   target.alpha=-al*al*u.trace.value;for(int i=0;i<3;++i){target.alpha+=u.beta.value[i]*u.alpha.d[i];target.beta[i]=al*al*u.chi.value*u.lambda.value[i];for(int j=0;j<3;++j)target.beta[i]+=u.beta.value[j]*u.beta.d[j][i]+.5*al*al*gu.inverse[i][j]*u.chi.d[j]-al*u.chi.value*gu.inverse[i][j]*u.alpha.d[j];}
   const auto vt=Fields(target);for(int i=0;i<4;++i)Maximum(core,vf[i],vt[i]);++core_rows;}
 }
 // Gauge-value-only tiny lapse check. No finite/uniform Fbar claim here.
 for(double r:{0.,.5,.9,.99999}){const double x[3]={r,0,0};const auto p=reference.At(r,0,0);const auto c=rwm::ReferenceConnection(p,x);
  for(double al:{1e-4,1e-100,1e-200,1e-300,1e-320})for(bool zero_beta:{false,true}){auto u=Off(p);u.alpha.value=al;if(zero_beta)for(int i=0;i<3;++i)u.beta.value[i]=0;hyp::GaugeRHS<double>f{};if(!rwm::Assemble(rwm::Gauge(p,u,c),p.omega,f))throw std::runtime_error("tiny positive gauge invalid");for(double v:Fields(f))tinymax=std::max(tinymax,std::abs(v));++tinyrows;}}
 }
 for(double a:{.5,.75,1.,2.})for(double r:{.01,.3,.5,.75,.9,.98})for(const auto&n:directions){
  hyp::LayerReference<double>reference(1,a,{true,.05,.95});const double x[3]={r*n[0],r*n[1],r*n[2]};const auto p=reference.At(x[0],x[1],x[2]);const auto pD=CastPoint<D>(p);const D xd[3]={x[0],x[1],x[2]};const auto c=rwm::ReferenceConnection(p,x);const auto cd=rwm::ReferenceConnection(pD,xd);const auto base=Off(p);
  for(int col=0;col<20;++col){const auto ud=Direction(base,col,0);hyp::GaugeRHS<D>gd{};if(!rwm::Assemble(rwm::Gauge(pD,ud,cd),D(p.omega),gd))throw std::runtime_error("dual gauge invalid");const auto ad=Fields(gd);D sf[4]{};rwm::ScaledSource(pD,ud,cd,sf);
   int level=0;for(double eps:{1e-4,5e-5,2.5e-5}){auto up=Values(Direction(base,col,eps)),um=Values(Direction(base,col,-eps));hyp::GaugeRHS<double>gp{},gm{};if(!rwm::Assemble(rwm::Gauge(p,up,c),p.omega,gp)||!rwm::Assemble(rwm::Gauge(p,um,c),p.omega,gm))throw std::runtime_error("FD gauge invalid");double sp[4]{},sm[4]{};rwm::ScaledSource(p,up,c,sp);rwm::ScaledSource(p,um,c,sm);const auto ap=Fields(gp),am=Fields(gm);
    for(int i=0;i<4;++i){Maximum(fdg[level],ad[i].d,(ap[i]-am[i])/(2*eps));Maximum(fds[level],sf[i].d,(sp[i]-sm[i])/(2*eps));}++level;}
   ++dualrows;
  }
 }
 dualgauge=fdg[2];dualsource=fds[2];
 std::cout<<std::setprecision(17)<<"{\"wide_arithmetic_sanity_error\":"<<wide_error<<",\"reference_rows\":"<<rows<<",\"offconstraint_rows\":"<<offrows<<",\"core_rows\":"<<core_rows<<",\"tiny_gauge_rows\":"<<tinyrows<<",\"dual_directions\":"<<dualrows<<",\"embedding_scaled_connection_error\":"<<embedding<<",\"ADM_scaled_connection_error\":"<<adm<<",\"ADM_scaled_temporal_connection_max\":"<<zero_time<<",\"full_conformal_connection_transform_error\":"<<transform<<",\"physical_conformal_source_error\":"<<sourceforms<<",\"factored_reference_max\":"<<factoredref<<",\"raw_reference_gauge_max\":"<<rawref<<",\"raw_reference_geometry_max\":"<<rawgeom<<",\"factored_unfactored_gauge_error\":"<<rawfactor<<",\"stationary_B_identity_error\":"<<Bidentity<<",\"actual_C0_full_Z_conformal_source_error\":"<<fullZsource<<",\"actual_C0_full_Z_physical_source_error\":"<<physicalsource<<",\"exact_core_harmonic_error\":"<<core<<",\"tiny_gauge_max\":"<<tinymax<<",\"dual_gauge_FD_errors\":["<<fdg[0]<<","<<fdg[1]<<","<<fdg[2]<<"],\"dual_scaled_source_FD_errors\":["<<fds[0]<<","<<fds[1]<<","<<fds[2]<<"]}\n";
}
