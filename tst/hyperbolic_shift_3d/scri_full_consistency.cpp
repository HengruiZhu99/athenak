// Consistency of the current SG1 boundary/adjacent-volume execution path.
// Reuse only analytic manufactured jets and norm helpers from the older audit;
// its old mesh, boundary mode and main are not used here.
#define main unused_legacy_scri_accuracy_main
#include "scri_sampled_accuracy.cpp"
#undef main
#include "manufactured_twist.hpp"
#include "fixed_composite_mesh.hpp"
#include "z4c/hyperboloidal/hyperbolic_composite_exchange.hpp"
#include "z4c/hyperboloidal/hyperbolic_sampled_normal_device.hpp"
#include "z4c/hyperboloidal/hyperbolic_scri_device.hpp"

namespace {
using Device=h::HyperbolicCompositeDevice;
using Surface=h::HyperbolicScriDevice;
using Samples=h::HyperbolicSampledNormalDevice;
using Exchange=h::HyperbolicCompositeExchange;
using A31=std::array<Real,31>;
Device::Buffer UploadAudit(const V&v){Device::Buffer b("audit upload",v.size());
 auto host=Kokkos::create_mirror_view(b);std::copy(v.begin(),v.end(),host.data());Kokkos::deep_copy(b,host);return b;}
V DownloadAudit(Device::ConstBuffer b){auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),b);
 return V(host.data(),host.data()+host.size());}
void Check(bool ok,const char*message){if(!ok)throw std::runtime_error(message);}
Scalar SinAudit(const Scalar&a){Scalar v{};v.value=std::sin(a.value);
 for(int i=0;i<3;++i){v.d[i]=std::cos(a.value)*a.d[i];for(int j=0;j<3;++j)
  v.dd[i][j]=std::cos(a.value)*a.dd[i][j]-v.value*a.d[i]*a.d[j];}return v;}
NativeJets Fixture(const Real*x,int kind,bool boundary=false){
 if(kind==2)return manufactured_twist::Fields(x);
 const bool angular=kind==1;
 auto v=Profile(x,angular,2,boundary);
 if(angular){
  // Nonconstant tangential B exercises the surface atlas and angular filter.
  // Direct-outer lapse has no B coupling, so its regularity/clock is unchanged.
  const Real a[3]={1.3,.7,-.9};Scalar phase{};
  for(int i=0;i<3;++i){phase.value+=a[i]*x[i];phase.d[i]=a[i];}
  for(int i=0;i<3;++i){const int j=(i+1)%3,k=(i+2)%3;Scalar tangent{};
   tangent.value=a[j]*x[k]-a[k]*x[j];tangent.d[k]=a[j];tangent.d[j]=-a[k];
   v.Put(19+i,Scale(Product(tangent,SinAudit(phase)),.003));}
 }
 return v;
}
A31 ExactPoleAudit(const Real*x,int kind,bool boundary=false){
 const auto v=Fixture(x,kind,boundary);const auto u=Model::Jet(x,v.v,v.d,v.dd);const auto o=Model::Omega(x,u,boundary);
 auto parts=h::ConformalRHS(u,o,4.,-.99);Check(h::HyperbolicDamping(x,u,o,parts),"invalid analytic pole");
 A31 out{};Model::StoreGeometry(parts.pole,out.data());return out;
}
A31 ExactNormalAudit(const Real*x,int kind){
 const auto v=Fixture(x,kind,true);const auto u=Model::Jet(x,v.v,v.d,v.dd);const auto o=Model::Omega(x,u,true);
 const auto dual=h::TensorPoleNormalJet(u,o,x,x,4.,-.99,1.,1.,true,true);
 A31 out{};Model::StoreGeometry(h::TensorScriLimit(h::Z4cRHS<Real>{},dual,1.),out.data());return out;
}
A31 GeometryArray(const h::Z4cRHS<Real>&v){A31 out{};Model::StoreGeometry(v,out.data());return out;}
A31 Minus(A31 a,const A31&b){for(int f=0;f<31;++f)a[f]-=b[f];return a;}
struct AuditNorm {
 Real weight=0,square=0,maximum=0,x[3]{};int field=-1;size_t points=0;
 void Add(const A31&v,Real w,const Real*where,int lo=0,int hi=31){
  weight+=w;++points;
  for(int f=lo;f<hi;++f){Check(std::isfinite(v[f]),"nonfinite audit residual");
   square+=w*Weight(f)*v[f]*v[f];if(std::abs(v[f])>maximum){maximum=std::abs(v[f]);field=f;
    for(int a=0;a<3;++a)x[a]=where[a];}}
 }
 void Write(std::ostream&o)const{o<<"{\"weight\":"<<weight<<",\"points\":"<<points
  <<",\"rms\":"<<(weight?std::sqrt(square/weight):0)<<",\"max_component\":"<<maximum
  <<",\"max_field\":"<<field<<",\"max_xyz\":["<<x[0]<<','<<x[1]<<','<<x[2]<<"]}";}
};
struct Regional {
 AuditNorm all,seam,interior;
 void Add(const A31&v,Real w,const Real*x,bool edge,int lo=0,int hi=31){
  all.Add(v,w,x,lo,hi);(edge?seam:interior).Add(v,w,x,lo,hi);}
 void Write(std::ostream&o)const{o<<"{\"all\":";all.Write(o);o<<",\"seam\":";seam.Write(o);o<<",\"interior\":";interior.Write(o);o<<'}';}
};
Real SurfaceArea(int n,int j,int k){const Real a=pi/4;
 const Real l=std::tan(a*(-1+2.*j/n)),r=std::tan(a*(-1+2.*(j+1)/n));
 const Real b=std::tan(a*(-1+2.*k/n)),t=std::tan(a*(-1+2.*(k+1)/n));
 return Primitive(r,t)-Primitive(l,t)-Primitive(r,b)+Primitive(l,b);}
std::array<Real,6> NormalWeights(){std::array<Real,6>w{};
 const Real offsets[7]={-5.5,-4.5,-3.5,-2.5,-1.5,-.5,0};
 for(int i=0;i<6;++i){cubed_sphere::Jet v(1);for(int j=0;j<7;++j){if(i!=j)
  v=v*(cubed_sphere::Jet::Variable(0,0)-cubed_sphere::Jet(offsets[j]))/cubed_sphere::Jet(offsets[i]-offsets[j]);}
  w[i]=v.d[0];}
 return w;
}
h::Z4cJet<Real> DiscreteJet(const Model&m,const V&padded,size_t p){
 V v(31);std::vector<std::array<Real,3>>d(31);
 std::vector<std::array<std::array<Real,3>,3>>dd(31);
 for(int f=0;f<31;++f){v[f]=padded[f*m.grid.count+m.grid.active[p]];Real df[3],ddf[3][3];
  m.BulkDerivatives(padded,f,p,df,ddf,f<=6||f==18||(f>=22&&f<25));
  for(int a=0;a<3;++a){d[f][a]=df[a];for(int b=0;b<3;++b)dd[f][a][b]=ddf[a][b];}}
 return Model::Jet(m.grid.geometry[p].x,v,d,dd);
}
A31 ExactBulk(const Real*x,int kind,h::Z4cRHSParts<Real>*saved=nullptr){
 const auto v=Fixture(x,kind);const auto u=Model::Jet(x,v.v,v.d,v.dd);const auto o=Model::Omega(x,u);
 auto parts=h::ConformalRHS(u,o,4.,-.99);Check(h::HyperbolicDamping(x,u,o,parts),"invalid analytic bulk");
 if(saved)*saved=parts;h::Z4cRHS<Real>rate{};Check(h::AssembleInterior(parts,o.omega,rate),"invalid analytic assembly");
 A31 result=GeometryArray(rate);const auto B=Model::Vector(19,v.v,v.d),D=Model::Vector(25,v.v,v.d);
 const auto gauge=h::HyperbolicGauge(x,u,B,D.value);Check(gauge.valid,"invalid analytic gauge");
 result[18]=gauge.lapse_regular+gauge.lapse_pole/o.omega;
 Real auxiliary[3];Check(h::HyperbolicAuxiliaryRate(u,D,rate.lambda,auxiliary),"invalid analytic auxiliary");
 for(int a=0;a<3;++a){result[19+a]=gauge.flux_rate[a];result[22+a]=gauge.shift_regular[a]+gauge.shift_pole[a]/o.omega;result[25+a]=auxiliary[a];}
 return result;
}

void Audit(Model&m,int kind,std::ostream&out){
 const int n=m.grid.n;const Real dt=1e-5;const auto weights=NormalWeights();
 Controls controls;JetControls(kind==1,2,controls);
 Check(controls.jet_first<1e-7&&controls.jet_second<1e-6&&controls.jet_mixed<1e-6,"analytic profile jets failed");
 Real fixture_first_error=0,fixture_second_error=0;
 const Real probe[3]={.37,-.41,.79},eps=1e-4;const auto oracle=Fixture(probe,kind);
 for(int a=0;a<3;++a){Real xp[3],xm[3];for(int b=0;b<3;++b){xp[b]=probe[b]+(a==b?eps:0);xm[b]=probe[b]-(a==b?eps:0);}
  const auto vp=Fixture(xp,kind),vm=Fixture(xm,kind);
  for(int f=0;f<25;++f){fixture_first_error=std::max(fixture_first_error,std::abs((vp.v[f]-vm.v[f])/(2*eps)-oracle.d[f][a]));
   if(f<=6||f==18||(f>=22&&f<25))fixture_second_error=std::max(fixture_second_error,std::abs((vp.v[f]-2*oracle.v[f]+vm.v[f])/(eps*eps)-oracle.dd[f][a][a]));}}
 Check(fixture_first_error<2e-7&&fixture_second_error<2e-6,"manufactured analytic jet oracle failed independent FD");
 for(size_t p=0;p<m.grid.active.size();++p){const auto v=Fixture(m.grid.geometry[p].x,kind);
  std::copy(v.v.begin(),v.v.end(),m.state.begin()+31*p);}
 for(size_t p=0;p<m.scri.rows.size();++p){const auto v=Fixture(m.scri.rows[p].geometry.x,kind,true);
  std::copy(v.v.begin(),v.v.end(),m.scri.values.begin()+31*p);}
 m.BoundaryValues();const V projected=m.scri.values;
 const auto active=UploadAudit(m.state),surface=UploadAudit(projected);
 Samples samples(m,0,m.scri.rows.size());Surface boundary(m,0,m.scri.rows.size());
 Exchange exchange(m,samples.FirstBlock(),samples.EndBlock());exchange.SetSurface(projected);
 const auto halo=exchange.Fill(active);Check(!halo.invalid,"invalid native exchange");
 Check(!samples.Evaluate(exchange.padded,surface).bad_rows,"invalid native samples");
 const V poles=DownloadAudit(samples.LocalPoleSamples());
 const V angular_rhs=m.angular_plan->Apply(projected,2);
 const auto boundary_summary=boundary.Evaluate(active,surface,dt,UploadAudit(angular_rhs),samples.LocalPoleSamples());
 Check(!boundary_summary.bad_rows,"invalid native boundary rate");
 const V surface_rate=DownloadAudit(boundary.LocalRHS()),dependent=DownloadAudit(boundary.LocalSurface());
 m.scri.values=dependent;exchange.SetSurface(dependent);Check(!exchange.Fill(active).invalid,"invalid final exchange");
 const auto final_surface=UploadAudit(dependent);const V padded=DownloadAudit(exchange.padded);
 const int first=samples.FirstBlock()*n*n*n,count=(samples.EndBlock()-samples.FirstBlock())*n*n*n;
 Device bulk(m.grid,first,count,m.radial_plan.get());
 Check(!bulk.Evaluate(exchange.padded,dt,false,.5,final_surface,{},true).bad_cells,"invalid adjacent-volume RHS");
 const V volume_rate=bulk.DownloadRHS();
 std::map<std::string,Regional> norms;
 Real exact_pole_max=0,exact_trace_max=0,analytic_normal_check=0,reconstruction=0;
 for(size_t p=0;p<m.scri.rows.size();++p){const auto&row=m.scri.rows[p];const auto*x=row.geometry.x;
  const Real area=SurfaceArea(n,row.j,row.k),step=row.radial_spacing;
  const bool seam=row.j<2||row.k<2||row.j>=n-2||row.k>=n-2;
  const auto v=Fixture(x,kind,true);const auto u=Model::Jet(x,v.v,v.d,v.dd);const auto o=Model::Omega(x,u,true);
  auto exact=h::ConformalRHS(u,o,4.,-.99);Check(h::HyperbolicDamping(x,u,o,exact),"invalid exact boundary");
  const auto exact_normal=ExactNormalAudit(x,kind);
  A31 reference=Minus(GeometryArray(exact.regular),exact_normal);
  const auto B=Model::Vector(19,v.v,v.d);const Real zero[3]{};
  const auto gauge=h::HyperbolicGauge(x,u,B,zero);Check(gauge.valid,"invalid exact boundary gauge");
  for(int a=0;a<3;++a)reference[19+a]=gauge.flux_rate[a];
  const auto trace=GeometryArray(h::ScriRegularityTraces(u,o));
  const auto exact_pole=GeometryArray(exact.pole);
  for(int f=0;f<18;++f){exact_pole_max=std::max(exact_pole_max,std::abs(exact_pole[f]));exact_trace_max=std::max(exact_trace_max,std::abs(trace[f]));}
  const auto sampled=m.scri.Sample(p,padded,projected,31);
  const auto du=Model::Jet(x,sampled.value,sampled.d,sampled.dd);const auto dom=Model::Omega(x,du,true);
  auto disc=h::ConformalRHS(du,dom,4.,-.99);Check(h::HyperbolicDamping(x,du,dom,disc),"invalid discrete endpoint");
  const auto dpole=GeometryArray(disc.pole);auto dtrace=GeometryArray(h::ScriRegularityTraces(du,dom));
  const auto metric=h::Geometry(du.metric);Real q=0;
  for(int a=0;a<3;++a)for(int b=0;b<3;++b)q+=metric.inverse[a][b]*x[a]*x[b];
  dtrace[0]=du.chi.value-1/q;dtrace[13]=du.trace.value+3;dtrace[14]=du.theta.value;
  A31 native_normal{},exact_samples{},penalty{},actual{},angular_q{},closure_error{};
  for(int s=0;s<6;++s){const size_t cp=size_t(row.block)*n*n*n+(row.k*n+row.j)*n+n-6+s;
   const auto ep=ExactPoleAudit(m.grid.geometry[cp].x,kind);
   for(int f=0;f<18;++f){native_normal[f]+=weights[s]*(poles[31*(6*p+s)+f]-dpole[f])/step;
    exact_samples[f]+=weights[s]*(ep[f]-exact_pole[f])/step;}}
  for(int f=0;f<31;++f){actual[f]=surface_rate[31*p+f];angular_q[f]=angular_rhs[31*p+f];
   penalty[f]=(4*dpole[f]-8*dtrace[f])/step;}
  auto expected_discrete=Minus(GeometryArray(disc.regular),native_normal);
  for(int f=0;f<18;++f){expected_discrete[f]+=penalty[f]+angular_q[f];reconstruction=std::max(reconstruction,std::abs(actual[f]-expected_discrete[f]));}
  // Null-condition time derivative, evaluated with analytic surface geometry.
  const auto exact_g=h::Geometry(u.metric);Real invn[3]{},exact_q=0,qdot=0;
  for(int a=0;a<3;++a)for(int b=0;b<3;++b)invn[a]+=exact_g.inverse[a][b]*x[b];
  for(int a=0;a<3;++a)exact_q+=x[a]*invn[a];
  for(int a=0;a<3;++a)for(int b=0;b<3;++b){int f=0;for(;f<6;++f)if((Model::ti[f]==a&&Model::tj[f]==b)||(Model::ti[f]==b&&Model::tj[f]==a))break;
   qdot-=invn[a]*reference[1+f]*invn[b];}
  const Real speed_t=(reference[0]*exact_q+u.chi.value*qdot)/(2*std::sqrt(u.chi.value*exact_q)),mu=h::ShiftCoefficients(1.,1.,u.chi.value).mu;
  for(int a=0;a<3;++a)closure_error[25+a]=dependent[31*p+25+a]+speed_t*x[a]/mu;
  norms["surface_full"].Add(Minus(actual,reference),area,x,seam);
  norms["surface_geometry"].Add(Minus(actual,reference),area,x,seam,0,18);
  norms["surface_B"].Add(Minus(actual,reference),area,x,seam,19,22);
  norms["surface_reference"].Add(reference,area,x,seam);
  norms["normal_native"].Add(Minus(native_normal,exact_normal),area,x,seam);
  norms["normal_exact_samples"].Add(Minus(exact_samples,exact_normal),area,x,seam);
  norms["normal_discretized_numerator_effect"].Add(Minus(native_normal,exact_samples),area,x,seam);
  norms["surface_regular"].Add(Minus(GeometryArray(disc.regular),GeometryArray(exact.regular)),area,x,seam);
  norms["surface_penalty"].Add(penalty,area,x,seam);
  norms["surface_angular_Q"].Add(angular_q,area,x,seam);
  norms["dependent_D"].Add(closure_error,area,x,seam);
  if(p%std::max(size_t(1),m.scri.rows.size()/24)==0){const Real eps=1e-4;A31 pole_values[4];const int offsets[4]={-2,-1,1,2};
   for(int s=0;s<4;++s){Real xx[3];for(int a=0;a<3;++a)xx[a]=x[a]*(1+offsets[s]*eps);pole_values[s]=ExactPoleAudit(xx,kind);}
   for(int f=0;f<18;++f)analytic_normal_check=std::max(analytic_normal_check,std::abs((pole_values[0][f]-8*pole_values[1][f]+8*pole_values[2][f]-pole_values[3][f])/(12*eps)-exact_normal[f]));}
  for(int depth=1;depth<=6;++depth){const size_t cp=size_t(row.block)*n*n*n+(row.k*n+row.j)*n+n-depth;const auto*xx=m.grid.geometry[cp].x;
   h::Z4cRHSParts<Real> exactparts{};const auto ref=ExactBulk(xx,kind,&exactparts);A31 actual_bulk{};
   for(int f=0;f<31;++f)actual_bulk[f]=volume_rate[31*(cp-first)+f];
   const auto discrete=DiscreteJet(m,padded,cp);const auto om=Model::Omega(xx,discrete);
   auto parts=h::ConformalRHS(discrete,om,4.,-.99);Check(h::HyperbolicDamping(xx,discrete,om,parts),"invalid discrete adjacent pole");
   A31 pole_error=Minus(GeometryArray(parts.pole),GeometryArray(exactparts.pole));for(auto&v:pole_error)v/=om.omega;
   h::Z4cRHS<Real>central{};Check(h::AssembleInterior(parts,om.omega,central),"invalid discrete adjacent assembly");
   const std::string key="row"+std::to_string(depth)+"_";
   norms[key+"full"].Add(Minus(actual_bulk,ref),area,xx,seam);
   norms[key+"pole_over_Omega"].Add(pole_error,area,xx,seam);
   norms[key+"geometry_central"].Add(Minus(GeometryArray(central),ref),area,xx,seam,0,18);
   norms[key+"transport_KO"].Add(Minus(actual_bulk,GeometryArray(central)),area,xx,seam,0,18);
  }
 }
 Check(exact_pole_max<3e-12&&exact_trace_max<3e-12,"fixture violates analytic scri regularity");
 Check(analytic_normal_check<3e-8,"analytic normal reference failed independent FD check");
 Check(reconstruction<2e-9,"native surface decomposition does not reconstruct actual device RHS");
 if(kind<2)Check(norms["surface_reference"].all.maximum>1e-4,"trivial manufactured surface reference");
 else Check(norms["surface_reference"].all.maximum<1e-10,"twist geometry is not analytically stationary");
 Check(std::abs(norms["surface_full"].all.weight-4*pi)<2e-12,"surface norm area mismatch");
 const char*fixture_names[]={"radial_p2","angular_p2_with_surface_B","nonconstant_twist_metric"};
 out<<"{\"fixture\":\""<<fixture_names[kind]<<"\",\"exact_pole_max\":"<<exact_pole_max
  <<",\"fixture_first_jet_FD_error\":"<<fixture_first_error<<",\"fixture_second_jet_FD_error\":"<<fixture_second_error
  <<",\"exact_regularity_max\":"<<exact_trace_max<<",\"normal_AD_vs_independent_FD\":"<<analytic_normal_check
  <<",\"surface_decomposition_error\":"<<reconstruction<<",\"null_max\":"<<boundary_summary.null_after
  <<",\"dependent_compatibility\":"<<boundary_summary.compatibility<<",\"norms\":{";
 bool first_key=true;for(const auto&[key,norm]:norms){if(!first_key)out<<',';first_key=false;out<<'"'<<key<<"\":";norm.Write(out);}out<<"}}";
}
}
int main(int argc,char**argv){Kokkos::initialize(argc,argv);int status=0;
 try{
  if(argc!=3)throw std::invalid_argument("usage: scri_full_consistency N fresh_output.json");
  const int n=std::stoi(argv[1]);Check(n>=12&&n<=64&&n%2==0,"bounded N must be even in12..64");
  Check(!std::ifstream(argv[2]).good(),"refusing existing report");
  Model model(hyperbolic_reproduction::ContinuousCoreMesh(n),false,true,2,
    Model::NormalDerivative::sampled_numerator,Model::RadialClosure::donor_seven);
  std::ostringstream out;out<<std::setprecision(17)<<"{\"n\":"<<n<<",\"h\":"<<.25/n
   <<",\"scope\":\"Native SG1 device surface and adjacent-volume RHS; analytic regular p2 fields with nonzero RHS, angular B, and nonconstant volume-preserving flat-metric pullback with stationary geometric RHS. Not an evolved solution or global hyperbolicity test.\",\"seam_definition\":\"two angular cell rows on each face, shrinking band\",\"fixtures\":[";
  Audit(model,0,out);out<<',';Audit(model,1,out);out<<',';Audit(model,2,out);out<<"]}\n";
  std::ofstream file(argv[2]);file<<out.str();file.close();Check(bool(file),"cannot save report");
  std::cout<<"Full scri consistency N="<<n<<" completed\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';status=1;}
 Kokkos::finalize();return status;
}
