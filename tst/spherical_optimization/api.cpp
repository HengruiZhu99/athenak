#include "athena.hpp"
#include "z4c/hyperboloidal/central_z4c_configuration.hpp"
#include "z4c/hyperboloidal/current_spherical.hpp"
#include "z4c/hyperboloidal/stationary_gauge_reference.hpp"
#include <memory>
#include <string>
namespace h=z4c::hyperboloidal;
namespace s=h::current_spherical;
namespace {
thread_local std::string error;
struct Lab {
 h::HyperbolicShiftParameters<double> p;
 std::unique_ptr<h::NativeWormhole> native;
 std::unique_ptr<h::StationaryGaugeReference> reference;
 Lab(int mode,double radius,double slicing_ratio,int separated,double compact_end=0) {
  if(mode==0 && compact_end!=0)throw std::invalid_argument("compact slicing requires the central configuration mode");
  if(mode==0){ // Historical current production formulation, explicit baseline.
   p.profile={true,.4,.54,.42};p.inner_begin=.62;p.inner_end=.78;
   p.reference_start_M=20;p.reference_end_M=40;p.stationary_outer_reference=true;
  }else if(mode==1){
   h::CentralZ4cConfiguration c;c.inner_z4c_radius_M=radius;c.slicing_end_ratio=slicing_ratio;c.compact_slicing_end=compact_end;
   p=c.Resolve().gauge;
  }else throw std::invalid_argument("unknown formulation mode");
  p.separated_shift_speeds=separated;
  native=std::make_unique<h::NativeWormhole>(1./3.,2e-13,p.reference_start_M,p.reference_end_M,
                       h::NativeWormhole::TransitionProfile::polynomial_c6,p.reference_compact_end);
  p.inner_gamma=std::pow(native->IsotropicScale(),2);p.Validate();
  reference=std::make_unique<h::StationaryGaugeReference>(p);
 }
};
}
extern "C" const char* spherical_error(){return error.c_str();}
extern "C" void* spherical_create(int mode,double radius,double ratio,int separated){
 try{return new Lab(mode,radius,ratio,separated);}catch(const std::exception&e){error=e.what();return nullptr;}}
extern "C" void* spherical_create_extended(int mode,double radius,double ratio,int separated,double compact_end){
 try{return new Lab(mode,radius,ratio,separated,compact_end);}catch(const std::exception&e){error=e.what();return nullptr;}}
extern "C" int spherical_compact_configuration(void*v,double*out){
 auto&n=*static_cast<Lab*>(v)->native;
 const double a[]={n.CompactEnd(),n.CompactShape(),n.TransitionEndAreal()/n.Mass(),n.CompactJoinResidual(),n.Tolerance()};
 std::copy(a,a+5,out);return 0;}
extern "C" void spherical_destroy(void*v){delete static_cast<Lab*>(v);}
extern "C" int spherical_configuration(void*v,double*out){
 auto&l=*static_cast<Lab*>(v);const auto&p=l.p;
 const double a[]={p.reference_start_M,p.reference_end_M,l.native->InnerRadius(),l.native->IsotropicScale(),
 p.profile.flat_radius,p.profile.outer_radius,p.profile.plateau,p.inner_begin,p.inner_end,
 p.restoration_begin,p.restoration_end,p.inner_gamma,l.native->OuterTransitionRadius()};
 std::copy(a,a+13,out);return 0;
}
extern "C" int spherical_initial(void*v,int n,const double*r,double*q,double*sources,double*omega){
 try{auto&l=*static_cast<Lab*>(v);
  for(int i=0;i<n;++i){const double x[3]={r[i],0,0};auto f=l.native->Fields(x);
   l.p.profile.TransformFields(r[i],f);const auto ref=l.reference->At(r[i]);
   const auto om=l.p.profile.Radial(r[i]);omega[3*i]=om.value;omega[3*i+1]=om.first;omega[3*i+2]=om.second;
   const double weight=h::Smoothstep9((r[i]-l.p.inner_begin)/(l.p.inner_end-l.p.inner_begin));
   f[22]=(1-weight)*r[i];
   if(r[i]>l.p.profile.flat_radius){
    const double blend=h::Smoothstep9((r[i]-l.p.profile.flat_radius)/(l.p.inner_begin-l.p.profile.flat_radius));
    const double A=(1+r[i]*r[i])/2;f[18]=(1-blend)*(f[18]+A)+blend*ref.alpha-A;f[22]=ref.shift_ratio*r[i];
   }
   s::State value={f[0],1,1,f[7],f[10],f[13]-3,0,0,f[18]+(1+r[i]*r[i])/2,f[22]-r[i],0,0};
   std::copy(value.begin(),value.end(),q+s::count*i);
   const double src[]={ref.alpha,ref.shift_ratio,ref.lapse_offset,ref.lapse_linear,ref.lapse_quadratic,ref.shift_offset};
   std::copy(src,src+6,sources+6*i);
  }return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int spherical_outer_reference_jets(void*v,int n,const double*r,double*d,double*dd){
 try{auto&l=*static_cast<Lab*>(v);
  for(int i=0;i<n;++i){
   if(r[i]<l.p.profile.outer_radius)throw std::invalid_argument("outer reference jet needs native conformal frame");
   const auto j=l.native->OuterRadialDerivatives(r[i]);
   s::State first{},second{};
   first[s::chi]=j.first[0];second[s::chi]=j.second[0];
   first[s::alpha]=j.first[3];second[s::alpha]=j.second[3];
   first[s::beta]=-1;
   std::copy(first.begin(),first.end(),d+12*i);std::copy(second.begin(),second.end(),dd+12*i);
  }return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int spherical_current_parts(void*v,int n,const double*r,const double*q,const double*d,const double*dd,
 const double*sources,double*regular,double*pole,double*traces,double*diagnostics){
 try{auto&l=*static_cast<Lab*>(v);
  for(int i=0;i<n;++i){const auto j=s::Lift(r[i],q+s::count*i,d+s::count*i,dd+s::count*i);const double x[3]={r[i],0,0};
   const auto*z=sources+6*i;h::StationaryGaugeSource<double> src{z[0],z[1],z[2],z[3],z[4],z[5]};
   const auto out=s::Evaluate(x,j,l.p,src);
   std::copy(out.regular.begin(),out.regular.end(),regular+s::count*i);
   std::copy(out.pole.begin(),out.pole.end(),pole+s::count*i);
   std::copy(out.trace.begin(),out.trace.end(),traces+s::count*i);
   if(diagnostics){const auto c=h::EvolvedConstraints(j.u,out.omega);const auto coeff=h::ShiftCoefficients(r[i],j.u.alpha.value,j.u.chi.value,l.p);
    const auto g=h::Geometry(j.u.metric);const double Q=g.inverse[0][0];
    double speed_chi=j.u.chi.value;
    if(l.p.cartesian_chi_guard && out.omega.omega>0)
      speed_chi+=coeff.inner*(std::max(speed_chi,l.p.inner_chi_guard/(out.omega.omega*out.omega.omega))-speed_chi);
    auto*z=diagnostics+12*i;
    z[0]=c.hamiltonian;z[1]=c.momentum[0];z[2]=j.u.theta.value;z[3]=c.z4.z_covector[0];
    z[4]=c.z4.determinant_residual;z[5]=c.z4.tracefree_residual;z[6]=c.null_residual;
    z[7]=std::abs(j.u.beta.value[0])+std::sqrt(std::max({j.u.alpha.value*j.u.alpha.value*speed_chi*Q,
     h::GaugeLapseSpeedSquared(j.u.alpha.value,speed_chi,coeff,Q,1.,l.p),h::GaugeShiftSpeedSquared(coeff,Q)}));
    z[8]=z[9]=std::numeric_limits<double>::quiet_NaN();
    if(r[i]<1){const auto sphere=h::SphereDiagnostics(r[i],j.u,out.omega);z[8]=sphere.mass;z[9]=sphere.expansion_out;}
    z[10]=j.u.alpha.value;z[11]=out.omega.omega>0?j.u.alpha.value/out.omega.omega:std::numeric_limits<double>::infinity();
   }
  }return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int spherical_scri(void*v,const double*q,const double*d,const double*dd,const double*normal,
 double spacing,double penalty,double*rate,double*dependent){
 try{auto&l=*static_cast<Lab*>(v);const auto j=s::Lift(1.,q,d,dd);const double x[3]={1,0,0};
  const auto parts=s::Evaluate(x,j,l.p,l.reference->At(1.));s::State normal_pole;std::copy(normal,normal+12,normal_pole.begin());
  const auto out=s::ScriRate(j,parts,normal_pole,spacing,penalty,l.p,dependent[0],dependent[1]);
  std::copy(out.begin(),out.end(),rate);return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}
// Full 3D first-order pseudodifferential symbol, including transverse sectors.
// State: dchi, five dg, five A, Khat, Theta, three Lambda, dalpha,
// three dbeta, three B, three D. Algebraic-constraint tangent basis.
// Original conformal variables are used; no singular endpoint normalization.
extern "C" int spherical_symbol(void*v,double r,const double*q,const double*n,double*P){
 try{auto&l=*static_cast<Lab*>(v);if(!(r>0&&r<1))throw std::invalid_argument("symbol is an interior chart");
  h::Z4cJet<double>base{};base.alpha.value=q[s::alpha];base.chi.value=q[s::chi];
  base.metric.g[0][0]=q[s::gr];base.metric.g[1][1]=base.metric.g[2][2]=q[s::gt];
  const auto g=h::Geometry(base.metric);if(!g.valid)throw std::invalid_argument("invalid symbol metric");
  const int ii[5]={0,0,0,1,1},jj[5]={0,1,2,1,2};
  auto basis=[&](int f,double b[3][3]){b[ii[f]][jj[f]]=b[jj[f]][ii[f]]=1;double tr=0;for(int i=0;i<3;++i)for(int j=0;j<3;++j)tr+=g.inverse[i][j]*b[i][j];b[2][2]-=tr/g.inverse[2][2];};
  h::OmegaJet<double>o{};o.omega=l.p.profile.Radial(r).value;const double x[3]={r,0,0},eps=1e-4;
  std::fill(P,P+26*26,0.);
  for(int col=0;col<26;++col)for(int sign:{-1,1}){double e=sign*eps;auto u=base;h::VectorJet<double>B{};double D[3]{};
   if(col>=6&&col<11){double b[3][3]{};basis(col-6,b);for(int i=0;i<3;++i)for(int j=0;j<3;++j)u.a.k[i][j]=e*b[i][j];}
   if(col==11)u.trace.value=e;if(col==12)u.theta.value=e;
   if(col>=13&&col<16)u.lambda.value[col-13]=e;
   if(col==16)for(int j=0;j<3;++j)u.alpha.d[j]=e*n[j];
   if(col>=17&&col<20)for(int j=0;j<3;++j)u.beta.d[j][col-17]=e*n[j];
   if(col>=20&&col<23)for(int j=0;j<3;++j)B.d[j][col-20]=e*n[j];
   if(col>=23)D[col-23]=e;
   auto parts=h::ConformalRHS(u,o,l.p.kappa1,l.p.kappa2);h::Z4cRHS<double>R{};
   if(!h::HyperbolicDamping(x,u,o,parts,l.p)||!h::AssembleInterior(parts,o.omega,R))throw std::runtime_error("symbol geometric upper failure");
   const auto gauge=h::HyperbolicGauge(x,u,B,D,l.p);double upper[26]{};
   upper[0]=R.chi;for(int a=0;a<5;++a)upper[1+a]=R.metric[ii[a]][jj[a]];
   if(col==11||col==12||(col>=20&&col<23))upper[16]=gauge.lapse_regular+gauge.lapse_pole/o.omega;
   for(int i=0;i<3;++i){if((col>=13&&col<16)||col>=23)upper[17+i]=gauge.shift_regular[i]+gauge.shift_pole[i]/o.omega;if(col==16)upper[20+i]=gauge.flux_rate[i];}
   u=base;
   if(col==0||col==16)for(int j=0;j<3;++j)for(int k=0;k<3;++k)(col==0?u.chi.dd[j][k]:u.alpha.dd[j][k])=e*n[j]*n[k];
   if(col>=1&&col<6){double b[3][3]{};basis(col-1,b);for(int j=0;j<3;++j)for(int k=0;k<3;++k)for(int a=0;a<3;++a)for(int bidx=0;bidx<3;++bidx)u.metric.ddg[j][k][a][bidx]=e*n[j]*n[k]*b[a][bidx];}
   if(col==11||col==12)for(int j=0;j<3;++j)(col==11?u.trace.d[j]:u.theta.d[j])=e*n[j];
   if(col>=13&&col<16)for(int j=0;j<3;++j)u.lambda.d[j][col-13]=e*n[j];
   if(col>=17&&col<20)for(int j=0;j<3;++j)for(int k=0;k<3;++k)u.beta.dd[j][k][col-17]=e*n[j]*n[k];
   parts=h::ConformalRHS(u,o,l.p.kappa1,l.p.kappa2);
   if(!h::HyperbolicDamping(x,u,o,parts,l.p)||!h::AssembleInterior(parts,o.omega,R))throw std::runtime_error("symbol geometric lower failure");
   for(int a=0;a<5;++a)upper[6+a]=R.a[ii[a]][jj[a]];
   upper[11]=R.trace;upper[12]=R.theta;
   for(int i=0;i<3;++i)upper[13+i]=upper[23+i]=R.lambda[i];
   for(int row=0;row<26;++row)P[26*row+col]+=sign*upper[row]/(2*eps);
  }
  // All evolved fields use the same component advection in this mode.
  for(int i=0;i<26;++i)P[26*i+i]+=q[s::beta]*n[0];
  return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int spherical_auxiliary(void*v,int n,const double*r,const double*q,const double*d,const double*lambda_rate,double*out){
 try{auto&l=*static_cast<Lab*>(v);s::State dd{};
  for(int i=0;i<n;++i){const auto j=s::Lift(r[i],q+12*i,d+12*i,dd.data());double input[3]={lambda_rate[i],0,0},rate[3];
   if(!h::HyperbolicAuxiliaryRate(j.u,j.D,input,rate,l.p.damping))throw std::runtime_error("nonfinite auxiliary");out[i]=rate[0];}
  return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int spherical_gauge_speeds(void*v,double r,const double*q,double*out){
 try{auto&l=*static_cast<Lab*>(v);const auto&p=l.p;double chi=q[s::chi];
  const auto c=h::ShiftCoefficients(r,q[s::alpha],chi,p);const double o=p.profile.Radial(r).value;
  if(p.cartesian_chi_guard && o>0)chi+=c.inner*(std::max(chi,p.inner_chi_guard/(o*o))-chi);
  out[0]=h::GaugeLapseSpeedSquared(q[s::alpha],chi,c,1/q[s::gr],1.,p);
  out[1]=h::GaugeShiftSpeedSquared(c,1/q[s::gr]);out[2]=q[s::alpha]*q[s::alpha]*chi/q[s::gr];out[3]=c.inner;return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}
// Unconstrained radial principal symbol of the twelve-field evolution, including
// determinant and trace-A sectors omitted from the 26-variable tangent symbol.
extern "C" int spherical_free_radial_symbol(void*v,double r,const double*q,double*P){
 try{auto&l=*static_cast<Lab*>(v);if(!(r>0&&r<1))throw std::invalid_argument("interior symbol only");
  h::Z4cJet<double> base{};base.chi.value=q[s::chi];base.alpha.value=q[s::alpha];
  base.metric.g[0][0]=q[s::gr];base.metric.g[1][1]=base.metric.g[2][2]=q[s::gt];
  h::OmegaJet<double>o{};o.omega=l.p.profile.Radial(r).value;const double x[3]={r,0,0},eps=1e-4;
  std::fill(P,P+144,0.);
  auto geometry=[&](const h::Z4cJet<double>&u){auto parts=h::ConformalRHS(u,o,l.p.kappa1,l.p.kappa2);h::Z4cRHS<double>rate{};
   if(!h::HyperbolicDamping(x,u,o,parts,l.p)||!h::AssembleInterior(parts,o.omega,rate))throw std::runtime_error("invalid free symbol");return s::Pack(rate);};
  for(int col=0;col<12;++col)for(int sign:{-1,1}){const double e=sign*eps;auto u=base;h::VectorJet<double>B{};double D[3]{};
   if(col==s::ar)u.a.k[0][0]=e;if(col==s::at)u.a.k[1][1]=u.a.k[2][2]=e;
   if(col==s::khat)u.trace.value=e;if(col==s::theta)u.theta.value=e;
   if(col==s::lambda)u.lambda.value[0]=e;if(col==s::beta)u.beta.d[0][0]=e;
   if(col==s::flux)B.d[0][0]=e;if(col==s::driver)D[0]=e;if(col==s::alpha)u.alpha.d[0]=e;
   auto upper=geometry(u);auto gauge=h::HyperbolicGauge(x,u,B,D,l.p);
   s::State value{};for(int f:{s::chi,s::gr,s::gt})value[f]=upper[f];
   if(col==s::khat||col==s::theta||col==s::flux)value[s::alpha]=gauge.lapse_regular+gauge.lapse_pole/o.omega;
   if(col==s::lambda||col==s::driver)value[s::beta]=gauge.shift_regular[0]+gauge.shift_pole[0]/o.omega;
   if(col==s::alpha)value[s::flux]=gauge.flux_rate[0];
   u=base;
   if(col==s::chi)u.chi.dd[0][0]=e;if(col==s::alpha)u.alpha.dd[0][0]=e;
   if(col==s::gr)u.metric.ddg[0][0][0][0]=e;
   if(col==s::gt)u.metric.ddg[0][0][1][1]=u.metric.ddg[0][0][2][2]=e;
   if(col==s::khat)u.trace.d[0]=e;if(col==s::theta)u.theta.d[0]=e;
   if(col==s::lambda)u.lambda.d[0][0]=e;if(col==s::beta)u.beta.dd[0][0][0]=e;
   auto lower=geometry(u);for(int f:{s::ar,s::at,s::khat,s::theta,s::lambda})value[f]=lower[f];
   value[s::driver]=lower[s::lambda];
   for(int row=0;row<12;++row)P[12*row+col]+=sign*value[row]/(2*eps);
  }
  for(int i=0;i<12;++i)P[12*i+i]+=q[s::beta];return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int spherical_expand_chart(int n,const double*q,const double*d,const double*dd,double*Q,double*D,double*DD){
 try{for(int i=0;i<n;++i){s::State a{},b{},c{};s::ExpandChart(q+10*i,d+10*i,dd+10*i,a,b,c);
  std::copy(a.begin(),a.end(),Q+12*i);std::copy(b.begin(),b.end(),D+12*i);std::copy(c.begin(),c.end(),DD+12*i);}return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int spherical_analytic_normal(void*v,const double*q,const double*d,const double*dd,double*out){
 try{auto&l=*static_cast<Lab*>(v);const auto j=s::Lift(1.,q,d,dd);const double x[3]={1,0,0};
  auto o=l.p.profile.At(x);h::SetStationaryOmegaNormal(j.u.alpha.value,j.u.beta.value,j.u.alpha.d,j.u.beta.d,o);
  const auto dual=h::TensorPoleNormalJet(j.u,o,x,x,l.p.kappa1,l.p.kappa2,1.,1.,true,true);
  const auto normal=s::Pack(h::TensorScriLimit(h::Z4cRHS<double>{},dual,1.));
  std::copy(normal.begin(),normal.end(),out);out[s::alpha]=dual.lapse.derivative;out[s::driver]=normal[s::lambda];return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}
