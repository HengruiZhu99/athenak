// Bounded CPU manufactured derivative accuracy; no time integration.
// The profiles fix boundary values, not all pole/regularity traces. Consequently
// this tests d_n N consistency, not regularity of a manufactured PDE solution.
#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "z4c/hyperboloidal/hyperbolic_composite.hpp"

namespace {
namespace h=z4c::hyperboloidal;
using Model=h::HyperbolicComposite;
using V=std::vector<Real>;
using Scalar=h::ScalarJet<Real>;
constexpr int fields=18;
constexpr Real pi=3.141592653589793238462643383279502884;
struct NativeJets {
  V v=V(31,0);
  std::vector<std::array<Real,3>> d=std::vector<std::array<Real,3>>(31);
  std::vector<std::array<std::array<Real,3>,3>> dd=
      std::vector<std::array<std::array<Real,3>,3>>(31);
  void Put(int f,const Scalar&a){v[f]=a.value;
    for(int i=0;i<3;++i){d[f][i]=a.d[i];for(int j=0;j<3;++j)dd[f][i][j]=a.dd[i][j];}}
};
Scalar Scale(Scalar a,Real b){a.value*=b;for(int i=0;i<3;++i){a.d[i]*=b;
  for(int j=0;j<3;++j)a.dd[i][j]*=b;}return a;}
Scalar Product(const Scalar&a,const Scalar&b){Scalar c;c.value=a.value*b.value;
  for(int i=0;i<3;++i){c.d[i]=a.d[i]*b.value+a.value*b.d[i];
    for(int j=0;j<3;++j)c.dd[i][j]=a.dd[i][j]*b.value+a.d[i]*b.d[j]+a.d[j]*b.d[i]+a.value*b.dd[i][j];}
  return c;}
Scalar Exp(const Scalar&a){Scalar c;c.value=std::exp(a.value);
  for(int i=0;i<3;++i){c.d[i]=c.value*a.d[i];
    for(int j=0;j<3;++j)c.dd[i][j]=c.value*(a.dd[i][j]+a.d[i]*a.d[j]);}return c;}
NativeJets Profile(const Real*x,bool angular,int power,bool boundary=false){
  Real s=0;for(int i=0;i<3;++i)s+=x[i]*x[i];if(boundary)s=1;
  constexpr Real k=2.3,phase=.37;const Real sn=std::sin(k*s+phase),cs=std::cos(k*s+phase);
  const Real t=1-s,fs=power==1?-sn+t*k*cs:-2*t*sn+t*t*k*cs;
  const Real fss=power==1?-2*k*cs-t*k*k*sn:2*sn-4*t*k*cs-t*t*k*k*sn;
  Scalar f;f.value=(power==1?t:t*t)*sn;
  for(int i=0;i<3;++i){f.d[i]=2*x[i]*fs;
    for(int j=0;j<3;++j)f.dd[i][j]=4*x[i]*x[j]*fss+(i==j?2*fs:0);}
  if(angular){const Real a[3]={1.3,.7,-.9};Real phase2=0;for(int i=0;i<3;++i)phase2+=a[i]*x[i];
    Scalar factor;factor.value=1+.25*std::sin(phase2);
    for(int i=0;i<3;++i){factor.d[i]=.25*a[i]*std::cos(phase2);
      for(int j=0;j<3;++j)factor.dd[i][j]=-.25*a[i]*a[j]*std::sin(phase2);}
    f=Product(f,factor);}
  NativeJets out;auto chi=Scale(f,.0125);chi.value+=1;out.Put(0,chi);
  const auto gx=Exp(Scale(f,.0125)),gy=Exp(Scale(f,-.0125));
  auto dx=gx,dy=gy;dx.value=std::expm1(.0125*f.value);dy.value=std::expm1(-.0125*f.value);
  out.Put(1,dx);out.Put(4,dy);
  out.Put(7,Scale(Product(gx,f),.01));out.Put(10,Scale(Product(gy,f),-.01));
  out.Put(13,Scale(f,.02));out.Put(14,Scale(f,-.007));
  for(int a=0;a<3;++a){Scalar coordinate;coordinate.value=x[a];coordinate.d[a]=1;
    out.Put(15+a,Scale(Product(coordinate,f),.008));}
  // Native alpha-reference, beta+x, B/D/J are exactly zero; Model::Jet adds
  // the analytic Minkowski lapse/shift and their exact Cartesian derivatives.
  return out;
}
std::array<Real,fields> Flatten(const h::Z4cRHS<Real>&u){std::array<Real,fields> a{};
  a[0]=u.chi;a[13]=u.trace;a[14]=u.theta;
  for(int t=0;t<6;++t){const int i=Model::ti[t],j=Model::tj[t];a[1+t]=u.metric[i][j];a[7+t]=u.a[i][j];}
  for(int i=0;i<3;++i)a[15+i]=u.lambda[i];return a;}
Real Weight(int f){return (f==2||f==3||f==5||f==8||f==9||f==11)?2:1;}
std::array<Real,fields> Difference(std::array<Real,fields>a,const std::array<Real,fields>&b){
  for(int i=0;i<fields;++i)a[i]-=b[i];return a;}
h::Z4cRHS<Real> AnalyticPole(const Real*x,bool angular,int power,bool boundary,Real*parity=nullptr){
  const auto fields=Profile(x,angular,power,boundary);const auto u=Model::Jet(x,fields.v,fields.d,fields.dd);
  const auto o=Model::Omega(x,u,boundary);auto parts=h::ConformalRHS(u,o,4.,-.99);
  if(!h::HyperbolicDamping(x,u,o,parts))throw std::runtime_error("invalid analytic pole");
  if(parity){Real beta[3]={-x[0],-x[1],-x[2]};const auto pole=h::TensorPoleSources(u,o,u.alpha.value,beta,4.,-.99,1.,1.,true,true);
    const auto error=Difference(Flatten(parts.pole),Flatten(pole.geometry));
    for(Real e:error)*parity=std::max(*parity,std::abs(e));}
  return parts.pole;
}
h::Z4cRHS<Real> AnalyticDerivative(const Real*x,bool angular,int power){
  const auto fields=Profile(x,angular,power,true);const auto u=Model::Jet(x,fields.v,fields.d,fields.dd);
  const auto o=Model::Omega(x,u,true);const auto pole=h::TensorPoleNormalJet(u,o,x,x,4.,-.99,1.,1.,true,true);
  return h::TensorScriLimit(h::Z4cRHS<Real>{},pole,1.);
}
struct Norm {Real area=0,squared=0,maximum=0;size_t points=0;
  void Add(const std::array<Real,fields>&value,Real mass,int begin=0,int end=fields){
    area+=mass;++points;for(int f=begin;f<end;++f){if(!std::isfinite(value[f]))throw std::runtime_error("nonfinite error");
      squared+=mass*Weight(f)*value[f]*value[f];maximum=std::max(maximum,std::abs(value[f]));}}
  void Write(std::ostream&out)const{out<<"{\"area\":"<<area<<",\"points\":"<<points
    <<",\"l2\":"<<std::sqrt(squared/area)<<",\"linf_component\":"<<maximum<<"}";}
};
constexpr const char*region_names[3]={"all","seam_adjacent","face_interior"};
constexpr const char*method_names[6]={"discrete_AD","sampled_native_numerators","sampled_exact_numerators","sampled_native_minus_exact_samples","full_rate_discrete_AD","full_rate_sampled_native"};
struct Errors {Norm region[3],groups[6];
  void Add(const std::array<Real,fields>&v,Real mass,bool seam){region[0].Add(v,mass);region[seam?1:2].Add(v,mass);
    groups[0].Add(v,mass,0,1);groups[1].Add(v,mass,7,13);groups[2].Add(v,mass,13,14);groups[3].Add(v,mass,14,15);groups[4].Add(v,mass,15,18);groups[5].Add(v,mass,1,7);}
  void Write(std::ostream&out)const{out<<"{\"regions\":{";
    for(int i=0;i<3;++i){if(i)out<<',';out<<'"'<<region_names[i]<<"\":";region[i].Write(out);}out<<"},\"groups\":{";
    const char*names[6]={"chi","A","P","Theta","Lambda","metric"};for(int i=0;i<6;++i){if(i)out<<',';out<<'"'<<names[i]<<"\":";groups[i].Write(out);}out<<"}}";}
};
Real Primitive(Real u,Real v){return std::atan2(u*v,std::sqrt(1+u*u+v*v));}
Real Area(int n,int j,int k){const Real l=-1+2.*j/n,r=-1+2.*(j+1)/n,b=-1+2.*k/n,t=-1+2.*(k+1)/n;
  return Primitive(r,t)-Primitive(l,t)-Primitive(r,b)+Primitive(l,b);}
struct Controls {Real moments=0,diagonal=0,jet_first=0,jet_second=0,jet_mixed=0,pole_parity=0,normal_fd=0;
  Real null_residual=0,determinant=0,trace=0,volume_determinant=0,volume_trace=0,exact_pole=0,exact_regularity=0;};
std::array<Real,6> Weights(Controls&control){
  const Real nodes[6]={-4.5,-3.5,-2.5,-1.5,-.5,0};std::array<Real,6>w{};
  for(int i=0;i<6;++i){cubed_sphere::Jet polynomial(1);
    for(int j=0;j<6;++j)if(i!=j)polynomial=polynomial*(cubed_sphere::Jet::Variable(0,0)-cubed_sphere::Jet(nodes[j]))/cubed_sphere::Jet(nodes[i]-nodes[j]);
    w[i]=polynomial.d[0];}
  for(int p=0;p<=5;++p){Real sum=0;for(int i=0;i<6;++i)sum+=w[i]*std::pow(nodes[i],p);
    control.moments=std::max(control.moments,std::abs(sum-(p==1?1:0)));}
  control.diagonal=std::abs(w[5]-1126./315);return w;
}
void JetControls(bool angular,int power,Controls&control){
  const Real x[3]={.3,-.2,.7},step=1e-4;const auto exact=Profile(x,angular,power);
  for(int axis=0;axis<3;++axis){Real plus[3]={x[0],x[1],x[2]},minus[3]={x[0],x[1],x[2]};plus[axis]+=step;minus[axis]-=step;
    const auto a=Profile(plus,angular,power),b=Profile(minus,angular,power);
    for(int f=0;f<31;++f){control.jet_first=std::max(control.jet_first,std::abs((a.v[f]-b.v[f])/(2*step)-exact.d[f][axis]));
      control.jet_second=std::max(control.jet_second,std::abs((a.v[f]-2*exact.v[f]+b.v[f])/(step*step)-exact.dd[f][axis][axis]));}}
  for(int i=0;i<3;++i)for(int j=i+1;j<3;++j){NativeJets corners[4];int index=0;
    for(int a:{-1,1})for(int b:{-1,1}){Real point[3]={x[0],x[1],x[2]};point[i]+=a*step;point[j]+=b*step;corners[index++]=Profile(point,angular,power);}
    for(int f=0;f<31;++f){const Real mixed=(corners[0].v[f]-corners[1].v[f]-corners[2].v[f]+corners[3].v[f])/(4*step*step);
      control.jet_mixed=std::max(control.jet_mixed,std::abs(mixed-exact.dd[f][i][j]));}}
}
void RunFixture(Model&model,bool angular,int power,std::ostream&out){
  const auto begin=std::chrono::steady_clock::now();const int n=model.grid.n;Controls controls;const auto weights=Weights(controls);JetControls(angular,power,controls);
  for(size_t p=0;p<model.grid.active.size();++p){const auto value=Profile(model.grid.geometry[p].x,angular,power);
    controls.volume_determinant=std::max(controls.volume_determinant,std::abs((1+value.v[1])*(1+value.v[4])-1));
    controls.volume_trace=std::max(controls.volume_trace,std::abs(value.v[7]/(1+value.v[1])+value.v[10]/(1+value.v[4])));
    std::copy(value.v.begin(),value.v.end(),model.state.begin()+31*p);}
  for(size_t p=0;p<model.scri.rows.size();++p){const auto value=Profile(model.scri.rows[p].geometry.x,angular,power,true);
    std::copy(value.v.begin(),value.v.end(),model.scri.values.begin()+31*p);}
  model.BoundaryValues();const auto padded=model.Padded(model.state);
  const auto native_rate=model.BoundaryRates(padded,.0001);Errors errors[6];Norm reference,full_reference;
  for(size_t p=0;p<model.scri.rows.size();++p){const auto&row=model.scri.rows[p];const auto*x=row.geometry.x;
    const Real area=Area(n,row.j,row.k);if(!(area>0))throw std::runtime_error("nonpositive area");
    const bool seam=row.j<2||row.k<2||row.j>=n-2||row.k>=n-2;
    const auto exact=Flatten(AnalyticDerivative(x,angular,power));reference.Add(exact,area);
    const auto sample=model.scri.Sample(p,padded,model.scri.values,31);
    const auto u=model.Jet(x,sample.value,sample.d,sample.dd);const auto omega=model.Omega(x,u,true);
    const auto ad=h::TensorPoleNormalJet(u,omega,x,x,4.,-.99,1.,1.,true,true);
    const auto current=Flatten(h::TensorScriLimit(h::Z4cRHS<Real>{},ad,1.));
    const auto sampled=Flatten(model.SampledPoleNormalDerivative(p,padded));
    std::array<Real,fields> pure{};const auto endpoint=Flatten(AnalyticPole(x,angular,power,true,&controls.pole_parity));
    for(int s=0;s<5;++s){const size_t active=row.block*n*n*n+(row.k*n+row.j)*n+n-5+s;
      const auto pole=Flatten(AnalyticPole(model.grid.geometry[active].x,angular,power,false,&controls.pole_parity));
      for(int f=0;f<fields;++f)pure[f]+=weights[s]*(pole[f]-endpoint[f])/row.radial_spacing;}
    errors[0].Add(Difference(current,exact),area,seam);errors[1].Add(Difference(sampled,exact),area,seam);
    errors[2].Add(Difference(pure,exact),area,seam);errors[3].Add(Difference(sampled,pure),area,seam);
    // Native complete free geometry at this h. Surface values are exactly the
    // Minkowski constants, so the optional angular-Q term vanishes exactly.
    // Keep the actual regular and penalty terms; replace only d_n N for the
    // sampled alternative. No dependent-D or bulk-rate claim is made here.
    const auto exact_fields=Profile(x,angular,power,true);
    const auto exact_u=model.Jet(x,exact_fields.v,exact_fields.d,exact_fields.dd);
    const auto exact_o=model.Omega(x,exact_u,true);auto parts=h::ConformalRHS(exact_u,exact_o,4.,-.99);
    if(!h::HyperbolicDamping(x,exact_u,exact_o,parts))throw std::runtime_error("invalid analytic full rate");
    const auto regular=Flatten(parts.regular);const auto trace=Flatten(h::ScriRegularityTraces(exact_u,exact_o));
    const auto g=h::Geometry(exact_u.metric);Real q=0;for(int i=0;i<3;++i)for(int j=0;j<3;++j)q+=g.inverse[i][j]*x[i]*x[j];
    auto traces=trace;traces[0]=exact_u.chi.value-1/q;traces[13]=exact_u.trace.value+3;traces[14]=exact_u.theta.value;
    std::array<Real,fields> full{},old{},replacement{};
    for(int f=0;f<fields;++f){controls.exact_pole=std::max(controls.exact_pole,std::abs(endpoint[f]));
      controls.exact_regularity=std::max(controls.exact_regularity,std::abs(traces[f]));
      full[f]=regular[f]-exact[f]+(4*endpoint[f]-8*traces[f])/row.radial_spacing;
      old[f]=native_rate[31*p+f];replacement[f]=old[f]+current[f]-sampled[f];}
    full_reference.Add(full,area);errors[4].Add(Difference(old,full),area,seam);errors[5].Add(Difference(replacement,full),area,seam);
    if(p%std::max(size_t(1),model.scri.rows.size()/24)==0){
      constexpr Real step=1e-4;std::array<Real,fields> values[4];const Real shifts[4]={-2,-1,1,2};
      for(int j=0;j<4;++j){Real point[3];for(int a=0;a<3;++a)point[a]=(1+shifts[j]*step)*x[a];values[j]=Flatten(AnalyticPole(point,angular,power,false));}
      for(int f=0;f<fields;++f){const Real finite=(values[0][f]-8*values[1][f]+8*values[2][f]-values[3][f])/(12*step);
        controls.normal_fd=std::max(controls.normal_fd,std::abs(finite-exact[f]));}}
    const auto geometry=h::Geometry(u.metric);Real a_trace=0;for(int i=0;i<3;++i)for(int j=0;j<3;++j)a_trace+=geometry.inverse[i][j]*u.a.k[i][j];
    // Surface values are exactly Minkowski; derivative jets remain manufactured.
    controls.determinant=std::max(controls.determinant,std::abs(geometry.determinant-1));
    controls.trace=std::max(controls.trace,std::abs(a_trace));
    controls.null_residual=std::max(controls.null_residual,model.NullResidual(x,&model.scri.values[31*p]));
  }
  if(controls.moments>1e-12||controls.diagonal>1e-14||controls.jet_first>1e-7||controls.jet_second>1e-6||controls.jet_mixed>1e-6||
     controls.pole_parity>2e-12||controls.normal_fd>2e-8||controls.null_residual>2e-14||
     controls.determinant>2e-14||controls.trace>2e-14||controls.volume_determinant>2e-14||controls.volume_trace>2e-14||std::abs(reference.area-4*pi)>5e-13)
    throw std::runtime_error("manufactured internal oracle check failed");
  if(power==2&&(controls.exact_pole>2e-12||controls.exact_regularity>2e-12))throw std::runtime_error("p2 analytic regularity failed");
  out<<"{\"fixture\":\""<<(angular?"angular":"radial")<<"_p"<<power<<"\",\"vanishing_power\":"<<power
     <<",\"exact_boundary_regular\":"<<(power==2?"true":"false")<<",\"reference\":";reference.Write(out);
  out<<",\"full_rate_reference\":";full_reference.Write(out);
  out<<",\"controls\":{\"weight_moment_error\":"<<controls.moments<<",\"weight_diagonal_error\":"<<controls.diagonal
     <<",\"analytic_first_jet_fd_error\":"<<controls.jet_first<<",\"analytic_second_jet_fd_error\":"<<controls.jet_second
     <<",\"analytic_mixed_hessian_fd_error\":"<<controls.jet_mixed
     <<",\"native_pole_algebra_error\":"<<controls.pole_parity<<",\"analytic_AD_vs_five_point_normal_error\":"<<controls.normal_fd
     <<",\"null_residual\":"<<controls.null_residual<<",\"surface_determinant_error\":"<<controls.determinant<<",\"surface_A_trace\":"<<controls.trace
     <<",\"volume_determinant_error\":"<<controls.volume_determinant<<",\"volume_A_trace\":"<<controls.volume_trace
     <<",\"exact_endpoint_pole_max\":"<<controls.exact_pole<<",\"exact_regularity_max\":"<<controls.exact_regularity<<"},\"errors\":{";
  for(int i=0;i<6;++i){if(i)out<<',';out<<'"'<<method_names[i]<<"\":";errors[i].Write(out);}out<<"},\"seconds\":"
      <<std::chrono::duration<Real>(std::chrono::steady_clock::now()-begin).count()<<"}";
}
}
int main(int argc,char**argv){
  try{if(argc!=3)throw std::invalid_argument("usage: scri_sampled_accuracy n output.json");
    const int n=std::stoi(argv[1]);if(n!=8&&n!=12&&n!=16&&n!=24)throw std::invalid_argument("unsupported bounded n");
    const auto begin=std::chrono::steady_clock::now();
    Model model(cubed_sphere::CompositeMesh(.125,{.45,.75,1},1,n,3),false,true,0);model.workers=1;
    if(std::ifstream(argv[2]).good())throw std::runtime_error("output must be new");std::ostringstream out;out<<std::setprecision(17);
    out<<"{\"status\":\"completed\",\"n\":"<<n<<",\"h\":"<<.25/n
       <<",\"backend\":\"CPU native reference\",\"scope\":\"Manufactured derivative and complete free-geometry surface-rate accuracy only; no evolution\",\"norm\":\"Absolute errors: reference-area normalized Cartesian Frobenius L2; Linf is maximum Cartesian stored component. No division by reference rate.\",\"regions\":\"Global norm primary; seam_adjacent is a shrinking two-cell band on every face, not a fixed physical region\",\"full_rate_contract\":\"Native AD BoundaryRates with only normal contribution replaced for sampled variant; no dependent-D/bulk-rate claim. Q exactly zero on constant manufactured boundary values. p1 target includes h-dependent nonzero SAT; p2 exact N=C=0.\",\"fixtures\":[";
    bool first=true;for(int power:{1,2})for(bool angular:{false,true}){if(!first)out<<',';first=false;RunFixture(model,angular,power,out);}out<<"],\"seconds\":"
        <<std::chrono::duration<Real>(std::chrono::steady_clock::now()-begin).count()<<"}\n";
    std::ofstream file(argv[2]);file<<out.str();file.close();if(!file)throw std::runtime_error("output failure");
    std::cout<<"manufactured derivative accuracy n="<<n<<" completed\n";return 0;
  }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
