// Linearized actual native22 Z4c/RWM source, including analytic scri quotients.
// AD amplitude/spatial directions are analysis only; no PDE fields are added.
#include "goal1_native_endpoint.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstdlib>
namespace cs=cubed_sphere;namespace h=z4c::hyperboloidal;
Real Read(){std::string token;if(!(std::cin>>token))throw std::runtime_error("truncated tensor fixture");char*end=nullptr;Real v=std::strtod(token.c_str(),&end);if(!end||*end||!std::isfinite(v))throw std::runtime_error("nonfinite tensor fixture");return v;}
Real Error(Real a,Real b){return std::abs(a-b)/std::max({Real(1),std::abs(a),std::abs(b)});}
cs::Jet Project(const cs::Jet&x,const Real d[3]){cs::Jet q(x.f);for(int i=0;i<3;++i)q.d[1]+=d[i]*x.d[i];return q;}
h::Z4cJet<cs::Jet> Tangent(const h::Z4cJet<Real>&base,const h::Z4cJet<Real>&delta,const Real d[3],Real unused=0){
 auto b=cs::native_endpoint::Spatial(base,unused),a=cs::native_endpoint::Spatial(delta,unused);std::vector<cs::Jet> flat;
 cs::typed_support::Map<cs::Jet>(a,[&](int,cs::Jet x){flat.push_back(x);return x;});
 return cs::typed_support::Map<cs::Jet>(b,[&](int n,cs::Jet x){auto q=Project(x,d);q.d[0]=flat[n].f;for(int i=0;i<3;++i)q.h[0][1]+=d[i]*flat[n].d[i];q.h[1][0]=q.h[0][1];return q;});
}
cs::native_endpoint::Parts EndpointParts(const h::LayerPoint<Real>&p,const h::Z4cJet<Real>&delta,const Real x[3],const Real d[3],Real unused=0){
 auto ref0=cs::native_endpoint::OuterReference(p,x),ref=cs::typed_support::Constant<cs::Jet>(p);
 ref.omega=Project(ref0.omega,d);ref.omega.f=0;ref.alpha=Project(ref0.alpha,d);ref.L=Project(ref0.L,d);
 ref.state=cs::typed_support::Map<cs::Jet>(ref0.state,[&](int,cs::Jet q){return Project(q,d);});
 for(int i=0;i<3;++i){ref.beta[i]=Project(ref0.beta[i],d);ref.domega[i]=Project(ref0.domega[i],d);ref.dalpha[i]=Project(ref0.dalpha[i],d);for(int j=0;j<3;++j)ref.omega_hessian[i][j]=Project(ref0.omega_hessian[i][j],d);}
 auto c0=cs::native_endpoint::OuterConnection(x),c=c0;
 for(int i=0;i<4;++i)for(int j=0;j<3;++j)for(int k=0;k<3;++k)c.scaled[i][j][k]=Project(c0.scaled[i][j][k],d);
 return cs::native_endpoint::NativeParts(Tangent(p.state,delta,d,unused),ref,c);
}
h::Z4cJet<Real> RayOnly(h::Z4cJet<Real>u,const Real n[3]){
 auto scalar=[&](h::ScalarJet<Real>&s){Real d=0,dd=0;for(int i=0;i<3;++i){d+=n[i]*s.d[i];for(int j=0;j<3;++j)dd+=n[i]*n[j]*s.dd[i][j];}for(int i=0;i<3;++i){s.d[i]=n[i]*d;for(int j=0;j<3;++j)s.dd[i][j]=n[i]*n[j]*dd;}};
 scalar(u.chi);scalar(u.alpha);scalar(u.trace);scalar(u.theta);
 for(int i=0;i<3;++i){h::ScalarJet<Real>b{},l{};for(int d=0;d<3;++d){b.d[d]=u.beta.d[d][i];l.d[d]=u.lambda.d[d][i];for(int e=0;e<3;++e)b.dd[d][e]=u.beta.dd[d][e][i];}scalar(b);scalar(l);for(int d=0;d<3;++d){u.beta.d[d][i]=b.d[d];u.lambda.d[d][i]=l.d[d];for(int e=0;e<3;++e)u.beta.dd[d][e][i]=b.dd[d][e];}
 for(int j=0;j<3;++j){h::ScalarJet<Real>g{},a{};for(int d=0;d<3;++d){g.d[d]=u.metric.dg[d][i][j];a.d[d]=u.a.dk[d][i][j];for(int e=0;e<3;++e)g.dd[d][e]=u.metric.ddg[d][e][i][j];}scalar(g);scalar(a);for(int d=0;d<3;++d){u.metric.dg[d][i][j]=g.d[d];u.a.dk[d][i][j]=a.d[d];for(int e=0;e<3;++e)u.metric.ddg[d][e][i][j]=g.dd[d][e];}}}return u;}
int main(int argc,char**argv){Kokkos::initialize(argc,argv);std::cout<<std::setprecision(17);int status=0;try{while(std::cin.peek()!=EOF){std::cin>>std::ws;if(std::cin.peek()==EOF)break;int id=Read();Real time=Read(),x[3];for(auto&v:x)v=Read();h::Z4cJet<Real>empty{};auto du=cs::typed_support::Map<Real>(empty,[](int,Real){return Read();});Real target[22];for(auto&v:target)v=Read();h::LayerReference<Real>descriptor(1.,.5,{true,.05,.95});auto p=descriptor.At(x[0],x[1],x[2]);bool endpoint=std::abs(p.omega)<1e-14;Real actual[22]{},pole=0,tangent=0,negative=0,unused=0,H=0,M=0,Z=0,det=0,trace=0,scale=1;
 cs::typed_support::Map<Real>(du,[&](int,Real v){scale=std::max(scale,std::abs(v));return v;});
 if(endpoint){
  Real gradient[3][22]{},regular[22]{};Real norm=0;
  for(int d=0;d<3;++d){Real direction[3]{};direction[d]=1;auto f=EndpointParts(p,du,x,direction);if(!f.valid)throw std::runtime_error("invalid endpoint tensor parts");norm+=p.domega[d]*p.domega[d];for(int n=0;n<22;++n){regular[n]=f.regular[n].d[0];gradient[d][n]=f.pole[n].h[0][1];pole=std::max(pole,std::abs(f.pole[n].d[0]));}}
  for(int n=0;n<22;++n){Real quotient=0;for(int d=0;d<3;++d)quotient+=p.domega[d]*gradient[d][n]/norm;actual[n]=regular[n]+quotient;for(int d=0;d<3;++d)tangent=std::max(tangent,std::abs(gradient[d][n]-p.domega[d]*quotient));}
  auto bad=du;for(int i=0;i<3;++i)for(int j=0;j<3;++j)bad.a.k[i][j]=0;Real direction[3]={1,0,0};auto omitted=EndpointParts(p,bad,x,direction),good=EndpointParts(p,du,x,direction);auto altered=EndpointParts(p,du,x,direction,.17);for(int n=0;n<22;++n)unused=std::max(unused,std::abs(altered.regular[n].d[0]-good.regular[n].d[0])+std::abs(altered.pole[n].h[0][1]-good.pole[n].h[0][1]));for(int n=0;n<22;++n)negative=std::max(negative,std::abs(omitted.regular[n].d[0]-good.regular[n].d[0])+std::abs(omitted.pole[n].d[0]-good.pole[n].d[0]));
 }else{
  Real direction[3]{};auto state=Tangent(p.state,du,direction);auto ref=cs::typed_support::Constant<cs::Jet>(p);auto c=cs::typed_support::Constant<cs::Jet>(rwm::ReferenceConnection(p,x));auto f=cs::typed_support::Core(state,ref,c);for(int n=0;n<22;++n)actual[n]=f[n].d[0];
  auto bad=state;for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int d=0;d<3;++d)for(int e=0;e<3;++e)bad.metric.ddg[d][e][i][j].d[0]=0;auto omitted=cs::typed_support::Core(bad,ref,c);for(int n=0;n<22;++n)negative=std::max(negative,std::abs(omitted[n].d[0]-actual[n]));
  auto test=state;for(int i=0;i<3;++i)for(int d=0;d<3;++d)for(int e=0;e<3;++e)test.lambda.dd[d][e][i].d[0]=.123*(1+i+d+e);auto altered=cs::typed_support::Core(test,ref,c);for(int n=0;n<22;++n)unused=std::max(unused,std::abs(altered[n].d[0]-actual[n]));
 }
 {Real direction[3]{};auto state=Tangent(p.state,du,direction);auto ref=cs::typed_support::Constant<cs::Jet>(p);if(endpoint)ref.omega=cs::Jet(0);
  h::OmegaJet<cs::Jet> omega{};omega.omega=ref.omega;for(int i=0;i<3;++i){omega.gradient[i]=ref.domega[i];for(int j=0;j<3;++j)omega.hessian[i][j]=ref.omega_hessian[i][j];}h::SetStationaryOmegaNormal(state.alpha.value,state.beta.value,state.alpha.d,state.beta.d,omega);auto constraints=h::EvolvedConstraints(state,omega);H=std::abs(constraints.hamiltonian.d[0]);for(int i=0;i<3;++i){M=std::max(M,std::abs(constraints.momentum[i].d[0]));Z=std::max(Z,std::abs(constraints.z4.z_covector[i].d[0]));}
  auto geometry=h::Geometry(state.metric);det=std::abs(geometry.determinant.d[0]);cs::Jet tr;for(int i=0;i<3;++i)for(int j=0;j<3;++j)tr+=geometry.inverse[i][j]*state.a.k[i][j];trace=std::abs(tr.d[0]);
 }
 Real transverse_negative=0,radius=std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
 if(radius>0){Real n[3];for(int d=0;d<3;++d)n[d]=x[d]/radius;auto ray=RayOnly(du,n);
 if(endpoint){Real direction[3]={1,0,0};auto f=EndpointParts(p,ray,x,direction),good=EndpointParts(p,du,x,direction);for(int k=0;k<22;++k)transverse_negative=std::max(transverse_negative,std::abs(f.regular[k].d[0]-good.regular[k].d[0])+std::abs(f.pole[k].h[0][1]-good.pole[k].h[0][1]));}
 else{Real direction[3]{};auto f=cs::typed_support::Core(Tangent(p.state,ray,direction),cs::typed_support::Constant<cs::Jet>(p),cs::typed_support::Constant<cs::Jet>(rwm::ReferenceConnection(p,x)));for(int k=0;k<22;++k)transverse_negative=std::max(transverse_negative,std::abs(f[k].d[0]-actual[k]));}}
 Real error=0;int worst=0;for(int n=0;n<22;++n){Real e=Error(actual[n],target[n]);if(e>error){error=e;worst=n;}}
 std::cout<<"{\"id\":"<<id<<",\"time\":"<<time<<",\"Omega\":"<<p.omega<<",\"endpoint\":"<<(endpoint?"true":"false")<<",\"scaled_source_error\":"<<error<<",\"worst_native_row\":"<<worst<<",\"pole_linear\":"<<pole<<",\"pole_tangent\":"<<tangent<<",\"H_linear\":"<<H<<",\"M_conformal_linear_max\":"<<M<<",\"Z_linear_max\":"<<Z<<",\"det_linear\":"<<det<<",\"Atrace_linear\":"<<trace<<",\"native_input_jet_max\":"<<scale<<",\"omitted_metric_Hessian_or_endpoint_A_negative\":"<<negative<<",\"unused_lambda_DD_change\":"<<unused<<",\"omitted_tangential_derivatives_negative\":"<<transverse_negative<<",\"actual_rhs\":[";for(int n=0;n<22;++n)std::cout<<(n?",":"")<<actual[n];std::cout<<"],\"target_rhs\":[";for(int n=0;n<22;++n)std::cout<<(n?",":"")<<target[n];std::cout<<"]}"<<std::endl;
 }}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;status=2;}Kokkos::finalize();return status;}
