// Fixed finite-boundary Taylor data from the original native initial-value problem.
// AD objects are evaluation devices, never independently evolved spatial fields.
#pragma once
#include <array>
#include <vector>
#include <stdexcept>
#include "goal1_native_ad.hpp"
namespace Kokkos {
inline cubed_sphere::Jet sqrt(const cubed_sphere::Jet&x){return cubed_sphere::Power(x,.5);}
inline cubed_sphere::Jet exp(const cubed_sphere::Jet&x){using cubed_sphere::Jet;Jet z(std::exp(x.f));for(int i=0;i<3;++i){z.d[i]=z.f*x.d[i];for(int j=0;j<3;++j)z.h[i][j]=z.f*(x.h[i][j]+x.d[i]*x.d[j]);}return z;}
inline cubed_sphere::Jet abs(const cubed_sphere::Jet&x){return x.f<0?-x:x;}
inline cubed_sphere::Jet pow(const cubed_sphere::Jet&x,const cubed_sphere::Jet&p){return exp(p*cubed_sphere::Log(x));}
inline cubed_sphere::Jet hypot(const cubed_sphere::Jet&a,const cubed_sphere::Jet&b){return sqrt(a*a+b*b);}
}
#include "goal1_typed_support.hpp"
namespace cubed_sphere {namespace native_boundary_history {
namespace h=z4c::hyperboloidal;using J=Jet;
constexpr int ma[6]={0,0,0,1,1,2},mb[6]={0,1,2,1,2,2};
inline bool Config(int n){return n<=6||n>=18;}
template<class T>h::Z4cJet<T> Initial(const h::LayerPoint<T>&p,const T x[3],Real amplitude,Real width){
 auto u=p.state;T r2=0;for(int i=0;i<3;++i)r2+=x[i]*x[i];const T b=1-r2,w2=width*width,e=Kokkos::exp(-r2/w2);
 const T phi=b*b*b*b*e,dr=e*(-4*b*b*b-b*b*b*b/w2),ddr=e*(12*b*b+8*b*b*b/w2+b*b*b*b/(w2*w2));
 T g[4]={1+.2*x[0]+.3*x[1]*x[2],.5*(1+.3*x[1]*x[2]),.1*x[0],.05*x[0]*x[1]},dg[4][3]{},ddg[4][3][3]{};
 dg[0][0]=.2;dg[0][1]=.3*x[2];dg[0][2]=.3*x[1];ddg[0][1][2]=ddg[0][2][1]=.3;
 dg[1][1]=.15*x[2];dg[1][2]=.15*x[1];ddg[1][1][2]=ddg[1][2][1]=.15;
 dg[2][0]=.1;dg[3][0]=.05*x[1];dg[3][1]=.05*x[0];ddg[3][0][1]=ddg[3][1][0]=.05;
 for(int k=0;k<4;++k){h::ScalarJet<T>a{};a.value=amplitude*phi*g[k];for(int i=0;i<3;++i){a.d[i]=amplitude*(2*x[i]*dr*g[k]+phi*dg[k][i]);for(int j=0;j<3;++j)a.dd[i][j]=amplitude*((T(i==j?2:0)*dr+4*x[i]*x[j]*ddr)*g[k]+2*dr*(x[i]*dg[k][j]+x[j]*dg[k][i])+phi*ddg[k][i][j]);}
  if(k==0){u.alpha.value+=a.value;for(int i=0;i<3;++i){u.alpha.d[i]+=a.d[i];for(int j=0;j<3;++j)u.alpha.dd[i][j]+=a.dd[i][j];}}
  else {u.beta.value[k-1]+=a.value;for(int i=0;i<3;++i){u.beta.d[i][k-1]+=a.d[i];for(int j=0;j<3;++j)u.beta.dd[i][j][k-1]+=a.dd[i][j];}}
 }return u;
}
inline J First(const J&v,int i){J z(v.d[i]);for(int j=0;j<3;++j)z.d[j]=v.h[i][j];return z;}
inline h::Z4cJet<J> FromComponents(const std::array<J,22>&f,Real unsupported=0){
 h::Z4cJet<J>u{};auto scalar=[&](const J&v){h::ScalarJet<J>s{};s.value=v;for(int i=0;i<3;++i){s.d[i]=First(v,i);for(int j=0;j<3;++j){s.dd[i][j]=J(v.h[i][j]);for(int k=0;k<3;++k)s.dd[i][j].d[k]=unsupported*(1+i+j+k);}}return s;};
 u.chi=scalar(f[0]);u.trace=scalar(f[7]);u.theta=scalar(f[17]);u.alpha=scalar(f[18]);
 for(int i=0;i<6;++i){int a=ma[i],b=mb[i];auto q=scalar(f[1+i]);u.metric.g[a][b]=u.metric.g[b][a]=q.value;u.a.k[a][b]=u.a.k[b][a]=f[8+i];for(int d=0;d<3;++d){u.metric.dg[d][a][b]=u.metric.dg[d][b][a]=q.d[d];u.a.dk[d][a][b]=u.a.dk[d][b][a]=First(f[8+i],d);for(int e=0;e<3;++e)u.metric.ddg[d][e][a][b]=u.metric.ddg[d][e][b][a]=q.dd[d][e];}}
 for(int i=0;i<3;++i){auto l=scalar(f[14+i]),b=scalar(f[19+i]);u.lambda.value[i]=l.value;u.beta.value[i]=b.value;for(int d=0;d<3;++d){u.lambda.d[d][i]=l.d[d];u.beta.d[d][i]=b.d[d];for(int e=0;e<3;++e){u.lambda.dd[d][e][i]=l.dd[d][e];u.beta.dd[d][e][i]=b.dd[d][e];}}}return u;
}
template<class F>h::LayerPoint<J> TransformReference(const h::LayerPoint<J>&p,F fn){
 h::LayerPoint<J>r{};r.omega=fn(p.omega);r.alpha=fn(p.alpha);r.k_bar=fn(p.k_bar);r.k_physical=fn(p.k_physical);r.hessian_omega=fn(p.hessian_omega);r.hessian_alpha=fn(p.hessian_alpha);r.einstein_nn=fn(p.einstein_nn);r.einstein_stress_diagonal=fn(p.einstein_stress_diagonal);r.radius=fn(p.radius);r.L=fn(p.L);r.b=fn(p.b);r.w_omega=fn(p.w_omega);r.n_residue=fn(p.n_residue);r.outgoing=fn(p.outgoing);r.ingoing=fn(p.ingoing);
 for(int i=0;i<3;++i){r.beta[i]=fn(p.beta[i]);r.domega[i]=fn(p.domega[i]);r.dalpha[i]=fn(p.dalpha[i]);r.einstein_momentum[i]=fn(p.einstein_momentum[i]);for(int j=0;j<3;++j)r.omega_hessian[i][j]=fn(p.omega_hessian[i][j]);}r.state=typed_support::Map<J>(p.state,[&](int,const J&v){return fn(v);});return r;
}
struct Data {
 h::Z4cJet<Real>initial{},first{},second{};
 Real raw_reference_RHS=0,initial_chain_error=0,source_gradient_error=0;
 // second contains only all22 values and11 configuration gradients. No absent
 // higher derivative can enter Boundary::Build/Rows. Never pass it to a PDE RHS.
 h::Z4cJet<Real> ForBoundaryRows(Real time,int order)const{
  if(order<0||order>2||time<0||time>.02000000000001)throw std::runtime_error("finite boundary history scope");std::vector<Real>f,s;typed_support::Map<Real>(first,[&](int,Real v){f.push_back(v);return v;});typed_support::Map<Real>(second,[&](int,Real v){s.push_back(v);return v;});return typed_support::Map<Real>(initial,[&](int i,Real v){return v+(order>=1?time*f[i]:0)+(order>=2?.5*time*time*s[i]:0);});
 }
};
inline Data Build(const Real x[3],Real amplitude,Real width,Real unsupported=0){
 if(!(width>0)||!std::isfinite(amplitude))throw std::runtime_error("finite boundary data parameters");J pos[3];for(int i=0;i<3;++i)pos[i]=J::Variable(x[i],i);h::LayerReference<J>descriptor(J(1),J(.5),{true,.05,.95});auto p=descriptor.At(pos[0],pos[1],pos[2]);auto c=rwm::ReferenceConnection(p,pos);auto u=Initial(p,pos,amplitude,width);auto f=typed_support::Core(u,p,c),ref=typed_support::Core(p.state,p,c);Data out;for(auto&r:ref)out.raw_reference_RHS=std::max(out.raw_reference_RHS,std::abs(r.f));out.initial=typed_support::Map<Real>(u,[](int,const J&v){return v.f;});auto dot=FromComponents(f,unsupported);out.first=typed_support::Map<Real>(dot,[](int,const J&v){return v.f;});std::array<J,22>second{};
 // Each outer two-direction jet differentiates time and one spatial direction.
 // Full actual native source supplies all second time values. Only configuration
 // mixed time/spatial derivatives are exposed: their rows use <=1 spatial order.
 for(int d=0;d<3;++d){std::vector<J>flat;typed_support::Map<J>(dot,[&](int,const J&v){flat.push_back(v);return v;});auto a=typed_support::Map<J>(u,[&](int n,const J&v){J q(v.f);q.d[0]=flat[n].f;q.d[1]=v.d[d];q.h[0][1]=q.h[1][0]=flat[n].d[d];return q;});auto stationary=[&](const J&v){J q(v.f);q.d[1]=v.d[d];return q;};auto p2=TransformReference(p,stationary);rwm::Connection<J>c2{};c2.valid=c.valid;for(int i=0;i<4;++i)for(int j=0;j<3;++j)for(int k=0;k<3;++k)c2.scaled[i][j][k]=stationary(c.scaled[i][j][k]);auto g=typed_support::Core(a,p2,c2);
  for(int n=0;n<22;++n){out.source_gradient_error=std::max(out.source_gradient_error,std::abs(g[n].d[1]-f[n].d[d]));second[n].f=g[n].d[0];if(Config(n))second[n].d[d]=g[n].h[0][1];}
 }
 auto two=FromComponents(second);out.second=typed_support::Map<Real>(two,[](int,const J&v){return v.f;});return out;
}
}}
