// An amplitude dual of a supplied spatial jet for equation analysis only.
// No wave momentum, independently evolved gradient or PDE field is introduced.
#pragma once
#include <vector>
#include "goal1_native_ad.hpp"
namespace cubed_sphere {namespace tensor_endpoint {
struct D {Jet f,a;D()=default;D(double x):f(x){}D(const Jet&x):f(x){}D(const Jet&x,const Jet&y):f(x),a(y){}};
inline D operator+(const D&x,const D&y){return{x.f+y.f,x.a+y.a};}
inline D operator-(const D&x,const D&y){return{x.f-y.f,x.a-y.a};}
inline D operator-(const D&x){return D(0)-x;}
inline D operator*(const D&x,const D&y){return{x.f*y.f,x.a*y.f+x.f*y.a};}
inline D operator/(const D&x,const D&y){return{x.f/y.f,(x.a*y.f-x.f*y.a)/(y.f*y.f)};}
inline D&operator+=(D&x,const D&y){x=x+y;return x;}inline D&operator-=(D&x,const D&y){x=x-y;return x;}inline D&operator*=(D&x,const D&y){x=x*y;return x;}inline D&operator/=(D&x,const D&y){x=x/y;return x;}
inline bool operator>(const D&x,const D&y){return x.f>y.f;}inline bool operator<(const D&x,const D&y){return x.f<y.f;}inline bool operator==(const D&x,const D&y){return x.f==y.f;}
}}
namespace Kokkos {inline bool isfinite(const cubed_sphere::tensor_endpoint::D&x){return cubed_sphere::JetFinite(x.f)&&cubed_sphere::JetFinite(x.a);}}
#include "goal1_native_endpoint.hpp"
namespace cubed_sphere {namespace tensor_endpoint {
namespace h=z4c::hyperboloidal;
struct Parts{std::array<D,22>regular,pole;h::Z4cJet<D>state;D normal;};
inline Parts LinearParts(const h::Z4cJet<Real>&delta,const h::LayerPoint<Real>&p,const Real x[3],Real unused=0){
 auto ref0=native_endpoint::OuterReference(p,x);ref0.omega.f=0;auto ref=typed_support::Constant<D>(p);ref.omega=D(ref0.omega);ref.alpha=D(ref0.alpha);ref.L=D(ref0.L);ref.state=typed_support::Map<D>(ref0.state,[](int,Jet q){return D(q);});for(int i=0;i<3;++i){ref.beta[i]=D(ref0.beta[i]);ref.domega[i]=D(ref0.domega[i]);ref.dalpha[i]=D(ref0.dalpha[i]);for(int j=0;j<3;++j)ref.omega_hessian[i][j]=D(ref0.omega_hessian[i][j]);}
 auto base=native_endpoint::Spatial(p.state,unused),du=native_endpoint::Spatial(delta,unused);std::vector<Jet>flat;typed_support::Map<Jet>(du,[&](int,Jet q){flat.push_back(q);return q;});auto u=typed_support::Map<D>(base,[&](int i,Jet q){return D(q,flat[i]);});
 auto c0=native_endpoint::OuterConnection(x);rwm::Connection<D>c{};c.valid=true;for(int k=0;k<4;++k)for(int i=0;i<3;++i)for(int j=0;j<3;++j)c.scaled[k][i][j]=D(c0.scaled[k][i][j]);h::OmegaJet<D>o{};o.omega=ref.omega;for(int i=0;i<3;++i){o.gradient[i]=ref.domega[i];for(int j=0;j<3;++j)o.hessian[i][j]=ref.omega_hessian[i][j];}h::SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,o);auto f=h::ConformalRHS(u,o,D(10)/u.alpha.value,D(0));auto g=rwm::Gauge(ref,u,c);if(!f.valid||!g.valid)throw std::runtime_error("invalid actual linear endpoint parts");return{native_endpoint::Rows(f.regular,g.regular),native_endpoint::Rows(f.pole,g.pole),u,o.normal};
}
}}
