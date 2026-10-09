#ifndef SCRATCH_DETACHED_HEIGHT_HPP_
#define SCRATCH_DETACHED_HEIGHT_HPP_
#include "z4c/hyperboloidal/layer_reference.hpp"
namespace z4c { namespace hyperboloidal {
template<typename T> KOKKOS_INLINE_FUNCTION
T DetachedScaledExp(T logbase,T coefficient) {
 if(coefficient==0)return 0;
 T v=Kokkos::exp(logbase+Kokkos::log(Kokkos::abs(coefficient)));
 return coefficient>0?v:-v;
}
// Counterfactual Minkowski geometry with a separate height cutoff; used only
// to construct BH initial jets. The actual Minkowski gauge/RHS reference stays
// untouched. This helper must never replace CartesianPatch.reference.
template<typename T> struct DetachedHeightGeometry:LayerReference<T> {
 T boost_power=T(1);
 LayerParameters height;
 DetachedHeightGeometry(const LayerReference<T>&reference,LayerParameters cutoff)
     :LayerReference<T>(reference),height(cutoff) {
  reference.Validate();height.Validate(reference.scri_radius,reference.curvature_radius);
  if(!reference.layer.enabled||!height.enabled)throw std::invalid_argument("detached height requires enabled compactification and height");
 }
 KOKKOS_INLINE_FUNCTION LayerPoint<T> At(T x,T y,T z) const {
  auto p=LayerReference<T>::At(x,y,z);
  const T r=p.radius,a=this->curvature_radius,s=this->scri_radius;
  if((r<=this->layer.r0&&r<=height.r0)||(r>=this->layer.r1&&r>=height.r1))return p;
  const auto w=SmoothCutoff(r,T(this->layer.r0),T(this->layer.r1));
  const auto hw=SmoothCutoff(r,T(height.r0),T(height.r1));
  const T outer=(s-r)*(s+r)/(2*a*s),op=-r/(a*s),opp=-1/(a*s);
  const Radial2<T> o{p.omega,w.d*(outer-1)+w.value*op,
    w.dd*(outer-1)+2*w.d*op+w.value*opp};
  const T o3=w.ddd*(outer-1)+3*w.dd*op+3*w.d*opp;
  const Radial2<T> ell{p.L,-r*o.dd,-o.dd-r*o3};
  const T width=T(height.r1-height.r0);
  Radial2<T> f{};
  if(r<=height.r0) { f={0,0,0}; }
  else if(r>=height.r1) { f={1,0,0}; }
  else {
   const T v=(r-T(height.r0))/width,t=(T(height.r1)-r)/width,g=-1/v+1/t;
   if(g<=0) {
   const T e=Kokkos::exp(g),inv=1/(1+e);
   const T g1=(1/(v*v)+1/(t*t))/width,
     g2=(-2/(v*v*v)+2/(t*t*t))/(width*width);
   const T logf=boost_power*(g-Kokkos::log1p(e));
   const T lf1=boost_power*g1*inv;
   const T lf2=boost_power*(g2*inv-e*g1*g1*inv*inv);
   f={Kokkos::exp(logf),DetachedScaledExp(logf,lf1),DetachedScaledExp(logf,lf1*lf1+lf2)};
  } else {
   f.value=Kokkos::pow(hw.value,boost_power);
   f.d=boost_power*Kokkos::pow(hw.value,boost_power-1)*hw.d;
   f.dd=boost_power*Kokkos::pow(hw.value,boost_power-1)*hw.dd
      +boost_power*(boost_power-1)*Kokkos::pow(hw.value,boost_power-2)*hw.d*hw.d;
  }
  }
  const Radial2<T> radius{r,1,0},ia{1/a,0,0},minus{-1,0,0};
  const auto b=radius*f*ia;
  const T av=Kokkos::hypot(o.value,b.value),ad=(o.value*o.d+b.value*b.d)/av;
  const Radial2<T> alpha{av,ad,
    (o.d*o.d+b.d*b.d+o.value*o.dd+b.value*b.dd-ad*ad)/av};
  const auto chi=Power(alpha/ell,T(2)/3);
  const auto radial_metric=chi*ell*ell/(alpha*alpha);
  const auto beta=minus*b*alpha/ell;
  const T kr=-b.d/ell.value,kt=-f.value/(a*ell.value),kb=kr+2*kt;
  const T diff=-r*f.d/(a*ell.value);
  const T ddiff=-(f.d+r*f.dd)/(a*ell.value)-diff*ell.d/ell.value;
  const Radial2<T> difference{diff,ddiff,0};
  const auto ar=Radial2<T>{T(2)/3,0,0}*radial_metric*difference;
  const auto at=Radial2<T>{-T(1)/3,0,0}*chi*difference;
  const T kp=-(3*f.value+r*o.value*f.d/ell.value)/a;
  const T dkp=-(3*f.d+((o.value+r*o.d)*f.d+r*o.value*f.dd)/ell.value
    -r*o.value*f.d*ell.d/(ell.value*ell.value))/a;
  Z4cJet<T> axis{};
  axis.alpha={alpha.value,{alpha.d,0,0},{{alpha.dd,0,0},{0,0,0},{0,0,0}}};
  axis.chi={chi.value,{chi.d,0,0},{{chi.dd,0,0},{0,0,0},{0,0,0}}};
  axis.trace.value=kp;axis.trace.d[0]=dkp;
  axis.beta.value[0]=beta.value;axis.beta.d[0][0]=beta.d;axis.beta.dd[0][0][0]=beta.dd;
  for(int i=0;i<3;++i) {
   const auto m=i==0?radial_metric:chi,k=i==0?ar:at;
   axis.metric.g[i][i]=m.value;axis.metric.dg[0][i][i]=m.d;axis.metric.ddg[0][0][i][i]=m.dd;
   axis.a.k[i][i]=k.value;axis.a.dk[0][i][i]=k.d;
  }
  const T n[3]={x/r,y/r,z/r};
  p.state=CartesianRadialJet(axis,r,n);
  const auto geometry=Geometry(p.state.metric);
  for(int i=0;i<3;++i) {
   p.state.lambda.value[i]=geometry.contracted[i];
   for(int d=0;d<3;++d)p.state.lambda.d[d][i]=geometry.dcontracted[d][i];
   p.beta[i]=p.state.beta.value[i];p.dalpha[i]=p.state.alpha.d[i];
  }
  p.alpha=alpha.value;p.k_physical=kp;p.k_bar=kb;p.b=b.value;
  p.outgoing=alpha.value*(alpha.value+b.value)/ell.value;
  p.ingoing=-alpha.value*o.value*o.value/(ell.value*(alpha.value+b.value));
  // Omega, its Cartesian jets, L, Nref and BoxOmega/Omega are production values.
  return p;
 }
};
} }
#endif
