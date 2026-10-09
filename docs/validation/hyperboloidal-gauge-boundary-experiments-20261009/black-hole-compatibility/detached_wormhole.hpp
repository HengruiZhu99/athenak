#ifndef SCRATCH_DETACHED_WORMHOLE_HPP_
#define SCRATCH_DETACHED_WORMHOLE_HPP_
#include "detached_height.hpp"
namespace z4c { namespace hyperboloidal {
template<typename T> class DetachedWormhole {
 public:
  DetachedWormhole(const LayerReference<T>&reference,T mass,LayerParameters height)
      :geometry_(reference,height),mass_(mass) {
   if(!(mass>0)||!std::isfinite(mass))throw std::invalid_argument("detached wormhole requires M>0");
   const T omega=reference.At(T(height.r0),0,0).omega;
   if(!(omega>0)||!(mass<2*T(height.r0)/omega))
    throw std::invalid_argument("detached height core must contain physical throat R=M/2");
  }
  const DetachedHeightGeometry<T>& geometry() const{return geometry_;}
  T mass() const{return mass_;}
  KOKKOS_INLINE_FUNCTION Z4cJet<T> At(T x,T y,T z) const {
   const T r=Kokkos::sqrt(x*x+y*y+z*z);
   if(r==0) {
    Z4cJet<T> u{};for(int i=0;i<3;++i){u.metric.g[i][i]=1;u.alpha.dd[i][i]=8/(mass_*mass_);}return u;
   }
   const auto p=geometry_.At(r,0,0);const auto &ur=p.state;
   const Radial2<T> one{1,0,0},radius{r,1,0},half{mass_/2,0,0};
   const Radial2<T> omega{p.omega,p.domega[0],p.omega_hessian[0][0]};
   const auto iw=SmoothCutoff(r,T(geometry_.layer.r0),T(geometry_.layer.r1));
   const T s=geometry_.scri_radius,a=geometry_.curvature_radius;
   const T outer=(s-r)*(s+r)/(2*a*s),op=-r/(a*s),opp=-1/(a*s);
   const T o3=iw.ddd*(outer-1)+3*iw.dd*op+3*iw.d*opp;
   const Radial2<T> ell{p.L,-r*omega.dd,-omega.dd-r*o3};
   const Radial2<T> alphaB{ur.alpha.value,ur.alpha.d[0],ur.alpha.dd[0][0]};
   const Radial2<T> chiB{ur.chi.value,ur.chi.d[0],ur.chi.dd[0][0]};
   const Radial2<T> betaB{ur.beta.value[0],ur.beta.d[0][0],ur.beta.dd[0][0][0]};
   const auto b=Radial2<T>{-1,0,0}*betaB*ell/alphaB;
   Radial2<T> w{};
   if(r>=T(geometry_.height.r1))w={1,0,0};
   else if(r>T(geometry_.height.r0))w=b*Radial2<T>{a,0,0}/radius;
   const auto inverse_psi=radius/(radius+half*omega);
   const auto ip2=inverse_psi*inverse_psi;
   const auto N=(radius-half*omega)/(radius+half*omega);
   const auto alpha=alphaB*((one-w)*ip2+w*N);
   const auto chi=chiB*ip2*ip2;
   const auto beta=betaB*N*ip2;
   Z4cJet<T> axis=ur;
   axis.alpha={alpha.value,{alpha.d,0,0},{{alpha.dd,0,0},{0,0,0},{0,0,0}}};
   axis.chi={chi.value,{chi.d,0,0},{{chi.dd,0,0},{0,0,0},{0,0,0}}};
   axis.beta={};axis.beta.value[0]=beta.value;axis.beta.d[0][0]=beta.d;axis.beta.dd[0][0][0]=beta.dd;
   axis.trace={};axis.a={};
   if(r>T(geometry_.height.r0)) {
    const Radial2<T> db{b.d,b.dd,0},domega{omega.d,omega.dd,0};
    const auto m=half*omega/radius,den=one-m*m;
    const auto krB=(Radial2<T>{-1,0,0}*omega*db+b*domega)/ell;
    const auto kr=ip2*(krB-Radial2<T>{2,0,0}*b*m/(radius*den));
    const auto kt=Radial2<T>{-1,0,0}*ip2*b*N/radius;
    const auto P=kr+Radial2<T>{2,0,0}*kt;
    axis.trace.value=P.value;axis.trace.d[0]=P.d;
    const auto DB=(Radial2<T>{-1,0,0}*db+b/radius)/ell;
    const auto shear=ip2*(DB-b*Radial2<T>{mass_,0,0}*(Radial2<T>{2,0,0}-m)/(radius*radius*den));
    const Radial2<T> gr{ur.metric.g[0][0],ur.metric.dg[0][0][0],0};
    const Radial2<T> gt{ur.metric.g[1][1],ur.metric.dg[0][1][1],0};
    const auto ar=Radial2<T>{T(2)/3,0,0}*gr*shear,at=Radial2<T>{-T(1)/3,0,0}*gt*shear;
    axis.a.k[0][0]=ar.value;axis.a.dk[0][0][0]=ar.d;
    for(int i=1;i<3;++i){axis.a.k[i][i]=at.value;axis.a.dk[0][i][i]=at.d;}
   }
   const T n[3]={x/r,y/r,z/r};auto u=CartesianRadialJet(axis,r,n);
   const auto actual_geometry=geometry_.At(x,y,z);
   u.metric=actual_geometry.state.metric;u.lambda=actual_geometry.state.lambda;
   return u;
  }
 private:
  DetachedHeightGeometry<T> geometry_;
  T mass_;
};
} }
#endif
