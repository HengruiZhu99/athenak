// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_WORMHOLE_CHART_HPP_
#define Z4C_HYPERBOLOIDAL_WORMHOLE_CHART_HPP_

#include "mesh/cubed_sphere/map.hpp"
#include "z4c/hyperboloidal/conformal_rhs.hpp"

namespace z4c { namespace hyperboloidal {
// Positive Cartesian coefficient chart. The puncture is a second AF infinity,
// never a regular center. Metric and evolved Lambda remain independent. These
// factors encode wormhole powers, not a boundary closure at the missing point.
struct WormholeChart {
  cubed_sphere::Jet chi, weight, alpha;
  Real chi_t=0, weight_t=0, alpha_t=0;
  WormholeChart(const Real x[3], const Real center[3], const Real velocity[3]) {
    using J=cubed_sphere::Jet;
    J z(0),rho2(0);Real rho2_t=0;
    for(int i=0;i<3;++i){const J y=J::Variable(x[i],i);
      z=z+y*y;rho2=rho2+(y-J(center[i]))*(y-J(center[i]));
      rho2_t-=2*(x[i]-center[i])*velocity[i];}
    const J omega=(J(1)-z)*J(.5), o4=omega*omega*omega*omega;
    const J rho4=rho2*rho2;
    chi=rho4/(rho4+o4);weight=rho2/(rho2+o4);alpha=(J(1)+z)*J(.5)*weight;
    // Coordinate time is at fixed x; Omega and the lapse clock are stationary.
    chi_t=2*rho2.f*rho2_t*o4.f/((rho4.f+o4.f)*(rho4.f+o4.f));
    weight_t=rho2_t*o4.f/((rho2.f+o4.f)*(rho2.f+o4.f));
    alpha_t=(1+z.f)*.5*weight_t;
  }
  explicit WormholeChart(const Real x[3]) : WormholeChart(x,zero,zero) {}
  inline static constexpr Real zero[3]={0,0,0};
  // Factor the actual initial isotropic inner profile rather than asking a
  // Cartesian polynomial to resolve its radial cusp. The compact C7 blend
  // returns to the generic factors before the physical slice transition.
  // This changes only the coefficient chart: no physical RHS is subtracted.
  static WormholeChart Isotropic(const Real x[3]) {
    WormholeChart result(x);using J=cubed_sphere::Jet;
    J z(0);for(int i=0;i<3;++i){const auto y=J::Variable(x[i],i);z=z+y*y;}
    const J r=cubed_sphere::Power(z,.5);
    if(r.f>=.3)return result;
    const J omega=(J(1)-z)*J(.5);
    constexpr Real mass=1./3,k=.37754694621653295,a=mass*k/2;
    const J rp=r+J(a),inner_chi=J(k*k)*z*z/(omega*omega*rp*rp*rp*rp);
    const J inner_alpha=omega*z/(rp*rp);
    J taper(1);
    if(r.f>.2){const J y=(J(.3)-r)*J(10),v=J(1)-y;
      const J y8=cubed_sphere::Power(y,8),v8=cubed_sphere::Power(v,8);taper=y8/(y8+v8);}
    result.chi=ChartBlend(inner_chi,result.chi,taper);
    result.alpha=ChartBlend(inner_alpha,result.alpha,taper);
    return result;
  }
  // Global smooth radial denominator: captures the AF radial cusp without a
  // compact matching collar. At scri the deviations vanish at least as Omega^4.
  static WormholeChart Asymptotic(const Real x[3]) {
    WormholeChart result(x);using J=cubed_sphere::Jet;
    J z(0);for(int i=0;i<3;++i){const auto y=J::Variable(x[i],i);z=z+y*y;}
    constexpr Real mass=1./3,k=.37754694621653295,a=mass*k/2;
    const J omega=(J(1)-z)*J(.5),r=cubed_sphere::Power(z,.5),rp=r+J(a);
    const J o4=omega*omega*omega*omega,rp2=rp*rp;
    result.chi=J(k*k)*z*z/(J(16)*o4*omega*omega*rp2*rp2+J(k*k)*z*z);
    result.alpha=(J(1)+z)*J(.5)*z/(z+J(16)*o4*rp2);
    return result;
  }
  static WormholeChart At(const Real x[3],int reference) {
    if(reference==1)return Isotropic(x);
    if(reference==2)return Asymptotic(x);
    return WormholeChart(x);
  }
 private:
  static cubed_sphere::Jet ChartBlend(const cubed_sphere::Jet &a,const cubed_sphere::Jet &b,
      const cubed_sphere::Jet &w){return w*a+(cubed_sphere::Jet(1)-w)*b;}
};

inline cubed_sphere::Jet ChartExponential(const cubed_sphere::Jet &q) {
  cubed_sphere::Jet out(std::exp(q.f));
  for(int i=0;i<3;++i){out.d[i]=out.f*q.d[i];for(int j=0;j<3;++j)
    out.h[i][j]=out.f*(q.h[i][j]+q.d[i]*q.d[j]);}
  return out;
}

// Selected spherical control: physical 1+log, multiplicative physical-log
// restoring (strength=1), collapsed-inner source. Algebraically identical to
// gauge.py, with inner leading terms canceled before floating-point evaluation.
// No symmetry assumption enters: the advection is beta^i partial_i alpha and
// trace is the independent physical Khat, not Khat+3 or a metric constraint.
template<typename T> KOKKOS_INLINE_FUNCTION
T WormholeLapseRate(const T x[3], const Z4cJet<T> &u, T nu=T(1), T strength=T(1)) {
  T z=0,advection=0,beta_x=0;
  for(int i=0;i<3;++i){z+=x[i]*x[i];advection+=u.beta.value[i]*u.alpha.d[i];beta_x+=u.beta.value[i]*x[i];}
  const T omega=(1-z)/2,A=(1+z)/2,a=u.alpha.value;
  const T taper=(1-z)*(1-z)*(1-z)*(1-z),xi=strength+(3*nu-strength)*taper;
  T source;
  if(z<T(.25)){
    const T one_minus_taper=z*(4+z*(-6+z*(4-z)));
    // D=12 nu Omega^2-2 xi A has zero constant coefficient.
    const T D=z*((3*nu-4*strength)+z*((2*strength-3*nu)
      +z*((2*strength-6*nu)+z*((9*nu-3*strength)+z*(strength-3*nu)))));
    source=-a*((a+4*nu*omega*omega)*u.trace.value+(3+2*xi)*a+D)
      +a*beta_x+z*(one_minus_taper+taper*a/A);
  } else {
    const T F=a*a+4*nu*omega*omega*a;
    source=-F*(u.trace.value+3)-2*xi*a*(a-A)+a*beta_x+z
      +taper*z*(a-A)/A;
  }
  return advection+source/omega;
}
}}  // namespace z4c::hyperboloidal
#endif
