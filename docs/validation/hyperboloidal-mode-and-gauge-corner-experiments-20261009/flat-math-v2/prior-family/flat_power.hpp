#ifndef SCRATCH_FLAT_POWER_HPP_
#define SCRATCH_FLAT_POWER_HPP_
#include "z4c/hyperboloidal/layer_reference.hpp"

namespace z4c {
namespace hyperboloidal {

template <typename T>
KOKKOS_INLINE_FUNCTION T ScaledPowerExp(T logbase, T coefficient) {
  if (coefficient == 0) return 0;
  const T magnitude = Kokkos::exp(logbase + Kokkos::log(Kokkos::abs(coefficient)));
  return coefficient > 0 ? magnitude : -magnitude;
}

// Scratch-only analytic height chosen from a monotone isotropic compactification.
// b=sqrt((-r Omega')*(2 Omega-r Omega')); alpha=L=Omega-r Omega', beta=-b n.
// The inner square root is evaluated logarithmically, including derivatives that
// remain representable after exp(g), Omega' or b itself underflows.
template <typename T>
struct FlatPowerReference : LayerReference<T> {
  int omega_power = 2;
  FlatPowerReference(T s, T a, LayerParameters parameters={}, int exponent=2)
      : LayerReference<T>(s,a,parameters), omega_power(exponent) {
    if(exponent<1 || exponent>4) throw std::invalid_argument("scratch cutoff exponent must lie in 1..4");
  }

  KOKKOS_INLINE_FUNCTION LayerPoint<T> At(T x, T y, T z) const {
    auto p = LayerReference<T>::At(x,y,z);
    const T r=p.radius, a=this->curvature_radius, s=this->scri_radius;
    if (!this->layer.enabled || r<=this->layer.r0 || r>=this->layer.r1) return p;
    const auto basew=SmoothCutoff(r,T(this->layer.r0),T(this->layer.r1));
    const T exponent=T(omega_power);
    CutoffJet<T> w{};
    w.value=Kokkos::pow(basew.value,exponent);
    if(basew.value>0) {
      w.d=exponent*Kokkos::pow(basew.value,exponent-1)*basew.d;
      w.dd=exponent*Kokkos::pow(basew.value,exponent-1)*basew.dd;
      if(omega_power>1) w.dd+=exponent*(exponent-1)*Kokkos::pow(basew.value,exponent-2)*basew.d*basew.d;
      w.ddd=exponent*Kokkos::pow(basew.value,exponent-1)*basew.ddd;
      if(omega_power>1) w.ddd+=3*exponent*(exponent-1)*Kokkos::pow(basew.value,exponent-2)*basew.d*basew.dd;
      if(omega_power>2) w.ddd+=exponent*(exponent-1)*(exponent-2)*Kokkos::pow(basew.value,exponent-3)*basew.d*basew.d*basew.d;
    }
    const T outer=(s-r)*(s+r)/(2*a*s), opd=-r/(a*s), opdd=-1/(a*s);
    const T o3=w.ddd*(outer-1)+3*w.dd*opd+3*w.d*opdd;
    const T n[3]={x/r,y/r,z/r};
    const Radial2<T> omega{(1-w.value)+w.value*outer,
      w.d*(outer-1)+w.value*opd,
      w.dd*(outer-1)+2*w.d*opd+w.value*opdd};
    const T op=omega.d, opp=omega.dd;
    const Radial2<T> ell{omega.value-r*op,-r*opp,-opp-r*o3};
    const Radial2<T> radius{r,1,0}, one{1,0,0};
    const Radial2<T> zjet{-r*op,-op-r*opp,-2*opp-r*o3};
    Radial2<T> b{};
    const T width=T(this->layer.r1-this->layer.r0);
    const T v=(r-T(this->layer.r0))/width, t=(T(this->layer.r1)-r)/width;
    const T g=-1/v+1/t;
    if(g<=0) {
      const T g1=(1/(v*v)+1/(t*t))/width;
      const T g2=(-2/(v*v*v)+2/(t*t*t))/(width*width);
      const T g3=(6/(v*v*v*v)+6/(t*t*t*t))/(width*width*width);
      const T e=Kokkos::exp(g);
      const Radial2<T> expjet{e,e*g1,e*(g1*g1+g2)};
      const auto inv=Power(one+expjet,T(-1));
      const Radial2<T> deficit{1-outer,-opd,-opdd};
      const Radial2<T> grad{g1,g2,g3};
      const Radial2<T> acceleration{-opd,-opdd,0};
      const auto factor=Radial2<T>{exponent,0,0}*deficit*grad*Power(inv,exponent+1)+acceleration*Power(inv,exponent);
      const T logz1=1/r+exponent*g1+factor.d/factor.value;
      const T ratio=factor.d/factor.value;
      const T logz2=-1/(r*r)+exponent*g2+factor.dd/factor.value-ratio*ratio;
      const auto shape=Radial2<T>{2,0,0}*omega+zjet;
      const T shape_ratio=shape.d/shape.value;
      const T logb=(Kokkos::log(r)+exponent*g+Kokkos::log(factor.value)
                      +Kokkos::log(shape.value))/2;
      const T logb1=(logz1+shape_ratio)/2;
      const T logb2=(logz2+shape.dd/shape.value-shape_ratio*shape_ratio)/2;
      b={Kokkos::exp(logb),ScaledPowerExp(logb,logb1),ScaledPowerExp(logb,logb1*logb1+logb2)};
    } else {
      b=Power(zjet,T(.5))*Power(Radial2<T>{2,0,0}*omega+zjet,T(.5));
    }
    const T kb=(-b.d-2*b.value/r)/ell.value;
    const T d=(-b.d+b.value/r)/ell.value;
    const T dp=(-b.dd+b.d/r-b.value/(r*r))/ell.value-d*ell.d/ell.value;
    const T kp=(-omega.value*(b.d+2*b.value/r)+3*b.value*omega.d)/ell.value;
    const T dkp=(-omega.value*(b.dd+2*b.d/r-2*b.value/(r*r))
                  +2*omega.d*(b.d-b.value/r)+3*b.value*omega.dd)/ell.value
                  -kp*ell.d/ell.value;
    Z4cJet<T> axis{};
    axis.chi.value=1;
    axis.alpha={ell.value,{ell.d,0,0},{{ell.dd,0,0},{0,0,0},{0,0,0}}};
    axis.beta.value[0]=-b.value;
    axis.beta.d[0][0]=-b.d;
    axis.beta.dd[0][0][0]=-b.dd;
    axis.trace.value=kp;axis.trace.d[0]=dkp;
    for(int i=0;i<3;++i) {
      axis.metric.g[i][i]=1;
      axis.a.k[i][i]=(i==0?T(2):T(-1))*d/3;
      axis.a.dk[0][i][i]=(i==0?T(2):T(-1))*dp/3;
    }
    p.state=CartesianRadialJet(axis,r,n);
    p.alpha=ell.value;p.k_physical=kp;p.k_bar=kb;p.b=b.value;
    for(int i=0;i<3;++i) {p.beta[i]=p.state.beta.value[i];p.dalpha[i]=p.state.alpha.d[i];}
    p.outgoing=ell.value+b.value;
    p.ingoing=-omega.value*omega.value/(ell.value+b.value);
    p.omega=omega.value; p.L=ell.value;
    for(int i=0;i<3;++i) {
      p.domega[i]=omega.d*n[i];
      for(int j=0;j<3;++j) p.omega_hessian[i][j]=omega.dd*n[i]*n[j]
         +omega.d*((i==j?1:0)-n[i]*n[j])/r;
    }
    p.n_residue=omega.d*omega.d/(ell.value*ell.value);
    p.w_omega=2*p.n_residue+omega.value/(ell.value*ell.value)
       *(omega.dd+2*omega.d/r-omega.d*ell.d/ell.value);
    return p;
  }
};

} // namespace hyperboloidal
} // namespace z4c
#endif
