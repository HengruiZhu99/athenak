#ifndef SCRATCH_WIDE_FLAT_PENROSE_OVERLAY_HPP_
#define SCRATCH_WIDE_FLAT_PENROSE_OVERLAY_HPP_
// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#ifndef Z4C_HYPERBOLOIDAL_LAYER_REFERENCE_HPP_
#define Z4C_HYPERBOLOIDAL_LAYER_REFERENCE_HPP_

#include "z4c/hyperboloidal/cmc_reference.hpp"
#include "z4c/hyperboloidal/cartesian_radial.hpp"

namespace z4c {
namespace hyperboloidal {

// Values and analytic radial derivatives through third order. The logistic is
// evaluated from its smaller exponential; no finite differences at endpoints.
template <typename T>
struct CutoffJet {
  T value, d, dd, ddd;
};

template <typename T>
KOKKOS_INLINE_FUNCTION CutoffJet<T> SmoothCutoff(T r, T r0, T r1) {
  if (r <= r0)
    return {0, 0, 0, 0};
  if (r >= r1)
    return {1, 0, 0, 0};
  const T width = r1 - r0, s = (r - r0) / width, t = (r1 - r) / width;
  const T g = -1 / s + 1 / t;
  const T e = Kokkos::exp(-Kokkos::abs(g));
  const T w = g <= 0 ? e / (1 + e) : 1 / (1 + e);
  // Once the exponential underflows, avoid 0*overflow in derivative factors.
  if (e == 0)
    return {w, 0, 0, 0};
  const T p = e / ((1 + e) * (1 + e)), h = 1 - 2 * w;
  const T dg = 1 / (s * s) + 1 / (t * t);
  const T ddg = -2 / (s * s * s) + 2 / (t * t * t);
  const T dddg = 6 / (s * s * s * s) + 6 / (t * t * t * t);
  return {w, p * dg / width, p * (h * dg * dg + ddg) / (width * width),
          p * ((h * h - 2 * p) * dg * dg * dg + 3 * h * dg * ddg + dddg) /
              (width * width * width)};
}

// Second radial jets; derivatives are ordinary derivatives, not Taylor weights.
template <typename T>
struct Radial2 {
  T value, d, dd;
};
template <typename T>
KOKKOS_INLINE_FUNCTION Radial2<T> operator+(Radial2<T> x, Radial2<T> y) {
  return {x.value + y.value, x.d + y.d, x.dd + y.dd};
}
template <typename T>
KOKKOS_INLINE_FUNCTION Radial2<T> operator-(Radial2<T> x, Radial2<T> y) {
  return {x.value - y.value, x.d - y.d, x.dd - y.dd};
}
template <typename T>
KOKKOS_INLINE_FUNCTION Radial2<T> operator*(Radial2<T> x, Radial2<T> y) {
  return {x.value * y.value, x.d * y.value + x.value * y.d,
          x.dd * y.value + 2 * x.d * y.d + x.value * y.dd};
}
template <typename T>
KOKKOS_INLINE_FUNCTION Radial2<T> Power(Radial2<T> x, T exponent) {
  const T v = Kokkos::pow(x.value, exponent), f = exponent / x.value;
  return {v, v * f * x.d, v * f * (x.dd + (exponent - 1) * x.d * x.d / x.value)};
}
template <typename T>
KOKKOS_INLINE_FUNCTION Radial2<T> operator/(Radial2<T> x, Radial2<T> y) {
  return x * Power(y, T(-1));
}

struct LayerParameters {
  bool enabled = false;
  double r0 = 0.35, r1 = 0.75;
  void Validate(double s, double a) const {
    if (!enabled)
      return;
    if (!std::isfinite(r0) || !std::isfinite(r1) || !(0 < r0 && r0 < r1 && r1 < s) ||
        !(a >= s / 2)) {
      throw std::invalid_argument("layer requires 0<r0<r1<S and a>=S/2");
    }
  }
};

template <typename T>
struct LayerPoint : CMCPoint<T> {
  Z4cJet<T> state;  // all spatial jets consumed by ConformalRHS
  T omega_hessian[3][3], radius, L, b, w_omega, n_residue;
  T outgoing, ingoing;
};

// Inherits the legacy descriptor so existing CMC-only utilities remain usable
// when enabled=false. Massive initial data are derived by a separate helper;
// this descriptor and all of its reference fields remain Minkowski.
template <typename T>
struct OriginalLayerReference : CMCReference<T> {
  LayerParameters layer;
  OriginalLayerReference(T s, T a, LayerParameters p = {}) : CMCReference<T>{s, a}, layer(p) {}
  void Validate() const {
    CMCReference<T>::Validate();
    layer.Validate(this->scri_radius, this->curvature_radius);
  }

  KOKKOS_INLINE_FUNCTION
  LayerPoint<T> At(T x, T y, T z) const {
    LayerPoint<T> p{};
    p.radius = Kokkos::sqrt(x * x + y * y + z * z);
    const T a = this->curvature_radius, s = this->scri_radius;
    // Exact outer branch also supplies the analytic continuation needed only
    // for reference-deviation ghosts. No PDE is evaluated beyond scri.
    if (!layer.enabled || p.radius >= layer.r1) {
      static_cast<CMCPoint<T>&>(p) = CMCReference<T>::At(x, y, z);
      if (layer.enabled)
        p.omega = (s - p.radius) * (s + p.radius) / (2 * a * s);
      auto& u = p.state;
      u.chi.value = 1;
      u.alpha.value = p.alpha;
      u.trace.value = p.k_physical;
      for (int i = 0; i < 3; ++i) {
        u.metric.g[i][i] = 1;
        u.alpha.d[i] = p.dalpha[i];
        u.alpha.dd[i][i] = p.hessian_alpha;
        u.beta.value[i] = p.beta[i];
        u.beta.d[i][i] = -1 / a;
        p.omega_hessian[i][i] = p.hessian_omega;
      }
      p.L = p.alpha;
      p.b = p.radius / a;
      p.n_residue = p.radius * p.radius / (a * a * s * s * p.L * p.L);
      p.w_omega = 2 * p.n_residue +
                  p.omega * (-3 / (a * s * p.L * p.L) +
                             p.radius * p.radius / (a * a * s * s * p.L * p.L * p.L));
      p.outgoing = CMCReference<T>::OutgoingSpeed(p.radius);
      p.ingoing = CMCReference<T>::IngoingSpeed(p.radius);
      return p;
    }
    if (p.radius <= layer.r0) {
      // No radial division in the exactly Cauchy neighborhood of the origin.
      p.omega = p.alpha = p.L = p.state.chi.value = p.state.alpha.value = 1;
      for (int i = 0; i < 3; ++i) p.state.metric.g[i][i] = 1;
      p.outgoing = 1;
      p.ingoing = -1;
      return p;
    }
    const T r = p.radius;
    const auto w = SmoothCutoff(r, T(layer.r0), T(layer.r1));
    const T outer = (s - r) * (s + r) / (2 * a * s), op = -r / (a * s),
            opp = -1 / (a * s);
    const Radial2<T> o{(1 - w.value) + w.value * outer, w.d * (outer - 1) + w.value * op,
                       w.dd * (outer - 1) + 2 * w.d * op + w.value * opp};
    const T o3 = w.ddd * (outer - 1) + 3 * w.dd * op + 3 * w.d * opp;
    const Radial2<T> b{r * w.value / a, (w.value + r * w.d) / a,
                       (2 * w.d + r * w.dd) / a};
    const Radial2<T> l{o.value - r * o.d, -r * o.dd, -o.dd - r * o3};
    const T av = Kokkos::hypot(o.value, b.value);
    const T ad = (o.value * o.d + b.value * b.d) / av;
    const Radial2<T> alpha{
        av, ad, (o.d * o.d + b.d * b.d + o.value * o.dd + b.value * b.dd - ad * ad) / av};
    const Radial2<T> chi = Power(alpha / l, T(2) / 3);
    const Radial2<T> radial_metric = chi * l * l / (alpha * alpha);
    const Radial2<T> beta = Radial2<T>{-1, 0, 0} * b * alpha / l;
    const T kr = -b.d / l.value, kt = -w.value / (a * l.value), kb = kr + 2 * kt;
    const T dkr = -b.dd / l.value + b.d * l.d / (l.value * l.value);
    const T dkt = -w.d / (a * l.value) + w.value * l.d / (a * l.value * l.value);
    const T dkb = dkr + 2 * dkt;
    const Radial2<T> ar{
        radial_metric.value * (kr - kb / 3),
        radial_metric.d * (kr - kb / 3) + radial_metric.value * (dkr - dkb / 3), 0};
    const Radial2<T> at{chi.value * (kt - kb / 3),
                        chi.d * (kt - kb / 3) + chi.value * (dkt - dkb / 3), 0};
    const T wn = b.value * o.d / l.value;
    const T dwn = (b.d * o.d + b.value * o.dd) / l.value - wn * l.d / l.value;
    // Physical trace is factored without cancellation of O*Kbar and 3*wn.
    p.k_physical = -(3 * w.value + r * o.value * w.d / l.value) / a;
    const T dkp = o.d * kb + o.value * dkb + 3 * dwn;
    Z4cJet<T> axis{};
    axis.alpha = {alpha.value, {alpha.d, 0, 0}, {{alpha.dd, 0, 0}, {0, 0, 0}, {0, 0, 0}}};
    axis.chi = {chi.value, {chi.d, 0, 0}, {{chi.dd, 0, 0}, {0, 0, 0}, {0, 0, 0}}};
    axis.trace.value = p.k_physical;
    axis.trace.d[0] = dkp;
    axis.beta.value[0] = beta.value;
    axis.beta.d[0][0] = beta.d;
    axis.beta.dd[0][0][0] = beta.dd;
    for (int i = 0; i < 3; ++i) {
      const auto m = i == 0 ? radial_metric : chi;
      const auto k = i == 0 ? ar : at;
      axis.metric.g[i][i] = m.value;
      axis.metric.dg[0][i][i] = m.d;
      axis.metric.ddg[0][0][i][i] = m.dd;
      axis.a.k[i][i] = k.value;
      axis.a.dk[0][i][i] = k.d;
    }
    const T n[3] = {x / r, y / r, z / r};
    p.state = CartesianRadialJet(axis, r, n);
    // Flat Cartesian connection convention: Lambda=-partial_j gtilde^{ij}.
    const auto geometry = Geometry(p.state.metric);
    for (int i = 0; i < 3; ++i) {
      p.state.lambda.value[i] = geometry.contracted[i];
      for (int j = 0; j < 3; ++j) p.state.lambda.d[j][i] = geometry.dcontracted[j][i];
      p.beta[i] = p.state.beta.value[i];
      p.dalpha[i] = p.state.alpha.d[i];
      p.domega[i] = o.d * n[i];
      for (int j = 0; j < 3; ++j) {
        p.omega_hessian[i][j] =
            o.dd * n[i] * n[j] + o.d * ((i == j ? 1 : 0) - n[i] * n[j]) / r;
      }
    }
    p.omega = o.value;
    p.alpha = alpha.value;
    p.k_bar = kb;
    p.L = l.value;
    p.b = b.value;
    p.n_residue = o.d * o.d / (l.value * l.value);
    p.w_omega = 2 * p.n_residue + o.value * (o.dd / (l.value * l.value) +
                                             2 * o.d / (r * l.value * l.value) -
                                             o.d * l.d / (l.value * l.value * l.value));
    p.outgoing = alpha.value * (alpha.value + b.value) / l.value;
    p.ingoing = -alpha.value * o.value * o.value / (l.value * (alpha.value + b.value));
    // Lambda second derivatives, A second derivatives and trace second
    // derivatives are not consumed by this kernel and are intentionally unset.
    return p;
  }
};

}  // namespace hyperboloidal
}  // namespace z4c
#endif  // Z4C_HYPERBOLOIDAL_LAYER_REFERENCE_HPP_

#ifndef SCRATCH_FLAT_POWER_HPP_
#define SCRATCH_FLAT_POWER_HPP_


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
struct FlatPowerReference : OriginalLayerReference<T> {
  int omega_power = 1;
  FlatPowerReference(T s, T a, LayerParameters parameters={}, int exponent=1)
      : OriginalLayerReference<T>(s,a,parameters), omega_power(exponent) {
    if(exponent<1 || exponent>4) throw std::invalid_argument("scratch cutoff exponent must lie in 1..4");
  }

  KOKKOS_INLINE_FUNCTION LayerPoint<T> At(T x, T y, T z) const {
    auto p = OriginalLayerReference<T>::At(x,y,z);
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

namespace z4c {namespace hyperboloidal {template<typename T> using LayerReference=FlatPowerReference<T>;}}
#endif
