// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#ifndef Z4C_HYPERBOLOIDAL_CMC_TRUMPET_HPP_
#define Z4C_HYPERBOLOIDAL_CMC_TRUMPET_HPP_

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include "z4c/hyperboloidal/spherical_tensor.hpp"

namespace z4c {
namespace hyperboloidal {

// Host-side Schwarzschild CMC trumpet initializer, S=a=1, K=-3.
// R is physical areal radius, J=-R+C/R^2, D=1-2M/R+J^2.
// The critical C gives a double zero of D at R0, an infinite proper-distance
// cylindrical end at isotropic r=0. This is not the C=0 wormhole exterior.
class CMCTrumpet {
 public:
  explicit CMCTrumpet(double mass) {
    if (!(mass > 0) || !std::isfinite(mass)) {
      throw std::invalid_argument("trumpet mass must be positive and finite");
    }
    double lo = 1.5*mass, hi = 2*mass;
    for (int i = 0; i < 100; ++i) {
      const double r = (lo+hi)/2;
      const double f = 2*r-3*mass-3*r*r*std::sqrt(2*mass/r-1);
      if (f > 0) hi = r;
      else lo = r;
    }
    radius_ = (lo+hi)/2;
    c_ = radius_*radius_*(std::sqrt(2*mass/radius_-1)+radius_);
    // F(u)=u^2 D(1/u)=(u0-u)^2 Q(u). Factorization avoids catastrophic
    // cancellation close to the trumpet. Coefficients use both polynomial ends.
    const double u0 = 1/radius_;
    q_[4] = c_*c_;
    q_[3] = 2*u0*q_[4];
    q_[2] = 3*u0*u0*q_[4];
    q_[1] = 2/(u0*u0*u0);
    q_[0] = 1/(u0*u0);
  }

  double radius() const { return radius_; }
  double integration_constant() const { return c_; }

  double InverseAreal(double r, double step = 0.001) const {
    if (!(r > 0 && r < 1) || !(step > 0) || !std::isfinite(step)) {
      throw std::invalid_argument("trumpet sample requires 0<r<1 and finite step");
    }
    // t=-log(r), du/dt=sqrt(F(u)); u=0 at scri. Fixed RK4 quadrature
    // avoids an arbitrary finite outer areal-radius cutoff or a puncture floor.
    const double end = -std::log(r);
    const int steps = std::max(1, static_cast<int>(std::ceil(end/step)));
    const double h = end/steps;
    double u = 0;
    for (int i = 0; i < steps; ++i) {
      const double k1 = RootF(u), k2 = RootF(u+h*k1/2);
      const double k3 = RootF(u+h*k2/2), k4 = RootF(u+h*k3);
      u += h*(k1+2*k2+2*k3+k4)/6;
    }
    return u;
  }

  void Values(double r, double v[NFIELDS], double step = 0.001) const {
    for (int f = 0; f < NFIELDS; ++f) v[f] = 0;
    const double u = InverseAreal(r, step), omega = (1-r*r)/2;
    const double ratio = r*u/omega;
    v[DCHI] = ratio*ratio-1;
    v[ARR] = -2*c_*u*u*u/omega;
    v[DALPHA] = omega*RootF(u)/u-(1+r*r)/2;
    v[DBETA] = c_*r*u*u*u;
  }

  Z4cJet<double> Jet(double r) const {
    double v[NFIELDS], d[NFIELDS]{}, dd[NFIELDS]{};
    Values(r, v);
    const double u = InverseAreal(r), o = (1-r*r)/2;
    double q = q_[4], dq = 0, ddq = 0;
    for (int i = 3; i >= 0; --i) {
      ddq = ddq*u+2*dq;
      dq = dq*u+q;
      q = q*u+q_[i];
    }
    const double root = std::sqrt(q), gap = 1/radius_-u;
    const double z = gap*root;
    const double dz = -root+gap*dq/(2*root);
    const double ddz = -dq/root+gap*(ddq/(2*root)-dq*dq/(4*q*root));
    const double du = -z/r, ddu = z*(dz+1)/(r*r);
    const double lu = du/u, dlu = ddu/u-lu*lu;
    const double lo = -r/o, dlo = -1/o-r*r/(o*o);
    const double chi = 1+v[DCHI], lc = 2*(1/r+lu-lo);
    d[DCHI] = chi*lc;
    dd[DCHI] = chi*(lc*lc+2*(-1/(r*r)+dlu-dlo));
    const double la = 3*lu-lo;
    d[ARR] = v[ARR]*la;
    dd[ARR] = v[ARR]*(la*la+3*dlu-dlo);
    const double alpha = o*z/u, lz = dz*du/z;
    const double dlz = (ddz*du*du+dz*ddu)/z-lz*lz;
    const double lapse_log = lo+lz-lu;
    d[DALPHA] = alpha*lapse_log-r;
    dd[DALPHA] = alpha*(lapse_log*lapse_log+dlo+dlz-dlu)-1;
    d[DBETA] = c_*(u*u*u+3*r*u*u*du);
    dd[DBETA] = c_*(6*u*u*du+6*r*u*du*du+3*r*u*u*ddu);
    return SphericalJet(r, v, d, dd, CMCReference<double>{1, 1});
  }

 private:
  double RootF(double u) const {
    double q = q_[4];
    for (int i = 3; i >= 0; --i) q = q*u+q_[i];
    return (1/radius_-u)*std::sqrt(q);
  }
  double radius_, c_, q_[5];
};

}  // namespace hyperboloidal
}  // namespace z4c
#endif  // Z4C_HYPERBOLOIDAL_CMC_TRUMPET_HPP_
