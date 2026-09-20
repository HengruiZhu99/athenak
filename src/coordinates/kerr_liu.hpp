#ifndef COORDINATES_KERR_LIU_HPP_
#define COORDINATES_KERR_LIU_HPP_
//========================================================================================
// AthenaK astrophysical plasma code. Licensed under the 3-clause BSD License.
//========================================================================================
//! Axis-regular Cartesian Liu-Etienne-Shapiro Kerr wormhole initial data.
//! Primary radial map and K_ij: arXiv:1001.4077, Eqs. 11 and 13--15.
//!
//! This is a standalone geometry provider, not an evolution/background registration.
//! The positive precollapsed lapse does NOT make these data stationary. Subtracting
//! their nonzero continuum RHS would define a different, reference-forced PDE.
//! r=0 and |a|>=M are excluded; no floor, excision or throat 0/0 is introduced.

#include <cmath>

#ifdef KOKKOS_INLINE_FUNCTION
#define KERR_LIU_INLINE KOKKOS_INLINE_FUNCTION
#else
#define KERR_LIU_INLINE inline
#endif

namespace kerr_liu {

enum class Gauge {
  precollapsed_zero_shift,
  signed_stationary_diagnostic  // Negative on the second sheet, zero at the throat.
};
enum class Status { success, invalid_parameters, puncture, nonfinite_geometry };

//! Cartesian value/first/second derivatives. Derivative indices precede tensor indices.
template <typename T>
struct Jet {
  T value = 0;
  T d[3] = {};
  T dd[3][3] = {};
  KERR_LIU_INLINE Jet() = default;
  KERR_LIU_INLINE Jet(T v) : value(v) {}
  KERR_LIU_INLINE static Jet Coordinate(T v, int axis) {
    Jet out(v);
    out.d[axis] = 1;
    return out;
  }
};

template <typename T>
KERR_LIU_INLINE Jet<T> operator+(const Jet<T> &a, const Jet<T> &b) {
  Jet<T> out(a.value + b.value);
  for (int i=0; i<3; ++i) {
    out.d[i] = a.d[i] + b.d[i];
    for (int j=0; j<3; ++j) out.dd[i][j] = a.dd[i][j] + b.dd[i][j];
  }
  return out;
}
template <typename T>
KERR_LIU_INLINE Jet<T> operator-(const Jet<T> &a) {
  Jet<T> out(-a.value);
  for (int i=0; i<3; ++i) {
    out.d[i] = -a.d[i];
    for (int j=0; j<3; ++j) out.dd[i][j] = -a.dd[i][j];
  }
  return out;
}
template <typename T>
KERR_LIU_INLINE Jet<T> operator-(const Jet<T> &a, const Jet<T> &b) { return a + (-b); }
template <typename T>
KERR_LIU_INLINE Jet<T> operator*(const Jet<T> &a, const Jet<T> &b) {
  Jet<T> out(a.value*b.value);
  for (int i=0; i<3; ++i) {
    out.d[i] = a.d[i]*b.value + a.value*b.d[i];
    for (int j=0; j<3; ++j) {
      out.dd[i][j] = a.dd[i][j]*b.value + a.value*b.dd[i][j]
                   + a.d[i]*b.d[j] + a.d[j]*b.d[i];
    }
  }
  return out;
}
template <typename T>
KERR_LIU_INLINE Jet<T> Power(const Jet<T> &a, T exponent) {
  const T v = std::pow(a.value, exponent);
  const T fp = exponent*v/a.value;
  const T fpp = exponent*(exponent-T(1))*v/(a.value*a.value);
  Jet<T> out(v);
  for (int i=0; i<3; ++i) {
    out.d[i] = fp*a.d[i];
    for (int j=0; j<3; ++j) out.dd[i][j] = fp*a.dd[i][j] + fpp*a.d[i]*a.d[j];
  }
  return out;
}
template <typename T>
KERR_LIU_INLINE Jet<T> operator/(const Jet<T> &a, const Jet<T> &b) {
  return a*Power(b, T(-1));
}
// Explicit scalar overloads keep template deduction usable for T-valued constants.
template <typename T> KERR_LIU_INLINE Jet<T> operator+(const Jet<T> &a,T b) {return a+Jet<T>(b);}
template <typename T> KERR_LIU_INLINE Jet<T> operator+(T a,const Jet<T> &b) {return Jet<T>(a)+b;}
template <typename T> KERR_LIU_INLINE Jet<T> operator-(const Jet<T> &a,T b) {return a-Jet<T>(b);}
template <typename T> KERR_LIU_INLINE Jet<T> operator-(T a,const Jet<T> &b) {return Jet<T>(a)-b;}
template <typename T> KERR_LIU_INLINE Jet<T> operator*(const Jet<T> &a,T b) {return a*Jet<T>(b);}
template <typename T> KERR_LIU_INLINE Jet<T> operator*(T a,const Jet<T> &b) {return Jet<T>(a)*b;}
template <typename T> KERR_LIU_INLINE Jet<T> operator/(const Jet<T> &a,T b) {return a/Jet<T>(b);}
template <typename T> KERR_LIU_INLINE Jet<T> operator/(T a,const Jet<T> &b) {return Jet<T>(a)/b;}

template <typename T>
KERR_LIU_INLINE Jet<T> Determinant(const Jet<T> g[3][3]) {
  return g[0][0]*g[1][1]*g[2][2] + T(2)*g[0][1]*g[0][2]*g[1][2]
       - g[0][0]*g[1][2]*g[1][2] - g[1][1]*g[0][2]*g[0][2]
       - g[2][2]*g[0][1]*g[0][1];
}
template <typename T>
KERR_LIU_INLINE void Inverse(const Jet<T> g[3][3], const Jet<T> &det,
                             Jet<T> gi[3][3]) {
  gi[0][0]=(g[1][1]*g[2][2]-g[1][2]*g[1][2])/det;
  gi[1][1]=(g[0][0]*g[2][2]-g[0][2]*g[0][2])/det;
  gi[2][2]=(g[0][0]*g[1][1]-g[0][1]*g[0][1])/det;
  gi[0][1]=gi[1][0]=(g[0][2]*g[1][2]-g[0][1]*g[2][2])/det;
  gi[0][2]=gi[2][0]=(g[0][1]*g[1][2]-g[0][2]*g[1][1])/det;
  gi[1][2]=gi[2][1]=(g[0][1]*g[0][2]-g[0][0]*g[1][2])/det;
}

template <typename T>
struct FirstDerivative { T value = 0; T d[3] = {}; };

template <typename T>
struct Geometry {
  T radius = 0, throat_radius = 0;
  Jet<T> gamma[3][3], K[3][3], alpha, beta[3];
  Jet<T> chi, conformal_metric[3][3], conformal_A[3][3], trace_K;
  FirstDerivative<T> conformal_Gamma[3];
  Jet<T> signed_stationary_lapse;
};

template <typename T>
KERR_LIU_INLINE bool Finite(const Jet<T> &x) {
  if (!std::isfinite(x.value)) return false;
  for (int i=0; i<3; ++i) {
    if (!std::isfinite(x.d[i])) return false;
    for (int j=0; j<3; ++j) if (!std::isfinite(x.dd[i][j])) return false;
  }
  return true;
}

//! Mass and spin parameter a have length units; a/M is the dimensionless spin.
//! Coordinates are centered on the hole, with its spin aligned with the z axis.
template <typename T>
KERR_LIU_INLINE Status Evaluate(T mass, T spin, const T xyz[3], Gauge gauge,
                                 Geometry<T> &out) {
  out = Geometry<T>();
  if (!(mass > T(0)) || !(std::abs(spin) < mass) ||
      !std::isfinite(mass) || !std::isfinite(spin)) return Status::invalid_parameters;
  if (gauge != Gauge::precollapsed_zero_shift &&
      gauge != Gauge::signed_stationary_diagnostic) return Status::invalid_parameters;
  Jet<T> x[3];
  for (int i=0; i<3; ++i) {
    if (!std::isfinite(xyz[i])) return Status::invalid_parameters;
    x[i] = Jet<T>::Coordinate(xyz[i], i);
  }
  const Jet<T> r2=x[0]*x[0]+x[1]*x[1]+x[2]*x[2];
  if (!(r2.value > T(0))) return Status::puncture;
  const Jet<T> r=Power(r2,T(.5));
  const T gap=std::sqrt(mass*mass-spin*spin);
  const T rp=mass+gap, rm=mass-gap, c=rp/T(4), a2=spin*spin;
  const Jet<T> R=(r+c)*(r+c)/r;
  const Jet<T> C=x[2]/r, sin2=T(1)-C*C;
  const Jet<T> Sigma=R*R+a2*C*C;
  // (R-rp)=(r-c)^2/r avoids subtracting almost equal radii at the throat.
  const Jet<T> Delta=(r-c)*(r-c)*(R-rm)/r;
  const Jet<T> bigA=(R*R+a2)*(R*R+a2)-a2*Delta*sin2;
  const Jet<T> F=Sigma*(r+c)*(r+c)/(r*r*r*(R-rm));
  const Jet<T> B=Sigma/r2;
  const Jet<T> azimuth=a2*(Sigma+T(2)*mass*R)/(Sigma*r2*r2);
  const Jet<T> P=T(3)*R*R*R*R+T(2)*a2*R*R-a2*a2
                -a2*(R*R-a2)*sin2;
  const Jet<T> common=mass*spin/(Sigma*Power(bigA*Sigma,T(.5)));
  const Jet<T> U=common*P*(T(1)+c/r)/Power(r*(R-rm),T(.5));
  const Jet<T> V=-T(2)*a2*common*R*(r-c)*Power((R-rm)/r,T(.5));
  const Jet<T> v[3]={-x[1],x[0],Jet<T>(T(0))};
  Jet<T> n[3];
  for (int i=0; i<3; ++i) n[i]=x[i]/r;
  for (int i=0; i<3; ++i) {
    for (int j=0; j<3; ++j) {
      out.gamma[i][j]=B*T(i==j)+(F-B)*n[i]*n[j]+azimuth*v[i]*v[j];
      out.K[i][j]=U/r2*(n[i]*v[j]+n[j]*v[i])
        +C*V/(r2*r)*((C*n[i]-T(i==2))*v[j]+(C*n[j]-T(j==2))*v[i]);
    }
  }
  const Jet<T> det=Determinant(out.gamma);
  if (!(det.value > T(0)) || !Finite(det)) return Status::nonfinite_geometry;
  out.chi=Power(det,-T(1)/T(3));
  Jet<T> inverse[3][3], conformal_inverse[3][3];
  Inverse(out.gamma,det,inverse);
  for (int i=0; i<3; ++i)
    for (int j=0; j<3; ++j) out.trace_K=out.trace_K+inverse[i][j]*out.K[i][j];
  for (int i=0; i<3; ++i) {
    for (int j=0; j<3; ++j) {
      out.conformal_metric[i][j]=out.chi*out.gamma[i][j];
      conformal_inverse[i][j]=inverse[i][j]/out.chi;
      out.conformal_A[i][j]=out.chi*(out.K[i][j]-out.gamma[i][j]*out.trace_K/T(3));
    }
  }
  for (int i=0; i<3; ++i) {
    for (int j=0; j<3; ++j) {
      out.conformal_Gamma[i].value-=conformal_inverse[i][j].d[j];
      for (int k=0; k<3; ++k) out.conformal_Gamma[i].d[k]-=conformal_inverse[i][j].dd[k][j];
    }
  }
  out.signed_stationary_lapse=(r-c)*Power((R-rm)*Sigma/(r*bigA),T(.5));
  if (gauge == Gauge::precollapsed_zero_shift) {
    out.alpha=Power(out.chi,T(.5));
  } else {
    out.alpha=out.signed_stationary_lapse;
    const Jet<T> omega=-T(2)*mass*spin*R/bigA;
    for (int i=0; i<3; ++i) out.beta[i]=omega*v[i];
  }
  out.radius=r.value; out.throat_radius=c;
  if (!Finite(out.alpha) || !Finite(out.chi) || !Finite(out.trace_K))
    return Status::nonfinite_geometry;
  for (int i=0; i<3; ++i) {
    if (!Finite(out.beta[i]) || !std::isfinite(out.conformal_Gamma[i].value))
      return Status::nonfinite_geometry;
    for (int j=0; j<3; ++j) {
      if (!Finite(out.gamma[i][j]) || !Finite(out.K[i][j]) ||
          !Finite(out.conformal_metric[i][j]) || !Finite(out.conformal_A[i][j]) ||
          !std::isfinite(out.conformal_Gamma[i].d[j])) return Status::nonfinite_geometry;
    }
  }
  return Status::success;
}

}  // namespace kerr_liu
#undef KERR_LIU_INLINE
#endif  // COORDINATES_KERR_LIU_HPP_
