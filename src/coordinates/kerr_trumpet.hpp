#ifndef COORDINATES_KERR_TRUMPET_HPP_
#define COORDINATES_KERR_TRUMPET_HPP_
//========================================================================================
// AthenaK astrophysical plasma code. Licensed under the 3-clause BSD License.
//========================================================================================
//! Stationary Cartesian Kerr trumpet, Dennison/Baumgarte/Montero arXiv:1409.1887.
//! Eqs. 12--18 with R0=M and r=R-M; spin is aligned with the z axis.
//! The lapse is positive for r>0 and tends to zero only at the puncture.
//! This is not ordinary stationary 1+log: the residual gauge must retain its
//! reference subtraction. No floors, clipping, excision or evolution are applied.
//! The conformal fields have directional limits at r=0 for nonzero spin, so
//! the puncture itself is excluded. |a|>=M is excluded as well.

#include <cmath>

#ifdef KOKKOS_INLINE_FUNCTION
#define KERR_TRUMPET_INLINE KOKKOS_INLINE_FUNCTION
#else
#define KERR_TRUMPET_INLINE inline
#endif

namespace kerr_trumpet {

enum class Status { success, invalid_parameters, puncture, nonfinite_geometry };

//! Cartesian value/first/second derivatives. Derivative indices precede tensor indices.
template <typename T>
struct Jet {
  T value = 0;
  T d[3] = {};
  T dd[3][3] = {};
  KERR_TRUMPET_INLINE Jet() = default;
  KERR_TRUMPET_INLINE Jet(T v) : value(v) {}
  KERR_TRUMPET_INLINE static Jet Coordinate(T v, int axis) {
    Jet out(v);
    out.d[axis] = 1;
    return out;
  }
};

template <typename T>
KERR_TRUMPET_INLINE Jet<T> operator+(const Jet<T> &a, const Jet<T> &b) {
  Jet<T> out(a.value + b.value);
  for (int i=0; i<3; ++i) {
    out.d[i] = a.d[i] + b.d[i];
    for (int j=0; j<3; ++j) out.dd[i][j] = a.dd[i][j] + b.dd[i][j];
  }
  return out;
}
template <typename T>
KERR_TRUMPET_INLINE Jet<T> operator-(const Jet<T> &a) {
  Jet<T> out(-a.value);
  for (int i=0; i<3; ++i) {
    out.d[i] = -a.d[i];
    for (int j=0; j<3; ++j) out.dd[i][j] = -a.dd[i][j];
  }
  return out;
}
template <typename T>
KERR_TRUMPET_INLINE Jet<T> operator-(const Jet<T> &a, const Jet<T> &b) { return a + (-b); }
template <typename T>
KERR_TRUMPET_INLINE Jet<T> operator*(const Jet<T> &a, const Jet<T> &b) {
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
KERR_TRUMPET_INLINE Jet<T> Power(const Jet<T> &a, T exponent) {
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
KERR_TRUMPET_INLINE Jet<T> operator/(const Jet<T> &a, const Jet<T> &b) {
  return a*Power(b, T(-1));
}
// Explicit scalar overloads keep template deduction usable for T-valued constants.
template <typename T> KERR_TRUMPET_INLINE Jet<T> operator+(const Jet<T> &a,T b) {return a+Jet<T>(b);}
template <typename T> KERR_TRUMPET_INLINE Jet<T> operator+(T a,const Jet<T> &b) {return Jet<T>(a)+b;}
template <typename T> KERR_TRUMPET_INLINE Jet<T> operator-(const Jet<T> &a,T b) {return a-Jet<T>(b);}
template <typename T> KERR_TRUMPET_INLINE Jet<T> operator-(T a,const Jet<T> &b) {return Jet<T>(a)-b;}
template <typename T> KERR_TRUMPET_INLINE Jet<T> operator*(const Jet<T> &a,T b) {return a*Jet<T>(b);}
template <typename T> KERR_TRUMPET_INLINE Jet<T> operator*(T a,const Jet<T> &b) {return Jet<T>(a)*b;}
template <typename T> KERR_TRUMPET_INLINE Jet<T> operator/(const Jet<T> &a,T b) {return a/Jet<T>(b);}
template <typename T> KERR_TRUMPET_INLINE Jet<T> operator/(T a,const Jet<T> &b) {return Jet<T>(a)/b;}

template <typename T>
KERR_TRUMPET_INLINE Jet<T> Determinant(const Jet<T> g[3][3]) {
  return g[0][0]*g[1][1]*g[2][2] + T(2)*g[0][1]*g[0][2]*g[1][2]
       - g[0][0]*g[1][2]*g[1][2] - g[1][1]*g[0][2]*g[0][2]
       - g[2][2]*g[0][1]*g[0][1];
}
template <typename T>
KERR_TRUMPET_INLINE void Inverse(const Jet<T> g[3][3], const Jet<T> &det,
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
  T radius = 0, horizon_radius = 0;
  Jet<T> gamma[3][3], alpha, beta[3];
  Jet<T> chi, conformal_metric[3][3];
  // K is formed from first derivatives of the stationary ADM fields. Its
  // first derivative uses their second derivatives. No K Hessian is supplied.
  FirstDerivative<T> K[3][3], conformal_A[3][3], trace_K;
  FirstDerivative<T> conformal_Gamma[3];
};

template <typename T>
KERR_TRUMPET_INLINE bool Finite(const Jet<T> &x) {
  if (!std::isfinite(x.value)) return false;
  for (int i=0; i<3; ++i) {
    if (!std::isfinite(x.d[i])) return false;
    for (int j=0; j<3; ++j) if (!std::isfinite(x.dd[i][j])) return false;
  }
  return true;
}
template <typename T>
KERR_TRUMPET_INLINE bool Finite(const FirstDerivative<T> &x) {
  if (!std::isfinite(x.value)) return false;
  for (int i=0; i<3; ++i) if (!std::isfinite(x.d[i])) return false;
  return true;
}

//! Mass M and spin a have length units; a/M is the dimensionless signed spin.
//! Input coordinates are centered on the hole. All derivatives are Cartesian.
//! R0=M gives the ADM falloff of the original Schwarzschild trumpet and avoids
//! the physical Kerr ring. This chart penetrates the outer horizon for |a|<M.
template <typename T>
KERR_TRUMPET_INLINE Status Evaluate(T mass, T spin, const T xyz[3],
                                     Geometry<T> &out) {
  out = Geometry<T>();
  if (!(mass > T(0)) || !(std::abs(spin) < mass) ||
      !std::isfinite(mass) || !std::isfinite(spin)) return Status::invalid_parameters;
  Jet<T> x[3];
  for (int i=0; i<3; ++i) {
    if (!std::isfinite(xyz[i])) return Status::invalid_parameters;
    x[i] = Jet<T>::Coordinate(xyz[i],i);
  }
  const Jet<T> r2=x[0]*x[0]+x[1]*x[1]+x[2]*x[2];
  if (!(r2.value > T(0))) return Status::puncture;
  const Jet<T> r=Power(r2,T(.5)), R=r+mass;
  const T a2=spin*spin, gap=std::sqrt(mass*mass-a2);
  Jet<T> n[3];
  for (int i=0; i<3; ++i) n[i]=x[i]/r;
  const Jet<T> w[3]={-n[1],n[0],Jet<T>(T(0))};
  const Jet<T> v[3]={-x[1],x[0],Jet<T>(T(0))};
  const Jet<T> Sigma=R*R+a2*n[2]*n[2];
  const Jet<T> A=R*R+a2;
  const Jet<T> X=A*A-a2*(x[0]*x[0]+x[1]*x[1]);
  // X-r^2 Sigma=(R^2+a^2)(2Mr+M^2+a^2)>0: 0<alpha<1.
  // In the (n,e_theta,e_phi) basis the radial/azimuthal metric block's
  // numerator determinant is X, and its diagonal Sigma is positive.
  // Thus gamma is positive definite everywhere off the excluded puncture.
  out.alpha=r*Power(Sigma/X,T(.5));
  const Jet<T> radial=gap*A/X;
  const Jet<T> angular=-spin*(T(2)*mass*r+mass*mass+a2)/X;
  for (int i=0; i<3; ++i) {
    out.beta[i]=radial*x[i]+angular*v[i];
    for (int j=0; j<3; ++j) {
      out.gamma[i][j]=(Sigma*T(i==j)
          +a2*(T(1)+T(2)*mass*R/Sigma)*w[i]*w[j]
          -spin*gap*(n[i]*w[j]+w[i]*n[j]))/r2;
    }
  }
  if (!(out.alpha.value>T(0)) || !Finite(out.alpha))
    return Status::nonfinite_geometry;
  // Stationarity: 0=-2 alpha K_ij+Lie_beta(gamma)_ij. Differentiate this
  // identity explicitly, without representing missing third derivatives.
  for (int i=0; i<3; ++i) {
    for (int j=0; j<3; ++j) {
      T lie=0, dlie[3]={};
      for (int k=0; k<3; ++k) {
        lie += out.beta[k].value*out.gamma[i][j].d[k]
             + out.gamma[k][j].value*out.beta[k].d[i]
             + out.gamma[i][k].value*out.beta[k].d[j];
        for (int p=0; p<3; ++p) {
          dlie[p] += out.beta[k].d[p]*out.gamma[i][j].d[k]
              +out.beta[k].value*out.gamma[i][j].dd[p][k]
              +out.gamma[k][j].d[p]*out.beta[k].d[i]
              +out.gamma[k][j].value*out.beta[k].dd[p][i]
              +out.gamma[i][k].d[p]*out.beta[k].d[j]
              +out.gamma[i][k].value*out.beta[k].dd[p][j];
        }
      }
      out.K[i][j].value=lie/(T(2)*out.alpha.value);
      for (int p=0; p<3; ++p) {
        out.K[i][j].d[p]=(dlie[p]-T(2)*out.K[i][j].value*out.alpha.d[p])
                       /(T(2)*out.alpha.value);
      }
    }
  }
  // Analytic determinant avoids subtractive cancellation between metric terms.
  const Jet<T> det=Sigma*X/(r2*r2*r2);
  if (!(det.value>T(0)) || !Finite(det)) return Status::nonfinite_geometry;
  out.chi=Power(det,-T(1)/T(3));
  Jet<T> inverse[3][3], conformal_inverse[3][3];
  Inverse(out.gamma,det,inverse);
  for (int i=0; i<3; ++i) {
    for (int j=0; j<3; ++j) {
      out.trace_K.value+=inverse[i][j].value*out.K[i][j].value;
      for (int p=0; p<3; ++p) {
        out.trace_K.d[p]+=inverse[i][j].d[p]*out.K[i][j].value
                        +inverse[i][j].value*out.K[i][j].d[p];
      }
    }
  }
  for (int i=0; i<3; ++i) {
    for (int j=0; j<3; ++j) {
      out.conformal_metric[i][j]=out.chi*out.gamma[i][j];
      conformal_inverse[i][j]=inverse[i][j]/out.chi;
      const T tf=out.K[i][j].value-out.gamma[i][j].value*out.trace_K.value/T(3);
      out.conformal_A[i][j].value=out.chi.value*tf;
      for (int p=0; p<3; ++p) {
        const T dtf=out.K[i][j].d[p]
            -(out.gamma[i][j].d[p]*out.trace_K.value
              +out.gamma[i][j].value*out.trace_K.d[p])/T(3);
        out.conformal_A[i][j].d[p]=out.chi.d[p]*tf+out.chi.value*dtf;
      }
    }
  }
  for (int i=0; i<3; ++i) {
    for (int j=0; j<3; ++j) {
      out.conformal_Gamma[i].value-=conformal_inverse[i][j].d[j];
      for (int k=0; k<3; ++k)
        out.conformal_Gamma[i].d[k]-=conformal_inverse[i][j].dd[k][j];
    }
  }
  out.radius=r.value; out.horizon_radius=gap;
  if (!Finite(out.chi) || !Finite(out.trace_K)) return Status::nonfinite_geometry;
  for (int i=0; i<3; ++i) {
    if (!Finite(out.beta[i]) || !Finite(out.conformal_Gamma[i]))
      return Status::nonfinite_geometry;
    for (int j=0; j<3; ++j) {
      if (!Finite(out.gamma[i][j]) || !Finite(out.K[i][j]) ||
          !Finite(out.conformal_metric[i][j]) || !Finite(out.conformal_A[i][j]))
        return Status::nonfinite_geometry;
    }
  }
  return Status::success;
}

}  // namespace kerr_trumpet
#undef KERR_TRUMPET_INLINE
#endif  // COORDINATES_KERR_TRUMPET_HPP_
