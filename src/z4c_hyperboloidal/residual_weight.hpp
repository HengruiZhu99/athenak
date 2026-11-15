// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_RESIDUAL_WEIGHT_HPP_
#define Z4C_HYPERBOLOIDAL_RESIDUAL_WEIGHT_HPP_
#include "z4c_hyperboloidal/mapped_geometry.hpp"
namespace z4c_hyperboloidal {
// Fixed geometric radii, copied by value into kernels. Zero radii select the
// original single-puncture representation (including its arithmetic).
struct ResidualWeight {
  double inner=0, outer=0;
  KOKKOS_INLINE_FUNCTION bool Regular() const {return inner>0;}
  KOKKOS_INLINE_FUNCTION mapped::Jet At(const double *x) const {
    using J=mapped::Jet;
    J r2;
    for(int a=0;a<3;++a){const J y=J::Variable(x[a],a);r2=r2+y*y;}
    if (!Regular() || r2.f>=outer*outer) return r2;
    if (r2.f<=inner*inner) return J(inner*inner);
    const J r=mapped::Power(r2,.5),z=(r-J(inner))/J(outer-inner);
    // Symmetric integral of 12012 z^6 (1-z)^6: C6 at both joins.
    const bool upper=z.f>.5;
    const J t=upper?J(1)-z:z;
    J s=mapped::Power(t,7)*(J(1716)+t*(J(-9009)+t*(J(20020)+t*
      (J(-24024)+t*(J(16380)+t*(J(-6006)+J(924)*t))))));
    if(upper)s=J(1)-s;
    return J(inner*inner)+s*(r2-J(inner*inner));
  }
  KOKKOS_INLINE_FUNCTION double Value(const double *x) const {
    // Preserve the old sqrt/square operation in single-hole residual formation.
    if(!Regular()){const double r=sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);return r*r;}
    return At(x).f;
  }
};
}
#endif
