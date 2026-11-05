// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_REFERENCE_VARIABLES_HPP_
#define Z4C_HYPERBOLOIDAL_REFERENCE_VARIABLES_HPP_
#include "z4c/hyperboloidal/conformal_rhs.hpp"
namespace z4c::hyperboloidal {
// A fixed scalar reference is a numerical variable definition, not a solution
// subtracted from the evolution equations. These Cartesian product/quotient
// rules also apply to nonspherical jets. No constraint or sign is projected.
template<class T> KOKKOS_INLINE_FUNCTION ScalarJet<T> ReferenceProduct(
    const ScalarJet<T>&v,const ScalarJet<T>&f) {
  ScalarJet<T> q{};q.value=f.value*v.value;
  for(int i=0;i<3;++i){q.d[i]=f.d[i]*v.value+f.value*v.d[i];
    for(int j=0;j<3;++j)q.dd[i][j]=f.dd[i][j]*v.value+
        f.d[i]*v.d[j]+f.d[j]*v.d[i]+f.value*v.dd[i][j];}
  return q;
}
template<class T> KOKKOS_INLINE_FUNCTION ScalarJet<T> ReferenceQuotient(
    const ScalarJet<T>&q,const ScalarJet<T>&f) {
  ScalarJet<T> v{};v.value=q.value/f.value;
  for(int i=0;i<3;++i)v.d[i]=(q.d[i]-f.d[i]*v.value)/f.value;
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)v.dd[i][j]=
      (q.dd[i][j]-f.dd[i][j]*v.value-f.d[i]*v.d[j]-f.d[j]*v.d[i])/f.value;
  return v;
}
// Supported fixed puncture factors are r^2 and r^4. The caller excludes r=0;
// their singular inverses are not a puncture regularity proof.
template<class T> KOKKOS_INLINE_FUNCTION ScalarJet<T> PunctureReference(
    const T x[3],int power) {
  ScalarJet<T> f{};const T r2=x[0]*x[0]+x[1]*x[1]+x[2]*x[2];
  f.value=power==2?r2:r2*r2;
  for(int i=0;i<3;++i){f.d[i]=(power==2?T(2):4*r2)*x[i];
    for(int j=0;j<3;++j)f.dd[i][j]=(i==j?(power==2?T(2):4*r2):T(0))+
        (power==2?T(0):8*x[i]*x[j]);}
  return f;
}
}
#endif
