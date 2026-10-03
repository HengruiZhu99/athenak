// Access the unchanged FastFlow basis through dense tables or separated factors.
#ifndef Z4C_FASTFLOW_HARMONIC_CACHE_HPP_
#define Z4C_FASTFLOW_HARMONIC_CACHE_HPP_

#include "athena.hpp"

enum FastFlowBasis {
  FF_Y0, FF_Yc, FF_Ys, FF_dY0dth, FF_dYcdth, FF_dYsdth, FF_dYcdph, FF_dYsdph,
  FF_dY0dth2, FF_dYcdth2, FF_dYsdth2, FF_dYcdph2, FF_dYsdph2,
  FF_dYcdthdph, FF_dYsdthdph, FF_basis_count
};

template<class Table, class Factor>
struct FastFlowHarmonicCache {
  Kokkos::Array<Table, FF_basis_count> dense;
  Factor polar, phase;
  int ntheta, lmax1;
  bool factorized;
  Real sqrt2;

  KOKKOS_INLINE_FUNCTION
  Real operator()(FastFlowBasis kind, int p, int q) const {
    if (!factorized) return dense[kind](p,q);
    const int itheta=p%ntheta, iphi=p/ntheta;
    if (kind==FF_Y0) return polar(0,itheta,q*lmax1);
    if (kind==FF_dY0dth) return polar(1,itheta,q*lmax1);
    if (kind==FF_dY0dth2) return polar(2,itheta,q*lmax1);
    const int m=q%lmax1;
    const Real c=phase(0,iphi,m), s=phase(1,iphi,m);
    // Match StableFastFlowHarmonic's multiplication order, then its sqrt(2).
    if (kind==FF_Yc) return sqrt2*(polar(0,itheta,q)*c);
    if (kind==FF_Ys) return sqrt2*(polar(0,itheta,q)*s);
    if (kind==FF_dYcdth) return sqrt2*(polar(1,itheta,q)*c);
    if (kind==FF_dYsdth) return sqrt2*(polar(1,itheta,q)*s);
    if (kind==FF_dYcdph) return sqrt2*((-m*polar(0,itheta,q))*s);
    if (kind==FF_dYsdph) return sqrt2*((m*polar(0,itheta,q))*c);
    if (kind==FF_dYcdth2) return sqrt2*(polar(2,itheta,q)*c);
    if (kind==FF_dYsdth2) return sqrt2*(polar(2,itheta,q)*s);
    if (kind==FF_dYcdph2) return sqrt2*((-m*m*polar(0,itheta,q))*c);
    if (kind==FF_dYsdph2) return sqrt2*((-m*m*polar(0,itheta,q))*s);
    if (kind==FF_dYcdthdph) return sqrt2*((-m*polar(1,itheta,q))*s);
    return sqrt2*((m*polar(1,itheta,q))*c);
  }
};
#endif  // Z4C_FASTFLOW_HARMONIC_CACHE_HPP_
