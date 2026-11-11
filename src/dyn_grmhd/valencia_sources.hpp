#ifndef DYN_GRMHD_VALENCIA_SOURCES_HPP_
#define DYN_GRMHD_VALENCIA_SOURCES_HPP_
#include "athena.hpp"
#include "utils/finite_diff.hpp"

// Geometry only: independent of EOS and of primitive recovery.
template<int NG, class ADM>
KOKKOS_INLINE_FUNCTION void ValenciaGeometryDerivatives(
    const ADM &adm, int ndim, const Real idx[3], int m, int k, int j, int i,
    Real da[3], Real db[3][3], Real dg[3][3][3]) {
  for (int c=0;c<3;++c) {
    da[c]=c<ndim ? Dx<NG>(c,idx,adm.alpha,m,k,j,i) : 0.0;
    for (int a=0;a<3;++a) {
      db[c][a]=c<ndim ? Dx<NG>(c,idx,adm.beta_u,m,a,k,j,i) : 0.0;
      for (int b=0;b<3;++b)
        dg[c][a][b]=c<ndim ? Dx<NG>(c,idx,adm.g_dd,m,a,b,k,j,i) : 0.0;
    }
  }
}

template<class ADM>
KOKKOS_INLINE_FUNCTION void ValenciaGeometryDerivatives(
    int ng, const ADM &adm, int ndim, const Real idx[3], int m, int k, int j, int i,
    Real da[3], Real db[3][3], Real dg[3][3][3]) {
  switch (ng) {
    case 2: ValenciaGeometryDerivatives<2>(adm,ndim,idx,m,k,j,i,da,db,dg); break;
    case 3: ValenciaGeometryDerivatives<3>(adm,ndim,idx,m,k,j,i,da,db,dg); break;
    case 4: ValenciaGeometryDerivatives<4>(adm,ndim,idx,m,k,j,i,da,db,dg); break;
    default: Kokkos::abort("Unsupported Valencia derivative ghost width");
  }
}

// Apply each contraction in the evolution operator's existing order. The caller
// supplies the addition, allowing the evolution update to retain its arithmetic.
template<class Add>
KOKKOS_INLINE_FUNCTION void ValenciaMomentumSources(
    Real E, const Real S[3], const Real W[3][3], Real alpha,
    const Real da[3], const Real db[3][3], const Real dg[3][3][3],
    Real scale, Add add) {
  for (int a=0;a<3;++a) {
    for (int b=0;b<3;++b) {
      for (int c=0;c<3;++c) add(a,0.5*scale*alpha*W[b][c]*dg[a][b][c]);
      add(a,scale*S[b]*db[a][b]);
    }
    add(a,-scale*E*da[a]);
  }
}
#endif
