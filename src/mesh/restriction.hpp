#ifndef MESH_RESTRICTION_HPP_
#define MESH_RESTRICTION_HPP_

//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file restriction.hpp
//! \brief restriction operators for cell-centered variables,
//! implemented as templated inline functions so they can be used for z4c
//! with different order of spatial differencing order.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"

struct CoarseRestrictionRange {
  int lower;
  int upper;
};

enum class O4RestrictionStencil1D : int {
  centered = 0,
  active_lower = 1,
  active_upper = 2,
};

KOKKOS_INLINE_FUNCTION
O4RestrictionStencil1D SelectO4RestrictionStencil(const int fine_pair_start,
                                                   const int active_start,
                                                   const int active_extent) {
  if (fine_pair_start == active_start) {
    return O4RestrictionStencil1D::active_lower;
  }
  if (fine_pair_start == active_start + active_extent - 2) {
    return O4RestrictionStencil1D::active_upper;
  }
  return O4RestrictionStencil1D::centered;
}

KOKKOS_INLINE_FUNCTION
int O4RestrictionReference(const int fine_pair_start,
                           const O4RestrictionStencil1D stencil) {
  if (stencil == O4RestrictionStencil1D::active_lower) return fine_pair_start;
  if (stencil == O4RestrictionStencil1D::active_upper) return fine_pair_start - 2;
  return fine_pair_start - 1;
}

KOKKOS_INLINE_FUNCTION
Real O4RestrictionWeight(const O4RestrictionStencil1D stencil, const int n) {
  constexpr Real centered[4] = {
      -1.0 / 16.0, 9.0 / 16.0, 9.0 / 16.0, -1.0 / 16.0};
  constexpr Real lower[4] = {
      5.0 / 16.0, 15.0 / 16.0, -5.0 / 16.0, 1.0 / 16.0};
  constexpr Real upper[4] = {
      1.0 / 16.0, -5.0 / 16.0, 15.0 / 16.0, 5.0 / 16.0};
  if (stencil == O4RestrictionStencil1D::active_lower) return lower[n];
  if (stencil == O4RestrictionStencil1D::active_upper) return upper[n];
  return centered[n];
}

// Return only coarse indices whose associated pair of fine cells is stored.
// Same-level receive buffers can contain an odd number of fine ghost cells;
// converting both endpoints with integer division otherwise creates a coarse
// target whose fine-cell pair lies partly outside the allocation.
KOKKOS_INLINE_FUNCTION
CoarseRestrictionRange CompleteFinePairCoarseRange(const int lower, const int upper,
                                                    const int coarse_start,
                                                    const int fine_start,
                                                    const int fine_extent) {
  CoarseRestrictionRange range{lower, upper};
  while (range.lower <= range.upper &&
         (range.lower - coarse_start)*2 + fine_start < 0) {
    ++range.lower;
  }
  while (range.lower <= range.upper &&
         (range.upper - coarse_start)*2 + fine_start + 1 >= fine_extent) {
    --range.upper;
  }
  return range;
}

template <int NGHOST>
KOKKOS_INLINE_FUNCTION
Real RestrictInterpolation(const int m, const int v, const int fk, const int fj,
                          const int fi, const int nx1, const int nx2, const int nx3,
                          const DvceArray5D<Real> &a,
                          const DualArray1D<Real> &restrict_2nd,
                          const DualArray1D<Real> &restrict_4th,
                          const DualArray1D<Real> &restrict_4th_edge) {
  // interpolated value at new grid point
  Real ivals = 0;

  bool offseti = (fi<nx1/2+NGHOST);
  bool offsetj = (fj<nx2/2+NGHOST);

  // A collapsed x3 direction has one stored plane and no ghost layers.  Apply
  // the same order-matched rule as the Cartesian path in the two active
  // directions only.  This mirrors the collapsed-direction handling in
  // ProlongInterpolation and leaves the established 3-D tensor product below
  // unchanged.
  if (nx3 == 1) {
    if constexpr (NGHOST == 2) {
      const int refi = offseti ? fi : fi - 1;
      const int refj = offsetj ? fj : fj - 1;
      for (int jj = 0; jj < NGHOST + 1; ++jj) {
        for (int ii = 0; ii < NGHOST + 1; ++ii) {
          const int wghti = offseti ? ii : NGHOST - ii;
          const int wghtj = offsetj ? jj : NGHOST - jj;
          ivals += restrict_2nd.d_view(wghti) * restrict_2nd.d_view(wghtj) *
                   a(m, v, fk, refj + jj, refi + ii);
        }
      }
    } else if constexpr (NGHOST == 3) {
      // The O4 evolution may retain four allocated ghost layers while using
      // a three-deep PDE stencil.  Infer the allocated active start from the
      // actual View extent, not from the interpolation selector.  At the first
      // and last active sibling pairs use mirror-paired, cubic-exact stencils
      // containing current active values only.
      const int active_is = (a.extent_int(4) - nx1) / 2;
      const int active_js = (a.extent_int(3) - nx2) / 2;
      const auto stencil_i = SelectO4RestrictionStencil(fi, active_is, nx1);
      const auto stencil_j = SelectO4RestrictionStencil(fj, active_js, nx2);
      const int refi = O4RestrictionReference(fi, stencil_i);
      const int refj = O4RestrictionReference(fj, stencil_j);
      for (int jj = 0; jj < 4; ++jj) {
        for (int ii = 0; ii < 4; ++ii) {
          ivals += O4RestrictionWeight(stencil_i, ii) *
                   O4RestrictionWeight(stencil_j, jj) *
                   a(m, v, fk, refj + jj, refi + ii);
        }
      }
    } else {
      static_assert(NGHOST == 4,
                    "Z4c restriction supports nghost=2, 3, or 4");
      int refi = offseti ? fi - 1 : fi - 2;
      int refj = offsetj ? fj - 1 : fj - 2;
      const int outer_i = nx1 + 2 * NGHOST - 2;
      const int outer_j = nx2 + 2 * NGHOST - 2;
      const bool edge_i = (fi == 0 || fi == NGHOST ||
                           fi == NGHOST + nx1 - 2 || fi == outer_i);
      const bool edge_j = (fj == 0 || fj == NGHOST ||
                           fj == NGHOST + nx2 - 2 || fj == outer_j);
      refi = (fi == NGHOST) ? refi + 1 : refi;
      refj = (fj == NGHOST) ? refj + 1 : refj;
      refi = (fi == NGHOST + nx1 - 2) ? refi - 1 : refi;
      refj = (fj == NGHOST + nx2 - 2) ? refj - 1 : refj;
      refi = (fi == 0) ? 0 : refi;
      refj = (fj == 0) ? 0 : refj;
      refi = (fi == outer_i) ? nx1 + NGHOST - 1 : refi;
      refj = (fj == outer_j) ? nx2 + NGHOST - 1 : refj;
      for (int jj = 0; jj < NGHOST + 1; ++jj) {
        for (int ii = 0; ii < NGHOST + 1; ++ii) {
          const int wghti = offseti ? ii : NGHOST - ii;
          const int wghtj = offsetj ? jj : NGHOST - jj;
          const Real wi = edge_i ? restrict_4th_edge.d_view(wghti)
                                 : restrict_4th.d_view(wghti);
          const Real wj = edge_j ? restrict_4th_edge.d_view(wghtj)
                                 : restrict_4th.d_view(wghtj);
          ivals += wi * wj * a(m, v, fk, refj + jj, refi + ii);
        }
      }
    }
    return ivals;
  }

  bool offsetk = (fk<nx3/2+NGHOST);

  if (NGHOST ==2) {
    int refi = (offseti) ? fi : fi-1;
    int refj = (offsetj) ? fj : fj-1;
    int refk = (offsetk) ? fk : fk-1;
    for (int ii=0; ii<NGHOST+1; ii++) {
      for (int jj=0; jj<NGHOST+1; jj++) {
        for (int kk=0; kk<NGHOST+1; kk++) {
          int wghti = (offseti) ? ii : NGHOST-ii;
          int wghtj = (offsetj) ? jj : NGHOST-jj;
          int wghtk = (offsetk) ? kk : NGHOST-kk;
          Real iwght = restrict_2nd.d_view(wghti)
                       *restrict_2nd.d_view(wghtj)
                       *restrict_2nd.d_view(wghtk);
          ivals += iwght*a(m,v,refk+kk,refj+jj,refi+ii);
        }
      }
    }
  }

  if (NGHOST == 3) {
    const int active_is = (a.extent_int(4) - nx1) / 2;
    const int active_js = (a.extent_int(3) - nx2) / 2;
    const int active_ks = (a.extent_int(2) - nx3) / 2;
    const auto stencil_i = SelectO4RestrictionStencil(fi, active_is, nx1);
    const auto stencil_j = SelectO4RestrictionStencil(fj, active_js, nx2);
    const auto stencil_k = SelectO4RestrictionStencil(fk, active_ks, nx3);
    const int refi = O4RestrictionReference(fi, stencil_i);
    const int refj = O4RestrictionReference(fj, stencil_j);
    const int refk = O4RestrictionReference(fk, stencil_k);
    for (int ii = 0; ii < 4; ++ii) {
      for (int jj = 0; jj < 4; ++jj) {
        for (int kk = 0; kk < 4; ++kk) {
          ivals += O4RestrictionWeight(stencil_i, ii) *
                   O4RestrictionWeight(stencil_j, jj) *
                   O4RestrictionWeight(stencil_k, kk) *
                   a(m, v, refk + kk, refj + jj, refi + ii);
        }
      }
    }
  }

  if (NGHOST ==4) {
    int refi = (offseti) ? fi-1 : fi-2;
    int refj = (offsetj) ? fj-1 : fj-2;
    int refk = (offsetk) ? fk-1 : fk-2;

    const int outer_i = nx1 + 2*NGHOST - 2;
    const int outer_j = nx2 + 2*NGHOST - 2;
    const int outer_k = nx3 + 2*NGHOST - 2;
    const bool edge_i = (fi == 0 || fi == NGHOST ||
                         fi == NGHOST + nx1 - 2 || fi == outer_i);
    const bool edge_j = (fj == 0 || fj == NGHOST ||
                         fj == NGHOST + nx2 - 2 || fj == outer_j);
    const bool edge_k = (fk == 0 || fk == NGHOST ||
                         fk == NGHOST + nx3 - 2 || fk == outer_k);

    // edge cases
    refi = (fi==NGHOST) ? refi+1 : refi;
    refj = (fj==NGHOST) ? refj+1 : refj;
    refk = (fk==NGHOST) ? refk+1 : refk;

    refi = (fi==NGHOST+nx1-2) ? refi-1 : refi;
    refj = (fj==NGHOST+nx2-2) ? refj-1 : refj;
    refk = (fk==NGHOST+nx3-2) ? refk-1 : refk;

    // FillCoarseInBndryCC also restricts the outermost complete pair in a
    // stored same-level ghost band.  Use the existing one-sided fourth-order
    // rule there, oriented by offset{i,j,k}; a centered five-point stencil
    // would read one cell below zero or one beyond the stored allocation.
    refi = (fi == 0) ? 0 : refi;
    refj = (fj == 0) ? 0 : refj;
    refk = (fk == 0) ? 0 : refk;
    refi = (fi == outer_i) ? nx1 + NGHOST - 1 : refi;
    refj = (fj == outer_j) ? nx2 + NGHOST - 1 : refj;
    refk = (fk == outer_k) ? nx3 + NGHOST - 1 : refk;

    for (int ii=0; ii<NGHOST+1; ii++) {
      for (int jj=0; jj<NGHOST+1; jj++) {
        for (int kk=0; kk<NGHOST+1; kk++) {
          int wghti = (offseti) ? ii : NGHOST-ii;
          int wghtj = (offsetj) ? jj : NGHOST-jj;
          int wghtk = (offsetk) ? kk : NGHOST-kk;
          Real iwght = 1;
          if (edge_i) {
            iwght *= restrict_4th_edge.d_view(wghti);
          } else {
            iwght *= restrict_4th.d_view(wghti);
          }
          if (edge_j) {
            iwght *= restrict_4th_edge.d_view(wghtj);
          } else {
            iwght *= restrict_4th.d_view(wghtj);
          }
          if (edge_k) {
            iwght *= restrict_4th_edge.d_view(wghtk);
          } else {
            iwght *= restrict_4th.d_view(wghtk);
          }
          ivals += iwght*a(m,v,refk+kk,refj+jj,refi+ii);
        }
      }
    }
  }
  return ivals;
}
#endif // MESH_RESTRICTION_HPP_
