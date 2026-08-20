#ifndef MESH_PROLONGATION_HPP_
#define MESH_PROLONGATION_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file prolongation.hpp
//! \brief prolongation operators for cell-centered and face-centered variables,
//! implemented as inline functions so they can be used both in Bval and AMR functions.

#include "z4c/z4c.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProlongCC()
//! \brief 2nd-order (piecewise-linear) prolongation operator for cell-centered variables

KOKKOS_INLINE_FUNCTION
void ProlongCC(const int m, const int v, const int k, const int j, const int i,
               const int fk, const int fj, const int fi,
               const bool multi_d, const bool three_d,
               const DvceArray5D<Real> &ca, const DvceArray5D<Real> &a) {
  // calculate x1-gradient using the min-mod limiter
  Real dl = ca(m,v,k,j,i  ) - ca(m,v,k,j,i-1);
  Real dr = ca(m,v,k,j,i+1) - ca(m,v,k,j,i  );
  Real dvar1 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));

  // calculate x2-gradient using the min-mod limiter
  Real dvar2 = 0.0;
  if (multi_d) {
    dl = ca(m,v,k,j  ,i) - ca(m,v,k,j-1,i);
    dr = ca(m,v,k,j+1,i) - ca(m,v,k,j  ,i);
    dvar2 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
  }

  // calculate x1-gradient using the min-mod limiter
  Real dvar3 = 0.0;
  if (three_d) {
    dl = ca(m,v,k  ,j,i) - ca(m,v,k-1,j,i);
    dr = ca(m,v,k+1,j,i) - ca(m,v,k  ,j,i);
    dvar3 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
  }

  // interpolate to the finer grid
  a(m,v,fk,fj,fi  ) = ca(m,v,k,j,i) - dvar1 - dvar2 - dvar3;
  a(m,v,fk,fj,fi+1) = ca(m,v,k,j,i) + dvar1 - dvar2 - dvar3;
  if (multi_d) {
    a(m,v,fk,fj+1,fi  ) = ca(m,v,k,j,i) - dvar1 + dvar2 - dvar3;
    a(m,v,fk,fj+1,fi+1) = ca(m,v,k,j,i) + dvar1 + dvar2 - dvar3;
  }
  if (three_d) {
    a(m,v,fk+1,fj  ,fi  ) = ca(m,v,k,j,i) - dvar1 - dvar2 + dvar3;
    a(m,v,fk+1,fj  ,fi+1) = ca(m,v,k,j,i) + dvar1 - dvar2 + dvar3;
    a(m,v,fk+1,fj+1,fi  ) = ca(m,v,k,j,i) - dvar1 + dvar2 + dvar3;
    a(m,v,fk+1,fj+1,fi+1) = ca(m,v,k,j,i) + dvar1 + dvar2 + dvar3;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ProlongFCSharedX1Face()
//! \brief 2nd-order (piecewise-linear) prolongation operator for face-centered variables
//! on shared X1-faces between fine and coarse cells

KOKKOS_INLINE_FUNCTION
void ProlongFCSharedX1Face(const int m, const int k, const int j, const int i,
                   const int fk, const int fj, const int fi,
                   const bool multi_d, const bool three_d,
                   const DvceArray4D<Real> &cbx1f, const DvceArray4D<Real> &bx1f) {
  // Prolongate b.x1f (v=0) by interpolating in x2/x3
  Real dvar2 = 0.0;
  if (multi_d) {
    Real dl = cbx1f(m,k,j  ,i) - cbx1f(m,k,j-1,i);
    Real dr = cbx1f(m,k,j+1,i) - cbx1f(m,k,j  ,i);
    dvar2 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
  }

  Real dvar3 = 0.0;
  if (three_d) {
    Real dl = cbx1f(m,k  ,j,i) - cbx1f(m,k-1,j,i);
    Real dr = cbx1f(m,k+1,j,i) - cbx1f(m,k  ,j,i);
    dvar3 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
  }

  bx1f(m,fk,fj,fi) = cbx1f(m,k,j,i) - dvar2 - dvar3;
  if (multi_d) {
    bx1f(m,fk,fj+1,fi) = cbx1f(m,k,j,i) + dvar2 - dvar3;
  }
  if (three_d) {
    bx1f(m,fk+1,fj  ,fi) = cbx1f(m,k,j,i) - dvar2 + dvar3;
    bx1f(m,fk+1,fj+1,fi) = cbx1f(m,k,j,i) + dvar2 + dvar3;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ProlongFCSharedX2Face()
//! \brief 2nd-order (piecewise-linear) prolongation operator for face-centered variables
//! on shared X2-faces between fine and coarse cells

KOKKOS_INLINE_FUNCTION
void ProlongFCSharedX2Face(const int m, const int k, const int j, const int i,
                   const int fk, const int fj, const int fi,
                   const bool three_d,
                   const DvceArray4D<Real> &cbx2f, const DvceArray4D<Real> &bx2f) {
  // Prolongate b.x2f (v=1) by interpolating in x1/x3
  Real dl = cbx2f(m,k,j,i  ) - cbx2f(m,k,j,i-1);
  Real dr = cbx2f(m,k,j,i+1) - cbx2f(m,k,j,i  );
  Real dvar1 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));

  Real dvar3 = 0.0;
  if (three_d) {
    dl = cbx2f(m,k  ,j,i) - cbx2f(m,k-1,j,i);
    dr = cbx2f(m,k+1,j,i) - cbx2f(m,k  ,j,i);
    dvar3 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
  }

  bx2f(m,fk  ,fj,fi  ) = cbx2f(m,k,j,i) - dvar1 - dvar3;
  bx2f(m,fk  ,fj,fi+1) = cbx2f(m,k,j,i) + dvar1 - dvar3;
  if (three_d) {
    bx2f(m,fk+1,fj,fi  ) = cbx2f(m,k,j,i) - dvar1 + dvar3;
    bx2f(m,fk+1,fj,fi+1) = cbx2f(m,k,j,i) + dvar1 + dvar3;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ProlongFCSharedX3Face()
//! \brief 2nd-order (piecewise-linear) prolongation operator for face-centered variables
//! on shared X3-faces between fine and coarse cells

KOKKOS_INLINE_FUNCTION
void ProlongFCSharedX3Face(const int m, const int k, const int j, const int i,
                   const int fk, const int fj, const int fi,
                   const bool multi_d,
                   const DvceArray4D<Real> &cbx3f, const DvceArray4D<Real> &bx3f) {
  // Prolongate b.x3f (v=2) by interpolating in x1/x2
  Real dl = cbx3f(m,k,j,i  ) - cbx3f(m,k,j,i-1);
  Real dr = cbx3f(m,k,j,i+1) - cbx3f(m,k,j,i  );
  Real dvar1 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));

  Real dvar2 = 0.0;
  if (multi_d) {
    dl = cbx3f(m,k,j  ,i) - cbx3f(m,k,j-1,i);
    dr = cbx3f(m,k,j+1,i) - cbx3f(m,k,j  ,i);
    dvar2 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
  }

  bx3f(m,fk,fj  ,fi  ) = cbx3f(m,k,j,i) - dvar1 - dvar2;
  bx3f(m,fk,fj  ,fi+1) = cbx3f(m,k,j,i) + dvar1 - dvar2;
  if (multi_d) {
    bx3f(m,fk,fj+1,fi  ) = cbx3f(m,k,j,i) - dvar1 + dvar2;
    bx3f(m,fk,fj+1,fi+1) = cbx3f(m,k,j,i) + dvar1 + dvar2;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ProlongInternalFC()
//! \brief 2nd-order prolongation operator for face-centered variables on internal edges
//! of new fine cells within one coarse cell using divergence-preserving interpolation
//! scheme of Toth & Roe, JCP 180, 736 (2002).

KOKKOS_INLINE_FUNCTION
void ProlongFCInternal(const int m, const int fk, const int fj, const int fi,
                       const bool three_d, const DvceFaceFld4D<Real> &b) {
  // Prolongate internal fields in 3D
  if (three_d) {
    Real Uxx  = 0.0, Vyy  = 0.0, Wzz  = 0.0;
    Real Uxyz = 0.0, Vxyz = 0.0, Wxyz = 0.0;
    for (int jj=0; jj<2; jj++) {
      int jsgn = 2*jj - 1;
      int fjj  = fj + jj, fjp = fj + 2*jj;
      for (int ii=0; ii<2; ii++) {
        int isgn = 2*ii - 1;
        int fii = fi + ii, fip = fi + 2*ii;
        Uxx += isgn*(jsgn*(b.x2f(m,fk  ,fjp,fii) + b.x2f(m,fk+1,fjp,fii)) +
                          (b.x3f(m,fk+2,fjj,fii) - b.x3f(m,fk  ,fjj,fii)));

        Vyy += jsgn*(     (b.x3f(m,fk+2,fjj,fii) - b.x3f(m,fk  ,fjj,fii)) +
                     isgn*(b.x1f(m,fk  ,fjj,fip) + b.x1f(m,fk+1,fjj,fip)));

        Wzz +=       isgn*(b.x1f(m,fk+1,fjj,fip) - b.x1f(m,fk  ,fjj,fip)) +
                     jsgn*(b.x2f(m,fk+1,fjp,fii) - b.x2f(m,fk  ,fjp,fii));

        Uxyz += isgn*jsgn*(b.x1f(m,fk+1,fjj,fip) - b.x1f(m,fk  ,fjj,fip));
        Vxyz += isgn*jsgn*(b.x2f(m,fk+1,fjp,fii) - b.x2f(m,fk  ,fjp,fii));
        Wxyz += isgn*jsgn*(b.x3f(m,fk+2,fjj,fii) - b.x3f(m,fk  ,fjj,fii));
      }
    }
    Uxx *= 0.125;  Vyy *= 0.125;  Wzz *= 0.125;
    Uxyz *= 0.0625; Vxyz *= 0.0625; Wxyz *= 0.0625;

    b.x1f(m,fk  ,fj  ,fi+1) = 0.5*(b.x1f(m,fk  ,fj  ,fi  ) + b.x1f(m,fk  ,fj  ,fi+2))
                            + Uxx - Vxyz - Wxyz;
    b.x1f(m,fk  ,fj+1,fi+1) = 0.5*(b.x1f(m,fk  ,fj+1,fi  ) + b.x1f(m,fk  ,fj+1,fi+2))
                            + Uxx - Vxyz + Wxyz;
    b.x1f(m,fk+1,fj  ,fi+1) = 0.5*(b.x1f(m,fk+1,fj  ,fi  ) + b.x1f(m,fk+1,fj  ,fi+2))
                            + Uxx + Vxyz - Wxyz;
    b.x1f(m,fk+1,fj+1,fi+1) = 0.5*(b.x1f(m,fk+1,fj+1,fi  ) + b.x1f(m,fk+1,fj+1,fi+2))
                            + Uxx + Vxyz + Wxyz;
    b.x2f(m,fk  ,fj+1,fi  ) = 0.5*(b.x2f(m,fk  ,fj  ,fi  ) + b.x2f(m,fk  ,fj+2,fi  ))
                            + Vyy - Uxyz - Wxyz;
    b.x2f(m,fk  ,fj+1,fi+1) = 0.5*(b.x2f(m,fk  ,fj  ,fi+1) + b.x2f(m,fk  ,fj+2,fi+1))
                            + Vyy - Uxyz + Wxyz;
    b.x2f(m,fk+1,fj+1,fi  ) = 0.5*(b.x2f(m,fk+1,fj  ,fi  ) + b.x2f(m,fk+1,fj+2,fi  ))
                            + Vyy + Uxyz - Wxyz;
    b.x2f(m,fk+1,fj+1,fi+1) = 0.5*(b.x2f(m,fk+1,fj  ,fi+1) + b.x2f(m,fk+1,fj+2,fi+1))
                            + Vyy + Uxyz + Wxyz;
    b.x3f(m,fk+1,fj  ,fi  ) = 0.5*(b.x3f(m,fk+2,fj  ,fi  ) + b.x3f(m,fk  ,fj  ,fi  ))
                            + Wzz - Uxyz - Vxyz;
    b.x3f(m,fk+1,fj  ,fi+1) = 0.5*(b.x3f(m,fk+2,fj  ,fi+1) + b.x3f(m,fk  ,fj  ,fi+1))
                            + Wzz - Uxyz + Vxyz;
    b.x3f(m,fk+1,fj+1,fi  ) = 0.5*(b.x3f(m,fk+2,fj+1,fi  ) + b.x3f(m,fk  ,fj+1,fi  ))
                            + Wzz + Uxyz - Vxyz;
    b.x3f(m,fk+1,fj+1,fi+1) = 0.5*(b.x3f(m,fk+2,fj+1,fi+1) + b.x3f(m,fk  ,fj+1,fi+1))
                            + Wzz + Uxyz + Vxyz;

  // Prolongate internal fields in 2D
  } else {
    Real tmp1 = 0.25*(b.x2f(m,fk,fj+2,fi+1) - b.x2f(m,fk,fj,  fi+1)
                    - b.x2f(m,fk,fj+2,fi  ) + b.x2f(m,fk,fj,  fi  ));
    Real tmp2 = 0.25*(b.x1f(m,fk,fj,  fi  ) - b.x1f(m,fk,fj,  fi+2)
                    - b.x1f(m,fk,fj+1,fi  ) + b.x1f(m,fk,fj+1,fi+2));
    b.x1f(m,fk,fj  ,fi+1) = 0.5*(b.x1f(m,fk,fj,  fi  ) + b.x1f(m,fk,fj,  fi+2)) + tmp1;
    b.x1f(m,fk,fj+1,fi+1) = 0.5*(b.x1f(m,fk,fj+1,fi  ) + b.x1f(m,fk,fj+1,fi+2)) + tmp1;
    b.x2f(m,fk,fj+1,fi  ) = 0.5*(b.x2f(m,fk,fj,  fi  ) + b.x2f(m,fk,fj+2,fi  )) + tmp2;
    b.x2f(m,fk,fj+1,fi+1) = 0.5*(b.x2f(m,fk,fj,  fi+1) + b.x2f(m,fk,fj+2,fi+1)) + tmp2;
  }
  return;
}

template <int NGHOST>
KOKKOS_INLINE_FUNCTION
Real ProlongWeight1D(const int n, const bool offset) {
  if constexpr (NGHOST == 2) {
    constexpr Real w[3] = {0.15625, 0.9375, -0.09375};
    return w[offset ? NGHOST-n : n];
  } else if constexpr (NGHOST == 3) {
    // Reflection-symmetric cubic interpolation.  The left child uses coarse
    // cells i-2,...,i+1 and the right child uses i-1,...,i+2; the reference
    // shift is applied in ProlongInterpolation below.
    constexpr Real wl[4] = {-5.0/128.0, 35.0/128.0, 105.0/128.0, -7.0/128.0};
    constexpr Real wr[4] = {-7.0/128.0, 105.0/128.0, 35.0/128.0, -5.0/128.0};
    return offset ? wr[n] : wl[n];
  } else {
    static_assert(NGHOST == 4, "Z4c prolongation supports nghost=2, 3, or 4");
    constexpr Real w[5] = {-0.02197265625, 0.205078125, 0.9228515625,
                           -0.123046875, 0.01708984375};
    return w[offset ? NGHOST-n : n];
  }
}

template <int NGHOST>
KOKKOS_INLINE_FUNCTION
Real ProlongInterpolation(const int m, const int v, int k, int j, int i,
                            const int nx1, const int nx2, const int nx3,
                            const bool offsetk, const bool offsetj, const bool offseti,
                        const DvceArray5D<Real> &ca, const DualArray3D<Real> &weights) {
  // interpolated value at new grid point
  Real ivals = 0;

  // A collapsed direction has one stored cell and no ghost layers.  Interpolate only
  // in the active plane; this is the 2-D tensor product of the same 1-D rule.  The
  // established 3-D coefficient tables remain the Cartesian path for NGHOST=2 and 4.
  const bool collapsed_x3 = (nx3 == 1);
  const int nk = collapsed_x3 ? 1 : NGHOST+1;
  for (int kk=0; kk<nk; kk++) {
    for (int jj=0; jj<NGHOST+1; jj++) {
      for (int ii=0; ii<NGHOST+1; ii++) {
        int wghti = (offseti) ? NGHOST-ii : ii;
        int wghtj = (offsetj) ? NGHOST-jj : jj;
        int wghtk = (offsetk) ? NGHOST-kk : kk;
        Real weight;
        if constexpr (NGHOST == 3) {
          weight = ProlongWeight1D<NGHOST>(ii, offseti) *
                   ProlongWeight1D<NGHOST>(jj, offsetj);
          if (!collapsed_x3) weight *= ProlongWeight1D<NGHOST>(kk, offsetk);
        } else if (collapsed_x3) {
          weight = ProlongWeight1D<NGHOST>(ii, offseti) *
                   ProlongWeight1D<NGHOST>(jj, offsetj);
        } else {
          weight = weights.d_view(wghtk,wghtj,wghti);
        }
        // For the O4-configured cubic pair, shift only the left-child stencil
        // one parent farther left.  The two child stencils are then exact
        // reflections about the coarse-cell center.  O2/O6 references remain
        // byte-for-byte equivalent to the historical expression.
        const int o4_left_k = (NGHOST == 3 && !offsetk && !collapsed_x3) ? 1 : 0;
        const int o4_left_j = (NGHOST == 3 && !offsetj) ? 1 : 0;
        const int o4_left_i = (NGHOST == 3 && !offseti) ? 1 : 0;
        const int ck = collapsed_x3 ? k : k-NGHOST/2+kk-o4_left_k;
        ivals += weight*ca(m,v,ck,j-NGHOST/2+jj-o4_left_j,
                          i-NGHOST/2+ii-o4_left_i);
      }
    }
  }

  return ivals;
}

//----------------------------------------------------------------------------------------
//! \fn HighOrderProlongCC()
//! \brief high-order prolongation operator for cell-centered variables

template <int NGHOST>
KOKKOS_INLINE_FUNCTION
void HighOrderProlongCC(const int m, const int v, const int k, const int j, const int i,
               const int fk, const int fj, const int fi, const int nx1, const int nx2,
               const int nx3, const DvceArray5D<Real> &ca, const DvceArray5D<Real> &a,
               const DualArray3D<Real> &weights) {
  // stencil size for interpolator
  a(m,v,fk  ,fj  ,fi  ) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1, nx2, nx3,
                                                        false,false,false, ca, weights);
  a(m,v,fk  ,fj  ,fi+1) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1, nx2, nx3,
                                                        false,false, true, ca, weights);
  a(m,v,fk  ,fj+1,fi  ) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1, nx2, nx3,
                                                        false, true,false, ca, weights);
  a(m,v,fk  ,fj+1,fi+1) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1, nx2, nx3,
                                                        false, true, true, ca, weights);
  if (nx3 > 1) {
    a(m,v,fk+1,fj  ,fi  ) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1,nx2,nx3,
                                                           true,false,false, ca, weights);
    a(m,v,fk+1,fj  ,fi+1) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1,nx2,nx3,
                                                           true,false, true, ca, weights);
    a(m,v,fk+1,fj+1,fi  ) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1,nx2,nx3,
                                                           true, true,false, ca, weights);
    a(m,v,fk+1,fj+1,fi+1) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1,nx2,nx3,
                                                           true, true, true, ca, weights);
  }
  return;
}

enum class ChiProlongationStatus : int {
  high_order = 0,
  limited = 1,
  invalid_parent = 2,
  invalid_limited = 3,
};

template <int NGHOST>
KOKKOS_INLINE_FUNCTION
bool ProlongationParentStencilFinitePositive(
    const int m, const int v, const int k, const int j, const int i,
    const int nx3, const DvceArray5D<Real> &ca) {
  const bool collapsed_x3 = (nx3 == 1);
  // The symmetric O4 child pair consumes the union i-2,...,i+2 in every
  // active direction.  The strict chi gate must validate that union, not only
  // either four-point child stencil in isolation.  O2/O6 retain their exact
  // historical parent inventory.
  const int lower = NGHOST == 3 ? -2 : -NGHOST / 2;
  const int upper = NGHOST == 3 ? 2 : NGHOST - NGHOST / 2;
  const int k_lower = collapsed_x3 ? 0 : lower;
  const int k_upper = collapsed_x3 ? 0 : upper;
  for (int dk = k_lower; dk <= k_upper; ++dk) {
    const int ck = collapsed_x3 ? k : k + dk;
    for (int dj = lower; dj <= upper; ++dj) {
      for (int di = lower; di <= upper; ++di) {
        const Real parent = ca(m, v, ck, j + dj, i + di);
        if (!Kokkos::isfinite(parent) || !(parent > 0.0)) return false;
      }
    }
  }
  return true;
}

KOKKOS_INLINE_FUNCTION
int ProlongationSiblingCount(const bool multi_d, const bool three_d) {
  return 2 * (multi_d ? 2 : 1) * (three_d ? 2 : 1);
}

KOKKOS_INLINE_FUNCTION
bool ProlongationSiblingGroupFinitePositive(
    const int m, const int v, const int fk, const int fj, const int fi,
    const bool multi_d, const bool three_d, const DvceArray5D<Real> &a) {
  const int nk = three_d ? 2 : 1;
  const int nj = multi_d ? 2 : 1;
  for (int dk = 0; dk < nk; ++dk) {
    for (int dj = 0; dj < nj; ++dj) {
      for (int di = 0; di < 2; ++di) {
        const Real child = a(m, v, fk + dk, fj + dj, fi + di);
        if (!Kokkos::isfinite(child) || !(child > 0.0)) return false;
      }
    }
  }
  return true;
}

//! \brief Validate the complete parent neighborhood consumed by ProlongCC.
KOKKOS_INLINE_FUNCTION
bool LimitedProlongationParentNeighborhoodFinitePositive(
    const int m, const int v, const int k, const int j, const int i,
    const bool multi_d, const bool three_d, const DvceArray5D<Real> &ca) {
  const int dk_lo = three_d ? -1 : 0;
  const int dk_hi = three_d ? 1 : 0;
  const int dj_lo = multi_d ? -1 : 0;
  const int dj_hi = multi_d ? 1 : 0;
  for (int dk = dk_lo; dk <= dk_hi; ++dk) {
    for (int dj = dj_lo; dj <= dj_hi; ++dj) {
      for (int di = -1; di <= 1; ++di) {
        const Real parent = ca(m, v, k + dk, j + dj, i + di);
        if (!Kokkos::isfinite(parent) || !(parent > 0.0)) return false;
      }
    }
  }
  return true;
}

//! \brief Always-limited O2 chi prolongation with unchanged strict positivity gates.
//!
//! This is the chi member of the diagnostic limited-O2 AMR transfer pair.  It is
//! not a floor or clip: invalid parent data and invalid reconstructed sibling groups
//! remain fatal at the existing caller gate.
KOKKOS_INLINE_FUNCTION
ChiProlongationStatus ProlongLimitedPositiveChiCC(
    const int m, const int v, const int k, const int j, const int i,
    const int fk, const int fj, const int fi, const bool multi_d,
    const bool three_d, const DvceArray5D<Real> &ca,
    const DvceArray5D<Real> &a) {
  if (!LimitedProlongationParentNeighborhoodFinitePositive(
          m, v, k, j, i, multi_d, three_d, ca)) {
    return ChiProlongationStatus::invalid_parent;
  }
  ProlongCC(m, v, k, j, i, fk, fj, fi, multi_d, three_d, ca, a);
  if (!ProlongationSiblingGroupFinitePositive(m, v, fk, fj, fi, multi_d,
                                              three_d, a)) {
    return ChiProlongationStatus::invalid_limited;
  }
  return ChiProlongationStatus::limited;
}

//! \brief Positivity-preserving Z4c chi prolongation for one complete child group.
//!
//! The existing high-order candidate is always generated first.  Acceptance also
//! requires every coarse value consumed by that interpolation to be finite and strictly
//! positive.  A rejected high-order group is replaced in full by the existing minmod
//! prolongation; no child is clipped or floored independently.
template <int NGHOST>
KOKKOS_INLINE_FUNCTION
ChiProlongationStatus ProlongPositiveChiCC(
    const int m, const int v, const int k, const int j, const int i,
    const int fk, const int fj, const int fi, const int nx1, const int nx2,
    const int nx3, const bool multi_d, const bool three_d,
    const DvceArray5D<Real> &ca, const DvceArray5D<Real> &a,
    const DualArray3D<Real> &weights) {
  HighOrderProlongCC<NGHOST>(m, v, k, j, i, fk, fj, fi, nx1, nx2, nx3,
                             ca, a, weights);
  if (!ProlongationParentStencilFinitePositive<NGHOST>(m, v, k, j, i, nx3,
                                                        ca)) {
    return ChiProlongationStatus::invalid_parent;
  }
  if (ProlongationSiblingGroupFinitePositive(m, v, fk, fj, fi, multi_d,
                                              three_d, a)) {
    return ChiProlongationStatus::high_order;
  }

  ProlongCC(m, v, k, j, i, fk, fj, fi, multi_d, three_d, ca, a);
  if (!ProlongationSiblingGroupFinitePositive(m, v, fk, fj, fi, multi_d,
                                              three_d, a)) {
    return ChiProlongationStatus::invalid_limited;
  }
  return ChiProlongationStatus::limited;
}

#endif // MESH_PROLONGATION_HPP_
