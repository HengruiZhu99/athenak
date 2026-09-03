//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c_bcs.cpp
//  \brief

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "pc_gh/pc_gh.hpp"
#include "z4c/z4c.hpp"

template<int order>
void BCHelper(MeshBlockPack *ppack, DualArray2D<Real> u_in, DvceArray5D<Real> u0,
              int is, int ie, int js, int je, int ks, int ke, int n1, int n2, int n3);

// A simple function for doing one-sided extrapolation.
// The off[xyz] variables control the direction of the extrapolation,
// and delta specifies how far to extrapolate to.
template<int order>
KOKKOS_INLINE_FUNCTION
Real Extrapolate(DvceArray5D<Real> u, const int m, const int n,
                 const int k, const int j, const int i,
                 const int offz, const int offy, const int offx,
                 const int delta);

// Return true when component n changes sign under reflection of coordinate axis.
// PC-GH stores vectors, symmetric tensors, and their first spatial derivatives;
// derivative indices contribute one additional odd factor in the reflected direction.
KOKKOS_INLINE_FUNCTION
constexpr bool ReflectOdd(const bool pcgh, const int n, const int axis) {
  if (!pcgh) {
    if (axis == 0) {
      return n == z4c::Z4c::I_Z4C_GXY || n == z4c::Z4c::I_Z4C_GXZ
          || n == z4c::Z4c::I_Z4C_AXY || n == z4c::Z4c::I_Z4C_AXZ
          || n == z4c::Z4c::I_Z4C_GAMX || n == z4c::Z4c::I_Z4C_BETAX;
    }
    if (axis == 1) {
      return n == z4c::Z4c::I_Z4C_GXY || n == z4c::Z4c::I_Z4C_GYZ
          || n == z4c::Z4c::I_Z4C_AXY || n == z4c::Z4c::I_Z4C_AYZ
          || n == z4c::Z4c::I_Z4C_GAMY || n == z4c::Z4c::I_Z4C_BETAY;
    }
    return n == z4c::Z4c::I_Z4C_GXZ || n == z4c::Z4c::I_Z4C_GYZ
        || n == z4c::Z4c::I_Z4C_AXZ || n == z4c::Z4c::I_Z4C_AYZ
        || n == z4c::Z4c::I_Z4C_GAMZ || n == z4c::Z4c::I_Z4C_BETAZ;
  }

  if ((n >= pc_gh::PcGh::I_ZX && n <= pc_gh::PcGh::I_ZZ)
      || (n >= pc_gh::PcGh::I_BETAX && n <= pc_gh::PcGh::I_BETAZ)
      || (n >= pc_gh::PcGh::I_P1 && n <= pc_gh::PcGh::I_P3)
      || (n >= pc_gh::PcGh::I_L1 && n <= pc_gh::PcGh::I_L3)) {
    int const first = (n <= pc_gh::PcGh::I_ZZ) ? pc_gh::PcGh::I_ZX
        : ((n <= pc_gh::PcGh::I_BETAZ) ? pc_gh::PcGh::I_BETAX
        : ((n <= pc_gh::PcGh::I_P3) ? pc_gh::PcGh::I_P1 : pc_gh::PcGh::I_L1));
    return n - first == axis;
  }

  int tensor_component = -1;
  int derivative_axis = -1;
  if (n >= pc_gh::PcGh::I_GTXX && n <= pc_gh::PcGh::I_GTZZ) {
    tensor_component = n - pc_gh::PcGh::I_GTXX;
  } else if (n >= pc_gh::PcGh::I_ATXX && n <= pc_gh::PcGh::I_ATZZ) {
    tensor_component = n - pc_gh::PcGh::I_ATXX;
  } else if (n >= pc_gh::PcGh::I_Q1XX && n <= pc_gh::PcGh::I_Q3ZZ) {
    int const offset = n - pc_gh::PcGh::I_Q1XX;
    derivative_axis = offset/6;
    tensor_component = offset % 6;
  }
  if (tensor_component >= 0) {
    int first_index = 0;
    int second_index = 0;
    if (tensor_component == 1) {
      second_index = 1;
    } else if (tensor_component == 2) {
      second_index = 2;
    } else if (tensor_component == 3) {
      first_index = second_index = 1;
    } else if (tensor_component == 4) {
      first_index = 1;
      second_index = 2;
    } else if (tensor_component == 5) {
      first_index = second_index = 2;
    }
    int reflected_indices = (first_index == axis) + (second_index == axis)
        + (derivative_axis == axis);
    return reflected_indices % 2 == 1;
  }

  if (n >= pc_gh::PcGh::I_B11 && n <= pc_gh::PcGh::I_B33) {
    int const offset = n - pc_gh::PcGh::I_B11;
    int const derivative = offset/3;
    int const vector_component = offset % 3;
    return ((derivative == axis) + (vector_component == axis)) == 1;
  }
  return false;
}

static_assert(!ReflectOdd(true, pc_gh::PcGh::I_W, 1));
static_assert(ReflectOdd(true, pc_gh::PcGh::I_BETAY, 1));
static_assert(ReflectOdd(true, pc_gh::PcGh::I_GTXY, 1));
static_assert(!ReflectOdd(true, pc_gh::PcGh::I_GTYY, 1));
static_assert(ReflectOdd(true, pc_gh::PcGh::I_Q2XX, 1));
static_assert(!ReflectOdd(true, pc_gh::PcGh::I_Q2XY, 1));
static_assert(ReflectOdd(true, pc_gh::PcGh::I_B12, 1));
static_assert(!ReflectOdd(true, pc_gh::PcGh::I_B22, 1));

// Linear extrapolation
template<>
KOKKOS_INLINE_FUNCTION
Real Extrapolate<2>(DvceArray5D<Real> u, const int m, const int n,
                    const int k, const int j, const int i,
                    const int offz, const int offy, const int offx,
                    const int delta) {
  Real f0 = u(m,n,k,j,i);
  Real f1 = u(m,n,k+offz,j+offy,i+offx);
  return f0 + (delta)*(f0 - f1);
}

// Quadratic extrapolation
template<>
KOKKOS_INLINE_FUNCTION
Real Extrapolate<3>(DvceArray5D<Real> u, const int m, const int n,
                    const int k, const int j, const int i,
                    const int offz, const int offy, const int offx,
                    const int delta) {
  Real f0 = u(m,n,k,j,i);
  Real f1 = u(m,n,k+offz,j+offy,i+offx);
  Real f2 = u(m,n,k+2*offz,j+2*offy,i+2*offx);
  return 0.5*(f0 * (1 + delta) * (2 + delta) +
              delta*(f2 + delta*f2 - 2*f1*(2 + delta)));
}

// Cubic extrapolation
template<>
KOKKOS_INLINE_FUNCTION
Real Extrapolate<4>(DvceArray5D<Real> u, const int m, const int n,
                    const int k, const int j, const int i,
                    const int offz, const int offy, const int offx,
                    const int delta) {
  Real f0 = u(m,n,k,j,i);
  Real f1 = u(m,n,k+offz,j+offy,i+offx);
  Real f2 = u(m,n,k+2*offz,j+2*offy,i+2*offx);
  Real f3 = u(m,n,k+3*offz,j+3*offy,i+3*offx);
  return (-3.0*f1*delta*(2 + delta)*(3 + delta) +
          f0*(1 + delta)*(2 + delta)*(3 + delta) +
          delta*(1 + delta)*(-f3*(2 + delta) + 3*f2*(3 + delta)))/6.0;
}

//----------------------------------------------------------------------------------------
// \!fn void MeshBoundaryValues::Z4cBCs()
// \brief Apply physical boundary conditions for all Z4c variables at faces of MB which
//  are at the edge of the computational domain
void MeshBoundaryValues::Z4cBCs(MeshBlockPack *ppack, DualArray2D<Real> u_in,
                                DvceArray5D<Real> u0, DvceArray5D<Real> coarse_u0) {
  auto &pm = ppack->pmesh;
  auto &indcs = ppack->pmesh->mb_indcs;
  int &ng = indcs.ng;

  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int is = indcs.is;
  int ie = indcs.ie;
  int js = indcs.js;
  int je = indcs.je;
  int ks = indcs.ks;
  int ke = indcs.ke;
  int const extrap_order = (ppack->pz4c != nullptr)
      ? ppack->pz4c->opt.extrap_order
      : ppack->ppcgh->opt.extrap_order;

  switch(extrap_order) {
    case 2:
      BCHelper<2>(ppack, u_in, u0, is, ie, js, je, ks, ke, n1, n2, n3);
      break;
    case 3:
      BCHelper<3>(ppack, u_in, u0, is, ie, js, je, ks, ke, n1, n2, n3);
      break;
    case 4:
      BCHelper<4>(ppack, u_in, u0, is, ie, js, je, ks, ke, n1, n2, n3);
      break;
  }
  if (pm->multilevel) {
    int cn1 = indcs.cnx1 + 2*ng;
    int cn2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*ng) : 1;
    int cn3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*ng) : 1;
    int cis = indcs.cis;
    int cie = indcs.cie;
    int cjs = indcs.cjs;
    int cje = indcs.cje;
    int cks = indcs.cks;
    int cke = indcs.cke;
    switch(extrap_order) {
      case 2:
        BCHelper<2>(ppack, u_in, coarse_u0, cis, cie, cjs, cje, cks, cke, cn1, cn2, cn3);
        break;
      case 3:
        BCHelper<3>(ppack, u_in, coarse_u0, cis, cie, cjs, cje, cks, cke, cn1, cn2, cn3);
        break;
      case 4:
        BCHelper<4>(ppack, u_in, coarse_u0, cis, cie, cjs, cje, cks, cke, cn1, cn2, cn3);
        break;
    }
  }
}

//void BoundaryValues::Z4cBCs(MeshBlockPack *ppack, DualArray2D<Real> u_in,
//                            DvceArray5D<Real> u0) {
template<int order>
void BCHelper(MeshBlockPack *ppack, DualArray2D<Real> u_in, DvceArray5D<Real> u0,
              int is, int ie, int js, int je, int ks, int ke, int n1, int n2, int n3) {
  // loop over all MeshBlocks in this MeshBlockPack
  auto &pm = ppack->pmesh;
  int &ng = ppack->pmesh->mb_indcs.ng;
  auto &mb_bcs = ppack->pmb->mb_bcs;

  int nvar = u0.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR
  int nmb = ppack->nmb_thispack;
  bool const pcgh = ppack->ppcgh != nullptr;

  // only apply BCs unless periodic or shear_periodic
  if (pm->mesh_bcs[BoundaryFace::inner_x1] != BoundaryFlag::periodic
      && pm->mesh_bcs[BoundaryFace::inner_x1] != BoundaryFlag::shear_periodic) {
    par_for("z4cbc_x1", DevExeSpace(), 0,(nmb-1),0,(nvar-1),0,(n3-1),0,(n2-1),
    KOKKOS_LAMBDA(int m, int n, int k, int j) {
      // apply physical boundaries to inner_x1
      switch (mb_bcs.d_view(m,BoundaryFace::inner_x1)) {
        case BoundaryFlag::reflect:
          for (int i=0; i<ng; ++i) {
            if (ReflectOdd(pcgh, n, 0)) {
              u0(m,n,k,j,is-i-1) = -u0(m,n,k,j,is+i);
            } else {
              u0(m,n,k,j,is-i-1) =  u0(m,n,k,j,is+i);
            }
          }
          break;
        case BoundaryFlag::diode:
        case BoundaryFlag::outflow:
        case BoundaryFlag::vacuum:
          for (int i=0; i<ng; ++i) {
            //u0(m,n,k,j,is-i-1) = u0(m,n,k,j,is);
            u0(m,n,k,j,is-i-1) = Extrapolate<order>(u0,m,n,k,j,is,0,0,1,i+1);
          }
          break;
        case BoundaryFlag::inflow:
          for (int i=0; i<ng; ++i) {
            u0(m,n,k,j,is-i-1) = u_in.d_view(n,BoundaryFace::inner_x1);
          }
          break;
        default:
          break;
      }

      // apply physical boundaries to outer_x1
      switch (mb_bcs.d_view(m,BoundaryFace::outer_x1)) {
        case BoundaryFlag::reflect:
          for (int i=0; i<ng; ++i) {
            if (ReflectOdd(pcgh, n, 0)) {
              u0(m,n,k,j,ie+i+1) = -u0(m,n,k,j,ie-i);
            } else {
              u0(m,n,k,j,ie+i+1) =  u0(m,n,k,j,ie-i);
            }
          }
          break;
        case BoundaryFlag::diode:
        case BoundaryFlag::outflow:
        case BoundaryFlag::vacuum:
          for (int i=0; i<ng; ++i) {
            //u0(m,n,k,j,ie+i+1) = u0(m,n,k,j,ie);
            u0(m,n,k,j,ie+i+1) = Extrapolate<order>(u0,m,n,k,j,ie,0,0,-1,i+1);
          }
          break;
        case BoundaryFlag::inflow:
          for (int i=0; i<ng; ++i) {
            u0(m,n,k,j,ie+i+1) = u_in.d_view(n,BoundaryFace::outer_x1);
          }
          break;
        default:
          break;
      }
    });
  }

  if (pm->one_d) return;

  // only apply BCs if not periodic
  if (pm->mesh_bcs[BoundaryFace::inner_x2] != BoundaryFlag::periodic) {
    par_for("z4cbc_x2", DevExeSpace(), 0,(nmb-1),0,(nvar-1),0,(n3-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int n, int k, int i) {
      // apply physical boundaries to inner_x2
      switch (mb_bcs.d_view(m,BoundaryFace::inner_x2)) {
        case BoundaryFlag::reflect:
          for (int j=0; j<ng; ++j) {
            if (ReflectOdd(pcgh, n, 1)) {
              u0(m,n,k,js-j-1,i) = -u0(m,n,k,js+j,i);
            } else {
              u0(m,n,k,js-j-1,i) =  u0(m,n,k,js+j,i);
            }
          }
          break;
        case BoundaryFlag::diode:
        case BoundaryFlag::outflow:
        case BoundaryFlag::vacuum:
          for (int j=0; j<ng; ++j) {
            //u0(m,n,k,js-j-1,i) = u0(m,n,k,js,i);
            u0(m,n,k,js-j-1,i) = Extrapolate<order>(u0,m,n,k,js,i,0,1,0,j+1);
          }
          break;
        case BoundaryFlag::inflow:
          for (int j=0; j<ng; ++j) {
            u0(m,n,k,js-j-1,i) = u_in.d_view(n,BoundaryFace::inner_x2);
          }
          break;
        default:
          break;
      }

      // apply physical boundaries to outer_x2
      switch (mb_bcs.d_view(m,BoundaryFace::outer_x2)) {
        case BoundaryFlag::reflect:
          for (int j=0; j<ng; ++j) {
            if (ReflectOdd(pcgh, n, 1)) {
              u0(m,n,k,je+j+1,i) = -u0(m,n,k,je-j,i);
            } else {
              u0(m,n,k,je+j+1,i) =  u0(m,n,k,je-j,i);
            }
          }
          break;
        case BoundaryFlag::diode:
        case BoundaryFlag::outflow:
        case BoundaryFlag::vacuum:
          for (int j=0; j<ng; ++j) {
            //u0(m,n,k,je+j+1,i) = u0(m,n,k,je,i);
            u0(m,n,k,je+j+1,i) = Extrapolate<order>(u0,m,n,k,je,i,0,-1,0,j+1);
          }
          break;
        case BoundaryFlag::inflow:
          for (int j=0; j<ng; ++j) {
            u0(m,n,k,je+j+1,i) = u_in.d_view(n,BoundaryFace::outer_x2);
          }
          break;
        default:
          break;
      }
    });
  }
  if (pm->two_d) return;

  // only apply BCs if not periodic
  if (pm->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::periodic) return;
  par_for("z4cbc_x3", DevExeSpace(), 0,(nmb-1),0,(nvar-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int n, int j, int i) {
    // apply physical boundaries to inner_x3
    switch (mb_bcs.d_view(m,BoundaryFace::inner_x3)) {
      case BoundaryFlag::reflect:
        for (int k=0; k<ng; ++k) {
          if (ReflectOdd(pcgh, n, 2)) {
            u0(m,n,ks-k-1,j,i) = -u0(m,n,ks+k,j,i);
          } else {
            u0(m,n,ks-k-1,j,i) =  u0(m,n,ks+k,j,i);
          }
        }
        break;
      case BoundaryFlag::diode:
      case BoundaryFlag::outflow:
      case BoundaryFlag::vacuum:
        for (int k=0; k<ng; ++k) {
          //u0(m,n,ks-k-1,j,i) = u0(m,n,ks,j,i);
          u0(m,n,ks-k-1,j,i) = Extrapolate<order>(u0,m,n,ks,j,i,1,0,0,k+1);
        }
        break;
      case BoundaryFlag::inflow:
        for (int k=0; k<ng; ++k) {
          u0(m,n,ks-k-1,j,i) = u_in.d_view(n,BoundaryFace::inner_x3);
        }
        break;
      default:
        break;
    }

    // apply physical boundaries to outer_x3
    switch (mb_bcs.d_view(m,BoundaryFace::outer_x3)) {
      case BoundaryFlag::reflect:
        for (int k=0; k<ng; ++k) {
          if (ReflectOdd(pcgh, n, 2)) {
            u0(m,n,ke+k+1,j,i) = -u0(m,n,ke-k,j,i);
          } else {
            u0(m,n,ke+k+1,j,i) =  u0(m,n,ke-k,j,i);
          }
        }
        break;
      case BoundaryFlag::diode:
      case BoundaryFlag::outflow:
      case BoundaryFlag::vacuum:
        for (int k=0; k<ng; ++k) {
          //u0(m,n,ke+k+1,j,i) = u0(m,n,ke,j,i);
          u0(m,n,ke+k+1,j,i) = Extrapolate<order>(u0,m,n,ke,j,i,-1,0,0,k+1);
        }
        break;
      case BoundaryFlag::inflow:
        for (int k=0; k<ng; ++k) {
          u0(m,n,ke+k+1,j,i) = u_in.d_view(n,BoundaryFace::outer_x3);
        }
        break;
      default:
        break;
    }
  });

  return;
}
