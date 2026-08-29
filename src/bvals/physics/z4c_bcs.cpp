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
#include "z4c/cartoon_axis_boundary.hpp"
#include "z4c/cartoon_vertex_axis.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_symmetry.hpp"

namespace {

void ValidateAxisBoundaryContract(MeshBlockPack *ppack,
                                  const DvceArray5D<Real> &state) {
  const auto &bcs = ppack->pmesh->mesh_bcs;
  const bool any_axis =
      bcs[BoundaryFace::inner_x1] == BoundaryFlag::axis ||
      bcs[BoundaryFace::outer_x1] == BoundaryFlag::axis ||
      bcs[BoundaryFace::inner_x2] == BoundaryFlag::axis ||
      bcs[BoundaryFace::outer_x2] == BoundaryFlag::axis ||
      bcs[BoundaryFace::inner_x3] == BoundaryFlag::axis ||
      bcs[BoundaryFace::outer_x3] == BoundaryFlag::axis;
  if (!any_axis) return;

  const auto &config = ppack->z4c_symmetry;
  const bool exact_half_plane_axis =
      config.mode == z4c::Z4cSymmetryMode::cartoon_so2 &&
      config.coordinate_map == z4c::Z4cCoordinateMap::half_rho_z_suppressed_y_v2 &&
      config.schema == z4c::Z4cSymmetryConfig::kHalfPlaneCartoonSchema &&
      bcs[BoundaryFace::inner_x1] == BoundaryFlag::axis &&
      bcs[BoundaryFace::outer_x1] != BoundaryFlag::axis &&
      bcs[BoundaryFace::inner_x2] != BoundaryFlag::axis &&
      bcs[BoundaryFace::outer_x2] != BoundaryFlag::axis &&
      bcs[BoundaryFace::inner_x3] != BoundaryFlag::axis &&
      bcs[BoundaryFace::outer_x3] != BoundaryFlag::axis &&
      state.extent_int(1) == z4c::Z4c::nz4c;
  if (!exact_half_plane_axis) {
    std::cerr << "### FATAL ERROR in " << __FILE__
              << ": boundary type axis requires the schema-2 half-plane Cartoon "
                 "Z4c state at inner_x1 only"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

}  // namespace

template<int order>
void BCHelper(MeshBlockPack *ppack, DualArray2D<Real> u_in, DvceArray5D<Real> u0,
              int is, int ie, int js, int je, int ks, int ke, int n1, int n2,
              int n3, int ghost_width);

// A simple function for doing one-sided extrapolation.
// The off[xyz] variables control the direction of the extrapolation,
// and delta specifies how far to extrapolate to.
template<int order>
KOKKOS_INLINE_FUNCTION
Real Extrapolate(DvceArray5D<Real> u, const int m, const int n,
                 const int k, const int j, const int i,
                 const int offz, const int offy, const int offx,
                 const int delta);

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
  const auto &layout = ppack->pz4c->layout;

  int n1 = layout.n1;
  int n2 = layout.n2;
  int n3 = layout.n3;
  int is = layout.is;
  int ie = layout.ie;
  int js = layout.js;
  int je = layout.je;
  int ks = layout.ks;
  int ke = layout.ke;
  auto &opt = ppack->pz4c->opt;
  ValidateAxisBoundaryContract(ppack, u0);

  switch(opt.extrap_order) {
    case 2:
      BCHelper<2>(ppack, u_in, u0, is, ie, js, je, ks, ke, n1, n2, n3,
                  layout.ng);
      break;
    case 3:
      BCHelper<3>(ppack, u_in, u0, is, ie, js, je, ks, ke, n1, n2, n3,
                  layout.ng);
      break;
    case 4:
      BCHelper<4>(ppack, u_in, u0, is, ie, js, je, ks, ke, n1, n2, n3,
                  layout.ng);
      break;
  }
  if (pm->multilevel) {
    int cn1 = layout.cn1;
    int cn2 = layout.cn2;
    int cn3 = layout.cn3;
    int cis = layout.cis;
    int cie = layout.cie;
    int cjs = layout.cjs;
    int cje = layout.cje;
    int cks = layout.cks;
    int cke = layout.cke;
    switch(opt.extrap_order) {
      case 2:
        BCHelper<2>(ppack, u_in, coarse_u0, cis, cie, cjs, cje, cks, cke,
                    cn1, cn2, cn3, layout.coarse_ng);
        break;
      case 3:
        BCHelper<3>(ppack, u_in, coarse_u0, cis, cie, cjs, cje, cks, cke,
                    cn1, cn2, cn3, layout.coarse_ng);
        break;
      case 4:
        BCHelper<4>(ppack, u_in, coarse_u0, cis, cie, cjs, cje, cks, cke,
                    cn1, cn2, cn3, layout.coarse_ng);
        break;
    }
  }
}

//void BoundaryValues::Z4cBCs(MeshBlockPack *ppack, DualArray2D<Real> u_in,
//                            DvceArray5D<Real> u0) {
template<int order>
void BCHelper(MeshBlockPack *ppack, DualArray2D<Real> u_in, DvceArray5D<Real> u0,
              int is, int ie, int js, int je, int ks, int ke, int n1, int n2,
              int n3, int ghost_width) {
  // loop over all MeshBlocks in this MeshBlockPack
  auto &pm = ppack->pmesh;
  const int ng = ghost_width;
  auto &mb_bcs = ppack->pmb->mb_bcs;

  int nvar = u0.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR
  int nmb = ppack->nmb_thispack;
  const bool vertex = ppack->pz4c->layout.centering ==
                      z4c::Z4cGridCentering::vertex;
  const bool compact_work = vertex && ppack->pz4c->opt.lean_runtime;

  // only apply BCs unless periodic or shear_periodic
  if (pm->mesh_bcs[BoundaryFace::inner_x1] != BoundaryFlag::periodic
      && pm->mesh_bcs[BoundaryFace::inner_x1] != BoundaryFlag::shear_periodic) {
    const auto boundary_work = ppack->pz4c->physical_boundary_x1_work.d_view;
    const int work_count = compact_work
        ? ppack->pz4c->nphysical_boundary_x1_work : nmb;
    par_for("z4cbc_x1", DevExeSpace(), 0,(work_count-1),0,(nvar-1),0,(n3-1),0,(n2-1),
    KOKKOS_LAMBDA(int row, int n, int k, int j) {
      const int m = compact_work ? boundary_work(row) : row;
      // apply physical boundaries to inner_x1
      switch (mb_bcs.d_view(m,BoundaryFace::inner_x1)) {
        case BoundaryFlag::axis:
          if (!(vertex
                    ? z4c::FillCenteredZ4cAxisGhostLine<z4c::VertexCenteredZ4c>(
                          u0, m, n, k, j, is, ng)
                    : z4c::FillZ4cAxisGhostLine(u0, m, n, k, j, is, ng))) {
            Kokkos::abort("invalid packed Z4c component in axis parity fill");
          }
          break;
        case BoundaryFlag::reflect:
          for (int i=0; i<ng; ++i) {
            if (n==z4c::Z4c::I_Z4C_GXY || n==z4c::Z4c::I_Z4C_GXZ ||
                n==z4c::Z4c::I_Z4C_AXY || n==z4c::Z4c::I_Z4C_AXZ ||
                n==z4c::Z4c::I_Z4C_GAMX || n==z4c::Z4c::I_Z4C_BETAX) {
              u0(m,n,k,j,is-i-1) = -u0(m,n,k,j,is+i+(vertex ? 1 : 0));
            } else {
              u0(m,n,k,j,is-i-1) =  u0(m,n,k,j,is+i+(vertex ? 1 : 0));
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
            if (n==z4c::Z4c::I_Z4C_GXY || n==z4c::Z4c::I_Z4C_GXZ ||
                n==z4c::Z4c::I_Z4C_AXY || n==z4c::Z4c::I_Z4C_AXZ ||
                n==z4c::Z4c::I_Z4C_GAMX || n==z4c::Z4c::I_Z4C_BETAX) {
              u0(m,n,k,j,ie+i+1) = -u0(m,n,k,j,ie-i-(vertex ? 1 : 0));
            } else {
              u0(m,n,k,j,ie+i+1) =  u0(m,n,k,j,ie-i-(vertex ? 1 : 0));
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
    const auto boundary_work = ppack->pz4c->physical_boundary_x2_work.d_view;
    const int work_count = compact_work
        ? ppack->pz4c->nphysical_boundary_x2_work : nmb;
    par_for("z4cbc_x2", DevExeSpace(), 0,(work_count-1),0,(nvar-1),0,(n3-1),0,(n1-1),
    KOKKOS_LAMBDA(int row, int n, int k, int i) {
      const int m = compact_work ? boundary_work(row) : row;
      // apply physical boundaries to inner_x2
      switch (mb_bcs.d_view(m,BoundaryFace::inner_x2)) {
        case BoundaryFlag::reflect:
          for (int j=0; j<ng; ++j) {
            if (n==z4c::Z4c::I_Z4C_GXY || n==z4c::Z4c::I_Z4C_GYZ ||
                n==z4c::Z4c::I_Z4C_AXY || n==z4c::Z4c::I_Z4C_AYZ ||
                n==z4c::Z4c::I_Z4C_GAMY || n==z4c::Z4c::I_Z4C_BETAY) {
              u0(m,n,k,js-j-1,i) = -u0(m,n,k,js+j+(vertex ? 1 : 0),i);
            } else {
              u0(m,n,k,js-j-1,i) =  u0(m,n,k,js+j+(vertex ? 1 : 0),i);
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
            if (n==z4c::Z4c::I_Z4C_GXY || n==z4c::Z4c::I_Z4C_GYZ ||
                n==z4c::Z4c::I_Z4C_AXY || n==z4c::Z4c::I_Z4C_AYZ ||
                n==z4c::Z4c::I_Z4C_GAMY || n==z4c::Z4c::I_Z4C_BETAY) {
              u0(m,n,k,je+j+1,i) = -u0(m,n,k,je-j-(vertex ? 1 : 0),i);
            } else {
              u0(m,n,k,je+j+1,i) =  u0(m,n,k,je-j-(vertex ? 1 : 0),i);
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
  const auto boundary_work = ppack->pz4c->physical_boundary_x3_work.d_view;
  const int work_count = compact_work
      ? ppack->pz4c->nphysical_boundary_x3_work : nmb;
  par_for("z4cbc_x3", DevExeSpace(), 0,(work_count-1),0,(nvar-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int row, int n, int j, int i) {
    const int m = compact_work ? boundary_work(row) : row;
    // apply physical boundaries to inner_x3
    switch (mb_bcs.d_view(m,BoundaryFace::inner_x3)) {
      case BoundaryFlag::reflect:
        for (int k=0; k<ng; ++k) {
          if (n==z4c::Z4c::I_Z4C_GXZ || n==z4c::Z4c::I_Z4C_GYZ ||
              n==z4c::Z4c::I_Z4C_AXZ || n==z4c::Z4c::I_Z4C_AYZ ||
              n==z4c::Z4c::I_Z4C_GAMZ || n==z4c::Z4c::I_Z4C_BETAZ) {
            u0(m,n,ks-k-1,j,i) = -u0(m,n,ks+k+(vertex ? 1 : 0),j,i);
          } else {
            u0(m,n,ks-k-1,j,i) =  u0(m,n,ks+k+(vertex ? 1 : 0),j,i);
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
          if (n==z4c::Z4c::I_Z4C_GXZ || n==z4c::Z4c::I_Z4C_GYZ ||
              n==z4c::Z4c::I_Z4C_AXZ || n==z4c::Z4c::I_Z4C_AYZ ||
              n==z4c::Z4c::I_Z4C_GAMZ || n==z4c::Z4c::I_Z4C_BETAZ) {
            u0(m,n,ke+k+1,j,i) = -u0(m,n,ke-k-(vertex ? 1 : 0),j,i);
          } else {
            u0(m,n,ke+k+1,j,i) =  u0(m,n,ke-k-(vertex ? 1 : 0),j,i);
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
