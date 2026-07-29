//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file scalar_field_bcs.cpp
//! \brief Cartesian physical boundaries for canonical scalar-field variables.

#include "athena.hpp"
#include "bvals/bvals.hpp"
#include "mesh/mesh.hpp"

namespace {

template <int ORDER>
KOKKOS_INLINE_FUNCTION
Real ExtrapolateScalar(const DvceArray5D<Real> &u, const int m, const int n,
                       const int k, const int j, const int i,
                       const int dk, const int dj, const int di,
                       const int distance) {
  const Real f0 = u(m, n, k, j, i);
  const Real f1 = u(m, n, k + dk, j + dj, i + di);
  if constexpr (ORDER == 2) {
    return f0 + distance*(f0 - f1);
  } else {
    const Real f2 = u(m, n, k + 2*dk, j + 2*dj, i + 2*di);
    if constexpr (ORDER == 3) {
      return 0.5*(f0*(1 + distance)*(2 + distance) +
                  distance*(f2 + distance*f2 -
                            2*f1*(2 + distance)));
    } else {
      const Real f3 = u(m, n, k + 3*dk, j + 3*dj, i + 3*di);
      return (-3.0*f1*distance*(2 + distance)*(3 + distance) +
              f0*(1 + distance)*(2 + distance)*(3 + distance) +
              distance*(1 + distance) *
                  (-f3*(2 + distance) + 3*f2*(3 + distance)))/6.0;
    }
  }
}

template <int ORDER, int DIRECTION, bool INNER>
void ApplyScalarFace(MeshBlockPack *ppack, const DualArray2D<Real> &u_in,
                     DvceArray5D<Real> u0, const int is, const int ie,
                     const int js, const int je, const int ks, const int ke,
                     const int n1, const int n2, const int n3) {
  constexpr BoundaryFace face =
      (DIRECTION == 0)
          ? (INNER ? BoundaryFace::inner_x1 : BoundaryFace::outer_x1)
          : ((DIRECTION == 1)
                 ? (INNER ? BoundaryFace::inner_x2 : BoundaryFace::outer_x2)
                 : (INNER ? BoundaryFace::inner_x3 : BoundaryFace::outer_x3));
  auto &mb_bcs = ppack->pmb->mb_bcs;
  const int ng = ppack->pmesh->mb_indcs.ng;
  const int nmb = ppack->nmb_thispack;
  const int nvar = u0.extent_int(1);
  const int first_extent = (DIRECTION == 0) ? n3 : ((DIRECTION == 1) ? n3 : n2);
  const int second_extent = (DIRECTION == 0) ? n2 : n1;

  par_for(
      "scalar field physical boundary", DevExeSpace(), 0, nmb - 1,
      0, nvar - 1, 0, first_extent - 1, 0, second_extent - 1,
      KOKKOS_LAMBDA(const int m, const int n, const int first,
                    const int second) {
        const BoundaryFlag flag = mb_bcs.d_view(m, face);
        for (int ghost = 0; ghost < ng; ++ghost) {
          int k = (DIRECTION == 2)
                      ? (INNER ? ks - ghost - 1 : ke + ghost + 1)
                      : first;
          int j = (DIRECTION == 1)
                      ? (INNER ? js - ghost - 1 : je + ghost + 1)
                      : ((DIRECTION == 2) ? first : second);
          int i = (DIRECTION == 0)
                      ? (INNER ? is - ghost - 1 : ie + ghost + 1)
                      : second;

          int mirror_k = k;
          int mirror_j = j;
          int mirror_i = i;
          int edge_k = k;
          int edge_j = j;
          int edge_i = i;
          if constexpr (DIRECTION == 0) {
            mirror_i = INNER ? is + ghost : ie - ghost;
            edge_i = INNER ? is : ie;
          } else if constexpr (DIRECTION == 1) {
            mirror_j = INNER ? js + ghost : je - ghost;
            edge_j = INNER ? js : je;
          } else {
            mirror_k = INNER ? ks + ghost : ke - ghost;
            edge_k = INNER ? ks : ke;
          }

          switch (flag) {
            case BoundaryFlag::reflect:
              // Both phi and Pi are Cartesian scalars and therefore reflection-even.
              u0(m, n, k, j, i) =
                  u0(m, n, mirror_k, mirror_j, mirror_i);
              break;
            case BoundaryFlag::inflow:
              u0(m, n, k, j, i) = u_in.d_view(n, face);
              break;
            case BoundaryFlag::diode:
            case BoundaryFlag::outflow:
            case BoundaryFlag::vacuum: {
              const int inward = INNER ? 1 : -1;
              const int dk = (DIRECTION == 2) ? inward : 0;
              const int dj = (DIRECTION == 1) ? inward : 0;
              const int di = (DIRECTION == 0) ? inward : 0;
              // Polynomial extrapolation is not an absorbing massive-field BC.
              u0(m, n, k, j, i) =
                  ExtrapolateScalar<ORDER>(u0, m, n, edge_k, edge_j, edge_i,
                                           dk, dj, di, ghost + 1);
              break;
            }
            default:
              break;
          }
        }
      });
}

template <int ORDER>
void ApplyScalarBoundaries(MeshBlockPack *ppack, const DualArray2D<Real> &u_in,
                           DvceArray5D<Real> u0, const int is, const int ie,
                           const int js, const int je, const int ks, const int ke,
                           const int n1, const int n2, const int n3) {
  Mesh *pmesh = ppack->pmesh;
  if (pmesh->mesh_bcs[BoundaryFace::inner_x1] != BoundaryFlag::periodic &&
      pmesh->mesh_bcs[BoundaryFace::inner_x1] !=
          BoundaryFlag::shear_periodic) {
    ApplyScalarFace<ORDER, 0, true>(
        ppack, u_in, u0, is, ie, js, je, ks, ke, n1, n2, n3);
  }
  if (pmesh->mesh_bcs[BoundaryFace::outer_x1] != BoundaryFlag::periodic &&
      pmesh->mesh_bcs[BoundaryFace::outer_x1] !=
          BoundaryFlag::shear_periodic) {
    ApplyScalarFace<ORDER, 0, false>(
        ppack, u_in, u0, is, ie, js, je, ks, ke, n1, n2, n3);
  }
  if (pmesh->one_d) {
    return;
  }

  if (pmesh->mesh_bcs[BoundaryFace::inner_x2] != BoundaryFlag::periodic) {
    ApplyScalarFace<ORDER, 1, true>(
        ppack, u_in, u0, is, ie, js, je, ks, ke, n1, n2, n3);
  }
  if (pmesh->mesh_bcs[BoundaryFace::outer_x2] != BoundaryFlag::periodic) {
    ApplyScalarFace<ORDER, 1, false>(
        ppack, u_in, u0, is, ie, js, je, ks, ke, n1, n2, n3);
  }
  if (pmesh->two_d) {
    return;
  }

  if (pmesh->mesh_bcs[BoundaryFace::inner_x3] != BoundaryFlag::periodic) {
    ApplyScalarFace<ORDER, 2, true>(
        ppack, u_in, u0, is, ie, js, je, ks, ke, n1, n2, n3);
  }
  if (pmesh->mesh_bcs[BoundaryFace::outer_x3] != BoundaryFlag::periodic) {
    ApplyScalarFace<ORDER, 2, false>(
        ppack, u_in, u0, is, ie, js, je, ks, ke, n1, n2, n3);
  }
}

template <int ORDER>
void ApplyScalarStorage(MeshBlockPack *ppack, const DualArray2D<Real> &u_in,
                        DvceArray5D<Real> u0, DvceArray5D<Real> coarse_u0) {
  auto &indcs = ppack->pmesh->mb_indcs;
  const int ng = indcs.ng;
  ApplyScalarBoundaries<ORDER>(
      ppack, u_in, u0, indcs.is, indcs.ie, indcs.js, indcs.je,
      indcs.ks, indcs.ke, indcs.nx1 + 2*ng,
      (indcs.nx2 > 1) ? indcs.nx2 + 2*ng : 1,
      (indcs.nx3 > 1) ? indcs.nx3 + 2*ng : 1);

  if (ppack->pmesh->multilevel) {
    ApplyScalarBoundaries<ORDER>(
        ppack, u_in, coarse_u0, indcs.cis, indcs.cie, indcs.cjs, indcs.cje,
        indcs.cks, indcs.cke, indcs.cnx1 + 2*ng,
        (indcs.cnx2 > 1) ? indcs.cnx2 + 2*ng : 1,
        (indcs.cnx3 > 1) ? indcs.cnx3 + 2*ng : 1);
  }
}

} // namespace

void MeshBoundaryValues::ScalarFieldBCs(
    MeshBlockPack *ppack, DualArray2D<Real> u_in, DvceArray5D<Real> u0,
    DvceArray5D<Real> coarse_u0, int extrap_order) {
  switch (extrap_order) {
    case 2:
      ApplyScalarStorage<2>(ppack, u_in, u0, coarse_u0);
      break;
    case 3:
      ApplyScalarStorage<3>(ppack, u_in, u0, coarse_u0);
      break;
    case 4:
      ApplyScalarStorage<4>(ppack, u_in, u0, coarse_u0);
      break;
    default:
      break;
  }
}
