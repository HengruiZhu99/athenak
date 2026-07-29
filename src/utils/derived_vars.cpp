//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file derived_variables.cpp
//! \brief Calculates derived variables used for outputs, mesh refinement criteria, etc.
//! Variables are only calculated over active zones (ghost zones excluded).

#include <iostream>
#include <sstream>
#include <string>   // std::string, to_string()

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coordinates.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "radiation/radiation.hpp"
#include "radiation/radiation_tetrad.hpp"
#include "particles/particles.hpp"
#include "scalar_field/scalar_field.hpp"
#include "utils/current.hpp"
#include "utils/finite_diff.hpp"

namespace {

template <int NGHOST>
void ComputeScalarDerived(const int quantity, const int index,
                          MeshBlockPack *pmbp, DvceArray5D<Real> dvars) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  auto &state = pmbp->pscalar->u0;
  auto &adm_vars = pmbp->padm->adm;
  const int nmb = pmbp->nmb_thispack;
  const int ncomp = pmbp->pscalar->ncomponents;
  const int ndim = 1 + static_cast<int>(pmbp->pmesh->multi_d) +
                   static_cast<int>(pmbp->pmesh->three_d);
  const scalar_field::PotentialData potential = pmbp->pscalar->potential;
  const bool use_excision = pmbp->pscalar->excision;
  auto &excision_mask = pmbp->pcoord->excision_floor;

  par_for(
      "scalar refinement diagnostic", DevExeSpace(), 0, nmb - 1,
      indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        const Real idx[3] = {
          Real(1.0)/size.d_view(m).dx1,
          Real(1.0)/size.d_view(m).dx2,
          Real(1.0)/size.d_view(m).dx3
        };
        Real phi[2] = {0.0, 0.0};
        Real pi[2] = {0.0, 0.0};
        Real gradient[2][3] = {
          {0.0, 0.0, 0.0},
          {0.0, 0.0, 0.0}
        };
        for (int component = 0; component < ncomp; ++component) {
          const int iphi = 2*component;
          phi[component] = state(m, iphi, k, j, i);
          pi[component] = state(m, iphi + 1, k, j, i);
          for (int direction = 0; direction < ndim; ++direction) {
            gradient[component][direction] =
                Dx<NGHOST>(direction, idx, state, m, iphi, k, j, i);
          }
        }
        if (quantity == 0) {
          dvars(m, index, k, j, i) =
              scalar_field::FieldAmplitude(ncomp, phi);
        } else if (use_excision && excision_mask(m, k, j, i)) {
          dvars(m, index, k, j, i) = 0.0;
        } else {
          const Real metric[6] = {
            adm_vars.g_dd(m, 0, 0, k, j, i),
            adm_vars.g_dd(m, 0, 1, k, j, i),
            adm_vars.g_dd(m, 0, 2, k, j, i),
            adm_vars.g_dd(m, 1, 1, k, j, i),
            adm_vars.g_dd(m, 1, 2, k, j, i),
            adm_vars.g_dd(m, 2, 2, k, j, i)
          };
          const scalar_field::MatterPoint matter =
              scalar_field::ComputeMatter(
                  ncomp, phi, pi, gradient, metric, potential);
          dvars(m, index, k, j, i) =
              (quantity == 1) ? matter.energy : matter.charge;
        }
      });
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn  ComputeDerivedVariable()
//! \brief Returns derived variable(s) specified by "name" in dvars(m,n,k,j,i) array
//! starting at n=index

void ComputeDerivedVariable(std::string name, int index, MeshBlockPack* pmbp,
                            DvceArray5D<Real> dvars) {
  int nmb = pmbp->nmb_thispack;
  auto &indcs = pmbp->pmesh->mb_indcs;
//  int &ng = indcs.ng;
//  int n1 = indcs.nx1 + 2*ng;
//  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
//  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;

  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  auto &size = pmbp->pmb->mb_size;

  if (name == "sf_amplitude" || name == "sf_energy" ||
      name == "sf_charge") {
    int quantity = 0;
    if (name == "sf_energy") {
      quantity = 1;
    } else if (name == "sf_charge") {
      quantity = 2;
    }
    switch (pmbp->pscalar->fd_stencil) {
      case 2:
        ComputeScalarDerived<2>(quantity, index, pmbp, dvars);
        break;
      case 3:
        ComputeScalarDerived<3>(quantity, index, pmbp, dvars);
        break;
      case 4:
        ComputeScalarDerived<4>(quantity, index, pmbp, dvars);
        break;
    }
    return;
  }

  // radiation coordinate frame energy density R^0^0
  if (name.compare("rad_coord_e") == 0) {
    // Coordinates
    auto &coord = pmbp->pcoord->coord_data;
    bool &flat = coord.is_minkowski;
    Real &spin = coord.bh_spin;

    // Radiation
    int nang1 = pmbp->prad->prgeo->nangles - 1;
    auto nh_c_ = pmbp->prad->nh_c;
    auto tet_c_ = pmbp->prad->tet_c;
    auto tetcov_c_ = pmbp->prad->tetcov_c;
    auto domega = pmbp->prad->prgeo->solid_angles;
    auto i0_ = pmbp->prad->i0;

    par_for("moments",DevExeSpace(),0,(nmb-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

      // Extract components of metric
      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x1v,x2v,x3v,flat,spin,glower,gupper);

      // coordinate component n^0
      Real n0 = tet_c_(m,0,0,k,j,i);

      // set coordinate frame component
      dvars(m,index,k,j,i) = 0.0;
      for (int n=0; n<=nang1; ++n) {
        Real nmun1 = 0.0; Real nmun2 = 0.0; Real n_0 = 0.0;
        for (int d=0; d<4; ++d) {
          nmun1 += tet_c_   (m,d,0,k,j,i)*nh_c_.d_view(n,d);
          nmun2 += tet_c_   (m,d,0,k,j,i)*nh_c_.d_view(n,d);
          n_0   += tetcov_c_(m,d,0,k,j,i)*nh_c_.d_view(n,d);
        }
        dvars(m,index,k,j,i) += (nmun1*nmun2*(i0_(m,n,k,j,i)/(n0*n_0))*domega.d_view(n));
      }
    });
  }
  return;
}
