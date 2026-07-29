//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file scalar_field_tmunu.cpp
//! \brief Adds canonical scalar matter to the shared ADM stress-energy accumulator.

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/coordinates.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "scalar_field/scalar_field.hpp"
#include "utils/finite_diff.hpp"
#include "z4c/tmunu.hpp"

namespace scalar_field {

//----------------------------------------------------------------------------------------
//! \brief Dispatch scalar stress-energy assembly using the configured FD stencil.

TaskStatus ScalarField::AddTmunu(Driver *driver, int stage) {
  switch (fd_stencil) {
    case 2:
      return AddTmunu<2>(driver, stage);
    case 3:
      return AddTmunu<3>(driver, stage);
    case 4:
      return AddTmunu<4>(driver, stage);
    default:
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
                << __LINE__ << std::endl
                << "Unsupported scalar-field finite-difference stencil selector "
                << fd_stencil << std::endl;
      std::exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
//! \brief Refresh scalar matter after the final RK stage for diagnostics and output.

TaskStatus ScalarField::AddTmunuFinal(Driver *driver, int stage) {
  if (stage != driver->nexp_stages) {
    return TaskStatus::complete;
  }
  return AddTmunu(driver, stage);
}

//----------------------------------------------------------------------------------------
//! \brief Add E, S_i, and S_ij for all active scalar components.

template <int NGHOST>
TaskStatus ScalarField::AddTmunu(Driver *driver, int stage) {
  (void)driver;
  (void)stage;
  if (!backreaction || pmy_pack->pz4c == nullptr ||
      pmy_pack->ptmunu == nullptr) {
    return TaskStatus::complete;
  }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  auto &adm_vars = pmy_pack->padm->adm;
  auto &excision_mask = pmy_pack->pcoord->excision_floor;
  auto &state = u0;
  auto &matter_vars = pmy_pack->ptmunu->tmunu;
  const int nmb = pmy_pack->nmb_thispack;
  const int ndim = 1 + static_cast<int>(pmy_pack->pmesh->multi_d) +
                   static_cast<int>(pmy_pack->pmesh->three_d);
  const int ncomp = ncomponents;
  const PotentialData pot = potential;
  const bool use_excision = excision;

  par_for(
      "add scalar field tmunu", DevExeSpace(), 0, nmb - 1,
      indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        // Excised scalar data is an interior regularization and must not source Z4c.
        if (use_excision && excision_mask(m, k, j, i)) {
          return;
        }

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

        const Real metric[6] = {
          adm_vars.g_dd(m, 0, 0, k, j, i),
          adm_vars.g_dd(m, 0, 1, k, j, i),
          adm_vars.g_dd(m, 0, 2, k, j, i),
          adm_vars.g_dd(m, 1, 1, k, j, i),
          adm_vars.g_dd(m, 1, 2, k, j, i),
          adm_vars.g_dd(m, 2, 2, k, j, i)
        };
        const MatterPoint matter =
            ComputeMatter(ncomp, phi, pi, gradient, metric, pot);

        matter_vars.E(m, k, j, i) += matter.energy;
        for (int direction = 0; direction < 3; ++direction) {
          matter_vars.S_d(m, direction, k, j, i) +=
              matter.momentum[direction];
        }
        matter_vars.S_dd(m, 0, 0, k, j, i) += matter.stress[0];
        matter_vars.S_dd(m, 0, 1, k, j, i) += matter.stress[1];
        matter_vars.S_dd(m, 0, 2, k, j, i) += matter.stress[2];
        matter_vars.S_dd(m, 1, 1, k, j, i) += matter.stress[3];
        matter_vars.S_dd(m, 1, 2, k, j, i) += matter.stress[4];
        matter_vars.S_dd(m, 2, 2, k, j, i) += matter.stress[5];
      });
  return TaskStatus::complete;
}

template TaskStatus ScalarField::AddTmunu<2>(Driver *driver, int stage);
template TaskStatus ScalarField::AddTmunu<3>(Driver *driver, int stage);
template TaskStatus ScalarField::AddTmunu<4>(Driver *driver, int stage);

}  // namespace scalar_field
