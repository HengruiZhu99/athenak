//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file minkowski.cpp
//! \brief Exact Minkowski initial data for regularized vacuum FO-GH evolution.

#include <cstdlib>
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "fo_gh/fo_gh.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"

namespace {

void CheckFoGhMinkowski(ParameterInput *pin, Mesh *pm) {
  (void)pin;
  auto *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const auto state = pmbp->pfogh->u0;
  const int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  Real max_error = 0.0;
  Kokkos::parallel_reduce(
      "fo_gh Minkowski error", Kokkos::RangePolicy<>(DevExeSpace(),
      0, pmbp->nmb_thispack*fo_gh::nvar*ncells),
      KOKKOS_LAMBDA(const int idx, Real &local_maximum) {
        int work = idx;
        const int i = work % indcs.nx1 + indcs.is;
        work /= indcs.nx1;
        const int j = work % indcs.nx2 + indcs.js;
        work /= indcs.nx2;
        const int k = work % indcs.nx3 + indcs.ks;
        work /= indcs.nx3;
        const int n = work % fo_gh::nvar;
        const int m = work/fo_gh::nvar;
        Real expected = 0.0;
        if (n == fo_gh::I_TGXX || n == fo_gh::I_TGYY ||
            n == fo_gh::I_TGZZ || n == fo_gh::I_CHI ||
            n == fo_gh::I_ALPHA) {
          expected = 1.0;
        }
        const Real error = Kokkos::abs(state(m, n, k, j, i) - expected);
        if (!Kokkos::isfinite(error)) {
          local_maximum = std::numeric_limits<Real>::infinity();
        } else {
          local_maximum = fmax(local_maximum, error);
        }
      }, Kokkos::Max<Real>(max_error));
  if (max_error > 1.0e-14) {
    std::cout << "FO-GH Minkowski evolution failed: max error = "
              << max_error << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "FO-GH Minkowski evolution passed: max error = "
            << max_error << std::endl;
}

} // namespace

void ProblemGenerator::FoGhMinkowski(ParameterInput *pin, const bool restart) {
  (void)pin;
  pgen_final_func = &CheckFoGhMinkowski;
  if (restart) {
    return;
  }
  auto *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pfogh == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "FO-GH Minkowski data require an <fo_gh> block."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int ncells1 = indcs.nx1 + 2*indcs.ng;
  const int ncells2 = indcs.nx2 + 2*indcs.ng;
  const int ncells3 = indcs.nx3 + 2*indcs.ng;
  const auto state = pmbp->pfogh->u0;
  par_for("fo_gh Minkowski", DevExeSpace(), 0, pmbp->nmb_thispack - 1,
  0, ncells3 - 1, 0, ncells2 - 1, 0, ncells1 - 1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    for (int n = 0; n < fo_gh::FoGh::nfo_gh; ++n) {
      state(m, n, k, j, i) = 0.0;
    }
    state(m, fo_gh::I_TGXX, k, j, i) = 1.0;
    state(m, fo_gh::I_TGYY, k, j, i) = 1.0;
    state(m, fo_gh::I_TGZZ, k, j, i) = 1.0;
    state(m, fo_gh::I_CHI, k, j, i) = 1.0;
    state(m, fo_gh::I_ALPHA, k, j, i) = 1.0;
  });
}
