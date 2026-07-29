//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file tmunu.hpp
//! \brief implementation of Tmunu class
#include <algorithm>

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "driver/driver.hpp"
#include "parameter_input.hpp"
#include "z4c/tmunu.hpp"
#include "mesh/mesh.hpp"
#include "tasklist/numerical_relativity.hpp"

char const * const Tmunu::Tmunu_names[Tmunu::N_Tmunu] = {
  "tmunu_Sxx", "tmunu_Sxy", "tmunu_Sxz", "tmunu_Syy", "tmunu_Syz", "tmunu_Szz",
  "tmunu_E", "tmunu_Sx", "tmunu_Sy", "tmunu_Sz",
};

//----------------------------------------------------------------------------------------
// constructor: initializes data structures and parameters
Tmunu::Tmunu(MeshBlockPack *ppack, ParameterInput *pin):
  u_tmunu("u_tmunu",1,1,1,1,1),
  pmy_pack(ppack) {
  int nmb = std::max((ppack->nmb_thispack), (ppack->pmesh->nmb_maxperrank));
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*(indcs.ng)) : 1;

  Kokkos::realloc(u_tmunu, nmb, N_Tmunu, ncells3, ncells2, ncells1);
  tmunu.S_dd.InitWithShallowSlice(u_tmunu, I_Tmunu_Sxx, I_Tmunu_Szz);
  tmunu.E.InitWithShallowSlice(u_tmunu, I_Tmunu_E);
  tmunu.S_d.InitWithShallowSlice(u_tmunu, I_Tmunu_Sx, I_Tmunu_Sz);
  Kokkos::deep_copy(u_tmunu, 0.0);
}

Tmunu::~Tmunu() {}

//----------------------------------------------------------------------------------------
//! \brief Queue the deterministic per-stage reset of the shared matter accumulator.

void Tmunu::QueueTmunuTasks() {
  using namespace numrel;  // NOLINT(build/namespaces)
  pmy_pack->pnr->QueueTask(
      &Tmunu::Clear, this, Tmunu_Clear, "Tmunu_Clear", Task_Run);
  pmy_pack->pnr->QueueTask(
      &Tmunu::ClearFinal, this, Tmunu_Clear, "Tmunu_ClearFinal", Task_End);
}

//----------------------------------------------------------------------------------------
//! \brief Clear all active-cell ADM matter variables before additive producers run.

TaskStatus Tmunu::Clear(Driver *driver, int stage) {
  (void)driver;
  (void)stage;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int nmb = pmy_pack->nmb_thispack;
  auto &data = u_tmunu;

  par_for(
      "clear tmunu accumulator", DevExeSpace(), 0, nmb - 1,
      0, N_Tmunu - 1, indcs.ks, indcs.ke, indcs.js, indcs.je,
      indcs.is, indcs.ie,
      KOKKOS_LAMBDA(const int m, const int n, const int k, const int j,
                    const int i) {
        data(m, n, k, j, i) = 0.0;
      });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \brief Clear the accumulator after the final RK stage before refreshing matter.

TaskStatus Tmunu::ClearFinal(Driver *driver, int stage) {
  if (stage != driver->nexp_stages) {
    return TaskStatus::complete;
  }
  return Clear(driver, stage);
}
