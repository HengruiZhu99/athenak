//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_update.cpp
//! \brief Performs update of z4c variables (u0) for each stage of explicit
//  SSP RK integrators (e.g. RK1, RK2, RK3, RK4). Update uses weighted average
//  and partial time step appropriate to stage.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "coordinates/coordinates.hpp"
#include "globals.hpp"
#include "z4c/z4c.hpp"
#include "z4c/stored_domain_bounds.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace z4c {

void Z4c::InitializePrescribedZeroShift() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const auto bounds = MakeStoredDomainBounds(indcs);
  auto state = u0;
  const int nmb = pmy_pack->nmb_thispack;
  const int last = opt.telegraph_lapse ? I_Z4C_BETAZ : I_Z4C_BZ;
  par_for("initialize prescribed zero shift", DevExeSpace(), 0, nmb - 1,
          I_Z4C_BETAX, last, bounds.ks, bounds.ke, bounds.js, bounds.je,
          bounds.is, bounds.ie,
          KOKKOS_LAMBDA(const int m, const int n, const int k, const int j,
                        const int i) { state(m, n, k, j, i) = 0.0; });
  Kokkos::fence();
}

void Z4c::CheckPrescribedZeroShiftInvariant(Driver *driver, int stage) {
  if (opt.shift_mode != Z4cShiftMode::prescribed_zero ||
      !opt.shift_invariant_diagnostic) return;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto state = u0;
  const int nmb = pmy_pack->nmb_thispack;
  const int last = opt.telegraph_lapse ? I_Z4C_BETAZ : I_Z4C_BZ;
  Real local_max = 0.0;
  Kokkos::parallel_reduce(
      "prescribed zero shift invariant",
      Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<5>>(
          {0, I_Z4C_BETAX, indcs.ks, indcs.js, indcs.is},
          {nmb, last + 1, indcs.ke + 1, indcs.je + 1, indcs.ie + 1}),
      KOKKOS_LAMBDA(const int m, const int n, const int k, const int j,
                    const int i, Real &value) {
        value = fmax(value, fabs(state(m, n, k, j, i)));
      }, Kokkos::Max<Real>(local_max));
  Real global_max = local_max;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(&local_max, &global_max, 1, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
#endif
  if (global_variable::my_rank == 0) {
    std::ifstream prior("shift_invariant_check.csv");
    const bool exists = prior.good();
    prior.close();
    std::ofstream out("shift_invariant_check.csv", std::ios::app);
    if (!exists) out << "cycle,time,rk_stage,max_abs_prescribed_shift_state\n";
    out << pmy_pack->pmesh->ncycle << ',' << std::setprecision(17)
        << pmy_pack->pmesh->time << ',' << stage << ',' << global_max << '\n';
    if (!out) {
      std::cerr << "### FATAL ERROR: failed to write shift invariant evidence"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  if (global_max != 0.0) {
    std::cerr << "### FATAL ERROR: prescribed-zero shift invariant failed at cycle "
              << pmy_pack->pmesh->ncycle << " stage " << stage
              << " max_abs_prescribed_shift_state=" << global_max << std::endl;
    std::exit(EXIT_FAILURE);
  }
}
//----------------------------------------------------------------------------------------
//! \fn  void Z4c::Update
//! \brief Explicit RK update
TaskStatus Z4c::ExpRKUpdate(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;


  Real &gam0 = pdriver->gam0[stage-1];
  Real &gam1 = pdriver->gam1[stage-1];
  Real beta_dt = (pdriver->beta[stage-1])*(pmy_pack->pmesh->dt);
  auto &u0 = pmy_pack->pz4c->u0;
  auto &u1 = pmy_pack->pz4c->u1;
  auto &u_rhs = pmy_pack->pz4c->u_rhs;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nvar = nz4c;
  const bool prescribed_zero_shift =
      opt.shift_mode == Z4cShiftMode::prescribed_zero;
  const bool telegraph_lapse = opt.telegraph_lapse;

  par_for("z4c RK update",DevExeSpace(),
      0,nmb1,0,nvar-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(const int m, const int n, const int k, const int j, const int i) {
    const bool prescribed_component =
        n >= I_Z4C_BETAX && n <= (telegraph_lapse ? I_Z4C_BETAZ : I_Z4C_BZ);
    if (prescribed_zero_shift && prescribed_component) {
      u0(m,n,k,j,i) = 0.0;
    } else {
      u0(m,n,k,j,i) = gam0*u0(m,n,k,j,i) + gam1*u1(m,n,k,j,i) +
                      beta_dt*u_rhs(m,n,k,j,i);
    }
  });
  CheckPrescribedZeroShiftInvariant(pdriver, stage);
  if (chi_parent_provenance != nullptr) {
    chi_parent_provenance->RecordCheckpoint(
        ChiProvenanceCheckpoint::s0_after_rk, stage, pbval_u);
  }
  return TaskStatus::complete;
}
} // namespace z4c
