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
#include "mesh/mesh_refinement.hpp"
#include "driver/driver.hpp"
#include "driver/classical_rk4.hpp"
#include "coordinates/coordinates.hpp"
#include "globals.hpp"
#include "z4c/z4c.hpp"

#include <cstdlib>
#include <numeric>
#include <fstream>
#include <iomanip>
#include <iostream>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace z4c {

void Z4c::RebuildSubcycleParents() {
  auto *mesh=pmy_pack->pmesh;
  if (global_variable::nranks != 1 || pmy_pack->gids != 0 ||
      pmy_pack->nmb_thispack != mesh->nmb_total ||
      layout.centering != Z4cGridCentering::vertex ||
      pmy_pack->z4c_symmetry.mode != Z4cSymmetryMode::cartoon_so2) {
    throw std::runtime_error("subcycling parents currently require single-pack VC Cartoon");
  }
  std::vector<subcycling::BlockKey> leaves;
  leaves.reserve(mesh->nmb_total);
  for (int m=0; m<mesh->nmb_total; ++m) {
    const auto &loc=mesh->lloc_eachmb[m];
    leaves.push_back({loc.level,loc.lx1,loc.lx2,loc.lx3});
  }
  auto hierarchy=std::make_unique<subcycling::Hierarchy>(leaves,mesh->root_level,2);
  subcycle_parents.Initialize(*hierarchy,u0,layout);
  subcycle_hierarchy=std::move(hierarchy);
}

void Z4c::InitializePrescribedZeroShift() {
  const auto bounds = layout;
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
  const auto bounds = layout;
  auto state = u0;
  const int nmb = pmy_pack->nmb_thispack;
  const int last = opt.telegraph_lapse ? I_Z4C_BETAZ : I_Z4C_BZ;
  Real local_max = 0.0;
  Kokkos::parallel_reduce(
      "prescribed zero shift invariant",
      Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<5>>(
          {0, I_Z4C_BETAX, bounds.ks, bounds.js, bounds.is},
          {nmb, last + 1, bounds.ke + 1, bounds.je + 1, bounds.ie + 1}),
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
  const auto bounds = layout;
  int is = bounds.is, ie = bounds.ie;
  int js = bounds.js, je = bounds.je;
  int ks = bounds.ks, ke = bounds.ke;


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

  if (pdriver->integrator == "rk4_classical") {
    // u1 is the beginning-of-step state, copied by CopyU only at stage 1.
    // The accumulator is scratch, never checkpointed or transferred by AMR.
    // Resize only at stage 1, after any between-step hierarchy changes.
    if (stage == 1 && classical_sum.extent(0) != nmb1 + 1) {
      Kokkos::realloc(classical_sum, nmb1 + 1, nz4c,
                      bounds.n3, bounds.n2, bounds.n1);
    }
#ifdef ATHENA_Z4C_KERNEL_TESTS
    // Test hook only: capture actual post-boundary RHS without consuming the
    // predictor or changing evolution. Runtime level-local stepping will select
    // only coarse donor blocks and attach this to each parent interval.
    if (std::getenv("ATHENA_TEST_RK_PREDICTOR") != nullptr) {
      if (stage == 1) {
        std::vector<int> ids(nmb1+1);
        std::iota(ids.begin(),ids.end(),0);
        coarse_rk_predictor.Begin(u1,bounds,ids,pmy_pack->pmesh->time,
                                  pmy_pack->pmesh->dt);
      }
      coarse_rk_predictor.Capture(u_rhs,stage);
    }
#endif
    auto sum = classical_sum;
    const Real dt = pmy_pack->pmesh->dt;
    const Real weight = classical_rk4::Weight(stage);
    const Real next = classical_rk4::NextFraction(stage);
    par_for("z4c classical RK4 update", DevExeSpace(),
        0, nmb1, 0, nvar-1, ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA(const int m, const int n, const int k, const int j,
                      const int i) {
      const bool prescribed_component = n >= I_Z4C_BETAX &&
          n <= (telegraph_lapse ? I_Z4C_BETAZ : I_Z4C_BZ);
      const Real rhs = u_rhs(m,n,k,j,i);
      sum(m,n,k,j,i) = (stage == 1 ? 0.0 : sum(m,n,k,j,i)) + weight*rhs;
      u0(m,n,k,j,i) = prescribed_zero_shift && prescribed_component ? 0.0 :
          u1(m,n,k,j,i) + dt*(stage == 4 ? sum(m,n,k,j,i) : next*rhs);
    });
  } else {
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
  }
  ApplyVertexAxisRegularity(u0, stage, "post_rk_state");
  CheckPrescribedZeroShiftInvariant(pdriver, stage);
  if (chi_parent_provenance != nullptr) {
    chi_parent_provenance->RecordCheckpoint(
        ChiProvenanceCheckpoint::s0_after_rk, stage, pbval_u);
  }
  CheckStateAdmissibility(pdriver, stage, Z4cStateCheckpoint::post_rk_update);
  if (layout.centering == Z4cGridCentering::vertex &&
      pmy_pack->pmesh->pmr != nullptr) {
    pmy_pack->pmesh->pmr->VCAMRLifecycleMarkFirstPostEventUpdate(u0, stage);
  }
  return TaskStatus::complete;
}
} // namespace z4c
