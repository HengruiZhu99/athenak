//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_tasks.cpp
//! \brief functions that control z4c tasks in the appropriate task list

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>
#include <cstdio>
#include <cstdlib>

#include <Kokkos_Timer.hpp>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "z4c/compact_object_tracker.hpp"
#include "z4c/cartoon_axis_boundary.hpp"
#include "z4c/cartoon_vertex_axis.hpp"
#include "z4c/fastflow.hpp"
#include "z4c/horizon_dump.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_vertex_topology.hpp"
#include "tasklist/numerical_relativity.hpp"
#include "z4c/cce/cce.hpp"

namespace z4c {

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::QueueZ4cTasks
//! \brief queue Z4c tasks into NumericalRelativity
void Z4c::QueueZ4cTasks() {
  printf("AssembleZ4cTasks\n");
  using namespace mhd;     // NOLINT(build/namespaces)
  using namespace numrel;  // NOLINT(build/namespaces)
  NumericalRelativity *pnr = pmy_pack->pnr;
  int const fd_stencil = opt.fd_stencil;

  // Start task list
  pnr->QueueTask(&Z4c::InitRecv, this, Z4c_Recv, "Z4c_Recv", Task_Start);
  pnr->QueueTask(&Z4c::InitRecvWeyl, this, Z4c_IRecvW, "Z4c_IRecvW", Task_Start);

  // Run task list
  pnr->QueueTask(&Z4c::CopyU, this, Z4c_CopyU, "Z4c_CopyU", Task_Run);
  pnr->QueueTask(&Z4c::FillAxisParityGhosts, this, Z4c_AxisGhosts,
                 "Z4c_AxisGhosts", Task_Run, {Z4c_CopyU});
  switch (fd_stencil) {
    case 2:
      pnr->QueueTask(&Z4c::CalcRHS<2>, this, Z4c_CalcRHS, "Z4c_CalcRHS",
                     Task_Run, {Z4c_AxisGhosts}, {MHD_SetTmunu});
      break;
    case 3:
      pnr->QueueTask(&Z4c::CalcRHS<3>, this, Z4c_CalcRHS, "Z4c_CalcRHS",
                     Task_Run, {Z4c_AxisGhosts}, {MHD_SetTmunu});
      break;
    case 4:
      pnr->QueueTask(&Z4c::CalcRHS<4>, this, Z4c_CalcRHS, "Z4c_CalcRHS",
                     Task_Run, {Z4c_AxisGhosts}, {MHD_SetTmunu});
      break;
    default:
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "Unsupported Z4c finite-difference stencil selector "
                << fd_stencil << std::endl;
      std::exit(EXIT_FAILURE);
  }
  pnr->QueueTask(&Z4c::Z4cBoundaryRHS, this, Z4c_SomBC, "Z4c_SomBC", Task_Run,
                 {Z4c_CalcRHS});
  pnr->QueueTask(&Z4c::ExpRKUpdate, this, Z4c_ExplRK, "Z4c_ExplRK", Task_Run,
                 {Z4c_SomBC},{MHD_EField});
  const bool vertex_centered =
      layout.centering == Z4cGridCentering::vertex;
  if (pmy_pack->pz4c->opt.floor_chi) {
    pnr->QueueTask(&Z4c::Z4cFloorChi, this, Z4c_ChiFloor, "Z4c_ChiFloor", Task_Run,
                   {Z4c_ExplRK});
  }
  if (vertex_centered) {
    // Native VC first reconciles shared nodes and builds a complete unprojected
    // stage boundary state.  The accepted final stage is projected and rebuilt
    // once more by FinalizeVertexAcceptedState below.
    pnr->QueueTask(&Z4c::RestrictU, this, Z4c_RestU, "Z4c_RestU", Task_Run,
                   pmy_pack->pz4c->opt.floor_chi
                       ? std::vector<TaskName>{Z4c_ChiFloor}
                       : std::vector<TaskName>{Z4c_ExplRK});
  } else {
    if (pmy_pack->pz4c->opt.floor_chi) {
      pnr->QueueTask(&Z4c::EnforceAlgConstr, this, Z4c_AlgC, "Z4c_AlgC",
                     Task_Run, {Z4c_ChiFloor});
    } else {
      pnr->QueueTask(&Z4c::EnforceAlgConstr, this, Z4c_AlgC, "Z4c_AlgC",
                     Task_Run, {Z4c_ExplRK});
    }
    pnr->QueueTask(&Z4c::RestrictU, this, Z4c_RestU, "Z4c_RestU",
                   Task_Run, {Z4c_AlgC});
  }
  pnr->QueueTask(&Z4c::SendU, this, Z4c_SendU, "Z4c_SendU", Task_Run, {Z4c_RestU});
  pnr->QueueTask(&Z4c::RecvU, this, Z4c_RecvU, "Z4c_RecvU", Task_Run, {Z4c_SendU});
  pnr->QueueTask(&Z4c::ApplyPhysicalBCs, this, Z4c_BCS, "Z4c_BCS", Task_Run, {Z4c_RecvU});
  pnr->QueueTask(&Z4c::Prolongate, this, Z4c_Prolong, "Z4c_Prolong", Task_Run, {Z4c_BCS});
  pnr->QueueTask(&Z4c::FillAxisParityGhosts, this, Z4c_AxisGhostsPost,
                 "Z4c_AxisGhostsPost", Task_Run, {Z4c_Prolong});
  if (vertex_centered) {
    pnr->QueueTask(&Z4c::FinalizeVertexAcceptedState, this, Z4c_VCFinalize,
                   "Z4c_VCFinalize", Task_Run, {Z4c_AxisGhostsPost});
    pnr->QueueTask(&Z4c::ConvertZ4cToADM, this, Z4c_Z4c2ADM, "Z4c_Z4c2ADM",
                   Task_Run, {Z4c_VCFinalize});
  } else {
    pnr->QueueTask(&Z4c::ConvertZ4cToADM, this, Z4c_Z4c2ADM, "Z4c_Z4c2ADM",
                   Task_Run, {Z4c_AxisGhostsPost});
  }
  if (pmy_pack->pdyngr != nullptr) {
    pnr->QueueTask(&Z4c::UpdateExcisionMasks, this, Z4c_Excise, "Z4c_Excise", Task_Run,
                   {Z4c_Z4c2ADM}, {Z4c_FastFlow});
  }
  pnr->QueueTask(&Z4c::NewTimeStep, this, Z4c_Newdt, "Z4c_Newdt", Task_Run,
                 {Z4c_Z4c2ADM});
  pnr->QueueTask(&Z4c::TrackCompactObjects, this, Z4c_PT, "Z4c_PT",
                 Task_Run, {Z4c_Z4c2ADM});
  pnr->QueueTask(&Z4c::FindHorizon, this, Z4c_FastFlow, "Z4c_FastFlow",
                 Task_Run, {Z4c_PT});

  // End task list
  pnr->QueueTask(&Z4c::ClearSend, this, Z4c_ClearS, "Z4c_ClearS", Task_End);
  pnr->QueueTask(&Z4c::ClearRecv, this, Z4c_ClearR, "Z4c_ClearR", Task_End, {Z4c_ClearS});
  /*pnr->QueueTask(&Z4c::Z4cToADM, this, Z4c_Z4c2ADM, "Z4c_Z4c2ADM", Task_End,
                 {Z4c_ClearR});*/
  pnr->QueueTask(&Z4c::ADMConstraints_, this, Z4c_ADMC, "Z4c_ADMC", Task_End,
  //               {Z4c_Z4c2ADM});
                 {Z4c_ClearR});
  pnr->QueueTask(&Z4c::CalcWeylScalar, this, Z4c_Weyl, "Z4c_Weyl", Task_End, {Z4c_ADMC});
  pnr->QueueTask(&Z4c::RestrictWeyl, this, Z4c_RestW, "Z4c_RestW", Task_End, {Z4c_Weyl});
  pnr->QueueTask(&Z4c::SendWeyl, this, Z4c_SendW, "Z4c_SendW", Task_End, {Z4c_RestW});
  pnr->QueueTask(&Z4c::RecvWeyl, this, Z4c_RecvW, "Z4c_RecvW", Task_End, {Z4c_SendW});
  pnr->QueueTask(&Z4c::ProlongateWeyl, this, Z4c_ProlW, "Z4c_ProlW", Task_End,
                 {Z4c_RecvW});
  pnr->QueueTask(&Z4c::ClearSendWeyl, this, Z4c_ClearSW, "Z4c_ClearS2", Task_End,
                 {Z4c_ProlW});
  pnr->QueueTask(&Z4c::ClearRecvWeyl, this, Z4c_ClearRW, "Z4c_ClearR2", Task_End,
                 {Z4c_ClearSW});
  pnr->QueueTask(&Z4c::CalcWaveForm, this, Z4c_Wave, "Z4c_Wave", Task_End,
                 {Z4c_ClearRW});
  pnr->QueueTask(&Z4c::CCEDump, this, Z4c_CCE, "CCEDump", Task_End, {Z4c_Wave});
  pnr->QueueTask(&Z4c::DumpHorizons, this, Z4c_DumpHorizon, "Z4c_DumpHorizon",
                Task_End, {Z4c_CCE});
}
//----------------------------------------------------------------------------------------
//! \fn  void Wave::InitRecv
//! \brief function to post non-blocking receives (with MPI), and initialize all boundary
//  receive status flags to waiting (with or without MPI) for Wave variables.

TaskStatus Z4c::InitRecv(Driver *pdrive, int stage) {
  TaskStatus tstat = layout.centering == Z4cGridCentering::vertex
      ? pbval_u_vc->InitRecv(nz4c) : pbval_u->InitRecv(nz4c);
  if (tstat != TaskStatus::complete) return tstat;
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Wave::ClearRecv
//! \brief Waits for all MPI receives to complete before allowing execution to continue

TaskStatus Z4c::ClearRecv(Driver *pdrive, int stage) {
  TaskStatus tstat = layout.centering == Z4cGridCentering::vertex
      ? pbval_u_vc->ClearRecv() : pbval_u->ClearRecv();
  if (tstat != TaskStatus::complete) return tstat;
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::ClearSend
//! \brief Waits for all MPI sends to complete before allowing execution to continue

TaskStatus Z4c::ClearSend(Driver *pdrive, int stage) {
  TaskStatus tstat = layout.centering == Z4cGridCentering::vertex
      ? pbval_u_vc->ClearSend() : pbval_u->ClearSend();
  if (tstat != TaskStatus::complete) return tstat;
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::CopyU
//! \brief  copy u0 --> u1 in first stage

TaskStatus Z4c::CopyU(Driver *pdrive, int stage) {
  auto integrator = pdrive->integrator;

  // Copy/accumulate every active degree of freedom.  A native vertex grid has
  // one more active point than the cell-centered mesh indices in each
  // non-collapsed direction, so using mb_indcs here silently omitted the
  // shared upper face/edge/corner vertices from RK4's low-storage accumulator.
  const auto bounds = layout;
  int is = bounds.is, ie = bounds.ie;
  int js = bounds.js, je = bounds.je;
  int ks = bounds.ks, ke = bounds.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nvar = nz4c;
  auto &u0 = pmy_pack->pz4c->u0;
  auto &u1 = pmy_pack->pz4c->u1;

  if (chi_parent_provenance != nullptr) {
    chi_parent_provenance->RecordBeforeCopy(pdrive, stage);
  }

  // hierarchical parallel loop that updates conserved variables to intermediate step
  // using weights and fractional time step appropriate to stages of time-integrator.
  // Important to use vector inner loop for good performance on cpus
  if (integrator == "rk4") {
    Real &delta = pdrive->delta[stage-1];
    if (stage == 1) {
      Kokkos::deep_copy(DevExeSpace(), u1, u0);
    } else {
      par_for("CopyCons", DevExeSpace(),0, nmb1, 0, nvar-1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int n, int k, int j, int i){
        u1(m,n,k,j,i) += delta*u0(m,n,k,j,i);
      });
    }
  } else {
    if (stage == 1) {
      Kokkos::deep_copy(DevExeSpace(), u1, u0);
    }
  }
  if (chi_parent_provenance != nullptr) {
    chi_parent_provenance->RecordAfterCopy(pdrive, stage);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Z4c::FillAxisParityGhosts
//! \brief Reconstruct only the derived negative-rho ghost storage for the current stage.
//!
//! The accepted-state boundary pass follows active-cell algebraic projection, exchange,
//! physical boundary conditions, and coarse-fine prolongation.  This named task rebuilds
//! the derived negative-rho storage before the next RHS without independently projecting
//! any ghost value.

TaskStatus Z4c::FillAxisParityGhosts(Driver *pdrive, int stage) {
  ReconstructAxisParityGhosts();
  return TaskStatus::complete;
}

void Z4c::ReconstructAxisParityGhosts() {
  const auto &config = pmy_pack->z4c_symmetry;
  if (config.mode != Z4cSymmetryMode::cartoon_so2 ||
      config.coordinate_map != Z4cCoordinateMap::half_rho_z_suppressed_y_v2 ||
      config.schema != Z4cSymmetryConfig::kHalfPlaneCartoonSchema) {
    return;
  }

  const int ng = layout.ng;
  const int n2 = layout.n2;
  const int n3 = layout.n3;
  const int is = layout.is;
  const int nmb = pmy_pack->nmb_thispack;
  auto &mb_bcs = pmy_pack->pmb->mb_bcs;
  auto &state = pmy_pack->pz4c->u0;

  if (layout.centering == Z4cGridCentering::vertex) {
    par_for("z4c VC half-plane axis parity", DevExeSpace(), 0, nmb - 1,
            0, nz4c - 1, 0, n3 - 1, 0, n2 - 1,
        KOKKOS_LAMBDA(const int m, const int n, const int k, const int j) {
          if (mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::axis &&
              !FillCenteredZ4cAxisGhostLine<VertexCenteredZ4c>(
                  state, m, n, k, j, is, ng)) {
            Kokkos::abort("invalid packed Z4c component in VC axis parity fill");
          }
        });
  } else {
    par_for("z4c CC half-plane axis parity", DevExeSpace(), 0, nmb - 1,
            0, nz4c - 1, 0, n3 - 1, 0, n2 - 1,
        KOKKOS_LAMBDA(const int m, const int n, const int k, const int j) {
          if (mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::axis &&
              !FillCenteredZ4cAxisGhostLine<CellCenteredZ4c>(
                  state, m, n, k, j, is, ng)) {
            Kokkos::abort("invalid packed Z4c component in CC axis parity fill");
          }
        });
  }
  Kokkos::fence();
}

//----------------------------------------------------------------------------------------
//! \brief Enforce exact SO(2) identities on the evolved VC axis and emit a deterministic
//! correction audit.  This is a regularity projection, not a floor or limiter.

void Z4c::ApplyVertexAxisRegularity(DvceArray5D<Real> &state, const int stage,
                                    const char *checkpoint) {
  if (layout.centering != Z4cGridCentering::vertex ||
      pmy_pack->z4c_symmetry.mode != Z4cSymmetryMode::cartoon_so2) {
    return;
  }
  const int nmb = pmy_pack->nmb_thispack;
  const int active_n2 = layout.je - layout.js + 1;
  const int active_n3 = layout.ke - layout.ks + 1;
  const int points_per_block = active_n2 * active_n3;
  DvceArray2D<Real> records("vertex axis regularity records",
                            nmb * points_per_block, 4);
  Kokkos::deep_copy(records, 0.0);
  auto &mb_bcs = pmy_pack->pmb->mb_bcs;
  const int is = layout.is;
  const int js = layout.js;
  const int ks = layout.ks;
  par_for("enforce evolved vertex axis regularity", DevExeSpace(), 0, nmb - 1,
          layout.ks, layout.ke, layout.js, layout.je,
      KOKKOS_LAMBDA(const int m, const int k, const int j) {
        if (mb_bcs.d_view(m, BoundaryFace::inner_x1) != BoundaryFlag::axis) return;
        const VertexAxisCorrection correction =
            EnforceVertexAxisZ4cPoint(state, m, k, j, is);
        const int record =
            m * points_per_block + (k - ks) * active_n2 + (j - js);
        records(record, 0) = correction.max_abs;
        records(record, 1) = correction.max_rel;
        records(record, 2) = static_cast<Real>(correction.component);
        records(record, 3) = static_cast<Real>(correction.nonfinite);
      });
  Kokkos::fence();

  const auto host_records =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), records);
  pmy_pack->pmb->mb_gid.sync_host();
  pmy_pack->pmb->mb_size.sync_host();
  Real local_max_abs = 0.0;
  Real local_max_rel = 0.0;
  int local_nonfinite = 0;
  unsigned long long local_key = std::numeric_limits<unsigned long long>::max();
  int local_component = -1;
  int local_m = -1;
  int local_j = -1;
  for (int m = 0; m < nmb; ++m) {
    for (int k = layout.ks; k <= layout.ke; ++k) {
      for (int j = layout.js; j <= layout.je; ++j) {
        const int record =
            m * points_per_block + (k - layout.ks) * active_n2 + (j - layout.js);
        const Real absolute = host_records(record, 0);
        const Real relative = host_records(record, 1);
        const int component = static_cast<int>(host_records(record, 2));
        local_nonfinite = std::max(local_nonfinite,
                                   static_cast<int>(host_records(record, 3)));
        local_max_rel = std::max(local_max_rel, relative);
        const unsigned long long ordinal =
            static_cast<unsigned long long>((k - layout.ks) * active_n2 +
                                            (j - layout.js));
        const unsigned long long key =
            (static_cast<unsigned long long>(pmy_pack->pmb->mb_gid.h_view(m)) << 32) |
            (ordinal << 8) | static_cast<unsigned long long>(component + 1);
        if (absolute > local_max_abs ||
            (absolute == local_max_abs && key < local_key)) {
          local_max_abs = absolute;
          local_key = key;
          local_component = component;
          local_m = m;
          local_j = j;
        }
      }
    }
  }
  Real global_max_abs = local_max_abs;
  Real global_max_rel = local_max_rel;
  int global_nonfinite = local_nonfinite;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &global_max_abs, 1, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &global_max_rel, 1, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &global_nonfinite, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
#endif
  if (local_max_abs != global_max_abs) {
    local_key = std::numeric_limits<unsigned long long>::max();
    local_component = -1;
    local_m = -1;
    local_j = -1;
  }
  unsigned long long global_key = local_key;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &global_key, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN,
                MPI_COMM_WORLD);
#endif
  const int selected_gid =
      global_key == std::numeric_limits<unsigned long long>::max()
          ? -1 : static_cast<int>(global_key >> 32);
  int selected_component = -1;
  Real selected_z = 0.0;
  if (local_key == global_key && local_m >= 0) {
    selected_component = local_component;
    const auto &size = pmy_pack->pmb->mb_size.h_view(local_m);
    selected_z = size.x2min +
                 static_cast<Real>(local_j - layout.js) * size.dx2;
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &selected_component, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &selected_z, 1, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
#endif
  if (global_variable::my_rank == 0 &&
      (global_max_abs > 0.0 || global_nonfinite != 0)) {
    std::ifstream prior("z4c_vertex_axis_regularity.csv");
    const bool exists = prior.good();
    prior.close();
    std::ofstream out("z4c_vertex_axis_regularity.csv", std::ios::app);
    if (!exists) {
      out << "cycle,time,rk_stage,checkpoint,max_abs,max_scaled,component,gid,z,nonfinite\n";
    }
    out << pmy_pack->pmesh->ncycle << ',' << std::setprecision(17)
        << pmy_pack->pmesh->time << ',' << stage << ',' << checkpoint << ','
        << global_max_abs << ',' << global_max_rel << ',' << selected_component
        << ',' << selected_gid << ',' << selected_z << ',' << global_nonfinite
        << '\n';
    if (!out) {
      std::cerr << "### FATAL ERROR: failed to write VC axis regularity evidence"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  if (global_nonfinite != 0 ||
      global_max_rel > opt.vertex_axis_correction_tolerance) {
    std::cerr << "### FATAL ERROR: VC axis regularity correction rejected at cycle "
              << pmy_pack->pmesh->ncycle << " stage " << stage
              << " checkpoint=" << checkpoint << " max_abs=" << global_max_abs
              << " max_scaled=" << global_max_rel << " tolerance="
              << opt.vertex_axis_correction_tolerance << " component="
              << selected_component << " gid=" << selected_gid << " z="
              << selected_z << " nonfinite=" << global_nonfinite << std::endl;
#if MPI_PARALLEL_ENABLED
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#else
    std::exit(EXIT_FAILURE);
#endif
  }
}

void Z4c::ReconstructConstraintAxisParityGhosts() {
  ReconstructConstraintAxisParityGhosts(u_con);
}

void Z4c::ReconstructConstraintAxisParityGhosts(
    DvceArray5D<Real> &constraints) {
  const auto &config = pmy_pack->z4c_symmetry;
  if (config.mode != Z4cSymmetryMode::cartoon_so2 ||
      config.coordinate_map != Z4cCoordinateMap::half_rho_z_suppressed_y_v2 ||
      config.schema != Z4cSymmetryConfig::kHalfPlaneCartoonSchema) {
    return;
  }

  const auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int ng = indcs.ng;
  const int n2 = indcs.nx2 > 1 ? indcs.nx2 + 2 * ng : 1;
  const int n3 = indcs.nx3 > 1 ? indcs.nx3 + 2 * ng : 1;
  const int is = indcs.is;
  const int nmb = pmy_pack->nmb_thispack;
  auto &mb_bcs = pmy_pack->pmb->mb_bcs;
  par_for("Z4c half-plane constraint axis parity", DevExeSpace(),
          0, nmb - 1, 0, ncon - 1, 0, n3 - 1, 0, n2 - 1,
      KOKKOS_LAMBDA(const int m, const int n, const int k, const int j) {
        if (mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::axis &&
            !FillConstraintAxisGhostLine(constraints, m, n, k, j, is, ng)) {
          Kokkos::abort("invalid constraint component in axis parity fill");
        }
      });
  Kokkos::fence();
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::SendU
//! \brief sends cell-centered conserved variables

TaskStatus Z4c::SendU(Driver *pdrive, int stage) {
  TaskStatus tstat = layout.centering == Z4cGridCentering::vertex
      ? pbval_u_vc->PackAndSendVC(u0, coarse_u0)
      : pbval_u->PackAndSendCC(u0, coarse_u0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::RecvU
//! \brief receives cell-centered conserved variables

TaskStatus Z4c::RecvU(Driver *pdrive, int stage) {
  TaskStatus tstat = layout.centering == Z4cGridCentering::vertex
      ? pbval_u_vc->RecvAndUnpackVC(u0, coarse_u0)
      : pbval_u->RecvAndUnpackCC(u0, coarse_u0);
  if (tstat == TaskStatus::complete &&
      layout.centering == Z4cGridCentering::vertex) {
    vertex_topology_plan->SynchronizeSharedNodes(u0);
  }
  if (tstat == TaskStatus::complete && chi_parent_provenance != nullptr) {
    chi_parent_provenance->RecordCheckpoint(
        ChiProvenanceCheckpoint::s2_after_receive, stage, pbval_u);
  }
  if (tstat == TaskStatus::complete) {
    CheckStateAdmissibility(pdrive, stage, Z4cStateCheckpoint::post_receive);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::EnforceAlgConstr
//! \brief

TaskStatus Z4c::EnforceAlgConstr(Driver *pdrive, int stage) {
  if (pmy_pack->pdyngr != nullptr || stage == pdrive->nexp_stages) {
    AlgConstr(pmy_pack, pdrive, stage);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \brief Project the accepted native-VC state only after its first canonical
//! shared-node synchronization, then rebuild every derived coarse/ghost value.

TaskStatus Z4c::FinalizeVertexAcceptedState(Driver *pdrive, int stage) {
  if (layout.centering != Z4cGridCentering::vertex ||
      stage != pdrive->nexp_stages) {
    return TaskStatus::complete;
  }

  AlgConstr(pmy_pack, pdrive, stage);
  ApplyVertexAxisRegularity(u0, stage, "post_accepted_projection");
  vertex_topology_plan->SynchronizeSharedNodes(u0);

  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictVC(u0, coarse_u0);
  }

  // Reuse the same native boundary objects only after the first pass's requests
  // are complete.  This blocking accepted-state pass runs once per full RK step.
  (void)pbval_u_vc->ClearSend();
  (void)pbval_u_vc->ClearRecv();
  (void)pbval_u_vc->InitRecv(nz4c);
  (void)pbval_u_vc->PackAndSendVC(u0, coarse_u0);
  (void)pbval_u_vc->ClearSend();
  (void)pbval_u_vc->ClearRecv();
  (void)pbval_u_vc->RecvAndUnpackVC(u0, coarse_u0);
  vertex_topology_plan->SynchronizeSharedNodes(u0);
  FillBuiltInPhysicalBoundaryGhosts();
  if (pmy_pack->pmesh->multilevel) {
    pbval_u_vc->ProlongateVC(u0, coarse_u0, opt.spatial_order, I_Z4C_CHI);
  }
  FillBuiltInPhysicalBoundaryGhosts();
  ApplyVertexAxisRegularity(u0, stage, "post_accepted_boundary");
  ReconstructAxisParityGhosts();
  CheckStateAdmissibility(pdrive, stage,
                          Z4cStateCheckpoint::post_amr_transfer);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::ADMToZ4c_
//! \brief

TaskStatus Z4c::ConvertZ4cToADM(Driver *pdrive, int stage) {
  if (pmy_pack->pdyngr != nullptr || stage == pdrive->nexp_stages) {
    Z4cToADM(pmy_pack);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void Z4c::UpdateExcisionMasks
//! \brief

TaskStatus Z4c::UpdateExcisionMasks(Driver *pdrive, int stage) {
  if (pmy_pack->pcoord->coord_data.bh_excise && stage == pdrive->nexp_stages) {
    pmy_pack->pcoord->UpdateExcisionMasks();
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::ADM_Constraints_
//! \brief

TaskStatus Z4c::ADMConstraints_(Driver *pdrive, int stage) {
  if (stage == pdrive->nexp_stages) {
    switch (opt.fd_stencil) {
      case 2: ADMConstraints<2>(pmy_pack);
              break;
      case 3: ADMConstraints<3>(pmy_pack);
              break;
      case 4: ADMConstraints<4>(pmy_pack);
              break;
      default:
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl
                  << "Unsupported Z4c finite-difference stencil selector "
                  << opt.fd_stencil << std::endl;
        std::exit(EXIT_FAILURE);
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::RestrictU
//! \brief

TaskStatus Z4c::RestrictU(Driver *pdrive, int stage) {
  // Only execute Mesh function with SMR/SMR
  if (pmy_pack->pmesh->multilevel) {
    if (layout.centering == Z4cGridCentering::vertex) {
      // RK updates are block-local.  Reconcile coincident active vertices before
      // injection so every fine child presents one canonical point value to its
      // parent, independently of MeshBlock or MPI ownership.
      vertex_topology_plan->SynchronizeSharedNodes(u0);
      pmy_pack->pmesh->pmr->RestrictVC(u0, coarse_u0);
    } else {
      pmy_pack->pmesh->pmr->RestrictCC(u0, coarse_u0, true);
    }
  }
  if (chi_parent_provenance != nullptr) {
    chi_parent_provenance->RecordCheckpoint(
        ChiProvenanceCheckpoint::s1_after_restriction, stage, pbval_u);
  }
  CheckStateAdmissibility(pdrive, stage, Z4cStateCheckpoint::post_restriction);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Z4c::Prolongate
//! \brief Wrapper task list function to prolongate conserved (or primitive) variables
//! at fine/coarse boundaries with SMR/AMR

TaskStatus Z4c::Prolongate(Driver *pdrive, int stage) {
  if (pmy_pack->pmesh->multilevel) {  // only prolongate with SMR/AMR
    if (layout.centering == Z4cGridCentering::vertex) {
      pbval_u_vc->ProlongateVC(u0, coarse_u0, opt.spatial_order,
                               I_Z4C_CHI);
      CheckStateAdmissibility(pdrive, stage,
                              Z4cStateCheckpoint::post_prolongation);
      CheckStateAdmissibility(pdrive, stage,
                              Z4cStateCheckpoint::post_amr_transfer);
      return TaskStatus::complete;
    }
    if (amr_jump_diagnostic != nullptr && stage > 0) {
      amr_jump_diagnostic->RecordRKStageCoarseFineExposure(stage);
    }
    // Populate only coarse-cache locations owned by this boundary operation.
    // For Z4c, same-level overlap is owner-authoritative after receive/local
    // copy, so FillCoarseInBndryCC deliberately preserves it; generic
    // finite-volume users retain their receiver-local refresh policy.
    pbval_u->FillCoarseInBndryCC(u0, coarse_u0, true);
    if (chi_parent_provenance != nullptr) {
      chi_parent_provenance->RecordCheckpoint(
          ChiProvenanceCheckpoint::s4_before_parent_gate, stage, pbval_u);
    }
    if (amr_jump_diagnostic != nullptr) {
      amr_jump_diagnostic->RecordSameLevelRefreshShadow();
      amr_jump_diagnostic->RecordT3(
          AMRJumpWriter::same_level_coarse_refresh, 3, false);
    }
    pbval_u->ProlongateCC(u0, coarse_u0, true);
    if (amr_jump_diagnostic != nullptr) {
      amr_jump_diagnostic->RecordT3(
          AMRJumpWriter::coarse_to_fine_prolongation, 4, false);
    }
  }
  CheckStateAdmissibility(pdrive, stage, Z4cStateCheckpoint::post_prolongation);
  CheckStateAdmissibility(pdrive, stage, Z4cStateCheckpoint::post_amr_transfer);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void Z4c::FillBuiltInPhysicalBoundaryGhosts
//! \brief Apply only the built-in Z4c physical boundary policy.
//!
//! This is separate from the task wrapper so AMR initialization can compose a second
//! physical-face pass after coarse/fine prolongation without invoking a user callback twice.

void Z4c::FillBuiltInPhysicalBoundaryGhosts() {
  if (!(pmy_pack->pmesh->strictly_periodic)) {
    if (layout.centering == Z4cGridCentering::vertex) {
      pbval_u_vc->Z4cBCs(pmy_pack, pbval_u_vc->u_in, u0, coarse_u0);
    } else {
      pbval_u->Z4cBCs(pmy_pack, pbval_u->u_in, u0, coarse_u0);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Z4c::ApplyPhysicalBCs
//! \brief Apply built-in and user Z4c physical boundary conditions once per task pass.

TaskStatus Z4c::ApplyPhysicalBCs(Driver *pdrive, int stage) {
  FillBuiltInPhysicalBoundaryGhosts();

  // Preserve the historical user callback count: once in the public task wrapper only.
  if (!(pmy_pack->pmesh->strictly_periodic)) {
    if (pmy_pack->pmesh->pgen->user_bcs) {
      (pmy_pack->pmesh->pgen->user_bcs_func)(pmy_pack->pmesh);
    }
  }
  if (chi_parent_provenance != nullptr) {
    chi_parent_provenance->RecordCheckpoint(
        ChiProvenanceCheckpoint::s3_after_boundary, stage, pbval_u);
  }
  CheckStateAdmissibility(pdrive, stage, Z4cStateCheckpoint::post_physical_bc);
  return TaskStatus::complete;
}

TaskStatus Z4c::TrackCompactObjects(Driver *pdrive, int stage) {
  if (stage == pdrive->nexp_stages) {
    for (auto & pt : ptracker) {
      pt->InterpolateVelocity(pmy_pack);
      pt->EvolveTracker(pmy_pack);
      pt->WriteTracker();
    }
  }
  return TaskStatus::complete;
}

TaskStatus Z4c::FindHorizon(Driver *pdrive, int stage) {
  if (stage == pdrive->nexp_stages) {
    const int accepted_cycle = pmy_pack->pmesh->ncycle + 1;
    const Real accepted_time = pmy_pack->pmesh->time + pmy_pack->pmesh->dt;
    for (auto & pahf : pfastflow) {
      if (!pahf->ShouldSearch(accepted_cycle, accepted_time)) continue;
      Kokkos::Timer horizon_timer;
      pahf->Find(accepted_cycle, accepted_time);
      pahf->Write(accepted_cycle, accepted_time);
      Real horizon_seconds = horizon_timer.seconds();
#if MPI_PARALLEL_ENABLED
      MPI_Allreduce(MPI_IN_PLACE, &horizon_seconds, 1, MPI_ATHENA_REAL, MPI_MAX,
                    MPI_COMM_WORLD);
#endif
      if (global_variable::my_rank == 0) {
        std::cout << "FastFlow wall time: cycle=" << accepted_cycle
                  << " time=" << accepted_time
                  << " seconds=" << horizon_seconds << std::endl;
      }
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
// ! \fn TaskList CCEDump
// ! \brief CCE initial data for Pittnull code (cce dumps for Pittnull).

TaskStatus Z4c::CCEDump(Driver *pdrive, int stage) {
  float time_32 = static_cast<float>(pmy_pack->pmesh->time);
  float next_32 = static_cast<float>(cce_dump_last_output_time+cce_dump_dt);
  if ((time_32 >= next_32)) {
    if (stage == pdrive->nexp_stages) {
      //printf("%s:(ctime,dt)=(%f,%f)",__func__,pmy_pack->pmesh->time,cce_dump_dt);
      for (auto cce : pmy_pack->pz4c_cce) {
        cce->InterpolateAndDecompose(pmy_pack);
      }
      cce_dump_last_output_time = time_32;
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void Z4c:Z4cFloorChi
//! \brief Floor chi, to prevent negative propagation.
TaskStatus Z4c::Z4cFloorChi(Driver *pdrive, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  int nmb = pmy_pack->nmb_thispack;

  auto &z4c = pmy_pack->pz4c->z4c;
  auto &opt = pmy_pack->pz4c->opt;

  par_for("z4c_floor_chi",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real chi = z4c.chi(m,k,j,i);

    if (!Kokkos::isfinite(chi) || chi < opt.chi_min_floor) {
      z4c.chi(m,k,j,i) = opt.chi_min_floor;
    }
  });

  return TaskStatus::complete;
}


//----------------------------------------------------------------------------------------
//! \fn  void Z4c::CalcWeylScalar_
//! \brief

TaskStatus Z4c::CalcWeylScalar(Driver *pdrive, int stage) {
  if (pmy_pack->pz4c->nrad == 0) {
    return TaskStatus::complete;
  } else {
    float time_32 = static_cast<float>(pmy_pack->pmesh->time);
    if (last_output_time==time_32 && stage == pdrive->nexp_stages) {
      switch (opt.fd_stencil) {
        case 2: Z4cWeyl<2>(pmy_pack);
                break;
        case 3: Z4cWeyl<3>(pmy_pack);
                break;
        case 4: Z4cWeyl<4>(pmy_pack);
                break;
        default:
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl
                    << "Unsupported Z4c finite-difference stencil selector "
                    << opt.fd_stencil << std::endl;
          std::exit(EXIT_FAILURE);
      }
    }
    return TaskStatus::complete;
  }
}

TaskStatus Z4c::CalcWaveForm(Driver *pdrive, int stage) {
  if (pmy_pack->pz4c->nrad == 0) {
    return TaskStatus::complete;
  } else {
    float time_32 = static_cast<float>(pmy_pack->pmesh->time);
    if ((last_output_time==time_32) && (stage == pdrive->nexp_stages)) {
      WaveExtr(pmy_pack);
    }
    return TaskStatus::complete;
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::SendWeyl
//! \brief sends cell-centered conserved variables

TaskStatus Z4c::SendWeyl(Driver *pdrive, int stage) {
  if (pmy_pack->pz4c->nrad == 0) {
    return TaskStatus::complete;
  } else {
    float time_32 = static_cast<float>(pmy_pack->pmesh->time);
    if ((last_output_time==time_32) && (stage == pdrive->nexp_stages)) {
      TaskStatus tstat = layout.centering == Z4cGridCentering::vertex
          ? pbval_weyl_vc->PackAndSendVC(u_weyl, coarse_u_weyl)
          : pbval_weyl->PackAndSendCC(u_weyl, coarse_u_weyl);
      return tstat;
    } else {
      return TaskStatus::complete;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::RecvWeyl
//! \brief receives cell-centered conserved variables

TaskStatus Z4c::RecvWeyl(Driver *pdrive, int stage) {
  if (pmy_pack->pz4c->nrad == 0) {
    return TaskStatus::complete;
  } else {
    float time_32 = static_cast<float>(pmy_pack->pmesh->time);
    if ((last_output_time==time_32) && (stage == pdrive->nexp_stages)) {
      TaskStatus tstat = layout.centering == Z4cGridCentering::vertex
          ? pbval_weyl_vc->RecvAndUnpackVC(u_weyl, coarse_u_weyl)
          : pbval_weyl->RecvAndUnpackCC(u_weyl, coarse_u_weyl);
      if (tstat == TaskStatus::complete &&
          layout.centering == Z4cGridCentering::vertex) {
        vertex_topology_plan->SynchronizeSharedNodes(u_weyl);
      }
      return tstat;
    } else {
      return TaskStatus::complete;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::RestrictU
//! \brief

TaskStatus Z4c::RestrictWeyl(Driver *pdrive, int stage) {
  if (pmy_pack->pz4c->nrad == 0) {
    return TaskStatus::complete;
  } else {
    float time_32 = static_cast<float>(pmy_pack->pmesh->time);
    if ((last_output_time==time_32) && (stage == pdrive->nexp_stages)) {
      if (pmy_pack->pmesh->multilevel) {
        if (layout.centering == Z4cGridCentering::vertex) {
          pmy_pack->pmesh->pmr->RestrictVC(u_weyl, coarse_u_weyl);
        } else {
          pmy_pack->pmesh->pmr->RestrictCC(u_weyl, coarse_u_weyl, true);
        }
      }
    }
    return TaskStatus::complete;
  }
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Z4c::ProlongateWeyl
//! \brief Wrapper task list function to prolongate weyl scalar
//! at fine/coarse boundaries with SMR/AMR

TaskStatus Z4c::ProlongateWeyl(Driver *pdrive, int stage) {
  if (pmy_pack->pz4c->nrad == 0) {
    return TaskStatus::complete;
  } else {
    float time_32 = static_cast<float>(pmy_pack->pmesh->time);
    if ((last_output_time==time_32) && (stage == pdrive->nexp_stages)) {
      if (pmy_pack->pmesh->multilevel) {
        if (layout.centering == Z4cGridCentering::vertex) {
          pbval_weyl_vc->ProlongateVC(u_weyl, coarse_u_weyl,
                                      opt.spatial_order);
        } else {
          pbval_weyl->ProlongateCC(u_weyl, coarse_u_weyl);
        }
      }
    }
    return TaskStatus::complete;
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void Wave::InitRecvWeyl
//! \brief function to post non-blocking receives (with MPI), and initialize all boundary
//  receive status flags to waiting (with or without MPI) for Wave variables.

TaskStatus Z4c::InitRecvWeyl(Driver *pdrive, int stage) {
  if (pmy_pack->pz4c->nrad == 0) {
    return TaskStatus::complete;
  } else {
    float time_32 = static_cast<float>(pmy_pack->pmesh->time);
    float next_32 = static_cast<float>(last_output_time+waveform_dt);
    if (((time_32 >= next_32) || (time_32 == 0)) && stage == pdrive->nexp_stages) {
      last_output_time = time_32;
      TaskStatus tstat = layout.centering == Z4cGridCentering::vertex
          ? pbval_weyl_vc->InitRecv(2) : pbval_weyl->InitRecv(2);
      return tstat;
    } else {
      return TaskStatus::complete;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void Wave::ClearRecvWeyl
//! \brief Waits for all MPI receives to complete before allowing execution to continue

TaskStatus Z4c::ClearRecvWeyl(Driver *pdrive, int stage) {
  if (pmy_pack->pz4c->nrad == 0) {
    return TaskStatus::complete;
  } else {
    float time_32 = static_cast<float>(pmy_pack->pmesh->time);
    if ((last_output_time==time_32) && (stage == pdrive->nexp_stages)) {
      TaskStatus tstat = layout.centering == Z4cGridCentering::vertex
          ? pbval_weyl_vc->ClearRecv() : pbval_weyl->ClearRecv();
      return tstat;
    } else {
      return TaskStatus::complete;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::ClearSendWeyl
//! \brief Waits for all MPI sends to complete before allowing execution to continue

TaskStatus Z4c::ClearSendWeyl(Driver *pdrive, int stage) {
  if (pmy_pack->pz4c->nrad == 0) {
    return TaskStatus::complete;
  } else {
    float time_32 = static_cast<float>(pmy_pack->pmesh->time);
    if ((last_output_time==time_32) && (stage == pdrive->nexp_stages)) {
      TaskStatus tstat = layout.centering == Z4cGridCentering::vertex
          ? pbval_weyl_vc->ClearSend() : pbval_weyl->ClearSend();
      return tstat;
    } else {
      return TaskStatus::complete;
    }
  }
}

TaskStatus Z4c::DumpHorizons(Driver *pdrive, int stage) {
  if (pmy_pack->pz4c->phorizon_dump.size() == 0 || stage != pdrive->nexp_stages) {
    return TaskStatus::complete;
  } else {
    float time_32 = static_cast<float>(pmy_pack->pmesh->time);
    float next_32 = static_cast<float>(pmy_pack->pz4c->phorizon_dump[0]
                                    ->horizon_last_output_time
                                    +pmy_pack->pz4c->phorizon_dump[0]->horizon_dt);
    if (((time_32 >= next_32) || (time_32 == 0))) {
      int i = 0;
      for (auto & hd : phorizon_dump) {
        hd->horizon_last_output_time = time_32;
        hd->SetGridAndInterpolate(pmy_pack->pz4c->ptracker[i]->GetPos());
        i++;
      }
    }
    return TaskStatus::complete;
  }

  return TaskStatus::complete;
}

} // namespace z4c
