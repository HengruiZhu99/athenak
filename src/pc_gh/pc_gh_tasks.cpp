//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pc_gh_tasks.cpp
//! \brief task-graph integration and communication wrappers for PC-GH

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "bvals/bvals.hpp"
#include "coordinates/cell_locations.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock_pack.hpp"
#include "pc_gh/pc_gh.hpp"
#include "tasklist/numerical_relativity.hpp"
#include "utils/horizon_dump.hpp"
#include "utils/compact_object_tracker.hpp"
#include "utils/finite_diff.hpp"
#include "utils/gr_wave.hpp"

namespace pc_gh {
namespace {

KOKKOS_INLINE_FUNCTION
Real FlatPcGhValue(int n) {
  return (n == PcGh::I_W || n == PcGh::I_RHO
          || n == PcGh::I_GTXX || n == PcGh::I_GTYY || n == PcGh::I_GTZZ)
      ? 1.0 : 0.0;
}

KOKKOS_INLINE_FUNCTION
void PcGhSommerfeld(DvceArray5D<Real> state, DvceArray5D<Real> state_rhs,
                    const RegionIndcs &indcs, const DualArray1D<RegionSize> &size,
                    int m, int k, int j, int i) {
  Real const x = CellCenterX(i - indcs.is, indcs.nx1,
                              size.d_view(m).x1min, size.d_view(m).x1max);
  Real const y = CellCenterX(j - indcs.js, indcs.nx2,
                              size.d_view(m).x2min, size.d_view(m).x2max);
  Real const z = CellCenterX(k - indcs.ks, indcs.nx3,
                              size.d_view(m).x3min, size.d_view(m).x3max);
  Real const radius = std::sqrt(x*x + y*y + z*z);
  Real const inv_radius = 1.0/radius;
  Real const inverse_spacing[3] = {1.0/size.d_view(m).dx1,
                                    1.0/size.d_view(m).dx2,
                                    1.0/size.d_view(m).dx3};
  Real const normal[3] = {x*inv_radius, y*inv_radius, z*inv_radius};
  for (int n = 0; n < PcGh::npcgh; ++n) {
    Real radial_derivative = 0.0;
    for (int d = 0; d < 3; ++d) {
      radial_derivative += normal[d]*Dx<2>(
          d, inverse_spacing, state, m, n, k, j, i);
    }
    state_rhs(m, n, k, j, i) = -radial_derivative
        - (state(m, n, k, j, i) - FlatPcGhValue(n))*inv_radius;
  }
}

}  // namespace

void PcGh::QueuePcGhTasks() {
  using namespace numrel;  // NOLINT(build/namespaces)
  NumericalRelativity *pnr = pmy_pack->pnr;

  pnr->QueueTask(&PcGh::InitRecv, this, PcGh_IRecv, "PcGh_IRecv", Task_Start);
  pnr->QueueTask(&PcGh::InitRecvWeyl, this, PcGh_IRecvW, "PcGh_IRecvW",
                 Task_Start);

  pnr->QueueTask(&PcGh::CopyU, this, PcGh_CopyU, "PcGh_CopyU", Task_Run);
  switch (opt.fd_stencil) {
    case 2:
      pnr->QueueTask(&PcGh::CalcRHS<2>, this, PcGh_CalcRHS, "PcGh_CalcRHS",
                     Task_Run, {PcGh_CopyU});
      break;
    case 3:
      pnr->QueueTask(&PcGh::CalcRHS<3>, this, PcGh_CalcRHS, "PcGh_CalcRHS",
                     Task_Run, {PcGh_CopyU});
      break;
    case 4:
      pnr->QueueTask(&PcGh::CalcRHS<4>, this, PcGh_CalcRHS, "PcGh_CalcRHS",
                     Task_Run, {PcGh_CopyU});
      break;
    default:
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << "\nUnsupported PC-GH stencil selector " << opt.fd_stencil
                << std::endl;
      std::exit(EXIT_FAILURE);
  }
  pnr->QueueTask(&PcGh::BoundaryRHS, this, PcGh_BoundaryRHS, "PcGh_BoundaryRHS",
                 Task_Run, {PcGh_CalcRHS});
  pnr->QueueTask(&PcGh::ExpRKUpdate, this, PcGh_ExplRK, "PcGh_ExplRK", Task_Run,
                 {PcGh_BoundaryRHS});
  pnr->QueueTask(&PcGh::RestrictU, this, PcGh_RestU, "PcGh_RestU", Task_Run,
                 {PcGh_ExplRK});
  pnr->QueueTask(&PcGh::SendU, this, PcGh_SendU, "PcGh_SendU", Task_Run,
                 {PcGh_RestU});
  pnr->QueueTask(&PcGh::RecvU, this, PcGh_RecvU, "PcGh_RecvU", Task_Run,
                 {PcGh_SendU});
  pnr->QueueTask(&PcGh::ApplyPhysicalBCs, this, PcGh_BCS, "PcGh_BCS", Task_Run,
                 {PcGh_RecvU});
  pnr->QueueTask(&PcGh::Prolongate, this, PcGh_Prolong, "PcGh_Prolong", Task_Run,
                 {PcGh_BCS});
  pnr->QueueTask(&PcGh::EnforceAlgebraicConstraints, this, PcGh_AlgC, "PcGh_AlgC",
                 Task_Run, {PcGh_Prolong});
  pnr->QueueTask(&PcGh::PrepareProjectionExchange, this, PcGh_PrepP,
                 "PcGh_PrepP", Task_Run, {PcGh_AlgC});
  pnr->QueueTask(&PcGh::EnforceReductionConstraints, this, PcGh_RedC,
                 "PcGh_RedC", Task_Run, {PcGh_PrepP});
  pnr->QueueTask(&PcGh::RestrictProjection, this, PcGh_RestP, "PcGh_RestP",
                 Task_Run, {PcGh_RedC});
  pnr->QueueTask(&PcGh::SendProjection, this, PcGh_SendP, "PcGh_SendP",
                 Task_Run, {PcGh_RestP});
  pnr->QueueTask(&PcGh::RecvProjection, this, PcGh_RecvP, "PcGh_RecvP",
                 Task_Run, {PcGh_SendP});
  pnr->QueueTask(&PcGh::ApplyProjectionPhysicalBCs, this, PcGh_BCSP,
                 "PcGh_BCSP", Task_Run, {PcGh_RecvP});
  pnr->QueueTask(&PcGh::ProlongateProjection, this, PcGh_ProlP, "PcGh_ProlP",
                 Task_Run, {PcGh_BCSP});
  pnr->QueueTask(&PcGh::ConvertToADM, this, PcGh_ToADM, "PcGh_ToADM", Task_Run,
                 {PcGh_ProlP});
  pnr->QueueTask(&PcGh::NewTimeStep, this, PcGh_Newdt, "PcGh_Newdt", Task_Run,
                 {PcGh_ToADM});

  pnr->QueueTask(&PcGh::ClearSend, this, PcGh_ClearS, "PcGh_ClearS", Task_End);
  pnr->QueueTask(&PcGh::ClearRecv, this, PcGh_ClearR, "PcGh_ClearR", Task_End,
                 {PcGh_ClearS});
  switch (opt.fd_stencil) {
    case 2:
      pnr->QueueTask(&PcGh::CalcConstraints<2>, this, PcGh_Constraints,
                     "PcGh_Constraints", Task_End, {PcGh_ClearR});
      break;
    case 3:
      pnr->QueueTask(&PcGh::CalcConstraints<3>, this, PcGh_Constraints,
                     "PcGh_Constraints", Task_End, {PcGh_ClearR});
      break;
    case 4:
      pnr->QueueTask(&PcGh::CalcConstraints<4>, this, PcGh_Constraints,
                     "PcGh_Constraints", Task_End, {PcGh_ClearR});
      break;
    default:
      std::abort();
  }
  pnr->QueueTask(&PcGh::CalcWeylScalar, this, PcGh_Weyl, "PcGh_Weyl",
                 Task_End, {PcGh_Constraints});
  pnr->QueueTask(&PcGh::RestrictWeyl, this, PcGh_RestW, "PcGh_RestW",
                 Task_End, {PcGh_Weyl});
  pnr->QueueTask(&PcGh::SendWeyl, this, PcGh_SendW, "PcGh_SendW",
                 Task_End, {PcGh_RestW});
  pnr->QueueTask(&PcGh::RecvWeyl, this, PcGh_RecvW, "PcGh_RecvW",
                 Task_End, {PcGh_SendW});
  pnr->QueueTask(&PcGh::ProlongateWeyl, this, PcGh_ProlW, "PcGh_ProlW",
                 Task_End, {PcGh_RecvW});
  pnr->QueueTask(&PcGh::ClearSendWeyl, this, PcGh_ClearSW, "PcGh_ClearSW",
                 Task_End, {PcGh_ProlW});
  pnr->QueueTask(&PcGh::ClearRecvWeyl, this, PcGh_ClearRW, "PcGh_ClearRW",
                 Task_End, {PcGh_ClearSW});
  pnr->QueueTask(&PcGh::CalcWaveForm, this, PcGh_Wave, "PcGh_Wave",
                 Task_End, {PcGh_ClearRW});
  pnr->QueueTask(&PcGh::TrackCompactObjects, this, PcGh_PT, "PcGh_PT",
                 Task_End, {PcGh_Wave});
  pnr->QueueTask(&PcGh::DumpHorizons, this, PcGh_DumpHorizon,
                 "PcGh_DumpHorizon", Task_End, {PcGh_PT});
}

TaskStatus PcGh::InitRecv(Driver *, int) {
  return pbval_u->InitRecv(npcgh);
}

TaskStatus PcGh::ClearRecv(Driver *, int) {
  return pbval_u->ClearRecv();
}

TaskStatus PcGh::ClearSend(Driver *, int) {
  return pbval_u->ClearSend();
}

TaskStatus PcGh::InitRecvWeyl(Driver *pdriver, int stage) {
  if (nrad == 0) return TaskStatus::complete;
  float const time_32 = static_cast<float>(pmy_pack->pmesh->time);
  float const next_32 = static_cast<float>(last_waveform_time + waveform_dt);
  if ((time_32 >= next_32 || time_32 == 0.0F)
      && stage == pdriver->nexp_stages) {
    last_waveform_time = time_32;
    return pbval_weyl->InitRecv(2);
  }
  return TaskStatus::complete;
}

TaskStatus PcGh::ClearRecvWeyl(Driver *pdriver, int stage) {
  float const time_32 = static_cast<float>(pmy_pack->pmesh->time);
  if (nrad > 0 && last_waveform_time == time_32
      && stage == pdriver->nexp_stages) return pbval_weyl->ClearRecv();
  return TaskStatus::complete;
}

TaskStatus PcGh::ClearSendWeyl(Driver *pdriver, int stage) {
  float const time_32 = static_cast<float>(pmy_pack->pmesh->time);
  if (nrad > 0 && last_waveform_time == time_32
      && stage == pdriver->nexp_stages) return pbval_weyl->ClearSend();
  return TaskStatus::complete;
}

TaskStatus PcGh::SendU(Driver *, int) {
  if (opt.reduction_monitor) BeginReductionTransfer(7);
  return pbval_u->PackAndSendCC(u0, coarse_u0);
}

TaskStatus PcGh::RecvU(Driver *, int) {
  TaskStatus const status = pbval_u->RecvAndUnpackCC(u0, coarse_u0);
  if (opt.reduction_monitor && status == TaskStatus::complete) EndReductionTransfer(7);
  return status;
}

TaskStatus PcGh::SendProjection(Driver *pdriver, int stage) {
  if (!opt.project_reduction_constraints || stage != pdriver->nexp_stages) {
    return TaskStatus::complete;
  }
  bool const monitor = opt.reduction_monitor || (opt.boundedness_output
      && (pmy_pack->pmesh->ncycle + 1) % opt.boundedness_dcycle == 0);
  if (monitor) BeginReductionTransfer(5);
  return pbval_u->PackAndSendCC(u0, coarse_u0);
}

TaskStatus PcGh::RecvProjection(Driver *pdriver, int stage) {
  if (!opt.project_reduction_constraints || stage != pdriver->nexp_stages) {
    return TaskStatus::complete;
  }
  TaskStatus const status = pbval_u->RecvAndUnpackCC(u0, coarse_u0);
  bool const monitor = status == TaskStatus::complete && (opt.reduction_monitor
      || (opt.boundedness_output
      && (pmy_pack->pmesh->ncycle + 1) % opt.boundedness_dcycle == 0));
  if (monitor) EndReductionTransfer(5);
  return status;
}

TaskStatus PcGh::SendWeyl(Driver *pdriver, int stage) {
  float const time_32 = static_cast<float>(pmy_pack->pmesh->time);
  if (nrad > 0 && last_waveform_time == time_32
      && stage == pdriver->nexp_stages) {
    return pbval_weyl->PackAndSendCC(u_weyl, coarse_u_weyl);
  }
  return TaskStatus::complete;
}

TaskStatus PcGh::RecvWeyl(Driver *pdriver, int stage) {
  float const time_32 = static_cast<float>(pmy_pack->pmesh->time);
  if (nrad > 0 && last_waveform_time == time_32
      && stage == pdriver->nexp_stages) {
    return pbval_weyl->RecvAndUnpackCC(u_weyl, coarse_u_weyl);
  }
  return TaskStatus::complete;
}

TaskStatus PcGh::RestrictU(Driver *pdriver, int stage) {
  bool const monitor = opt.reduction_monitor || (stage == pdriver->nexp_stages
      && opt.boundedness_output
      && (pmy_pack->pmesh->ncycle + 1) % opt.boundedness_dcycle == 0);
  if (monitor) BeginReductionTransfer(0);
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(u0, coarse_u0, true);
  }
  if (monitor) EndReductionTransfer(0);
  return TaskStatus::complete;
}

TaskStatus PcGh::Prolongate(Driver *pdriver, int stage) {
  bool const monitor = opt.reduction_monitor || (stage == pdriver->nexp_stages
      && opt.boundedness_output
      && (pmy_pack->pmesh->ncycle + 1) % opt.boundedness_dcycle == 0);
  if (monitor) BeginReductionTransfer(1);
  if (pmy_pack->pmesh->multilevel) {
    pbval_u->ProlongateCC(u0, coarse_u0, true);
  }
  if (monitor) EndReductionTransfer(1);
  return TaskStatus::complete;
}

TaskStatus PcGh::RestrictWeyl(Driver *pdriver, int stage) {
  float const time_32 = static_cast<float>(pmy_pack->pmesh->time);
  if (nrad > 0 && last_waveform_time == time_32
      && stage == pdriver->nexp_stages && pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(u_weyl, coarse_u_weyl, true);
  }
  return TaskStatus::complete;
}

TaskStatus PcGh::RestrictProjection(Driver *pdriver, int stage) {
  bool const active = opt.project_reduction_constraints
      && stage == pdriver->nexp_stages && pmy_pack->pmesh->multilevel;
  bool const monitor = active && (opt.reduction_monitor || (opt.boundedness_output
      && (pmy_pack->pmesh->ncycle + 1) % opt.boundedness_dcycle == 0));
  if (active) {
    if (monitor) BeginReductionTransfer(4);
    pmy_pack->pmesh->pmr->RestrictCC(u0, coarse_u0, true);
    if (monitor) EndReductionTransfer(4);
  }
  return TaskStatus::complete;
}

TaskStatus PcGh::ProlongateWeyl(Driver *pdriver, int stage) {
  float const time_32 = static_cast<float>(pmy_pack->pmesh->time);
  if (nrad > 0 && last_waveform_time == time_32
      && stage == pdriver->nexp_stages && pmy_pack->pmesh->multilevel) {
    pbval_weyl->ProlongateCC(u_weyl, coarse_u_weyl);
  }
  return TaskStatus::complete;
}

TaskStatus PcGh::ProlongateProjection(Driver *pdriver, int stage) {
  bool const active = opt.project_reduction_constraints
      && stage == pdriver->nexp_stages && pmy_pack->pmesh->multilevel;
  bool const monitor = active && (opt.reduction_monitor || (opt.boundedness_output
      && (pmy_pack->pmesh->ncycle + 1) % opt.boundedness_dcycle == 0));
  if (active) {
    if (monitor) BeginReductionTransfer(6);
    pbval_u->ProlongateCC(u0, coarse_u0, true);
    if (monitor) EndReductionTransfer(6);
  }
  return TaskStatus::complete;
}

TaskStatus PcGh::ApplyPhysicalBCs(Driver *, int) {
  if (opt.reduction_monitor) BeginReductionTransfer(8);
  if (!pmy_pack->pmesh->strictly_periodic) {
    pbval_u->Z4cBCs(pmy_pack, pbval_u->u_in, u0, coarse_u0);
    if (pmy_pack->pmesh->pgen->user_bcs) {
      pmy_pack->pmesh->pgen->user_bcs_func(pmy_pack->pmesh);
    }
  }
  if (opt.reduction_monitor) EndReductionTransfer(8);
  return TaskStatus::complete;
}

TaskStatus PcGh::ApplyProjectionPhysicalBCs(Driver *pdriver, int stage) {
  if (!opt.project_reduction_constraints || stage != pdriver->nexp_stages) {
    return TaskStatus::complete;
  }
  if (!pmy_pack->pmesh->strictly_periodic) {
    pbval_u->Z4cBCs(pmy_pack, pbval_u->u_in, u0, coarse_u0);
    if (pmy_pack->pmesh->pgen->user_bcs) {
      pmy_pack->pmesh->pgen->user_bcs_func(pmy_pack->pmesh);
    }
  }
  return TaskStatus::complete;
}

TaskStatus PcGh::PrepareProjectionExchange(Driver *pdriver, int stage) {
  if (!opt.project_reduction_constraints || stage != pdriver->nexp_stages) {
    return TaskStatus::complete;
  }
  TaskStatus status = pbval_u->ClearSend();
  if (status != TaskStatus::complete) return status;
  status = pbval_u->ClearRecv();
  if (status != TaskStatus::complete) return status;
  return pbval_u->InitRecv(npcgh);
}

TaskStatus PcGh::BoundaryRHS(Driver *, int) {
  auto &mesh = pmy_pack->pmesh;
  if (mesh->strictly_periodic) return TaskStatus::complete;
  auto &mb_bcs = pmy_pack->pmb->mb_bcs;
  auto &indcs = mesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  int const nmb = pmy_pack->nmb_thispack;
  auto state = u0;
  auto state_rhs = u_rhs;

  par_for("PC-GH Sommerfeld x1", DevExeSpace(), 0, nmb - 1,
  indcs.ks, indcs.ke, indcs.js, indcs.je,
  KOKKOS_LAMBDA(int m, int k, int j) {
    if (mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::outflow) {
      PcGhSommerfeld(state, state_rhs, indcs, size, m, k, j, indcs.is);
    }
    if (mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::outflow) {
      PcGhSommerfeld(state, state_rhs, indcs, size, m, k, j, indcs.ie);
    }
  });
  if (mesh->multi_d) {
    par_for("PC-GH Sommerfeld x2", DevExeSpace(), 0, nmb - 1,
    indcs.ks, indcs.ke, indcs.is, indcs.ie,
    KOKKOS_LAMBDA(int m, int k, int i) {
      if (mb_bcs.d_view(m, BoundaryFace::inner_x2) == BoundaryFlag::outflow) {
        PcGhSommerfeld(state, state_rhs, indcs, size, m, k, indcs.js, i);
      }
      if (mb_bcs.d_view(m, BoundaryFace::outer_x2) == BoundaryFlag::outflow) {
        PcGhSommerfeld(state, state_rhs, indcs, size, m, k, indcs.je, i);
      }
    });
  }
  if (mesh->three_d) {
    par_for("PC-GH Sommerfeld x3", DevExeSpace(), 0, nmb - 1,
    indcs.js, indcs.je, indcs.is, indcs.ie,
    KOKKOS_LAMBDA(int m, int j, int i) {
      if (mb_bcs.d_view(m, BoundaryFace::inner_x3) == BoundaryFlag::outflow) {
        PcGhSommerfeld(state, state_rhs, indcs, size, m, indcs.ks, j, i);
      }
      if (mb_bcs.d_view(m, BoundaryFace::outer_x3) == BoundaryFlag::outflow) {
        PcGhSommerfeld(state, state_rhs, indcs, size, m, indcs.ke, j, i);
      }
    });
  }
  return TaskStatus::complete;
}

TaskStatus PcGh::EnforceAlgebraicConstraints(Driver *pdriver, int stage) {
  bool const monitor = opt.reduction_monitor || (stage == pdriver->nexp_stages
      && opt.boundedness_output
      && (pmy_pack->pmesh->ncycle + 1) % opt.boundedness_dcycle == 0);
  if (monitor) BeginReductionTransfer(2);
  ProjectAlgebraic(pmy_pack);
  if (opt.project_gauge_constraints) ProjectGaugeConstraints(pmy_pack);
  if (monitor) EndReductionTransfer(2);
  return TaskStatus::complete;
}

TaskStatus PcGh::EnforceReductionConstraints(Driver *pdriver, int stage) {
  if (!opt.project_reduction_constraints || stage != pdriver->nexp_stages) {
    return TaskStatus::complete;
  }
  bool const monitor = opt.reduction_monitor || (stage == pdriver->nexp_stages
      && opt.boundedness_output
      && (pmy_pack->pmesh->ncycle + 1) % opt.boundedness_dcycle == 0);
  if (monitor) BeginReductionTransfer(3);
  switch (opt.fd_stencil) {
    case 2: ProjectReduction<2>(pmy_pack); break;
    case 3: ProjectReduction<3>(pmy_pack); break;
    case 4: ProjectReduction<4>(pmy_pack); break;
    default: std::abort();
  }
  if (monitor) EndReductionTransfer(3);
  return TaskStatus::complete;
}

TaskStatus PcGh::ConvertToADM(Driver *pdriver, int stage) {
  // Evolution, CFL, regular Cartesian output, and the PC-GH horizon path do not need
  // physical ADM fields.  Enable this explicitly only for requested masked ADM output.
  if ((opt.reconstruct_adm_output || nrad > 0) && stage == pdriver->nexp_stages) {
    PcGhToADM(pmy_pack);
  }
  return TaskStatus::complete;
}

TaskStatus PcGh::CalcWeylScalar(Driver *pdriver, int stage) {
  float const time_32 = static_cast<float>(pmy_pack->pmesh->time);
  if (nrad == 0 || last_waveform_time != time_32
      || stage != pdriver->nexp_stages) return TaskStatus::complete;
  switch (opt.fd_stencil) {
    case 2: gr_wave::CalculateWeyl<2>(pmy_pack, u_weyl); break;
    case 3: gr_wave::CalculateWeyl<3>(pmy_pack, u_weyl); break;
    case 4: gr_wave::CalculateWeyl<4>(pmy_pack, u_weyl); break;
    default: std::abort();
  }
  return TaskStatus::complete;
}

TaskStatus PcGh::CalcWaveForm(Driver *pdriver, int stage) {
  float const time_32 = static_cast<float>(pmy_pack->pmesh->time);
  if (nrad > 0 && last_waveform_time == time_32
      && stage == pdriver->nexp_stages) {
    gr_wave::ExtractWaveform(
        pmy_pack, spherical_grids, u_weyl, psi_out, "waveforms");
  }
  return TaskStatus::complete;
}

TaskStatus PcGh::TrackCompactObjects(Driver *pdriver, int stage) {
  if (stage == pdriver->nexp_stages) {
    for (auto &tracker : ptracker) {
      tracker->InterpolateVelocity(pmy_pack);
      tracker->EvolveTracker(pmy_pack);
      tracker->WriteTracker();
    }
  }
  return TaskStatus::complete;
}

TaskStatus PcGh::DumpHorizons(Driver *pdriver, int stage) {
  if (phorizon_dump.empty() || stage != pdriver->nexp_stages) {
    return TaskStatus::complete;
  }
  float const time_32 = static_cast<float>(pmy_pack->pmesh->time);
  float const next_32 = static_cast<float>(
      phorizon_dump.front()->horizon_last_output_time
      + phorizon_dump.front()->horizon_dt);
  if (time_32 >= next_32 || time_32 == 0.0F) {
    for (auto &dump : phorizon_dump) {
      dump->horizon_last_output_time = time_32;
      dump->SetGridAndInterpolatePcGh(
          dump->pos, opt.physical_output_inner_radius);
    }
  }
  return TaskStatus::complete;
}

}  // namespace pc_gh
