//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pc_gh_tasks.cpp
//! \brief task-graph integration and communication wrappers for PC-GH

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "bvals/bvals.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock_pack.hpp"
#include "pc_gh/pc_gh.hpp"
#include "tasklist/numerical_relativity.hpp"

namespace pc_gh {

void PcGh::QueuePcGhTasks() {
  using namespace numrel;  // NOLINT(build/namespaces)
  NumericalRelativity *pnr = pmy_pack->pnr;

  pnr->QueueTask(&PcGh::InitRecv, this, PcGh_IRecv, "PcGh_IRecv", Task_Start);

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
  pnr->QueueTask(&PcGh::ConvertToADM, this, PcGh_ToADM, "PcGh_ToADM", Task_Run,
                 {PcGh_AlgC});
  pnr->QueueTask(&PcGh::NewTimeStep, this, PcGh_Newdt, "PcGh_Newdt", Task_Run,
                 {PcGh_ToADM});

  pnr->QueueTask(&PcGh::ClearSend, this, PcGh_ClearS, "PcGh_ClearS", Task_End);
  pnr->QueueTask(&PcGh::ClearRecv, this, PcGh_ClearR, "PcGh_ClearR", Task_End,
                 {PcGh_ClearS});
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

TaskStatus PcGh::SendU(Driver *, int) {
  return pbval_u->PackAndSendCC(u0, coarse_u0);
}

TaskStatus PcGh::RecvU(Driver *, int) {
  return pbval_u->RecvAndUnpackCC(u0, coarse_u0);
}

TaskStatus PcGh::RestrictU(Driver *, int) {
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(u0, coarse_u0, true);
  }
  return TaskStatus::complete;
}

TaskStatus PcGh::Prolongate(Driver *, int) {
  if (pmy_pack->pmesh->multilevel) {
    pbval_u->ProlongateCC(u0, coarse_u0, true);
  }
  return TaskStatus::complete;
}

TaskStatus PcGh::ApplyPhysicalBCs(Driver *, int) {
  if (!pmy_pack->pmesh->strictly_periodic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << '\n'
              << "Nonperiodic PC-GH state boundary conditions are not implemented"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return TaskStatus::complete;
}

TaskStatus PcGh::BoundaryRHS(Driver *, int) {
  // Periodic domains require no separate RHS boundary operation.  This explicit task
  // remains the fail-closed insertion point for a later derived GH characteristic BC.
  return TaskStatus::complete;
}

TaskStatus PcGh::EnforceAlgebraicConstraints(Driver *, int) {
  ProjectAlgebraic(pmy_pack);
  return TaskStatus::complete;
}

TaskStatus PcGh::ConvertToADM(Driver *pdriver, int stage) {
  if (stage == pdriver->nexp_stages) PcGhToADM(pmy_pack);
  return TaskStatus::complete;
}

}  // namespace pc_gh
