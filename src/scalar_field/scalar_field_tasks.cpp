//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file scalar_field_tasks.cpp
//! \brief Task graph and boundary-communication wrappers for canonical scalar fields.

#include <cstdlib>
#include <iostream>
#include <vector>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/coordinates.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "scalar_field/scalar_field.hpp"
#include "tasklist/numerical_relativity.hpp"

namespace scalar_field {

//----------------------------------------------------------------------------------------
//! \fn void ScalarField::QueueScalarFieldTasks()
//! \brief Queue scalar-field tasks into the numerical-relativity task framework.

void ScalarField::QueueScalarFieldTasks() {
  using namespace numrel;  // NOLINT(build/namespaces)

  NumericalRelativity *pnr = pmy_pack->pnr;

  // Post receives before each RK stage.
  pnr->QueueTask(&ScalarField::InitRecv, this, SF_Recv, "SF_Recv", Task_Start);

  // Evolve the current stage, then synchronize the updated field and its ghost zones.
  pnr->QueueTask(&ScalarField::SetADM, this, SF_SetADM, "SF_SetADM", Task_Run);
  pnr->QueueTask(&ScalarField::CopyU, this, SF_CopyU, "SF_CopyU", Task_Run,
                 {SF_SetADM});
  switch (fd_stencil) {
    case 2:
      pnr->QueueTask(&ScalarField::CalcRHS<2>, this, SF_CalcRHS, "SF_CalcRHS",
                     Task_Run, {SF_CopyU});
      break;
    case 3:
      pnr->QueueTask(&ScalarField::CalcRHS<3>, this, SF_CalcRHS, "SF_CalcRHS",
                     Task_Run, {SF_CopyU});
      break;
    case 4:
      pnr->QueueTask(&ScalarField::CalcRHS<4>, this, SF_CalcRHS, "SF_CalcRHS",
                     Task_Run, {SF_CopyU});
      break;
    default:
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "Unsupported scalar-field finite-difference stencil selector "
                << fd_stencil << std::endl;
      std::exit(EXIT_FAILURE);
  }
  std::vector<TaskName> tmunu_dependencies = {SF_CalcRHS};
  std::vector<TaskName> tmunu_optional;
  if (pmy_pack->ptmunu != nullptr && pmy_pack->pz4c != nullptr) {
    tmunu_dependencies.push_back(Tmunu_Clear);
    tmunu_optional.push_back(MHD_AddTmunu);
  }
  switch (fd_stencil) {
    case 2:
      pnr->QueueTask(&ScalarField::AddTmunu<2>, this, SF_AddTmunu,
                     "SF_AddTmunu", Task_Run, tmunu_dependencies,
                     tmunu_optional);
      break;
    case 3:
      pnr->QueueTask(&ScalarField::AddTmunu<3>, this, SF_AddTmunu,
                     "SF_AddTmunu", Task_Run, tmunu_dependencies,
                     tmunu_optional);
      break;
    case 4:
      pnr->QueueTask(&ScalarField::AddTmunu<4>, this, SF_AddTmunu,
                     "SF_AddTmunu", Task_Run, tmunu_dependencies,
                     tmunu_optional);
      break;
  }
  pnr->QueueTask(&ScalarField::ExpRKUpdate, this, SF_ExplRK, "SF_ExplRK",
                 Task_Run, {SF_CalcRHS, SF_AddTmunu});
  pnr->QueueTask(&ScalarField::RestrictU, this, SF_RestU, "SF_RestU",
                 Task_Run, {SF_ExplRK});
  pnr->QueueTask(&ScalarField::SendU, this, SF_SendU, "SF_SendU",
                 Task_Run, {SF_RestU});
  pnr->QueueTask(&ScalarField::RecvU, this, SF_RecvU, "SF_RecvU",
                 Task_Run, {SF_SendU});
  pnr->QueueTask(&ScalarField::ApplyPhysicalBCs, this, SF_BCS, "SF_BCS",
                 Task_Run, {SF_RecvU});
  pnr->QueueTask(&ScalarField::Prolongate, this, SF_Prolong, "SF_Prolong",
                 Task_Run, {SF_BCS});
  pnr->QueueTask(&ScalarField::SetADMFinal, this, SF_SetADMFinal,
                 "SF_SetADMFinal", Task_Run, {SF_Prolong}, {MHD_Newdt});
  pnr->QueueTask(&ScalarField::NewTimeStep, this, SF_Newdt, "SF_Newdt",
                 Task_Run, {SF_SetADMFinal}, {Z4c_Z4c2ADM});

  // Complete nonblocking communication before the next RK stage posts new receives.
  pnr->QueueTask(&ScalarField::ClearSend, this, SF_ClearS, "SF_ClearS", Task_End);
  pnr->QueueTask(&ScalarField::ClearRecv, this, SF_ClearR, "SF_ClearR",
                 Task_End, {SF_ClearS});

  std::vector<TaskName> final_tmunu_dependencies;
  std::vector<TaskName> final_tmunu_optional;
  if (pmy_pack->ptmunu != nullptr && pmy_pack->pz4c != nullptr) {
    final_tmunu_dependencies.push_back(Tmunu_Clear);
    final_tmunu_optional.push_back(MHD_AddTmunu);
  }
  pnr->QueueTask(&ScalarField::AddTmunuFinal, this, SF_AddTmunu,
                 "SF_AddTmunuFinal", Task_End, final_tmunu_dependencies,
                 final_tmunu_optional);
}

//----------------------------------------------------------------------------------------
//! \brief Post nonblocking receives for scalar-field boundary data.

TaskStatus ScalarField::InitRecv(Driver *driver, int stage) {
  return pbval_u->InitRecv(nvar);
}

//----------------------------------------------------------------------------------------
//! \brief Wait for all scalar-field receives to complete.

TaskStatus ScalarField::ClearRecv(Driver *driver, int stage) {
  return pbval_u->ClearRecv();
}

//----------------------------------------------------------------------------------------
//! \brief Wait for all scalar-field sends to complete.

TaskStatus ScalarField::ClearSend(Driver *driver, int stage) {
  return pbval_u->ClearSend();
}

//----------------------------------------------------------------------------------------
//! \brief Refresh an externally prescribed ADM metric at this RHS stage time.

TaskStatus ScalarField::SetADM(Driver *driver, int stage) {
  if (pmy_pack->pz4c == nullptr && pmy_pack->padm->is_dynamic) {
    pmy_pack->padm->SetADMVariablesAtTime(
        pmy_pack, driver->StageTime(pmy_pack->pmesh, stage));
    if (excision) {
      pmy_pack->pcoord->UpdateExcisionMasks();
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \brief Leave prescribed ADM data synchronized with the completed cycle.

TaskStatus ScalarField::SetADMFinal(Driver *driver, int stage) {
  if (stage == driver->nexp_stages && pmy_pack->pz4c == nullptr &&
      pmy_pack->padm->is_dynamic) {
    pmy_pack->padm->SetADMVariablesAtTime(
        pmy_pack, driver->StageEndTime(pmy_pack->pmesh, stage));
    if (excision) {
      pmy_pack->pcoord->UpdateExcisionMasks();
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \brief Maintain the second low-storage RK register.

TaskStatus ScalarField::CopyU(Driver *driver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is;
  const int ie = indcs.ie;
  const int js = indcs.js;
  const int je = indcs.je;
  const int ks = indcs.ks;
  const int ke = indcs.ke;
  const int nmb1 = pmy_pack->nmb_thispack - 1;

  if (driver->integrator == "rk4") {
    if (stage == 1) {
      Kokkos::deep_copy(DevExeSpace(), u1, u0);
    } else {
      const Real delta = driver->delta[stage - 1];
      par_for(
          "scalar field copy RK register", DevExeSpace(), 0, nmb1, 0, nvar - 1,
          ks, ke, js, je, is, ie,
          KOKKOS_LAMBDA(const int m, const int n, const int k, const int j,
                        const int i) {
            u1(m, n, k, j, i) += delta*u0(m, n, k, j, i);
          });
    }
  } else if (stage == 1) {
    Kokkos::deep_copy(DevExeSpace(), u1, u0);
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \brief Restrict the current scalar state onto the coarse grid for SMR/AMR.

TaskStatus ScalarField::RestrictU(Driver *driver, int stage) {
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(u0, coarse_u0, true);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \brief Pack and send the current scalar state to neighboring MeshBlocks.

TaskStatus ScalarField::SendU(Driver *driver, int stage) {
  return pbval_u->PackAndSendCC(u0, coarse_u0);
}

//----------------------------------------------------------------------------------------
//! \brief Receive and unpack the current scalar state from neighboring MeshBlocks.

TaskStatus ScalarField::RecvU(Driver *driver, int stage) {
  return pbval_u->RecvAndUnpackCC(u0, coarse_u0);
}

//----------------------------------------------------------------------------------------
//! \brief Apply scalar physical boundaries and the enrolled user boundary callback.

TaskStatus ScalarField::ApplyPhysicalBCs(Driver *driver, int stage) {
  if (!(pmy_pack->pmesh->strictly_periodic)) {
    pbval_u->ScalarFieldBCs(pmy_pack, pbval_u->u_in, u0, coarse_u0,
                           extrap_order);
    if (pmy_pack->pmesh->pgen->user_bcs) {
      (pmy_pack->pmesh->pgen->user_bcs_func)(pmy_pack->pmesh);
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \brief Prolongate scalar data into fine-grid ghost zones at coarse-fine interfaces.

TaskStatus ScalarField::Prolongate(Driver *driver, int stage) {
  if (pmy_pack->pmesh->multilevel) {
    pbval_u->ProlongateCC(u0, coarse_u0, true);
  }
  return TaskStatus::complete;
}

}  // namespace scalar_field
