#ifndef Z4C_HYPERBOLOIDAL_TASKS_HPP_
#define Z4C_HYPERBOLOIDAL_TASKS_HPP_
//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file tasks.hpp
//! \brief Stage dependencies queued into AthenaK's NumericalRelativity task list.

#include "tasklist/numerical_relativity.hpp"
#include "z4c_hyperboloidal/rhs_subset.hpp"

namespace z4c_hyperboloidal {

// Queue is normally NumericalRelativity. The small template also lets the
// dependency contract run against the original TaskList with delayed transport
// in tests. It is not a scheduler: Driver still executes before/stagen/after.
// Module adapters must perform real work or return incomplete while polling;
// a partially integrated module must not instantiate this graph with no-op RHSs.
template<class Queue, class Module>
void QueueZ4cHyperboloidalStageTasks(Queue *queue, Module *module) {
  using namespace numrel;  // NOLINT(build/namespaces)

  // Posting receives does not require current-stage field values.
  queue->QueueTask(&Module::InitRecv, module, Hyp_Recv, "Hyp_Recv", Task_Start);

  // Copy the accepted base before surface projection changes dependent values.
  // Send both sparse plans before launching independent volume work. Packing
  // copies donor values, so subsequent halo writes cannot race MPI send buffers.
  queue->QueueTask(&Module::CopyU, module, Hyp_CopyU, "Hyp_CopyU", Task_Run);
  queue->QueueTask(&Module::PrepareStage, module, Hyp_Prepare, "Hyp_Prepare",
                   Task_Run, {Hyp_CopyU});
  queue->QueueTask(&Module::SendBulk, module, Hyp_SendBulk, "Hyp_SendBulk",
                   Task_Run, {Hyp_Prepare});
  queue->QueueTask(&Module::SendSurface, module, Hyp_SendSurface, "Hyp_SendSurface",
                   Task_Run, {Hyp_Prepare});
  queue->QueueTask(&Module::FillLocalHalos, module, Hyp_FillLocal, "Hyp_FillLocal",
                   Task_Run, {Hyp_SendBulk});
  queue->QueueTask([=](Driver *driver, int stage) {
    return module->CalcRHS(driver, stage, RHSSubset::Interior);
  }, Hyp_InteriorRHS, "Hyp_InteriorRHS", Task_Run, {Hyp_Prepare});

  // An incomplete receive yields to every other ready task on the rank.
  // Surface sampling owns a separate direct-donor plan; it need not wait for
  // interpolated volume ghosts. Interior volume work remains independent.
  queue->QueueTask(&Module::RecvBulk, module, Hyp_RecvBulk, "Hyp_RecvBulk",
                   Task_Run, {Hyp_SendBulk});
  queue->QueueTask(&Module::FillRemoteHalos, module, Hyp_FillRemote, "Hyp_FillRemote",
                   Task_Run, {Hyp_RecvBulk, Hyp_FillLocal});
  queue->QueueTask(&Module::RecvSurface, module, Hyp_RecvSurface, "Hyp_RecvSurface",
                   Task_Run, {Hyp_SendSurface});
  queue->QueueTask(&Module::SolveNormalCompatibility, module, Hyp_Normal, "Hyp_Normal",
                   Task_Run, {Hyp_RecvSurface});
  queue->QueueTask(&Module::CalcSurfaceRHS, module, Hyp_SurfaceRHS, "Hyp_SurfaceRHS",
                   Task_Run, {Hyp_Normal});
  queue->QueueTask([=](Driver *driver, int stage) {
    return module->CalcRHS(driver, stage, RHSSubset::Dependent);
  }, Hyp_OuterRHS, "Hyp_DependentRHS", Task_Run, {Hyp_FillRemote, Hyp_Normal});

  // All disjoint volume batches and the separately evolved surface must have
  // current-stage rates before advancing either state. Reproject after every
  // update, including stage four, so accepted state and diagnostics agree.
  queue->QueueTask(&Module::ExpRKUpdate, module, Hyp_ExplRK, "Hyp_ExplRK", Task_Run,
                   {Hyp_InteriorRHS, Hyp_OuterRHS, Hyp_SurfaceRHS});
  queue->QueueTask(&Module::ProjectSurface, module, Hyp_Project, "Hyp_Project",
                   Task_Run, {Hyp_ExplRK});

  // End tasks retire persistent requests before the next stage reuses buffers.
  // They do not impose a rank-wide barrier or a stage collective.
  queue->QueueTask(&Module::ClearSend, module, Hyp_ClearS, "Hyp_ClearS", Task_End);
  queue->QueueTask(&Module::ClearRecv, module, Hyp_ClearR, "Hyp_ClearR", Task_End);
}

}  // namespace z4c_hyperboloidal
#endif  // Z4C_HYPERBOLOIDAL_TASKS_HPP_
