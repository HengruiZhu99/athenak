//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_hyperboloidal_tasks.cpp
//! \brief Native block boundary tasks for hyperboloidal Z4c.

#include <stdexcept>

#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "z4c_hyperboloidal/mapped_halo.hpp"
#include "z4c_hyperboloidal/surface_exchange.hpp"
#include "z4c_hyperboloidal/tasks.hpp"
#include "z4c_hyperboloidal/z4c_hyperboloidal.hpp"

namespace z4c_hyperboloidal {
namespace {
template<class Plan>
Plan &RequirePlan(Plan *plan) {
  if (plan==nullptr) throw std::logic_error("Hyperboloidal stage plans are not built");
  return *plan;
}

TaskStatus Join(TaskStatus a, TaskStatus b) {
  if (a==TaskStatus::fail || b==TaskStatus::fail) return TaskStatus::fail;
  return a==TaskStatus::complete && b==TaskStatus::complete
      ? TaskStatus::complete : TaskStatus::incomplete;
}
}  // namespace

void Z4cHyperboloidal::QueueZ4cHyperboloidalTasks() {
  // MeshBlockPack assembles these native queues before the problem generator
  // initializes fields. Only bind tasks here; plans are required at execution.
  QueueZ4cHyperboloidalStageTasks(pmy_pack->pnr,this);
}

TaskStatus Z4cHyperboloidal::InitRecv(Driver *driver, int stage) {
  if (!initialized_ || !driver->UsesClassicalRK4() || stage<1 || stage>4)
    throw std::logic_error("Hyperboloidal receive task needs initialized classical RK4");
  auto &bulk=RequirePlan(mapped_halo_.get());
  auto &surface=RequirePlan(surface_exchange_.get());
  const auto a=bulk.StartReceives(),b=surface.StartReceives();
  return Join(a,b);
}

TaskStatus Z4cHyperboloidal::PrepareStage(Driver *, int) {
  EnforceAlgebraicConstraints();
  const auto status=RequirePlan(surface_exchange_.get()).Project();
  if (status!=TaskStatus::complete) return status;
  FormResidual();
  return TaskStatus::complete;
}

TaskStatus Z4cHyperboloidal::SendBulk(Driver *, int) {
  return RequirePlan(mapped_halo_.get()).PackAndSend(residual,u0);
}

TaskStatus Z4cHyperboloidal::FillLocalHalos(Driver *, int) {
  return RequirePlan(mapped_halo_.get()).FillLocal(residual,u0);
}

TaskStatus Z4cHyperboloidal::RecvBulk(Driver *, int) {
  return RequirePlan(mapped_halo_.get()).Receive();
}

TaskStatus Z4cHyperboloidal::FillRemoteHalos(Driver *, int) {
  return RequirePlan(mapped_halo_.get()).FillRemote(residual,u0);
}

TaskStatus Z4cHyperboloidal::SendSurface(Driver *, int) {
  return RequirePlan(surface_exchange_.get()).PackAndSend();
}

TaskStatus Z4cHyperboloidal::RecvSurface(Driver *, int) {
  return RequirePlan(surface_exchange_.get()).Receive();
}

TaskStatus Z4cHyperboloidal::SolveNormalCompatibility(Driver *, int) {
  auto &surface=RequirePlan(surface_exchange_.get());
  const auto status=surface.Sample();
  if (status!=TaskStatus::complete) return status;
  return surface.Solve();
}

TaskStatus Z4cHyperboloidal::CalcSurfaceRHS(Driver *, int) {
  return RequirePlan(surface_exchange_.get()).Evaluate();
}

TaskStatus Z4cHyperboloidal::ProjectSurface(Driver *, int) {
  return RequirePlan(surface_exchange_.get()).Project();
}

TaskStatus Z4cHyperboloidal::ClearSend(Driver *, int) {
  const auto bulk=RequirePlan(mapped_halo_.get()).ClearSend();
  const auto surface=RequirePlan(surface_exchange_.get()).ClearSend();
  return Join(bulk,surface);
}

TaskStatus Z4cHyperboloidal::ClearRecv(Driver *, int) {
  const auto bulk=RequirePlan(mapped_halo_.get()).ClearRecv();
  const auto surface=RequirePlan(surface_exchange_.get()).ClearRecv();
  return Join(bulk,surface);
}

}  // namespace z4c_hyperboloidal
