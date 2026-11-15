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
  // Existing NR end-stage queue: all stage sends are retired before a tracker
  // launches its own collective mapped exchange. Single-hole path is a no-op.
  pmy_pack->pnr->QueueTask(&Z4cHyperboloidal::TrackBinary,this,numrel::Z4c_PT,
      "Hyp_TrackBinary",numrel::Task_End,{numrel::Hyp_ClearS,numrel::Hyp_ClearR});
}

TaskStatus Z4cHyperboloidal::InitRecv(Driver *driver, int stage) {
  if (!initialized_ || !driver->UsesClassicalRK4() || stage<1 || stage>4)
    throw std::logic_error("Hyperboloidal receive task needs initialized classical RK4");
  auto &bulk=RequirePlan(mapped_halo_.get());
  auto &surface=RequirePlan(surface_exchange_.get());
  const auto a=bulk.StartReceives(),b=surface.StartReceives();
  return Join(Join(a,b),RequirePlan(agreement_plan_.get()).StartReceives());
}

TaskStatus Z4cHyperboloidal::PrepareStage(Driver *, int) {
  EnforceAlgebraicConstraints();
  const auto status=RequirePlan(surface_exchange_.get()).Project();
  if (status!=TaskStatus::complete) return status;
  FormResidual();
  return TaskStatus::complete;
}

TaskStatus Z4cHyperboloidal::SendBulk(Driver *, int) {
  const auto bulk=RequirePlan(mapped_halo_.get()).PackAndSend(residual,u0);
  return Join(bulk,RequirePlan(agreement_plan_.get()).PackAndSend(residual));
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
  InvalidateDiagnostics();
  return RequirePlan(surface_exchange_.get()).Project();
}

TaskStatus Z4cHyperboloidal::ClearSend(Driver *, int) {
  const auto bulk=RequirePlan(mapped_halo_.get()).ClearSend();
  const auto surface=RequirePlan(surface_exchange_.get()).ClearSend();
  return Join(Join(bulk,surface),RequirePlan(agreement_plan_.get()).ClearSend());
}

TaskStatus Z4cHyperboloidal::ClearRecv(Driver *, int) {
  const auto bulk=RequirePlan(mapped_halo_.get()).ClearRecv();
  const auto surface=RequirePlan(surface_exchange_.get()).ClearRecv();
  return Join(Join(bulk,surface),RequirePlan(agreement_plan_.get()).ClearRecv());
}

TaskStatus Z4cHyperboloidal::ApplyAgreement(Driver *,int) {
  RequirePlan(agreement_plan_.get());
  const auto status=agreement_plan_->Receive();
  if (status!=TaskStatus::complete) return status;
  agreement_plan_->Evaluate();agreement_plan_->AddRHS(*this);
  return TaskStatus::complete;
}

std::array<Real,12> Z4cHyperboloidal::AgreementHistorySums() {
  if (!plans_ready_) BuildStagePlans();
  std::array<Real,12> out{};
  out[8]=npoints;out[9]=mapped_halo_->Rows().size()+mapped_halo_->ExteriorTargets().size();
  auto *plan=agreement_plan_.get();

  RequirePlan(plan);
  FormResidual();plan->StartReceives();plan->PackAndSend(residual);
  while(plan->Receive()!=TaskStatus::complete) {}
  plan->Evaluate();const auto values=plan->History();
  while(plan->ClearSend()!=TaskStatus::complete) {}
  while(plan->ClearRecv()!=TaskStatus::complete) {}
  for(int f=0;f<8;++f)out[f]=values[f];
  out[10]=plan->Exchange().RemoteDonors();
  out[11]=plan->Exchange().SendDonors();
  return out;
}
}  // namespace z4c_hyperboloidal
