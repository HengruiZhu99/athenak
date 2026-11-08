//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_hyperboloidal_lifecycle.cpp
//! \brief Native initialization and fixed-step policy for hyperboloidal Z4c.

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "z4c_hyperboloidal/mapped_halo.hpp"
#include "z4c_hyperboloidal/surface_exchange.hpp"
#include "z4c_hyperboloidal/z4c_hyperboloidal.hpp"

namespace z4c_hyperboloidal {
namespace {
TaskStatus BothComplete(TaskStatus a,TaskStatus b) {
  if (a==TaskStatus::fail || b==TaskStatus::fail)
    throw std::runtime_error("Hyperboloidal initialization exchange failed");
  return a==TaskStatus::complete && b==TaskStatus::complete
      ? TaskStatus::complete : TaskStatus::incomplete;
}
}  // namespace

void Z4cHyperboloidal::InitializeBoundaryValues(Driver *driver) {
  if (!initialized_)
    throw std::logic_error("Hyperboloidal initial state or restart is not ready");
  if (!plans_ready_) BuildStagePlans();

  // The problem generator or restart reader owns the evolved fields and chi
  // records. Regenerate only dependent values and communication workspace here.
  EnforceAlgebraicConstraints();
  auto &bulk=*mapped_halo_;
  auto &surface=*surface_exchange_;
  surface.Project();
  FormResidual();
  bulk.StartReceives();surface.StartReceives();
  bulk.PackAndSend(residual,u0);surface.PackAndSend();
  bulk.FillLocal(residual,u0);

  // Reuse native incomplete-task polling instead of treating an unfinished
  // receive as a completed initialization. Both plans must retire their buffers
  // before the first RK stage (including ranks with no owned scri patch).
  TaskList initialize;
  TaskID none(0);
  auto received_bulk=initialize.AddTask([&](Driver *,int) {
    return bulk.Receive();
  },none);
  auto filled_bulk=initialize.AddTask([&](Driver *,int) {
    return bulk.FillRemote(residual,u0);
  },received_bulk);
  auto received_surface=initialize.AddTask([&](Driver *,int) {
    return surface.Receive();
  },none);
  auto solved_surface=initialize.AddTask([&](Driver *,int) {
    const auto status=surface.Sample();
    return status==TaskStatus::complete ? surface.Solve() : status;
  },received_surface);
  initialize.AddTask([&](Driver *,int) {
    const auto bulk_status=bulk.ClearSend(),surface_status=surface.ClearSend();
    return BothComplete(bulk_status,surface_status);
  },none);
  auto used_receives=filled_bulk | solved_surface;
  initialize.AddTask([&](Driver *,int) {
    const auto bulk_status=bulk.ClearRecv(),surface_status=surface.ClearRecv();
    return BothComplete(bulk_status,surface_status);
  },used_receives);
  while (initialize.DoAvailable(driver,0)!=TaskListStatus::complete) {}
}

TaskStatus Z4cHyperboloidal::NewTimeStep(Driver *driver,int) {
  if (!driver->UsesClassicalRK4())
    throw std::invalid_argument("Hyperboloidal fixed steps require rk4_classical");
  if (!std::isfinite(requested_dt) || requested_dt<=0 ||
      requested_dt/std::min(parameters.tau,Real(InnerTelegrapher::tau))>.25)
    throw std::invalid_argument("Hyperboloidal fixed timestep exceeds the relaxation limit");
  // This is an absolute time step, not a characteristic inverse speed to be
  // multiplied by cfl_number. The pinned device policy reports characteristic
  // CFL and retains the relaxation check; it has no additional .4 CFL rejection.
  dtnew=requested_dt;
  return TaskStatus::complete;
}

Real Z4cHyperboloidal::StageMaxCompatibility() const {
  if (!plans_ready_) throw std::logic_error("Hyperboloidal stage diagnostics are not ready");
  return surface_exchange_->MaxCompatibility();
}

Real Z4cHyperboloidal::StageMaxClock() const {
  if (!plans_ready_) throw std::logic_error("Hyperboloidal stage diagnostics are not ready");
  return surface_exchange_->MaxClock();
}
}  // namespace z4c_hyperboloidal
