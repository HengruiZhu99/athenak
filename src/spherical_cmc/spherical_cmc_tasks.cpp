// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#include <map>
#include <memory>
#include <string>

#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "spherical_cmc/spherical_cmc.hpp"

namespace spherical_cmc {
void SphericalCMC::AssembleTasks(std::map<std::string, std::shared_ptr<TaskList>> tl) {
  TaskID none(0);
  auto copy = tl["stagen"]->AddTask(&SphericalCMC::CopyState, this, none);
  auto solve = tl["stagen"]->AddTask(&SphericalCMC::SolveGauge, this, copy);
  auto update = tl["stagen"]->AddTask(&SphericalCMC::RKUpdate, this, solve);
  auto project = tl["stagen"]->AddTask(&SphericalCMC::SolveGauge, this, update);
  tl["stagen"]->AddTask(&SphericalCMC::NewTimeStep, this, project);
}
TaskStatus SphericalCMC::CopyState(Driver *, int stage) {
  if (stage == 1) Kokkos::deep_copy(u1, u0);
  return TaskStatus::complete;
}
TaskStatus SphericalCMC::SolveGauge(Driver *, int) {
  Refresh();
  return TaskStatus::complete;
}
TaskStatus SphericalCMC::RKUpdate(Driver *driver, int stage) {
  Kokkos::deep_copy(saved, u1);
  const int is = pmy_pack->pmesh->mb_indcs.is;
  for (int i = 0; i <= radial->n + 1; ++i) {
    radial->a[i] = driver->gam0[stage - 1] * radial->a[i] +
                   driver->gam1[stage - 1] * saved(0, AGR, 0, 0, is + i - 1) +
                   driver->beta[stage - 1] * pmy_pack->pmesh->dt * radial->rhs[i];
  }
  // Store only the evolved metric; all dependent fields are solved by the next task.
  if (radial->inner_radius == 0) radial->a[0] = 1;
  for (int i = 0; i <= radial->n + 1; ++i) host(0, AGR, 0, 0, is + i - 1) = radial->a[i];
  Kokkos::deep_copy(u0, host);
  return TaskStatus::complete;
}
}  // namespace spherical_cmc
