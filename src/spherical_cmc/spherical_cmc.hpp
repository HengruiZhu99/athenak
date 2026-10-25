// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#ifndef SPHERICAL_CMC_SPHERICAL_CMC_HPP_
#define SPHERICAL_CMC_SPHERICAL_CMC_HPP_

#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "spherical_cmc/radial_solver.hpp"
#include "tasklist/task_list.hpp"

class MeshBlockPack;
class ParameterInput;
class Driver;
struct HistoryData;

namespace spherical_cmc {
enum Field {
  AGR,
  CHI,
  ALPHA,
  BETA,
  KHAT,
  THETA,
  ARR,
  GAMMA,
  HAM,
  CMC,
  ARHS,
  MOM,
  MASS_ERROR,
  NVAR
};

class SphericalCMC {
 public:
  SphericalCMC(MeshBlockPack *pack, ParameterInput *pin);
  void Initialize(ParameterInput *pin, bool restart);
  void AssembleTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  TaskStatus CopyState(Driver *driver, int stage);
  TaskStatus SolveGauge(Driver *driver, int stage);
  TaskStatus RKUpdate(Driver *driver, int stage);
  TaskStatus NewTimeStep(Driver *driver, int stage);
  void Refresh();
  void History(HistoryData *data);

  DvceArray5D<Real> u0, u1;
  Real dtnew = 0;
  static const char *names[NVAR];

 private:
  MeshBlockPack *pmy_pack;
  std::unique_ptr<RadialSolver> radial;
  HostArray5D<Real> host, saved;
  void LoadMetric();
  void StoreFields();
};
}  // namespace spherical_cmc
#endif  // SPHERICAL_CMC_SPHERICAL_CMC_HPP_
