//========================================================================================
// Separate 50-field puncture-adapted reference-frame first-order GH module.
// Licensed under the 3-clause BSD License, see LICENSE for details.
//========================================================================================
#ifndef REF_GH_REF_GH_HPP_
#define REF_GH_REF_GH_HPP_

#include "athena.hpp"
#include "parameter_input.hpp"
#include "ref_gh/ref_gh_state.hpp"
#include "tasklist/task_list.hpp"

class Driver;
class MeshBlockPack;
class MeshBoundaryValuesCC;

namespace ref_gh {

class RefGh {
 public:
  static constexpr int nref_gh = nvar;
  static constexpr int ncon = 6;
  static char const * const StateNames[nref_gh];
  static char const * const ConstraintNames[ncon];

  struct Options {
    int fd_order;
    int extrap_order;
    int reference_kind;
    Real gamma0;
    Real diss;
    Real fail_closed_dt;
    Real reference_mass;
    Real reference_center[3];  // NOLINT(runtime/arrays)
  } opt;

  RefGh(MeshBlockPack *ppack, ParameterInput *pin);
  ~RefGh();

  DvceArray5D<Real> u0;
  DvceArray5D<Real> u1;
  DvceArray5D<Real> u_rhs;
  DvceArray5D<Real> u_con;
  DvceArray5D<Real> coarse_u0;
  DvceArray2D<Real> reference_table;
  Real dtnew;
  Real max_char_speed;
  MeshBoundaryValuesCC *pbval_u;

  template <int FDNG>
  TaskStatus CalcRHS(Driver *driver, int stage);
  template <int FDNG>
  void CalcConstraints();
  void QueueTasks();
  void RefGhToADM();
  void UpdateDiagnostics();
  static void SetADMVariables(MeshBlockPack *pack);
  TaskStatus InitRecv(Driver *driver, int stage);
  TaskStatus ClearRecv(Driver *driver, int stage);
  TaskStatus ClearSend(Driver *driver, int stage);
  TaskStatus CopyU(Driver *driver, int stage);
  TaskStatus ExpRKUpdate(Driver *driver, int stage);
  TaskStatus RestrictU(Driver *driver, int stage);
  TaskStatus SendU(Driver *driver, int stage);
  TaskStatus RecvU(Driver *driver, int stage);
  TaskStatus Prolongate(Driver *driver, int stage);
  TaskStatus ApplyPhysicalBCs(Driver *driver, int stage);
  TaskStatus NewTimeStep(Driver *driver, int stage);

 private:
  MeshBlockPack *pmy_pack;
};

}  // namespace ref_gh

#endif  // REF_GH_REF_GH_HPP_
