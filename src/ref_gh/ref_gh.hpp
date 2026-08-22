//========================================================================================
// Separate 50-field puncture-adapted reference-frame first-order GH module.
// Licensed under the 3-clause BSD License, see LICENSE for details.
//========================================================================================
#ifndef REF_GH_REF_GH_HPP_
#define REF_GH_REF_GH_HPP_

#include <cstdint>

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
  // First six slots are the native GH/reduction constraints.  The remaining
  // pointwise diagnostic slots are deliberately not evolved state and are
  // refreshed only when diagnostics/history are requested.
  static constexpr int kNativeConstraints = 6;
  static constexpr int kDiagnosticOffset = kNativeConstraints;
  static constexpr int kSourceDiagnostics = 9;
  static constexpr int kMetricConditionDiagnostic =
      kDiagnosticOffset + kSourceDiagnostics;
  static constexpr int ncon = kMetricConditionDiagnostic + 1;
  static char const * const StateNames[nref_gh];
  static char const * const ConstraintNames[ncon];

  struct Options {
    int fd_order;
    int extrap_order;
    int reference_kind;
    bool reference_time_dependent;
    bool reference_controlled;
    bool controller_enabled;
    int source_kind;
    bool debug_task_fences;
    bool validate_reference_cache;
    Real gamma0;
    Real diss;
    Real fail_closed_dt;
    Real reference_mass;
    Real reference_center[3];  // NOLINT(runtime/arrays)
    Real r_core0;
    Real tau_core;
    Real kappa_core;
    Real tau_transition;
    Real r_fit_min;
    Real r_fit_max;
    Real regularization_outer_start;
    Real regularization_outer_end;
    Real controller_zeta;
    Real controller_omega_q;
    Real controller_omega_p;
    Real controller_acceleration_limit;
    Real controller_delta_bound;
    Real controller_rate_bound;
  } opt;

  struct ControllerState {
    Real delta_q;
    Real delta_q_dot;
    Real delta_p;
    Real delta_p_dot;
  };

  struct ControllerDiagnostics {
    Real e_G;
    Real e_alpha;
    Real fitting_cell_count;
    Real lambda_min;
    Real lambda_max;
    Real det_g_third_min;
    Real det_g_third_max;
    Real condition_max;
    Real relative_lapse_min;
    Real relative_lapse_max;
    Real v2_max;
    Real psi_max;
    Real inverse_psi_max;
    Real physical_lapse_min;
    Real physical_lapse_max;
    Real r_core;
    Real transition_amplitude;
    bool feedback_active;
    bool fitting_shell_valid;
  };

  RefGh(MeshBlockPack *ppack, ParameterInput *pin);
  ~RefGh();

  DvceArray5D<Real> u0;
  DvceArray5D<Real> u1;
  DvceArray5D<Real> u_rhs;
  DvceArray5D<Real> u_con;
  DvceArray5D<Real> coarse_u0;
  DvceArray5D<Real> reference_provider;
  DvceArray5D<Real> reference_workspace;
  DvceArray5D<Real> reference_evolution;
  DvceArray5D<Real> reference_diagnostic;
  DvceArray2D<Real> reference_table;
  Real reference_cache_time;
  Real reference_diagnostic_time;
  std::uint64_t controller_generation;
  std::uint64_t reference_cache_generation;
  std::uint64_t reference_diagnostic_generation;
  ControllerState controller;
  ControllerState controller_base;
  ControllerState controller_rhs;
  ControllerDiagnostics controller_diagnostics;
  bool reference_cache_oracle_validated;
  bool reference_diagnostic_oracle_validated;
  Real dtnew;
  Real max_char_speed;
  MeshBoundaryValuesCC *pbval_u;

  template <int FDNG>
  TaskStatus CalcRHS(Driver *driver, int stage);
  template <int FDNG>
  void CalcConstraints();
  void QueueTasks();
  void RefGhToADM();
  void CacheMetricCondition();
  void UpdateDiagnostics();
  static void SetADMVariables(MeshBlockPack *pack);
  TaskStatus InitRecv(Driver *driver, int stage);
  TaskStatus ClearRecv(Driver *driver, int stage);
  TaskStatus ClearSend(Driver *driver, int stage);
  TaskStatus CopyU(Driver *driver, int stage);
  TaskStatus MeasureController(Driver *driver, int stage);
  void MeasureControllerAtTime(Real stage_time);
  TaskStatus UpdateReferenceGeometry(Driver *driver, int stage);
  TaskStatus ExpRKUpdate(Driver *driver, int stage);
  TaskStatus RestrictU(Driver *driver, int stage);
  TaskStatus SendU(Driver *driver, int stage);
  TaskStatus RecvU(Driver *driver, int stage);
  TaskStatus Prolongate(Driver *driver, int stage);
  TaskStatus ApplyPhysicalBCs(Driver *driver, int stage);
  TaskStatus NewTimeStep(Driver *driver, int stage);
  void DebugFence(const char *label) const;

 private:
  Real StageTime(const Driver *driver, int stage) const;
  void FillReferenceCache(Real time, bool include_diagnostics);
  void PersistControllerState();
  MeshBlockPack *pmy_pack;
  ParameterInput *pinput;
};

}  // namespace ref_gh

#endif  // REF_GH_REF_GH_HPP_
