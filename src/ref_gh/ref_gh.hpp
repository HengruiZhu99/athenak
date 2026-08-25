//========================================================================================
// Puncture-adapted reference-frame first-order GH plus hyperbolic gauge driver.
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
    int continuation_mode;
    int source_kind;
    bool debug_task_fences;
    bool validate_reference_cache;
    bool max_location_diagnostics;
    int transition_path;
    int phi_ordering;
    Real gamma0;
    Real gamma2;
    bool gauge_driver_enabled;
    bool gauge_reference_subtraction;
    Real gauge_mu;
    Real gauge_eta;
    Real shift_nu;
    Real shift_eta;
    bool exclude_puncture_stencil_diagnostics;
    Real diss;
    Real fail_closed_dt;
    Real reference_mass;
    Real reference_center[3];  // NOLINT(runtime/arrays)
    Real generic_gaussian_width;
    Real generic_q_initial;
    Real generic_q_final;
    Real generic_transition_time;
    Real r_core0;
    Real tau_core;
    Real kappa_core;
    Real transition_width;
    Real tau_transition;
    Real r_fit_min;
    Real r_fit_max;
    Real controller_fit_buffer_cells;
    Real regularization_outer_start;
    Real regularization_outer_end;
    Real controller_zeta;
    Real controller_omega_q;
    Real controller_omega_p;
    Real controller_acceleration_limit;
    Real controller_delta_bound;
    Real controller_rate_bound;
    Real continuation_v_max;
    Real continuation_tau_v;
    Real continuation_xi_end_start;
    Real continuation_risk_slow;
    Real continuation_risk_stop;
    Real continuation_condition_stop;
    Real continuation_lapse_min_stop;
    Real continuation_lapse_max_stop;
    Real continuation_v2_stop;
    Real continuation_gh_warning;
    Real continuation_reduction_warning;
    Real continuation_curl_warning;
    Real continuation_growth_time;
  } opt;

  struct ControllerState {
    Real delta_q;
    Real delta_q_dot;
    Real delta_p;
    Real delta_p_dot;
    Real xi;
    Real xi_dot;
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
    Real xi_ddot;
    Real v_cmd;
    Real risk;
    Real risk_condition;
    Real risk_lapse_min;
    Real risk_lapse_max;
    Real risk_v2;
    Real risk_factor;
    Real endpoint_factor;
    Real gh_l2;
    Real reduction_l2;
    Real curl_l2;
    bool feedback_active;
    bool fitting_shell_valid;
    bool constraint_veto;
    bool controller_frozen;
    bool controller_completed;
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
  Real max_location_diagnostic_time;
  int max_location_diagnostic_cycle;
  std::uint64_t controller_generation;
  std::uint64_t reference_cache_generation;
  std::uint64_t reference_diagnostic_generation;
  ControllerState controller;
  ControllerState controller_base;
  ControllerState controller_rhs;
  ControllerDiagnostics controller_diagnostics;
  bool continuation_constraint_veto;
  bool continuation_frozen;
  bool continuation_completed;
  Real continuation_veto_start_time;
  Real continuation_veto_start_level;
  Real continuation_veto_last_level;
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
  void AppendMaxLocationDiagnostics();
  static void SetADMVariables(MeshBlockPack *pack);
  TaskStatus InitRecv(Driver *driver, int stage);
  TaskStatus ClearRecv(Driver *driver, int stage);
  TaskStatus ClearSend(Driver *driver, int stage);
  TaskStatus CopyU(Driver *driver, int stage);
  TaskStatus MeasureController(Driver *driver, int stage);
  void MeasureControllerAtTime(Real stage_time);
  bool UpdateContinuationConstraintVeto(Real time);
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
