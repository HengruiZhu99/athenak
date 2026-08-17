//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file amr_jump_diagnostic.hpp
//! \brief Default-off configuration and accounting primitives for a bounded Z4c AMR
//! jump diagnosis.

#ifndef Z4C_AMR_JUMP_DIAGNOSTIC_HPP_
#define Z4C_AMR_JUMP_DIAGNOSTIC_HPP_

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"

class Driver;
class MeshBlockPack;
class MeshRefinement;

namespace z4c {

inline constexpr const char *kAMRJumpDiagnosticSchema =
    "athenak_z4c_amr_jump_diagnostic_v1";

enum class AMRJumpPhase : int {
  t0_accepted_old_hierarchy = 0,
  t1_balanced_topology_proposal = 1,
  t2_redistributed_refined_active = 2,
  t3_boundary_reconstruction = 3,
  t4_projected_z4c = 4,
  t5_accepted_new_hierarchy = 5,
};

enum class AMRJumpWriter : int {
  accepted_old_state = 0,
  topology_only = 1,
  refine_or_derefine_transfer = 2,
  restrict = 3,
  mpi_receive = 4,
  physical_or_axis_bc = 5,
  same_level_coarse_refresh = 6,
  coarse_to_fine_prolongation = 7,
  algebraic_projection = 8,
  adm_or_constraint_recomputation = 9,
};

enum class AMRJumpHierarchyControl : int {
  dynamic = 0,
  freeze_after_target = 1,
  buffered_freeze_after_target = 2,
};

inline const char *AMRJumpHierarchyControlName(
    const AMRJumpHierarchyControl control) {
  switch (control) {
    case AMRJumpHierarchyControl::dynamic:
      return "dynamic";
    case AMRJumpHierarchyControl::freeze_after_target:
      return "freeze_after_target";
    case AMRJumpHierarchyControl::buffered_freeze_after_target:
      return "buffered_freeze_after_target";
  }
  return "unknown";
}

inline const char *AMRJumpPhaseName(const AMRJumpPhase phase) {
  switch (phase) {
    case AMRJumpPhase::t0_accepted_old_hierarchy:
      return "T0_ACCEPTED_OLD_HIERARCHY";
    case AMRJumpPhase::t1_balanced_topology_proposal:
      return "T1_BALANCED_TOPOLOGY_PROPOSAL";
    case AMRJumpPhase::t2_redistributed_refined_active:
      return "T2_REDISTRIBUTED_REFINED_ACTIVE";
    case AMRJumpPhase::t3_boundary_reconstruction:
      return "T3_BOUNDARY_RECONSTRUCTION";
    case AMRJumpPhase::t4_projected_z4c:
      return "T4_PROJECTED_Z4C";
    case AMRJumpPhase::t5_accepted_new_hierarchy:
      return "T5_ACCEPTED_NEW_HIERARCHY";
  }
  return "UNKNOWN_PHASE";
}

inline const char *AMRJumpWriterName(const AMRJumpWriter writer) {
  switch (writer) {
    case AMRJumpWriter::accepted_old_state:
      return "ACCEPTED_OLD_STATE";
    case AMRJumpWriter::topology_only:
      return "TOPOLOGY_ONLY";
    case AMRJumpWriter::refine_or_derefine_transfer:
      return "REFINE_OR_DEREFINE_TRANSFER";
    case AMRJumpWriter::restrict:
      return "RESTRICT";
    case AMRJumpWriter::mpi_receive:
      return "MPI_RECEIVE";
    case AMRJumpWriter::physical_or_axis_bc:
      return "PHYSICAL_OR_AXIS_BC";
    case AMRJumpWriter::same_level_coarse_refresh:
      return "SAME_LEVEL_COARSE_REFRESH";
    case AMRJumpWriter::coarse_to_fine_prolongation:
      return "COARSE_TO_FINE_PROLONGATION";
    case AMRJumpWriter::algebraic_projection:
      return "ALGEBRAIC_PROJECTION";
    case AMRJumpWriter::adm_or_constraint_recomputation:
      return "ADM_OR_CONSTRAINT_RECOMPUTATION";
  }
  return "UNKNOWN_WRITER";
}

struct AMRJumpDiagnosticConfig {
  bool enabled = false;
  int target_level_before = 2;
  int target_level_after = 3;
  int target_cycle = -1;
  int post_cycles = 8;
  std::string output_basename = "z4c_amr_jump";
  // Optional transfer policy applied only to the exact matched T1--T5
  // transaction.  The preceding evolution, including the RK step that reaches
  // the target cycle, retains the production <z4c>/amr_transfer policy.
  std::string target_transfer;
  // At the final pre-projection T3 boundary state and at T5, recompute
  // constraints with the O2, O4, and O6 derivative providers. This never
  // changes production state.
  bool derivative_order_audit = false;
  AMRJumpHierarchyControl hierarchy_control =
      AMRJumpHierarchyControl::dynamic;
};

struct AMRJumpDiagnosticContext {
  bool cartoon = false;
  bool adaptive = false;
  bool multilevel = false;
  int root_level = 0;
  int maximum_level = 0;
  int nranks = 1;
};

inline bool AMRJumpOutputBasenameIsSafe(const std::string &basename) {
  if (basename.empty() || basename == "." || basename == ".." ||
      basename.front() == '.') {
    return false;
  }
  return std::all_of(basename.begin(), basename.end(), [](const unsigned char c) {
    return std::isalnum(c) != 0 || c == '_' || c == '-' || c == '.';
  });
}

inline std::string ValidateAMRJumpDiagnosticConfig(
    const AMRJumpDiagnosticConfig &config,
    const AMRJumpDiagnosticContext &context) {
  if (!config.target_transfer.empty() && !config.enabled) {
    return "amr_jump_target_transfer requires amr_jump_diagnostic=true";
  }
  if (config.derivative_order_audit && !config.enabled) {
    return "amr_jump_derivative_order_audit requires amr_jump_diagnostic=true";
  }
  if (!config.enabled) {
    if (config.hierarchy_control != AMRJumpHierarchyControl::dynamic) {
      return "amr_jump_hierarchy_control requires amr_jump_diagnostic=true";
    }
    return {};
  }
  if (config.target_level_before < context.root_level) {
    return "amr_jump_target_level_before is below the root level";
  }
  if (config.target_level_after != config.target_level_before + 1) {
    return "amr_jump target levels must describe one 2:1 level transition";
  }
  if (config.target_level_after > context.maximum_level) {
    return "amr_jump_target_level_after exceeds the configured maximum level";
  }
  if (config.target_cycle < -1) {
    return "amr_jump_target_cycle must be nonnegative or -1 for the first match";
  }
  if (!config.target_transfer.empty()) {
    if (config.target_transfer != "high_order" &&
        config.target_transfer != "limited_o2") {
      return "amr_jump_target_transfer must be high_order or limited_o2";
    }
    if (config.target_cycle < 0) {
      return "amr_jump_target_transfer requires an explicit target cycle";
    }
  }
  if (config.post_cycles < 0) {
    return "amr_jump_post_cycles must be nonnegative";
  }
  if (!AMRJumpOutputBasenameIsSafe(config.output_basename)) {
    return "amr_jump_output_basename must be a non-hidden portable basename";
  }
  if (config.hierarchy_control != AMRJumpHierarchyControl::dynamic &&
      config.target_cycle < 0) {
    return "amr_jump_hierarchy_control requires an explicit target cycle";
  }
  if (context.nranks <= 0) {
    return "AMR jump diagnostic requires a positive MPI rank count";
  }
  if (config.enabled && !context.cartoon) {
    return "amr_jump_diagnostic requires Cartoon SO(2) symmetry";
  }
  if (config.enabled && (!context.adaptive || !context.multilevel)) {
    return "amr_jump_diagnostic requires adaptive multilevel evolution";
  }
  return {};
}

inline bool IsKnownAMRJumpParameter(const std::string &name) {
  constexpr std::array<const char *, 9> known = {
      "amr_jump_diagnostic", "amr_jump_target_level_before",
      "amr_jump_target_level_after", "amr_jump_target_cycle",
      "amr_jump_post_cycles",
      "amr_jump_output_basename", "amr_jump_hierarchy_control",
      "amr_jump_target_transfer", "amr_jump_derivative_order_audit"};
  return std::any_of(known.begin(), known.end(), [&name](const char *candidate) {
    return name == candidate;
  });
}

inline AMRJumpDiagnosticConfig ReadAMRJumpDiagnosticConfig(
    ParameterInput *pin, const AMRJumpDiagnosticContext &context) {
  if (pin == nullptr) {
    throw std::invalid_argument("AMR jump diagnostic requires ParameterInput");
  }
  for (const auto &block : pin->block) {
    if (block.block_name != "z4c") continue;
    for (const auto &line : block.line) {
      if (line.param_name.rfind("amr_jump_", 0) == 0 &&
          !IsKnownAMRJumpParameter(line.param_name)) {
        throw std::invalid_argument("unknown <z4c>/" + line.param_name);
      }
    }
  }

  AMRJumpDiagnosticConfig config;
  config.enabled =
      pin->GetOrAddBoolean("z4c", "amr_jump_diagnostic", false);
  if (pin->DoesParameterExist("z4c", "amr_jump_target_level_before")) {
    config.target_level_before =
        pin->GetInteger("z4c", "amr_jump_target_level_before");
  }
  if (pin->DoesParameterExist("z4c", "amr_jump_target_level_after")) {
    config.target_level_after =
        pin->GetInteger("z4c", "amr_jump_target_level_after");
  }
  if (pin->DoesParameterExist("z4c", "amr_jump_target_cycle")) {
    config.target_cycle = pin->GetInteger("z4c", "amr_jump_target_cycle");
  }
  if (pin->DoesParameterExist("z4c", "amr_jump_post_cycles")) {
    config.post_cycles = pin->GetInteger("z4c", "amr_jump_post_cycles");
  }
  if (pin->DoesParameterExist("z4c", "amr_jump_output_basename")) {
    config.output_basename =
        pin->GetString("z4c", "amr_jump_output_basename");
  }
  if (pin->DoesParameterExist("z4c", "amr_jump_target_transfer")) {
    const std::string target_transfer =
        pin->GetString("z4c", "amr_jump_target_transfer");
    if (target_transfer != "none") config.target_transfer = target_transfer;
  }
  if (pin->DoesParameterExist("z4c", "amr_jump_derivative_order_audit")) {
    config.derivative_order_audit =
        pin->GetBoolean("z4c", "amr_jump_derivative_order_audit");
  }
  if (pin->DoesParameterExist("z4c", "amr_jump_hierarchy_control")) {
    const std::string control =
        pin->GetString("z4c", "amr_jump_hierarchy_control");
    if (control == "dynamic") {
      config.hierarchy_control = AMRJumpHierarchyControl::dynamic;
    } else if (control == "freeze_after_target") {
      config.hierarchy_control = AMRJumpHierarchyControl::freeze_after_target;
    } else if (control == "buffered_freeze_after_target") {
      config.hierarchy_control =
          AMRJumpHierarchyControl::buffered_freeze_after_target;
    } else {
      throw std::invalid_argument(
          "unknown <z4c>/amr_jump_hierarchy_control=" + control);
    }
  }
  const std::string error = ValidateAMRJumpDiagnosticConfig(config, context);
  if (!error.empty()) throw std::invalid_argument(error);
  return config;
}

struct AMRJumpLedgerIncrement {
  AMRJumpPhase phase = AMRJumpPhase::t0_accepted_old_hierarchy;
  AMRJumpWriter writer = AMRJumpWriter::accepted_old_state;
  int ordinal = 0;
  int owner_rank = -1;
  std::uint64_t point_id = 0;
  Real before = 0.0;
  Real after = 0.0;
};

struct AMRJumpLedgerClosure {
  bool finite = true;
  bool ordered = true;
  bool unique_ownership = true;
  Real telescoped_change = 0.0;
  Real direct_change = 0.0;
  Real absolute_residual = 0.0;
  Real tolerance = 0.0;

  bool closed() const {
    return finite && ordered && unique_ownership &&
           absolute_residual <= tolerance;
  }
};

//! Runtime capture for the bounded, default-off AMR transaction diagnostic.
//!
//! The implementation deliberately lives outside the integrator task graph.  A Z4c
//! object owns this class only when <z4c>/amr_jump_diagnostic=true; the default path
//! therefore allocates no diagnostic state and performs no diagnostic fence, reduction,
//! or file operation.
class AMRJumpDiagnosticRuntime {
 public:
  AMRJumpDiagnosticRuntime(MeshBlockPack *pack,
                           const AMRJumpDiagnosticConfig &config);
  ~AMRJumpDiagnosticRuntime() = default;

  AMRJumpDiagnosticRuntime(const AMRJumpDiagnosticRuntime &) = delete;
  AMRJumpDiagnosticRuntime &operator=(const AMRJumpDiagnosticRuntime &) = delete;

  void BeginTransaction(const MeshRefinement &refinement);
  bool ShouldFreezeHierarchy() const;
  bool ShouldBufferTargetCycle(int cycle) const;
  int target_level_before() const { return config_.target_level_before; }
  void RecordHierarchyControl(int original_refine, int original_derefine,
                              int buffered_refine, int suppressed_refine,
                              int suppressed_derefine);
  void CancelTransaction();
  void RecordTopologyProposal(const MeshRefinement &refinement,
                              int old_nmb, int new_nmb, int nnew, int ndel);
  void RecordT2();
  void RecordT3(AMRJumpWriter writer, int ordinal, bool final_boundary_state);
  void RecordRestrictionShadow();
  void RecordSameLevelRefreshShadow();
  void RecordT4();
  void RecordT5();
  void AfterAcceptedCycle(Driver *driver);
  void RecordRKStageCoarseFineExposure(int stage);

  bool transaction_active() const { return transaction_active_; }
  bool detailed_event_active() const { return detailed_event_active_; }
  bool target_seen() const { return target_seen_; }
  int target_cycle() const { return target_cycle_; }

 private:
  MeshBlockPack *pack_ = nullptr;
  AMRJumpDiagnosticConfig config_;
  bool transaction_active_ = false;
  bool output_initialized_ = false;
  bool pending_t0_ = false;
  bool detailed_event_active_ = false;
  bool target_seen_ = false;
  bool target_transfer_active_ = false;
  int saved_amr_transfer_ = -1;
  int target_cycle_ = -1;
  int old_max_level_ = -1;
  int new_max_level_ = -1;
  int old_nmb_total_ = 0;
  int t3_last_ordinal_ = -1;
  std::uint64_t local_x_cf_ = 0;
  std::string rank_root_;
  std::string pending_event_root_;
  std::string event_root_;
  std::vector<int> old_ranks_;
  std::vector<int> old_flags_;
  std::vector<std::int32_t> old_lx1_;
  std::vector<std::int32_t> old_lx2_;
  std::vector<std::int32_t> old_lx3_;
  std::vector<std::int32_t> old_levels_;

  void EnsureOutputInitialized();
  void CapturePhase(AMRJumpPhase phase, AMRJumpWriter writer, int ordinal,
                    bool constraints_valid, bool include_coarse);
  void WriteCurrentTopology(const std::string &path) const;
  void WriteAcceptedTopologySnapshot() const;
  void WriteCompactTransaction(int nnew, int ndel) const;
  void WriteAcceptedCycleAggregate() const;
  void RestoreTargetTransfer();
  void DiscardPendingT0();
};

inline AMRJumpLedgerClosure CloseAMRJumpLedger(
    const Real initial, const Real final,
    const std::vector<AMRJumpLedgerIncrement> &increments,
    const Real relative_tolerance =
        256.0 * std::numeric_limits<Real>::epsilon()) {
  AMRJumpLedgerClosure closure;
  closure.direct_change = final - initial;
  int previous_phase = static_cast<int>(AMRJumpPhase::t0_accepted_old_hierarchy);
  int previous_ordinal = -1;
  std::vector<std::pair<std::uint64_t, int>> ownership;
  ownership.reserve(increments.size());
  for (const auto &increment : increments) {
    const int phase = static_cast<int>(increment.phase);
    closure.ordered = closure.ordered &&
                      (phase > previous_phase ||
                       (phase == previous_phase &&
                        increment.ordinal > previous_ordinal));
    if (phase > previous_phase) previous_ordinal = -1;
    previous_phase = phase;
    previous_ordinal = increment.ordinal;
    closure.finite = closure.finite && std::isfinite(increment.before) &&
                     std::isfinite(increment.after);
    closure.telescoped_change += increment.after - increment.before;
    ownership.emplace_back(increment.point_id, increment.owner_rank);
  }
  std::sort(ownership.begin(), ownership.end());
  for (std::size_t i = 1; i < ownership.size(); ++i) {
    if (ownership[i].first == ownership[i - 1].first &&
        ownership[i].second != ownership[i - 1].second) {
      closure.unique_ownership = false;
    }
  }
  closure.absolute_residual =
      std::fabs(closure.telescoped_change - closure.direct_change);
  const Real scale = std::max(
      {Real(1.0), std::fabs(initial), std::fabs(final),
       std::fabs(closure.telescoped_change)});
  closure.tolerance = relative_tolerance * scale;
  closure.finite = closure.finite && std::isfinite(initial) &&
                   std::isfinite(final) &&
                   std::isfinite(closure.telescoped_change) &&
                   std::isfinite(closure.absolute_residual);
  return closure;
}

}  // namespace z4c

#endif  // Z4C_AMR_JUMP_DIAGNOSTIC_HPP_
