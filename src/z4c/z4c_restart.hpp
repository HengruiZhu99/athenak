//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c_restart.hpp
//! \brief Immutable host-only Z4c restart configuration and integration-state carrier.

#ifndef Z4C_Z4C_RESTART_HPP_
#define Z4C_Z4C_RESTART_HPP_

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "z4c/z4c_symmetry.hpp"

class ParameterInput;

namespace z4c {

inline constexpr const char *kZ4cRestartBlock = "z4c_restart";

struct Z4cCentralRestartState {
  static constexpr int kCurrentSchema = 2;

  int schema = kCurrentSchema;
  bool initialized = false;
  double proper_time = 0.0;
  double previous_lapse = 1.0;
  double constraint_norm = 0.0;
  double abs_kretschmann = 0.0;
  int sample_gid = -1;
  int sample_level = -1;
  int last_cycle = -1;
  double last_time = 0.0;
};

//! Restart-authoritative state for the Cartoon m=0 FastFlow implementation.
struct Z4cM0FastFlowRestartState {
  static constexpr int kCurrentSchema = 2;

  int schema = kCurrentSchema;
  std::vector<double> coefficients;
  std::string surface_mode = "none";
  std::string selected_branch = "none";
  int center_count = 0;
  double center_z0 = 0.0;
  double center_z1 = 0.0;
  std::string status = "not_started";
  std::string failure_code = "none";
  int last_search_cycle = -1;
  double last_search_time = 0.0;
  double time_first_found = -1.0;
  bool converged = false;
};

//! Text-declared topology which must agree with the binary restart header.
struct Z4cMeshRestartState {
  int nx1 = 1;
  int nx2 = 1;
  int nx3 = 1;
  int meshblock_nx1 = 1;
  int meshblock_nx2 = 1;
  int meshblock_nx3 = 1;
};

struct Z4cRestartState {
  static constexpr int kLegacyCellCarrierSchema = 1;
  static constexpr int kCurrentCarrierSchema = 2;

  // Default construction remains the historical cell-centered carrier.
  // Native vertex centering opts into schema 2 through the validated factory.
  int carrier_schema = kLegacyCellCarrierSchema;
  Z4cSymmetryConfig config;
  Z4cGridLayout layout;
  int requested_spatial_order = 2;
  int effective_spatial_order = 2;
  Z4cMeshRestartState mesh;
  Z4cCentralRestartState central;
  Z4cM0FastFlowRestartState fastflow;
};

struct Z4cRestartSnapshot {
  bool present = false;
  Z4cRestartState state;
};

struct Z4cRestartResult {
  bool valid = false;
  std::string error;
};

inline Z4cRestartResult ValidateZ4cCentralRestartState(
    const Z4cCentralRestartState &state) {
  if (state.schema != Z4cCentralRestartState::kCurrentSchema ||
      !std::isfinite(state.proper_time) || state.proper_time < 0.0 ||
      !std::isfinite(state.previous_lapse) || state.previous_lapse < 0.0 ||
      !std::isfinite(state.constraint_norm) || state.constraint_norm < 0.0 ||
      !std::isfinite(state.abs_kretschmann) || state.abs_kretschmann < 0.0 ||
      !std::isfinite(state.last_time) || state.last_cycle < -1 ||
      state.sample_gid < -1 || state.sample_level < -1) {
    return {false, "invalid axis-central integration state in <z4c_restart>"};
  }
  if (state.initialized) {
    if (state.last_cycle < 0 || state.sample_gid < 0 || state.sample_level < 0) {
      return {false, "initialized axis-central state lacks sample metadata"};
    }
  } else if (state.proper_time != 0.0 || state.previous_lapse != 1.0 ||
             state.constraint_norm != 0.0 || state.abs_kretschmann != 0.0 ||
             state.sample_gid != -1 || state.sample_level != -1 ||
             state.last_cycle != -1 || state.last_time != 0.0) {
    return {false, "uninitialized axis-central state contains evolved values"};
  }
  return {true, ""};
}

inline Z4cRestartResult UpdateZ4cCentralRestartState(
    Z4cCentralRestartState *state, const double lapse,
    const double constraint_norm, const double abs_kretschmann,
    const int sample_gid, const int sample_level, const int cycle,
    const double time, const bool restart_initialization) {
  if (state == nullptr || !std::isfinite(lapse) || lapse < 0.0 ||
      !std::isfinite(constraint_norm) || constraint_norm < 0.0 ||
      !std::isfinite(abs_kretschmann) || abs_kretschmann < 0.0 ||
      sample_gid < 0 || sample_level < 0 || cycle < 0 || !std::isfinite(time)) {
    return {false, "invalid axis-central diagnostic sample"};
  }
  if (!state->initialized) {
    if (restart_initialization) {
      return {false, "Cartoon restart lacks initialized axis-central integration state"};
    }
    state->initialized = true;
    state->proper_time = 0.0;
  } else {
    const double scale = std::max({1.0, std::fabs(time), std::fabs(state->last_time)});
    const double tolerance = 32.0 * std::numeric_limits<double>::epsilon() * scale;
    if (restart_initialization) {
      if (state->last_cycle != cycle || std::fabs(state->last_time - time) > tolerance) {
        return {false, "axis-central restart cycle/time does not match the binary state"};
      }
    } else {
      const double dt = time - state->last_time;
      if (state->last_cycle < 0 || cycle != state->last_cycle + 1 ||
          !std::isfinite(dt) || dt <= tolerance) {
        return {false, "axis-central proper-time update is not one forward accepted step"};
      }
      state->proper_time += 0.5 * (state->previous_lapse + lapse) * dt;
    }
  }
  state->previous_lapse = lapse;
  state->constraint_norm = constraint_norm;
  state->abs_kretschmann = abs_kretschmann;
  state->sample_gid = sample_gid;
  state->sample_level = sample_level;
  state->last_cycle = cycle;
  state->last_time = time;
  return ValidateZ4cCentralRestartState(*state);
}

Z4cRestartState MakeDefaultZ4cRestartState(const Z4cSymmetryConfig &config,
                                           int requested_spatial_order,
                                           int nghost, int nx1, int nx2, int nx3,
                                           int meshblock_nx1, int meshblock_nx2,
                                           int meshblock_nx3);
Z4cRestartResult CaptureZ4cRestartSnapshot(ParameterInput *pin,
                                           Z4cRestartSnapshot *snapshot);
Z4cRestartResult ValidateAndRestoreZ4cRestartSnapshot(
    ParameterInput *pin, const Z4cRestartSnapshot &snapshot);
Z4cRestartResult ValidateZ4cRestartBinaryDimensions(
    ParameterInput *pin, int nx1, int nx2, int nx3, int meshblock_nx1,
    int meshblock_nx2, int meshblock_nx3);
void StoreZ4cRestartState(ParameterInput *pin, const Z4cRestartState &state);

}  // namespace z4c

#endif  // Z4C_Z4C_RESTART_HPP_
