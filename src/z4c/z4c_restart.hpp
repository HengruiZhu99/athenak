//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c_restart.hpp
//! \brief Immutable host-only Z4c restart configuration and integration-state carrier.

#ifndef Z4C_Z4C_RESTART_HPP_
#define Z4C_Z4C_RESTART_HPP_

#include <string>
#include <vector>

#include "z4c/z4c_symmetry.hpp"

class ParameterInput;

namespace z4c {

inline constexpr const char *kZ4cRestartBlock = "z4c_restart";

struct Z4cCentralRestartState {
  static constexpr int kCurrentSchema = 1;

  int schema = kCurrentSchema;
  double proper_time = 0.0;
  double previous_lapse = 1.0;
  int last_cycle = -1;
  double last_time = 0.0;
};

//! Reserved state for the later m=0 FastFlow implementation.
struct Z4cM0FastFlowRestartState {
  static constexpr int kCurrentSchema = 1;

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
  static constexpr int kCurrentCarrierSchema = 1;

  int carrier_schema = kCurrentCarrierSchema;
  Z4cSymmetryConfig config;
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
