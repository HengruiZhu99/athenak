//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c_restart.cpp
//! \brief Immutable host-only Z4c restart carrier implementation.

#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "parameter_input.hpp"
#include "z4c/z4c_restart.hpp"

namespace z4c {
namespace {

Z4cRestartResult Invalid(const std::string &message) { return {false, message}; }

std::string FormatDouble(const double value) {
  std::ostringstream stream;
  stream << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
  return stream.str();
}

std::string FormatCoefficients(const std::vector<double> &coefficients) {
  if (coefficients.empty()) return "none";
  std::ostringstream stream;
  stream << std::setprecision(std::numeric_limits<double>::max_digits10);
  for (std::size_t n = 0; n < coefficients.size(); ++n) {
    if (n != 0) stream << ',';
    stream << coefficients[n];
  }
  return stream.str();
}

bool ParseCoefficients(const std::string &text, const int expected_count,
                       std::vector<double> *coefficients) {
  coefficients->clear();
  if (expected_count == 0) return text == "none";
  std::istringstream stream(text);
  std::string token;
  while (std::getline(stream, token, ',')) {
    try {
      std::size_t parsed = 0;
      const double value = std::stod(token, &parsed);
      if (parsed != token.size() || !std::isfinite(value)) return false;
      coefficients->push_back(value);
    } catch (...) {
      return false;
    }
  }
  return static_cast<int>(coefficients->size()) == expected_count;
}

bool Has(ParameterInput *pin, const char *key) {
  return pin->DoesParameterExist(kZ4cRestartBlock, key);
}

bool ReadInteger(ParameterInput *pin, const char *key, int *value) {
  const std::string text = pin->GetString(kZ4cRestartBlock, key);
  try {
    std::size_t parsed = 0;
    const long long candidate = std::stoll(text, &parsed);
    if (parsed != text.size() || candidate < std::numeric_limits<int>::min() ||
        candidate > std::numeric_limits<int>::max()) {
      return false;
    }
    *value = static_cast<int>(candidate);
    return true;
  } catch (...) {
    return false;
  }
}

bool ReadDouble(ParameterInput *pin, const char *key, double *value) {
  const std::string text = pin->GetString(kZ4cRestartBlock, key);
  try {
    std::size_t parsed = 0;
    *value = std::stod(text, &parsed);
    return parsed == text.size();
  } catch (...) {
    return false;
  }
}

bool ReadBoolean(ParameterInput *pin, const char *key, bool *value) {
  const std::string text = pin->GetString(kZ4cRestartBlock, key);
  if (text == "1" || text == "true") {
    *value = true;
    return true;
  }
  if (text == "0" || text == "false") {
    *value = false;
    return true;
  }
  return false;
}

Z4cRestartResult InvalidType(ParameterInput *pin, const char *key,
                              const char *type) {
  return Invalid(std::string("invalid typed <") + kZ4cRestartBlock + ">/" + key +
                 "='" + pin->GetString(kZ4cRestartBlock, key) +
                 "'; expected " + type);
}

Z4cRestartResult RequireKeys(ParameterInput *pin) {
  for (const char *key : {
           "carrier_schema", "symmetry", "coordinate_map", "symmetry_schema",
           "requested_spatial_order", "effective_spatial_order", "stencil_width",
           "central_schema", "central_proper_time", "central_previous_lapse",
           "central_last_cycle", "central_last_time", "fastflow_schema",
           "fastflow_coefficient_count", "fastflow_coefficients",
           "fastflow_surface_mode", "fastflow_selected_branch", "fastflow_center_count",
           "fastflow_center_z0", "fastflow_center_z1", "fastflow_status",
           "fastflow_failure_code", "fastflow_last_search_cycle",
           "fastflow_last_search_time", "fastflow_converged"}) {
    if (!Has(pin, key)) {
      return Invalid(std::string("partial <") + kZ4cRestartBlock + "> carrier: missing <" +
                     kZ4cRestartBlock + ">/" + key);
    }
  }
  return {true, ""};
}

Z4cRestartResult ReadState(ParameterInput *pin, Z4cRestartState *state) {
  const auto required = RequireKeys(pin);
  if (!required.valid) return required;

#define READ_INTEGER(KEY, DESTINATION)                      \
  if (!ReadInteger(pin, KEY, &DESTINATION)) {              \
    return InvalidType(pin, KEY, "integer");              \
  }
#define READ_DOUBLE(KEY, DESTINATION)                       \
  if (!ReadDouble(pin, KEY, &DESTINATION)) {               \
    return InvalidType(pin, KEY, "finite real");          \
  }
#define READ_BOOLEAN(KEY, DESTINATION)                      \
  if (!ReadBoolean(pin, KEY, &DESTINATION)) {              \
    return InvalidType(pin, KEY, "boolean");              \
  }
  READ_INTEGER("carrier_schema", state->carrier_schema)
  if (state->carrier_schema != Z4cRestartState::kCurrentCarrierSchema) {
    return Invalid("unsupported <z4c_restart>/carrier_schema=" +
                   std::to_string(state->carrier_schema));
  }

  const std::string symmetry = pin->GetString(kZ4cRestartBlock, "symmetry");
  const std::string coordinate_map =
      pin->GetString(kZ4cRestartBlock, "coordinate_map");
  if (symmetry == "cartesian3d") {
    state->config.mode = Z4cSymmetryMode::cartesian3d;
  } else if (symmetry == "cartoon_so2") {
    state->config.mode = Z4cSymmetryMode::cartoon_so2;
  } else {
    return Invalid("invalid <z4c_restart>/symmetry='" + symmetry + "'");
  }
  if (coordinate_map == "cartesian_xyz") {
    state->config.coordinate_map = Z4cCoordinateMap::cartesian_xyz;
  } else if (coordinate_map == "signed_rho_z_suppressed_y_v1") {
    state->config.coordinate_map = Z4cCoordinateMap::signed_rho_z_suppressed_y_v1;
  } else {
    return Invalid("invalid <z4c_restart>/coordinate_map='" + coordinate_map + "'");
  }
  READ_INTEGER("symmetry_schema", state->config.schema)
  READ_INTEGER("requested_spatial_order", state->requested_spatial_order)
  READ_INTEGER("effective_spatial_order", state->effective_spatial_order)
  READ_INTEGER("stencil_width", state->config.stencil_width)

  const bool map_matches_mode =
      (state->config.mode == Z4cSymmetryMode::cartesian3d &&
       state->config.coordinate_map == Z4cCoordinateMap::cartesian_xyz) ||
      (state->config.mode == Z4cSymmetryMode::cartoon_so2 &&
       state->config.coordinate_map == Z4cCoordinateMap::signed_rho_z_suppressed_y_v1);
  if (!map_matches_mode || state->config.schema != Z4cSymmetryConfig::kCurrentSchema) {
    return Invalid("inconsistent symmetry/map/schema in <z4c_restart>");
  }
  const int nghost = pin->GetInteger("mesh", "nghost");
  const int expected_order =
      EffectiveZ4cSpatialOrder(state->requested_spatial_order, nghost);
  if (state->effective_spatial_order != expected_order ||
      state->config.stencil_width != expected_order / 2 + 1 ||
      (expected_order != 2 && expected_order != 4 && expected_order != 6)) {
    return Invalid("inconsistent spatial-order/stencil state in <z4c_restart>");
  }

  READ_INTEGER("central_schema", state->central.schema)
  READ_DOUBLE("central_proper_time", state->central.proper_time)
  READ_DOUBLE("central_previous_lapse", state->central.previous_lapse)
  READ_INTEGER("central_last_cycle", state->central.last_cycle)
  READ_DOUBLE("central_last_time", state->central.last_time)
  if (state->central.schema != Z4cCentralRestartState::kCurrentSchema ||
      !std::isfinite(state->central.proper_time) ||
      !std::isfinite(state->central.previous_lapse) ||
      !std::isfinite(state->central.last_time) || state->central.last_cycle < -1) {
    return Invalid("invalid axis-central integration state in <z4c_restart>");
  }

  READ_INTEGER("fastflow_schema", state->fastflow.schema)
  int coefficient_count = 0;
  READ_INTEGER("fastflow_coefficient_count", coefficient_count)
  if (state->fastflow.schema != Z4cM0FastFlowRestartState::kCurrentSchema ||
      coefficient_count < 0 || coefficient_count > 4096 ||
      !ParseCoefficients(pin->GetString(kZ4cRestartBlock, "fastflow_coefficients"),
                         coefficient_count, &state->fastflow.coefficients)) {
    return Invalid("invalid m=0 FastFlow coefficient state in <z4c_restart>");
  }
  state->fastflow.surface_mode =
      pin->GetString(kZ4cRestartBlock, "fastflow_surface_mode");
  state->fastflow.selected_branch =
      pin->GetString(kZ4cRestartBlock, "fastflow_selected_branch");
  READ_INTEGER("fastflow_center_count", state->fastflow.center_count)
  READ_DOUBLE("fastflow_center_z0", state->fastflow.center_z0)
  READ_DOUBLE("fastflow_center_z1", state->fastflow.center_z1)
  state->fastflow.status = pin->GetString(kZ4cRestartBlock, "fastflow_status");
  state->fastflow.failure_code =
      pin->GetString(kZ4cRestartBlock, "fastflow_failure_code");
  READ_INTEGER("fastflow_last_search_cycle", state->fastflow.last_search_cycle)
  READ_DOUBLE("fastflow_last_search_time", state->fastflow.last_search_time)
  READ_BOOLEAN("fastflow_converged", state->fastflow.converged)
  if (state->fastflow.center_count < 0 || state->fastflow.center_count > 2 ||
      !std::isfinite(state->fastflow.center_z0) ||
      !std::isfinite(state->fastflow.center_z1) ||
      state->fastflow.last_search_cycle < -1 ||
      !std::isfinite(state->fastflow.last_search_time)) {
    return Invalid("invalid m=0 FastFlow integration state in <z4c_restart>");
  }
  return {true, ""};
#undef READ_INTEGER
#undef READ_DOUBLE
#undef READ_BOOLEAN
}

Z4cRestartResult Conflict(const char *block, const char *key,
                          const std::string &stored,
                          const std::string &requested) {
  return Invalid(std::string("conflicting restart override <") + block + ">/" + key +
                 ": stored='" + stored + "' requested='" + requested + "'");
}

template <typename Stored, typename Requested>
Z4cRestartResult Compare(const char *block, const char *key, const Stored &stored,
                         const Requested &requested) {
  if (stored == requested) return {true, ""};
  std::ostringstream stored_stream;
  std::ostringstream requested_stream;
  stored_stream << stored;
  requested_stream << requested;
  return Conflict(block, key, stored_stream.str(), requested_stream.str());
}

Z4cRestartResult CompareCarrierParameters(ParameterInput *pin,
                                           const Z4cRestartState &stored) {
#define COMPARE_INTEGER(KEY, STORED)                              \
  {                                                              \
    int requested = 0;                                           \
    if (!ReadInteger(pin, KEY, &requested)) {                     \
      return Conflict(kZ4cRestartBlock, KEY, std::to_string(STORED), \
                      pin->GetString(kZ4cRestartBlock, KEY));     \
    }                                                            \
    const auto result = Compare(kZ4cRestartBlock, KEY, STORED, requested); \
    if (!result.valid) return result;                             \
  }
#define COMPARE_DOUBLE(KEY, STORED)                               \
  {                                                              \
    double requested = 0.0;                                      \
    if (!ReadDouble(pin, KEY, &requested)) {                      \
      return Conflict(kZ4cRestartBlock, KEY, FormatDouble(STORED), \
                      pin->GetString(kZ4cRestartBlock, KEY));     \
    }                                                            \
    const auto result = Compare(kZ4cRestartBlock, KEY, STORED, requested); \
    if (!result.valid) return result;                             \
  }
#define COMPARE_STRING(KEY, STORED)                              \
  {                                                             \
    const auto result = Compare(kZ4cRestartBlock, KEY, STORED,  \
        pin->GetString(kZ4cRestartBlock, KEY));                  \
    if (!result.valid) return result;                            \
  }
#define COMPARE_BOOLEAN(KEY, STORED)                              \
  {                                                             \
    bool requested = false;                                     \
    if (!ReadBoolean(pin, KEY, &requested)) {                    \
      return Conflict(kZ4cRestartBlock, KEY, STORED ? "1" : "0", \
                      pin->GetString(kZ4cRestartBlock, KEY));    \
    }                                                           \
    const auto result = Compare(kZ4cRestartBlock, KEY, STORED, requested); \
    if (!result.valid) return result;                            \
  }
  COMPARE_INTEGER("carrier_schema", stored.carrier_schema)
  COMPARE_STRING("symmetry", std::string(ToString(stored.config.mode)))
  COMPARE_STRING("coordinate_map",
                 std::string(ToString(stored.config.coordinate_map)))
  COMPARE_INTEGER("symmetry_schema", stored.config.schema)
  COMPARE_INTEGER("requested_spatial_order", stored.requested_spatial_order)
  COMPARE_INTEGER("effective_spatial_order", stored.effective_spatial_order)
  COMPARE_INTEGER("stencil_width", stored.config.stencil_width)
  COMPARE_INTEGER("central_schema", stored.central.schema)
  COMPARE_DOUBLE("central_proper_time", stored.central.proper_time)
  COMPARE_DOUBLE("central_previous_lapse", stored.central.previous_lapse)
  COMPARE_INTEGER("central_last_cycle", stored.central.last_cycle)
  COMPARE_DOUBLE("central_last_time", stored.central.last_time)
  COMPARE_INTEGER("fastflow_schema", stored.fastflow.schema)
  COMPARE_INTEGER("fastflow_coefficient_count",
                  static_cast<int>(stored.fastflow.coefficients.size()))
  COMPARE_STRING("fastflow_surface_mode", stored.fastflow.surface_mode)
  COMPARE_STRING("fastflow_selected_branch", stored.fastflow.selected_branch)
  COMPARE_INTEGER("fastflow_center_count", stored.fastflow.center_count)
  COMPARE_DOUBLE("fastflow_center_z0", stored.fastflow.center_z0)
  COMPARE_DOUBLE("fastflow_center_z1", stored.fastflow.center_z1)
  COMPARE_STRING("fastflow_status", stored.fastflow.status)
  COMPARE_STRING("fastflow_failure_code", stored.fastflow.failure_code)
  COMPARE_INTEGER("fastflow_last_search_cycle", stored.fastflow.last_search_cycle)
  COMPARE_DOUBLE("fastflow_last_search_time", stored.fastflow.last_search_time)
  COMPARE_BOOLEAN("fastflow_converged", stored.fastflow.converged)
#undef COMPARE_INTEGER
#undef COMPARE_DOUBLE
#undef COMPARE_STRING
#undef COMPARE_BOOLEAN

  const std::string requested_coefficients =
      pin->GetString(kZ4cRestartBlock, "fastflow_coefficients");
  std::vector<double> parsed_coefficients;
  if (!ParseCoefficients(requested_coefficients,
                         static_cast<int>(stored.fastflow.coefficients.size()),
                         &parsed_coefficients) ||
      parsed_coefficients != stored.fastflow.coefficients) {
    return Conflict(kZ4cRestartBlock, "fastflow_coefficients",
                    FormatCoefficients(stored.fastflow.coefficients),
                    requested_coefficients);
  }
  return {true, ""};
}

}  // namespace

Z4cRestartState MakeDefaultZ4cRestartState(const Z4cSymmetryConfig &config,
                                           const int requested_spatial_order,
                                           const int nghost) {
  Z4cRestartState state;
  state.config = config;
  state.requested_spatial_order = requested_spatial_order;
  state.effective_spatial_order =
      EffectiveZ4cSpatialOrder(requested_spatial_order, nghost);
  return state;
}

Z4cRestartResult CaptureZ4cRestartSnapshot(ParameterInput *pin,
                                           Z4cRestartSnapshot *snapshot) {
  snapshot->present = false;
  if (!pin->DoesBlockExist(kZ4cRestartBlock)) return {true, ""};
  const auto result = ReadState(pin, &snapshot->state);
  if (!result.valid) return result;
  snapshot->present = true;
  return {true, ""};
}

Z4cRestartResult ValidateAndRestoreZ4cRestartSnapshot(
    ParameterInput *pin, const Z4cRestartSnapshot &snapshot) {
  if (!snapshot.present) return {true, ""};
  if (!pin->DoesBlockExist(kZ4cRestartBlock)) {
    return Invalid("restart override removed authoritative <z4c_restart> carrier");
  }
  auto result = RequireKeys(pin);
  if (!result.valid) return result;
  result = CompareCarrierParameters(pin, snapshot.state);
  if (!result.valid) return result;
  const int nghost = pin->GetInteger("mesh", "nghost");
  const int override_effective_order = EffectiveZ4cSpatialOrder(
      snapshot.state.requested_spatial_order, nghost);
  result = Compare("z4c", "effective_spatial_order",
                   snapshot.state.effective_spatial_order,
                   override_effective_order);
  if (!result.valid) return result;
  Z4cRestartState requested;
  result = ReadState(pin, &requested);
  if (!result.valid) return result;

  const std::string requested_symmetry =
      pin->DoesParameterExist("z4c", "symmetry")
          ? pin->GetString("z4c", "symmetry") : "cartesian3d";
  const std::string requested_map =
      pin->DoesParameterExist("z4c", "coordinate_map")
          ? pin->GetString("z4c", "coordinate_map")
          : (requested_symmetry == "cartoon_so2"
                 ? "signed_rho_z_suppressed_y_v1" : "cartesian_xyz");
  const int requested_schema = pin->DoesParameterExist("z4c", "symmetry_schema")
                                   ? pin->GetInteger("z4c", "symmetry_schema")
                                   : Z4cSymmetryConfig::kCurrentSchema;
  const int requested_order = pin->DoesParameterExist("z4c", "spatial_order")
                                  ? pin->GetInteger("z4c", "spatial_order")
                                  : 2 * (nghost - 1);
  if (requested_symmetry != ToString(snapshot.state.config.mode)) {
    return Conflict("z4c", "symmetry", ToString(snapshot.state.config.mode),
                    requested_symmetry);
  }
  if (requested_map != ToString(snapshot.state.config.coordinate_map)) {
    return Conflict("z4c", "coordinate_map",
                    ToString(snapshot.state.config.coordinate_map),
                    requested_map);
  }
  result = Compare("z4c", "symmetry_schema", snapshot.state.config.schema,
                   requested_schema);
  if (!result.valid) return result;
  result = Compare("z4c", "spatial_order", snapshot.state.requested_spatial_order,
                   requested_order);
  if (!result.valid) return result;
  result = Compare("z4c", "effective_spatial_order",
                   snapshot.state.effective_spatial_order,
                   EffectiveZ4cSpatialOrder(requested_order, nghost));
  if (!result.valid) return result;

  StoreZ4cRestartState(pin, snapshot.state);
  pin->SetString("z4c", "symmetry", ToString(snapshot.state.config.mode));
  pin->SetString("z4c", "coordinate_map", ToString(snapshot.state.config.coordinate_map));
  pin->SetInteger("z4c", "symmetry_schema", snapshot.state.config.schema);
  pin->SetInteger("z4c", "spatial_order", snapshot.state.requested_spatial_order);
  pin->SetString("z4c", "restart_symmetry", ToString(snapshot.state.config.mode));
  pin->SetString("z4c", "restart_coordinate_map",
                 ToString(snapshot.state.config.coordinate_map));
  pin->SetInteger("z4c", "restart_symmetry_schema", snapshot.state.config.schema);
  return {true, ""};
}

void StoreZ4cRestartState(ParameterInput *pin, const Z4cRestartState &state) {
  pin->SetInteger(kZ4cRestartBlock, "carrier_schema", state.carrier_schema);
  pin->SetString(kZ4cRestartBlock, "symmetry", ToString(state.config.mode));
  pin->SetString(kZ4cRestartBlock, "coordinate_map",
                 ToString(state.config.coordinate_map));
  pin->SetInteger(kZ4cRestartBlock, "symmetry_schema", state.config.schema);
  pin->SetInteger(kZ4cRestartBlock, "requested_spatial_order",
                  state.requested_spatial_order);
  pin->SetInteger(kZ4cRestartBlock, "effective_spatial_order",
                  state.effective_spatial_order);
  pin->SetInteger(kZ4cRestartBlock, "stencil_width", state.config.stencil_width);
  pin->SetInteger(kZ4cRestartBlock, "central_schema", state.central.schema);
  pin->SetString(kZ4cRestartBlock, "central_proper_time",
                 FormatDouble(state.central.proper_time));
  pin->SetString(kZ4cRestartBlock, "central_previous_lapse",
                 FormatDouble(state.central.previous_lapse));
  pin->SetInteger(kZ4cRestartBlock, "central_last_cycle", state.central.last_cycle);
  pin->SetString(kZ4cRestartBlock, "central_last_time",
                 FormatDouble(state.central.last_time));
  pin->SetInteger(kZ4cRestartBlock, "fastflow_schema", state.fastflow.schema);
  pin->SetInteger(kZ4cRestartBlock, "fastflow_coefficient_count",
                  static_cast<int>(state.fastflow.coefficients.size()));
  pin->SetString(kZ4cRestartBlock, "fastflow_coefficients",
                 FormatCoefficients(state.fastflow.coefficients));
  pin->SetString(kZ4cRestartBlock, "fastflow_surface_mode",
                 state.fastflow.surface_mode);
  pin->SetString(kZ4cRestartBlock, "fastflow_selected_branch",
                 state.fastflow.selected_branch);
  pin->SetInteger(kZ4cRestartBlock, "fastflow_center_count",
                  state.fastflow.center_count);
  pin->SetString(kZ4cRestartBlock, "fastflow_center_z0",
                 FormatDouble(state.fastflow.center_z0));
  pin->SetString(kZ4cRestartBlock, "fastflow_center_z1",
                 FormatDouble(state.fastflow.center_z1));
  pin->SetString(kZ4cRestartBlock, "fastflow_status", state.fastflow.status);
  pin->SetString(kZ4cRestartBlock, "fastflow_failure_code",
                 state.fastflow.failure_code);
  pin->SetInteger(kZ4cRestartBlock, "fastflow_last_search_cycle",
                  state.fastflow.last_search_cycle);
  pin->SetString(kZ4cRestartBlock, "fastflow_last_search_time",
                 FormatDouble(state.fastflow.last_search_time));
  pin->SetBoolean(kZ4cRestartBlock, "fastflow_converged", state.fastflow.converged);
}

}  // namespace z4c
