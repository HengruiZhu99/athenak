#ifndef PC_GH_STATE_LAYOUT_HPP_
#define PC_GH_STATE_LAYOUT_HPP_

#include <stdexcept>
#include <string>
#include "parameter_input.hpp"

namespace pc_gh {
struct StateLayout {
  std::string name;
  int version;
  int fields;
};
inline StateLayout LayoutForFormulation(const std::string &formulation) {
  if (formulation == "legacy") return {"legacy_pcgh55", 1, 55};
  if (formulation == "intrinsic_clean") return {"intrinsic_pcgh50", 1, 50};
  throw std::runtime_error("unknown PC-GH formulation: " + formulation);
}
inline std::string RequestedFormulation(ParameterInput *pin) {
  return pin->DoesParameterExist("pc_gh", "formulation")
      ? pin->GetString("pc_gh", "formulation") : "legacy";
}
struct SavedStateLayout {
  bool has_pcgh = false;
  bool tagged = false;
  StateLayout layout{"", 0, 0};
};
// Capture this from the restart header BEFORE input/command-line overrides.
inline SavedStateLayout ReadSavedStateLayout(ParameterInput *pin) {
  SavedStateLayout saved;
  saved.has_pcgh = pin->DoesBlockExist("pc_gh");
  if (!saved.has_pcgh) return saved;
  bool name = pin->DoesParameterExist("pc_gh", "restart_layout");
  bool version = pin->DoesParameterExist("pc_gh", "restart_layout_version");
  bool fields = pin->DoesParameterExist("pc_gh", "restart_layout_fields");
  if (!(name || version || fields)) return saved;
  if (!(name && version && fields)) {
    throw std::runtime_error("incomplete PC-GH restart layout metadata");
  }
  saved.tagged = true;
  saved.layout = {pin->GetString("pc_gh", "restart_layout"),
                  pin->GetInteger("pc_gh", "restart_layout_version"),
                  pin->GetInteger("pc_gh", "restart_layout_fields")};
  auto expected = LayoutForFormulation(RequestedFormulation(pin));
  if (saved.layout.name != expected.name || saved.layout.version != expected.version
      || saved.layout.fields != expected.fields) {
    throw std::runtime_error("PC-GH restart layout metadata is unknown or inconsistent");
  }
  return saved;
}
inline void ValidateRestartLayout(const SavedStateLayout &saved, ParameterInput *pin) {
  bool requested = pin->DoesBlockExist("pc_gh");
  if (saved.has_pcgh != requested) {
    throw std::runtime_error("cannot add/remove PC-GH physics when reading a restart");
  }
  if (!requested) return;
  auto expected = LayoutForFormulation(RequestedFormulation(pin));
  auto actual = saved.layout;
  if (!saved.tagged) {
    // A bare array length cannot distinguish historical 55-field formulations.
    const std::string declaration = pin->DoesParameterExist("pc_gh", "restart_untagged_layout")
        ? pin->GetString("pc_gh", "restart_untagged_layout") : "";
    if (declaration != "legacy_pcgh55") {
      throw std::runtime_error("untagged PC-GH restart: establish collision-lineage "
          "provenance, then explicitly set pc_gh/restart_untagged_layout=legacy_pcgh55");
    }
    actual = LayoutForFormulation("legacy");
  }
  if (actual.name != expected.name || actual.version != expected.version
      || actual.fields != expected.fields) {
    throw std::runtime_error("incompatible PC-GH restart layout; conversion is required");
  }
  // Overrides may not even alter the protected descriptive metadata.
  for (const char *key : {"restart_layout", "restart_layout_version", "restart_layout_fields"}) {
    if (!pin->DoesParameterExist("pc_gh", key)) continue;
    std::string expected_value = std::string(key) == "restart_layout" ? actual.name
        : std::to_string(std::string(key) == "restart_layout_version" ? actual.version
                                                                         : actual.fields);
    if (pin->GetString("pc_gh", key) != expected_value) {
      throw std::runtime_error("PC-GH restart layout metadata cannot be overridden");
    }
  }
}
inline void WriteStateLayout(ParameterInput *pin, const StateLayout &layout) {
  pin->SetString("pc_gh", "restart_layout", layout.name);
  pin->SetInteger("pc_gh", "restart_layout_version", layout.version);
  pin->SetInteger("pc_gh", "restart_layout_fields", layout.fields);
}
}  // namespace pc_gh
#endif  // PC_GH_STATE_LAYOUT_HPP_
