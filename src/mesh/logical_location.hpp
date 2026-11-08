#ifndef MESH_LOGICAL_LOCATION_HPP_
#define MESH_LOGICAL_LOCATION_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>

// Logical indices are relative to a chart root. Chart zero is the Cartesian root;
// shell charts have independent trees but share global MeshBlock IDs and ownership.
// Keep chart_id last so existing four-element Cartesian aggregates remain valid.
struct LogicalLocation {
  std::int32_t lx1, lx2, lx3, level;
  std::int32_t chart_id = 0;
  static constexpr int nint = 5;
};
static_assert(std::is_standard_layout<LogicalLocation>::value &&
              std::is_trivially_copyable<LogicalLocation>::value &&
              sizeof(LogicalLocation) == LogicalLocation::nint*sizeof(std::int32_t),
              "LogicalLocation MPI representation must be five contiguous int32s");

// Restart records are explicit, so adding chart identity does not change the
// interpretation of existing four-integer Cartesian restart files.
namespace logical_location {
constexpr int current_version = 2;

inline std::size_t RecordSize(int version) {
  if (version != 1 && version != current_version) {
    throw std::runtime_error("Unsupported LogicalLocation restart version");
  }
  return (version == 1 ? 4 : 5)*sizeof(std::int32_t);
}

inline LogicalLocation Decode(const void *record, int version) {
  std::int32_t fields[5] = {};
  std::memcpy(fields, record, RecordSize(version));
  return {fields[0], fields[1], fields[2], fields[3], fields[4]};
}

inline void Encode(void *record, const LogicalLocation &loc, int version) {
  if (version == 1 && loc.chart_id != 0) {
    throw std::runtime_error("Cartesian restart format cannot store a shell chart");
  }
  const std::int32_t fields[5] = {loc.lx1, loc.lx2, loc.lx3, loc.level, loc.chart_id};
  std::memcpy(record, fields, RecordSize(version));
}
}  // namespace logical_location

#endif  // MESH_LOGICAL_LOCATION_HPP_
