//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file vertex_topology.hpp
//! \brief Integer-only canonical identity and role metadata for vertex-centered fields.

#ifndef BVALS_VERTEX_TOPOLOGY_HPP_
#define BVALS_VERTEX_TOPOLOGY_HPP_

#include <cstdint>
#include <type_traits>

#include <Kokkos_Macros.hpp>

namespace vertex_topology {

enum class VertexNodeRole : std::uint8_t {
  independent_interior = 0,
  shared_same_level,
  shared_coarse_fine_coincident,
  hanging_fine_interface,
  physical_boundary,
  axis,
  ghost,
};

struct CanonicalVertexDirectionConfig {
  int root_level = 0;
  int configured_max_level = 0;
  int root_meshblocks = 1;
  int intervals_per_meshblock = 1;
  bool collapsed = false;
  bool periodic = false;
};

struct CanonicalVertexDirectionResult {
  std::uint64_t key = 0;
  bool valid = false;
};

struct CanonicalVertexKey {
  std::uint64_t i1 = 0;
  std::uint64_t i2 = 0;
  std::uint64_t i3 = 0;
  bool valid = false;
};

struct VertexRoleInput {
  bool ghost = false;
  bool axis = false;
  bool physical_boundary = false;
  bool same_level_shared = false;
  bool coarse_fine_interface = false;
  bool fine_side = false;
  bool coincident_with_coarse = false;
};

//! Compact per-local-copy device record.  Contributor lists live in a separate
//! deterministic gather plan; this record is deliberately POD and backend portable.
struct VertexTopologyRecord {
  CanonicalVertexKey key;
  VertexNodeRole role = VertexNodeRole::ghost;
  std::uint8_t topological_multiplicity = 0;
  std::uint8_t canonical_diagnostic_owner = 0;
};

static_assert(std::is_trivially_copyable_v<CanonicalVertexKey>);
static_assert(std::is_trivially_copyable_v<VertexTopologyRecord>);

KOKKOS_INLINE_FUNCTION constexpr bool SafeLeftShift(
    const std::uint64_t value, const int shift, std::uint64_t *result) {
  if (shift < 0 || shift >= 64) return false;
  constexpr std::uint64_t maximum = ~static_cast<std::uint64_t>(0);
  if (shift > 0 && value > (maximum >> shift)) return false;
  *result = value << shift;
  return true;
}

//! Form one direction of the dyadic global vertex key.  `local_vertex_offset`
//! addresses active vertices only, in [0,nx].  Periodic upper endpoints are
//! canonicalized to zero before promotion to the configured maximum level.
KOKKOS_INLINE_FUNCTION constexpr CanonicalVertexDirectionResult
MakeCanonicalVertexDirectionKey(
    const CanonicalVertexDirectionConfig config, const int level,
    const std::int64_t logical_meshblock, const int local_vertex_offset) {
  if (config.collapsed) return {0, true};
  if (config.root_level < 0 || level < config.root_level ||
      config.configured_max_level < level || config.root_meshblocks <= 0 ||
      config.intervals_per_meshblock <= 0 || logical_meshblock < 0 ||
      local_vertex_offset < 0 ||
      local_vertex_offset > config.intervals_per_meshblock) {
    return {0, false};
  }
  const auto nx = static_cast<std::uint64_t>(config.intervals_per_meshblock);
  const auto location = static_cast<std::uint64_t>(logical_meshblock);
  constexpr std::uint64_t maximum = ~static_cast<std::uint64_t>(0);
  if (location > (maximum - static_cast<std::uint64_t>(local_vertex_offset)) / nx) {
    return {0, false};
  }
  std::uint64_t level_node = location * nx +
                             static_cast<std::uint64_t>(local_vertex_offset);
  if (config.periodic) {
    std::uint64_t root_period = 0;
    if (static_cast<std::uint64_t>(config.root_meshblocks) > maximum / nx) {
      return {0, false};
    }
    root_period = static_cast<std::uint64_t>(config.root_meshblocks) * nx;
    std::uint64_t level_period = 0;
    if (!SafeLeftShift(root_period, level - config.root_level, &level_period) ||
        level_node > level_period) {
      return {0, false};
    }
    if (level_node == level_period) level_node = 0;
  }
  std::uint64_t canonical = 0;
  if (!SafeLeftShift(level_node, config.configured_max_level - level,
                     &canonical)) {
    return {0, false};
  }
  return {canonical, true};
}

KOKKOS_INLINE_FUNCTION constexpr CanonicalVertexKey MakeCanonicalVertexKey(
    const CanonicalVertexDirectionConfig x1,
    const CanonicalVertexDirectionConfig x2,
    const CanonicalVertexDirectionConfig x3, const int level,
    const std::int64_t lx1, const std::int64_t lx2, const std::int64_t lx3,
    const int ox1, const int ox2, const int ox3) {
  const auto k1 = MakeCanonicalVertexDirectionKey(x1, level, lx1, ox1);
  const auto k2 = MakeCanonicalVertexDirectionKey(x2, level, lx2, ox2);
  const auto k3 = MakeCanonicalVertexDirectionKey(x3, level, lx3, ox3);
  return {k1.key, k2.key, k3.key, k1.valid && k2.valid && k3.valid};
}

//! A fine vertex coincides with the next-coarser lattice exactly when every
//! noncollapsed level-local integer coordinate is even.
KOKKOS_INLINE_FUNCTION constexpr bool FineVertexCoincidentWithCoarse(
    const std::int64_t level_i1, const std::int64_t level_i2,
    const std::int64_t level_i3, const bool x1_collapsed,
    const bool x2_collapsed, const bool x3_collapsed) {
  return (x1_collapsed || (level_i1 & 1LL) == 0) &&
         (x2_collapsed || (level_i2 & 1LL) == 0) &&
         (x3_collapsed || (level_i3 & 1LL) == 0);
}

KOKKOS_INLINE_FUNCTION constexpr VertexNodeRole ClassifyVertexNodeRole(
    const VertexRoleInput input) {
  if (input.ghost) return VertexNodeRole::ghost;
  if (input.axis) return VertexNodeRole::axis;
  if (input.physical_boundary) return VertexNodeRole::physical_boundary;
  if (input.coarse_fine_interface) {
    if (input.fine_side && !input.coincident_with_coarse) {
      return VertexNodeRole::hanging_fine_interface;
    }
    return VertexNodeRole::shared_coarse_fine_coincident;
  }
  if (input.same_level_shared) return VertexNodeRole::shared_same_level;
  return VertexNodeRole::independent_interior;
}

KOKKOS_INLINE_FUNCTION constexpr const char *VertexNodeRoleName(
    const VertexNodeRole role) {
  switch (role) {
    case VertexNodeRole::independent_interior: return "independent_interior";
    case VertexNodeRole::shared_same_level: return "shared_same_level";
    case VertexNodeRole::shared_coarse_fine_coincident:
      return "shared_coarse_fine_coincident";
    case VertexNodeRole::hanging_fine_interface: return "hanging_fine_interface";
    case VertexNodeRole::physical_boundary: return "physical_boundary";
    case VertexNodeRole::axis: return "axis";
    case VertexNodeRole::ghost: return "ghost";
  }
  return "invalid";
}

}  // namespace vertex_topology

#endif  // BVALS_VERTEX_TOPOLOGY_HPP_
