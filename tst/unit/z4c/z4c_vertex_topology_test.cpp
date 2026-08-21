//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c_vertex_topology_test.cpp
//! \brief Canonical identity and role tests for VC faces, edges, and corners.

#include <iostream>
#include <string>

#include <Kokkos_Core.hpp>

#include "athena.hpp"
#include "bvals/vertex_topology.hpp"

namespace {

using vertex_topology::CanonicalVertexDirectionConfig;
using vertex_topology::ClassifyVertexNodeRole;
using vertex_topology::MakeCanonicalVertexDirectionKey;
using vertex_topology::MakeCanonicalVertexKey;
using vertex_topology::VertexNodeRole;
using vertex_topology::VertexRoleInput;

CanonicalVertexDirectionConfig Direction(const int nx, const bool collapsed = false,
                                         const bool periodic = false) {
  return {2, 6, 4, nx, collapsed, periodic};
}

bool CheckCanonicalIdentity() {
  const auto x = Direction(8);
  const auto collapsed = Direction(1, true);
  // Adjacent same-level blocks share the exact endpoint key.
  const auto left = MakeCanonicalVertexDirectionKey(x, 4, 5, 8);
  const auto right = MakeCanonicalVertexDirectionKey(x, 4, 6, 0);
  if (!left.valid || !right.valid || left.key != right.key) return false;

  // A coarse vertex and a coincident fine vertex share one promoted key.
  const auto coarse = MakeCanonicalVertexDirectionKey(x, 3, 2, 3);
  const auto fine = MakeCanonicalVertexDirectionKey(x, 4, 4, 6);
  if (!coarse.valid || !fine.valid || coarse.key != fine.key) return false;

  // Periodic upper endpoint canonicalizes to the lower endpoint.
  const auto periodic = Direction(8, false, true);
  const auto lower = MakeCanonicalVertexDirectionKey(periodic, 4, 0, 0);
  const auto upper = MakeCanonicalVertexDirectionKey(periodic, 4, 15, 8);
  if (!lower.valid || !upper.valid || lower.key != 0 || upper.key != lower.key) {
    return false;
  }

  // Collapsed coordinates have the unique key zero regardless of logical metadata.
  const auto singleton = MakeCanonicalVertexDirectionKey(collapsed, 4, 99, 17);
  if (!singleton.valid || singleton.key != 0) return false;

  // Invalid active offsets and overflowing promotion fail closed.
  if (MakeCanonicalVertexDirectionKey(x, 4, 1, 9).valid) return false;
  CanonicalVertexDirectionConfig overflow{0, 63, 1, 8, false, false};
  if (MakeCanonicalVertexDirectionKey(overflow, 0, 1, 0).valid) return false;
  return true;
}

bool CheckFaceEdgeCornerKeys(const int dimensions) {
  const auto active = Direction(8);
  const auto collapsed = Direction(1, true);
  const auto x1 = active;
  const auto x2 = dimensions >= 2 ? active : collapsed;
  const auto x3 = dimensions >= 3 ? active : collapsed;
  const int nx2 = dimensions >= 2 ? 8 : 0;
  const int nx3 = dimensions >= 3 ? 8 : 0;
  // Every nonempty subset of active directions represents a face/edge/corner.
  const int masks = (1 << dimensions);
  for (int mask = 1; mask < masks; ++mask) {
    const int o1 = (mask & 1) ? 8 : 3;
    const int o2 = dimensions >= 2 && (mask & 2) ? nx2 : (dimensions >= 2 ? 3 : 0);
    const int o3 = dimensions >= 3 && (mask & 4) ? nx3 : (dimensions >= 3 ? 3 : 0);
    const auto first = MakeCanonicalVertexKey(x1, x2, x3, 4, 2, 2, 2,
                                               o1, o2, o3);
    const auto second = MakeCanonicalVertexKey(
        x1, x2, x3, 4, 2 + ((mask & 1) ? 1 : 0),
        2 + ((mask & 2) ? 1 : 0), 2 + ((mask & 4) ? 1 : 0),
        (mask & 1) ? 0 : o1, (mask & 2) ? 0 : o2, (mask & 4) ? 0 : o3);
    if (!first.valid || !second.valid || first.i1 != second.i1 ||
        first.i2 != second.i2 || first.i3 != second.i3) {
      return false;
    }
  }
  return true;
}

bool CheckRolesAndDeviceRecord() {
  if (ClassifyVertexNodeRole({true, false, false, false, false, false, false}) !=
          VertexNodeRole::ghost ||
      ClassifyVertexNodeRole({false, true, false, true, false, false, false}) !=
          VertexNodeRole::axis ||
      ClassifyVertexNodeRole({false, false, true, false, false, false, false}) !=
          VertexNodeRole::physical_boundary ||
      ClassifyVertexNodeRole({false, false, false, true, false, false, false}) !=
          VertexNodeRole::shared_same_level ||
      ClassifyVertexNodeRole({false, false, false, false, true, true, false}) !=
          VertexNodeRole::hanging_fine_interface ||
      ClassifyVertexNodeRole({false, false, false, false, true, true, true}) !=
          VertexNodeRole::shared_coarse_fine_coincident ||
      ClassifyVertexNodeRole({}) != VertexNodeRole::independent_interior) {
    return false;
  }
  if (!vertex_topology::FineVertexCoincidentWithCoarse(4, 8, 0, false, false,
                                                        true) ||
      vertex_topology::FineVertexCoincidentWithCoarse(4, 7, 0, false, false,
                                                       true)) {
    return false;
  }

  Kokkos::View<vertex_topology::VertexTopologyRecord *> records("VC roles", 3);
  Kokkos::parallel_for(
      "construct VC role records", Kokkos::RangePolicy<DevExeSpace>(0, 3),
      KOKKOS_LAMBDA(const int index) {
        VertexRoleInput input;
        input.same_level_shared = index == 0;
        input.coarse_fine_interface = index > 0;
        input.fine_side = index > 0;
        input.coincident_with_coarse = index == 1;
        records(index).role = ClassifyVertexNodeRole(input);
        records(index).topological_multiplicity = static_cast<std::uint8_t>(index + 2);
      });
  Kokkos::fence();
  const auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), records);
  return host(0).role == VertexNodeRole::shared_same_level &&
         host(1).role == VertexNodeRole::shared_coarse_fine_coincident &&
         host(2).role == VertexNodeRole::hanging_fine_interface &&
         host(0).topological_multiplicity == 2 &&
         host(2).topological_multiplicity == 4;
}

bool CheckCartoonHalfPlane() {
  const auto radial = Direction(8);
  const auto axial = Direction(8);
  const auto suppressed = Direction(1, true);
  const auto axis_low = MakeCanonicalVertexKey(radial, axial, suppressed, 4,
                                                0, 3, 0, 0, 4, 0);
  const auto axis_high = MakeCanonicalVertexKey(radial, axial, suppressed, 5,
                                                 0, 6, 117, 0, 8, 99);
  return axis_low.valid && axis_high.valid && axis_low.i1 == 0 &&
         axis_low.i3 == 0 && axis_high.i1 == 0 && axis_high.i3 == 0 &&
         axis_low.i2 == axis_high.i2 &&
         ClassifyVertexNodeRole({false, true, false, false, true, true, true}) ==
             VertexNodeRole::axis;
}

}  // namespace

int main(int argc, char **argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  bool passed = CheckCanonicalIdentity() && CheckFaceEdgeCornerKeys(1) &&
                CheckFaceEdgeCornerKeys(2) && CheckFaceEdgeCornerKeys(3) &&
                CheckRolesAndDeviceRecord() && CheckCartoonHalfPlane();
  if (!passed) {
    std::cerr << "canonical VC topology contract failed\n";
    return 1;
  }
  std::cout << "canonical VC topology contract passed\n";
  return 0;
}
