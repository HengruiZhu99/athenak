//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c_vertex_topology.hpp
//! \brief Production topology-derived role map for native VC Z4c storage.

#ifndef Z4C_Z4C_VERTEX_TOPOLOGY_HPP_
#define Z4C_Z4C_VERTEX_TOPOLOGY_HPP_

#include <vector>

#include "athena.hpp"
#include "bvals/vertex_topology.hpp"
#include "z4c/z4c_grid.hpp"

class MeshBlockPack;

namespace z4c {

struct VertexContributor {
  std::uint64_t key1 = 0;
  std::uint64_t key2 = 0;
  std::uint64_t key3 = 0;
  int level = 0;
  int lx1 = 0;
  int lx2 = 0;
  int lx3 = 0;
  int gid = -1;
  int m = -1;
  int k = -1;
  int j = -1;
  int i = -1;
};

static_assert(std::is_trivially_copyable_v<VertexContributor>);

bool VertexContributorLess(const VertexContributor &left,
                           const VertexContributor &right);

constexpr bool VertexContributorHasAuthority(const int contributor_level,
                                              const int maximum_level) {
  return contributor_level == maximum_level;
}

class Z4cVertexTopologyPlan {
 public:
  Z4cVertexTopologyPlan() = default;
  void Rebuild(MeshBlockPack *pack, const Z4cGridLayout &layout);
  void SynchronizeSharedNodes(
      DvceArray5D<Real> &state,
      const char *diagnostic_environment =
          "ATHENA_Z4C_VC_SYNC_DIAGNOSTIC") const;

  DualArray4D<vertex_topology::VertexTopologyRecord> records;
  std::uint64_t generation = 0;
  std::uint64_t active_records = 0;
  std::uint64_t shared_records = 0;
  std::uint64_t hanging_records = 0;
  std::vector<VertexContributor> local_contributors;
  std::vector<VertexContributor> global_contributors;
  std::vector<int> contributor_counts;
  std::vector<int> contributor_displacements;
  std::vector<int> sorted_global_indices;
  std::vector<int> global_group_for_contributor;
  std::vector<int> local_group;
  DualArray2D<int> local_indices;
  mutable std::uint64_t synchronization_calls = 0;
};

}  // namespace z4c

#endif  // Z4C_Z4C_VERTEX_TOPOLOGY_HPP_
