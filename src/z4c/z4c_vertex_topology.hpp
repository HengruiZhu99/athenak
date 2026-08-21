//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c_vertex_topology.hpp
//! \brief Production topology-derived role map for native VC Z4c storage.

#ifndef Z4C_Z4C_VERTEX_TOPOLOGY_HPP_
#define Z4C_Z4C_VERTEX_TOPOLOGY_HPP_

#include "athena.hpp"
#include "bvals/vertex_topology.hpp"
#include "z4c/z4c_grid.hpp"

class MeshBlockPack;

namespace z4c {

class Z4cVertexTopologyPlan {
 public:
  Z4cVertexTopologyPlan() = default;
  void Rebuild(MeshBlockPack *pack, const Z4cGridLayout &layout);

  DualArray4D<vertex_topology::VertexTopologyRecord> records;
  std::uint64_t generation = 0;
  std::uint64_t active_records = 0;
  std::uint64_t shared_records = 0;
  std::uint64_t hanging_records = 0;
};

}  // namespace z4c

#endif  // Z4C_Z4C_VERTEX_TOPOLOGY_HPP_
