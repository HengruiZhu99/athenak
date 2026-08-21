//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c_vertex_topology.cpp
//! \brief Build canonical native-VC identity and role records from Mesh topology.

#include "z4c/z4c_vertex_topology.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <set>

#include "bvals/bvals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "mesh/nghbr_index.hpp"

namespace z4c {
namespace {

bool IsNonperiodicPhysicalBoundary(const BoundaryFlag flag) {
  return flag != BoundaryFlag::block && flag != BoundaryFlag::periodic &&
         flag != BoundaryFlag::shear_periodic && flag != BoundaryFlag::axis;
}

void AddNeighborIndices(const int ox1, const int ox2, const int ox3,
                        std::set<int> *indices) {
  for (int f2 = 0; f2 <= 1; ++f2) {
    for (int f1 = 0; f1 <= 1; ++f1) {
      const int index = NeighborIndex(ox1, ox2, ox3, f1, f2);
      if (index >= 0) indices->insert(index);
    }
  }
}

vertex_topology::CanonicalVertexDirectionConfig DirectionConfig(
    const Mesh *mesh, const int direction, const int intervals,
    const bool collapsed) {
  const BoundaryFace inner = static_cast<BoundaryFace>(2 * direction);
  const BoundaryFace outer = static_cast<BoundaryFace>(2 * direction + 1);
  const bool periodic =
      mesh->mesh_bcs[inner] == BoundaryFlag::periodic &&
      mesh->mesh_bcs[outer] == BoundaryFlag::periodic;
  const int root_blocks = direction == 0 ? mesh->nmb_rootx1
                          : (direction == 1 ? mesh->nmb_rootx2
                                            : mesh->nmb_rootx3);
  return {mesh->root_level, mesh->max_level, root_blocks, intervals,
          collapsed, periodic};
}

}  // namespace

void Z4cVertexTopologyPlan::Rebuild(MeshBlockPack *pack,
                                    const Z4cGridLayout &layout) {
  if (layout.centering != Z4cGridCentering::vertex) {
    std::cerr << "### FATAL ERROR: VC topology plan requested for non-VC layout"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  Mesh *mesh = pack->pmesh;
  MeshBlock *blocks = pack->pmb;
  blocks->mb_gid.sync_host();
  blocks->mb_lev.sync_host();
  blocks->mb_bcs.sync_host();
  blocks->nghbr.sync_host();

  const int nmb = pack->nmb_thispack;
  Kokkos::realloc(records, nmb, layout.n3, layout.n2, layout.n1);
  auto host = records.h_view;
  const auto x1 = DirectionConfig(mesh, 0, layout.nx1, layout.nx1 <= 1);
  const auto x2 = DirectionConfig(mesh, 1, layout.nx2, layout.nx2 <= 1);
  const auto x3 = DirectionConfig(mesh, 2, layout.nx3, layout.nx3 <= 1);
  std::uint64_t local_active = 0;
  std::uint64_t local_shared = 0;
  std::uint64_t local_hanging = 0;

  for (int m = 0; m < nmb; ++m) {
    const int gid = blocks->mb_gid.h_view(m);
    const LogicalLocation location = mesh->lloc_eachmb[gid];
    for (int k = 0; k < layout.n3; ++k) {
      for (int j = 0; j < layout.n2; ++j) {
        for (int i = 0; i < layout.n1; ++i) {
          auto &record = host(m, k, j, i);
          record = {};
          const bool ghost = i < layout.is || i > layout.ie ||
                             j < layout.js || j > layout.je ||
                             k < layout.ks || k > layout.ke;
          if (ghost) {
            record.role = vertex_topology::VertexNodeRole::ghost;
            continue;
          }
          ++local_active;
          const int oi = i - layout.is;
          const int oj = j - layout.js;
          const int ok = k - layout.ks;
          record.key = vertex_topology::MakeCanonicalVertexKey(
              x1, x2, x3, location.level, location.lx1, location.lx2,
              location.lx3, oi, oj, ok);
          if (!record.key.valid) {
            std::cerr << "### FATAL ERROR: canonical VC key overflow/invalid input for gid "
                      << gid << " local_vertex=(" << oi << ',' << oj << ',' << ok
                      << ") level=" << location.level << std::endl;
            std::exit(EXIT_FAILURE);
          }

          const bool lower1 = layout.nx1 > 1 && i == layout.is;
          const bool upper1 = layout.nx1 > 1 && i == layout.ie;
          const bool lower2 = layout.nx2 > 1 && j == layout.js;
          const bool upper2 = layout.nx2 > 1 && j == layout.je;
          const bool lower3 = layout.nx3 > 1 && k == layout.ks;
          const bool upper3 = layout.nx3 > 1 && k == layout.ke;
          const bool axis = lower1 &&
              blocks->mb_bcs.h_view(m, BoundaryFace::inner_x1) == BoundaryFlag::axis;
          const bool physical =
              (lower1 && IsNonperiodicPhysicalBoundary(
                             blocks->mb_bcs.h_view(m, BoundaryFace::inner_x1))) ||
              (upper1 && IsNonperiodicPhysicalBoundary(
                             blocks->mb_bcs.h_view(m, BoundaryFace::outer_x1))) ||
              (lower2 && IsNonperiodicPhysicalBoundary(
                             blocks->mb_bcs.h_view(m, BoundaryFace::inner_x2))) ||
              (upper2 && IsNonperiodicPhysicalBoundary(
                             blocks->mb_bcs.h_view(m, BoundaryFace::outer_x2))) ||
              (lower3 && IsNonperiodicPhysicalBoundary(
                             blocks->mb_bcs.h_view(m, BoundaryFace::inner_x3))) ||
              (upper3 && IsNonperiodicPhysicalBoundary(
                             blocks->mb_bcs.h_view(m, BoundaryFace::outer_x3)));

          int choices1[2] = {0, 0};
          int choices2[2] = {0, 0};
          int choices3[2] = {0, 0};
          int count1 = 1, count2 = 1, count3 = 1;
          if (lower1 || upper1) choices1[count1++] = lower1 ? -1 : 1;
          if (lower2 || upper2) choices2[count2++] = lower2 ? -1 : 1;
          if (lower3 || upper3) choices3[count3++] = lower3 ? -1 : 1;
          bool same_level = false;
          bool coarse_fine = false;
          bool fine_side = false;
          std::set<int> neighbor_gids;
          for (int a = 0; a < count1; ++a) {
            for (int b = 0; b < count2; ++b) {
              for (int c = 0; c < count3; ++c) {
                if (choices1[a] == 0 && choices2[b] == 0 && choices3[c] == 0) {
                  continue;
                }
                std::set<int> indices;
                AddNeighborIndices(choices1[a], choices2[b], choices3[c], &indices);
                for (const int index : indices) {
                  if (index >= blocks->nnghbr) continue;
                  const NeighborBlock &neighbor = blocks->nghbr.h_view(m, index);
                  if (neighbor.gid < 0) continue;
                  neighbor_gids.insert(neighbor.gid);
                  if (neighbor.lev == location.level) same_level = true;
                  if (neighbor.lev != location.level) coarse_fine = true;
                  if (neighbor.lev < location.level) fine_side = true;
                }
              }
            }
          }
          const std::int64_t level_i1 =
              static_cast<std::int64_t>(location.lx1) * layout.nx1 + oi;
          const std::int64_t level_i2 =
              static_cast<std::int64_t>(location.lx2) * layout.nx2 + oj;
          const std::int64_t level_i3 =
              static_cast<std::int64_t>(location.lx3) * layout.nx3 + ok;
          const bool coincident = vertex_topology::FineVertexCoincidentWithCoarse(
              level_i1, level_i2, level_i3, layout.nx1 <= 1,
              layout.nx2 <= 1, layout.nx3 <= 1);
          const vertex_topology::VertexRoleInput role_input{
              false, axis, physical, same_level, coarse_fine, fine_side, coincident};
          record.role = vertex_topology::ClassifyVertexNodeRole(role_input);
          record.topological_multiplicity = static_cast<std::uint8_t>(
              std::min<std::size_t>(255, neighbor_gids.size() + 1));
          // The deterministic gather plan will refine this ownership bit using the
          // complete contributor order.  Interior nodes are unambiguous already.
          record.canonical_diagnostic_owner =
              record.role == vertex_topology::VertexNodeRole::independent_interior;
          if (record.role == vertex_topology::VertexNodeRole::shared_same_level ||
              record.role ==
                  vertex_topology::VertexNodeRole::shared_coarse_fine_coincident) {
            ++local_shared;
          }
          if (record.role ==
              vertex_topology::VertexNodeRole::hanging_fine_interface) {
            ++local_hanging;
          }
        }
      }
    }
  }
  records.template modify<HostMemSpace>();
  records.template sync<DevExeSpace>();
  active_records = local_active;
  shared_records = local_shared;
  hanging_records = local_hanging;
  ++generation;
}

}  // namespace z4c
