//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c_vertex_topology.cpp
//! \brief Build canonical native-VC identity and role records from Mesh topology.

#include "z4c/z4c_vertex_topology.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <tuple>

#include "bvals/bvals.hpp"
#include "globals.hpp"
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
  // AthenaK subdivides faces with (f1,f2), edges with f1 only, and corners
  // not at all.  Passing f2=1 for an edge numerically aliases its buffer ID
  // into the corner range and can misclassify a same-level shared vertex as a
  // hanging coarse/fine node.
  const int codimension = std::abs(ox1) + std::abs(ox2) + std::abs(ox3);
  const int f1_count = codimension <= 2 ? 2 : 1;
  const int f2_count = codimension == 1 ? 2 : 1;
  for (int f2 = 0; f2 < f2_count; ++f2) {
    for (int f1 = 0; f1 < f1_count; ++f1) {
      indices->insert(NeighborIndex(ox1, ox2, ox3, f1, f2));
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

bool VertexContributorLess(const VertexContributor &left,
                           const VertexContributor &right) {
  return std::tie(left.key1, left.key2, left.key3, left.level, left.lx1,
                  left.lx2, left.lx3, left.gid, left.k, left.j, left.i) <
         std::tie(right.key1, right.key2, right.key3, right.level, right.lx1,
                  right.lx2, right.lx3, right.gid, right.k, right.j, right.i);
}

void Z4cVertexTopologyPlan::ConfigureRuntime(
    const bool use_single_rank_device_sync,
    const bool use_synchronization_postcondition,
    const int requested_maximum_variables) {
  if (requested_maximum_variables <= 0) {
    std::cerr << "### FATAL ERROR: native VC synchronization requires a positive "
                 "maximum variable count"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  single_rank_device_sync = use_single_rank_device_sync;
  synchronization_postcondition = use_synchronization_postcondition;
  maximum_variables = requested_maximum_variables;
  device_local_group = DualArray1D<int>("VC shared local groups", 1);
  device_authority_contributors =
      DualArray1D<int>("VC shared authority contributors", 1);
  device_authority_begin =
      DualArray1D<int>("VC shared authority begin", 1);
  device_authority_end = DualArray1D<int>("VC shared authority end", 1);
  device_group_values = DvceArray2D<Real>(
      "VC shared group values", 1, maximum_variables);
}

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
          int minimum_same_level_gid = gid;
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
                  if (neighbor.lev == location.level) {
                    same_level = true;
                    minimum_same_level_gid =
                        std::min(minimum_same_level_gid, neighbor.gid);
                  }
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
          // Axis, physical-boundary, and hanging roles intentionally do not enter
          // the evolution gather.  They still need exactly one deterministic
          // same-level diagnostic copy.  Shared coincident roles are refined by
          // the complete global contributor order below.
          record.canonical_diagnostic_owner =
              (record.role == vertex_topology::VertexNodeRole::independent_interior ||
               record.role == vertex_topology::VertexNodeRole::axis ||
               record.role == vertex_topology::VertexNodeRole::physical_boundary ||
               record.role ==
                   vertex_topology::VertexNodeRole::hanging_fine_interface) &&
              gid == minimum_same_level_gid;
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

  local_contributors.clear();
  for (int m = 0; m < nmb; ++m) {
    const int gid = blocks->mb_gid.h_view(m);
    const LogicalLocation location = mesh->lloc_eachmb[gid];
    for (int k = layout.ks; k <= layout.ke; ++k) {
      for (int j = layout.js; j <= layout.je; ++j) {
        for (int i = layout.is; i <= layout.ie; ++i) {
          auto &record = host(m, k, j, i);
          if (record.topological_multiplicity <= 1 ||
              record.role ==
                  vertex_topology::VertexNodeRole::hanging_fine_interface) {
            continue;
          }
          local_contributors.push_back(
              {record.key.i1, record.key.i2, record.key.i3, location.level,
               location.lx1, location.lx2, location.lx3, gid, m, k, j, i});
        }
      }
    }
  }
  std::sort(local_contributors.begin(), local_contributors.end(),
            VertexContributorLess);
  contributor_counts.assign(global_variable::nranks, 0);
  const int local_count = static_cast<int>(local_contributors.size());
#if MPI_PARALLEL_ENABLED
  MPI_Allgather(&local_count, 1, MPI_INT, contributor_counts.data(), 1, MPI_INT,
                MPI_COMM_WORLD);
#else
  contributor_counts[0] = local_count;
#endif
  contributor_displacements.assign(global_variable::nranks, 0);
  for (int rank = 1; rank < global_variable::nranks; ++rank) {
    contributor_displacements[rank] = contributor_displacements[rank - 1] +
                                      contributor_counts[rank - 1];
  }
  const int global_count =
      contributor_displacements.back() + contributor_counts.back();
  global_contributors.resize(global_count);
#if MPI_PARALLEL_ENABLED
  std::vector<int> byte_counts(global_variable::nranks);
  std::vector<int> byte_displacements(global_variable::nranks);
  for (int rank = 0; rank < global_variable::nranks; ++rank) {
    byte_counts[rank] = contributor_counts[rank] * sizeof(VertexContributor);
    byte_displacements[rank] =
        contributor_displacements[rank] * sizeof(VertexContributor);
  }
  MPI_Allgatherv(local_contributors.data(),
                 local_count * sizeof(VertexContributor), MPI_BYTE,
                 global_contributors.data(), byte_counts.data(),
                 byte_displacements.data(), MPI_BYTE, MPI_COMM_WORLD);
#else
  global_contributors = local_contributors;
#endif
  sorted_global_indices.resize(global_count);
  std::iota(sorted_global_indices.begin(), sorted_global_indices.end(), 0);
  std::sort(sorted_global_indices.begin(), sorted_global_indices.end(),
      [this](const int left, const int right) {
        return VertexContributorLess(global_contributors[left],
                                     global_contributors[right]);
      });
  global_group_for_contributor.assign(global_count, -1);
  int group = -1;
  VertexContributor prior;
  bool have_prior = false;
  for (const int global_index : sorted_global_indices) {
    const auto &current = global_contributors[global_index];
    if (!have_prior || current.key1 != prior.key1 || current.key2 != prior.key2 ||
        current.key3 != prior.key3) {
      ++group;
      prior = current;
      have_prior = true;
    }
    global_group_for_contributor[global_index] = group;
  }
  group_count = group + 1;
  local_group.resize(local_count);
  const int my_displacement =
      contributor_displacements[global_variable::my_rank];
  Kokkos::realloc(local_indices, local_count, 4);
  for (int local = 0; local < local_count; ++local) {
    const int global_index = my_displacement + local;
    local_group[local] = global_group_for_contributor[global_index];
    const auto &contributor = local_contributors[local];
    local_indices.h_view(local, 0) = contributor.m;
    local_indices.h_view(local, 1) = contributor.k;
    local_indices.h_view(local, 2) = contributor.j;
    local_indices.h_view(local, 3) = contributor.i;
  }
  local_indices.template modify<HostMemSpace>();
  local_indices.template sync<DevExeSpace>();

  Kokkos::realloc(device_local_group, local_count);
  for (int local = 0; local < local_count; ++local) {
    device_local_group.h_view(local) = local_group[local];
  }
  device_local_group.template modify<HostMemSpace>();
  device_local_group.template sync<DevExeSpace>();

  Kokkos::realloc(device_authority_begin, group_count);
  Kokkos::realloc(device_authority_end, group_count);
  Kokkos::realloc(device_group_values, group_count, maximum_variables);
  std::vector<int> authority_level(group_count,
                                   std::numeric_limits<int>::min());
  for (const int global_index : sorted_global_indices) {
    const int owner_group = global_group_for_contributor[global_index];
    authority_level[owner_group] = std::max(
        authority_level[owner_group], global_contributors[global_index].level);
  }
  std::vector<int> authority_begin(group_count, 0);
  std::vector<int> authority_end(group_count, 0);
  std::vector<int> authority_contributors;
  authority_contributors.reserve(global_count);
  for (int owner_group = 0; owner_group < group_count; ++owner_group) {
    authority_begin[owner_group] =
        static_cast<int>(authority_contributors.size());
    for (const int global_index : sorted_global_indices) {
      if (global_group_for_contributor[global_index] == owner_group &&
          global_contributors[global_index].level ==
              authority_level[owner_group]) {
        authority_contributors.push_back(global_index);
      }
    }
    authority_end[owner_group] =
        static_cast<int>(authority_contributors.size());
    if (authority_begin[owner_group] == authority_end[owner_group]) {
      std::cerr << "### FATAL ERROR: VC shared group has no finest-level authority"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    device_authority_begin.h_view(owner_group) = authority_begin[owner_group];
    device_authority_end.h_view(owner_group) = authority_end[owner_group];
  }
  Kokkos::realloc(device_authority_contributors,
                  authority_contributors.size());
  for (std::size_t contributor = 0;
       contributor < authority_contributors.size(); ++contributor) {
    device_authority_contributors.h_view(contributor) =
        authority_contributors[contributor];
  }
  device_authority_contributors.template modify<HostMemSpace>();
  device_authority_contributors.template sync<DevExeSpace>();
  device_authority_begin.template modify<HostMemSpace>();
  device_authority_begin.template sync<DevExeSpace>();
  device_authority_end.template modify<HostMemSpace>();
  device_authority_end.template sync<DevExeSpace>();
  // Mark exactly the first contributor in the canonical global order as diagnostic
  // owner.  Evolution synchronization still averages every contributor.
  for (int sorted = 0; sorted < global_count; ++sorted) {
    const int global_index = sorted_global_indices[sorted];
    const int owner_group = global_group_for_contributor[global_index];
    if (sorted > 0 &&
        global_group_for_contributor[sorted_global_indices[sorted - 1]] ==
            owner_group) {
      continue;
    }
    if (global_index < my_displacement ||
        global_index >= my_displacement + local_count) {
      continue;
    }
    const auto &owner = local_contributors[global_index - my_displacement];
    host(owner.m, owner.k, owner.j, owner.i).canonical_diagnostic_owner = 1;
  }
  records.template modify<HostMemSpace>();
  records.template sync<DevExeSpace>();
  ++generation;
}

void Z4cVertexTopologyPlan::SynchronizeSharedNodes(
    DvceArray5D<Real> &state, const char *diagnostic_environment) const {
  ++synchronization_calls;
  const int local_count = static_cast<int>(local_contributors.size());
  const int global_count = static_cast<int>(global_contributors.size());
  const int nvar = state.extent_int(1);
  if (global_count == 0 || nvar == 0) return;
  const char *diagnostic_path = std::getenv(diagnostic_environment);
  const bool diagnostic_requested =
      diagnostic_path != nullptr && diagnostic_path[0] != '\0';
  if (single_rank_device_sync && global_variable::nranks == 1 &&
      !diagnostic_requested) {
    if (nvar > maximum_variables) {
      std::cerr << "### FATAL ERROR: native VC synchronization received " << nvar
                << " variables, exceeding configured capacity "
                << maximum_variables << std::endl;
      std::exit(EXIT_FAILURE);
    }
    const auto indices = local_indices.d_view;
    const auto local_groups = device_local_group.d_view;
    const auto authority_contributors =
        device_authority_contributors.d_view;
    const auto authority_begin = device_authority_begin.d_view;
    const auto authority_end = device_authority_end.d_view;
    auto group_values = device_group_values;
    par_for("compute deterministic VC shared averages on device",
            DevExeSpace(), 0, group_count - 1, 0, nvar - 1,
        KOKKOS_LAMBDA(const int group, const int variable) {
          Real sum = 0.0;
          for (int entry = authority_begin(group);
               entry < authority_end(group); ++entry) {
            const int contributor = authority_contributors(entry);
            const Real value = state(indices(contributor, 0), variable,
                                     indices(contributor, 1),
                                     indices(contributor, 2),
                                     indices(contributor, 3));
            if (!Kokkos::isfinite(value)) {
              Kokkos::abort("nonfinite VC shared contributor on device");
            }
            sum += value;
          }
          group_values(group, variable) =
              sum / static_cast<Real>(authority_end(group) -
                                      authority_begin(group));
        });
    par_for("apply deterministic VC shared averages on device",
            DevExeSpace(), 0, local_count - 1, 0, nvar - 1,
        KOKKOS_LAMBDA(const int contributor, const int variable) {
          state(indices(contributor, 0), variable, indices(contributor, 1),
                indices(contributor, 2), indices(contributor, 3)) =
              group_values(local_groups(contributor), variable);
        });
    if (!synchronization_postcondition) return;

    DvceArray1D<unsigned long long> mismatches(
        "VC shared device sync mismatches", 1);
    Kokkos::deep_copy(mismatches, 0ULL);
    par_for("verify deterministic VC shared averages on device",
            DevExeSpace(), 0, local_count - 1, 0, nvar - 1,
        KOKKOS_LAMBDA(const int contributor, const int variable) {
          const Real expected =
              group_values(local_groups(contributor), variable);
          const Real actual =
              state(indices(contributor, 0), variable, indices(contributor, 1),
                    indices(contributor, 2), indices(contributor, 3));
          if (actual != expected) Kokkos::atomic_inc(&mismatches(0));
        });
    Kokkos::fence();
    const auto host_mismatches =
        Kokkos::create_mirror_view_and_copy(HostMemSpace(), mismatches);
    if (host_mismatches(0) != 0) {
      std::cerr << "### FATAL ERROR: one-rank device VC synchronization left "
                << host_mismatches(0) << " mismatched contributor values"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    return;
  }
  DvceArray2D<Real> packed("VC shared contributors", local_count, nvar);
  const auto indices = local_indices.d_view;
  if (local_count > 0) {
    par_for("pack VC shared contributors", DevExeSpace(), 0, local_count - 1,
            0, nvar - 1,
        KOKKOS_LAMBDA(const int contributor, const int variable) {
          packed(contributor, variable) =
              state(indices(contributor, 0), variable, indices(contributor, 1),
                    indices(contributor, 2), indices(contributor, 3));
        });
  }
  Kokkos::fence();
  const auto host_packed =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), packed);
  std::vector<Real> local_values(local_count * nvar);
  for (int contributor = 0; contributor < local_count; ++contributor) {
    for (int variable = 0; variable < nvar; ++variable) {
      local_values[contributor * nvar + variable] = host_packed(contributor, variable);
    }
  }
  std::vector<Real> global_values(global_count * nvar);
#if MPI_PARALLEL_ENABLED
  std::vector<int> value_counts(global_variable::nranks);
  std::vector<int> value_displacements(global_variable::nranks);
  for (int rank = 0; rank < global_variable::nranks; ++rank) {
    value_counts[rank] = contributor_counts[rank] * nvar;
    value_displacements[rank] = contributor_displacements[rank] * nvar;
  }
  MPI_Allgatherv(local_values.data(), local_count * nvar, MPI_ATHENA_REAL,
                 global_values.data(), value_counts.data(),
                 value_displacements.data(), MPI_ATHENA_REAL, MPI_COMM_WORLD);
#else
  global_values = local_values;
#endif
  const int groups = global_group_for_contributor.empty()
                         ? 0
                         : 1 + *std::max_element(global_group_for_contributor.begin(),
                                                 global_group_for_contributor.end());
  std::vector<Real> averages(groups * nvar, 0.0);
  std::vector<int> multiplicity(groups, 0);
  std::vector<int> authority_multiplicity(groups, 0);
  std::vector<int> authority_level(groups, std::numeric_limits<int>::min());
  for (const int global_index : sorted_global_indices) {
    const int group = global_group_for_contributor[global_index];
    ++multiplicity[group];
    authority_level[group] =
        std::max(authority_level[group], global_contributors[global_index].level);
  }
  for (const int global_index : sorted_global_indices) {
    const int group = global_group_for_contributor[global_index];
    const bool authoritative = VertexContributorHasAuthority(
        global_contributors[global_index].level, authority_level[group]);
    if (authoritative) ++authority_multiplicity[group];
    for (int variable = 0; variable < nvar; ++variable) {
      const Real value = global_values[global_index * nvar + variable];
      if (!std::isfinite(value)) {
        std::cerr << "### FATAL ERROR: nonfinite VC shared contributor key=("
                  << global_contributors[global_index].key1 << ','
                  << global_contributors[global_index].key2 << ','
                  << global_contributors[global_index].key3 << ") variable="
                  << variable << std::endl;
#if MPI_PARALLEL_ENABLED
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#else
        std::exit(EXIT_FAILURE);
#endif
      }
      if (authoritative) averages[group * nvar + variable] += value;
    }
  }
  for (int group = 0; group < groups; ++group) {
    if (authority_multiplicity[group] <= 0) {
      std::cerr << "### FATAL ERROR: VC shared group has no finest-level authority"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    for (int variable = 0; variable < nvar; ++variable) {
      averages[group * nvar + variable] /=
          static_cast<Real>(authority_multiplicity[group]);
    }
  }

  // Default-off localization evidence for the native-VC AMR discriminator.  The
  // contributor values have already been gathered in deterministic canonical order,
  // so this diagnostic observes (but does not alter) the exact inputs to the shared-node
  // reconciliation.  A caller-provided path keeps ordinary runs and restart state free
  // of new parameters or side effects.
  if (diagnostic_requested && global_variable::my_rank == 0) {
    std::vector<Real> minima(groups * nvar,
                             std::numeric_limits<Real>::max());
    std::vector<Real> maxima(groups * nvar,
                             -std::numeric_limits<Real>::max());
    std::vector<int> minimum_levels(groups,
                                    std::numeric_limits<int>::max());
    std::vector<int> maximum_levels(groups,
                                    std::numeric_limits<int>::min());
    for (const int global_index : sorted_global_indices) {
      const int group = global_group_for_contributor[global_index];
      minimum_levels[group] =
          std::min(minimum_levels[group], global_contributors[global_index].level);
      maximum_levels[group] =
          std::max(maximum_levels[group], global_contributors[global_index].level);
      for (int variable = 0; variable < nvar; ++variable) {
        const Real value = global_values[global_index * nvar + variable];
        minima[group * nvar + variable] =
            std::min(minima[group * nvar + variable], value);
        maxima[group * nvar + variable] =
            std::max(maxima[group * nvar + variable], value);
      }
    }
    std::ifstream prior_output(diagnostic_path);
    const bool output_exists = prior_output.good();
    prior_output.close();
    std::ofstream output(diagnostic_path, std::ios::app);
    if (!output) {
      std::cerr << "### FATAL ERROR: cannot open native VC synchronization "
                   "diagnostic path "
                << diagnostic_path << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (!output_exists) {
      output << "call,generation,group,key1,key2,key3,variable,"
                "minimum_level,maximum_level,multiplicity,authority_level,"
                "authority_multiplicity,minimum,maximum,"
                "spread,mean,contributor_level,contributor_lx1,"
                "contributor_lx2,contributor_lx3,contributor_gid,"
                "contributor_k,contributor_j,contributor_i,value\n";
    }
    output << std::setprecision(std::numeric_limits<Real>::max_digits10);
    const bool include_same_level =
        std::getenv("ATHENA_Z4C_VC_SYNC_DIAGNOSTIC_INCLUDE_SAME_LEVEL") != nullptr;
    for (const int global_index : sorted_global_indices) {
      const int group = global_group_for_contributor[global_index];
      if (!include_same_level && minimum_levels[group] == maximum_levels[group]) {
        continue;
      }
      const auto &contributor = global_contributors[global_index];
      for (int variable = 0; variable < nvar; ++variable) {
        const int offset = group * nvar + variable;
        output << synchronization_calls << ',' << generation << ',' << group
               << ',' << contributor.key1 << ',' << contributor.key2 << ','
               << contributor.key3 << ',' << variable << ','
               << minimum_levels[group] << ',' << maximum_levels[group] << ','
               << multiplicity[group] << ',' << authority_level[group] << ','
               << authority_multiplicity[group] << ',' << minima[offset] << ','
               << maxima[offset] << ',' << maxima[offset] - minima[offset]
               << ',' << averages[offset] << ',' << contributor.level << ','
               << contributor.lx1 << ',' << contributor.lx2 << ','
               << contributor.lx3 << ',' << contributor.gid << ','
               << contributor.k << ',' << contributor.j << ','
               << contributor.i << ','
               << global_values[global_index * nvar + variable] << '\n';
      }
    }
    if (!output) {
      std::cerr << "### FATAL ERROR: failed while writing native VC "
                   "synchronization diagnostic"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  auto host_replacements = Kokkos::create_mirror_view(packed);
  for (int contributor = 0; contributor < local_count; ++contributor) {
    const int group = local_group[contributor];
    for (int variable = 0; variable < nvar; ++variable) {
      host_replacements(contributor, variable) = averages[group * nvar + variable];
    }
  }
  Kokkos::deep_copy(packed, host_replacements);
  if (local_count > 0) {
    par_for("apply deterministic VC shared averages", DevExeSpace(),
            0, local_count - 1, 0, nvar - 1,
        KOKKOS_LAMBDA(const int contributor, const int variable) {
          state(indices(contributor, 0), variable, indices(contributor, 1),
                indices(contributor, 2), indices(contributor, 3)) =
              packed(contributor, variable);
        });
  }
  // The lean contract already made this postcondition optional on the
  // single-rank device path.  Honor the same option here: all contributors
  // have just been assigned from the same canonical replacement buffer, and
  // the following exact check is observational.  Keep it enabled for the
  // default exhaustive path and whenever localization evidence is requested.
  if (!synchronization_postcondition && !diagnostic_requested) return;
  Kokkos::fence();

  // This is intentionally an exact postcondition, not a tolerance check: every
  // contributor is assigned the identical canonical value above.  A mismatch
  // therefore diagnoses an indexing/write defect rather than floating-point
  // reduction order.
  DvceArray1D<unsigned long long> mismatches("VC shared sync mismatches", 1);
  Kokkos::deep_copy(mismatches, 0ULL);
  if (local_count > 0) {
    par_for("verify deterministic VC shared averages", DevExeSpace(),
            0, local_count - 1, 0, nvar - 1,
        KOKKOS_LAMBDA(const int contributor, const int variable) {
          const Real actual =
              state(indices(contributor, 0), variable, indices(contributor, 1),
                    indices(contributor, 2), indices(contributor, 3));
          if (actual != packed(contributor, variable)) {
            Kokkos::atomic_inc(&mismatches(0));
          }
        });
  }
  Kokkos::fence();
  const auto host_mismatches =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), mismatches);
  unsigned long long global_mismatches = host_mismatches(0);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &global_mismatches, 1, MPI_UNSIGNED_LONG_LONG,
                MPI_SUM, MPI_COMM_WORLD);
#endif
  if (global_mismatches != 0) {
    std::cerr << "### FATAL ERROR: canonical VC shared-node synchronization left "
              << global_mismatches << " mismatched contributor values" << std::endl;
#if MPI_PARALLEL_ENABLED
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#else
    std::exit(EXIT_FAILURE);
#endif
  }
}

}  // namespace z4c
