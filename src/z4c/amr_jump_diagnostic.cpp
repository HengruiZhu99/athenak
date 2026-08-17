//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file amr_jump_diagnostic.cpp
//! \brief Runtime capture for the bounded Cartoon Z4c AMR-jump diagnosis.

#include "z4c/amr_jump_diagnostic.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include <Kokkos_Core.hpp>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "driver/driver.hpp"
#include "globals.hpp"
#include "bvals/bvals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/nghbr_index.hpp"
#include "mesh/restriction.hpp"
#include "z4c/stored_domain_bounds.hpp"
#include "z4c/cartoon_meridional_sampler.hpp"
#include "z4c/z4c.hpp"

namespace z4c {
namespace {

namespace fs = std::filesystem;

[[noreturn]] void DiagnosticFailure(const std::string &message) {
  throw std::runtime_error("Z4c AMR jump diagnostic: " + message);
}

std::string RankTag() {
  std::ostringstream stream;
  stream << "rank" << std::setw(4) << std::setfill('0')
         << global_variable::my_rank;
  return stream.str();
}

std::string CycleTag(const int cycle) {
  std::ostringstream stream;
  stream << "c" << std::setw(8) << std::setfill('0') << cycle;
  return stream.str();
}

void CreateDirectory(const fs::path &path) {
  std::error_code error;
  fs::create_directories(path, error);
  if (error) {
    DiagnosticFailure("cannot create directory '" + path.string() + "': " +
                      error.message());
  }
}

void WriteTextAtomically(const fs::path &path, const std::string &contents) {
  CreateDirectory(path.parent_path());
  const fs::path temporary = path.string() + ".tmp";
  {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) DiagnosticFailure("cannot open '" + temporary.string() + "'");
    output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    if (!output) DiagnosticFailure("cannot write '" + temporary.string() + "'");
  }
  std::error_code error;
  fs::rename(temporary, path, error);
  if (error) {
    DiagnosticFailure("cannot promote '" + path.string() + "': " + error.message());
  }
}

void AppendText(const fs::path &path, const std::string &contents) {
  CreateDirectory(path.parent_path());
  std::ofstream output(path, std::ios::binary | std::ios::app);
  if (!output) DiagnosticFailure("cannot append '" + path.string() + "'");
  output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  if (!output) DiagnosticFailure("cannot append bytes to '" + path.string() + "'");
}

template <typename View>
void WriteViewBinary(const fs::path &path, const View &view, const int nmb) {
  if (nmb < 0 || nmb > view.extent_int(0)) {
    DiagnosticFailure("invalid local MeshBlock count for '" + path.string() + "'");
  }
  const auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), view);
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) DiagnosticFailure("cannot open binary output '" + path.string() + "'");
  for (int m = 0; m < nmb; ++m) {
    for (int n = 0; n < view.extent_int(1); ++n) {
      for (int k = 0; k < view.extent_int(2); ++k) {
        for (int j = 0; j < view.extent_int(3); ++j) {
          for (int i = 0; i < view.extent_int(4); ++i) {
            const Real value = host(m, n, k, j, i);
            output.write(reinterpret_cast<const char *>(&value), sizeof(Real));
          }
        }
      }
    }
  }
  if (!output) DiagnosticFailure("cannot write binary output '" + path.string() + "'");
}

template <typename View>
std::string ViewShapeJSON(const View &view, const int nmb) {
  std::ostringstream stream;
  stream << "[" << nmb << "," << view.extent_int(1) << ","
         << view.extent_int(2) << "," << view.extent_int(3) << ","
         << view.extent_int(4) << "]";
  return stream.str();
}

int MaximumLevel(const LogicalLocation *locations, const int count) {
  int level = std::numeric_limits<int>::min();
  for (int index = 0; index < count; ++index) {
    level = std::max(level, static_cast<int>(locations[index].level));
  }
  return count == 0 ? -1 : level;
}

struct CoarseFineFaceInventory {
  std::uint64_t incidents = 0;
  std::string locations;
};

CoarseFineFaceInventory LocalCoarseFineFaceInventory(MeshBlockPack *pack) {
  CoarseFineFaceInventory inventory;
  if (pack == nullptr || pack->pmesh == nullptr || !pack->pmesh->two_d) {
    return inventory;
  }
  auto &neighbors = pack->pmb->nghbr.h_view;
  auto &levels = pack->pmb->mb_lev.h_view;
  auto &gids = pack->pmb->mb_gid.h_view;
  constexpr std::array<const char *, 4> names = {
      "inner_x1", "outer_x1", "inner_x2", "outer_x2"};
  const std::array<std::array<int, 2>, 4> slots = {{
      {{NeighborIndex(-1, 0, 0, 0, 0), NeighborIndex(-1, 0, 0, 1, 0)}},
      {{NeighborIndex(1, 0, 0, 0, 0), NeighborIndex(1, 0, 0, 1, 0)}},
      {{NeighborIndex(0, -1, 0, 0, 0), NeighborIndex(0, -1, 0, 1, 0)}},
      {{NeighborIndex(0, 1, 0, 0, 0), NeighborIndex(0, 1, 0, 1, 0)}}}};
  std::ostringstream locations;
  bool first = true;
  for (int m = 0; m < pack->nmb_thispack; ++m) {
    for (int face = 0; face < 4; ++face) {
      bool coarse_fine = false;
      for (const int slot : slots[face]) {
        coarse_fine = coarse_fine ||
            (neighbors(m, slot).gid >= 0 && neighbors(m, slot).lev != levels(m));
      }
      if (!coarse_fine) continue;
      ++inventory.incidents;
      if (!first) locations << ";";
      first = false;
      locations << gids(m) << ":" << names[face];
    }
  }
  inventory.locations = locations.str();
  return inventory;
}

struct Aggregate {
  Real coordinate_volume = 0.0;
  Real proper_volume = 0.0;
  Real c_integral = 0.0;
  Real h_integral = 0.0;
  Real m_integral = 0.0;
  Real z_integral = 0.0;
  Real min_chi = std::numeric_limits<Real>::infinity();
  Real min_detg = std::numeric_limits<Real>::infinity();
  Real max_c = -std::numeric_limits<Real>::infinity();
  int min_chi_gid = -1;
  int min_chi_i = -1;
  int min_chi_j = -1;
  Real min_chi_rho = std::numeric_limits<Real>::quiet_NaN();
  Real min_chi_z = std::numeric_limits<Real>::quiet_NaN();
  std::uint64_t active_cells = 0;
  std::uint64_t nonpositive_detg = 0;
  std::uint64_t nonfinite_chi = 0;
  std::uint64_t nonfinite_constraints = 0;
};

Aggregate ComputeAggregate(MeshBlockPack *pack,
                           const DvceArray5D<Real> &z4c_state,
                           const DvceArray5D<Real> &adm_state,
                           const DvceArray5D<Real> &constraints) {
  Aggregate result;
  const auto z4c_host =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), z4c_state);
  const auto adm_host =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), adm_state);
  const auto con_host =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), constraints);
  auto &indcs = pack->pmesh->mb_indcs;
  auto &sizes = pack->pmb->mb_size.h_view;
  auto &gids = pack->pmb->mb_gid.h_view;
  for (int m = 0; m < pack->nmb_thispack; ++m) {
    const RegionSize &size = sizes(m);
    for (int k = indcs.ks; k <= indcs.ke; ++k) {
      for (int j = indcs.js; j <= indcs.je; ++j) {
        const Real z = CellCenterX(j - indcs.js, indcs.nx2, size.x2min, size.x2max);
        for (int i = indcs.is; i <= indcs.ie; ++i) {
          const Real rho = CellCenterX(i - indcs.is, indcs.nx1,
                                       size.x1min, size.x1max);
          const Real gxx = adm_host(m, adm::ADM::I_ADM_GXX, k, j, i);
          const Real gxy = adm_host(m, adm::ADM::I_ADM_GXY, k, j, i);
          const Real gxz = adm_host(m, adm::ADM::I_ADM_GXZ, k, j, i);
          const Real gyy = adm_host(m, adm::ADM::I_ADM_GYY, k, j, i);
          const Real gyz = adm_host(m, adm::ADM::I_ADM_GYZ, k, j, i);
          const Real gzz = adm_host(m, adm::ADM::I_ADM_GZZ, k, j, i);
          const Real detg = adm::SpatialDet(gxx, gxy, gxz, gyy, gyz, gzz);
          const Real coordinate_measure =
              kCartoonTwoPi * rho * size.dx1 * size.dx2;
          const Real proper_measure =
              coordinate_measure * std::sqrt(std::fabs(detg));
          const Real c = con_host(m, Z4c::I_CON_C, k, j, i);
          const Real h = con_host(m, Z4c::I_CON_H, k, j, i);
          const Real momentum = con_host(m, Z4c::I_CON_M, k, j, i);
          const Real z_constraint = con_host(m, Z4c::I_CON_Z, k, j, i);
          if (!std::isfinite(c) || !std::isfinite(h) ||
              !std::isfinite(momentum) || !std::isfinite(z_constraint)) {
            ++result.nonfinite_constraints;
          }
          result.coordinate_volume += coordinate_measure;
          result.proper_volume += proper_measure;
          result.c_integral += c * proper_measure;
          result.h_integral += h * h * proper_measure;
          result.m_integral += momentum * proper_measure;
          result.z_integral += z_constraint * proper_measure;
          result.min_detg = std::min(result.min_detg, detg);
          result.max_c = std::max(result.max_c, c);
          if (!(detg > 0.0) || !std::isfinite(detg)) ++result.nonpositive_detg;
          const Real chi = z4c_host(m, Z4c::I_Z4C_CHI, k, j, i);
          if (!std::isfinite(chi)) ++result.nonfinite_chi;
          if (chi < result.min_chi || !std::isfinite(chi)) {
            result.min_chi = chi;
            result.min_chi_gid = gids(m);
            result.min_chi_i = i;
            result.min_chi_j = j;
            result.min_chi_rho = rho;
            result.min_chi_z = z;
          }
          ++result.active_cells;
        }
      }
    }
  }
  return result;
}

std::string StrictJSONReal(const Real value) {
  if (!std::isfinite(value)) return "null";
  std::ostringstream stream;
  stream << std::setprecision(17) << value;
  return stream.str();
}

std::string AggregateJSON(const Aggregate &aggregate, const Mesh *mesh,
                          const char *schema) {
  std::ostringstream stream;
  stream << std::setprecision(17)
         << "{\"schema\":\"" << schema << "\","
         << "\"rank\":" << global_variable::my_rank << ","
         << "\"cycle\":" << mesh->ncycle << ","
         << "\"time\":" << mesh->time << ","
         << "\"active_cells\":" << aggregate.active_cells << ","
         << "\"coordinate_ring_volume\":"
         << StrictJSONReal(aggregate.coordinate_volume) << ","
         << "\"proper_volume\":" << StrictJSONReal(aggregate.proper_volume) << ","
         << "\"C_norm2\":" << StrictJSONReal(aggregate.c_integral) << ","
         << "\"H_norm2\":" << StrictJSONReal(aggregate.h_integral) << ","
         << "\"M_norm2\":" << StrictJSONReal(aggregate.m_integral) << ","
         << "\"Z_norm2\":" << StrictJSONReal(aggregate.z_integral) << ","
         << "\"min_chi\":" << StrictJSONReal(aggregate.min_chi) << ","
         << "\"nonfinite_chi\":" << aggregate.nonfinite_chi << ","
         << "\"min_chi_gid\":" << aggregate.min_chi_gid << ","
         << "\"min_chi_i\":" << aggregate.min_chi_i << ","
         << "\"min_chi_j\":" << aggregate.min_chi_j << ","
         << "\"min_chi_rho\":" << StrictJSONReal(aggregate.min_chi_rho) << ","
         << "\"min_chi_z\":" << StrictJSONReal(aggregate.min_chi_z) << ","
         << "\"min_det_gamma\":" << StrictJSONReal(aggregate.min_detg) << ","
         << "\"nonpositive_or_nonfinite_det_gamma\":"
         << aggregate.nonpositive_detg << ","
         << "\"nonfinite_constraints\":" << aggregate.nonfinite_constraints << ","
         << "\"max_C_local\":" << StrictJSONReal(aggregate.max_c) << "}";
  return stream.str();
}

struct RestrictionStencil2D {
  int refi = 0;
  int refj = 0;
  int width = 0;
  bool edge_i = false;
  bool edge_j = false;
};

RestrictionStencil2D RestrictionStencilForTarget(const int stencil,
                                                  const int fi,
                                                  const int fj,
                                                  const int nx1,
                                                  const int nx2) {
  RestrictionStencil2D result;
  const bool offset_i = fi < nx1 / 2 + stencil;
  const bool offset_j = fj < nx2 / 2 + stencil;
  if (stencil == 2) {
    result.refi = offset_i ? fi : fi - 1;
    result.refj = offset_j ? fj : fj - 1;
    result.width = 3;
  } else if (stencil == 3) {
    result.refi = fi - 1;
    result.refj = fj - 1;
    result.width = 4;
  } else if (stencil == 4) {
    result.refi = offset_i ? fi - 1 : fi - 2;
    result.refj = offset_j ? fj - 1 : fj - 2;
    const int outer_i = nx1 + 2 * stencil - 2;
    const int outer_j = nx2 + 2 * stencil - 2;
    result.edge_i = (fi == 0 || fi == stencil ||
                     fi == stencil + nx1 - 2 || fi == outer_i);
    result.edge_j = (fj == 0 || fj == stencil ||
                     fj == stencil + nx2 - 2 || fj == outer_j);
    if (fi == stencil) ++result.refi;
    if (fj == stencil) ++result.refj;
    if (fi == stencil + nx1 - 2) --result.refi;
    if (fj == stencil + nx2 - 2) --result.refj;
    if (fi == 0) result.refi = 0;
    if (fj == 0) result.refj = 0;
    if (fi == outer_i) result.refi = nx1 + stencil - 1;
    if (fj == outer_j) result.refj = nx2 + stencil - 1;
    result.width = 5;
  } else {
    DiagnosticFailure("unsupported restriction stencil in shadow diagnostic");
  }
  return result;
}

const char *RuleClass(const RestrictionStencil2D &stencil) {
  if (stencil.edge_i && stencil.edge_j) return "edge-edge";
  if (stencil.edge_i || stencil.edge_j) return "edge-center";
  return "center-center";
}

template <typename HostView>
void AppendChiShadowRecord(std::ostringstream &output, const HostView &fine,
                           const HostView &coarse, const RegionSize &size,
                           const int m, const int gid, const int neighbor_slot,
                           const int writer_ordinal, const int coarse_k,
                           const int coarse_j, const int coarse_i,
                           const int fine_k, const int fine_j, const int fine_i,
                           const int cis, const int cjs, const int nx1,
                           const int nx2, const int stencil) {
  const RestrictionStencil2D source =
      RestrictionStencilForTarget(stencil, fine_i, fine_j, nx1, nx2);
  Real source_min = std::numeric_limits<Real>::infinity();
  Real source_max = -std::numeric_limits<Real>::infinity();
  bool source_finite_positive = true;
  std::ostringstream values;
  values << std::setprecision(17);
  for (int sj = 0; sj < source.width; ++sj) {
    for (int si = 0; si < source.width; ++si) {
      const Real value = fine(m, Z4c::I_Z4C_CHI, fine_k,
                              source.refj + sj, source.refi + si);
      if (sj != 0 || si != 0) values << ";";
      values << value;
      source_min = std::min(source_min, value);
      source_max = std::max(source_max, value);
      source_finite_positive =
          source_finite_positive && std::isfinite(value) && value > 0.0;
    }
  }
  const Real shadow = 0.25 *
      (fine(m, Z4c::I_Z4C_CHI, fine_k, fine_j, fine_i) +
       fine(m, Z4c::I_Z4C_CHI, fine_k, fine_j, fine_i + 1) +
       fine(m, Z4c::I_Z4C_CHI, fine_k, fine_j + 1, fine_i) +
       fine(m, Z4c::I_Z4C_CHI, fine_k, fine_j + 1, fine_i + 1));
  const Real production =
      coarse(m, Z4c::I_Z4C_CHI, coarse_k, coarse_j, coarse_i);
  const Real rho = CellCenterX(coarse_i - cis, nx1 / 2,
                               size.x1min, size.x1max);
  const Real z = CellCenterX(coarse_j - cjs, nx2 / 2,
                             size.x2min, size.x2max);
  const Real absolute_difference = std::fabs(production - shadow);
  const Real relative_difference =
      absolute_difference / std::max({std::fabs(production), std::fabs(shadow),
                                      std::numeric_limits<Real>::min()});
  output << std::setprecision(17) << global_variable::my_rank << "," << m
         << "," << gid << "," << neighbor_slot << "," << writer_ordinal
         << "," << coarse_i << "," << coarse_j << "," << fine_i << ","
         << fine_j << "," << rho << "," << z << "," << RuleClass(source)
         << "," << (source.edge_i ? 1 : 0) << ","
         << (source.edge_j ? 1 : 0) << "," << source.refi << ","
         << source.refj << "," << source.width << "," << production << ","
         << shadow << "," << absolute_difference << ","
         << relative_difference << "," << source_min << "," << source_max
         << "," << (source_finite_positive ? 1 : 0) << ",\""
         << values.str() << "\"\n";
}

Z4cAMRTransfer TransferFromName(const std::string &name) {
  if (name == "high_order") return Z4cAMRTransfer::high_order;
  if (name == "limited_o2") return Z4cAMRTransfer::limited_o2;
  DiagnosticFailure("unknown target transaction transfer '" + name + "'");
}

}  // namespace

AMRJumpDiagnosticRuntime::AMRJumpDiagnosticRuntime(
    MeshBlockPack *pack, const AMRJumpDiagnosticConfig &config)
    : pack_(pack), config_(config) {
  if (pack_ == nullptr || pack_->pmesh == nullptr) {
    DiagnosticFailure("requires live MeshBlockPack and Mesh objects");
  }
  rank_root_ = (fs::path(config_.output_basename) / RankTag()).string();
}

void AMRJumpDiagnosticRuntime::EnsureOutputInitialized() {
  if (output_initialized_) return;
  CreateDirectory(rank_root_);
  std::ostringstream schema;
  schema << "{\"schema\":\"" << kAMRJumpDiagnosticSchema << "\","
         << "\"real_bytes\":" << sizeof(Real) << ","
         << "\"layout\":\"m,n,k,j,i_row_major_stream\","
         << "\"z4c_components\":[";
  for (int n = 0; n < Z4c::nz4c; ++n) {
    if (n != 0) schema << ",";
    schema << "\"" << Z4c::Z4c_names[n] << "\"";
  }
  schema << "],\"constraint_components\":[";
  for (int n = 0; n < Z4c::ncon; ++n) {
    if (n != 0) schema << ",";
    schema << "\"" << Z4c::Constraint_names[n] << "\"";
  }
  const std::string pre_target_transfer =
      Z4cAMRTransferName(pack_->pz4c->opt.amr_transfer);
  const std::string target_transfer = config_.target_transfer.empty()
                                          ? pre_target_transfer
                                          : config_.target_transfer;
  schema << "],\"hierarchy_control\":\""
         << AMRJumpHierarchyControlName(config_.hierarchy_control)
         << "\",\"amr_transfer\":\""
         << target_transfer << "\",\"pre_target_amr_transfer\":\""
         << pre_target_transfer
         << "\",\"target_transaction_only_transfer\":"
         << (config_.target_transfer.empty() ? "false" : "true")
         << ",\"derivative_order_audit\":"
         << (config_.derivative_order_audit ? "true" : "false") << "}";
  WriteTextAtomically(fs::path(rank_root_) / "schema.json", schema.str() + "\n");
  output_initialized_ = true;
}

bool AMRJumpDiagnosticRuntime::ShouldFreezeHierarchy() const {
  return target_seen_ &&
         config_.hierarchy_control != AMRJumpHierarchyControl::dynamic;
}

bool AMRJumpDiagnosticRuntime::ShouldBufferTargetCycle(const int cycle) const {
  return !target_seen_ &&
         config_.hierarchy_control ==
             AMRJumpHierarchyControl::buffered_freeze_after_target &&
         cycle == config_.target_cycle;
}

void AMRJumpDiagnosticRuntime::RecordHierarchyControl(
    const int original_refine, const int original_derefine,
    const int buffered_refine, const int suppressed_refine,
    const int suppressed_derefine) {
  EnsureOutputInitialized();
  std::ostringstream record;
  record << std::setprecision(17)
         << "{\"schema\":\"athenak_z4c_amr_hierarchy_control_v1\","
         << "\"rank\":" << global_variable::my_rank << ",\"cycle\":"
         << pack_->pmesh->ncycle << ",\"time\":" << pack_->pmesh->time
         << ",\"control\":\""
         << AMRJumpHierarchyControlName(config_.hierarchy_control) << "\","
         << "\"target_seen\":" << (target_seen_ ? "true" : "false") << ","
         << "\"original_refine\":" << original_refine << ","
         << "\"original_derefine\":" << original_derefine << ","
         << "\"buffered_refine_added\":" << buffered_refine << ","
         << "\"suppressed_refine\":" << suppressed_refine << ","
         << "\"suppressed_derefine\":" << suppressed_derefine << "}\n";
  AppendText(fs::path(rank_root_) / "hierarchy_control.jsonl", record.str());
}

void AMRJumpDiagnosticRuntime::BeginTransaction(
    const MeshRefinement &refinement) {
  // Z4c is constructed before main() applies -d via ChangeRunDir().  Defer every
  // diagnostic filesystem side effect until the first runtime AMR transaction so
  // relative output basenames are rooted in the requested run directory.
  EnsureOutputInitialized();
  if (transaction_active_) {
    DiagnosticFailure("attempted to begin an overlapping AMR transaction");
  }
  Mesh *mesh = pack_->pmesh;
  bool any_flag = false;
  for (int gid = 0; gid < mesh->nmb_total; ++gid) {
    any_flag = any_flag || refinement.refine_flag.h_view(gid) != 0;
  }
  if (!any_flag) return;

  transaction_active_ = true;
  old_nmb_total_ = mesh->nmb_total;
  old_max_level_ =
      MaximumLevel(mesh->lloc_eachmb, mesh->nmb_total) - mesh->root_level;
  old_ranks_.resize(mesh->nmb_total);
  old_flags_.resize(mesh->nmb_total);
  old_lx1_.resize(mesh->nmb_total);
  old_lx2_.resize(mesh->nmb_total);
  old_lx3_.resize(mesh->nmb_total);
  old_levels_.resize(mesh->nmb_total);
  for (int gid = 0; gid < mesh->nmb_total; ++gid) {
    const LogicalLocation &location = mesh->lloc_eachmb[gid];
    old_ranks_[gid] = mesh->rank_eachmb[gid];
    old_flags_[gid] = refinement.refine_flag.h_view(gid);
    old_lx1_[gid] = location.lx1;
    old_lx2_[gid] = location.lx2;
    old_lx3_[gid] = location.lx3;
    old_levels_[gid] = location.level;
  }

  pending_event_root_ =
      (fs::path(rank_root_) /
       ("event_" + CycleTag(mesh->ncycle) + "_pending"))
          .string();
  std::error_code cleanup_error;
  fs::remove_all(pending_event_root_, cleanup_error);
  if (cleanup_error) {
    DiagnosticFailure("cannot clear stale pending event: " + cleanup_error.message());
  }
  CreateDirectory(pending_event_root_);
  pending_t0_ = true;
  CapturePhase(AMRJumpPhase::t0_accepted_old_hierarchy,
               AMRJumpWriter::accepted_old_state, 0, true, true);

  std::ostringstream topology;
  topology << "old_gid,rank,level,lx1,lx2,lx3,requested_flag\n";
  for (int gid = 0; gid < old_nmb_total_; ++gid) {
    topology << gid << "," << old_ranks_[gid] << "," << old_levels_[gid]
             << "," << old_lx1_[gid] << "," << old_lx2_[gid] << ","
             << old_lx3_[gid] << "," << old_flags_[gid] << "\n";
  }
  WriteTextAtomically(fs::path(pending_event_root_) / "t0_old_topology.csv",
                      topology.str());
}

void AMRJumpDiagnosticRuntime::CancelTransaction() {
  if (!transaction_active_) return;
  RestoreTargetTransfer();
  DiscardPendingT0();
  transaction_active_ = false;
  detailed_event_active_ = false;
}

void AMRJumpDiagnosticRuntime::RecordTopologyProposal(
    const MeshRefinement &refinement, const int old_nmb, const int new_nmb,
    const int nnew, const int ndel) {
  if (!transaction_active_ || !pending_t0_) {
    DiagnosticFailure("topology proposal has no matching T0 transaction");
  }
  if (old_nmb != old_nmb_total_) {
    DiagnosticFailure("old MeshBlock count changed between T0 and T1");
  }
  Mesh *mesh = pack_->pmesh;
  new_max_level_ =
      MaximumLevel(refinement.new_lloc_eachmb, new_nmb) - mesh->root_level;
  detailed_event_active_ = new_max_level_ != old_max_level_;
  if (detailed_event_active_) {
    event_root_ =
        (fs::path(rank_root_) /
         ("event_" + CycleTag(mesh->ncycle) + "_l" +
          std::to_string(old_max_level_) + "_to_l" +
          std::to_string(new_max_level_)))
            .string();
    std::error_code error;
    fs::rename(pending_event_root_, event_root_, error);
    if (error) {
      DiagnosticFailure("cannot promote detailed event directory: " + error.message());
    }
    pending_t0_ = false;
    std::ostringstream topology;
    topology << "new_gid,new_rank,new_level,new_lx1,new_lx2,new_lx3,old_gid,"
                "old_rank,old_level,old_lx1,old_lx2,old_lx3,requested_flag\n";
    for (int gid = 0; gid < new_nmb; ++gid) {
      const int old_gid = refinement.newtoold[gid];
      topology << gid << "," << refinement.new_rank_eachmb[gid] << ","
               << refinement.new_lloc_eachmb[gid].level << ","
               << refinement.new_lloc_eachmb[gid].lx1 << ","
               << refinement.new_lloc_eachmb[gid].lx2 << ","
               << refinement.new_lloc_eachmb[gid].lx3 << "," << old_gid
               << "," << old_ranks_.at(old_gid) << ","
               << old_levels_.at(old_gid) << "," << old_lx1_.at(old_gid)
               << "," << old_lx2_.at(old_gid) << ","
               << old_lx3_.at(old_gid) << "," << old_flags_.at(old_gid)
               << "\n";
    }
    WriteTextAtomically(fs::path(event_root_) / "t1_topology_proposal.csv",
                        topology.str());
    std::ostringstream phase;
    phase << std::setprecision(17)
          << "{\"schema\":\"" << kAMRJumpDiagnosticSchema << "\","
          << "\"phase\":\"T1_BALANCED_TOPOLOGY_PROPOSAL\","
          << "\"writer\":\"TOPOLOGY_ONLY\",\"rank\":"
          << global_variable::my_rank << ",\"cycle\":" << mesh->ncycle
          << ",\"time\":" << mesh->time << ",\"old_nmb\":" << old_nmb
          << ",\"new_nmb\":" << new_nmb << ",\"nnew\":" << nnew
          << ",\"ndel\":" << ndel << ",\"old_max_level\":"
          << old_max_level_ << ",\"new_max_level\":" << new_max_level_
          << "}";
    WriteTextAtomically(fs::path(event_root_) / "t1_phase.json",
                        phase.str() + "\n");
    t3_last_ordinal_ = -1;
    const bool cycle_matches =
        config_.target_cycle < 0 || mesh->ncycle == config_.target_cycle;
    if (!target_seen_ && old_max_level_ == config_.target_level_before &&
        new_max_level_ == config_.target_level_after && cycle_matches) {
      if (!config_.target_transfer.empty()) {
        if (target_transfer_active_) {
          DiagnosticFailure("target transfer override is already active");
        }
        saved_amr_transfer_ =
            static_cast<int>(pack_->pz4c->opt.amr_transfer);
        pack_->pz4c->opt.amr_transfer =
            TransferFromName(config_.target_transfer);
        target_transfer_active_ = true;
      }
      target_seen_ = true;
      target_cycle_ = mesh->ncycle;
    }
  } else {
    DiscardPendingT0();
  }
  WriteCompactTransaction(nnew, ndel);
}

void AMRJumpDiagnosticRuntime::RecordT2() {
  if (!transaction_active_) return;
  if (detailed_event_active_) {
    CapturePhase(AMRJumpPhase::t2_redistributed_refined_active,
                 AMRJumpWriter::refine_or_derefine_transfer, 0, false, true);
  }
}

void AMRJumpDiagnosticRuntime::RecordT3(const AMRJumpWriter writer,
                                        const int ordinal,
                                        const bool final_boundary_state) {
  if (!transaction_active_ || !detailed_event_active_) return;
  if (ordinal <= t3_last_ordinal_) {
    DiagnosticFailure("T3 writer ordinals are not strictly increasing");
  }
  t3_last_ordinal_ = ordinal;
  CapturePhase(AMRJumpPhase::t3_boundary_reconstruction, writer, ordinal,
               final_boundary_state, true);
}

void AMRJumpDiagnosticRuntime::RecordRestrictionShadow() {
  if (!transaction_active_ || !detailed_event_active_) return;
  Mesh *mesh = pack_->pmesh;
  Z4c *z4c = pack_->pz4c;
  if (!mesh->two_d || mesh->mb_indcs.nx3 != 1) {
    DiagnosticFailure("restriction shadow requires the 2-D Cartoon layout");
  }
  Kokkos::fence();
  const auto fine = Kokkos::create_mirror_view_and_copy(HostMemSpace(), z4c->u0);
  const auto coarse =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), z4c->coarse_u0);
  auto &indcs = mesh->mb_indcs;
  std::ostringstream output;
  output << "rank,local_m,gid,neighbor_slot,writer_ordinal,coarse_i,coarse_j,"
            "fine_i,fine_j,rho,z,rule_class,edge_i,edge_j,source_ref_i,"
            "source_ref_j,source_width,production_chi,shadow_chi,abs_diff,"
            "rel_diff,source_min_chi,source_max_chi,source_finite_positive,"
            "source_stencil_row_major\n";
  for (int m = 0; m < pack_->nmb_thispack; ++m) {
    const int gid = pack_->pmb->mb_gid.h_view(m);
    const RegionSize &size = pack_->pmb->mb_size.h_view(m);
    for (int j = indcs.cjs; j <= indcs.cje; ++j) {
      for (int i = indcs.cis; i <= indcs.cie; ++i) {
        const int fine_i = 2 * i - indcs.cis;
        const int fine_j = 2 * j - indcs.cjs;
        AppendChiShadowRecord(output, fine, coarse, size, m, gid, -1, 0,
                              indcs.cks, j, i, indcs.ks, fine_j, fine_i,
                              indcs.cis, indcs.cjs, indcs.nx1, indcs.nx2,
                              z4c->opt.fd_stencil);
      }
    }
  }
  WriteTextAtomically(fs::path(event_root_) / "t3_00_restrict_shadow_chi.csv",
                      output.str());
}

void AMRJumpDiagnosticRuntime::RecordSameLevelRefreshShadow() {
  if (!transaction_active_ || !detailed_event_active_) return;
  Mesh *mesh = pack_->pmesh;
  Z4c *z4c = pack_->pz4c;
  if (!mesh->two_d || mesh->mb_indcs.nx3 != 1) {
    DiagnosticFailure("same-level shadow requires the 2-D Cartoon layout");
  }
  Kokkos::fence();
  const auto fine = Kokkos::create_mirror_view_and_copy(HostMemSpace(), z4c->u0);
  const auto coarse =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), z4c->coarse_u0);
  const auto neighbors = Kokkos::create_mirror_view_and_copy(
      HostMemSpace(), pack_->pmb->nghbr.d_view);
  const auto levels = Kokkos::create_mirror_view_and_copy(
      HostMemSpace(), pack_->pmb->mb_lev.d_view);
  auto &indcs = mesh->mb_indcs;
  std::ostringstream output;
  output << "rank,local_m,gid,neighbor_slot,writer_ordinal,coarse_i,coarse_j,"
            "fine_i,fine_j,rho,z,rule_class,edge_i,edge_j,source_ref_i,"
            "source_ref_j,source_width,production_chi,shadow_chi,abs_diff,"
            "rel_diff,source_min_chi,source_max_chi,source_finite_positive,"
            "source_stencil_row_major\n";
  for (int m = 0; m < pack_->nmb_thispack; ++m) {
    const int gid = pack_->pmb->mb_gid.h_view(m);
    const RegionSize &size = pack_->pmb->mb_size.h_view(m);
    for (int n = 0; n < pack_->pmb->nnghbr; ++n) {
      if (neighbors(m, n).gid < 0 || neighbors(m, n).lev != levels(m)) continue;
      const MeshBufferIndcs &same = z4c->pbval_u->recvbuf[n].isame[0];
      int il = (same.bis + indcs.cis) / 2;
      int iu = (same.bie + indcs.cis) / 2;
      int jl = (same.bjs + indcs.cjs) / 2;
      int ju = (same.bje + indcs.cjs) / 2;
      const auto irange = CompleteFinePairCoarseRange(
          il, iu, indcs.cis, indcs.is, fine.extent_int(4));
      const auto jrange = CompleteFinePairCoarseRange(
          jl, ju, indcs.cjs, indcs.js, fine.extent_int(3));
      for (int j = jrange.lower; j <= jrange.upper; ++j) {
        for (int i = irange.lower; i <= irange.upper; ++i) {
          const int fine_i = (i - indcs.cis) * 2 + indcs.is;
          const int fine_j = (j - indcs.cjs) * 2 + indcs.js;
          AppendChiShadowRecord(output, fine, coarse, size, m, gid, n, 3,
                                indcs.cks, j, i, indcs.ks, fine_j, fine_i,
                                indcs.cis, indcs.cjs, indcs.nx1, indcs.nx2,
                                z4c->opt.fd_stencil);
        }
      }
    }
  }
  WriteTextAtomically(
      fs::path(event_root_) / "t3_03_same_level_refresh_shadow_chi.csv",
      output.str());
}

void AMRJumpDiagnosticRuntime::RecordT4() {
  if (!transaction_active_ || !detailed_event_active_) return;
  CapturePhase(AMRJumpPhase::t4_projected_z4c,
               AMRJumpWriter::algebraic_projection, 0, true, true);
}

void AMRJumpDiagnosticRuntime::RecordT5() {
  if (!transaction_active_) return;
  if (detailed_event_active_) {
    CapturePhase(AMRJumpPhase::t5_accepted_new_hierarchy,
                 AMRJumpWriter::adm_or_constraint_recomputation, 0, true, true);
  }
  std::ostringstream end;
  end << std::setprecision(17)
      << "{\"schema\":\"athenak_z4c_amr_transaction_end_v1\","
      << "\"rank\":" << global_variable::my_rank << ",\"cycle\":"
      << pack_->pmesh->ncycle << ",\"time\":" << pack_->pmesh->time
      << ",\"old_max_level\":" << old_max_level_
      << ",\"new_max_level\":" << new_max_level_ << "}\n";
  AppendText(fs::path(rank_root_) / "transactions.jsonl", end.str());
  WriteAcceptedTopologySnapshot();
  const bool record_target_transfer =
      detailed_event_active_ && target_transfer_active_;
  const std::string effective_target_transfer =
      Z4cAMRTransferName(pack_->pz4c->opt.amr_transfer);
  RestoreTargetTransfer();
  if (record_target_transfer) {
    std::ostringstream lifecycle;
    lifecycle << "{\"schema\":\"athenak_z4c_amr_target_transfer_lifecycle_v1\","
              << "\"rank\":" << global_variable::my_rank
              << ",\"cycle\":" << pack_->pmesh->ncycle
              << ",\"target_transfer\":\"" << effective_target_transfer
              << "\",\"restored_transfer\":\""
              << Z4cAMRTransferName(pack_->pz4c->opt.amr_transfer)
              << "\",\"restored_after_t5\":true}\n";
    WriteTextAtomically(
        fs::path(event_root_) / "target_transfer_lifecycle.json",
        lifecycle.str());
  }
  transaction_active_ = false;
  detailed_event_active_ = false;
  event_root_.clear();
}

void AMRJumpDiagnosticRuntime::RestoreTargetTransfer() {
  if (!target_transfer_active_) return;
  if (saved_amr_transfer_ < 0) {
    DiagnosticFailure("target transfer override has no saved production policy");
  }
  pack_->pz4c->opt.amr_transfer =
      static_cast<Z4cAMRTransfer>(saved_amr_transfer_);
  saved_amr_transfer_ = -1;
  target_transfer_active_ = false;
}

void AMRJumpDiagnosticRuntime::RecordRKStageCoarseFineExposure(const int stage) {
  if (!target_seen_) return;
  EnsureOutputInitialized();
  const CoarseFineFaceInventory faces = LocalCoarseFineFaceInventory(pack_);
  local_x_cf_ += faces.incidents;
  std::ostringstream record;
  record << std::setprecision(17)
         << "{\"schema\":\"athenak_z4c_amr_rk_stage_exposure_v1\","
         << "\"rank\":" << global_variable::my_rank << ",\"cycle\":"
         << pack_->pmesh->ncycle << ",\"time\":" << pack_->pmesh->time
         << ",\"stage\":" << stage
         << ",\"coarse_fine_leaf_face_incidents\":" << faces.incidents
         << ",\"cumulative_X_CF\":" << local_x_cf_
         << ",\"locations\":\"" << faces.locations << "\"}\n";
  AppendText(fs::path(rank_root_) / "rk_stage_exposure.jsonl", record.str());
}

void AMRJumpDiagnosticRuntime::AfterAcceptedCycle(Driver *driver) {
  if (!target_seen_ || driver == nullptr) return;
  const int cycle = pack_->pmesh->ncycle;
  if (cycle < target_cycle_ || cycle > target_cycle_ + config_.post_cycles) return;
  WriteAcceptedCycleAggregate();
  if (cycle == target_cycle_ + config_.post_cycles) {
    driver->user_stop = true;
    if (config_.post_cycles == 0) {
      driver->user_stop_reason =
          "completed requested zero-PDE Z4c AMR-jump diagnostic at target "
          "cycle " + std::to_string(target_cycle_) +
          " after T5 and before the next RHS";
    } else {
      driver->user_stop_reason =
          "completed requested Z4c AMR-jump diagnostic window: target cycle " +
          std::to_string(target_cycle_) + " plus " +
          std::to_string(config_.post_cycles) + " accepted cycles";
    }
  }
}

void AMRJumpDiagnosticRuntime::CapturePhase(
    const AMRJumpPhase phase, const AMRJumpWriter writer, const int ordinal,
    const bool constraints_valid, const bool include_coarse) {
  const fs::path root = pending_t0_ ? pending_event_root_ : event_root_;
  if (root.empty()) DiagnosticFailure("phase capture has no event directory");
  std::ostringstream tag;
  tag << "t" << static_cast<int>(phase) << "_" << std::setw(2)
      << std::setfill('0') << ordinal << "_" << AMRJumpWriterName(writer);
  const fs::path phase_root = root / tag.str();
  CreateDirectory(phase_root);
  Mesh *mesh = pack_->pmesh;
  Z4c *z4c = pack_->pz4c;
  const int nmb = pack_->nmb_thispack;
  Kokkos::fence();
  WriteViewBinary(phase_root / "u0.bin", z4c->u0, nmb);
  if (include_coarse && mesh->multilevel) {
    WriteViewBinary(phase_root / "coarse_u0.bin", z4c->coarse_u0, nmb);
  }

  DvceArray5D<Real> scratch_adm("amr jump scratch adm", 1, 1, 1, 1, 1);
  DvceArray5D<Real> scratch_constraints("amr jump scratch constraints", 1, 1, 1, 1, 1);
  const DvceArray5D<Real> *adm_state = nullptr;
  const DvceArray5D<Real> *constraint_state = nullptr;
  if (constraints_valid) {
    if (phase == AMRJumpPhase::t0_accepted_old_hierarchy ||
        phase == AMRJumpPhase::t5_accepted_new_hierarchy) {
      adm_state = &pack_->padm->u_adm;
      constraint_state = &z4c->u_con;
    } else {
      z4c->EvaluateDiagnosticConstraints(scratch_adm, scratch_constraints);
      adm_state = &scratch_adm;
      constraint_state = &scratch_constraints;
    }
    WriteViewBinary(phase_root / "adm.bin", *adm_state, nmb);
    WriteViewBinary(phase_root / "constraints.bin", *constraint_state, nmb);
    const Aggregate aggregate =
        ComputeAggregate(pack_, z4c->u0, *adm_state, *constraint_state);
    WriteTextAtomically(phase_root / "aggregate.json",
                        AggregateJSON(aggregate, mesh,
                                      "athenak_z4c_amr_phase_aggregate_v1") +
                            "\n");
    if (phase == AMRJumpPhase::t5_accepted_new_hierarchy &&
        config_.derivative_order_audit) {
      constexpr std::array<int, 3> stencils = {2, 3, 4};
      constexpr std::array<const char *, 3> labels = {"o2", "o4", "o6"};
      for (std::size_t index = 0; index < stencils.size(); ++index) {
        z4c->EvaluateDiagnosticConstraints(
            scratch_adm, scratch_constraints, stencils[index]);
        WriteViewBinary(phase_root /
                            (std::string("constraints_") + labels[index] + ".bin"),
                        scratch_constraints, nmb);
      }
    }
  }
  WriteCurrentTopology((phase_root / "topology.csv").string());

  std::ostringstream metadata;
  metadata << std::setprecision(17)
           << "{\"schema\":\"" << kAMRJumpDiagnosticSchema << "\","
           << "\"phase\":\"" << AMRJumpPhaseName(phase) << "\","
           << "\"writer\":\"" << AMRJumpWriterName(writer) << "\","
           << "\"ordinal\":" << ordinal << ",\"rank\":"
           << global_variable::my_rank << ",\"cycle\":" << mesh->ncycle
           << ",\"time\":" << mesh->time << ",\"constraints_valid\":"
           << (constraints_valid ? "true" : "false")
           << ",\"derivative_order_audit\":"
           << ((phase == AMRJumpPhase::t5_accepted_new_hierarchy &&
                config_.derivative_order_audit) ? "true" : "false")
           << ",\"u0_shape\":" << ViewShapeJSON(z4c->u0, nmb)
           << ",\"active_bounds\":{\"is\":" << mesh->mb_indcs.is
           << ",\"ie\":" << mesh->mb_indcs.ie << ",\"js\":"
           << mesh->mb_indcs.js << ",\"je\":" << mesh->mb_indcs.je
           << ",\"ks\":" << mesh->mb_indcs.ks << ",\"ke\":"
           << mesh->mb_indcs.ke << "},\"meshblock_active_shape\":["
           << mesh->mb_indcs.nx3 << "," << mesh->mb_indcs.nx2 << ","
           << mesh->mb_indcs.nx1 << "],\"nghost\":"
           << mesh->mb_indcs.ng << ",\"coarse_active_bounds\":{\"cis\":"
           << mesh->mb_indcs.cis << ",\"cie\":" << mesh->mb_indcs.cie
           << ",\"cjs\":" << mesh->mb_indcs.cjs << ",\"cje\":"
           << mesh->mb_indcs.cje << ",\"cks\":" << mesh->mb_indcs.cks
           << ",\"cke\":" << mesh->mb_indcs.cke
           << "},\"root_level\":" << mesh->root_level;
  if (include_coarse && mesh->multilevel) {
    metadata << ",\"coarse_u0_shape\":"
             << ViewShapeJSON(z4c->coarse_u0, nmb);
  }
  if (constraints_valid) {
    metadata << ",\"adm_shape\":" << ViewShapeJSON(*adm_state, nmb)
             << ",\"constraint_shape\":"
             << ViewShapeJSON(*constraint_state, nmb);
  }
  metadata << "}";
  WriteTextAtomically(phase_root / "phase.json", metadata.str() + "\n");
}

void AMRJumpDiagnosticRuntime::WriteCurrentTopology(
    const std::string &path) const {
  Mesh *mesh = pack_->pmesh;
  std::ostringstream output;
  output << std::setprecision(17)
         << "local_m,gid,owner_rank,level,lx1,lx2,lx3,x1min,x1max,x2min,x2max,"
            "dx1,dx2,inner_x1_bc,outer_x1_bc,inner_x2_bc,outer_x2_bc\n";
  for (int m = 0; m < pack_->nmb_thispack; ++m) {
    const int gid = pack_->pmb->mb_gid.h_view(m);
    const LogicalLocation &location = mesh->lloc_eachmb[gid];
    const RegionSize &size = pack_->pmb->mb_size.h_view(m);
    output << m << "," << gid << "," << mesh->rank_eachmb[gid] << ","
           << location.level << "," << location.lx1 << "," << location.lx2
           << "," << location.lx3 << "," << size.x1min << "," << size.x1max
           << "," << size.x2min << "," << size.x2max << "," << size.dx1
           << "," << size.dx2 << ","
           << static_cast<int>(pack_->pmb->mb_bcs.h_view(m, inner_x1)) << ","
           << static_cast<int>(pack_->pmb->mb_bcs.h_view(m, outer_x1)) << ","
           << static_cast<int>(pack_->pmb->mb_bcs.h_view(m, inner_x2)) << ","
           << static_cast<int>(pack_->pmb->mb_bcs.h_view(m, outer_x2)) << "\n";
  }
  WriteTextAtomically(path, output.str());
}

void AMRJumpDiagnosticRuntime::WriteAcceptedTopologySnapshot() const {
  const fs::path directory = fs::path(rank_root_) / "accepted_topologies";
  std::ostringstream name;
  name << CycleTag(pack_->pmesh->ncycle) << ".csv";
  WriteCurrentTopology((directory / name.str()).string());
}

void AMRJumpDiagnosticRuntime::WriteCompactTransaction(const int nnew,
                                                       const int ndel) const {
  std::ostringstream output;
  output << std::setprecision(17)
         << "{\"schema\":\"athenak_z4c_amr_transaction_v1\","
         << "\"rank\":" << global_variable::my_rank << ",\"cycle\":"
         << pack_->pmesh->ncycle << ",\"time\":" << pack_->pmesh->time
         << ",\"old_nmb\":" << old_nmb_total_ << ",\"new_nmb\":"
         << old_nmb_total_ + nnew - ndel << ",\"nnew\":" << nnew
         << ",\"ndel\":" << ndel << ",\"old_max_level\":"
         << old_max_level_ << ",\"new_max_level\":" << new_max_level_
         << ",\"detailed\":"
         << (detailed_event_active_ ? "true" : "false") << "}\n";
  AppendText(fs::path(rank_root_) / "transactions.jsonl", output.str());
}

void AMRJumpDiagnosticRuntime::WriteAcceptedCycleAggregate() const {
  const Aggregate aggregate = ComputeAggregate(pack_, pack_->pz4c->u0,
                                               pack_->padm->u_adm,
                                               pack_->pz4c->u_con);
  const CoarseFineFaceInventory faces = LocalCoarseFineFaceInventory(pack_);
  std::string record = AggregateJSON(
      aggregate, pack_->pmesh, "athenak_z4c_amr_post_event_cycle_v1");
  if (record.empty() || record.back() != '}') {
    DiagnosticFailure("post-event aggregate did not produce a JSON object");
  }
  record.pop_back();
  std::ostringstream suffix;
  suffix << ",\"coarse_fine_leaf_face_incidents\":" << faces.incidents
         << ",\"cumulative_X_CF\":" << local_x_cf_
         << ",\"coarse_fine_face_locations\":\"" << faces.locations
         << "\"}";
  AppendText(fs::path(rank_root_) / "post_event_cycles.jsonl",
             record + suffix.str() + "\n");
}

void AMRJumpDiagnosticRuntime::DiscardPendingT0() {
  if (pending_event_root_.empty()) return;
  std::error_code error;
  fs::remove_all(pending_event_root_, error);
  if (error) {
    DiagnosticFailure("cannot discard pending T0 capture: " + error.message());
  }
  pending_event_root_.clear();
  pending_t0_ = false;
}

}  // namespace z4c
