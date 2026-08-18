//========================================================================================
//! \file chi_parent_provenance.cpp
//! \brief Default-off, state-preserving provenance audit for invalid coarse Z4c chi.

#include "z4c/chi_parent_provenance.hpp"

#include <sys/stat.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "bvals/bvals.hpp"
#include "coordinates/cell_locations.hpp"
#include "driver/driver.hpp"
#include "globals.hpp"
#include "mesh/amr_history_format.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "parameter_input.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_amr.hpp"

namespace z4c {
namespace {

[[noreturn]] void DiagnosticFatal(const std::string &message) {
  std::cerr << "### FATAL ERROR: chi parent provenance diagnostic: " << message
            << std::endl;
  std::exit(EXIT_FAILURE);
}

bool SafeBasename(const std::string &value) {
  if (value.empty() || value == "." || value == ".." || value.front() == '.') {
    return false;
  }
  return std::all_of(value.begin(), value.end(), [](const unsigned char c) {
    return std::isalnum(c) != 0 || c == '_' || c == '-' || c == '.';
  });
}

bool InvalidChi(const Real value) {
  return !std::isfinite(value) || !(value > 0.0);
}

const char *Classification(const Real value) {
  if (std::isnan(value)) return "NaN";
  if (std::isinf(value)) return std::signbit(value) ? "-Inf" : "+Inf";
  if (value == 0.0) return "zero";
  if (value < 0.0) return "negative";
  return "finite_positive";
}

const char *CheckpointName(const ChiProvenanceCheckpoint checkpoint) {
  switch (checkpoint) {
    case ChiProvenanceCheckpoint::s0_after_rk: return "S0_AFTER_EXP_RK_UPDATE";
    case ChiProvenanceCheckpoint::s1_after_restriction:
      return "S1_AFTER_RESTRICT_U";
    case ChiProvenanceCheckpoint::s2_after_receive: return "S2_AFTER_RECV_U";
    case ChiProvenanceCheckpoint::s3_after_boundary:
      return "S3_AFTER_PHYSICAL_AXIS_BC";
    case ChiProvenanceCheckpoint::s4_before_parent_gate:
      return "S4_BEFORE_PARENT_STENCIL_GATE";
  }
  return "UNKNOWN";
}

std::string TreeChecksum(const Mesh *mesh) {
  std::vector<amr_history::Location> leaves;
  leaves.reserve(mesh->nmb_total);
  for (int gid = 0; gid < mesh->nmb_total; ++gid) {
    const auto &loc = mesh->lloc_eachmb[gid];
    leaves.push_back({loc.level, loc.lx1, loc.lx2, loc.lx3});
  }
  std::sort(leaves.begin(), leaves.end());
  return amr_history::TreeChecksum(leaves);
}

struct CellIndex {
  int m = 0;
  int k = 0;
  int j = 0;
  int i = 0;
  bool operator<(const CellIndex &other) const {
    return std::tie(m, k, j, i) < std::tie(other.m, other.k, other.j, other.i);
  }
};

struct MinimumSummary {
  Real minimum = std::numeric_limits<Real>::infinity();
  CellIndex cell;
  std::uint64_t count = 0;
  std::uint64_t nonpositive = 0;
  std::uint64_t nonfinite = 0;
};

bool Inside(const MeshBufferIndcs &range, const CellIndex &cell) {
  return cell.i >= range.bis && cell.i <= range.bie &&
         cell.j >= range.bjs && cell.j <= range.bje &&
         cell.k >= range.bks && cell.k <= range.bke;
}

std::set<CellIndex> ConsumedCoarseCells(MeshBlockPack *pack,
                                        MeshBoundaryValuesCC *boundary) {
  std::set<CellIndex> cells;
  const auto &indcs = pack->pmesh->mb_indcs;
  const int stencil = pack->pz4c->opt.fd_stencil;
  const int lower = stencil / 2;
  const int nk = indcs.nx3 == 1 ? 1 : stencil + 1;
  auto &neighbors = pack->pmb->nghbr;
  auto &levels = pack->pmb->mb_lev;
  for (int m = 0; m < pack->nmb_thispack; ++m) {
    for (int n = 0; n < pack->pmb->nnghbr; ++n) {
      if (neighbors.h_view(m, n).gid < 0 ||
          neighbors.h_view(m, n).lev >= levels.h_view(m)) {
        continue;
      }
      const auto &range = boundary->recvbuf[n].iprol[0];
      for (int k = range.bks; k <= range.bke; ++k) {
        for (int j = range.bjs; j <= range.bje; ++j) {
          for (int i = range.bis; i <= range.bie; ++i) {
            for (int kk = 0; kk < nk; ++kk) {
              const int ck = indcs.nx3 == 1 ? k : k - lower + kk;
              for (int jj = 0; jj <= stencil; ++jj) {
                for (int ii = 0; ii <= stencil; ++ii) {
                  cells.insert({m, ck, j - lower + jj, i - lower + ii});
                }
              }
            }
          }
        }
      }
    }
  }
  return cells;
}

std::string Join(const std::vector<Real> &values) {
  std::ostringstream output;
  output << std::setprecision(17);
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) output << ';';
    output << values[index];
  }
  return output.str();
}

std::string JsonReal(const Real value) {
  if (!std::isfinite(value)) return "null";
  std::ostringstream output;
  output << std::setprecision(17) << value;
  return output.str();
}

}  // namespace

struct ChiParentProvenanceRuntime::HostSnapshot {
  int nmb = 0;
  int n3 = 0;
  int n2 = 0;
  int n1 = 0;
  std::vector<Real> values;

  Real at(const CellIndex &cell) const {
    if (cell.m < 0 || cell.m >= nmb || cell.k < 0 || cell.k >= n3 ||
        cell.j < 0 || cell.j >= n2 || cell.i < 0 || cell.i >= n1) {
      return std::numeric_limits<Real>::quiet_NaN();
    }
    const std::size_t index = static_cast<std::size_t>(cell.i) +
        static_cast<std::size_t>(n1) *
            (static_cast<std::size_t>(cell.j) +
             static_cast<std::size_t>(n2) *
                 (static_cast<std::size_t>(cell.k) +
                  static_cast<std::size_t>(n3) * cell.m));
    return values[index];
  }
};

const char *ChiWriterProvenanceName(const ChiWriterProvenance writer) {
  switch (writer) {
    case ChiWriterProvenance::local_restriction_centered:
      return "LOCAL_RESTRICTION_CENTERED";
    case ChiWriterProvenance::local_restriction_radial_edge:
      return "LOCAL_RESTRICTION_RADIAL_EDGE";
    case ChiWriterProvenance::local_restriction_z_edge:
      return "LOCAL_RESTRICTION_Z_EDGE";
    case ChiWriterProvenance::local_restriction_corner:
      return "LOCAL_RESTRICTION_CORNER";
    case ChiWriterProvenance::same_level_owner_receive:
      return "SAME_LEVEL_OWNER_RECEIVE";
    case ChiWriterProvenance::coarser_neighbor_receive:
      return "COARSER_NEIGHBOR_RECEIVE";
    case ChiWriterProvenance::axis_boundary_fill: return "AXIS_BOUNDARY_FILL";
    case ChiWriterProvenance::outer_physical_boundary_fill:
      return "OUTER_PHYSICAL_BOUNDARY_FILL";
    case ChiWriterProvenance::preexisting_unchanged_cache:
      return "PREEXISTING_UNCHANGED_CACHE";
    case ChiWriterProvenance::unknown: return "UNKNOWN";
  }
  return "UNKNOWN";
}

ChiParentProvenanceConfig ReadChiParentProvenanceConfig(ParameterInput *pin) {
  ChiParentProvenanceConfig config;
  config.enabled =
      pin->GetOrAddBoolean("z4c", "chi_parent_provenance_diagnostic", false);
  config.start_time =
      pin->GetOrAddReal("z4c", "chi_parent_provenance_start_time", 0.0);
  config.output_basename = pin->GetOrAddString(
      "z4c", "chi_parent_provenance_output", "chi_parent_provenance");
  // Restart files produced by the uninstrumented source do not contain the new
  // keys, and Athena's command-line layer only overrides pre-existing keys.
  // Exact environment overrides therefore provide the bounded diagnostic seam
  // without rewriting an authenticated restart. They are ignored unless set.
  if (const char *enabled = std::getenv("ATHENA_CHI_PARENT_PROVENANCE_DIAGNOSTIC")) {
    if (std::strcmp(enabled, "1") != 0) {
      DiagnosticFatal("ATHENA_CHI_PARENT_PROVENANCE_DIAGNOSTIC must equal 1");
    }
    config.enabled = true;
  }
  if (const char *start = std::getenv("ATHENA_CHI_PARENT_PROVENANCE_START_TIME")) {
    char *end = nullptr;
    const double parsed = std::strtod(start, &end);
    if (end == start || *end != '\0' || !std::isfinite(parsed) || parsed < 0.0) {
      DiagnosticFatal("ATHENA_CHI_PARENT_PROVENANCE_START_TIME is invalid");
    }
    config.start_time = parsed;
  }
  if (const char *output = std::getenv("ATHENA_CHI_PARENT_PROVENANCE_OUTPUT")) {
    config.output_basename = output;
  }
  if (const char *trace = std::getenv("ATHENA_CHI_CONTROL_TARGET_TRACE")) {
    if (std::strcmp(trace, "1") != 0) {
      DiagnosticFatal("ATHENA_CHI_CONTROL_TARGET_TRACE must equal 1");
    }
    config.control_target_trace = true;
  }
  if (config.enabled && !SafeBasename(config.output_basename)) {
    DiagnosticFatal("output basename is not a safe relative path");
  }
  return config;
}

namespace {

ChiParentProvenanceRuntime::HostSnapshot CaptureChi(
    MeshBlockPack *pack, const DvceArray5D<Real> &source,
    const std::string &label) {
  ChiParentProvenanceRuntime::HostSnapshot result;
  const int nmb = pack->nmb_thispack;
  result.nmb = nmb;
  result.n3 = source.extent_int(2);
  result.n2 = source.extent_int(3);
  result.n1 = source.extent_int(4);
  // Fixing the variable index directly creates a LayoutStride subview, for
  // which Kokkos has no portable Cuda-to-Host deep-copy mechanism.  Pack only
  // chi into a contiguous device view first; this also avoids copying the other
  // 24 evolved variables at every diagnostic checkpoint.
  DvceArray4D<Real> packed(label, result.nmb, result.n3, result.n2, result.n1);
  par_for(label + "_kernel", DevExeSpace(), 0, result.nmb - 1,
          0, result.n3 - 1, 0, result.n2 - 1, 0, result.n1 - 1,
          KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    packed(m, k, j, i) = source(m, Z4c::I_Z4C_CHI, k, j, i);
  });
  auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), packed);
  result.values.resize(static_cast<std::size_t>(result.nmb) * result.n3 *
                       result.n2 * result.n1);
  for (int m = 0; m < result.nmb; ++m) {
    for (int k = 0; k < result.n3; ++k) {
      for (int j = 0; j < result.n2; ++j) {
        for (int i = 0; i < result.n1; ++i) {
          const CellIndex cell{m, k, j, i};
          const std::size_t index = static_cast<std::size_t>(i) +
              static_cast<std::size_t>(result.n1) *
                  (static_cast<std::size_t>(j) +
                   static_cast<std::size_t>(result.n2) *
                       (static_cast<std::size_t>(k) +
                        static_cast<std::size_t>(result.n3) * m));
          result.values[index] = host(m, k, j, i);
        }
      }
    }
  }
  return result;
}

ChiParentProvenanceRuntime::HostSnapshot CaptureFineChi(MeshBlockPack *pack) {
  return CaptureChi(pack, pack->pz4c->u0, "chi_parent_provenance_fine_pack");
}

ChiParentProvenanceRuntime::HostSnapshot CaptureCoarseChi(MeshBlockPack *pack) {
  return CaptureChi(pack, pack->pz4c->coarse_u0,
                    "chi_parent_provenance_coarse_pack");
}

MinimumSummary ActiveFineMinimum(MeshBlockPack *pack,
                                 const ChiParentProvenanceRuntime::HostSnapshot &state) {
  MinimumSummary result;
  const auto &indcs = pack->pmesh->mb_indcs;
  for (int m = 0; m < state.nmb; ++m) {
    for (int k = indcs.ks; k <= indcs.ke; ++k) {
      for (int j = indcs.js; j <= indcs.je; ++j) {
        for (int i = indcs.is; i <= indcs.ie; ++i) {
          const CellIndex cell{m, k, j, i};
          const Real value = state.at(cell);
          ++result.count;
          if (!std::isfinite(value)) ++result.nonfinite;
          else if (!(value > 0.0)) ++result.nonpositive;
          if (std::isfinite(value) && value < result.minimum) {
            result.minimum = value;
            result.cell = cell;
          }
        }
      }
    }
  }
  return result;
}

MinimumSummary ConsumedCoarseMinimum(
    const std::set<CellIndex> &cells,
    const ChiParentProvenanceRuntime::HostSnapshot &state) {
  MinimumSummary result;
  for (const CellIndex &cell : cells) {
    const Real value = state.at(cell);
    ++result.count;
    if (!std::isfinite(value)) ++result.nonfinite;
    else if (!(value > 0.0)) ++result.nonpositive;
    if (std::isfinite(value) && value < result.minimum) {
      result.minimum = value;
      result.cell = cell;
    }
  }
  return result;
}

std::pair<Real, Real> FineCoordinates(MeshBlockPack *pack,
                                      const CellIndex &cell) {
  const auto &indcs = pack->pmesh->mb_indcs;
  const auto &size = pack->pmb->mb_size.h_view(cell.m);
  return {size.x1min + (cell.i - indcs.is + 0.5) * size.dx1,
          size.x2min + (cell.j - indcs.js + 0.5) * size.dx2};
}

std::pair<Real, Real> CoarseCoordinates(MeshBlockPack *pack,
                                        const CellIndex &cell) {
  const auto &indcs = pack->pmesh->mb_indcs;
  const auto &size = pack->pmb->mb_size.h_view(cell.m);
  return {size.x1min + (cell.i - indcs.cis + 0.5) * 2.0 * size.dx1,
          size.x2min + (cell.j - indcs.cjs + 0.5) * 2.0 * size.dx2};
}

bool SameBits(const Real left, const Real right) {
  return std::memcmp(&left, &right, sizeof(Real)) == 0;
}

struct WriterInfo {
  ChiWriterProvenance writer = ChiWriterProvenance::unknown;
  std::string first_phase = "UNKNOWN";
  int owner_gid = -1;
  int owner_i = -1;
  int owner_j = -1;
  int owner_k = -1;
  std::string owner_storage = "UNKNOWN";
};

ChiWriterProvenance LocalRestrictionClass(MeshBlockPack *pack,
                                          const CellIndex &cell) {
  const auto &indcs = pack->pmesh->mb_indcs;
  const int finei = 2 * cell.i - indcs.cis;
  const int finej = 2 * cell.j - indcs.cjs;
  const int outer_i = indcs.nx1 + 2 * indcs.ng - 2;
  const int outer_j = indcs.nx2 + 2 * indcs.ng - 2;
  const bool radial = finei == 0 || finei == indcs.ng ||
      finei == indcs.ng + indcs.nx1 - 2 || finei == outer_i;
  const bool zedge = finej == 0 || finej == indcs.ng ||
      finej == indcs.ng + indcs.nx2 - 2 || finej == outer_j;
  if (radial && zedge) return ChiWriterProvenance::local_restriction_corner;
  if (radial) return ChiWriterProvenance::local_restriction_radial_edge;
  if (zedge) return ChiWriterProvenance::local_restriction_z_edge;
  return ChiWriterProvenance::local_restriction_centered;
}

bool InActiveCoarse(const RegionIndcs &indcs, const CellIndex &cell) {
  return cell.i >= indcs.cis && cell.i <= indcs.cie &&
         cell.j >= indcs.cjs && cell.j <= indcs.cje &&
         cell.k >= indcs.cks && cell.k <= indcs.cke;
}

WriterInfo IdentifyWriter(
    MeshBlockPack *pack, MeshBoundaryValuesCC *boundary, const CellIndex &cell,
    const ChiParentProvenanceRuntime::HostSnapshot &s1,
    const ChiParentProvenanceRuntime::HostSnapshot &s2,
    const ChiParentProvenanceRuntime::HostSnapshot &s3, const Real current) {
  WriterInfo result;
  const auto &indcs = pack->pmesh->mb_indcs;
  const int gid = pack->pmb->mb_gid.h_view(cell.m);
  const int level = pack->pmb->mb_lev.h_view(cell.m);
  const Real v1 = s1.at(cell);
  const Real v2 = s2.at(cell);
  const Real v3 = s3.at(cell);

  int receive_slot = -1;
  ChiWriterProvenance receive_writer = ChiWriterProvenance::unknown;
  for (int n = 0; n < pack->pmb->nnghbr; ++n) {
    const auto &neighbor = pack->pmb->nghbr.h_view(cell.m, n);
    if (neighbor.gid < 0) continue;
    if (neighbor.lev < level && Inside(boundary->recvbuf[n].icoar[0], cell)) {
      if (receive_slot >= 0 && receive_slot != n) {
        receive_writer = ChiWriterProvenance::unknown;
        receive_slot = -2;
        break;
      }
      receive_slot = n;
      receive_writer = ChiWriterProvenance::coarser_neighbor_receive;
    } else if (neighbor.lev == level &&
               Inside(boundary->recvbuf[n].isame_z4c, cell)) {
      if (receive_slot >= 0 && receive_slot != n) {
        receive_writer = ChiWriterProvenance::unknown;
        receive_slot = -2;
        break;
      }
      receive_slot = n;
      receive_writer = ChiWriterProvenance::same_level_owner_receive;
    }
  }

  const auto &bcs = pack->pmb->mb_bcs;
  const bool axis_cell = cell.i < indcs.cis &&
      bcs.h_view(cell.m, BoundaryFace::inner_x1) == BoundaryFlag::axis;
  const bool outer_cell =
      (cell.i < indcs.cis &&
       bcs.h_view(cell.m, BoundaryFace::inner_x1) != BoundaryFlag::block &&
       bcs.h_view(cell.m, BoundaryFace::inner_x1) != BoundaryFlag::axis) ||
      (cell.i > indcs.cie &&
       bcs.h_view(cell.m, BoundaryFace::outer_x1) != BoundaryFlag::block) ||
      (cell.j < indcs.cjs &&
       bcs.h_view(cell.m, BoundaryFace::inner_x2) != BoundaryFlag::block) ||
      (cell.j > indcs.cje &&
       bcs.h_view(cell.m, BoundaryFace::outer_x2) != BoundaryFlag::block);

  if (InActiveCoarse(indcs, cell) && InvalidChi(v1)) {
    result.writer = LocalRestrictionClass(pack, cell);
    result.first_phase = "S1";
    result.owner_gid = gid;
    result.owner_i = cell.i;
    result.owner_j = cell.j;
    result.owner_k = cell.k;
    result.owner_storage = "coarse_u0_active";
  } else if (receive_slot >= 0 && InvalidChi(v2) &&
             (!InvalidChi(v1) || !SameBits(v1, v2))) {
    result.writer = receive_writer;
    result.first_phase = "S2";
    const auto &neighbor = pack->pmb->nghbr.h_view(cell.m, receive_slot);
    result.owner_gid = neighbor.gid;
    result.owner_storage = receive_writer ==
            ChiWriterProvenance::same_level_owner_receive
        ? "owner_coarse_u0"
        : "owner_u0";
  } else if (axis_cell && InvalidChi(v3) &&
             (!InvalidChi(v2) || !SameBits(v2, v3))) {
    result.writer = ChiWriterProvenance::axis_boundary_fill;
    result.first_phase = "S3";
    result.owner_gid = gid;
    result.owner_storage = "coarse_u0_axis_ghost";
  } else if (outer_cell && InvalidChi(v3) &&
             (!InvalidChi(v2) || !SameBits(v2, v3))) {
    result.writer = ChiWriterProvenance::outer_physical_boundary_fill;
    result.first_phase = "S3";
    result.owner_gid = gid;
    result.owner_storage = "coarse_u0_physical_ghost";
  } else if (!InvalidChi(v3) && InvalidChi(current)) {
    result.writer = ChiWriterProvenance::unknown;
    result.first_phase = "S4";
  } else {
    result.writer = ChiWriterProvenance::preexisting_unchanged_cache;
    result.first_phase = "PREEXISTING";
  }
  return result;
}

struct RestrictionDiagnostic {
  bool available = false;
  bool all_fine_positive = false;
  Real fine_min = std::numeric_limits<Real>::infinity();
  Real fine_max = -std::numeric_limits<Real>::infinity();
  Real high = std::numeric_limits<Real>::quiet_NaN();
  Real average = std::numeric_limits<Real>::quiet_NaN();
  Real log_candidate = std::numeric_limits<Real>::quiet_NaN();
  Real parity_centered = std::numeric_limits<Real>::quiet_NaN();
  std::vector<Real> fine_values;
  std::vector<Real> weights;
};

RestrictionDiagnostic RestrictionCandidates(
    MeshBlockPack *pack, const CellIndex &cell,
    const ChiParentProvenanceRuntime::HostSnapshot &fine) {
  RestrictionDiagnostic result;
  const auto &indcs = pack->pmesh->mb_indcs;
  if (!InActiveCoarse(indcs, cell) || indcs.nx3 != 1 ||
      pack->pz4c->opt.fd_stencil != 4) {
    return result;
  }
  const int fi = 2 * cell.i - indcs.cis;
  const int fj = 2 * cell.j - indcs.cjs;
  const bool offset_i = fi < indcs.nx1 / 2 + indcs.ng;
  const bool offset_j = fj < indcs.nx2 / 2 + indcs.ng;
  int refi = offset_i ? fi - 1 : fi - 2;
  int refj = offset_j ? fj - 1 : fj - 2;
  const int outer_i = indcs.nx1 + 2 * indcs.ng - 2;
  const int outer_j = indcs.nx2 + 2 * indcs.ng - 2;
  const bool edge_i = fi == 0 || fi == indcs.ng ||
      fi == indcs.ng + indcs.nx1 - 2 || fi == outer_i;
  const bool edge_j = fj == 0 || fj == indcs.ng ||
      fj == indcs.ng + indcs.nx2 - 2 || fj == outer_j;
  if (fi == indcs.ng) ++refi;
  if (fj == indcs.ng) ++refj;
  if (fi == indcs.ng + indcs.nx1 - 2) --refi;
  if (fj == indcs.ng + indcs.nx2 - 2) --refj;
  if (fi == 0) refi = 0;
  if (fj == 0) refj = 0;
  if (fi == outer_i) refi = indcs.nx1 + indcs.ng - 1;
  if (fj == outer_j) refj = indcs.nx2 + indcs.ng - 1;
  constexpr std::array<Real, 5> centered = {
      -5.0 / 128.0, 60.0 / 128.0, 90.0 / 128.0,
      -20.0 / 128.0, 3.0 / 128.0};
  constexpr std::array<Real, 5> edge = {
      35.0 / 128.0, 140.0 / 128.0, -70.0 / 128.0,
      28.0 / 128.0, -5.0 / 128.0};
  result.available = true;
  result.all_fine_positive = true;
  Real log_sum = 0.0;
  Real parity_sum = 0.0;
  for (int jj = 0; jj < 5; ++jj) {
    for (int ii = 0; ii < 5; ++ii) {
      const int wi_index = offset_i ? ii : 4 - ii;
      const int wj_index = offset_j ? jj : 4 - jj;
      const Real wi = edge_i ? edge[wi_index] : centered[wi_index];
      const Real wj = edge_j ? edge[wj_index] : centered[wj_index];
      const Real weight = wi * wj;
      const Real value = fine.at({cell.m, indcs.ks, refj + jj, refi + ii});
      result.fine_values.push_back(value);
      result.weights.push_back(weight);
      result.fine_min = std::min(result.fine_min, value);
      result.fine_max = std::max(result.fine_max, value);
      result.all_fine_positive = result.all_fine_positive && !InvalidChi(value);
      if (!InvalidChi(value)) log_sum += weight * std::log(value);

      if (fi == indcs.ng) {
        const int centered_refi = fi - 1;
        int parity_i = centered_refi + ii;
        if (parity_i < indcs.is) parity_i = 2 * indcs.is - 1 - parity_i;
        const Real pvalue = fine.at({cell.m, indcs.ks, refj + jj, parity_i});
        parity_sum += centered[ii] * wj * pvalue;
      }
    }
  }
  // high starts as NaN to make unavailable diagnostics explicit; replace it with
  // the exact tensor sum now that all terms have been gathered.
  result.high = 0.0;
  for (std::size_t q = 0; q < result.fine_values.size(); ++q) {
    result.high += result.weights[q] * result.fine_values[q];
  }
  const Real f00 = fine.at({cell.m, indcs.ks, fj, fi});
  const Real f10 = fine.at({cell.m, indcs.ks, fj, fi + 1});
  const Real f01 = fine.at({cell.m, indcs.ks, fj + 1, fi});
  const Real f11 = fine.at({cell.m, indcs.ks, fj + 1, fi + 1});
  result.average = 0.25 * (f00 + f10 + f01 + f11);
  if (result.all_fine_positive) result.log_candidate = std::exp(log_sum);
  if (fi == indcs.ng) result.parity_centered = parity_sum;
  return result;
}

std::uint64_t OrderedRealBits(const Real value) {
  static_assert(sizeof(Real) == sizeof(std::uint64_t),
                "the provenance audit requires binary64 Real");
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & (std::uint64_t{1} << 63)) != 0
             ? ~bits + 1
             : bits | (std::uint64_t{1} << 63);
}

long long UlpDistance(const Real left, const Real right) {
  if (!std::isfinite(left) || !std::isfinite(right)) {
    return SameBits(left, right) ? 0 : std::numeric_limits<long long>::max();
  }
  const std::uint64_t a = OrderedRealBits(left);
  const std::uint64_t b = OrderedRealBits(right);
  const std::uint64_t distance = a > b ? a - b : b - a;
  return distance > static_cast<std::uint64_t>(std::numeric_limits<long long>::max())
             ? std::numeric_limits<long long>::max()
             : static_cast<long long>(distance);
}

const char *StorageClass(const RegionIndcs &indcs, const CellIndex &cell) {
  return cell.i >= indcs.is && cell.i <= indcs.ie &&
                 cell.j >= indcs.js && cell.j <= indcs.je &&
                 cell.k >= indcs.ks && cell.k <= indcs.ke
             ? "active"
             : "ghost";
}

struct OwnerCell {
  bool found = false;
  CellIndex cell;
  int gid = -1;
};

OwnerCell FindActiveOwner(MeshBlockPack *pack, const int level,
                          const Real rho, const Real z) {
  const auto &indcs = pack->pmesh->mb_indcs;
  const Real epsilon = 64.0 * std::numeric_limits<Real>::epsilon() *
                       std::max({Real{1.0}, std::fabs(rho), std::fabs(z)});
  OwnerCell result;
  for (int m = 0; m < pack->nmb_thispack; ++m) {
    if (pack->pmb->mb_lev.h_view(m) != level) continue;
    const auto &size = pack->pmb->mb_size.h_view(m);
    if (rho < size.x1min - epsilon || rho >= size.x1max - epsilon ||
        z < size.x2min - epsilon || z >= size.x2max - epsilon) {
      continue;
    }
    const int i = indcs.is + static_cast<int>(std::floor((rho - size.x1min) / size.dx1));
    const int j = indcs.js + static_cast<int>(std::floor((z - size.x2min) / size.dx2));
    if (i < indcs.is || i > indcs.ie || j < indcs.js || j > indcs.je) continue;
    if (result.found) DiagnosticFatal("physical stencil value has multiple active owners");
    result.found = true;
    result.cell = {m, indcs.ks, j, i};
    result.gid = pack->pmb->mb_gid.h_view(m);
  }
  return result;
}

Real HostAdvectiveDerivative(const int order, const std::array<Real, 9> &v,
                             const Real velocity, const Real inverse_dx) {
  Real left = 0.0;
  Real right = 0.0;
  if (order == 2) {
    left = 0.5 * v[2] - 2.0 * v[3] + 1.5 * v[4];
    right = -0.5 * v[6] + 2.0 * v[5] - 1.5 * v[4];
  } else if (order == 4) {
    left = -v[1] / 12.0 + 6.0 * v[2] / 12.0 - 18.0 * v[3] / 12.0 +
           10.0 * v[4] / 12.0 + 3.0 * v[5] / 12.0;
    right = v[7] / 12.0 - 6.0 * v[6] / 12.0 + 18.0 * v[5] / 12.0 -
            10.0 * v[4] / 12.0 - 3.0 * v[3] / 12.0;
  } else if (order == 6) {
    left = v[0] / 60.0 - 2.0 * v[1] / 15.0 + 0.5 * v[2] -
           4.0 * v[3] / 3.0 + 7.0 * v[4] / 12.0 + 2.0 * v[5] / 5.0 -
           v[6] / 30.0;
    right = -v[8] / 60.0 + 2.0 * v[7] / 15.0 - 0.5 * v[6] +
            4.0 * v[5] / 3.0 - 7.0 * v[4] / 12.0 - 2.0 * v[3] / 5.0 +
            v[2] / 30.0;
  } else {
    DiagnosticFatal("unsupported shadow derivative order");
  }
  return velocity * (velocity < 0.0 ? left : right) * inverse_dx;
}

Real HostKODissipationO6(const std::array<Real, 9> &v, const Real inverse_dx,
                         const Real diss) {
  const Real raw = v[0] + v[8] - 8.0 * (v[1] + v[7]) +
                   28.0 * (v[2] + v[6]) - 56.0 * (v[3] + v[5]) +
                   70.0 * v[4];
  return raw * inverse_dx * diss;
}

std::string PhaseClassification(const Real affine_base, const Real candidate_full,
                                const Real candidate_no_adv,
                                const Real candidate_no_k,
                                const Real candidate_no_ko,
                                const bool accumulator_valid,
                                const bool copy_valid,
                                const bool stencil_valid) {
  if (!accumulator_valid || !copy_valid) return "RK_ACCUMULATOR_FAILURE";
  if (!stencil_valid) return "SAME_LEVEL_STENCIL_OR_GHOST_BUG";
  if (!(affine_base > 0.0)) return "RK_AFFINE_COMBINATION_FAILURE";
  if (candidate_full > 0.0 && std::isfinite(candidate_full)) return "NOT_ESTABLISHED";
  const bool advective = candidate_no_adv > 0.0;
  const bool curvature = candidate_no_k > 0.0;
  const bool ko = candidate_no_ko > 0.0;
  const int necessary = static_cast<int>(advective) + static_cast<int>(curvature) +
                        static_cast<int>(ko);
  if (necessary != 1) return "MIXED_TEMPORAL_SPATIAL_FAILURE";
  if (curvature) return "CURVATURE_SOURCE_STIFFNESS";
  if (advective) return "ADVECTION_DOMINATED_FAILURE";
  return "KO_DOMINATED_FAILURE";
}

}  // namespace

ChiParentProvenanceRuntime::ChiParentProvenanceRuntime(
    MeshBlockPack *pack, const ChiParentProvenanceConfig &config)
    : pack_(pack), config_(config), output_root_(config.output_basename) {
  // This runtime is created at the end of Z4c's constructor, before the owning
  // unique_ptr has been assigned back to MeshBlockPack::pz4c.
  if (pack_ == nullptr || pack_->pmesh == nullptr) {
    DiagnosticFatal("constructed without a complete MeshBlockPack");
  }
  if (global_variable::nranks != 1) {
    DiagnosticFatal("the bounded provenance diagnostic currently requires one MPI rank");
  }
  if (!pack_->pmesh->multilevel || pack_->pmesh->mb_indcs.nx3 != 1) {
    DiagnosticFatal("requires a multilevel collapsed-x3 calculation");
  }
  struct stat info;
  if (stat(output_root_.c_str(), &info) == 0) {
    DiagnosticFatal("fresh output directory already exists: " + output_root_);
  }
  if (mkdir(output_root_.c_str(), 0775) != 0) {
    DiagnosticFatal("cannot create output directory: " + output_root_);
  }
  {
    std::ofstream output(output_root_ + "/phase1_stage_minima.csv");
    output << "cycle,time_hex,rk_stage,checkpoint,tree_checksum,active_fine_min,"
              "active_min_gid,active_min_i,active_min_j,active_min_rho,"
              "active_min_z,active_nonpositive,active_nonfinite,"
              "consumed_coarse_min,coarse_min_gid,coarse_min_i,coarse_min_j,"
              "coarse_min_rho,coarse_min_z,coarse_nonpositive,coarse_nonfinite,"
              "consumed_coarse_cells\n";
  }
  {
    std::ofstream output(output_root_ + "/shadow_amr_requests.jsonl");
  }
  {
    std::ofstream output(output_root_ + "/replay_time_alignment.jsonl");
  }
  {
    std::ofstream output(output_root_ + "/preupdate_candidate_minima.csv");
    output << "cycle,time_hex,rk_stage,tree_checksum,gamma0,gamma1,beta_stage,"
              "dt,dt_hex,candidate_min,min_gid,min_i,min_j,min_rho,min_z,"
              "nonpositive,nonfinite\n";
  }
  if (config_.control_target_trace) {
    std::ofstream output(output_root_ + "/control_target_stage_decomposition.csv");
    output << "cycle,time_hex,rk_stage,gid,i,j,rho,z,gamma0,gamma1,beta_stage,"
              "dt,chi_old,chi_accumulator,affine_base,adv_rho,adv_z,adv_y,"
              "lie_divergence,curvature_source,ko_rho,ko_z,ko_y,stored_rhs,"
              "rhs_increment,candidate\n";
  }
}

ChiParentProvenanceRuntime::~ChiParentProvenanceRuntime() {
  delete fine_s0_;
  delete coarse_s1_;
  delete coarse_s2_;
  delete coarse_s3_;
  delete u0_before_copy_;
  delete u1_before_copy_;
  delete u1_after_copy_;
}

void ChiParentProvenanceRuntime::RecordBeforeCopy(Driver *driver, const int stage) {
  Mesh *mesh = pack_->pmesh;
  if (stage <= 0 || mesh->time < config_.start_time) return;
  delete u0_before_copy_;
  delete u1_before_copy_;
  delete u1_after_copy_;
  u0_before_copy_ = new HostSnapshot(CaptureChi(
      pack_, pack_->pz4c->u0, "chi_provenance_u0_before_copy"));
  u1_before_copy_ = new HostSnapshot(CaptureChi(
      pack_, pack_->pz4c->u1, "chi_provenance_u1_before_copy"));
  u1_after_copy_ = nullptr;
  copy_cycle_ = mesh->ncycle;
  copy_stage_ = stage;
}

void ChiParentProvenanceRuntime::RecordAfterCopy(Driver *driver, const int stage) {
  Mesh *mesh = pack_->pmesh;
  if (stage <= 0 || mesh->time < config_.start_time) return;
  if (copy_cycle_ != mesh->ncycle || copy_stage_ != stage ||
      u0_before_copy_ == nullptr || u1_before_copy_ == nullptr) {
    DiagnosticFatal("CopyU after-snapshot lacks its matching before-snapshot");
  }
  delete u1_after_copy_;
  u1_after_copy_ = new HostSnapshot(CaptureChi(
      pack_, pack_->pz4c->u1, "chi_provenance_u1_after_copy"));
}

void ChiParentProvenanceRuntime::AnalyzePreUpdate(Driver *driver, const int stage) {
  Mesh *mesh = pack_->pmesh;
  if (stage <= 0 || mesh->time < config_.start_time) return;
  if (copy_cycle_ != mesh->ncycle || copy_stage_ != stage ||
      u0_before_copy_ == nullptr || u1_before_copy_ == nullptr ||
      u1_after_copy_ == nullptr) {
    DiagnosticFatal("pre-update audit lacks a complete CopyU snapshot sequence");
  }
  const auto &indcs = mesh->mb_indcs;
  const int nmb = pack_->nmb_thispack;
  const Real gamma0 = driver->gam0[stage - 1];
  const Real gamma1 = driver->gam1[stage - 1];
  const Real beta_stage = driver->beta[stage - 1];
  const Real beta_dt = beta_stage * mesh->dt;
  auto u0 = pack_->pz4c->u0;
  auto u1 = pack_->pz4c->u1;
  auto rhs = pack_->pz4c->u_rhs;
  DvceArray4D<Real> packed_candidate(
      "chi preupdate candidate", nmb, u0.extent_int(2), u0.extent_int(3),
      u0.extent_int(4));
  par_for("chi preupdate candidate kernel", DevExeSpace(), 0, nmb - 1,
          indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
          KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    packed_candidate(m, k, j, i) = EvaluateChiRKCandidate(
        gamma0, gamma1, beta_dt, u0(m, Z4c::I_Z4C_CHI, k, j, i),
        u1(m, Z4c::I_Z4C_CHI, k, j, i),
        rhs(m, Z4c::I_Z4C_CHI, k, j, i)).candidate;
  });
  auto candidate = Kokkos::create_mirror_view_and_copy(HostMemSpace(), packed_candidate);
  MinimumSummary summary;
  std::vector<CellIndex> invalid;
  for (int m = 0; m < nmb; ++m) {
    for (int k = indcs.ks; k <= indcs.ke; ++k) {
      for (int j = indcs.js; j <= indcs.je; ++j) {
        for (int i = indcs.is; i <= indcs.ie; ++i) {
          const CellIndex cell{m, k, j, i};
          const Real value = candidate(m, k, j, i);
          ++summary.count;
          if (!std::isfinite(value)) {
            ++summary.nonfinite;
            invalid.push_back(cell);
          } else {
            if (!(value > 0.0)) {
              ++summary.nonpositive;
              invalid.push_back(cell);
            }
            if (value < summary.minimum) {
              summary.minimum = value;
              summary.cell = cell;
            }
          }
        }
      }
    }
  }
  const auto minimum_xy = FineCoordinates(pack_, summary.cell);
  const int minimum_gid = pack_->pmb->mb_gid.h_view(summary.cell.m);
  {
    std::ofstream output(output_root_ + "/preupdate_candidate_minima.csv",
                         std::ios::app);
    output << std::setprecision(17) << mesh->ncycle << ','
           << amr_history::HexReal(mesh->time) << ',' << stage << ','
           << TreeChecksum(mesh) << ',' << gamma0 << ',' << gamma1 << ','
           << beta_stage << ',' << mesh->dt << ','
           << amr_history::HexReal(mesh->dt) << ',' << summary.minimum << ','
           << minimum_gid << ',' << summary.cell.i << ',' << summary.cell.j << ','
           << minimum_xy.first << ',' << minimum_xy.second << ','
           << summary.nonpositive << ',' << summary.nonfinite << '\n';
    output.flush();
    if (!output) DiagnosticFatal("failed to write pre-update candidate minima");
  }
  if (config_.control_target_trace) {
    const HostSnapshot trace_old = CaptureChi(
        pack_, pack_->pz4c->u0, "chi_control_trace_u0");
    const HostSnapshot trace_acc = CaptureChi(
        pack_, pack_->pz4c->u1, "chi_control_trace_u1");
    const HostSnapshot trace_rhs = CaptureChi(
        pack_, pack_->pz4c->u_rhs, "chi_control_trace_rhs");
    auto trace_terms = Kokkos::create_mirror_view_and_copy(
        HostMemSpace(), pack_->pz4c->chi_provenance_terms);
    std::ofstream trace(output_root_ + "/control_target_stage_decomposition.csv",
                        std::ios::app);
    for (int m = 0; m < nmb; ++m) {
      const int gid = pack_->pmb->mb_gid.h_view(m);
      if (gid != 35 && gid != 60) continue;
      for (int j = indcs.js; j <= indcs.je; ++j) {
        for (int i = indcs.is; i <= indcs.ie; ++i) {
          const CellIndex cell{m, indcs.ks, j, i};
          const auto xy = FineCoordinates(pack_, cell);
          if (std::fabs(xy.first - 5.138671875) > 0.25 * pack_->pmb->mb_size.h_view(m).dx1 ||
              std::fabs(std::fabs(xy.second) - 0.001953125) >
                  0.25 * pack_->pmb->mb_size.h_view(m).dx2) {
            continue;
          }
          const Real old = trace_old.at(cell);
          const Real acc = trace_acc.at(cell);
          const Real stored_rhs = trace_rhs.at(cell);
          const auto arithmetic = EvaluateChiRKCandidate(
              gamma0, gamma1, beta_dt, old, acc, stored_rhs);
          trace << std::setprecision(17) << mesh->ncycle << ','
                << amr_history::HexReal(mesh->time) << ',' << stage << ',' << gid
                << ',' << i << ',' << j << ',' << xy.first << ',' << xy.second
                << ',' << gamma0 << ',' << gamma1 << ',' << beta_stage << ','
                << mesh->dt << ',' << old << ',' << acc << ','
                << arithmetic.affine_base;
          for (int term : {chi_adv_rho, chi_adv_z, chi_adv_y,
                           chi_lie_divergence, chi_curvature_source,
                           chi_ko_rho, chi_ko_z, chi_ko_y}) {
            trace << ',' << trace_terms(m, term, cell.k, j, i);
          }
          trace << ',' << stored_rhs << ',' << arithmetic.rhs_increment << ','
                << arithmetic.candidate << '\n';
        }
      }
    }
    trace.flush();
    if (!trace) DiagnosticFatal("failed to write control target trace");
  }
  if (invalid.empty()) return;

  // The first invalid candidate is the bounded trigger.  Everything below is
  // read-only evidence gathered before ExpRKUpdate is allowed to consume it.
  const HostSnapshot old_state = CaptureChi(
      pack_, pack_->pz4c->u0, "chi_provenance_preupdate_u0");
  const HostSnapshot accumulator = CaptureChi(
      pack_, pack_->pz4c->u1, "chi_provenance_preupdate_u1");
  const HostSnapshot rhs_state = CaptureChi(
      pack_, pack_->pz4c->u_rhs, "chi_provenance_preupdate_rhs");
  auto host_all = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pack_->pz4c->u0);
  auto host_mu = Kokkos::create_mirror_view_and_copy(
      HostMemSpace(), pack_->pz4c->u_telegraph_mu);
  auto host_terms = Kokkos::create_mirror_view_and_copy(
      HostMemSpace(), pack_->pz4c->chi_provenance_terms);

  std::ofstream accumulator_output(output_root_ + "/rk_accumulator_audit.csv");
  accumulator_output << "cycle,rk_stage,gid,i,j,rho,z,delta,u0_before_copy,"
                        "u1_before_copy,expected_u1_after,u1_after_copy,"
                        "copy_ulp_difference,copy_exact,u0_after_copy\n";
  std::ofstream rhs_output(output_root_ + "/chi_rhs_term_decomposition.csv");
  rhs_output << "cycle,rk_stage,gid,i,j,rho,z,adv_rho,adv_z,adv_y,"
                "lie_divergence,adv_directional_sum,adv_total_production,"
                "adv_directional_residual,curvature_source,rhs_before_ko,"
                "ko_rho,rhs_after_ko_rho,ko_z,rhs_after_ko_z,ko_y,"
                "rhs_after_ko_y,rhs_after_ko,stored_rhs_after_boundary,"
                "boundary_rhs_change,rhs_sum_residual,rhs_sum_ulp\n";
  std::ofstream counterfactual_output(
      output_root_ + "/chi_candidate_counterfactuals.csv");
  counterfactual_output << "cycle,rk_stage,gid,i,j,rho,z,chi_old,"
                           "chi_accumulator,gamma0,gamma1,beta_stage,dt,"
                           "affine_base,rhs_increment,candidate_base,"
                           "candidate_adv,candidate_K,candidate_KO,"
                           "candidate_no_K,candidate_no_adv,candidate_no_KO,"
                           "candidate_full,candidate_device,long_double_candidate,"
                           "host_device_ulp\n";
  std::ofstream stiffness_output(output_root_ + "/local_stiffness_metrics.csv");
  stiffness_output << "cycle,rk_stage,gid,i,j,rho,z,alpha,lapse,Khat,Theta,K,"
                      "chi_guarded,beta_rho,beta_z,beta_y,telegraph_mu,"
                      "telegraph_tau,dx_rho,dx_z,S_K,S_adv_rho,S_adv_z,"
                      "S_adv_total,S_tel,relative_RHS_increment,"
                      "positivity_step_fraction\n";
  std::ofstream stencil_output(output_root_ + "/chi_stencil_values.csv");
  stencil_output << "target_gid,direction,offset,local_gid,local_k,local_j,local_i,"
                    "rho,z,storage,value\n";
  std::ofstream owner_output(output_root_ + "/chi_stencil_owner_comparison.csv");
  owner_output << "target_gid,direction,offset,local_gid,local_k,local_j,local_i,"
                  "rho,z,storage,local_value,owner_found,owner_gid,owner_k,"
                  "owner_j,owner_i,owner_value,same_bits,ulp_difference,level\n";
  std::ofstream derivative_output(output_root_ + "/derivative_order_comparison.csv");
  derivative_output << "target_gid,direction,velocity,inverse_dx,order,"
                       "block_local,stitched_owner,difference,ulp_difference\n";
  std::ofstream ko_output(output_root_ + "/ko_directional_comparison.csv");
  ko_output << "target_gid,direction,inverse_dx,diss,production_term,"
               "block_local_shadow,stitched_owner_shadow,block_production_ulp,"
               "stitched_production_ulp\n";

  bool accumulator_valid = true;
  bool copy_valid = true;
  bool stencil_valid = true;
  std::string primary_classification = "NOT_ESTABLISHED";
  const Real delta = driver->delta[stage - 1];
  const Real tiny = std::numeric_limits<Real>::epsilon();
  for (const CellIndex &cell : invalid) {
    const int gid = pack_->pmb->mb_gid.h_view(cell.m);
    const auto xy = FineCoordinates(pack_, cell);
    const Real u0_before = u0_before_copy_->at(cell);
    const Real u1_before = u1_before_copy_->at(cell);
    const Real expected_u1 = stage == 1 ? u0_before : u1_before + delta * u0_before;
    const Real u1_after = u1_after_copy_->at(cell);
    const long long copy_ulp = UlpDistance(expected_u1, u1_after);
    const bool copy_exact = SameBits(expected_u1, u1_after);
    copy_valid = copy_valid && copy_exact;
    accumulator_valid = accumulator_valid && std::isfinite(u1_before) &&
                        std::isfinite(u1_after);
    accumulator_output << std::setprecision(17) << mesh->ncycle << ',' << stage
      << ',' << gid << ',' << cell.i << ',' << cell.j << ',' << xy.first << ','
      << xy.second << ',' << delta << ',' << u0_before << ',' << u1_before << ','
      << expected_u1 << ',' << u1_after << ',' << copy_ulp << ','
      << (copy_exact ? "true" : "false") << ',' << old_state.at(cell) << '\n';

    const Real adv_rho = host_terms(cell.m, chi_adv_rho, cell.k, cell.j, cell.i);
    const Real adv_z = host_terms(cell.m, chi_adv_z, cell.k, cell.j, cell.i);
    const Real adv_y = host_terms(cell.m, chi_adv_y, cell.k, cell.j, cell.i);
    const Real lie_div = host_terms(cell.m, chi_lie_divergence, cell.k, cell.j, cell.i);
    const Real adv_directional_sum = ((adv_rho + adv_z) + adv_y) + lie_div;
    const Real adv_total = host_terms(
        cell.m, chi_adv_total_production, cell.k, cell.j, cell.i);
    const Real curvature = host_terms(
        cell.m, chi_curvature_source, cell.k, cell.j, cell.i);
    const Real rhs_before_ko = host_terms(
        cell.m, chi_rhs_before_ko, cell.k, cell.j, cell.i);
    const Real ko_rho = host_terms(cell.m, chi_ko_rho, cell.k, cell.j, cell.i);
    const Real after_ko_rho = host_terms(
        cell.m, chi_rhs_after_ko_rho, cell.k, cell.j, cell.i);
    const Real ko_z = host_terms(cell.m, chi_ko_z, cell.k, cell.j, cell.i);
    const Real after_ko_z = host_terms(
        cell.m, chi_rhs_after_ko_z, cell.k, cell.j, cell.i);
    const Real ko_y = host_terms(cell.m, chi_ko_y, cell.k, cell.j, cell.i);
    const Real after_ko_y = host_terms(
        cell.m, chi_rhs_after_ko_y, cell.k, cell.j, cell.i);
    const Real after_ko = host_terms(
        cell.m, chi_rhs_after_ko, cell.k, cell.j, cell.i);
    const Real stored_rhs = rhs_state.at(cell);
    const Real production_sum = (((adv_total + curvature) + ko_rho) + ko_z) + ko_y;
    rhs_output << std::setprecision(17) << mesh->ncycle << ',' << stage << ','
      << gid << ',' << cell.i << ',' << cell.j << ',' << xy.first << ','
      << xy.second << ',' << adv_rho << ',' << adv_z << ',' << adv_y << ','
      << lie_div << ',' << adv_directional_sum << ',' << adv_total << ','
      << (adv_total - adv_directional_sum) << ',' << curvature << ',' << rhs_before_ko
      << ',' << ko_rho << ',' << after_ko_rho << ',' << ko_z << ',' << after_ko_z
      << ',' << ko_y << ',' << after_ko_y << ',' << after_ko << ',' << stored_rhs
      << ',' << (stored_rhs - after_ko) << ',' << (stored_rhs - production_sum)
      << ',' << UlpDistance(stored_rhs, production_sum) << '\n';

    const Real old = old_state.at(cell);
    const Real acc = accumulator.at(cell);
    const ChiRKArithmetic arithmetic = EvaluateChiRKCandidate(
        gamma0, gamma1, beta_dt, old, acc, stored_rhs);
    const Real delta_adv = beta_dt * adv_total;
    const Real delta_k = beta_dt * curvature;
    const Real delta_ko = beta_dt * (ko_rho + ko_z + ko_y);
    const Real candidate_adv = arithmetic.affine_base + delta_adv;
    const Real candidate_k = arithmetic.affine_base + delta_k;
    const Real candidate_ko = arithmetic.affine_base + delta_ko;
    const Real candidate_no_k = (arithmetic.affine_base + delta_adv) + delta_ko;
    const Real candidate_no_adv = (arithmetic.affine_base + delta_k) + delta_ko;
    const Real candidate_no_ko = (arithmetic.affine_base + delta_adv) + delta_k;
    const Real candidate_full = ((arithmetic.affine_base + delta_adv) + delta_k) +
                                delta_ko;
    const long double long_candidate =
        static_cast<long double>(gamma0) * static_cast<long double>(old) +
        static_cast<long double>(gamma1) * static_cast<long double>(acc) +
        static_cast<long double>(beta_dt) * static_cast<long double>(stored_rhs);
    counterfactual_output << std::setprecision(17) << mesh->ncycle << ',' << stage
      << ',' << gid << ',' << cell.i << ',' << cell.j << ',' << xy.first << ','
      << xy.second << ',' << old << ',' << acc << ',' << gamma0 << ',' << gamma1
      << ',' << beta_stage << ',' << mesh->dt << ',' << arithmetic.affine_base
      << ',' << arithmetic.rhs_increment << ',' << arithmetic.affine_base << ','
      << candidate_adv << ',' << candidate_k << ',' << candidate_ko << ','
      << candidate_no_k << ',' << candidate_no_adv << ',' << candidate_no_ko
      << ',' << candidate_full << ',' << candidate(cell.m, cell.k, cell.j, cell.i)
      << ',' << std::setprecision(21) << long_candidate << ','
      << UlpDistance(arithmetic.candidate,
                     candidate(cell.m, cell.k, cell.j, cell.i)) << '\n';

    const Real alpha = host_all(cell.m, Z4c::I_Z4C_ALPHA, cell.k, cell.j, cell.i);
    const Real khat = host_all(cell.m, Z4c::I_Z4C_KHAT, cell.k, cell.j, cell.i);
    const Real theta = host_all(cell.m, Z4c::I_Z4C_THETA, cell.k, cell.j, cell.i);
    const Real kval = khat + 2.0 * theta;
    const Real beta_rho = host_all(cell.m, Z4c::I_Z4C_BETAX, cell.k, cell.j, cell.i);
    const Real beta_z = host_all(cell.m, Z4c::I_Z4C_BETAY, cell.k, cell.j, cell.i);
    const Real beta_y = host_all(cell.m, Z4c::I_Z4C_BETAZ, cell.k, cell.j, cell.i);
    const Real mu = host_mu(cell.m, 0, cell.k, cell.j, cell.i);
    const auto &size = pack_->pmb->mb_size.h_view(cell.m);
    const Real s_k = mesh->dt * std::fabs((2.0 / 3.0) * alpha * kval);
    const Real s_adv_rho = std::fabs(beta_rho) * mesh->dt / size.dx1;
    const Real s_adv_z = std::fabs(beta_z) * mesh->dt / size.dx2;
    const Real s_tel = mesh->dt * mu / pack_->pz4c->opt.telegraph_tau;
    const Real denominator = std::max(
        std::fabs(arithmetic.affine_base),
        tiny * std::max({Real{1.0}, std::fabs(old), std::fabs(acc)}));
    const Real positivity_fraction = arithmetic.affine_base > 0.0 && stored_rhs < 0.0
        ? arithmetic.affine_base / (-beta_dt * stored_rhs)
        : std::numeric_limits<Real>::quiet_NaN();
    stiffness_output << std::setprecision(17) << mesh->ncycle << ',' << stage
      << ',' << gid << ',' << cell.i << ',' << cell.j << ',' << xy.first << ','
      << xy.second << ',' << alpha << ',' << alpha << ',' << khat << ',' << theta
      << ',' << kval << ',' << old << ',' << beta_rho << ',' << beta_z << ','
      << beta_y << ',' << mu << ',' << pack_->pz4c->opt.telegraph_tau << ','
      << size.dx1 << ',' << size.dx2 << ',' << s_k << ',' << s_adv_rho << ','
      << s_adv_z << ',' << (s_adv_rho + s_adv_z) << ',' << s_tel << ','
      << std::fabs(arithmetic.rhs_increment) / denominator << ','
      << positivity_fraction << '\n';

    for (int direction = 0; direction < 2; ++direction) {
      std::array<Real, 9> local_values{};
      std::array<Real, 9> owner_values{};
      bool direction_valid = true;
      const Real inverse_dx = direction == 0 ? 1.0 / size.dx1 : 1.0 / size.dx2;
      const Real velocity = direction == 0 ? beta_rho : beta_z;
      for (int offset = -4; offset <= 4; ++offset) {
        CellIndex local = cell;
        if (direction == 0) local.i += offset;
        else local.j += offset;
        const auto local_xy = FineCoordinates(pack_, local);
        const Real local_value = old_state.at(local);
        const OwnerCell owner = FindActiveOwner(
            pack_, pack_->pmb->mb_lev.h_view(cell.m), local_xy.first, local_xy.second);
        const Real owner_value = owner.found ? old_state.at(owner.cell)
                                             : std::numeric_limits<Real>::quiet_NaN();
        local_values[offset + 4] = local_value;
        owner_values[offset + 4] = owner_value;
        const bool same = owner.found && SameBits(local_value, owner_value);
        if (std::string(StorageClass(indcs, local)) == "ghost") {
          direction_valid = direction_valid && same;
        }
        stencil_output << std::setprecision(17) << gid << ','
          << (direction == 0 ? "rho" : "z") << ',' << offset << ',' << gid
          << ',' << local.k << ',' << local.j << ',' << local.i << ','
          << local_xy.first << ',' << local_xy.second << ','
          << StorageClass(indcs, local) << ',' << local_value << '\n';
        owner_output << std::setprecision(17) << gid << ','
          << (direction == 0 ? "rho" : "z") << ',' << offset << ',' << gid
          << ',' << local.k << ',' << local.j << ',' << local.i << ','
          << local_xy.first << ',' << local_xy.second << ','
          << StorageClass(indcs, local) << ',' << local_value << ','
          << (owner.found ? "true" : "false") << ',' << owner.gid << ','
          << owner.cell.k << ',' << owner.cell.j << ',' << owner.cell.i << ','
          << owner_value << ',' << (same ? "true" : "false") << ','
          << (owner.found ? UlpDistance(local_value, owner_value) : -1) << ','
          << pack_->pmb->mb_lev.h_view(cell.m) << '\n';
      }
      stencil_valid = stencil_valid && direction_valid;
      for (const int order : {2, 4, 6}) {
        const Real block = HostAdvectiveDerivative(
            order, local_values, velocity, inverse_dx);
        const Real stitched = HostAdvectiveDerivative(
            order, owner_values, velocity, inverse_dx);
        derivative_output << std::setprecision(17) << gid << ','
          << (direction == 0 ? "rho" : "z") << ',' << velocity << ','
          << inverse_dx << ',' << order << ',' << block << ',' << stitched << ','
          << (block - stitched) << ',' << UlpDistance(block, stitched) << '\n';
      }
      const Real production_ko = direction == 0 ? ko_rho : ko_z;
      const Real block_ko = HostKODissipationO6(
          local_values, inverse_dx, pack_->pz4c->diss);
      const Real stitched_ko = HostKODissipationO6(
          owner_values, inverse_dx, pack_->pz4c->diss);
      ko_output << std::setprecision(17) << gid << ','
        << (direction == 0 ? "rho" : "z") << ',' << inverse_dx << ','
        << pack_->pz4c->diss << ',' << production_ko << ',' << block_ko << ','
        << stitched_ko << ',' << UlpDistance(block_ko, production_ko) << ','
        << UlpDistance(stitched_ko, production_ko) << '\n';
    }
    const std::string classification = PhaseClassification(
        arithmetic.affine_base, arithmetic.candidate, candidate_no_adv,
        candidate_no_k, candidate_no_ko, accumulator_valid, copy_valid,
        stencil_valid);
    if (primary_classification == "NOT_ESTABLISHED") {
      primary_classification = classification;
    } else if (primary_classification != classification) {
      primary_classification = "MIXED_TEMPORAL_SPATIAL_FAILURE";
    }
  }

  // Preserve a compact, stitched-source-capable active patch for optional bounded
  // high-frequency analysis.  Only the blocks containing invalid candidates are dumped.
  std::set<int> target_blocks;
  for (const auto &cell : invalid) target_blocks.insert(cell.m);
  std::ofstream patch(output_root_ + "/local_patch_all_fields.csv");
  patch << "gid,level,lx1,lx2,k,j,i,rho,z";
  for (int n = 0; n < Z4c::nz4c; ++n) patch << ',' << Z4c::Z4c_names[n];
  patch << '\n';
  for (const int m : target_blocks) {
    const int gid = pack_->pmb->mb_gid.h_view(m);
    const auto &location = mesh->lloc_eachmb[gid];
    for (int j = indcs.js; j <= indcs.je; ++j) {
      for (int i = indcs.is; i <= indcs.ie; ++i) {
        const CellIndex cell{m, indcs.ks, j, i};
        const auto xy = FineCoordinates(pack_, cell);
        patch << std::setprecision(17) << gid << ','
              << location.level - mesh->root_level << ',' << location.lx1 << ','
              << location.lx2 << ',' << cell.k << ',' << j << ',' << i << ','
              << xy.first << ',' << xy.second;
        for (int n = 0; n < Z4c::nz4c; ++n) {
          patch << ',' << host_all(m, n, cell.k, j, i);
        }
        patch << '\n';
      }
    }
  }
  {
    std::ofstream output(output_root_ + "/rk_stage3_candidate_summary.json");
    output << std::setprecision(17)
      << "{\"schema\":\"athenak_chi_rk_candidate_v1\",\"classification\":\""
      << primary_classification << "\",\"cycle\":" << mesh->ncycle
      << ",\"stage\":" << stage << ",\"time_hex\":\""
      << amr_history::HexReal(mesh->time) << "\",\"tree_checksum\":\""
      << TreeChecksum(mesh) << "\",\"gamma0\":" << gamma0
      << ",\"gamma1\":" << gamma1 << ",\"beta_stage\":" << beta_stage
      << ",\"delta\":" << delta << ",\"dt\":" << mesh->dt
      << ",\"dt_hex\":\"" << amr_history::HexReal(mesh->dt)
      << "\",\"invalid_candidates\":" << invalid.size()
      << ",\"copy_exact\":" << (copy_valid ? "true" : "false")
      << ",\"stencil_owner_exact\":" << (stencil_valid ? "true" : "false")
      << "}\n";
  }
  {
    std::ofstream output(output_root_ + "/phase1_disposition.json");
    output << "{\"classification\":\"" << primary_classification
           << "\",\"cycle\":" << mesh->ncycle << ",\"stage\":" << stage
           << ",\"time_hex\":\"" << amr_history::HexReal(mesh->time)
           << "\",\"invalid_candidates\":" << invalid.size() << "}\n";
  }
  // ATHENA_ERROR aborts without unwinding these local streams.  Flush every
  // detailed artifact explicitly before the fail-closed stop.
  accumulator_output.flush();
  rhs_output.flush();
  counterfactual_output.flush();
  stiffness_output.flush();
  stencil_output.flush();
  owner_output.flush();
  derivative_output.flush();
  ko_output.flush();
  patch.flush();
  if (!accumulator_output || !rhs_output || !counterfactual_output ||
      !stiffness_output || !stencil_output || !owner_output ||
      !derivative_output || !ko_output || !patch) {
    DiagnosticFatal("failed to flush complete pre-update diagnostic evidence");
  }
  DiagnosticFatal("pre-update chi candidate is nonpositive or nonfinite; exact audit complete");
}

void ChiParentProvenanceRuntime::RecordCheckpoint(
    const ChiProvenanceCheckpoint checkpoint, const int stage,
    MeshBoundaryValuesCC *boundary) {
  Mesh *mesh = pack_->pmesh;
  if (mesh->time < config_.start_time) return;
  if (checkpoint == ChiProvenanceCheckpoint::s0_after_rk) {
    cycle_ = mesh->ncycle;
    stage_ = stage;
    delete fine_s0_;
    delete coarse_s1_;
    delete coarse_s2_;
    delete coarse_s3_;
    fine_s0_ = new HostSnapshot(CaptureFineChi(pack_));
    coarse_s1_ = nullptr;
    coarse_s2_ = nullptr;
    coarse_s3_ = nullptr;
  } else if (fine_s0_ == nullptr) {
    // Driver initialization performs boundary/restriction tasks before the
    // first evolution-stage RK update. They are outside the S0--S4 audit.
    return;
  } else if (cycle_ != mesh->ncycle) {
    // Regridding and post-step boundary initialization can run after ncycle is
    // incremented but before the next cycle's first RK update.
    return;
  } else if (stage_ != stage) {
    DiagnosticFatal("checkpoint sequence does not begin at S0 for this RK stage");
  }

  // No task between S0 and S4 changes active u0. Reuse the S0 host image so
  // the diagnostic does not add four redundant device-to-host copies per RK stage.
  HostSnapshot fine = *fine_s0_;
  HostSnapshot coarse = CaptureCoarseChi(pack_);
  if (checkpoint == ChiProvenanceCheckpoint::s1_after_restriction) {
    delete coarse_s1_;
    coarse_s1_ = new HostSnapshot(coarse);
  } else if (checkpoint == ChiProvenanceCheckpoint::s2_after_receive) {
    delete coarse_s2_;
    coarse_s2_ = new HostSnapshot(coarse);
  } else if (checkpoint == ChiProvenanceCheckpoint::s3_after_boundary) {
    delete coarse_s3_;
    coarse_s3_ = new HostSnapshot(coarse);
  }

  const MinimumSummary active = ActiveFineMinimum(pack_, fine);
  const auto consumed = ConsumedCoarseCells(pack_, boundary);
  const MinimumSummary coarse_min = ConsumedCoarseMinimum(consumed, coarse);
  const auto active_xy = FineCoordinates(pack_, active.cell);
  const auto coarse_xy = CoarseCoordinates(pack_, coarse_min.cell);
  const int active_gid = pack_->pmb->mb_gid.h_view(active.cell.m);
  const int coarse_gid = pack_->pmb->mb_gid.h_view(coarse_min.cell.m);
  std::ofstream output(output_root_ + "/phase1_stage_minima.csv",
                       std::ios::app);
  output << std::setprecision(17) << mesh->ncycle << ','
         << amr_history::HexReal(mesh->time) << ',' << stage << ','
         << CheckpointName(checkpoint) << ',' << TreeChecksum(mesh) << ','
         << active.minimum << ',' << active_gid << ',' << active.cell.i << ','
         << active.cell.j << ',' << active_xy.first << ',' << active_xy.second
         << ',' << active.nonpositive << ',' << active.nonfinite << ','
         << coarse_min.minimum << ',' << coarse_gid << ',' << coarse_min.cell.i
         << ',' << coarse_min.cell.j << ',' << coarse_xy.first << ','
         << coarse_xy.second << ',' << coarse_min.nonpositive << ','
         << coarse_min.nonfinite << ',' << coarse_min.count << '\n';
  output.flush();
  if (!output) DiagnosticFatal("failed to write stage minima");
  if (checkpoint == ChiProvenanceCheckpoint::s0_after_rk &&
      (active.nonpositive != 0 || active.nonfinite != 0)) {
    std::ofstream cells(output_root_ + "/active_fine_failure.csv");
    cells << "cycle,time_hex,rk_stage,tree_checksum,gid,relative_level,lx1,lx2,"
             "lx3,local_k,local_j,local_i,rho,z,value,classification\n";
    const auto &indcs = mesh->mb_indcs;
    for (int m = 0; m < fine.nmb; ++m) {
      const int gid = pack_->pmb->mb_gid.h_view(m);
      const auto &location = mesh->lloc_eachmb[gid];
      for (int k = indcs.ks; k <= indcs.ke; ++k) {
        for (int j = indcs.js; j <= indcs.je; ++j) {
          for (int i = indcs.is; i <= indcs.ie; ++i) {
            const CellIndex cell{m, k, j, i};
            const Real value = fine.at(cell);
            if (!InvalidChi(value)) continue;
            const auto xy = FineCoordinates(pack_, cell);
            cells << std::setprecision(17) << mesh->ncycle << ','
                  << amr_history::HexReal(mesh->time) << ',' << stage << ','
                  << TreeChecksum(mesh) << ',' << gid << ','
                  << location.level - mesh->root_level << ',' << location.lx1
                  << ',' << location.lx2 << ',' << location.lx3 << ',' << k
                  << ',' << j << ',' << i << ',' << xy.first << ','
                  << xy.second << ',' << value << ',' << Classification(value)
                  << '\n';
          }
        }
      }
    }
    cells.flush();
    if (!cells) DiagnosticFatal("failed to write active fine failure cells");
    std::ofstream disposition(output_root_ + "/phase1_disposition.json");
    disposition << "{\"classification\":\"ACTIVE_FINE_CHI_FAILURE\","
                << "\"cycle\":" << mesh->ncycle << ",\"stage\":" << stage
                << ",\"time_hex\":\"" << amr_history::HexReal(mesh->time)
                << "\",\"active_nonpositive\":" << active.nonpositive
                << ",\"active_nonfinite\":" << active.nonfinite << "}\n";
    disposition.flush();
    DiagnosticFatal("active fine chi first became invalid immediately after RK update");
  }
}

void ChiParentProvenanceRuntime::AnalyzeBoundaryFailure(
    MeshBoundaryValuesCC *boundary,
    const unsigned long long invalid_parent_stencils,
    const unsigned long long first_rejected_key) {
  if (pack_->pmesh->time < config_.start_time) return;
  if (fine_s0_ == nullptr || coarse_s1_ == nullptr || coarse_s2_ == nullptr ||
      coarse_s3_ == nullptr) {
    DiagnosticFatal("boundary failure lacks a complete S0--S3 snapshot sequence");
  }
  const HostSnapshot current = CaptureCoarseChi(pack_);
  const auto &indcs = pack_->pmesh->mb_indcs;
  const int stencil = pack_->pz4c->opt.fd_stencil;
  const int lower = stencil / 2;
  const int nk = indcs.nx3 == 1 ? 1 : stencil + 1;
  auto &neighbors = pack_->pmb->nghbr;
  auto &levels = pack_->pmb->mb_lev;

  struct InvalidRecord {
    CellIndex cell;
    Real value = 0.0;
    WriterInfo writer;
    std::uint64_t multiplicity = 0;
    int storage_level = -1;
    long long global_i = 0;
    long long global_j = 0;
    Real rho = 0.0;
    Real z = 0.0;
  };
  using GlobalKey = std::tuple<int, long long, long long, int>;
  std::map<GlobalKey, InvalidRecord> invalid;
  std::uint64_t rejected_targets = 0;

  const int first_gid = static_cast<int>(first_rejected_key >> 36);
  const int first_slot = static_cast<int>((first_rejected_key >> 30) & 0x3fULL);
  const int first_k = static_cast<int>((first_rejected_key >> 20) & 0x3ffULL);
  const int first_j = static_cast<int>((first_rejected_key >> 10) & 0x3ffULL);
  const int first_i = static_cast<int>(first_rejected_key & 0x3ffULL);
  std::ofstream first(output_root_ + "/first_invalid_parent_stencil.csv");
  first << "target_gid,neighbor_slot,target_k,target_j,target_i,stencil_k,"
           "stencil_j,stencil_i,value,classification,writer,first_invalid_phase,"
           "owner_gid,owner_storage,rho,z\n";

  const Real root_dx1 = (pack_->pmesh->mesh_size.x1max -
      pack_->pmesh->mesh_size.x1min) / pack_->pmesh->mesh_indcs.nx1;
  const Real root_dx2 = (pack_->pmesh->mesh_size.x2max -
      pack_->pmesh->mesh_size.x2min) / pack_->pmesh->mesh_indcs.nx2;

  for (int m = 0; m < pack_->nmb_thispack; ++m) {
    const int gid = pack_->pmb->mb_gid.h_view(m);
    const int relative_level = levels.h_view(m) - pack_->pmesh->root_level;
    for (int n = 0; n < pack_->pmb->nnghbr; ++n) {
      if (neighbors.h_view(m, n).gid < 0 ||
          neighbors.h_view(m, n).lev >= levels.h_view(m)) continue;
      const auto &range = boundary->recvbuf[n].iprol[0];
      for (int k = range.bks; k <= range.bke; ++k) {
        for (int j = range.bjs; j <= range.bje; ++j) {
          for (int i = range.bis; i <= range.bie; ++i) {
            std::vector<std::pair<CellIndex, Real>> bad;
            for (int kk = 0; kk < nk; ++kk) {
              const int ck = indcs.nx3 == 1 ? k : k - lower + kk;
              for (int jj = 0; jj <= stencil; ++jj) {
                for (int ii = 0; ii <= stencil; ++ii) {
                  const CellIndex cell{m, ck, j - lower + jj,
                                       i - lower + ii};
                  const Real value = current.at(cell);
                  if (InvalidChi(value)) bad.emplace_back(cell, value);
                  if (gid == first_gid && n == first_slot && k == first_k &&
                      j == first_j && i == first_i) {
                    const WriterInfo writer = IdentifyWriter(
                        pack_, boundary, cell, *coarse_s1_, *coarse_s2_,
                        *coarse_s3_, value);
                    const auto xy = CoarseCoordinates(pack_, cell);
                    first << std::setprecision(17) << gid << ',' << n << ',' << k
                          << ',' << j << ',' << i << ',' << cell.k << ','
                          << cell.j << ',' << cell.i << ',' << value << ','
                          << Classification(value) << ','
                          << ChiWriterProvenanceName(writer.writer) << ','
                          << writer.first_phase << ',' << writer.owner_gid << ','
                          << writer.owner_storage << ',' << xy.first << ','
                          << xy.second << '\n';
                  }
                }
              }
            }
            if (bad.empty()) continue;
            ++rejected_targets;
            for (const auto &entry : bad) {
              const CellIndex &cell = entry.first;
              const Real value = entry.second;
              const WriterInfo writer = IdentifyWriter(
                  pack_, boundary, cell, *coarse_s1_, *coarse_s2_,
                  *coarse_s3_, value);
              const auto xy = CoarseCoordinates(pack_, cell);
              const int storage_level = relative_level - 1;
              const Real dx1 = std::ldexp(root_dx1, -storage_level);
              const Real dx2 = std::ldexp(root_dx2, -storage_level);
              const long long gi = std::llround(
                  (xy.first - pack_->pmesh->mesh_size.x1min) / dx1 - 0.5);
              const long long gj = std::llround(
                  (xy.second - pack_->pmesh->mesh_size.x2min) / dx2 - 0.5);
              const GlobalKey key{storage_level, gi, gj, writer.owner_gid};
              auto found = invalid.find(key);
              if (found == invalid.end()) {
                InvalidRecord record;
                record.cell = cell;
                record.value = value;
                record.writer = writer;
                record.multiplicity = 1;
                record.storage_level = storage_level;
                record.global_i = gi;
                record.global_j = gj;
                record.rho = xy.first;
                record.z = xy.second;
                invalid.emplace(key, record);
              } else {
                ++found->second.multiplicity;
              }
            }
          }
        }
      }
    }
  }
  first.flush();
  if (rejected_targets != invalid_parent_stencils) {
    DiagnosticFatal("offline rejected-target enumeration does not match device count");
  }

  std::ofstream unique(output_root_ + "/unique_invalid_coarse_cells.csv");
  unique << "storage_level,global_i,global_j,receiver_gid,local_k,local_j,local_i,"
            "rho,z,value,classification,writer,first_invalid_phase,owner_gid,"
            "owner_storage,multiplicity,axis_distance,receiver_block_edge_distance_cells,"
            "s1_value,s2_value,s3_value,s4_value\n";
  std::ofstream multiplicity(
      output_root_ + "/invalid_cell_consumption_multiplicity.csv");
  multiplicity << "storage_level,global_i,global_j,owner_gid,rejected_targets_consuming\n";
  std::ofstream candidates(output_root_ + "/restriction_candidate_comparison.csv");
  candidates << "storage_level,global_i,global_j,receiver_gid,local_j,local_i,"
                "stencil_class,all_fine_positive,fine_min,fine_max,production_s1,"
                "recomputed_high,convex_2x2_average,log_restriction_candidate,"
                "parity_folded_axis_candidate,fine_stencil_values,tensor_weights\n";

  std::map<ChiWriterProvenance, std::uint64_t> writer_counts;
  bool restriction_signature = false;
  bool communication_signature = false;
  bool boundary_signature = false;
  Real minimum_invalid = std::numeric_limits<Real>::infinity();
  const InvalidRecord *first_record = nullptr;
  for (const auto &item : invalid) {
    const auto &key = item.first;
    const InvalidRecord &record = item.second;
    ++writer_counts[record.writer.writer];
    if (first_record == nullptr || record.value < minimum_invalid) {
      first_record = &record;
      minimum_invalid = record.value;
    }
    const int gid = pack_->pmb->mb_gid.h_view(record.cell.m);
    const int edge_distance = std::min(
        std::min(std::abs(record.cell.i - indcs.cis),
                 std::abs(record.cell.i - indcs.cie)),
        std::min(std::abs(record.cell.j - indcs.cjs),
                 std::abs(record.cell.j - indcs.cje)));
    unique << std::setprecision(17) << record.storage_level << ','
           << record.global_i << ',' << record.global_j << ',' << gid << ','
           << record.cell.k << ',' << record.cell.j << ',' << record.cell.i << ','
           << record.rho << ',' << record.z << ',' << record.value << ','
           << Classification(record.value) << ','
           << ChiWriterProvenanceName(record.writer.writer) << ','
           << record.writer.first_phase << ',' << record.writer.owner_gid << ','
           << record.writer.owner_storage << ',' << record.multiplicity << ','
           << record.rho << ',' << edge_distance << ','
           << coarse_s1_->at(record.cell) << ',' << coarse_s2_->at(record.cell)
           << ',' << coarse_s3_->at(record.cell) << ',' << record.value << '\n';
    multiplicity << std::get<0>(key) << ',' << std::get<1>(key) << ','
                 << std::get<2>(key) << ',' << std::get<3>(key) << ','
                 << record.multiplicity << '\n';

    if (record.writer.writer >= ChiWriterProvenance::local_restriction_centered &&
        record.writer.writer <= ChiWriterProvenance::local_restriction_corner) {
      const RestrictionDiagnostic diagnostic =
          RestrictionCandidates(pack_, record.cell, *fine_s0_);
      if (diagnostic.available) {
        candidates << std::setprecision(17) << record.storage_level << ','
                   << record.global_i << ',' << record.global_j << ',' << gid << ','
                   << record.cell.j << ',' << record.cell.i << ','
                   << ChiWriterProvenanceName(record.writer.writer) << ','
                   << (diagnostic.all_fine_positive ? "true" : "false") << ','
                   << diagnostic.fine_min << ',' << diagnostic.fine_max << ','
                   << coarse_s1_->at(record.cell) << ',' << diagnostic.high << ','
                   << diagnostic.average << ',' << diagnostic.log_candidate << ','
                   << diagnostic.parity_centered << ','
                   << Join(diagnostic.fine_values) << ',' << Join(diagnostic.weights)
                   << '\n';
        if (diagnostic.all_fine_positive && InvalidChi(diagnostic.high) &&
            !InvalidChi(diagnostic.average)) {
          restriction_signature = true;
        }
      }
    } else if (record.writer.writer ==
                   ChiWriterProvenance::same_level_owner_receive ||
               record.writer.writer ==
                   ChiWriterProvenance::coarser_neighbor_receive) {
      communication_signature = true;
    } else if (record.writer.writer == ChiWriterProvenance::axis_boundary_fill ||
               record.writer.writer ==
                   ChiWriterProvenance::outer_physical_boundary_fill) {
      boundary_signature = true;
    }
  }
  unique.flush();
  multiplicity.flush();
  candidates.flush();

  std::string disposition = "NOT_ESTABLISHED";
  const int mechanism_count = static_cast<int>(restriction_signature) +
      static_cast<int>(communication_signature) +
      static_cast<int>(boundary_signature);
  if (mechanism_count > 1) disposition = "MIXED_MECHANISM";
  else if (restriction_signature) {
    disposition = "HIGH_ORDER_RESTRICTION_BREAKS_POSITIVITY";
  } else if (communication_signature || boundary_signature) {
    disposition = "COMMUNICATION_OR_BOUNDARY_BREAKS_POSITIVITY";
  }

  if (first_record == nullptr) DiagnosticFatal("device reported no recoverable invalid cell");
  std::ofstream first_json(output_root_ + "/first_invalid_coarse_cell.json");
  first_json << std::setprecision(17)
             << "{\"schema\":\"athenak_z4c_chi_parent_provenance_v1\","
             << "\"phase1_classification\":\"" << disposition << "\","
             << "\"cycle\":" << pack_->pmesh->ncycle << ",\"stage\":"
             << stage_ << ",\"time_hex\":\""
             << amr_history::HexReal(pack_->pmesh->time) << "\","
             << "\"device_rejected_parent_stencils\":"
             << invalid_parent_stencils << ",\"enumerated_rejected_targets\":"
             << rejected_targets << ",\"unique_invalid_coarse_cells\":"
             << invalid.size() << ",\"minimum_invalid_value\":"
             << JsonReal(minimum_invalid) << ",\"first_writer\":\""
             << ChiWriterProvenanceName(first_record->writer.writer) << "\","
             << "\"first_invalid_phase\":\""
             << first_record->writer.first_phase << "\",\"rho\":"
             << first_record->rho << ",\"z\":" << first_record->z
             << ",\"writer_counts\":{";
  bool first_count = true;
  for (const auto &count : writer_counts) {
    if (!first_count) first_json << ',';
    first_count = false;
    first_json << '\"' << ChiWriterProvenanceName(count.first) << "\":"
               << count.second;
  }
  first_json << "}}\n";
  first_json.flush();
  std::ofstream phase1(output_root_ + "/phase1_disposition.json");
  phase1 << "{\"classification\":\"" << disposition
         << "\",\"cycle\":" << pack_->pmesh->ncycle << ",\"stage\":"
         << stage_ << ",\"unique_invalid_coarse_cells\":" << invalid.size()
         << ",\"rejected_parent_stencils\":" << invalid_parent_stencils
         << "}\n";
  phase1.flush();
  std::cerr << "CHI_PARENT_PROVENANCE classification=" << disposition
            << " unique_invalid_coarse_cells=" << invalid.size()
            << " rejected_parent_stencils=" << invalid_parent_stencils
            << std::endl;
}

void ChiParentProvenanceRuntime::RecordShadowAMRRequests(
    const std::size_t next_event, const std::string &next_event_time_hex,
    const std::string &tree_checksum) {
  if (pack_->pmesh->time < config_.start_time) return;
  Mesh *mesh = pack_->pmesh;
  const int begin = mesh->gids_eachrank[global_variable::my_rank];
  bool any = false;
  for (int m = 0; m < pack_->nmb_thispack; ++m) {
    if (mesh->pmr->refine_flag.h_view(begin + m) != 0) any = true;
  }
  if (!any) return;
  const HostSnapshot fine = CaptureFineChi(pack_);
  const auto &indcs = mesh->mb_indcs;
  std::ofstream output(output_root_ + "/shadow_amr_requests.jsonl",
                       std::ios::app);
  for (int m = 0; m < pack_->nmb_thispack; ++m) {
    const int gid = begin + m;
    const int flag = mesh->pmr->refine_flag.h_view(gid);
    if (flag == 0) continue;
    Real dmax = 0.0;
    for (int j = indcs.js; j <= indcs.je; ++j) {
      for (int i = indcs.is; i <= indcs.ie; ++i) {
        const Real di = fine.at({m, indcs.ks, j, i + 1}) -
                        fine.at({m, indcs.ks, j, i - 1});
        const Real dj = fine.at({m, indcs.ks, j + 1, i}) -
                        fine.at({m, indcs.ks, j - 1, i});
        dmax = std::max(dmax, std::sqrt(di * di + dj * dj));
      }
    }
    const auto &loc = mesh->lloc_eachmb[gid];
    output << std::setprecision(17)
           << "{\"schema\":\"athenak_amr_shadow_request_v1\",\"cycle\":"
           << mesh->ncycle << ",\"actual_time_hex\":\""
           << amr_history::HexReal(mesh->time) << "\",\"next_authority_event\":"
           << next_event << ",\"next_authority_time_hex\":\""
           << next_event_time_hex << "\",\"tree_checksum\":\""
           << tree_checksum << "\",\"gid\":" << gid << ",\"level\":"
           << loc.level - mesh->root_level << ",\"lx1\":" << loc.lx1
           << ",\"lx2\":" << loc.lx2 << ",\"lx3\":" << loc.lx3
           << ",\"requested_flag\":" << flag << ",\"raw_dchi_max\":"
           << dmax << ",\"dchi_threshold\":"
           << pack_->pz4c->pamr->dchi_thresh << ",\"derefine_threshold\":"
           << pack_->pz4c->pamr->dchi_derefine_factor *
                  pack_->pz4c->pamr->dchi_thresh
           << "}\n";
  }
  output.flush();
  if (!output) DiagnosticFatal("failed to append shadow AMR requests");
}

void ChiParentProvenanceRuntime::RecordReplayAlignment(
    const std::size_t event, const std::string &authority_time_hex,
    const std::string &actual_time_hex, const double signed_difference,
    const long long ulp_difference, const bool preceding_timestep_clipped) {
  std::ofstream output(output_root_ + "/replay_time_alignment.jsonl",
                       std::ios::app);
  output << std::setprecision(17)
         << "{\"schema\":\"athenak_amr_replay_time_alignment_v1\","
         << "\"event\":" << event << ",\"authority_time_hex\":\""
         << authority_time_hex << "\",\"actual_mesh_time_hex\":\""
         << actual_time_hex << "\",\"signed_difference\":"
         << signed_difference << ",\"ulp_difference\":" << ulp_difference
         << ",\"preceding_timestep_clipped\":"
         << (preceding_timestep_clipped ? "true" : "false") << "}\n";
  output.flush();
  if (!output) DiagnosticFatal("failed to append replay time alignment");
}

}  // namespace z4c
