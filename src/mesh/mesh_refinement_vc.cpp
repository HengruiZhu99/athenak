//========================================================================================
// AthenaK astrophysical plasma code
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mesh_refinement_vc.cpp
//! \brief Native vertex-centered refinement and derefinement operations.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/vertex_amr.hpp"
#include "z4c/z4c.hpp"

namespace {

struct VCAuditMismatch {
  bool found = false;
  int slot = -1;
  int variable = -1;
  int k = -1;
  int j = -1;
  int i = -1;
  Real expected = 0.0;
  Real actual = 0.0;
  Real absolute = 0.0;
  unsigned long long ulp = 0;
};

struct VCAuditFamily {
  int old_gid = -1;
  int new_gid = -1;
  int source_m = -1;
  int destination_m = -1;
  std::vector<Real> oracle;
  bool a5_staging_matches = false;
  bool a5_destination_matches = false;
  bool a6_parent_matches = false;
  std::string oracle_hash;
  std::string a5_staging_hash;
  std::string a5_destination_hash;
  std::string a6_parent_hash;
};

struct VCDerefineSlotAudit {
  bool active = false;
  std::string path;
  int cycle = -1;
  Real time = 0.0;
  int first_old = 0;
  int first_new = 0;
  int old_count = 0;
  int new_count = 0;
  int nvar = 0;
  int is = 0;
  int ie = -1;
  int js = 0;
  int je = -1;
  int ks = 0;
  int ke = -1;
  std::vector<Real> pre_a;
  std::vector<Real> a5_a;
  std::vector<Real> a6_a;
  std::vector<int> oldtonew;
  std::vector<int> newtoold;
  std::vector<int> old_flags;
  std::vector<VCAuditFamily> families;
  std::vector<int> a5_modified_live_old_gids;
  std::vector<int> a6_bad_unaffected_old_gids;
  VCAuditMismatch first_a5_live_mismatch;
  VCAuditMismatch first_a6_unaffected_mismatch;
  VCAuditMismatch first_a6_parent_mismatch;
};

VCDerefineSlotAudit vc_slot_audit;

std::size_t AuditPointsPerSlot(const VCDerefineSlotAudit &audit) {
  return static_cast<std::size_t>(audit.nvar) * (audit.ke - audit.ks + 1) *
      (audit.je - audit.js + 1) * (audit.ie - audit.is + 1);
}

std::size_t AuditIndex(const VCDerefineSlotAudit &audit, const int slot,
                       const int variable, const int k, const int j,
                       const int i) {
  const std::size_t ni = audit.ie - audit.is + 1;
  const std::size_t nj = audit.je - audit.js + 1;
  const std::size_t nk = audit.ke - audit.ks + 1;
  return (((static_cast<std::size_t>(slot) * audit.nvar + variable) * nk +
           (k - audit.ks)) * nj + (j - audit.js)) * ni + (i - audit.is);
}

template <typename View>
std::vector<Real> CaptureAuditActive(const View &view,
                                     const VCDerefineSlotAudit &audit,
                                     const int slots) {
  Kokkos::fence();
  const auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), view);
  std::vector<Real> values(static_cast<std::size_t>(slots) *
                           AuditPointsPerSlot(audit));
  for (int slot = 0; slot < slots; ++slot) {
    for (int variable = 0; variable < audit.nvar; ++variable) {
      for (int k = audit.ks; k <= audit.ke; ++k) {
        for (int j = audit.js; j <= audit.je; ++j) {
          for (int i = audit.is; i <= audit.ie; ++i) {
            values[AuditIndex(audit, slot, variable, k, j, i)] =
                host(slot, variable, k, j, i);
          }
        }
      }
    }
  }
  return values;
}

std::string AuditHash(const Real *values, const std::size_t count) {
  std::uint64_t hash = 1469598103934665603ULL;
  const auto *bytes = reinterpret_cast<const unsigned char *>(values);
  for (std::size_t index = 0; index < count * sizeof(Real); ++index) {
    hash ^= bytes[index];
    hash *= 1099511628211ULL;
  }
  std::ostringstream result;
  result << std::hex << std::setfill('0') << std::setw(16) << hash;
  return result.str();
}

std::string AuditSlotHash(const std::vector<Real> &values,
                          const VCDerefineSlotAudit &audit, const int slot) {
  const std::size_t count = AuditPointsPerSlot(audit);
  return AuditHash(values.data() + static_cast<std::size_t>(slot) * count, count);
}

unsigned long long AuditULPDistance(const Real left, const Real right) {
  if (!std::isfinite(left) || !std::isfinite(right)) {
    return std::numeric_limits<unsigned long long>::max();
  }
  if constexpr (sizeof(Real) == sizeof(std::uint64_t)) {
    std::int64_t a = 0, b = 0;
    std::memcpy(&a, &left, sizeof(Real));
    std::memcpy(&b, &right, sizeof(Real));
    if (a < 0) a = std::numeric_limits<std::int64_t>::min() - a;
    if (b < 0) b = std::numeric_limits<std::int64_t>::min() - b;
    const auto ua = static_cast<std::uint64_t>(a);
    const auto ub = static_cast<std::uint64_t>(b);
    return ua >= ub ? ua - ub : ub - ua;
  } else {
    std::int32_t a = 0, b = 0;
    std::memcpy(&a, &left, sizeof(Real));
    std::memcpy(&b, &right, sizeof(Real));
    if (a < 0) a = std::numeric_limits<std::int32_t>::min() - a;
    if (b < 0) b = std::numeric_limits<std::int32_t>::min() - b;
    const auto ua = static_cast<std::uint32_t>(a);
    const auto ub = static_cast<std::uint32_t>(b);
    return ua >= ub ? ua - ub : ub - ua;
  }
}

VCAuditMismatch CompareAuditSlot(const std::vector<Real> &expected,
                                 const int expected_slot,
                                 const std::vector<Real> &actual,
                                 const int actual_slot,
                                 const VCDerefineSlotAudit &audit) {
  VCAuditMismatch mismatch;
  mismatch.slot = actual_slot;
  for (int variable = 0; variable < audit.nvar; ++variable) {
    for (int k = audit.ks; k <= audit.ke; ++k) {
      for (int j = audit.js; j <= audit.je; ++j) {
        for (int i = audit.is; i <= audit.ie; ++i) {
          const Real lhs = expected[AuditIndex(
              audit, expected_slot, variable, k, j, i)];
          const Real rhs = actual[AuditIndex(
              audit, actual_slot, variable, k, j, i)];
          if (std::memcmp(&lhs, &rhs, sizeof(Real)) != 0) {
            mismatch.found = true;
            mismatch.variable = variable;
            mismatch.k = k;
            mismatch.j = j;
            mismatch.i = i;
            mismatch.expected = lhs;
            mismatch.actual = rhs;
            mismatch.absolute = std::fabs(lhs - rhs);
            mismatch.ulp = AuditULPDistance(lhs, rhs);
            return mismatch;
          }
        }
      }
    }
  }
  return mismatch;
}

VCAuditMismatch CompareAuditOracle(const std::vector<Real> &oracle,
                                   const std::vector<Real> &actual,
                                   const int actual_slot,
                                   const VCDerefineSlotAudit &audit) {
  return CompareAuditSlot(oracle, 0, actual, actual_slot, audit);
}

void WriteMismatchJSON(std::ostream &output, const VCAuditMismatch &mismatch) {
  output << "{\"found\":" << (mismatch.found ? "true" : "false")
         << ",\"slot\":" << mismatch.slot
         << ",\"variable\":" << mismatch.variable
         << ",\"k\":" << mismatch.k << ",\"j\":" << mismatch.j
         << ",\"i\":" << mismatch.i << ",\"expected\":"
         << std::setprecision(17) << mismatch.expected << ",\"actual\":"
         << mismatch.actual << ",\"absolute\":" << mismatch.absolute
         << ",\"ulp\":" << mismatch.ulp << "}";
}

enum class VCAMRDeviceError : long long {
  none = 0,
  new_gid_out_of_range = 1,
  old_gid_out_of_range = 2,
  local_meshblock_out_of_range = 3,
  active_bounds_out_of_range = 4,
  coarse_bounds_out_of_range = 5,
};

template <typename Record>
KOKKOS_INLINE_FUNCTION void RecordFirstVCAMRError(
    const Record &record, const VCAMRDeviceError error, const long long phase,
    const long long detail0, const long long detail1, const long long detail2,
    const long long detail3) {
  const long long code = static_cast<long long>(error);
  if (Kokkos::atomic_compare_exchange(&record(0), 0LL, code) == 0LL) {
    record(1) = phase;
    record(2) = detail0;
    record(3) = detail1;
    record(4) = detail2;
    record(5) = detail3;
  }
}

void CheckVCAMRDeviceError(const DvceArray1D<long long> &record,
                           const char *operation) {
  Kokkos::fence();
  const auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), record);
  if (host(0) == 0) return;
  std::cerr << "### FATAL ERROR: native VC AMR lifecycle validation failed operation="
            << operation << " code=" << host(0) << " phase=A" << host(1)
            << " details=" << host(2) << ',' << host(3) << ',' << host(4)
            << ',' << host(5) << std::endl;
#if MPI_PARALLEL_ENABLED
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
  std::exit(EXIT_FAILURE);
}

[[noreturn]] void AbortVC(const char *message) {
  if (global_variable::my_rank == 0) {
    std::cerr << "### FATAL ERROR: " << message << std::endl;
  }
#if MPI_PARALLEL_ENABLED
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
  std::exit(EXIT_FAILURE);
}

void BeginVCDerefineSlotAudit(
    Mesh *mesh, const int *oldtonew, const int *newtoold,
    const int *new_rank_eachmb, const int *new_gids_eachrank,
    const int *new_nmb_eachrank, const DualArray1D<int> &refine_flag,
    const DvceArray5D<Real> &a, const DvceArray5D<Real> &ca,
    const z4c::Z4cGridLayout &layout) {
  const char *path = std::getenv("ATHENA_Z4C_VC_DEREFINE_SLOT_AUDIT");
  vc_slot_audit = VCDerefineSlotAudit{};
  if (path == nullptr || path[0] == '\0') return;
  const int rank = global_variable::my_rank;
  const int first_old = mesh->gids_eachrank[rank];
  const int old_count = mesh->nmb_eachrank[rank];
  const int first_new = new_gids_eachrank[rank];
  const int new_count = new_nmb_eachrank[rank];
  const int last_old = first_old + old_count - 1;
  const int nleaf = mesh->three_d ? 8 : mesh->two_d ? 4 : 2;
  std::vector<int> family_old_gids;
  for (int old_gid = first_old; old_gid <= last_old; ++old_gid) {
    if (refine_flag.h_view(old_gid) >= -1) continue;
    const auto &location = mesh->lloc_eachmb[old_gid];
    if ((location.lx1 & 1) != 0 ||
        ((mesh->two_d || mesh->three_d) && (location.lx2 & 1) != 0) ||
        (mesh->three_d && (location.lx3 & 1) != 0)) {
      continue;
    }
    const int new_gid = oldtonew[old_gid];
    if (new_rank_eachmb[new_gid] != rank) continue;
    bool all_local = old_gid + nleaf - 1 <= last_old;
    for (int child = 0; child < nleaf && all_local; ++child) {
      all_local = mesh->rank_eachmb[old_gid + child] == rank;
    }
    if (all_local) family_old_gids.push_back(old_gid);
  }
  if (family_old_gids.size() < 2) return;

  auto &audit = vc_slot_audit;
  audit.active = true;
  audit.path = path;
  audit.cycle = mesh->ncycle;
  audit.time = mesh->time;
  audit.first_old = first_old;
  audit.first_new = first_new;
  audit.old_count = old_count;
  audit.new_count = new_count;
  audit.nvar = a.extent_int(1);
  audit.is = layout.is;
  audit.ie = layout.ie;
  audit.js = layout.js;
  audit.je = layout.je;
  audit.ks = layout.ks;
  audit.ke = layout.ke;
  audit.oldtonew.assign(oldtonew + first_old,
                        oldtonew + first_old + old_count);
  audit.newtoold.assign(newtoold + first_new,
                        newtoold + first_new + new_count);
  audit.old_flags.reserve(old_count);
  for (int old_gid = first_old; old_gid <= last_old; ++old_gid) {
    audit.old_flags.push_back(refine_flag.h_view(old_gid));
  }
  audit.pre_a = CaptureAuditActive(a, audit, old_count);

  Kokkos::fence();
  const auto coarse = Kokkos::create_mirror_view_and_copy(HostMemSpace(), ca);
  const std::size_t points = AuditPointsPerSlot(audit);
  for (const int old_gid : family_old_gids) {
    VCAuditFamily family;
    family.old_gid = old_gid;
    family.new_gid = oldtonew[old_gid];
    family.source_m = old_gid - first_old;
    family.destination_m = family.new_gid - first_new;
    family.oracle.resize(points);
    for (int variable = 0; variable < audit.nvar; ++variable) {
      for (int k = layout.ks; k <= layout.ke; ++k) {
        for (int j = layout.js; j <= layout.je; ++j) {
          for (int i = layout.is; i <= layout.ie; ++i) {
            const int qi = i - layout.is;
            const int qj = layout.nx2 <= 1 ? 0 : j - layout.js;
            const int qk = layout.nx3 <= 1 ? 0 : k - layout.ks;
            Real sum = 0.0;
            int count = 0;
            for (int child = 0; child < nleaf; ++child) {
              const int bx = child & 1;
              const int by = mesh->two_d || mesh->three_d
                  ? (child >> 1) & 1 : 0;
              const int bz = mesh->three_d ? (child >> 2) & 1 : 0;
              const bool x_has = bx == 0 ? qi <= layout.cnx1
                                          : qi >= layout.cnx1;
              const bool y_has = layout.nx2 <= 1 ||
                  (by == 0 ? qj <= layout.cnx2 : qj >= layout.cnx2);
              const bool z_has = layout.nx3 <= 1 ||
                  (bz == 0 ? qk <= layout.cnx3 : qk >= layout.cnx3);
              if (!x_has || !y_has || !z_has) continue;
              const int ci = layout.cis + qi - bx * layout.cnx1;
              const int cj = layout.nx2 <= 1 ? 0
                  : layout.cjs + qj - by * layout.cnx2;
              const int ck = layout.nx3 <= 1 ? 0
                  : layout.cks + qk - bz * layout.cnx3;
              sum += coarse(family.source_m + child, variable, ck, cj, ci);
              ++count;
            }
            family.oracle[AuditIndex(audit, 0, variable, k, j, i)] =
                sum / static_cast<Real>(count);
          }
        }
      }
    }
    family.oracle_hash = AuditHash(family.oracle.data(), points);
    audit.families.push_back(std::move(family));
  }
}

void RecordVCDerefineSlotAuditA5(const DvceArray5D<Real> &a) {
  auto &audit = vc_slot_audit;
  if (!audit.active) return;
  audit.a5_a = CaptureAuditActive(a, audit, audit.old_count);
  for (auto &family : audit.families) {
    const auto staging = CompareAuditOracle(
        family.oracle, audit.a5_a, family.source_m, audit);
    const auto destination = CompareAuditOracle(
        family.oracle, audit.a5_a, family.destination_m, audit);
    family.a5_staging_matches = !staging.found;
    family.a5_destination_matches = !destination.found;
    family.a5_staging_hash = AuditSlotHash(audit.a5_a, audit, family.source_m);
    family.a5_destination_hash =
        AuditSlotHash(audit.a5_a, audit, family.destination_m);
  }
  for (int local_old = 0; local_old < audit.old_count; ++local_old) {
    if (audit.old_flags[local_old] != 0) continue;
    const auto mismatch = CompareAuditSlot(
        audit.pre_a, local_old, audit.a5_a, local_old, audit);
    if (mismatch.found) {
      audit.a5_modified_live_old_gids.push_back(audit.first_old + local_old);
      if (!audit.first_a5_live_mismatch.found) {
        audit.first_a5_live_mismatch = mismatch;
      }
    }
  }
}

void FinishVCDerefineSlotAuditA6(const DvceArray5D<Real> &a) {
  auto &audit = vc_slot_audit;
  if (!audit.active) return;
  audit.a6_a = CaptureAuditActive(a, audit, audit.new_count);
  for (auto &family : audit.families) {
    const auto mismatch = CompareAuditOracle(
        family.oracle, audit.a6_a, family.destination_m, audit);
    family.a6_parent_matches = !mismatch.found;
    family.a6_parent_hash =
        AuditSlotHash(audit.a6_a, audit, family.destination_m);
    if (mismatch.found && !audit.first_a6_parent_mismatch.found) {
      audit.first_a6_parent_mismatch = mismatch;
    }
  }
  for (int local_old = 0; local_old < audit.old_count; ++local_old) {
    if (audit.old_flags[local_old] != 0) continue;
    const int old_gid = audit.first_old + local_old;
    const int new_gid = audit.oldtonew[local_old];
    const int local_new = new_gid - audit.first_new;
    if (local_new < 0 || local_new >= audit.new_count) continue;
    const auto mismatch = CompareAuditSlot(
        audit.pre_a, local_old, audit.a6_a, local_new, audit);
    if (mismatch.found) {
      audit.a6_bad_unaffected_old_gids.push_back(old_gid);
      if (!audit.first_a6_unaffected_mismatch.found) {
        audit.first_a6_unaffected_mismatch = mismatch;
      }
    }
  }

  std::ofstream output(audit.path, std::ios::out | std::ios::trunc);
  if (!output) AbortVC("cannot open native VC derefine slot audit output");
  output << std::setprecision(17)
         << "{\n  \"schema\": \"athenak_vc_derefine_slot_audit_v1\",\n"
         << "  \"cycle\": " << audit.cycle << ",\n  \"time\": "
         << audit.time << ",\n  \"rank\": " << global_variable::my_rank
         << ",\n  \"first_old\": " << audit.first_old
         << ",\n  \"first_new\": " << audit.first_new
         << ",\n  \"old_count\": " << audit.old_count
         << ",\n  \"new_count\": " << audit.new_count
         << ",\n  \"variables\": " << audit.nvar << ",\n  \"families\": [\n";
  for (std::size_t index = 0; index < audit.families.size(); ++index) {
    const auto &family = audit.families[index];
    output << "    {\"old_gid\":" << family.old_gid
           << ",\"new_gid\":" << family.new_gid
           << ",\"source_m\":" << family.source_m
           << ",\"destination_m\":" << family.destination_m
           << ",\"oracle_hash\":\"" << family.oracle_hash
           << "\",\"a5_staging_hash\":\"" << family.a5_staging_hash
           << "\",\"a5_destination_hash\":\""
           << family.a5_destination_hash << "\",\"a6_parent_hash\":\""
           << family.a6_parent_hash << "\",\"a5_staging_matches\":"
           << (family.a5_staging_matches ? "true" : "false")
           << ",\"a5_destination_matches\":"
           << (family.a5_destination_matches ? "true" : "false")
           << ",\"a6_parent_matches\":"
           << (family.a6_parent_matches ? "true" : "false") << "}"
           << (index + 1 == audit.families.size() ? "\n" : ",\n");
  }
  const auto write_ints = [&output](const std::vector<int> &values) {
    output << "[";
    for (std::size_t index = 0; index < values.size(); ++index) {
      if (index != 0) output << ",";
      output << values[index];
    }
    output << "]";
  };
  output << "  ],\n  \"a5_modified_live_old_gids\": ";
  write_ints(audit.a5_modified_live_old_gids);
  output << ",\n  \"a6_bad_unaffected_old_gids\": ";
  write_ints(audit.a6_bad_unaffected_old_gids);
  output << ",\n  \"first_a5_live_mismatch\": ";
  WriteMismatchJSON(output, audit.first_a5_live_mismatch);
  output << ",\n  \"first_a6_unaffected_mismatch\": ";
  WriteMismatchJSON(output, audit.first_a6_unaffected_mismatch);
  output << ",\n  \"first_a6_parent_mismatch\": ";
  WriteMismatchJSON(output, audit.first_a6_parent_mismatch);
  output << "\n}\n";
  output.close();
  if (!output) AbortVC("failed to write native VC derefine slot audit output");
  audit.active = false;
}

}  // namespace

void MeshRefinement::CopyVC(DvceArray5D<Real> &a) {
  // Moving a complete native array between MeshBlock slots is centering independent.
  CopyCC(a);
  FinishVCDerefineSlotAuditA6(a);
}

void MeshRefinement::RestrictVC(DvceArray5D<Real> &u, DvceArray5D<Real> &cu) {
  auto *z4c = pmy_mesh->pmb_pack->pz4c;
  if (z4c == nullptr ||
      z4c->layout.centering != z4c::Z4cGridCentering::vertex) {
    AbortVC("RestrictVC requires native vertex-centered Z4c storage");
  }
  const auto layout = z4c->layout;
  const int nmb = pmy_mesh->pmb_pack->nmb_thispack;
  const int nvar = u.extent_int(1);
  par_for("inject native VC restriction", DevExeSpace(), 0, nmb - 1,
          0, nvar - 1, layout.cks, layout.cke,
          layout.cjs, layout.cje, layout.cis, layout.cie,
      KOKKOS_LAMBDA(const int m, const int v, const int k,
                    const int j, const int i) {
        vertex_amr::InjectRestrictVCPoint(
            m, v, k, j, i, layout.is, layout.js, layout.ks,
            layout.cis, layout.cjs, layout.cks,
            layout.nx2 <= 1, layout.nx3 <= 1, u, cu);
      });
}

void MeshRefinement::CopyForRefinementVC(DvceArray5D<Real> &a,
                                          DvceArray5D<Real> &ca) {
  const auto layout = pmy_mesh->pmb_pack->pz4c->layout;
  const int refine_halo = vertex_amr::RequiredRefinementHaloForTransferOrder(
      pmy_mesh->pmb_pack->pz4c->opt.vertex_prolongation_order);
  if (refine_halo <= 0 || refine_halo > layout.ng ||
      refine_halo > layout.coarse_ng) {
    AbortVC("native VC refinement halo is incompatible with allocated storage");
  }
  const int il = layout.cis - refine_halo;
  const int iu = layout.cie + refine_halo;
  const int jl = layout.nx2 <= 1 ? 0 : layout.cjs - refine_halo;
  const int ju = layout.nx2 <= 1 ? 0 : layout.cje + refine_halo;
  const int kl = layout.nx3 <= 1 ? 0 : layout.cks - refine_halo;
  const int ku = layout.nx3 <= 1 ? 0 : layout.cke + refine_halo;
  const std::pair<int, int> idst(il, iu + 1);
  const std::pair<int, int> jdst(jl, ju + 1);
  const std::pair<int, int> kdst(kl, ku + 1);

  const int nmbs = new_gids_eachrank[global_variable::my_rank];
  const int nmbe = nmbs + new_nmb_eachrank[global_variable::my_rank] - 1;
  for (int newm = nmbs; newm <= nmbe; ++newm) {
    const int oldm = newtoold[newm];
    if (refine_flag.h_view(oldm) <= 0) continue;
    if (new_rank_eachmb[oldtonew[oldm]] != global_variable::my_rank ||
        new_rank_eachmb[newm] != global_variable::my_rank) {
      continue;
    }
    const int source_m = oldtonew[oldm] - nmbs;
    const int destination_m = newm - nmbs;
    const LogicalLocation &location = new_lloc_eachmb[newm];
    const int ox1 = location.lx1 & 1;
    const int ox2 = location.lx2 & 1;
    const int ox3 = location.lx3 & 1;
    const std::pair<int, int> isrc(
        layout.is + ox1 * layout.cnx1 - refine_halo,
        layout.is + (ox1 + 1) * layout.cnx1 + refine_halo + 1);
    const std::pair<int, int> jsrc = layout.nx2 <= 1
        ? std::pair<int, int>(0, 1)
        : std::pair<int, int>(
              layout.js + ox2 * layout.cnx2 - refine_halo,
              layout.js + (ox2 + 1) * layout.cnx2 + refine_halo + 1);
    const std::pair<int, int> ksrc = layout.nx3 <= 1
        ? std::pair<int, int>(0, 1)
        : std::pair<int, int>(
              layout.ks + ox3 * layout.cnx3 - refine_halo,
              layout.ks + (ox3 + 1) * layout.cnx3 + refine_halo + 1);
    if (VCAMRLifecycleDiagnosticEnabled()) {
      const auto valid_range = [](const std::pair<int, int> &range,
                                  const int extent) {
        return range.first >= 0 && range.second >= range.first &&
               range.second <= extent;
      };
      const bool same_shape =
          isrc.second - isrc.first == idst.second - idst.first &&
          jsrc.second - jsrc.first == jdst.second - jdst.first &&
          ksrc.second - ksrc.first == kdst.second - kdst.first;
      const bool collapsed_exact =
          (layout.nx2 > 1 || (jsrc.first == 0 && jsrc.second == 1 &&
                             jdst.first == 0 && jdst.second == 1)) &&
          (layout.nx3 > 1 || (ksrc.first == 0 && ksrc.second == 1 &&
                             kdst.first == 0 && kdst.second == 1));
      if (source_m < 0 || source_m >= a.extent_int(0) || destination_m < 0 ||
          destination_m >= ca.extent_int(0) ||
          !valid_range(isrc, a.extent_int(4)) ||
          !valid_range(jsrc, a.extent_int(3)) ||
          !valid_range(ksrc, a.extent_int(2)) ||
          !valid_range(idst, ca.extent_int(4)) ||
          !valid_range(jdst, ca.extent_int(3)) ||
          !valid_range(kdst, ca.extent_int(2)) || !same_shape ||
          !collapsed_exact || a.extent_int(1) != ca.extent_int(1)) {
        std::cerr << "### FATAL ERROR: native VC CopyForRefinementVC bounds mismatch"
                  << " new_gid=" << newm << " old_gid=" << oldm
                  << " source_m=" << source_m
                  << " destination_m=" << destination_m << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
    auto source = Kokkos::subview(a, source_m, Kokkos::ALL, ksrc, jsrc, isrc);
    auto destination =
        Kokkos::subview(ca, destination_m, Kokkos::ALL, kdst, jdst, idst);
    Kokkos::deep_copy(DevExeSpace(), destination, source);
  }
}

void MeshRefinement::RefineVC(DualArray1D<int> &new_to_old,
                              DvceArray5D<Real> &a,
                              DvceArray5D<Real> &ca) {
  auto *z4c = pmy_mesh->pmb_pack->pz4c;
  const auto layout = z4c->layout;
  const int transfer_order = z4c->opt.vertex_prolongation_order;
  if (!vertex_amr::IsSupportedTransferOrder(transfer_order)) {
    AbortVC("RefineVC requires transfer order 4, 6, or 8");
  }
  const int required =
      vertex_amr::RequiredRefinementHaloForTransferOrder(transfer_order);
  if (layout.coarse_ng < required) {
    AbortVC("native VC coarse ghost allocation is too narrow for midpoint interpolation");
  }
  const int nmb = new_nmb_eachrank[global_variable::my_rank];
  const int first_gid = new_gids_eachrank[global_variable::my_rank];
  const int nvar = a.extent_int(1);
  // Capture only device Views in the device closure.  Capturing a DualView wrapper
  // makes its host-side bookkeeping part of the kernel object; CUDA happened to
  // tolerate that representation, while SYCL exposes the invalid host indirection.
  const auto flags = refine_flag.d_view;
  const auto new_to_old_device = new_to_old.d_view;
  if (VCAMRLifecycleDiagnosticEnabled()) {
    DvceArray1D<long long> error("native VC AMR first device error", 6);
    Kokkos::deep_copy(error, 0LL);
    const long long new_extent = new_to_old_device.extent_int(0);
    const long long old_extent = flags.extent_int(0);
    const long long fine_m_extent = a.extent_int(0);
    const long long coarse_m_extent = ca.extent_int(0);
    const long long fine_i_extent = a.extent_int(4);
    const long long fine_j_extent = a.extent_int(3);
    const long long fine_k_extent = a.extent_int(2);
    const long long coarse_i_extent = ca.extent_int(4);
    const long long coarse_j_extent = ca.extent_int(3);
    const long long coarse_k_extent = ca.extent_int(2);
    Kokkos::parallel_for(
        "validate native VC refine lifecycle",
        Kokkos::RangePolicy<DevExeSpace>(0, nmb),
        KOKKOS_LAMBDA(const int m) {
          const long long new_gid = static_cast<long long>(m) + first_gid;
          if (new_gid < 0 || new_gid >= new_extent) {
            RecordFirstVCAMRError(error, VCAMRDeviceError::new_gid_out_of_range,
                                  10, m, new_gid, new_extent, first_gid);
            return;
          }
          const long long old_gid = new_to_old_device(new_gid);
          if (old_gid < 0 || old_gid >= old_extent) {
            RecordFirstVCAMRError(error, VCAMRDeviceError::old_gid_out_of_range,
                                  10, m, new_gid, old_gid, old_extent);
            return;
          }
          if (flags(old_gid) <= 0) return;
          if (m < 0 || m >= fine_m_extent || m >= coarse_m_extent) {
            RecordFirstVCAMRError(
                error, VCAMRDeviceError::local_meshblock_out_of_range, 10,
                m, fine_m_extent, coarse_m_extent, old_gid);
            return;
          }
          if (layout.is < 0 || layout.ie >= fine_i_extent ||
              layout.js < 0 || layout.je >= fine_j_extent ||
              layout.ks < 0 || layout.ke >= fine_k_extent) {
            RecordFirstVCAMRError(
                error, VCAMRDeviceError::active_bounds_out_of_range, 10,
                m, layout.ie, layout.je, layout.ke);
            return;
          }
          if (layout.cis - required < 0 ||
              layout.cie + required >= coarse_i_extent ||
              (layout.nx2 > 1 && (layout.cjs - required < 0 ||
                                  layout.cje + required >= coarse_j_extent)) ||
              (layout.nx3 > 1 && (layout.cks - required < 0 ||
                                  layout.cke + required >= coarse_k_extent))) {
            RecordFirstVCAMRError(
                error, VCAMRDeviceError::coarse_bounds_out_of_range, 10,
                m, layout.cie + required, layout.cje + required,
                layout.cke + required);
          }
        });
    CheckVCAMRDeviceError(error, "RefineVC_preflight");
  }
  DvceArray1D<unsigned long long> invalid("invalid VC refined chi", 1);
  Kokkos::deep_copy(invalid, 0ULL);
  par_for("native VC refine", DevExeSpace(), 0, nmb - 1, 0, nvar - 1,
          layout.ks, layout.ke, layout.js, layout.je, layout.is, layout.ie,
      KOKKOS_LAMBDA(const int m, const int v, const int k,
                    const int j, const int i) {
        if (flags(new_to_old_device(m + first_gid)) <= 0) return;
        Real value = 0.0;
        if (transfer_order == 4) {
          value = vertex_amr::ProlongVCPoint<4>(
              m, v, k, j, i, layout.is, layout.js, layout.ks,
              layout.cis, layout.cjs, layout.cks,
              layout.nx2 <= 1, layout.nx3 <= 1, ca, a);
        } else if (transfer_order == 6) {
          value = vertex_amr::ProlongVCPoint<6>(
              m, v, k, j, i, layout.is, layout.js, layout.ks,
              layout.cis, layout.cjs, layout.cks,
              layout.nx2 <= 1, layout.nx3 <= 1, ca, a);
        } else {
          value = vertex_amr::ProlongVCPoint<8>(
              m, v, k, j, i, layout.is, layout.js, layout.ks,
              layout.cis, layout.cjs, layout.cks,
              layout.nx2 <= 1, layout.nx3 <= 1, ca, a);
        }
        if (v == z4c::Z4c::I_Z4C_CHI &&
            (!Kokkos::isfinite(value) || !(value > 0.0))) {
          Kokkos::atomic_inc(&invalid(0));
        }
      });
  const auto invalid_host =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), invalid);
  unsigned long long invalid_global = invalid_host(0);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &invalid_global, 1, MPI_UNSIGNED_LONG_LONG,
                MPI_SUM, MPI_COMM_WORLD);
#endif
  if (invalid_global != 0) {
    AbortVC("native VC prolongation produced nonfinite/nonpositive active chi");
  }
}

void MeshRefinement::DerefineVCSameRank(DvceArray5D<Real> &a,
                                         DvceArray5D<Real> &ca) {
  const auto layout = pmy_mesh->pmb_pack->pz4c->layout;
  int nleaf = pmy_mesh->three_d ? 8 : pmy_mesh->two_d ? 4 : 2;
  const int first_old = pmy_mesh->gids_eachrank[global_variable::my_rank];
  const int last_old = first_old +
      pmy_mesh->nmb_eachrank[global_variable::my_rank] - 1;
  const int nvar = a.extent_int(1);
  const bool two_d = pmy_mesh->two_d;
  const bool three_d = pmy_mesh->three_d;

  BeginVCDerefineSlotAudit(
      pmy_mesh, oldtonew, newtoold, new_rank_eachmb, new_gids_eachrank,
      new_nmb_eachrank, refine_flag, a, ca, layout);

  for (int oldm = first_old; oldm <= last_old; ++oldm) {
    if (refine_flag.h_view(oldm) >= -1) continue;
    const auto &lower_child = pmy_mesh->lloc_eachmb[oldm];
    if ((lower_child.lx1 & 1) != 0 ||
        ((two_d || three_d) && (lower_child.lx2 & 1) != 0) ||
        (three_d && (lower_child.lx3 & 1) != 0)) {
      continue;
    }
    const int newm = oldtonew[oldm];
    if (new_rank_eachmb[newm] != global_variable::my_rank) continue;
    bool all_siblings_local = oldm + nleaf - 1 <= last_old;
    for (int child = 0; child < nleaf && all_siblings_local; ++child) {
      all_siblings_local =
          pmy_mesh->rank_eachmb[oldm + child] == global_variable::my_rank;
    }
    // A family split across ranks is reconstructed by the AMR receive/unpack path.
    if (!all_siblings_local) continue;
    // A5 still operates on the old MeshBlock-slot layout.  Stage the parent in
    // its old lower-child slot; A6 CopyVC/CopyCC relocates that slot to the new
    // parent slot.  Writing directly to the new slot here can clobber a live old
    // source before A6 has copied it.
    const int source_base = oldm - first_old;
    const int staging_m = source_base;
    // Every target node is assigned by one deterministic thread.  At shared sibling
    // planes all available copies are checked and averaged in logical child order.
    DvceArray1D<unsigned long long> inconsistent("inconsistent VC siblings", 1);
    Kokkos::deep_copy(inconsistent, 0ULL);
    par_for("native VC deterministic derefine", DevExeSpace(), 0, nvar - 1,
            layout.ks, layout.ke, layout.js, layout.je, layout.is, layout.ie,
        KOKKOS_LAMBDA(const int v, const int k, const int j, const int i) {
          const int qi = i - layout.is;
          const int qj = layout.nx2 <= 1 ? 0 : j - layout.js;
          const int qk = layout.nx3 <= 1 ? 0 : k - layout.ks;
          Real sum = 0.0;
          Real minimum = std::numeric_limits<Real>::max();
          Real maximum = -std::numeric_limits<Real>::max();
          int count = 0;
          for (int child = 0; child < nleaf; ++child) {
            const int bx = child & 1;
            const int by = two_d || three_d ? (child >> 1) & 1 : 0;
            const int bz = three_d ? (child >> 2) & 1 : 0;
            const bool x_has = bx == 0 ? qi <= layout.cnx1 : qi >= layout.cnx1;
            const bool y_has = layout.nx2 <= 1 ||
                (by == 0 ? qj <= layout.cnx2 : qj >= layout.cnx2);
            const bool z_has = layout.nx3 <= 1 ||
                (bz == 0 ? qk <= layout.cnx3 : qk >= layout.cnx3);
            if (!x_has || !y_has || !z_has || oldm + child > last_old) continue;
            const int ci = layout.cis + qi - bx * layout.cnx1;
            const int cj = layout.nx2 <= 1 ? 0
                : layout.cjs + qj - by * layout.cnx2;
            const int ck = layout.nx3 <= 1 ? 0
                : layout.cks + qk - bz * layout.cnx3;
            const Real value = ca(source_base + child, v, ck, cj, ci);
            sum += value;
            minimum = value < minimum ? value : minimum;
            maximum = value > maximum ? value : maximum;
            ++count;
          }
          if (count == 0 || !Kokkos::isfinite(minimum) || !Kokkos::isfinite(maximum)) {
            Kokkos::atomic_inc(&inconsistent(0));
            return;
          }
          const Real scale = fmax(1.0, fmax(fabs(minimum), fabs(maximum)));
          if (maximum - minimum > 64.0 * std::numeric_limits<Real>::epsilon() * scale) {
            Kokkos::atomic_inc(&inconsistent(0));
          }
          a(staging_m, v, k, j, i) = sum / static_cast<Real>(count);
        });
    const auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), inconsistent);
    if (host(0) != 0) {
      AbortVC("materially inconsistent coincident VC sibling values during derefinement");
    }
  }
  RecordVCDerefineSlotAuditA5(a);
}
