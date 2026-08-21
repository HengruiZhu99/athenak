//========================================================================================
// AthenaK astrophysical plasma code
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mesh_refinement_vc.cpp
//! \brief Native vertex-centered refinement and derefinement operations.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <utility>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/vertex_amr.hpp"
#include "z4c/z4c.hpp"

namespace {

[[noreturn]] void AbortVC(const char *message) {
  if (global_variable::my_rank == 0) {
    std::cerr << "### FATAL ERROR: " << message << std::endl;
  }
#if MPI_PARALLEL_ENABLED
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
  std::exit(EXIT_FAILURE);
}

}  // namespace

void MeshRefinement::CopyVC(DvceArray5D<Real> &a) {
  // Moving a complete native array between MeshBlock slots is centering independent.
  CopyCC(a);
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
  const int il = layout.cis - layout.coarse_ng;
  const int iu = layout.cie + layout.coarse_ng;
  const int jl = layout.nx2 <= 1 ? 0 : layout.cjs - layout.coarse_ng;
  const int ju = layout.nx2 <= 1 ? 0 : layout.cje + layout.coarse_ng;
  const int kl = layout.nx3 <= 1 ? 0 : layout.cks - layout.coarse_ng;
  const int ku = layout.nx3 <= 1 ? 0 : layout.cke + layout.coarse_ng;
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
        layout.is + ox1 * layout.cnx1 - layout.coarse_ng,
        layout.is + (ox1 + 1) * layout.cnx1 + layout.coarse_ng + 1);
    const std::pair<int, int> jsrc = layout.nx2 <= 1
        ? std::pair<int, int>(0, 1)
        : std::pair<int, int>(
              layout.js + ox2 * layout.cnx2 - layout.coarse_ng,
              layout.js + (ox2 + 1) * layout.cnx2 + layout.coarse_ng + 1);
    const std::pair<int, int> ksrc = layout.nx3 <= 1
        ? std::pair<int, int>(0, 1)
        : std::pair<int, int>(
              layout.ks + ox3 * layout.cnx3 - layout.coarse_ng,
              layout.ks + (ox3 + 1) * layout.cnx3 + layout.coarse_ng + 1);
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
  const int order = z4c->opt.spatial_order;
  if (order != 2 && order != 4 && order != 6) {
    AbortVC("RefineVC requires O2, O4, or O6 spatial order");
  }
  const int required = order == 6 ? vertex_amr::RequiredCoarseGhostWidth<6>()
                       : order == 4 ? vertex_amr::RequiredCoarseGhostWidth<4>()
                                    : vertex_amr::RequiredCoarseGhostWidth<2>();
  if (layout.coarse_ng < required) {
    AbortVC("native VC coarse ghost allocation is too narrow for midpoint interpolation");
  }
  const int nmb = new_nmb_eachrank[global_variable::my_rank];
  const int first_gid = new_gids_eachrank[global_variable::my_rank];
  const int nvar = a.extent_int(1);
  auto &flags = refine_flag;
  DvceArray1D<unsigned long long> invalid("invalid VC refined chi", 1);
  Kokkos::deep_copy(invalid, 0ULL);
  par_for("native VC refine", DevExeSpace(), 0, nmb - 1, 0, nvar - 1,
          layout.ks, layout.ke, layout.js, layout.je, layout.is, layout.ie,
      KOKKOS_LAMBDA(const int m, const int v, const int k,
                    const int j, const int i) {
        if (flags.d_view(new_to_old.d_view(m + first_gid)) <= 0) return;
        Real value = 0.0;
        if (order == 2) {
          value = vertex_amr::ProlongVCPoint<2>(
              m, v, k, j, i, layout.is, layout.js, layout.ks,
              layout.cis, layout.cjs, layout.cks,
              layout.nx2 <= 1, layout.nx3 <= 1, ca, a);
        } else if (order == 4) {
          value = vertex_amr::ProlongVCPoint<4>(
              m, v, k, j, i, layout.is, layout.js, layout.ks,
              layout.cis, layout.cjs, layout.cks,
              layout.nx2 <= 1, layout.nx3 <= 1, ca, a);
        } else {
          value = vertex_amr::ProlongVCPoint<6>(
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
  const int first_new = new_gids_eachrank[global_variable::my_rank];
  const int nvar = a.extent_int(1);
  const bool two_d = pmy_mesh->two_d;
  const bool three_d = pmy_mesh->three_d;

  for (int oldm = first_old; oldm <= last_old; ++oldm) {
    if (refine_flag.h_view(oldm) >= -1) continue;
    const auto &lower_child = pmy_mesh->lloc_eachmb[oldm];
    if ((lower_child.lx1 & 1) != 0 ||
        (two_d && (lower_child.lx2 & 1) != 0) ||
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
    const int destination_m = newm - first_new;
    // Every target node is assigned by one deterministic thread.  At shared sibling
    // planes all available copies are checked and averaged in logical child order.
    DvceArray1D<unsigned long long> inconsistent("inconsistent VC siblings", 1);
    Kokkos::deep_copy(inconsistent, 0ULL);
    const int source_base = oldm - first_old;
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
          a(destination_m, v, k, j, i) = sum / static_cast<Real>(count);
        });
    const auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), inconsistent);
    if (host(0) != 0) {
      AbortVC("materially inconsistent coincident VC sibling values during derefinement");
    }
  }
}
