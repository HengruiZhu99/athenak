//========================================================================================
// AthenaK astrophysical plasma code
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file bvals_vc.cpp
//! \brief Native same-level vertex-centered boundary communication.

#include "bvals.hpp"

#include <cstdlib>
#include <iostream>

#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/vertex_amr.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_vertex_topology.hpp"

namespace {

[[noreturn]] void AbortVCCommunication(const char *message) {
  std::cerr << "### FATAL ERROR: native VC boundary communication: "
            << message << std::endl;
#if MPI_PARALLEL_ENABLED
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
  std::exit(EXIT_FAILURE);
}

}  // namespace

TaskStatus MeshBoundaryValuesVC::PackAndSendVC(DvceArray5D<Real> &a,
                                               DvceArray5D<Real> &ca) {
  const int nmb = pmy_pack->nmb_thispack;
  const int nnghbr = pmy_pack->pmb->nnghbr;
  const int nvar = a.extent_int(1);
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &mbgid = pmy_pack->pmb->mb_gid;

  const int my_rank = global_variable::my_rank;
  const bool multilevel = pmy_pack->pmesh->multilevel;
  const bool fold_variables = pmy_pack->pz4c != nullptr &&
                              pmy_pack->pz4c->opt.lean_runtime;
  auto &sbuf = sendbuf;
  auto &rbuf = recvbuf;
  const int neighbor_teams = nmb * nnghbr;
  const int league_size = fold_variables ? neighbor_teams
                                         : neighbor_teams * nvar;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), league_size, Kokkos::AUTO);
  Kokkos::parallel_for("Pack VC same-level buffers", policy,
      KOKKOS_LAMBDA(TeamMember_t member) {
        const int league = member.league_rank();
        const int neighbor = fold_variables ? league : league / nvar;
        const int m = neighbor / nnghbr;
        const int n = neighbor % nnghbr;
        const int fixed_v = fold_variables ? 0 : league % nvar;
        if (nghbr.d_view(m, n).gid < 0) return;
        const int neighbor_level = nghbr.d_view(m, n).lev;
        const int local_level = mblev.d_view(m);
        const auto bounds = neighbor_level < local_level ? sbuf[n].icoar[0]
                            : neighbor_level == local_level ? sbuf[n].isame[0]
                                                            : sbuf[n].ifine[0];
        const int ni = bounds.bie - bounds.bis + 1;
        const int nj = bounds.bje - bounds.bjs + 1;
        const int nk = bounds.bke - bounds.bks + 1;
        const int destination_m = nghbr.d_view(m, n).gid - mbgid.d_view(0);
        const int destination_n = nghbr.d_view(m, n).dest;
        const int points_per_variable = nk * nj;
        const int work = fold_variables ? nvar * points_per_variable
                                        : points_per_variable;
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(member, work),
            [&](const int q) {
              const int v = fold_variables ? q / points_per_variable : fixed_v;
              const int kj = fold_variables ? q % points_per_variable : q;
              const int k = bounds.bks + kj / nj;
              const int j = bounds.bjs + kj % nj;
              Kokkos::parallel_for(
                  Kokkos::ThreadVectorRange(member, bounds.bis, bounds.bie + 1),
                  [&](const int i) {
                    const int offset = i - bounds.bis +
                        ni * (j - bounds.bjs + nj * (k - bounds.bks + nk * v));
                    if (nghbr.d_view(m, n).rank == my_rank) {
                      rbuf[destination_n].vars(destination_m, offset) =
                          neighbor_level < local_level ? ca(m, v, k, j, i)
                                                       : a(m, v, k, j, i);
                    } else {
                      sbuf[n].vars(m, offset) =
                          neighbor_level < local_level ? ca(m, v, k, j, i)
                                                       : a(m, v, k, j, i);
                    }
                  });
            });
      });

  // A fine block's coarse cache also needs same-level coarse neighbors to complete
  // high-order parent stencils at face/edge/corner intersections.
  if (multilevel) {
    Kokkos::parallel_for("Pack VC same-level coarse buffers", policy,
        KOKKOS_LAMBDA(TeamMember_t member) {
          const int league = member.league_rank();
          const int neighbor = fold_variables ? league : league / nvar;
          const int m = neighbor / nnghbr;
          const int n = neighbor % nnghbr;
          const int fixed_v = fold_variables ? 0 : league % nvar;
          if (nghbr.d_view(m, n).gid < 0 ||
              nghbr.d_view(m, n).lev != mblev.d_view(m)) return;
          const auto bounds = sbuf[n].isame_z4c;
          const int ni = bounds.bie - bounds.bis + 1;
          const int nj = bounds.bje - bounds.bjs + 1;
          const int nk = bounds.bke - bounds.bks + 1;
          const int destination_m = nghbr.d_view(m, n).gid - mbgid.d_view(0);
          const int destination_n = nghbr.d_view(m, n).dest;
          const int base = nvar * sbuf[n].isame_ndat;
          const int points_per_variable = nk * nj;
          const int work = fold_variables ? nvar * points_per_variable
                                          : points_per_variable;
          Kokkos::parallel_for(Kokkos::TeamThreadRange<>(member, work),
              [&](const int q) {
                const int v = fold_variables ? q / points_per_variable : fixed_v;
                const int kj = fold_variables ? q % points_per_variable : q;
                const int k = bounds.bks + kj / nj;
                const int j = bounds.bjs + kj % nj;
                Kokkos::parallel_for(
                    Kokkos::ThreadVectorRange(member, bounds.bis, bounds.bie + 1),
                    [&](const int i) {
                      const int offset = base + i - bounds.bis +
                          ni * (j - bounds.bjs + nj * (k - bounds.bks + nk * v));
                      if (nghbr.d_view(m, n).rank == my_rank) {
                        rbuf[destination_n].vars(destination_m, offset) =
                            ca(m, v, k, j, i);
                      } else {
                        sbuf[n].vars(m, offset) = ca(m, v, k, j, i);
                      }
                    });
              });
        });
  }

#if MPI_PARALLEL_ENABLED
  Kokkos::fence();
  bool no_errors = true;
  for (int m = 0; m < nmb; ++m) {
    for (int n = 0; n < nnghbr; ++n) {
      const auto neighbor = nghbr.h_view(m, n);
      if (neighbor.gid < 0 || neighbor.rank == my_rank) continue;
      const int lid = neighbor.gid - pmy_pack->pmesh->gids_eachrank[neighbor.rank];
      const int tag = CreateBvals_MPI_Tag(lid, neighbor.dest);
      auto send_ptr = Kokkos::subview(sendbuf[n].vars, m, Kokkos::ALL);
      int points = 0;
      if (neighbor.lev < mblev.h_view(m)) {
        points = sendbuf[n].icoar_ndat;
      } else if (neighbor.lev == mblev.h_view(m)) {
        points = multilevel ? sendbuf[n].isame_z4c_ndat
                            : sendbuf[n].isame_ndat;
      } else {
        points = sendbuf[n].ifine_ndat;
      }
      const int data_size = nvar * points;
      const int error = MPI_Isend(send_ptr.data(), data_size, MPI_ATHENA_REAL,
                                  neighbor.rank, tag, comm_vars,
                                  &(sendbuf[n].vars_req[m]));
      if (error != MPI_SUCCESS) no_errors = false;
    }
  }
  if (!no_errors) {
    std::cerr << "### FATAL ERROR: MPI error posting native VC sends" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

TaskStatus MeshBoundaryValuesVC::RecvAndUnpackVC(DvceArray5D<Real> &a,
                                                 DvceArray5D<Real> &ca) {
  const int nmb = pmy_pack->nmb_thispack;
  const int nnghbr = pmy_pack->pmb->nnghbr;
  const int nvar = a.extent_int(1);
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  const bool multilevel = pmy_pack->pmesh->multilevel;
  const bool fold_variables = pmy_pack->pz4c != nullptr &&
                              pmy_pack->pz4c->opt.lean_runtime;

#if MPI_PARALLEL_ENABLED
  bool incomplete = false;
  bool no_errors = true;
  for (int m = 0; m < nmb; ++m) {
    for (int n = 0; n < nnghbr; ++n) {
      const auto neighbor = nghbr.h_view(m, n);
      if (neighbor.gid < 0 || neighbor.rank == global_variable::my_rank) continue;
      int complete = 0;
      const int error = MPI_Test(&(recvbuf[n].vars_req[m]), &complete,
                                 MPI_STATUS_IGNORE);
      if (error != MPI_SUCCESS) no_errors = false;
      if (complete == 0) incomplete = true;
    }
  }
  if (!no_errors) {
    std::cerr << "### FATAL ERROR: MPI error testing native VC receives" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  if (incomplete) return TaskStatus::incomplete;
#endif

  auto &rbuf = recvbuf;
  const auto vertex_layout = layout;
  const int neighbor_teams = nmb * nnghbr;
  const int league_size = fold_variables ? neighbor_teams
                                         : neighbor_teams * nvar;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), league_size, Kokkos::AUTO);
  Kokkos::parallel_for("Unpack VC same-level buffers", policy,
      KOKKOS_LAMBDA(TeamMember_t member) {
        const int league = member.league_rank();
        const int neighbor = fold_variables ? league : league / nvar;
        const int m = neighbor / nnghbr;
        const int n = neighbor % nnghbr;
        const int fixed_v = fold_variables ? 0 : league % nvar;
        if (nghbr.d_view(m, n).gid < 0) return;
        const int neighbor_level = nghbr.d_view(m, n).lev;
        const int local_level = mblev.d_view(m);
        const auto bounds = neighbor_level < local_level ? rbuf[n].icoar[0]
                            : neighbor_level == local_level ? rbuf[n].isame[0]
                                                            : rbuf[n].ifine[0];
        const int ni = bounds.bie - bounds.bis + 1;
        const int nj = bounds.bje - bounds.bjs + 1;
        const int nk = bounds.bke - bounds.bks + 1;
        const int points_per_variable = nk * nj;
        const int work = fold_variables ? nvar * points_per_variable
                                        : points_per_variable;
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(member, work),
            [&](const int q) {
              const int v = fold_variables ? q / points_per_variable : fixed_v;
              const int kj = fold_variables ? q % points_per_variable : q;
              const int k = bounds.bks + kj / nj;
              const int j = bounds.bjs + kj % nj;
              Kokkos::parallel_for(
                  Kokkos::ThreadVectorRange(member, bounds.bis, bounds.bie + 1),
                  [&](const int i) {
                    const int offset = i - bounds.bis +
                        ni * (j - bounds.bjs + nj * (k - bounds.bks + nk * v));
                    if (neighbor_level < local_level) {
                      ca(m, v, k, j, i) = rbuf[n].vars(m, offset);
                      return;
                    }
                    // Same-level and fine-to-coarse messages include shared active
                    // vertices for matched geometry, but those values are reconciled by
                    // the canonical deterministic contributor reduction below.
                    const bool active =
                        i >= vertex_layout.is && i <= vertex_layout.ie &&
                        j >= vertex_layout.js && j <= vertex_layout.je &&
                        k >= vertex_layout.ks && k <= vertex_layout.ke;
                    if (!active) a(m, v, k, j, i) = rbuf[n].vars(m, offset);
                  });
            });
      });
  if (multilevel) {
    Kokkos::parallel_for("Unpack VC same-level coarse buffers", policy,
        KOKKOS_LAMBDA(TeamMember_t member) {
          const int league = member.league_rank();
          const int neighbor = fold_variables ? league : league / nvar;
          const int m = neighbor / nnghbr;
          const int n = neighbor % nnghbr;
          const int fixed_v = fold_variables ? 0 : league % nvar;
          if (nghbr.d_view(m, n).gid < 0 ||
              nghbr.d_view(m, n).lev != mblev.d_view(m)) return;
          const auto bounds = rbuf[n].isame_z4c;
          const int ni = bounds.bie - bounds.bis + 1;
          const int nj = bounds.bje - bounds.bjs + 1;
          const int nk = bounds.bke - bounds.bks + 1;
          const int base = nvar * rbuf[n].isame_ndat;
          const int points_per_variable = nk * nj;
          const int work = fold_variables ? nvar * points_per_variable
                                          : points_per_variable;
          Kokkos::parallel_for(Kokkos::TeamThreadRange<>(member, work),
              [&](const int q) {
                const int v = fold_variables ? q / points_per_variable : fixed_v;
                const int kj = fold_variables ? q % points_per_variable : q;
                const int k = bounds.bks + kj / nj;
                const int j = bounds.bjs + kj % nj;
                Kokkos::parallel_for(
                    Kokkos::ThreadVectorRange(member, bounds.bis, bounds.bie + 1),
                    [&](const int i) {
                      const int offset = base + i - bounds.bis +
                          ni * (j - bounds.bjs + nj * (k - bounds.bks + nk * v));
                      ca(m, v, k, j, i) = rbuf[n].vars(m, offset);
                    });
              });
        });
  }
  return TaskStatus::complete;
}

void MeshBoundaryValuesVC::ProlongateVC(DvceArray5D<Real> &a,
                                        DvceArray5D<Real> &ca,
                                        const int transfer_order,
                                        const int positive_component) {
  if (!vertex_amr::IsSupportedTransferOrder(transfer_order)) {
    AbortVCCommunication("ProlongateVC requires transfer order 4, 6, or 8");
  }
  const int nmb = pmy_pack->nmb_thispack;
  const int nnghbr = pmy_pack->pmb->nnghbr;
  const int nvar = a.extent_int(1);
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto records = pmy_pack->pz4c->vertex_topology_plan->records.d_view;
  auto iprol = prolongation_bounds.d_view;
  const bool compact_work = pmy_pack->pz4c->opt.lean_runtime;
  const auto prolongation_work = prolongation_neighbor_work.d_view;
  const int neighbor_work = compact_work ? nprolongation_neighbor_work
                                         : nmb * nnghbr;
  const auto vertex_layout = layout;
  DvceArray1D<unsigned long long> invalid;
  if (!compact_work) {
    invalid = DvceArray1D<unsigned long long>(
        "invalid VC boundary prolongation", 1);
    Kokkos::deep_copy(invalid, 0ULL);
  }
  Kokkos::TeamPolicy<> policy(DevExeSpace(), neighbor_work * nvar, Kokkos::AUTO);
  Kokkos::parallel_for("Prolong VC coarse-fine boundaries", policy,
      KOKKOS_LAMBDA(TeamMember_t member) {
        const int league = member.league_rank();
        const int row = league / nvar;
        const int m = compact_work ? prolongation_work(row, 0)
                                   : row / nnghbr;
        const int n = compact_work ? prolongation_work(row, 1)
                                   : row % nnghbr;
        const int v = league % nvar;
        if (nghbr.d_view(m, n).gid < 0 ||
            nghbr.d_view(m, n).lev >= mblev.d_view(m)) return;
        const auto coarse = iprol(n);
        const int raw_il = vertex_layout.is +
                           2 * (coarse.bis - vertex_layout.cis);
        const int raw_iu = vertex_layout.is +
                           2 * (coarse.bie - vertex_layout.cis);
        const int raw_jl = vertex_layout.js +
                           2 * (coarse.bjs - vertex_layout.cjs);
        const int raw_ju = vertex_layout.js +
                           2 * (coarse.bje - vertex_layout.cjs);
        const int raw_kl = vertex_layout.ks +
                           2 * (coarse.bks - vertex_layout.cks);
        const int raw_ku = vertex_layout.ks +
                           2 * (coarse.bke - vertex_layout.cks);
        const int il = vertex_layout.sis > raw_il ? vertex_layout.sis : raw_il;
        const int iu = vertex_layout.sie < raw_iu ? vertex_layout.sie : raw_iu;
        const int jl = vertex_layout.collapse_x2 ? 0
            : (vertex_layout.sjs > raw_jl ? vertex_layout.sjs : raw_jl);
        const int ju = vertex_layout.collapse_x2 ? 0
            : (vertex_layout.sje < raw_ju ? vertex_layout.sje : raw_ju);
        const int kl = vertex_layout.collapse_x3 ? 0
            : (vertex_layout.sks > raw_kl ? vertex_layout.sks : raw_kl);
        const int ku = vertex_layout.collapse_x3 ? 0
            : (vertex_layout.ske < raw_ku ? vertex_layout.ske : raw_ku);
        const int ni = iu - il + 1;
        const int nj = ju - jl + 1;
        const int nk = ku - kl + 1;
        if (ni <= 0 || nj <= 0 || nk <= 0) return;
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(member, nk * nj),
            [&](const int kj) {
              const int k = kl + kj / nj;
              const int j = jl + kj % nj;
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, il, iu + 1),
                  [&](const int i) {
                    const bool active =
                        i >= vertex_layout.is && i <= vertex_layout.ie &&
                        j >= vertex_layout.js && j <= vertex_layout.je &&
                        k >= vertex_layout.ks && k <= vertex_layout.ke;
                    if (active && records(m, k, j, i).role !=
                        vertex_topology::VertexNodeRole::hanging_fine_interface) {
                      return;
                    }
                    Real value = 0.0;
                    if (transfer_order == 4) {
                      value = vertex_amr::ProlongVCPoint<4>(
                          m, v, k, j, i, vertex_layout.is, vertex_layout.js,
                          vertex_layout.ks, vertex_layout.cis, vertex_layout.cjs,
                          vertex_layout.cks, vertex_layout.collapse_x2,
                          vertex_layout.collapse_x3, ca, a);
                    } else if (transfer_order == 6) {
                      value = vertex_amr::ProlongVCPoint<6>(
                          m, v, k, j, i, vertex_layout.is, vertex_layout.js,
                          vertex_layout.ks, vertex_layout.cis, vertex_layout.cjs,
                          vertex_layout.cks, vertex_layout.collapse_x2,
                          vertex_layout.collapse_x3, ca, a);
                    } else {
                      value = vertex_amr::ProlongVCPoint<8>(
                          m, v, k, j, i, vertex_layout.is, vertex_layout.js,
                          vertex_layout.ks, vertex_layout.cis, vertex_layout.cjs,
                          vertex_layout.cks, vertex_layout.collapse_x2,
                          vertex_layout.collapse_x3, ca, a);
                    }
                    if (!Kokkos::isfinite(value) ||
                        (v == positive_component && !(value > 0.0))) {
                      if (compact_work) {
                        Kokkos::abort(
                            "coarse/fine interpolation produced invalid state");
                      } else {
                        Kokkos::atomic_inc(&invalid(0));
                      }
                    }
                  });
            });
      });
  // The lean path fails directly on device and remains ordered before the next
  // task in the same execution space.  The exhaustive path retains its
  // deterministic global count and host-side report.
  if (compact_work) return;
  Kokkos::fence();
  const auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), invalid);
  unsigned long long global_invalid = host(0);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &global_invalid, 1, MPI_UNSIGNED_LONG_LONG,
                MPI_SUM, MPI_COMM_WORLD);
#endif
  if (global_invalid != 0) {
    AbortVCCommunication("coarse/fine interpolation produced invalid state");
  }
}
