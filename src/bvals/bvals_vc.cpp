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

namespace {

[[noreturn]] void AbortUnsupportedVCLevel(const int gid, const int neighbor_gid,
                                          const int level,
                                          const int neighbor_level) {
  std::cerr << "### FATAL ERROR: native VC coarse/fine communication is not enabled: "
            << "gid=" << gid << " level=" << level
            << " neighbor_gid=" << neighbor_gid
            << " neighbor_level=" << neighbor_level << std::endl;
#if MPI_PARALLEL_ENABLED
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
  std::exit(EXIT_FAILURE);
}

}  // namespace

TaskStatus MeshBoundaryValuesVC::PackAndSendVC(DvceArray5D<Real> &a) {
  const int nmb = pmy_pack->nmb_thispack;
  const int nnghbr = pmy_pack->pmb->nnghbr;
  const int nvar = a.extent_int(1);
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &mbgid = pmy_pack->pmb->mb_gid;

  // Do not silently reinterpret VC data with CC coarse/fine geometry.
  for (int m = 0; m < nmb; ++m) {
    for (int n = 0; n < nnghbr; ++n) {
      const auto neighbor = nghbr.h_view(m, n);
      if (neighbor.gid >= 0 && neighbor.lev != mblev.h_view(m)) {
        AbortUnsupportedVCLevel(mbgid.h_view(m), neighbor.gid,
                                mblev.h_view(m), neighbor.lev);
      }
    }
  }

  const int my_rank = global_variable::my_rank;
  auto &sbuf = sendbuf;
  auto &rbuf = recvbuf;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmb * nnghbr * nvar, Kokkos::AUTO);
  Kokkos::parallel_for("Pack VC same-level buffers", policy,
      KOKKOS_LAMBDA(TeamMember_t member) {
        const int league = member.league_rank();
        const int m = league / (nnghbr * nvar);
        const int n = (league / nvar) % nnghbr;
        const int v = league % nvar;
        if (nghbr.d_view(m, n).gid < 0) return;
        const auto bounds = sbuf[n].isame[0];
        const int ni = bounds.bie - bounds.bis + 1;
        const int nj = bounds.bje - bounds.bjs + 1;
        const int nk = bounds.bke - bounds.bks + 1;
        const int destination_m = nghbr.d_view(m, n).gid - mbgid.d_view(0);
        const int destination_n = nghbr.d_view(m, n).dest;
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(member, nk * nj),
            [&](const int kj) {
              const int k = bounds.bks + kj / nj;
              const int j = bounds.bjs + kj % nj;
              Kokkos::parallel_for(
                  Kokkos::ThreadVectorRange(member, bounds.bis, bounds.bie + 1),
                  [&](const int i) {
                    const int offset = i - bounds.bis +
                        ni * (j - bounds.bjs + nj * (k - bounds.bks + nk * v));
                    if (nghbr.d_view(m, n).rank == my_rank) {
                      rbuf[destination_n].vars(destination_m, offset) = a(m, v, k, j, i);
                    } else {
                      sbuf[n].vars(m, offset) = a(m, v, k, j, i);
                    }
                  });
            });
      });

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
      const int data_size = nvar * sendbuf[n].isame_ndat;
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

TaskStatus MeshBoundaryValuesVC::RecvAndUnpackVC(DvceArray5D<Real> &a) {
  const int nmb = pmy_pack->nmb_thispack;
  const int nnghbr = pmy_pack->pmb->nnghbr;
  const int nvar = a.extent_int(1);
  auto &nghbr = pmy_pack->pmb->nghbr;

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
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmb * nnghbr * nvar, Kokkos::AUTO);
  Kokkos::parallel_for("Unpack VC same-level buffers", policy,
      KOKKOS_LAMBDA(TeamMember_t member) {
        const int league = member.league_rank();
        const int m = league / (nnghbr * nvar);
        const int n = (league / nvar) % nnghbr;
        const int v = league % nvar;
        if (nghbr.d_view(m, n).gid < 0) return;
        const auto bounds = rbuf[n].isame[0];
        const int ni = bounds.bie - bounds.bis + 1;
        const int nj = bounds.bje - bounds.bjs + 1;
        const int nk = bounds.bke - bounds.bks + 1;
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(member, nk * nj),
            [&](const int kj) {
              const int k = bounds.bks + kj / nj;
              const int j = bounds.bjs + kj % nj;
              Kokkos::parallel_for(
                  Kokkos::ThreadVectorRange(member, bounds.bis, bounds.bie + 1),
                  [&](const int i) {
                    // Preserve all shared active values.  They are reconciled afterward
                    // by the canonical deterministic contributor reduction.
                    const bool active =
                        i >= vertex_layout.is && i <= vertex_layout.ie &&
                        j >= vertex_layout.js && j <= vertex_layout.je &&
                        k >= vertex_layout.ks && k <= vertex_layout.ke;
                    if (!active) {
                      const int offset = i - bounds.bis +
                          ni * (j - bounds.bjs + nj * (k - bounds.bks + nk * v));
                      a(m, v, k, j, i) = rbuf[n].vars(m, offset);
                    }
                  });
            });
      });
  return TaskStatus::complete;
}
