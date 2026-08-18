//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file prolongation.cpp
//! \brief functions to prolongate data at boundaries for cell-centered and face-centered
//! variables. Functions are members of MeshBoundaryValuesCC or MeshBoundaryValuesFC
//! classes.

#include <cstdlib>
#include <iostream>
#include <iomanip>    // std::setprecision()
#include <limits>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals.hpp"
#include "coarse_cache_ownership.hpp"
#include "mesh/prolongation.hpp" // implements prolongation operators
#include "mesh/restriction.hpp" // implements restriction operators
#include "z4c/z4c.hpp"

#include "coordinates/cell_locations.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif
//----------------------------------------------------------------------------------------
//! \fn void FillCoarseInBndryCC()
//! \brief To ensure that the coarse array is up-to-date in all neighboring cells touched
//! by the prolongation interpolation stencil, data is restricted to coarse array in
//! boundaries between MeshBlocks at the same level.

void MeshBoundaryValuesCC::FillCoarseInBndryCC(DvceArray5D<Real> &a,
                                               DvceArray5D<Real> &ca,
                                               bool is_z4c) {
  // Z4c already sends the owner-computed coarse representation to same-level
  // neighbors through isame_z4c.  Those received ghost values are the
  // authoritative cache representation.  Recomputing them from receiver-local
  // fine data selects a different block-edge stencil and creates a mixed cache.
  // The complete production iprol inventory is supplied by local restriction,
  // same/coarser receives, and physical boundaries before this point.
  if (!ShouldLocallyRefreshSameLevelCoarseCache(is_z4c)) return;

  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  //bool not_z4c = (pmbp->pz4c == nullptr)? true : false;

  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR
  int nmnv = nmb*nnghbr*nvar;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &rbuf = recvbuf;
  auto &indcs  = pmy_pack->pmesh->mb_indcs;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  auto &nx1 = pmy_pack->pmesh->mb_indcs.nx1;
  auto &nx2 = pmy_pack->pmesh->mb_indcs.nx2;
  auto &nx3 = pmy_pack->pmesh->mb_indcs.nx3;
  const int fine_n1 = a.extent_int(4);
  const int fine_n2 = a.extent_int(3);
  const int fine_n3 = a.extent_int(2);
  auto& restrict_2nd = pmy_pack->pmesh->pmr->weights.restrict_2nd;
  auto& restrict_4th = pmy_pack->pmesh->pmr->weights.restrict_4th;
  auto& restrict_4th_edge = pmy_pack->pmesh->pmr->weights.restrict_4th_edge;
  const int z4c_stencil =
      is_z4c ? pmy_pack->pz4c->opt.fd_stencil : indcs.ng;
  const bool limited_o2 =
      is_z4c && pmy_pack->pz4c->opt.amr_transfer ==
                    z4c::Z4cAMRTransfer::limited_o2;
  if (is_z4c && (z4c_stencil < 2 || z4c_stencil > 4 ||
                 z4c_stencil > indcs.ng)) {
    std::cerr << "### FATAL ERROR in " << __FILE__
              << ": invalid Z4c boundary restriction stencil "
              << z4c_stencil << " for allocated nghost=" << indcs.ng
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Restrict data into coarse array in any boundary filled with data from the same
  // level.  This ensures data in the coarse array at corners where one direction is a
  // coarser level and the other the same level is filled properly.
  // (Only needed in multidimensions)

  if (multi_d) {
    auto &cis = indcs.cis;
    auto &cjs = indcs.cjs;
    auto &cks = indcs.cks;
    // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of variables)
    Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
    Kokkos::parallel_for("ProlCCSame", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
      const int m = (tmember.league_rank())/(nnghbr*nvar);
      const int n = (tmember.league_rank() - m*(nnghbr*nvar))/nvar;
      const int v = (tmember.league_rank() - m*(nnghbr*nvar) - n*nvar);

      // only restrict when neighbor exists and is at SAME level
      if ((nghbr.d_view(m,n).gid >= 0) && (nghbr.d_view(m,n).lev == mblev.d_view(m))) {
        // loop over indices for receives at same level, but convert loop limits to
        // coarse array
        int il = (rbuf[n].isame[0].bis + cis)/2;
        int iu = (rbuf[n].isame[0].bie + cis)/2;
        int jl = (rbuf[n].isame[0].bjs + cjs)/2;
        int ju = (rbuf[n].isame[0].bje + cjs)/2;
        int kl = (rbuf[n].isame[0].bks + cks)/2;
        int ku = (rbuf[n].isame[0].bke + cks)/2;

        // Integer endpoint conversion can manufacture an unrepresentable
        // coarse target when a same-level receive ghost band has odd width.
        // Retain exactly the targets backed by a complete stored fine pair.
        const auto irange = CompleteFinePairCoarseRange(
            il, iu, indcs.cis, indcs.is, fine_n1);
        const auto jrange = CompleteFinePairCoarseRange(
            jl, ju, indcs.cjs, indcs.js, fine_n2);
        il = irange.lower;
        iu = irange.upper;
        jl = jrange.lower;
        ju = jrange.upper;
        if (three_d) {
          const auto krange = CompleteFinePairCoarseRange(
              kl, ku, indcs.cks, indcs.ks, fine_n3);
          kl = krange.lower;
          ku = krange.upper;
        }

        const int ni = iu - il + 1;
        const int nj = ju - jl + 1;
        const int nk = ku - kl + 1;
        const int nkji = nk*nj*ni;
        const int nji  = nj*ni;

        // Middle loop over k,j,i
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji),[&](const int idx) {
          int k = idx/nji;
          int j = (idx - k*nji)/ni;
          int i = (idx - k*nji - j*ni) + il;
          j += jl;
          k += kl;

          // indices refer to coarse array.  So must compute indices for fine array
          int finei = (i - indcs.cis)*2 + indcs.is;
          int finej = (j - indcs.cjs)*2 + indcs.js;
          int finek = (k - indcs.cks)*2 + indcs.ks;

          // restrict in 2D
          if (!(three_d)) {
            if (!is_z4c || limited_o2) {
              ca(m,v,kl,j,i) =
                  0.25*(a(m,v,kl,finej  ,finei) + a(m,v,kl,finej  ,finei+1)
                      + a(m,v,kl,finej+1,finei) + a(m,v,kl,finej+1,finei+1));
            } else {
              switch (z4c_stencil) {
                case 2: ca(m,v,kl,j,i) = RestrictInterpolation<2>(m,v,indcs.ks,
                            finej,finei,nx1,nx2,nx3,a,restrict_2nd,restrict_4th,
                            restrict_4th_edge);
                        break;
                case 3: ca(m,v,kl,j,i) = RestrictInterpolation<3>(m,v,indcs.ks,
                            finej,finei,nx1,nx2,nx3,a,restrict_2nd,restrict_4th,
                            restrict_4th_edge);
                        break;
                case 4: ca(m,v,kl,j,i) = RestrictInterpolation<4>(m,v,indcs.ks,
                            finej,finei,nx1,nx2,nx3,a,restrict_2nd,restrict_4th,
                            restrict_4th_edge);
                        break;
              }
            }
          // restrict in 3D
          } else {
            if (!is_z4c || limited_o2) {
              ca(m,v,k,j,i) = 0.125*(
                  a(m,v,finek  ,finej  ,finei) + a(m,v,finek  ,finej  ,finei+1)
                + a(m,v,finek  ,finej+1,finei) + a(m,v,finek  ,finej+1,finei+1)
                + a(m,v,finek+1,finej,  finei) + a(m,v,finek+1,finej,  finei+1)
                + a(m,v,finek+1,finej+1,finei) + a(m,v,finek+1,finej+1,finei+1));
            } else {
                switch (z4c_stencil) {
                  case 2: ca(m,v,k,j,i) = RestrictInterpolation<2>(m,v,finek,finej,finei,
                              nx1,nx2,nx3,a,restrict_2nd,restrict_4th,restrict_4th_edge);
                          break;
                  case 3: ca(m,v,k,j,i) = RestrictInterpolation<3>(m,v,finek,finej,finei,
                              nx1,nx2,nx3,a,restrict_2nd,restrict_4th,restrict_4th_edge);
                          break;
                  case 4: ca(m,v,k,j,i) = RestrictInterpolation<4>(m,v,finek,finej,finei,
                              nx1,nx2,nx3,a,restrict_2nd,restrict_4th,restrict_4th_edge);
                          break;
                }
            }
          }
        });
      }
      tmember.team_barrier();
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ProlongateCC()
//! \brief Prolongate data at boundaries for cell-centered data.
//! Code here is based on MeshRefinement::ProlongateCellCenteredValues() in C++ version

void MeshBoundaryValuesCC::ProlongateCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca,
    bool is_z4c) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;

  // ptr to z4c, which requires different prolongation/restriction scheme
  //bool not_z4c = (pmbp->pz4c == nullptr)? true : false;

  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR
  int nmnv = nmb*nnghbr*nvar;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &rbuf = recvbuf;
  auto &indcs  = pmy_pack->pmesh->mb_indcs;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  auto &nx1 = indcs.nx1;
  auto &nx2 = indcs.nx2;
  auto &nx3 = indcs.nx3;
  auto& prolong_2nd = pmy_pack->pmesh->pmr->weights.prolong_2nd;
  auto& prolong_4th = pmy_pack->pmesh->pmr->weights.prolong_4th;
  const int z4c_stencil =
      is_z4c ? pmy_pack->pz4c->opt.fd_stencil : indcs.ng;
  const bool limited_o2 =
      is_z4c && pmy_pack->pz4c->opt.amr_transfer ==
                    z4c::Z4cAMRTransfer::limited_o2;
  if (is_z4c && (z4c_stencil < 2 || z4c_stencil > 4 ||
                 z4c_stencil > indcs.ng)) {
    std::cerr << "### FATAL ERROR in " << __FILE__
              << ": invalid Z4c boundary prolongation stencil "
              << z4c_stencil << " for allocated nghost=" << indcs.ng
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  DvceArray1D<unsigned long long> chi_prolongation_counts(
      "Z4c boundary chi prolongation status counts", 4);
  Kokkos::deep_copy(chi_prolongation_counts, 0ULL);
  DvceArray1D<unsigned long long> chi_first_rejected_keys(
      "Z4c boundary first rejected chi groups", 2);
  Kokkos::deep_copy(chi_first_rejected_keys,
                    std::numeric_limits<unsigned long long>::max());
  auto &mb_gid = pmy_pack->pmb->mb_gid;

  // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of variables)
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("ProlCC", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(nnghbr*nvar);
    const int n = (tmember.league_rank() - m*(nnghbr*nvar))/nvar;
    const int v = (tmember.league_rank() - m*(nnghbr*nvar) - n*nvar);

    // only prolongate when neighbor exists and is at coarser level
    if ((nghbr.d_view(m,n).gid >= 0) && (nghbr.d_view(m,n).lev < mblev.d_view(m))) {
      // loop over indices for prolongation on this buffer
      int il = rbuf[n].iprol[0].bis;
      int iu = rbuf[n].iprol[0].bie;
      int jl = rbuf[n].iprol[0].bjs;
      int ju = rbuf[n].iprol[0].bje;
      int kl = rbuf[n].iprol[0].bks;
      int ku = rbuf[n].iprol[0].bke;
      const int ni = iu - il + 1;
      const int nj = ju - jl + 1;
      const int nk = ku - kl + 1;
      const int nkji = nk*nj*ni;
      const int nji  = nj*ni;

      // Middle loop over k,j,i
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
        int k = idx/nji;
        int j = (idx - k*nji)/ni;
        int i = (idx - k*nji - j*ni) + il;
        j += jl;
        k += kl;

        // indices for prolongation refer to coarse array.  So must compute
        // indices for fine array
        int fi = (i - indcs.cis)*2 + indcs.is;
        int fj = (j - indcs.cjs)*2 + indcs.js;
        int fk = (k - indcs.cks)*2 + indcs.ks;
        // call inlined prolongation operator for CC variables
        if (!is_z4c) {
          ProlongCC(m,v,k,j,i,fk,fj,fi,multi_d,three_d,ca,a);
        } else if (limited_o2) {
          if (v == z4c::Z4c::I_Z4C_CHI) {
            const ChiProlongationStatus status = ProlongLimitedPositiveChiCC(
                m,v,k,j,i,fk,fj,fi,multi_d,three_d,ca,a);
            Kokkos::atomic_inc(&chi_prolongation_counts(static_cast<int>(status)));
            if (status == ChiProlongationStatus::invalid_parent ||
                status == ChiProlongationStatus::invalid_limited) {
              const int rejection =
                  status == ChiProlongationStatus::invalid_parent ? 0 : 1;
              const unsigned long long key =
                  (static_cast<unsigned long long>(mb_gid.d_view(m)) << 36) |
                  (static_cast<unsigned long long>(n) << 30) |
                  (static_cast<unsigned long long>(k) << 20) |
                  (static_cast<unsigned long long>(j) << 10) |
                  static_cast<unsigned long long>(i);
              Kokkos::atomic_min(&chi_first_rejected_keys(rejection), key);
            }
          } else {
            ProlongCC(m,v,k,j,i,fk,fj,fi,multi_d,three_d,ca,a);
          }
        } else if (v == z4c::Z4c::I_Z4C_CHI) {
          ChiProlongationStatus status = ChiProlongationStatus::invalid_limited;
          switch (z4c_stencil) {
            case 2:
              status = ProlongPositiveChiCC<2>(m,v,k,j,i,fk,fj,fi,nx1,nx2,nx3,
                                               multi_d,three_d,ca,a,prolong_2nd);
              break;
            case 3:
              status = ProlongPositiveChiCC<3>(m,v,k,j,i,fk,fj,fi,nx1,nx2,nx3,
                                               multi_d,three_d,ca,a,prolong_4th);
              break;
            case 4:
              status = ProlongPositiveChiCC<4>(m,v,k,j,i,fk,fj,fi,nx1,nx2,nx3,
                                               multi_d,three_d,ca,a,prolong_4th);
              break;
          }
          Kokkos::atomic_inc(&chi_prolongation_counts(static_cast<int>(status)));
          if (status == ChiProlongationStatus::invalid_parent ||
              status == ChiProlongationStatus::invalid_limited) {
            const int rejection =
                status == ChiProlongationStatus::invalid_parent ? 0 : 1;
            const unsigned long long key =
                (static_cast<unsigned long long>(mb_gid.d_view(m)) << 36) |
                (static_cast<unsigned long long>(n) << 30) |
                (static_cast<unsigned long long>(k) << 20) |
                (static_cast<unsigned long long>(j) << 10) |
                static_cast<unsigned long long>(i);
            Kokkos::atomic_min(&chi_first_rejected_keys(rejection), key);
          }
        } else {
          switch (z4c_stencil) {
            case 2: HighOrderProlongCC<2>(m,v,k,j,i,fk,fj,fi,nx1,nx2,nx3,
                                          ca,a,prolong_2nd);
                    break;
            case 3: HighOrderProlongCC<3>(m,v,k,j,i,fk,fj,fi,nx1,nx2,nx3,
                                          ca,a,prolong_4th);
                    break;
            case 4: HighOrderProlongCC<4>(m,v,k,j,i,fk,fj,fi,nx1,nx2,nx3,
                                          ca,a,prolong_4th);
                    break;
          }
        }
      });
    }
    tmember.team_barrier();
  });

  if (is_z4c) {
    const auto host_counts =
        Kokkos::create_mirror_view_and_copy(HostMemSpace(), chi_prolongation_counts);
    unsigned long long local_counts[4] = {
        host_counts(0), host_counts(1), host_counts(2), host_counts(3)};
    unsigned long long global_counts[4] = {
        local_counts[0], local_counts[1], local_counts[2], local_counts[3]};
    const auto host_rejected_keys =
        Kokkos::create_mirror_view_and_copy(HostMemSpace(), chi_first_rejected_keys);
    unsigned long long global_rejected_keys[2] = {
        host_rejected_keys(0), host_rejected_keys(1)};
    std::vector<unsigned long long> fallback_each_rank(global_variable::nranks, 0ULL);
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(local_counts, global_counts, 4, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, global_rejected_keys, 2,
                  MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
    MPI_Gather(&local_counts[1], 1, MPI_UNSIGNED_LONG_LONG,
               fallback_each_rank.data(), 1, MPI_UNSIGNED_LONG_LONG, 0,
               MPI_COMM_WORLD);
#else
    fallback_each_rank[0] = local_counts[1];
#endif
    if (global_variable::my_rank == 0 &&
        (global_counts[1] != 0 || global_counts[2] != 0 || global_counts[3] != 0)) {
      std::cout << "BOUNDARY_Z4C_CHI_PROLONGATION cycle=" << pmy_pack->pmesh->ncycle
                << " transfer="
                << z4c::Z4cAMRTransferName(pmy_pack->pz4c->opt.amr_transfer)
                << " local_fallback_groups=";
      for (int rank = 0; rank < global_variable::nranks; ++rank) {
        if (rank > 0) std::cout << ",";
        std::cout << fallback_each_rank[rank];
      }
      std::cout << " global_fallback_groups=" << global_counts[1]
                << " global_high_order_groups=" << global_counts[0]
                << " invalid_parent_stencils=" << global_counts[2]
                << " invalid_limited_groups=" << global_counts[3]
                << std::endl;
    }
    if (global_counts[2] != 0 || global_counts[3] != 0) {
      if (global_counts[2] != 0 &&
          pmy_pack->pz4c->chi_parent_provenance != nullptr) {
        pmy_pack->pz4c->chi_parent_provenance->AnalyzeBoundaryFailure(
            this, global_counts[2], global_rejected_keys[0]);
      }
      if (global_variable::my_rank == 0) {
        const unsigned long long key =
            global_counts[2] != 0 ? global_rejected_keys[0]
                                  : global_rejected_keys[1];
        const int gid = static_cast<int>(key >> 36);
        const int neighbor_slot = static_cast<int>((key >> 30) & 0x3fULL);
        const int coarse_k = static_cast<int>((key >> 20) & 0x3ffULL);
        const int coarse_j = static_cast<int>((key >> 10) & 0x3ffULL);
        const int coarse_i = static_cast<int>(key & 0x3ffULL);
        const Mesh *mesh = pmy_pack->pmesh;
        const LogicalLocation &location = mesh->lloc_eachmb[gid];
        const int relative_level = location.level - mesh->root_level;
        const Real level_scale = std::ldexp(1.0, relative_level);
        const Real block_dx1 =
            (mesh->mesh_size.x1max - mesh->mesh_size.x1min) /
            (static_cast<Real>(mesh->nmb_rootx1) * level_scale);
        const Real block_dx2 =
            (mesh->mesh_size.x2max - mesh->mesh_size.x2min) /
            (static_cast<Real>(mesh->nmb_rootx2) * level_scale);
        const Real block_x1min = mesh->mesh_size.x1min + location.lx1 * block_dx1;
        const Real block_x2min = mesh->mesh_size.x2min + location.lx2 * block_dx2;
        const Real rho = block_x1min +
            (coarse_i - indcs.cis + 0.5) * 2.0 * block_dx1 / nx1;
        const Real z = block_x2min +
            (coarse_j - indcs.cjs + 0.5) * 2.0 * block_dx2 / nx2;
        std::cerr << std::setprecision(17)
                  << "BOUNDARY_Z4C_CHI_FIRST_REJECTED cycle="
                  << pmy_pack->pmesh->ncycle
                  << " kind=" << (global_counts[2] != 0 ? "invalid_parent"
                                                              : "invalid_limited")
                  << " gid=" << gid << " level=" << relative_level
                  << " lx1=" << location.lx1 << " lx2=" << location.lx2
                  << " neighbor_slot=" << neighbor_slot
                  << " coarse_i=" << coarse_i << " coarse_j=" << coarse_j
                  << " coarse_k=" << coarse_k << " rho=" << rho
                  << " z=" << z << std::endl;
        std::cerr << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl
                  << "Z4c chi boundary prolongation rejected " << global_counts[2]
                  << " parent stencils and " << global_counts[3]
                  << " limited sibling groups; chi must remain finite and strictly "
                     "positive without floors or clipping"
                  << std::endl;
      }
      std::exit(EXIT_FAILURE);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void FillCoarseInBndryFC()
//! \brief As in the case of cell-centered variables, to ensure that the coarse field is
//! up-to-date in all neighboring cells touched by the prolongation interpolation stencil,
//! data is also restricted to coarse array in boundaries between MeshBlocks at the same
//! level.

void MeshBoundaryValuesFC::FillCoarseInBndryFC(DvceFaceFld4D<Real> &b,
                                           DvceFaceFld4D<Real> &cb) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;

  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &indcs  = pmy_pack->pmesh->mb_indcs;
  auto &mblev = pmy_pack->pmb->mb_lev;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  // Restrict data into coarse array in any boundary filled with data from the same
  // level. (Only needed in multidimensions)

  if (multi_d) {
    int nmnv = 3*nmb*nnghbr;
    auto &rbuf = recvbuf;
    auto &cis = indcs.cis;
    auto &cjs = indcs.cjs;
    auto &cks = indcs.cks;
    // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of variables)
    Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
    Kokkos::parallel_for("ProlFCSame", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
      const int m = (tmember.league_rank())/(3*nnghbr);
      const int n = (tmember.league_rank() - m*(3*nnghbr))/3;
      const int v = (tmember.league_rank() - m*(3*nnghbr) - 3*n);

      // only restrict when neighbor exists and is at SAME level
      if ((nghbr.d_view(m,n).gid >= 0) && (nghbr.d_view(m,n).lev == mblev.d_view(m))) {
        // loop over indices for receives at same level, but convert loop limits to
        // coarse array
        int il = (rbuf[n].isame[v].bis + cis)/2;
        int iu = (rbuf[n].isame[v].bie + cis)/2;
        int jl = (rbuf[n].isame[v].bjs + cjs)/2;
        int ju = (rbuf[n].isame[v].bje + cjs)/2;
        int kl = (rbuf[n].isame[v].bks + cks)/2;
        int ku = (rbuf[n].isame[v].bke + cks)/2;

        const int ni = iu - il + 1;
        const int nj = ju - jl + 1;
        const int nk = ku - kl + 1;
        const int nkji = nk*nj*ni;
        const int nji  = nj*ni;

        // Middle loop over k,j,i
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji),[&](const int idx) {
          int k = idx/nji;
          int j = (idx - k*nji)/ni;
          int i = (idx - k*nji - j*ni) + il;
          j += jl;
          k += kl;

          // indices refer to coarse array.  So must compute indices for fine array
          int fk = (k - indcs.cks)*2 + indcs.ks;
          int fj = (j - indcs.cjs)*2 + indcs.js;
          int fi = (i - indcs.cis)*2 + indcs.is;

          // restrict in 2D
          if (!(three_d)) {
            if (v==0) {
              cb.x1f(m,kl,j,i) = 0.5*(b.x1f(m,kl,fj,fi) + b.x1f(m,kl,fj+1,fi));
            } else if (v==1) {
              cb.x2f(m,kl,j,i) = 0.5*(b.x2f(m,kl,fj,fi) + b.x2f(m,kl,fj,fi+1));
            } else {
              Real b3c = 0.25*(b.x3f(m,kl,fj  ,fi) + b.x3f(m,kl,fj  ,fi+1)
                             + b.x3f(m,kl,fj+1,fi) + b.x3f(m,kl,fj+1,fi+1));
              cb.x3f(m,kl  ,j,i) = b3c;
              cb.x3f(m,kl+1,j,i) = b3c;
            }

          // restrict in 3D
          } else {
            if (v==0) {
              cb.x1f(m,k,j,i) = 0.25*(b.x1f(m,fk  ,fj,fi) + b.x1f(m,fk  ,fj+1,fi)
                                    + b.x1f(m,fk+1,fj,fi) + b.x1f(m,fk+1,fj+1,fi));
            } else if (v==1) {
              cb.x2f(m,k,j,i) = 0.25*(b.x2f(m,fk  ,fj,fi) + b.x2f(m,fk  ,fj,fi+1)
                                    + b.x2f(m,fk+1,fj,fi) + b.x2f(m,fk+1,fj,fi+1));
            } else {
              cb.x3f(m,k,j,i) = 0.25*(b.x3f(m,fk,fj  ,fi) + b.x3f(m,fk,fj  ,fi+1)
                                    + b.x3f(m,fk,fj+1,fi) + b.x3f(m,fk,fj+1,fi+1));
            }
          }
        });
      }
      tmember.team_barrier();
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ProlongateFC()
//! \brief Prolongate data at boundaries for face-centered data (e.g. magnetic fields).

void MeshBoundaryValuesFC::ProlongateFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;

  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &indcs  = pmy_pack->pmesh->mb_indcs;
  auto &mblev = pmy_pack->pmb->mb_lev;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  // Prolongate b.x1f/b.x2f/b.x3f at all shared coarse/fine cell edges
  // Code here is based on MeshRefinement::ProlongateSharedFieldX1/2/3() and
  // MeshRefinement::ProlongateInternalField() in C++ version

  // Outer loop over (# of MeshBlocks)*(# of buffers)*(three field components)
  {int nmnv = 3*nmb*nnghbr;
  auto &rbuf = recvbuf;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("ProFC-2d-shared", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(3*nnghbr);
    const int n = (tmember.league_rank() - m*(3*nnghbr))/3;
    const int v = (tmember.league_rank() - m*(3*nnghbr) - 3*n);

    // only prolongate when neighbor exists and is at coarser level
    if ((nghbr.d_view(m,n).gid >= 0) && (nghbr.d_view(m,n).lev < mblev.d_view(m))) {
      int il = rbuf[n].iprol[v].bis;
      int iu = rbuf[n].iprol[v].bie;
      int jl = rbuf[n].iprol[v].bjs;
      int ju = rbuf[n].iprol[v].bje;
      int kl = rbuf[n].iprol[v].bks;
      int ku = rbuf[n].iprol[v].bke;
      const int ni = iu - il + 1;
      const int nj = ju - jl + 1;
      const int nk = ku - kl + 1;
      const int nkji = nk*nj*ni;
      const int nji  = nj*ni;

      // Middle loop over k,j,i
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember,nkji),[&](const int idx) {
        int k = idx/nji;
        int j = (idx - k*nji)/ni;
        int i = (idx - k*nji - j*ni) + il;
        j += jl;
        k += kl;

        int fi = (i - indcs.cis)*2 + indcs.is;                   // fine i
        int fj = (multi_d)? ((j - indcs.cjs)*2 + indcs.js) : j;  // fine j
        int fk = (three_d)? ((k - indcs.cks)*2 + indcs.ks) : k;  // fine k

        // Prolongate face-centered fields at shared faces betwen fine and coarse cells
        // by calling inlined prolongation operator for FC variables
        if (v==0) {
          ProlongFCSharedX1Face(m,k,j,i,fk,fj,fi,multi_d,three_d,cb.x1f,b.x1f);
        } else if (v==1) {
          ProlongFCSharedX2Face(m,k,j,i,fk,fj,fi,three_d,cb.x2f,b.x2f);
        } else {
          ProlongFCSharedX3Face(m,k,j,i,fk,fj,fi,multi_d,cb.x3f,b.x3f);
        }
      });
    }
    tmember.team_barrier();
  });}

  // Now prolongate b.x1f/b.x2f/b.x3f at interior fine cells using the 2nd-order
  // divergence-preserving interpolation scheme of Toth & Roe, JCP 180, 736 (2002).
  // Note prolongation at shared coarse/fine cell edges must be completed first as
  // interpolation formulae use these values.

  // Outer loop over (# of MeshBlocks)*(# of buffers)
  {int nmn = nmb*nnghbr;
  bool &one_d = pmy_pack->pmesh->one_d;
  auto &rbuf = recvbuf;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmn, Kokkos::AUTO);
  Kokkos::parallel_for("ProFC-2d-int", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(nnghbr);
    const int n = (tmember.league_rank() - m*(nnghbr));

    // only prolongate when neighbor exists and is at coarser level
    if ((nghbr.d_view(m,n).gid >= 0) && (nghbr.d_view(m,n).lev < mblev.d_view(m))) {
      // use prolongation indices of different field components for interior fine cells
      int il = rbuf[n].iprol[2].bis;
      int iu = rbuf[n].iprol[2].bie;
      int jl = rbuf[n].iprol[0].bjs;
      int ju = rbuf[n].iprol[0].bje;
      int kl = rbuf[n].iprol[1].bks;
      int ku = rbuf[n].iprol[1].bke;
      const int ni = iu - il + 1;
      const int nj = ju - jl + 1;
      const int nk = ku - kl + 1;
      const int nkji = nk*nj*ni;
      const int nji  = nj*ni;

      // Middle loop over k,j,i
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember,nkji),[&](const int idx) {
        int k = idx/nji;
        int j = (idx - k*nji)/ni;
        int i = (idx - k*nji - j*ni) + il;
        j += jl;
        k += kl;

        int fi = (i - indcs.cis)*2 + indcs.is;   // fine i
        int fj = (j - indcs.cjs)*2 + indcs.js;   // fine j
        int fk = (k - indcs.cks)*2 + indcs.ks;   // fine k

        if (one_d) {
          // In 1D, interior face field is trivial
          b.x1f(m,fk,fj,fi+1) = 0.5*(b.x1f(m,fk,fj,fi) + b.x1f(m,fk,fj,fi+2));
        } else {
          // in multi-D call inlined prolongation operator for FC fields at internal faces
          ProlongFCInternal(m,fk,fj,fi,three_d,b);
        }
      });
    }
    tmember.team_barrier();
  });}

  return;
}
