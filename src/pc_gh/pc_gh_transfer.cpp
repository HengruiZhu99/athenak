//========================================================================================
// AthenaK astrophysical plasma code
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
#include <stdexcept>
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "pc_gh/pc_gh.hpp"
#include "pc_gh/transfer_target.hpp"
#include "pc_gh/intrinsic_transfer_target.hpp"
#include "pc_gh/intrinsic_restriction.hpp"

namespace pc_gh {

void PcGh::RestrictIntrinsic2D(DvceArray5D<Real> &state, DvceArray5D<Real> &coarse) {
  auto u=state,cu=coarse;
  const auto a=pmy_pack->pmesh->mb_indcs;
  par_for("intrinsic point restriction 2D",DevExeSpace(),0,pmy_pack->nmb_thispack-1,
    0,EvolvedVariables()-1,a.cjs,a.cje,a.cis,a.cie,
    KOKKOS_LAMBDA(int m,int n,int j,int i) {
      cu(m,n,a.cks,j,i)=intrinsic::PointRestrict2D(u,m,n,a.ks,2*j-a.cjs,
          2*i-a.cis,a.is,a.ie,a.js,a.je);
    });
}


template <int ORDER>
void PcGh::TransferResidualGhosts() {
  auto state = u0;
  auto residual = transfer_residual;
  auto size = pmy_pack->pmb->mb_size;
  auto bcs = pmy_pack->pmb->mb_bcs;
  auto ind = pmy_pack->pmesh->mb_indcs;
  const int nmb = pmy_pack->nmb_thispack;
  const int nk = state.extent_int(2), nj = state.extent_int(3);
  const int ni = state.extent_int(4);
  const bool intrinsic_layout = IsIntrinsic();
  const int first_aux = intrinsic_layout ? intrinsic::P : I_P1;
  const int nvars = EvolvedVariables();
  const bool collision = opt.lapse_projection_target == "collision_factorized";
  Kokkos::deep_copy(residual,state);
  par_for("PC-GH source residual",DevExeSpace(),0,nmb-1,first_aux,nvars-1,
  0,nk-1,0,nj-1,0,ni-1,KOKKOS_LAMBDA(int m,int n,int k,int j,int i) {
    Real idx[3] = {1.0/size.d_view(m).dx1,1.0/size.d_view(m).dx2,
                   1.0/size.d_view(m).dx3};
    residual(m,n,k,j,i) -= intrinsic_layout
        ? intrinsic::IntrinsicTransferTarget<ORDER>(state,m,n,k,j,i,idx)
        : LegacyBoundaryTransferTarget<ORDER>(state,m,n,k,j,i,idx,collision,ind,bcs);
  });
  Kokkos::fence();
  // A separate communicator avoids interacting with in-flight ordinary transfers.
  // Restrict the residual itself: E_c = R_E E_f, then prolong/interchange E.
  // The private coarse buffer stores E in auxiliary slots, not physical G.
  if (pmy_pack->pmesh->multilevel) {
    if (intrinsic_point_restriction)
      RestrictIntrinsic2D(transfer_residual,coarse_transfer_residual);
    else pmy_pack->pmesh->pmr->RestrictCC(transfer_residual,coarse_transfer_residual,true);
  }
  pbval_residual->InitRecv(nvars);
  pbval_residual->PackAndSendCC(transfer_residual,coarse_transfer_residual);
  while (pbval_residual->RecvAndUnpackCC(transfer_residual,coarse_transfer_residual)
         != TaskStatus::complete) {}
  if (!pmy_pack->pmesh->strictly_periodic) {
    // Residuals carry exactly the same reflection tensor parity as auxiliaries.
    // Outflow extrapolates E; it does not extrapolate an independently reset G.
    pbval_residual->Z4cBCs(pmy_pack,pbval_residual->u_in,transfer_residual,
                           coarse_transfer_residual);
  }
  if (pmy_pack->pmesh->multilevel) {
    pbval_residual->ProlongateCC(transfer_residual,coarse_transfer_residual,true);
  }
  if (!pmy_pack->pmesh->strictly_periodic) {
    // Prolongation changes tangential ghosts after the first physical fill.
    // Refresh physical faces/edges/corners from these newly available values.
    pbval_residual->Z4cBCs(pmy_pack,pbval_residual->u_in,transfer_residual,
                           coarse_transfer_residual);
  }
  while (pbval_residual->ClearSend() != TaskStatus::complete) {}
  while (pbval_residual->ClearRecv() != TaskStatus::complete) {}
  // Preserve all active G and every primary, including transferred primary ghosts.
  // Reconstruct fine OR coarse leaf ghosts from that leaf's actual primary state.
  par_for("PC-GH residual ghost reconstruction",DevExeSpace(),0,nmb-1,first_aux,nvars-1,
  0,nk-1,0,nj-1,0,ni-1,KOKKOS_LAMBDA(int m,int n,int k,int j,int i) {
    if (i >= ind.is && i <= ind.ie && j >= ind.js && j <= ind.je
        && k >= ind.ks && k <= ind.ke) return;
    Real idx[3] = {1.0/size.d_view(m).dx1,1.0/size.d_view(m).dx2,
                   1.0/size.d_view(m).dx3};
    const Real target = intrinsic_layout
        ? intrinsic::IntrinsicTransferTarget<ORDER>(state,m,n,k,j,i,idx)
        : LegacyBoundaryTransferTarget<ORDER>(state,m,n,k,j,i,idx,collision,ind,bcs);
    state(m,n,k,j,i) = target+residual(m,n,k,j,i);
  });
  Kokkos::fence();
}

void PcGh::CompleteCoherentTransfer(int operation) {
  if (opt.coherent_transfer == "none") return;
  if (!pmy_pack->pmesh->strictly_periodic) {
    // Complete primary physical corners after ordinary prolongation. Record
    // this independently from the auxiliary-only correction below.
    BeginStateBudget(13);
    pbval_u->Z4cBCs(pmy_pack,pbval_u->u_in,u0,coarse_u0);
    EndStateBudget(13);
  }
  BeginStateBudget(operation);
  switch (opt.spatial_order) {
    case 2: TransferResidualGhosts<2>(); break;
    case 4: TransferResidualGhosts<4>(); break;
    case 6: TransferResidualGhosts<6>(); break;
    default: throw std::runtime_error("unsupported coherent transfer order");
  }
  EndStateBudget(operation);
}
}  // namespace pc_gh
