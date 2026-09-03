//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pc_gh_adm.cpp
//! \brief ADM conversion for the puncture-regular 55-field PC-GH state

#include <cmath>
#include <iostream>
#include <limits>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "pc_gh/pc_gh.hpp"
#include "utils/finite_diff.hpp"

namespace pc_gh {

template <int FD_STENCIL>
void PcGh::ADMToPcGh(MeshBlockPack *pmbp) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int const isg = indcs.is - indcs.ng;
  int const ieg = indcs.ie + indcs.ng;
  int const jsg = pmbp->pmesh->multi_d ? indcs.js - indcs.ng : indcs.js;
  int const jeg = pmbp->pmesh->multi_d ? indcs.je + indcs.ng : indcs.je;
  int const ksg = pmbp->pmesh->three_d ? indcs.ks - indcs.ng : indcs.ks;
  int const keg = pmbp->pmesh->three_d ? indcs.ke + indcs.ng : indcs.ke;
  int const nmb = pmbp->nmb_thispack;
  bool const multi_d = pmbp->pmesh->multi_d;
  bool const three_d = pmbp->pmesh->three_d;
  auto &pc = pmbp->ppcgh->u;
  auto &state = pmbp->ppcgh->u0;
  auto &adm_vars = pmbp->padm->adm;
  Real const division_floor = opt.initial_data_division_floor;
  Kokkos::deep_copy(state, 0.0);

  par_for("ADM to regular PC-GH primary fields", DevExeSpace(),
  0, nmb - 1, ksg, keg, jsg, jeg, isg, ieg,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> gamma_u;
    Real const det_gamma = adm::SpatialDet(
        adm_vars.g_dd(m, 0, 0, k, j, i), adm_vars.g_dd(m, 0, 1, k, j, i),
        adm_vars.g_dd(m, 0, 2, k, j, i), adm_vars.g_dd(m, 1, 1, k, j, i),
        adm_vars.g_dd(m, 1, 2, k, j, i), adm_vars.g_dd(m, 2, 2, k, j, i));
    Real const chi = std::cbrt(1.0/det_gamma);
    Real const w = std::sqrt(chi);
    Real const alpha = adm_vars.alpha(m, k, j, i);
    pc.w(m, k, j, i) = w;
    // This quotient is the defining ADM-to-PC-GH map and cannot be removed when the
    // input supplies alpha and gamma rather than rho.  It is confined to initialization;
    // the unguarded w is monitored below, and invalid values fail instead of being hidden.
    Real const guarded_w = (std::isfinite(w) && w >= 0.0)
        ? std::fmax(w, division_floor) : NAN;
    pc.rho(m, k, j, i) = (std::isfinite(w) && w >= 0.0
                            && std::isfinite(alpha) && alpha >= 0.0)
        ? alpha/guarded_w : NAN;
    adm::SpatialInv(1.0/det_gamma,
        adm_vars.g_dd(m, 0, 0, k, j, i), adm_vars.g_dd(m, 0, 1, k, j, i),
        adm_vars.g_dd(m, 0, 2, k, j, i), adm_vars.g_dd(m, 1, 1, k, j, i),
        adm_vars.g_dd(m, 1, 2, k, j, i), adm_vars.g_dd(m, 2, 2, k, j, i),
        &gamma_u(0, 0), &gamma_u(0, 1), &gamma_u(0, 2),
        &gamma_u(1, 1), &gamma_u(1, 2), &gamma_u(2, 2));

    Real trace_k = 0.0;
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        trace_k += gamma_u(a, b)*adm_vars.vK_dd(m, a, b, k, j, i);
      }
    }
    pc.K(m, k, j, i) = trace_k;
    pc.Cperp(m, k, j, i) = 0.0;
    for (int a = 0; a < 3; ++a) {
      pc.Z(m, a, k, j, i) = 0.0;
      pc.beta(m, a, k, j, i) = adm_vars.beta_u(m, a, k, j, i);
      for (int b = a; b < 3; ++b) {
        pc.gtilde(m, a, b, k, j, i) = chi*adm_vars.g_dd(m, a, b, k, j, i);
        pc.Atilde(m, a, b, k, j, i) = chi*(
            adm_vars.vK_dd(m, a, b, k, j, i)
            - trace_k*adm_vars.g_dd(m, a, b, k, j, i)/3.0);
      }
    }
  });
  Kokkos::fence();

  int const ni = ieg - isg + 1;
  int const nj = jeg - jsg + 1;
  int const nk = keg - ksg + 1;
  int const ncells = nmb*ni*nj*nk;
  Real minimum_w = std::numeric_limits<Real>::max();
  int guarded_cells = 0;
  Kokkos::parallel_reduce("PC-GH initial unguarded minimum w",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, ncells),
  KOKKOS_LAMBDA(int idx, Real &minimum) {
    int const m = idx/(ni*nj*nk);
    int const q = idx - m*ni*nj*nk;
    int const kk = q/(ni*nj);
    int const jj = (q - kk*ni*nj)/ni;
    int const ii = q - kk*ni*nj - jj*ni;
    minimum = std::fmin(minimum, pc.w(m, ksg + kk, jsg + jj, isg + ii));
  }, Kokkos::Min<Real>(minimum_w));
  Kokkos::parallel_reduce("PC-GH initial division-guard count",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, ncells),
  KOKKOS_LAMBDA(int idx, int &count) {
    int const m = idx/(ni*nj*nk);
    int const q = idx - m*ni*nj*nk;
    int const kk = q/(ni*nj);
    int const jj = (q - kk*ni*nj)/ni;
    int const ii = q - kk*ni*nj - jj*ni;
    Real const unguarded_w = pc.w(m, ksg + kk, jsg + jj, isg + ii);
    if (std::isfinite(unguarded_w) && unguarded_w >= 0.0
        && unguarded_w < division_floor) ++count;
  }, Kokkos::Sum<int>(guarded_cells));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &minimum_w, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &guarded_cells, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (global_variable::my_rank == 0) {
    std::cout << "PC-GH ADM initialization: min unguarded w=" << minimum_w
              << ", division_floor=" << division_floor
              << ", guarded cells=" << guarded_cells << std::endl;
  }
  ValidateState("ADM-to-PC-GH primary conversion", false, false);

  par_for("ADM to regular PC-GH first derivatives", DevExeSpace(),
  0, nmb - 1, indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real idx[3] = {1.0/size.d_view(m).dx1,
                   1.0/size.d_view(m).dx2,
                   1.0/size.d_view(m).dx3};
    for (int d = 0; d < 3; ++d) {
      bool const active = (d == 0) || (d == 1 && multi_d) || (d == 2 && three_d);
      pc.p(m, d, k, j, i) = active ? Dx<FD_STENCIL>(
          d, idx, pc.w, m, k, j, i) : 0.0;
      pc.L(m, d, k, j, i) = active ? 2.0*Dx<FD_STENCIL>(
          d, idx, adm_vars.alpha, m, k, j, i) : 0.0;
      for (int a = 0; a < 3; ++a) {
        state(m, BIndex(d, a), k, j, i) = active ? Dx<FD_STENCIL>(
            d, idx, pc.beta, m, a, k, j, i) : 0.0;
        for (int b = a; b < 3; ++b) {
          state(m, QIndex(d, a, b), k, j, i) = active ? Dx<FD_STENCIL>(
              d, idx, pc.gtilde, m, a, b, k, j, i) : 0.0;
        }
      }
    }
  });
  Kokkos::fence();
  ValidateState("ADM-to-PC-GH derivative conversion", false, false);
}

void PcGh::PcGhToADM(MeshBlockPack *pmbp, bool masked_only) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int const nmb = pmbp->nmb_thispack;
  int const isg = indcs.is - indcs.ng;
  int const ieg = indcs.ie + indcs.ng;
  int const jsg = pmbp->pmesh->multi_d ? indcs.js - indcs.ng : indcs.js;
  int const jeg = pmbp->pmesh->multi_d ? indcs.je + indcs.ng : indcs.je;
  int const ksg = pmbp->pmesh->three_d ? indcs.ks - indcs.ng : indcs.ks;
  int const keg = pmbp->pmesh->three_d ? indcs.ke + indcs.ng : indcs.ke;
  auto &pc = pmbp->ppcgh->u;
  auto &adm_vars = pmbp->padm->adm;
  auto &constraints = pmbp->ppcgh->u_con;
  Real const inner_radius = opt.physical_output_inner_radius;
  Real const center_x = opt.gauge_center[0];
  Real const center_y = opt.gauge_center[1];
  Real const center_z = opt.gauge_center[2];

  par_for("regular PC-GH to masked ADM diagnostics", DevExeSpace(),
  0, nmb - 1, ksg, keg, jsg, jeg, isg, ieg,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real const x = CellCenterX(i - indcs.is, indcs.nx1, size.d_view(m).x1min,
                               size.d_view(m).x1max) - center_x;
    Real const y = CellCenterX(j - indcs.js, indcs.nx2, size.d_view(m).x2min,
                               size.d_view(m).x2max) - center_y;
    Real const z = CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(m).x3min,
                               size.d_view(m).x3max) - center_z;
    bool const valid = !masked_only || x*x + y*y + z*z >= inner_radius*inner_radius;
    constraints(m, I_CON_PHYSICAL_VALID, k, j, i) = valid ? 1.0 : 0.0;
    adm_vars.alpha(m, k, j, i) = pc.rho(m, k, j, i)*pc.w(m, k, j, i);
    for (int a = 0; a < 3; ++a) {
      adm_vars.beta_u(m, a, k, j, i) = pc.beta(m, a, k, j, i);
    }
    if (valid) {
      Real const w2 = pc.w(m, k, j, i)*pc.w(m, k, j, i);
      Real const inv_w2 = 1.0/w2;
      adm_vars.psi4(m, k, j, i) = inv_w2;
      for (int a = 0; a < 3; ++a) {
        for (int b = a; b < 3; ++b) {
          adm_vars.g_dd(m, a, b, k, j, i) =
              pc.gtilde(m, a, b, k, j, i)*inv_w2;
          adm_vars.vK_dd(m, a, b, k, j, i) = (
              pc.Atilde(m, a, b, k, j, i)
              + pc.gtilde(m, a, b, k, j, i)*pc.K(m, k, j, i)/3.0)*inv_w2;
        }
      }
    } else {
      // The external horizon adapter requires a dense Cartesian cube. This finite
      // output-only extension is explicitly invalid and is never read by the evolution.
      adm_vars.psi4(m, k, j, i) = 1.0;
      for (int a = 0; a < 3; ++a) {
        for (int b = a; b < 3; ++b) {
          adm_vars.g_dd(m, a, b, k, j, i) = (a == b) ? 1.0 : 0.0;
          adm_vars.vK_dd(m, a, b, k, j, i) = 0.0;
        }
      }
    }
  });
}

template void PcGh::ADMToPcGh<2>(MeshBlockPack *pmbp);
template void PcGh::ADMToPcGh<3>(MeshBlockPack *pmbp);
template void PcGh::ADMToPcGh<4>(MeshBlockPack *pmbp);

}  // namespace pc_gh
