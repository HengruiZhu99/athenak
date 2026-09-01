//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pc_gh_adm.cpp
//! \brief exact ADM/PC-GH state conversion and first-derivative initialization

#include <cmath>

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "coordinates/adm.hpp"
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

  par_for("ADM to PC-GH primary fields", DevExeSpace(),
  0, nmb - 1, ksg, keg, jsg, jeg, isg, ieg,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> gamma_u;
    Real const det_gamma = adm::SpatialDet(
        adm_vars.g_dd(m, 0, 0, k, j, i), adm_vars.g_dd(m, 0, 1, k, j, i),
        adm_vars.g_dd(m, 0, 2, k, j, i), adm_vars.g_dd(m, 1, 1, k, j, i),
        adm_vars.g_dd(m, 1, 2, k, j, i), adm_vars.g_dd(m, 2, 2, k, j, i));
    Real const chi = std::cbrt(1.0/det_gamma);
    pc.chi(m, k, j, i) = chi;
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
    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        pc.gtilde(m, a, b, k, j, i) = chi*adm_vars.g_dd(m, a, b, k, j, i);
        pc.Atilde(m, a, b, k, j, i) = chi*(
            adm_vars.vK_dd(m, a, b, k, j, i)
            - trace_k*adm_vars.g_dd(m, a, b, k, j, i)/3.0);
      }
    }
    Real const alpha = adm_vars.alpha(m, k, j, i);
    pc.A(m, k, j, i) = alpha*alpha;
    pc.pi(m, k, j, i) = -trace_k;
    for (int a = 0; a < 3; ++a) {
      pc.beta(m, a, k, j, i) = adm_vars.beta_u(m, a, k, j, i);
    }
  });
  Kokkos::fence();

  par_for("ADM to PC-GH first derivatives", DevExeSpace(),
  0, nmb - 1, indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> gtilde_u;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> christoffel;
    Real idx[3] = {1.0/size.d_view(m).dx1,
                   1.0/size.d_view(m).dx2,
                   1.0/size.d_view(m).dx3};
    Real const det_gtilde = adm::SpatialDet(
        pc.gtilde(m, 0, 0, k, j, i), pc.gtilde(m, 0, 1, k, j, i),
        pc.gtilde(m, 0, 2, k, j, i), pc.gtilde(m, 1, 1, k, j, i),
        pc.gtilde(m, 1, 2, k, j, i), pc.gtilde(m, 2, 2, k, j, i));
    adm::SpatialInv(1.0/det_gtilde,
        pc.gtilde(m, 0, 0, k, j, i), pc.gtilde(m, 0, 1, k, j, i),
        pc.gtilde(m, 0, 2, k, j, i), pc.gtilde(m, 1, 1, k, j, i),
        pc.gtilde(m, 1, 2, k, j, i), pc.gtilde(m, 2, 2, k, j, i),
        &gtilde_u(0, 0), &gtilde_u(0, 1), &gtilde_u(0, 2),
        &gtilde_u(1, 1), &gtilde_u(1, 2), &gtilde_u(2, 2));

    for (int d = 0; d < 3; ++d) {
      bool const active = (d == 0) || (d == 1 && multi_d) || (d == 2 && three_d);
      pc.X(m, d, k, j, i) = active ? Dx<FD_STENCIL>(
          d, idx, pc.chi, m, k, j, i) : 0.0;
      pc.Y(m, d, k, j, i) = active ? Dx<FD_STENCIL>(
          d, idx, pc.A, m, k, j, i) : 0.0;
      for (int a = 0; a < 3; ++a) {
        state(m, BIndex(d, a), k, j, i) = active ? Dx<FD_STENCIL>(
            d, idx, pc.beta, m, a, k, j, i) : 0.0;
        for (int b = a; b < 3; ++b) {
          state(m, QIndex(d, a, b), k, j, i) = active ? Dx<FD_STENCIL>(
              d, idx, pc.gtilde, m, a, b, k, j, i) : 0.0;
        }
      }
    }

    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        for (int c = b; c < 3; ++c) {
          christoffel(a, b, c) = 0.0;
          for (int d = 0; d < 3; ++d) {
            christoffel(a, b, c) += 0.5*gtilde_u(a, d)*(
                state(m, QIndex(b, d, c), k, j, i)
                + state(m, QIndex(c, d, b), k, j, i)
                - state(m, QIndex(d, b, c), k, j, i));
          }
        }
      }
      pc.Lambda(m, a, k, j, i) = 0.0;
      for (int b = 0; b < 3; ++b) {
        for (int c = 0; c < 3; ++c) {
          pc.Lambda(m, a, k, j, i) += gtilde_u(b, c)*christoffel(a, b, c);
        }
      }
    }
  });
}

void PcGh::PcGhToADM(MeshBlockPack *pmbp) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  int const nmb = pmbp->nmb_thispack;
  int const isg = indcs.is - indcs.ng;
  int const ieg = indcs.ie + indcs.ng;
  int const jsg = pmbp->pmesh->multi_d ? indcs.js - indcs.ng : indcs.js;
  int const jeg = pmbp->pmesh->multi_d ? indcs.je + indcs.ng : indcs.je;
  int const ksg = pmbp->pmesh->three_d ? indcs.ks - indcs.ng : indcs.ks;
  int const keg = pmbp->pmesh->three_d ? indcs.ke + indcs.ng : indcs.ke;
  auto &pc = pmbp->ppcgh->u;
  auto &adm_vars = pmbp->padm->adm;

  par_for("PC-GH to ADM", DevExeSpace(),
  0, nmb - 1, ksg, keg, jsg, jeg, isg, ieg,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real const chi = pc.chi(m, k, j, i);
    adm_vars.psi4(m, k, j, i) = 1.0/chi;
    adm_vars.alpha(m, k, j, i) = std::sqrt(pc.A(m, k, j, i));
    for (int a = 0; a < 3; ++a) {
      adm_vars.beta_u(m, a, k, j, i) = pc.beta(m, a, k, j, i);
      for (int b = a; b < 3; ++b) {
        adm_vars.g_dd(m, a, b, k, j, i) = pc.gtilde(m, a, b, k, j, i)/chi;
        adm_vars.vK_dd(m, a, b, k, j, i) = (
            pc.Atilde(m, a, b, k, j, i)
            + pc.gtilde(m, a, b, k, j, i)*pc.K(m, k, j, i)/3.0)/chi;
      }
    }
  });
}

template void PcGh::ADMToPcGh<2>(MeshBlockPack *pmbp);
template void PcGh::ADMToPcGh<3>(MeshBlockPack *pmbp);
template void PcGh::ADMToPcGh<4>(MeshBlockPack *pmbp);

}  // namespace pc_gh
