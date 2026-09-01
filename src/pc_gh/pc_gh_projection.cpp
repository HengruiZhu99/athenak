//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pc_gh_projection.cpp
//! \brief consistent PC-GH conformal-metric, Q, and trace-free-A projections

#include <cmath>

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "coordinates/adm.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "pc_gh/pc_gh.hpp"

namespace pc_gh {

void PcGh::ProjectAlgebraic(MeshBlockPack *pmbp) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  int const isg = indcs.is - indcs.ng;
  int const ieg = indcs.ie + indcs.ng;
  int const jsg = pmbp->pmesh->multi_d ? indcs.js - indcs.ng : indcs.js;
  int const jeg = pmbp->pmesh->multi_d ? indcs.je + indcs.ng : indcs.je;
  int const ksg = pmbp->pmesh->three_d ? indcs.ks - indcs.ng : indcs.ks;
  int const keg = pmbp->pmesh->three_d ? indcs.ke + indcs.ng : indcs.ke;
  int const nmb = pmbp->nmb_thispack;
  auto &pc = pmbp->ppcgh->u;
  auto &state = pmbp->ppcgh->u0;

  par_for("PC-GH algebraic projection", DevExeSpace(),
  0, nmb - 1, ksg, keg, jsg, jeg, isg, ieg,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> old_g;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> old_g_inv;
    Real const det_g = adm::SpatialDet(
        pc.gtilde(m, 0, 0, k, j, i), pc.gtilde(m, 0, 1, k, j, i),
        pc.gtilde(m, 0, 2, k, j, i), pc.gtilde(m, 1, 1, k, j, i),
        pc.gtilde(m, 1, 2, k, j, i), pc.gtilde(m, 2, 2, k, j, i));
    Real const scale = std::cbrt(1.0/det_g);
    adm::SpatialInv(1.0/det_g,
        pc.gtilde(m, 0, 0, k, j, i), pc.gtilde(m, 0, 1, k, j, i),
        pc.gtilde(m, 0, 2, k, j, i), pc.gtilde(m, 1, 1, k, j, i),
        pc.gtilde(m, 1, 2, k, j, i), pc.gtilde(m, 2, 2, k, j, i),
        &old_g_inv(0, 0), &old_g_inv(0, 1), &old_g_inv(0, 2),
        &old_g_inv(1, 1), &old_g_inv(1, 2), &old_g_inv(2, 2));

    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        old_g(a, b) = pc.gtilde(m, a, b, k, j, i);
      }
    }

    Real trace_a = 0.0;
    Real trace_q[3] = {0.0, 0.0, 0.0};
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        trace_a += old_g_inv(a, b)*pc.Atilde(m, a, b, k, j, i);
        for (int d = 0; d < 3; ++d) {
          trace_q[d] += old_g_inv(a, b)*state(m, QIndex(d, a, b), k, j, i);
        }
      }
    }

    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        pc.gtilde(m, a, b, k, j, i) = scale*old_g(a, b);
        pc.Atilde(m, a, b, k, j, i) =
            pc.Atilde(m, a, b, k, j, i) - old_g(a, b)*trace_a/3.0;
        for (int d = 0; d < 3; ++d) {
          int const q = QIndex(d, a, b);
          state(m, q, k, j, i) = scale*(
              state(m, q, k, j, i) - old_g(a, b)*trace_q[d]/3.0);
        }
      }
    }
  });
}

}  // namespace pc_gh
