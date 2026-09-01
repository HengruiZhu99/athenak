//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pc_gh_newdt.cpp
//! \brief characteristic CFL timestep for the PC-GH system

#include <algorithm>
#include <cmath>
#include <limits>

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "coordinates/adm.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "pc_gh/pc_gh.hpp"

namespace pc_gh {

TaskStatus PcGh::NewTimeStep(Driver *pdriver, int stage) {
  if (stage != pdriver->nexp_stages) return TaskStatus::complete;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  auto &pc = u;
  int const nx1 = indcs.nx1;
  int const nx2 = indcs.nx2;
  int const nx3 = indcs.nx3;
  int const is = indcs.is;
  int const js = indcs.js;
  int const ks = indcs.ks;
  int const nkji = nx3*nx2*nx1;
  int const nmkji = pmy_pack->nmb_thispack*nkji;
  bool const multi_d = pmy_pack->pmesh->multi_d;
  bool const three_d = pmy_pack->pmesh->three_d;
  bool const use_z4c_mp = (opt.gauge == "z4c_mp"
                            || opt.gauge == "z4c_mp_hyperbolic");
  Real local_dt = std::numeric_limits<float>::max();

  Kokkos::parallel_reduce("PC-GH characteristic dt",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(int idx, Real &min_dt) {
    int const m = idx/nkji;
    int const q = idx - m*nkji;
    int const k0 = q/(nx2*nx1);
    int const j0 = (q - k0*nx2*nx1)/nx1;
    int const i0 = q - k0*nx2*nx1 - j0*nx1;
    int const i = is + i0;
    int const j = js + j0;
    int const k = ks + k0;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> g_inv;
    Real const det_g = adm::SpatialDet(
        pc.gtilde(m, 0, 0, k, j, i), pc.gtilde(m, 0, 1, k, j, i),
        pc.gtilde(m, 0, 2, k, j, i), pc.gtilde(m, 1, 1, k, j, i),
        pc.gtilde(m, 1, 2, k, j, i), pc.gtilde(m, 2, 2, k, j, i));
    adm::SpatialInv(1.0/det_g,
        pc.gtilde(m, 0, 0, k, j, i), pc.gtilde(m, 0, 1, k, j, i),
        pc.gtilde(m, 0, 2, k, j, i), pc.gtilde(m, 1, 1, k, j, i),
        pc.gtilde(m, 1, 2, k, j, i), pc.gtilde(m, 2, 2, k, j, i),
        &g_inv(0, 0), &g_inv(0, 1), &g_inv(0, 2),
        &g_inv(1, 1), &g_inv(1, 2), &g_inv(2, 2));
    Real const alpha = std::sqrt(pc.A(m, k, j, i));
    Real const chi = pc.chi(m, k, j, i);
    Real const physical_factor = alpha*std::sqrt(chi);
    Real gauge_factor = physical_factor;
    if (use_z4c_mp) {
      gauge_factor = std::fmax(gauge_factor, 2.0/std::sqrt(3.0));
      gauge_factor = std::fmax(gauge_factor, std::sqrt(2.0*alpha*chi));
    }
    Real const speed1 = std::fabs(pc.beta(m, 0, k, j, i))
                        + gauge_factor*std::sqrt(g_inv(0, 0));
    min_dt = std::fmin(min_dt, size.d_view(m).dx1/speed1);
    if (multi_d) {
      Real const speed2 = std::fabs(pc.beta(m, 1, k, j, i))
                          + gauge_factor*std::sqrt(g_inv(1, 1));
      min_dt = std::fmin(min_dt, size.d_view(m).dx2/speed2);
    }
    if (three_d) {
      Real const speed3 = std::fabs(pc.beta(m, 2, k, j, i))
                          + gauge_factor*std::sqrt(g_inv(2, 2));
      min_dt = std::fmin(min_dt, size.d_view(m).dx3/speed3);
    }
  }, Kokkos::Min<Real>(local_dt));
  dtnew = local_dt;
  return TaskStatus::complete;
}

}  // namespace pc_gh
