//========================================================================================
//! \file ref_gh_diagnostics.cpp
//! \brief ADM reconstruction and constraint refresh for reference-frame GH.
//========================================================================================
#include <cmath>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "ref_gh/ref_gh.hpp"
#include "ref_gh/ref_gh_geometry.hpp"
#include "ref_gh/reference_cache.hpp"

namespace ref_gh {

void RefGh::SetADMVariables(MeshBlockPack *pack) { pack->prefgh->RefGhToADM(); }

void RefGh::RefGhToADM() {
  if (pmy_pack->padm == nullptr) return;
  FillReferenceCache(pmy_pack->pmesh->time, false);
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const int n2 = indcs.nx2 + 2*indcs.ng;
  const int n3 = indcs.nx3 + 2*indcs.ng;
  const auto state = u0;
  const auto reference_cache = reference_evolution;
  const auto reference_extra = reference_diagnostic;
  const auto adm_vars = pmy_pack->padm->adm;
  par_for("ref_gh to ADM", DevExeSpace(), 0, pmy_pack->nmb_thispack - 1,
  0, n3 - 1, 0, n2 - 1, 0, n1 - 1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const ReferenceCachePoint reference{
        reference_cache, reference_extra, m, k, j, i};
    Real psi[4][4], pi[4][4], phi[3][4][4], d_psi[4][4][4]; // NOLINT
    Real metric[4][4], d_metric[4][4][4]; // NOLINT
    CoordinateGhGeometry geometry;
    Real determinant = 0.0;
    if (!LoadPointGeometry(state, reference, m, k, j, i, psi, pi, phi, d_psi,
                           metric, d_metric, geometry, determinant)) {
      adm_vars.alpha(m, k, j, i) = NAN;
      return;
    }
    adm_vars.alpha(m, k, j, i) = geometry.lapse;
    for (int a = 0; a < 3; ++a) {
      adm_vars.beta_u(m, a, k, j, i) = geometry.shift[a];
      for (int b = a; b < 3; ++b) {
        adm_vars.g_dd(m, a, b, k, j, i) = metric[a + 1][b + 1];
        adm_vars.vK_dd(m, a, b, k, j, i) =
            -geometry.lapse*geometry.christoffel[0][a + 1][b + 1];
      }
    }
    const Real det_spatial = adm::SpatialDet(
        metric[1][1], metric[1][2], metric[1][3], metric[2][2],
        metric[2][3], metric[3][3]);
    adm_vars.psi4(m, k, j, i) = Kokkos::pow(det_spatial, 1.0/3.0);
  });
}

void RefGh::CacheMetricCondition() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  const auto constraints = u_con;
  const auto adm_vars = pmy_pack->padm->adm;
  Kokkos::parallel_for(
      "ref_gh cache metric condition", Kokkos::RangePolicy<>(DevExeSpace(),
      0, pmy_pack->nmb_thispack*ncells), KOKKOS_LAMBDA(const int idx) {
        int work = idx;
        const int i = work % indcs.nx1 + indcs.is; work /= indcs.nx1;
        const int j = work % indcs.nx2 + indcs.js; work /= indcs.nx2;
        const int k = work % indcs.nx3 + indcs.ks;
        const int m = work/indcs.nx3;
        const Real frame_scale = constraints(
            m, kMetricConditionDiagnostic, k, j, i);
        const Real scale2 = frame_scale*frame_scale;
        constraints(m, kMetricConditionDiagnostic, k, j, i) =
            SymmetricConditionNumber3(
                scale2*adm_vars.g_dd(m, 0, 0, k, j, i),
                scale2*adm_vars.g_dd(m, 0, 1, k, j, i),
                scale2*adm_vars.g_dd(m, 0, 2, k, j, i),
                scale2*adm_vars.g_dd(m, 1, 1, k, j, i),
                scale2*adm_vars.g_dd(m, 1, 2, k, j, i),
                scale2*adm_vars.g_dd(m, 2, 2, k, j, i));
      });
}

void RefGh::UpdateDiagnostics() {
  FillReferenceCache(pmy_pack->pmesh->time, true);
  DebugFence("ref_gh diagnostics reference");
  RefGhToADM();
  DebugFence("ref_gh diagnostics ADM reconstruction");
  switch (opt.fd_order) {
    case 2: CalcConstraints<2>(); break;
    case 4: CalcConstraints<3>(); break;
    case 6: CalcConstraints<4>(); break;
  }
  DebugFence("ref_gh diagnostics constraints");
  CacheMetricCondition();
  DebugFence("ref_gh diagnostics metric condition");
}

}  // namespace ref_gh
