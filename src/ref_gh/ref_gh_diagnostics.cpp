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

namespace ref_gh {

void RefGh::SetADMVariables(MeshBlockPack *pack) { pack->prefgh->RefGhToADM(); }

void RefGh::RefGhToADM() {
  if (pmy_pack->padm == nullptr) return;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const int n2 = indcs.nx2 + 2*indcs.ng;
  const int n3 = indcs.nx3 + 2*indcs.ng;
  const auto state = u0;
  const auto table = reference_table;
  const auto adm_vars = pmy_pack->padm->adm;
  const int reference_kind = opt.reference_kind;
  const Real mass = opt.reference_mass;
  const Real center_x = opt.reference_center[0];
  const Real center_y = opt.reference_center[1];
  const Real center_z = opt.reference_center[2];
  const Real time = pmy_pack->pmesh->time;
  par_for("ref_gh to ADM", DevExeSpace(), 0, pmy_pack->nmb_thispack - 1,
  0, n3 - 1, 0, n2 - 1, 0, n1 - 1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                               size.d_view(m).x1min, size.d_view(m).x1max);
    const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                               size.d_view(m).x2min, size.d_view(m).x2max);
    const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                               size.d_view(m).x3min, size.d_view(m).x3max);
    const ReferenceGeometry reference = GetReferenceGeometry(
        reference_kind, table, mass, center_x, center_y, center_z, time, x, y, z);
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

void RefGh::UpdateDiagnostics() {
  RefGhToADM();
  switch (opt.fd_order) {
    case 2: CalcConstraints<2>(); break;
    case 4: CalcConstraints<3>(); break;
    case 6: CalcConstraints<4>(); break;
  }
}

}  // namespace ref_gh
