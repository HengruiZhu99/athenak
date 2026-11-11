//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2026 the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
#include "utils/circumbinary_budget.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/adm.hpp"
#include "cyclic_zoom/cyclic_zoom.hpp"
#include "dyn_grmhd/valencia_diagnostics.hpp"

circumbinary::RegionMoments ReduceBinaryRegions(MeshBlockPack* pack,
                                                circumbinary::BudgetGeometry geo,
                                                bool with_torque) {
  auto ind = pack->pmesh->mb_indcs;
  auto size = pack->pmb->mb_size;
  auto cons = pack->pmhd->u0;
  auto prim = pack->pmhd->w0;
  auto bcc = pack->pmhd->bcc0;
  auto adm = pack->padm->adm;
  Real gamma = pack->pmhd->peos->eos_data.gamma;
  bool frozen = pack->pmesh->pzoom && pack->pmesh->pzoom->zstate.zone > 0;
  ZoomRegion zoom{};
  if (frozen) {
    zoom = pack->pmesh->pzoom->zregion;
  }
  int cells = ind.nx1 * ind.nx2 * ind.nx3;
  circumbinary::RegionMoments sums;
  Kokkos::parallel_reduce(
      "binary_region_moments",
      Kokkos::RangePolicy<DevExeSpace>(0, pack->nmb_thispack * cells),
      KOKKOS_LAMBDA(int index, circumbinary::RegionMoments& sum) {
        int m = index / cells, cell = index % cells;
        int i = ind.is + cell % ind.nx1, j = ind.js + (cell / ind.nx1) % ind.nx2;
        int k = ind.ks + cell / (ind.nx1 * ind.nx2);
        auto s = size.d_view(m);
        Real x = CellCenterX(i - ind.is, ind.nx1, s.x1min, s.x1max);
        Real y = CellCenterX(j - ind.js, ind.nx2, s.x2min, s.x2max);
        Real z = CellCenterX(k - ind.ks, ind.nx3, s.x3min, s.x3max);
        Real f[4] = {}, mx[4] = {}, my[4] = {}, mz[4] = {};
        int n = geo.subcells;
        Real dv = s.dx1 * s.dx2 * s.dx3 / (n * n * n);
        for (int c = 0; c < n; ++c) {
          for (int b = 0; b < n; ++b) {
            for (int a = 0; a < n; ++a) {
              Real X = x + ((a + 0.5) / n - 0.5) * s.dx1;
              Real Y = y + ((b + 0.5) / n - 0.5) * s.dx2;
              Real Z = z + ((c + 0.5) / n - 0.5) * s.dx3;
              int r = geo.Region(X, Y, Z);
              if (r < 0) {
                continue;
              }
              if (frozen && zoom.IsInRegion(X, Y, Z)) {
                sum.bad[r] += 1;
                continue;
              }
              f[r] += 1;
              mx[r] += X;
              my[r] += Y;
              mz[r] += Z;
            }
          }
        }
        for (int r = 0; r < 4; ++r) {
          if (f[r] > 0) {
            Real ux = cons(m, IM1, k, j, i), uy = cons(m, IM2, k, j, i);
            if (!Kokkos::isfinite(ux) || !Kokkos::isfinite(uy)) {
              sum.bad[r] += 1;
              continue;
            }
            // Conserved storage is already densitized. Fractional Cartesian first
            // moments.
            sum.J[r] += (mx[r] * uy - my[r] * ux) * dv;
            sum.volume[r] += f[r] * dv;
            if (with_torque) {
              Real idx[3] = {1 / s.dx1, 1 / s.dx2, 1 / s.dx3}, tau[3];
              ValenciaTorqueCell(adm, prim, bcc, ind.ng, 3, gamma, idx, m, k, j, i,
                                 mx[r] / f[r], my[r] / f[r], mz[r] / f[r], tau);
              if (Kokkos::isfinite(tau[2])) {
                sum.torque[r] += tau[2] * f[r] * dv;
              } else {
                sum.bad[r] += 1;
              }
            }
          }
        }
      },
      Kokkos::Sum<circumbinary::RegionMoments>(sums));
  return sums;
}
