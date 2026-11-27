//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2026 the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
#include "utils/circumbinary_budget.hpp"
#include <vector>
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif
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

circumbinary::ShellMoments ReduceHorizonShell(MeshBlockPack* pack,
                                              circumbinary::HorizonShell shell,
                                              bool with_sources) {
  auto ind = pack->pmesh->mb_indcs;
  auto sizes = pack->pmb->mb_size.d_view;
  auto u = pack->pmhd->u0;
  auto w = pack->pmhd->w0;
  auto bcc = pack->pmhd->bcc0;
  auto adm = pack->padm->adm;
  auto reset = pack->pmhd->diagnostic_reset;
  auto coord = pack->pcoord;
  auto floor = coord->excision_floor, flux = coord->excision_flux;
  auto weight = coord->excision_weight;
  bool excise = coord->coord_data.bh_excise, smooth = coord->coord_data.smooth_excise;
  Real gamma = pack->pmhd->peos->eos_data.gamma;
  auto zoom = pack->pmesh->pzoom;
  bool frozen = zoom && zoom->zstate.zone > 0;
  ZoomRegion region{};
  if (frozen) region = zoom->zregion;
  int cells = ind.nx1 * ind.nx2 * ind.nx3;
  circumbinary::ShellMoments result;
  Kokkos::parallel_reduce(
      "horizon_shell_moments",
      Kokkos::RangePolicy<DevExeSpace>(0, pack->nmb_thispack * cells),
      KOKKOS_LAMBDA(int index, circumbinary::ShellMoments &sum) {
        int m = index / cells, cell = index % cells;
        int i = ind.is + cell % ind.nx1,
            j = ind.js + (cell / ind.nx1) % ind.nx2;
        int k = ind.ks + cell / (ind.nx1 * ind.nx2);
        auto s = sizes(m);
        Real center[3] = {CellCenterX(i - ind.is, ind.nx1, s.x1min, s.x1max),
                          CellCenterX(j - ind.js, ind.nx2, s.x2min, s.x2max),
                          CellCenterX(k - ind.ks, ind.nx3, s.x3min, s.x3max)};
        Real dx[3] = {s.dx1, s.dx2, s.dx3}, first[3] = {};
        Real volume = shell.CellMoments(center, dx, first);
        Real lo = center[2] - 0.5 * dx[2], hi = center[2] + 0.5 * dx[2];
        if (volume <= 0)
          return;
        bool contaminated =
            with_sources &&
            (reset(m, k, j, i) ||
             (excise && (floor(m, k, j, i) || flux(m, k, j, i) ||
                         (smooth && weight(m, k, j, i) > 0))));
        if (contaminated ||
            (frozen &&
             region.Overlaps(center[0] - 0.5 * dx[0], center[0] + 0.5 * dx[0],
                             center[1] - 0.5 * dx[1], center[1] + 0.5 * dx[1],
                             lo, hi, region.radius))) {
          sum.rejected += volume;
          return;
        }
        Real idx[3], p[3], force[3] = {}, torque[3] = {};
        for (int d = 0; d < 3; ++d) {
          idx[d] = 1 / dx[d];
          first[d] /= volume;
          p[d] = u(m, IM1 + d, k, j, i);
        }
        if (with_sources) {
          // Integrate the known affine grid shift Omega x r at the same
          // centroid as r x G. Using the cell-center shift leaves an artificial
          // cut-cell torque in rotating coordinates even for identical fluid
          // states.
          Real shift_offset[3] = {-shell.frame_omega * (first[1] - center[1]),
                                  shell.frame_omega * (first[0] - center[0]),
                                  0};
          ValenciaTorqueCell(adm, w, bcc, ind.ng, 3, gamma, idx, m, k, j, i,
                             first[0], first[1], first[2], torque, force,
                             shift_offset);
        }
        // Only Jz uses Cartesian first moments in this reduction.
        Real J = first[0] * p[1] - first[1] * p[0];
        bool finite = Kokkos::isfinite(J * volume) &&
                      Kokkos::isfinite(torque[2] * volume);
        for (int d = 0; d < 3; ++d)
          finite &= Kokkos::isfinite(p[d] * volume) &&
                    Kokkos::isfinite(force[d] * volume);
        if (!finite) {
          sum.rejected += volume;
          return;
        }
        for (int d = 0; d < 3; ++d) {
          sum.stored[d] += p[d] * volume;
          sum.source[d] += force[d] * volume;
        }
        sum.stored[3] += J * volume;
        sum.source[3] += torque[2] * volume;
        sum.sampled += volume;
      },
      Kokkos::Sum<circumbinary::ShellMoments>(result));
  return result;
}

void MeasureTransferInventory(MeshBlockPack* pack, const Real origin[3],
                              circumbinary::HorizonShell shell, Real values[4]) {
  // Only dispatch cells in local leaf blocks intersecting the sphere's bounding box.
  const auto& size = pack->pmb->mb_size;
  std::vector<int> candidates;
  for (int m = 0; m < pack->nmb_thispack; ++m) {
    const auto& b = size.h_view(m);
    if (b.x1max > origin[0] - shell.outer && b.x1min < origin[0] + shell.outer &&
        b.x2max > origin[1] - shell.outer && b.x2min < origin[1] + shell.outer &&
        b.x3max > origin[2] - shell.outer && b.x3min < origin[2] + shell.outer)
      candidates.push_back(m);
  }
  DvceArray1D<int> blocks("transfer_blocks", candidates.size());
  auto mirror = Kokkos::create_mirror_view(blocks);
  for (size_t i = 0; i < candidates.size(); ++i) mirror(i) = candidates[i];
  Kokkos::deep_copy(blocks, mirror);
  const auto sizes = size.d_view;
  const auto ind = pack->pmesh->mb_indcs;
  const int cells = ind.nx1 * ind.nx2 * ind.nx3;
  const Real cx = origin[0], cy = origin[1], cz = origin[2];
  auto u = pack->pmhd->u0;
  auto reset = pack->pmhd->diagnostic_reset;
  auto c = pack->pcoord;
  auto floor = c->excision_floor, flux = c->excision_flux;
  auto weight = c->excision_weight;
  bool excise = c->coord_data.bh_excise, smooth = c->coord_data.smooth_excise;
  auto zoom = pack->pmesh->pzoom;
  bool frozen = zoom && zoom->zstate.zone > 0;
  ZoomRegion region{};
  if (frozen) region = zoom->zregion;
  Kokkos::parallel_reduce("component_transfer_inventory",
      Kokkos::RangePolicy<DevExeSpace>(0, candidates.size() * cells),
      KOKKOS_LAMBDA(int n, Real& accepted, Real& rejected, Real& mass, Real& jz) {
        int m = blocks(n / cells), cell = n % cells;
        int i = ind.is + cell % ind.nx1, j = ind.js + (cell / ind.nx1) % ind.nx2;
        int k = ind.ks + cell / (ind.nx1 * ind.nx2);
        auto b = sizes(m);
        Real x = CellCenterX(i - ind.is, ind.nx1, b.x1min, b.x1max);
        Real y = CellCenterX(j - ind.js, ind.nx2, b.x2min, b.x2max);
        Real z = CellCenterX(k - ind.ks, ind.nx3, b.x3min, b.x3max);
        Real relative[3] = {x - cx, y - cy, z - cz}, dx[3] = {b.dx1, b.dx2, b.dx3}, first[3] = {};
        Real dv = shell.CellMoments(relative, dx, first);
        if (dv <= 0) return;
        Real dm = u(m, IDN, k, j, i) * dv;
        Real dj = (first[0] + cx * dv) * u(m, IM2, k, j, i) -
                  (first[1] + cy * dv) * u(m, IM1, k, j, i);
        const bool masked = reset(m, k, j, i) ||
            (excise && (floor(m, k, j, i) || flux(m, k, j, i) ||
                        (smooth && weight(m, k, j, i) > 0)));
        const bool overlaps_frozen = frozen &&
            region.Overlaps(x - 0.5 * b.dx1, x + 0.5 * b.dx1, y - 0.5 * b.dx2,
                            y + 0.5 * b.dx2, z - 0.5 * b.dx3, z + 0.5 * b.dx3,
                            region.radius);
        if (masked || overlaps_frozen || !Kokkos::isfinite(dm) || dm <= 0 ||
            !Kokkos::isfinite(dj)) {
          rejected += dv;
          return;
        }
        accepted += dv;
        mass += dm;
        jz += dj;
      }, Kokkos::Sum<Real>(values[0]), Kokkos::Sum<Real>(values[1]),
         Kokkos::Sum<Real>(values[2]), Kokkos::Sum<Real>(values[3]));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, values, 4, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
}
