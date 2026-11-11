//========================================================================================
// flux_generalized.cpp — integrate fluxes on arbitrary r(θ,φ) surfaces (Valencia MHD)
//----------------------------------------------------------------------------------------
#include "flux_generalized.hpp"
#include <algorithm>
#include <string>
#include <vector>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <limits>
#include "globals.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "eos/eos.hpp"
#include "coordinates/adm.hpp"
#include "utils/surface_grid.hpp"
#include "outputs/outputs.hpp"
#include "z4c/z4c.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "coordinates/coordinates.hpp"
#include "dyn_grmhd/valencia_surface_flux.hpp"

//----------------------------------------------------------------------------------------
// Main Function Definition (no EOS templates; gamma-law assumed)
//----------------------------------------------------------------------------------------
SurfaceFluxResult IntegrateSurface(MeshBlockPack* pmbp,
                                   SphericalSurfaceGrid* current_surf) {
  if (!pmbp->padm || !pmbp->pmhd || !pmbp->pdyngr ||
      pmbp->pdyngr->eos_policy != DynGRMHD_EOS::eos_ideal ||
      !pmbp->pmhd->peos->eos_data.is_ideal) {
    Kokkos::abort("Surface diagnostics require ideal gamma-law Valencia MHD and ADM");
  }
  if (!pmbp->pmesh->three_d) {
    Kokkos::abort("Closed surface integrals require three spatial dimensions");
  }
  SurfaceFluxResult result;
  // Gamma-law EOS parameter; must match valencia_stress_energy.hpp usage
  const Real gamma_gas = pmbp->pmhd->peos->eos_data.gamma;

  DualArray2D<Real> dSigma;

  // Ownership is rebuilt even for stationary surfaces: AMR, redistribution, restart,
  // and zoom restoration can change interpolation support without changing the shape.
  current_surf->RebuildAll();
  current_surf->QualifyStencils();
  const int np = current_surf->Npts();

  // Interpolate primitives and densitized B to the surface
  const int nvars_mhd = pmbp->pmhd->nmhd + pmbp->pmhd->nscalars;
  DualArray2D<Real> w_vals =
      current_surf->InterpolateCached(0, pmbp->pmhd->w0, 0, nvars_mhd);
  auto w_vals_d = w_vals.d_view;

  DualArray2D<Real> B_vals = current_surf->InterpolateCached(1, pmbp->pmhd->bcc0, 0, 3);
  auto B_vals_d = B_vals.d_view;

  // Lapse/shift from ADM or Z4C
  auto gauge_source_array = pmbp->padm->u_adm;
  int ab_start = adm::ADM::I_ADM_ALPHA;
  int ab_end = adm::ADM::I_ADM_BETAZ + 1;
  if (pmbp->pz4c != nullptr) {
    gauge_source_array = pmbp->pz4c->u0;
    ab_start = z4c::Z4c::I_Z4C_ALPHA;
    ab_end = z4c::Z4c::I_Z4C_BETAZ + 1;
  }
  DualArray2D<Real> ab_vals =
      current_surf->InterpolateCached(2, gauge_source_array, ab_start, ab_end);
  auto ab_vals_d = ab_vals.d_view;

  // Cartesian coordinates and interpolation indices
  auto coords = current_surf->Coords().d_view;
  auto interp_ind = current_surf->InterpIndices().d_view;
  auto owners = current_surf->Owners().d_view;
  auto reasons = current_surf->StencilReasons().d_view;
  auto motion = current_surf->Motion().d_view;
  int inherited_reason = current_surf->inherited_stencil_reason;

  // Surface metric and dSigma_i
  current_surf->InterpolateMetric();
  current_surf->BuildSurfaceCovectors(dSigma);
  auto g_surf_d = current_surf->Metric().d_view;
  auto dSigma_d = dSigma.d_view;

  FluxAccumulator fluxes;

  //--- 3. Parallel integration on the surface ---
  Kokkos::parallel_reduce(
      "flux_integral_from_prims", Kokkos::RangePolicy<DevExeSpace>(0, np),
      KOKKOS_LAMBDA(const int p, FluxAccumulator& update) {
        if (interp_ind(p, 0) == -1 || owners(p) != 1) {
          return;
        }
        update.owned += 1.0;
        int reason = reasons(p) | inherited_reason;
        for (int b = 0; b < 6; ++b) {
          if (reason & (1 << b)) {
            update.bad[b] += 1.0;
          }
        }

        // --- A. Local primitives and geometry ---
        const Real rho = w_vals_d(p, IDN);
        const Real pgas = w_vals_d(p, IPR);

        Real prim_Wv_u[3] = {w_vals_d(p, IVX), w_vals_d(p, IVY), w_vals_d(p, IVZ)};

        const Real alp = ab_vals_d(p, 0);
        const Real beta_u[3] = {ab_vals_d(p, 1), ab_vals_d(p, 2), ab_vals_d(p, 3)};

        // Spatial metric gamma_ij on the surface: store as 1D symmetric array
        Real g_dd_1d[6];
        g_dd_1d[0] = g_surf_d(p, 0);
        g_dd_1d[1] = g_surf_d(p, 1);
        g_dd_1d[2] = g_surf_d(p, 2);
        g_dd_1d[3] = g_surf_d(p, 3);
        g_dd_1d[4] = g_surf_d(p, 4);
        g_dd_1d[5] = g_surf_d(p, 5);

        // det(gamma)
        Real detg = adm::SpatialDet(g_dd_1d[0], g_dd_1d[1], g_dd_1d[2], g_dd_1d[3],
                                    g_dd_1d[4], g_dd_1d[5]);
        // Sylvester criterion, not determinant alone. No silent repairs.
        bool metric_ok = Kokkos::isfinite(alp) && alp > 0.0;
        for (int c = 0; c < 6; ++c) {
          metric_ok &= Kokkos::isfinite(g_dd_1d[c]);
        }
        metric_ok &= g_dd_1d[0] > 0.0 &&
                     g_dd_1d[0] * g_dd_1d[3] - SQR(g_dd_1d[1]) > 0.0 && detg > 0.0;
        if (!metric_ok) {
          update.bad[1] += 1.0;
          return;
        }
        bool fluid_ok =
            Kokkos::isfinite(rho) && rho > 0.0 && Kokkos::isfinite(pgas) && pgas >= 0.0;
        for (int c = 0; c < 3; ++c) {
          fluid_ok &= Kokkos::isfinite(prim_Wv_u[c]) && Kokkos::isfinite(beta_u[c]) &&
                      Kokkos::isfinite(B_vals_d(p, c)) && Kokkos::isfinite(motion(p, c));
        }
        if (!fluid_ok) {
          update.bad[2] += 1.0;
          return;
        }
        if (reason) {
          return;
        }

        // dS_i (covariant surface element)
        const Real dS_d[3] = {dSigma_d(p, 0), dSigma_d(p, 1), dSigma_d(p, 2)};

        Real B_tilde[3] = {B_vals_d(p, 0), B_vals_d(p, 1), B_vals_d(p, 2)};
        Real position[3] = {coords(p, 0), coords(p, 1), coords(p, 2)};
        Real velocity[3] = {motion(p, 0), motion(p, 1), motion(p, 2)};
        FluxAccumulator point;
        ValenciaSurfacePoint(rho, pgas, gamma_gas, prim_Wv_u, B_tilde, g_dd_1d, detg, alp,
                             beta_u, position, velocity, dS_d, point);
        Real checks[] = {point.mdot,         point.edot_fluid,   point.edot_em,
                         point.phiB,         point.area,         point.pdot_x_fluid,
                         point.pdot_y_fluid, point.pdot_z_fluid, point.pdot_x_em,
                         point.pdot_y_em,    point.pdot_z_em,    point.ldot_x_fluid,
                         point.ldot_y_fluid, point.ldot_z_fluid, point.ldot_x_em,
                         point.ldot_y_em,    point.ldot_z_em};
        bool finite = point.area > 0.0;
        for (int c = 0; c < 17; ++c) {
          finite &= Kokkos::isfinite(checks[c]);
        }
        for (int c = 0; c < 19; ++c) {
          finite &= Kokkos::isfinite(point.moving[c]);
        }
        if (!finite) {
          update.bad[2] += 1.0;
          return;
        }
        update += point;
        update.admissible += 1.0;
      },
      Kokkos::Sum<FluxAccumulator>(fluxes));

  Real legacy[17] = {fluxes.mdot,         fluxes.edot_fluid,   fluxes.edot_em,
                     fluxes.pdot_x_fluid, fluxes.pdot_y_fluid, fluxes.pdot_z_fluid,
                     fluxes.pdot_x_em,    fluxes.pdot_y_em,    fluxes.pdot_z_em,
                     fluxes.ldot_x_fluid, fluxes.ldot_y_fluid, fluxes.ldot_z_fluid,
                     fluxes.ldot_x_em,    fluxes.ldot_y_em,    fluxes.ldot_z_em,
                     fluxes.phiB,         fluxes.area};
  for (int c = 0; c < 17; ++c) {
    result.values[c] = legacy[c];
  }
  for (int c = 0; c < 19; ++c) {
    result.values[17 + c] = fluxes.moving[c];
  }
  Real stats[9] = {fluxes.owned, fluxes.admissible, fluxes.covered_area};
  for (int c = 0; c < 6; ++c) {
    stats[3 + c] = fluxes.bad[c];
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, result.values.data(), 36, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, stats, 9, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  Real dx = 0.0;
  for (int p = 0; p < np; ++p) {
    if (current_surf->Owners().h_view(p) != 1) {
      result.reason |= 1;
    }
    dx = std::max(dx, current_surf->LocalSpacing().h_view(p));
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &dx, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  for (int c = 0; c < 6; ++c) {
    if (stats[3 + c] > 0.0) {
      result.reason |= 1 << c;
    }
  }
  for (int c = 0; c < 36; ++c) {
    if (!std::isfinite(result.values[c])) {
      result.reason |= 4;
    }
  }
  result.valid = result.reason == 0 && stats[1] == np;
  result.values[36] = result.valid ? 1.0 : 0.0;
  result.values[37] = result.reason;
  result.values[38] = stats[0] / np;
  result.values[39] = stats[1] / np;
  result.values[40] = stats[2];
  result.values[41] = dx;
  result.values[42] = current_surf->min_cells_per_radius;
  result.values[43] = result.values[17];
  if (!result.valid) {
    for (int c = 0; c < 36; ++c) {
      result.values[c] = std::numeric_limits<Real>::quiet_NaN();
    }
  }
  return result;
}

void TorusFluxes_General(HistoryData* pdata, MeshBlockPack* pack,
                         const std::vector<SphericalSurfaceGrid*>& surfaces) {
  const int count = static_cast<int>(surfaces.size());
  if (count * kSurfaceChannels > NHISTORY_VARIABLES) {
    Kokkos::abort("Requested surface channels exceed history capacity");
  }
  // Preserve the complete legacy ordering, then append explicitly corrected channels.
  static const char* names[kSurfaceChannels] = {"mdot_",
                                                "edot_f_",
                                                "edot_em_",
                                                "pxdot_f_",
                                                "pydot_f_",
                                                "pzdot_f_",
                                                "pxdot_em_",
                                                "pydot_em_",
                                                "pzdot_em_",
                                                "lxdot_f_",
                                                "lydot_f_",
                                                "lzdot_f_",
                                                "lxdot_em_",
                                                "lydot_em_",
                                                "lzdot_em_",
                                                "phiB_",
                                                "area_",
                                                "M0_net_in_moving_",
                                                "M0_in_moving_",
                                                "M0_out_moving_",
                                                "Eeu_f_in_moving_",
                                                "Eeu_em_in_moving_",
                                                "Px_f_in_moving_",
                                                "Py_f_in_moving_",
                                                "Pz_f_in_moving_",
                                                "Px_em_in_moving_",
                                                "Py_em_in_moving_",
                                                "Pz_em_in_moving_",
                                                "Jx_f_in_moving_",
                                                "Jy_f_in_moving_",
                                                "Jz_f_in_moving_",
                                                "Jx_em_in_moving_",
                                                "Jy_em_in_moving_",
                                                "Jz_em_in_moving_",
                                                "Et_f_in_moving_",
                                                "Et_em_in_moving_",
                                                "valid_",
                                                "reason_",
                                                "ownership_fraction_",
                                                "valid_sample_fraction_",
                                                "partial_covered_area_",
                                                "stencil_dx_max_",
                                                "required_cells_per_radius_",
                                                "partial_M0_net_in_moving_"};
  pdata->nhist = count * kSurfaceChannels;
  for (int g = 0; g < count; ++g) {
    auto result = IntegrateSurface(pack, surfaces[g]);
    for (int c = 0; c < kSurfaceChannels; ++c) {
      int index = c < 17 ? 17 * g + c : 17 * count + 27 * g + c - 17;
      pdata->label[index] = std::string(names[c]) + surfaces[g]->Label();
      pdata->hdata[index] = global_variable::my_rank == 0 ? result.values[c] : 0.0;
    }
  }
}
