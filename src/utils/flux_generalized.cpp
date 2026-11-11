//========================================================================================
// flux_generalized.cpp — integrate fluxes on arbitrary r(θ,φ) surfaces (Valencia MHD)
//----------------------------------------------------------------------------------------
#include "flux_generalized.hpp"
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
#include "dyn_grmhd/valencia_stress_energy.hpp"

// Expanded accumulator for separated fluid/EM fluxes and all axes
struct FluxAccumulator {
  Real moving[19]={};
  Real owned=0, admissible=0, covered_area=0, dx_sum=0;
  Real bad[6]={};
  Real mdot = 0, edot_fluid = 0, edot_em = 0, phiB = 0, area = 0;
  // Linear momentum fluxes (fluid, EM)
  Real pdot_x_fluid = 0, pdot_y_fluid = 0, pdot_z_fluid = 0;
  Real pdot_x_em    = 0, pdot_y_em    = 0, pdot_z_em    = 0;
  // Angular momentum fluxes (fluid, EM for x, y, z axes)
  Real ldot_x_fluid = 0, ldot_y_fluid = 0, ldot_z_fluid = 0;
  Real ldot_x_em    = 0, ldot_y_em    = 0, ldot_z_em    = 0;

  KOKKOS_INLINE_FUNCTION
  FluxAccumulator& operator+=(const FluxAccumulator& src) {
    for (int i=0;i<19;++i) moving[i]+=src.moving[i];
    owned+=src.owned; admissible+=src.admissible; covered_area+=src.covered_area;
    for (int i=0;i<6;++i) bad[i]+=src.bad[i];
    mdot        += src.mdot;
    edot_fluid  += src.edot_fluid;
    edot_em     += src.edot_em;
    phiB        += src.phiB;
    area        += src.area;
    pdot_x_fluid += src.pdot_x_fluid;
    pdot_y_fluid += src.pdot_y_fluid;
    pdot_z_fluid += src.pdot_z_fluid;
    pdot_x_em    += src.pdot_x_em;
    pdot_y_em    += src.pdot_y_em;
    pdot_z_em    += src.pdot_z_em;
    ldot_x_fluid += src.ldot_x_fluid;
    ldot_y_fluid += src.ldot_y_fluid;
    ldot_z_fluid += src.ldot_z_fluid;
    ldot_x_em    += src.ldot_x_em;
    ldot_y_em    += src.ldot_y_em;
    ldot_z_em    += src.ldot_z_em;
    return *this;
  }
};

namespace Kokkos {
  template<>
  struct reduction_identity<FluxAccumulator> {
    KOKKOS_INLINE_FUNCTION
    static FluxAccumulator sum() { return FluxAccumulator{}; }
  };
} // namespace Kokkos

//----------------------------------------------------------------------------------------
// Main Function Definition (no EOS templates; gamma-law assumed)
//----------------------------------------------------------------------------------------
SurfaceFluxResult IntegrateSurface(MeshBlockPack *pmbp, SphericalSurfaceGrid *current_surf) {
  if (!pmbp->padm || !pmbp->pmhd || !pmbp->pdyngr ||
      pmbp->pdyngr->eos_policy != DynGRMHD_EOS::eos_ideal)
    Kokkos::abort("Surface diagnostics require ideal gamma-law Valencia MHD and ADM");
  if (!pmbp->pmesh->three_d)
    Kokkos::abort("Closed surface integrals require three spatial dimensions");
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
        current_surf->InterpolateToSurface(pmbp->pmhd->w0, 0, nvars_mhd);
    auto w_vals_d = w_vals.d_view;

    DualArray2D<Real> B_vals =
        current_surf->InterpolateToSurface(pmbp->pmhd->bcc0, 0, 3);
    auto B_vals_d = B_vals.d_view;

    // Lapse/shift from ADM or Z4C
    auto gauge_source_array = pmbp->padm->u_adm;
    int ab_start = adm::ADM::I_ADM_ALPHA;
    int ab_end   = adm::ADM::I_ADM_BETAZ + 1;
    if (pmbp->pz4c != nullptr) {
      gauge_source_array = pmbp->pz4c->u0;
      ab_start = z4c::Z4c::I_Z4C_ALPHA;
      ab_end   = z4c::Z4c::I_Z4C_BETAZ + 1;
    }
    DualArray2D<Real> ab_vals =
        current_surf->InterpolateToSurface(gauge_source_array, ab_start, ab_end);
    auto ab_vals_d = ab_vals.d_view;

    // Cartesian coordinates and interpolation indices
    auto coords     = current_surf->Coords().d_view;
    auto interp_ind = current_surf->InterpIndices().d_view;
    auto owners = current_surf->Owners().d_view;
    auto reasons = current_surf->StencilReasons().d_view;
    auto motion = current_surf->Motion().d_view;

    // Surface metric and dSigma_i
    current_surf->InterpolateMetric();
    current_surf->BuildSurfaceCovectors(dSigma);
    auto g_surf_d = current_surf->Metric().d_view;
    auto dSigma_d = dSigma.d_view;

    FluxAccumulator fluxes;

    //--- 3. Parallel integration on the surface ---
    Kokkos::parallel_reduce("flux_integral_from_prims",
      Kokkos::RangePolicy<DevExeSpace>(0, np),
      KOKKOS_LAMBDA(const int p, FluxAccumulator& update) {
        if (interp_ind(p,0) == -1 || owners(p)!=1) return;
        update.owned+=1.0;
        int reason=reasons(p);
        for (int b=0;b<6;++b) if (reason & (1<<b)) update.bad[b]+=1.0;

        // --- A. Local primitives and geometry ---
        const Real rho  = w_vals_d(p, IDN);
        const Real pgas = w_vals_d(p, IPR);

        Real prim_Wv_u[3] = {
          w_vals_d(p, IVX),
          w_vals_d(p, IVY),
          w_vals_d(p, IVZ)
        };

        const Real alp = ab_vals_d(p, 0);
        const Real beta_u[3] = {
          ab_vals_d(p, 1),
          ab_vals_d(p, 2),
          ab_vals_d(p, 3)
        };

        // Spatial metric gamma_ij on the surface: store as 1D symmetric array
        Real g_dd_1d[6];
        g_dd_1d[0] = g_surf_d(p,0);
        g_dd_1d[1] = g_surf_d(p,1);
        g_dd_1d[2] = g_surf_d(p,2);
        g_dd_1d[3] = g_surf_d(p,3);
        g_dd_1d[4] = g_surf_d(p,4);
        g_dd_1d[5] = g_surf_d(p,5);

        // det(gamma)
        Real detg = adm::SpatialDet(g_dd_1d[0], g_dd_1d[1], g_dd_1d[2],
                                    g_dd_1d[3], g_dd_1d[4], g_dd_1d[5]);
        // Sylvester criterion, not determinant alone. No silent repairs.
        bool metric_ok = Kokkos::isfinite(alp) && alp>0.0;
        for (int c=0;c<6;++c) metric_ok &= Kokkos::isfinite(g_dd_1d[c]);
        metric_ok &= g_dd_1d[0]>0.0 &&
                     g_dd_1d[0]*g_dd_1d[3]-SQR(g_dd_1d[1])>0.0 && detg>0.0;
        if (!metric_ok) { update.bad[1]+=1.0; return; }
        bool fluid_ok=Kokkos::isfinite(rho) && rho>0.0 && Kokkos::isfinite(pgas) && pgas>=0.0;
        for (int c=0;c<3;++c) {
          fluid_ok &= Kokkos::isfinite(prim_Wv_u[c]) && Kokkos::isfinite(beta_u[c]) &&
                      Kokkos::isfinite(B_vals_d(p,c)) && Kokkos::isfinite(motion(p,c));
        }
        if (!fluid_ok) { update.bad[2]+=1.0; return; }
        if (reason) return;
        update.admissible+=1.0;

        // dS_i (covariant surface element)
        const Real dS_d[3] = {
          dSigma_d(p,0),
          dSigma_d(p,1),
          dSigma_d(p,2)
        };

        // Inverse metric g^{ij}
        Real g_uu_1d[6];
        adm::SpatialInv(1.0/detg,
                        g_dd_1d[0], g_dd_1d[1], g_dd_1d[2],
                        g_dd_1d[3], g_dd_1d[4], g_dd_1d[5],
                        &g_uu_1d[0], &g_uu_1d[1], &g_uu_1d[2],
                        &g_uu_1d[3], &g_uu_1d[4], &g_uu_1d[5]);

        Real g_u[3][3];
        g_u[0][0] = g_uu_1d[0];
        g_u[0][1] = g_uu_1d[1];
        g_u[0][2] = g_uu_1d[2];
        g_u[1][0] = g_uu_1d[1];
        g_u[1][1] = g_uu_1d[3];
        g_u[1][2] = g_uu_1d[4];
        g_u[2][0] = g_uu_1d[2];
        g_u[2][1] = g_uu_1d[4];
        g_u[2][2] = g_uu_1d[5];

        // --- B. Eulerian velocity and Eulerian B from Valencia helpers ---
        Real W, v_u[3], v_d[3];
        ValenciaPrimsToEulerianVelocity(prim_Wv_u, g_dd_1d, W, v_u, v_d);

        Real B_tilde_u[3] = {
          B_vals_d(p,0),
          B_vals_d(p,1),
          B_vals_d(p,2)
        };

        Real sqrt_gamma_surf, B_u[3], B_d[3];
        ValenciaEulerianMagneticFromDensitized(B_tilde_u, g_dd_1d,
                                               sqrt_gamma_surf, B_u, B_d);

        // Comoving magnetic field
        Real Bv, b0, b_u[3], b_d[3], b2;
        ValenciaComovingB(v_u, v_d, B_u, B_d, W, Bv, b0, b_u, b_d, b2);

        // --- C. Decomposed stress-energy (fluid + EM) from one source of truth ---
        Real E_fluid, E_em;
        Real S_fluid_d[3], S_em_d[3];
        Real S_fluid_dd[3][3], S_em_dd[3][3];

        ValenciaStressEnergyDecomposed(rho, pgas, gamma_gas,
                                       v_u, v_d, W,
                                       B_u, B_d,
                                       b0, b_d, b2,
                                       g_dd_1d,
                                       E_fluid, E_em,
                                       S_fluid_d, S_em_d,
                                       S_fluid_dd, S_em_dd);

        const Real E_tot = E_fluid + E_em;

        // Raise S_j to S^i = g^{ij} S_j
        Real S_fluid_u[3] = {0.0,0.0,0.0};
        Real S_em_u[3]    = {0.0,0.0,0.0};
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            S_fluid_u[i] += g_u[i][j] * S_fluid_d[j];
            S_em_u[i]    += g_u[i][j] * S_em_d[j];
          }
        }

        // Mixed stresses S^i_j = g^{ik} S_kj
        Real S_fluid_ud[3][3], S_em_ud[3][3];
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            Real val_f = 0.0;
            Real val_em = 0.0;
            for (int k = 0; k < 3; ++k) {
              val_f  += g_u[i][k] * S_fluid_dd[k][j];
              val_em += g_u[i][k] * S_em_dd[k][j];
            }
            S_fluid_ud[i][j] = val_f;
            S_em_ud[i][j]    = val_em;
          }
        }

        // --- D. Mass flux ---
        Real u_u[3] = {
          W * (v_u[0] - beta_u[0]/alp),
          W * (v_u[1] - beta_u[1]/alp),
          W * (v_u[2] - beta_u[2]/alp)
        };
        const Real mass_flux_vec[3] = {
          alp * rho * u_u[0],
          alp * rho * u_u[1],
          alp * rho * u_u[2]
        };

        // Negative sign for inward accretion convention
        update.mdot -= (mass_flux_vec[0]*dS_d[0]
                      + mass_flux_vec[1]*dS_d[1]
                      + mass_flux_vec[2]*dS_d[2]);

        // --- E. Energy fluxes (fluid + EM) ---
        Real FE_fluid[3], FE_em[3];
        for (int i = 0; i < 3; ++i) {
          FE_fluid[i] = alp * S_fluid_u[i] - beta_u[i] * E_fluid;
          FE_em[i]    = alp * S_em_u[i]    - beta_u[i] * E_em;
        }

        update.edot_fluid -= (FE_fluid[0]*dS_d[0]
                            + FE_fluid[1]*dS_d[1]
                            + FE_fluid[2]*dS_d[2]);
        update.edot_em    -= (FE_em[0]*dS_d[0]
                            + FE_em[1]*dS_d[1]
                            + FE_em[2]*dS_d[2]);

        // --- F. Momentum fluxes ---
        Real FS_fluid[3][3], FS_em[3][3];
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            FS_fluid[i][j] = alp * S_fluid_ud[i][j] - beta_u[i] * S_fluid_d[j];
            FS_em[i][j]    = alp * S_em_ud[i][j]    - beta_u[i] * S_em_d[j];
          }
        }

        Real d_pdot_fluid[3] = {0.0,0.0,0.0};
        Real d_pdot_em[3]    = {0.0,0.0,0.0};
        for (int j = 0; j < 3; ++j) {
          for (int i = 0; i < 3; ++i) {
            d_pdot_fluid[j] += FS_fluid[i][j]*dS_d[i];
            d_pdot_em[j]    += FS_em[i][j]   *dS_d[i];
          }
        }

        // Store linear momentum fluxes
        update.pdot_x_fluid += d_pdot_fluid[0];
        update.pdot_y_fluid += d_pdot_fluid[1];
        update.pdot_z_fluid += d_pdot_fluid[2];

        update.pdot_x_em    += d_pdot_em[0];
        update.pdot_y_em    += d_pdot_em[1];
        update.pdot_z_em    += d_pdot_em[2];

        // --- G. Angular-momentum fluxes ---
        // Angular momentum density uses J_a = ε_{aℓm} x^ℓ S_m (covariant S_m).
        // Therefore flux through surface is
        //   Ldot_a = ∮ ε_{aℓm} x^ℓ (F^j{}_m dΣ_j)
        // and we already computed A_m := (F^j{}_m dΣ_j) as d_pdot_*[m] above.
        {
          const Real x = coords(p,0);
          const Real y = coords(p,1);
          const Real z = coords(p,2);

          // A_m = F^j{}_m dΣ_j  (covariant in m)
          const Real A_fluid_d[3] = { d_pdot_fluid[0], d_pdot_fluid[1], d_pdot_fluid[2] };
          const Real A_em_d[3]    = { d_pdot_em[0],    d_pdot_em[1],    d_pdot_em[2]    };

          // Ldot = r × A
          update.ldot_x_fluid += (y * A_fluid_d[2] - z * A_fluid_d[1]);
          update.ldot_y_fluid += (z * A_fluid_d[0] - x * A_fluid_d[2]);
          update.ldot_z_fluid += (x * A_fluid_d[1] - y * A_fluid_d[0]);

          update.ldot_x_em    += (y * A_em_d[2] - z * A_em_d[1]);
          update.ldot_y_em    += (z * A_em_d[0] - x * A_em_d[2]);
          update.ldot_z_em    += (x * A_em_d[1] - y * A_em_d[0]);
        }

        // Every current below is UNDENSITIZED; dS_d already includes sqrt(gamma).
        Real wn=0.0;
        for (int i=0;i<3;++i) wn+=motion(p,i)*dS_d[i];
        Real dmass=rho*W*wn;
        for (int i=0;i<3;++i) dmass-=mass_flux_vec[i]*dS_d[i];
        update.moving[0]+=dmass;
        update.moving[1]+=fmax(dmass,0.0);
        update.moving[2]+=fmax(-dmass,0.0);
        for (int part=0;part<2;++part) {
          Real E=part==0 ? E_fluid : E_em;
          Real S[3], P[3]={}, FE[3], FS[3][3];
          for (int j=0;j<3;++j) {
            S[j]=part==0 ? S_fluid_d[j] : S_em_d[j];
            FE[j]=part==0 ? FE_fluid[j] : FE_em[j];
            for (int i=0;i<3;++i) FS[i][j]=part==0 ? FS_fluid[i][j] : FS_em[i][j];
          }
          Real ein=E*wn, et=alp*E, etin;
          for (int j=0;j<3;++j) {
            ein-=FE[j]*dS_d[j]; et-=beta_u[j]*S[j];
            P[j]=S[j]*wn;
            for (int i=0;i<3;++i) P[j]-=FS[i][j]*dS_d[i];
            update.moving[5+3*part+j]+=P[j];
          }
          update.moving[3+part]+=ein;
          Real x=coords(p,0),y=coords(p,1),z=coords(p,2);
          update.moving[11+3*part]+=y*P[2]-z*P[1];
          update.moving[12+3*part]+=z*P[0]-x*P[2];
          update.moving[13+3*part]+=x*P[1]-y*P[0];
          etin=et*wn;
          for (int i=0;i<3;++i) {
            Real ft=alp*FE[i];
            for (int j=0;j<3;++j) ft-=beta_u[j]*FS[i][j];
            etin-=ft*dS_d[i];
          }
          update.moving[17+part]+=etin;
        }

        // --- H. Phi_B (magnetic flux) ---
        {
          // The EHT definition is Flux = 0.5 * Integral |B . dA|
          // B_u is the Eulerian magnetic field (already available).
          // dS_d is the covariant surface area element (normal vector * area).

          const Real flux_dot_dS = 
            B_u[0]*dS_d[0] + B_u[1]*dS_d[1] + B_u[2]*dS_d[2];

          update.phiB += 0.5 * Kokkos::fabs(flux_dot_dS);
        }

        // --- I. Surface area ---
        Real dS_u[3] = {0.0,0.0,0.0};
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            dS_u[i] += g_u[i][j]*dS_d[j];
          }
        }
        const Real dS_sq = dS_d[0]*dS_u[0] + dS_d[1]*dS_u[1] + dS_d[2]*dS_u[2];
        update.area += (dS_sq > 0.0 ? sqrt(dS_sq) : 0.0);
        update.covered_area += (dS_sq > 0.0 ? sqrt(dS_sq) : 0.0);
      },
      Kokkos::Sum<FluxAccumulator>(fluxes));

    Real legacy[17]={fluxes.mdot,fluxes.edot_fluid,fluxes.edot_em,
      fluxes.pdot_x_fluid,fluxes.pdot_y_fluid,fluxes.pdot_z_fluid,
      fluxes.pdot_x_em,fluxes.pdot_y_em,fluxes.pdot_z_em,
      fluxes.ldot_x_fluid,fluxes.ldot_y_fluid,fluxes.ldot_z_fluid,
      fluxes.ldot_x_em,fluxes.ldot_y_em,fluxes.ldot_z_em,fluxes.phiB,fluxes.area};
    for (int c=0;c<17;++c) result.values[c]=legacy[c];
    for (int c=0;c<19;++c) result.values[17+c]=fluxes.moving[c];
    Real stats[9]={fluxes.owned,fluxes.admissible,fluxes.covered_area};
    for (int c=0;c<6;++c) stats[3+c]=fluxes.bad[c];
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE,result.values.data(),36,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,stats,9,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
#endif
    Real dx=0.0;
    for (int p=0;p<np;++p) {
      if (current_surf->Owners().h_view(p)!=1) result.reason|=1;
      dx=std::max(dx,current_surf->LocalSpacing().h_view(p));
    }
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE,&dx,1,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
#endif
    for (int c=0;c<6;++c) if (stats[3+c]>0.0) result.reason|=1<<c;
    for (int c=0;c<36;++c) if (!std::isfinite(result.values[c])) result.reason|=4;
    result.valid=result.reason==0 && stats[1]==np;
    result.values[36]=result.valid ? 1.0 : 0.0;
    result.values[37]=result.reason;
    result.values[38]=stats[0]/np;
    result.values[39]=stats[1]/np;
    result.values[40]=stats[2];
    result.values[41]=dx;
    result.values[42]=current_surf->min_cells_per_radius;
    result.values[43]=result.values[17];
    if (!result.valid) for (int c=0;c<36;++c)
      result.values[c]=std::numeric_limits<Real>::quiet_NaN();
    return result;
}

void TorusFluxes_General(HistoryData *pdata, MeshBlockPack *pack,
                        const std::vector<SphericalSurfaceGrid*>& surfaces) {
  const int count=static_cast<int>(surfaces.size());
  if (count*kSurfaceChannels>NHISTORY_VARIABLES)
    Kokkos::abort("Requested surface channels exceed history capacity");
  // Preserve the complete legacy ordering, then append explicitly corrected channels.
  static const char* names[kSurfaceChannels]={
    "mdot_","edot_f_","edot_em_","pxdot_f_","pydot_f_","pzdot_f_",
    "pxdot_em_","pydot_em_","pzdot_em_","lxdot_f_","lydot_f_","lzdot_f_",
    "lxdot_em_","lydot_em_","lzdot_em_","phiB_","area_",
    "M0_net_in_moving_","M0_in_moving_","M0_out_moving_",
    "Eeu_f_in_moving_","Eeu_em_in_moving_",
    "Px_f_in_moving_","Py_f_in_moving_","Pz_f_in_moving_",
    "Px_em_in_moving_","Py_em_in_moving_","Pz_em_in_moving_",
    "Jx_f_in_moving_","Jy_f_in_moving_","Jz_f_in_moving_",
    "Jx_em_in_moving_","Jy_em_in_moving_","Jz_em_in_moving_",
    "Et_f_in_moving_","Et_em_in_moving_","valid_","reason_",
    "ownership_fraction_","valid_sample_fraction_","partial_covered_area_",
    "stencil_dx_max_","required_cells_per_radius_","partial_M0_net_in_moving_"};
  pdata->nhist=count*kSurfaceChannels;
  for (int g=0;g<count;++g) {
    auto result=IntegrateSurface(pack,surfaces[g]);
    for (int c=0;c<kSurfaceChannels;++c) {
      int index=c<17 ? 17*g+c : 17*count+27*g+c-17;
      pdata->label[index]=std::string(names[c])+surfaces[g]->Label();
      pdata->hdata[index]=global_variable::my_rank==0 ? result.values[c] : 0.0;
    }
  }
}
