//========================================================================================
// surface_grid.cpp — Implementation for the SphericalSurfaceGrid class
//----------------------------------------------------------------------------------------

#include "surface_grid.hpp"

#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coordinates.hpp"
#include "cyclic_zoom/cyclic_zoom.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif
#include "coordinates/adm.hpp" // For adm::SpatialDet and metric indices

#include <cmath>
#include <iostream>
#include <cstdlib>
#include <algorithm> // For std::min

//----------------------------------------------------------------------------------------
// Constructor
SphericalSurfaceGrid::SphericalSurfaceGrid(MeshBlockPack* pack,
                                           int ntheta, int nphi,
                                           RFunc r_of_thph,
                                           const std::string& name,
                                           const Real* center,
                                           int interp_order)
    : pmy_pack(pack), tag(name), metric_is_flat_(true), radius_function_(r_of_thph), interp_order_(interp_order) {
  // --- basic validation ---
  if (pmy_pack == nullptr || pmy_pack->pmesh == nullptr || pmy_pack->pmb == nullptr) {
    std::cerr << "### FATAL: SphericalSurfaceGrid requires a valid MeshBlockPack/mesh"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (ntheta < 3 || nphi < 3) {
    std::cerr << "### FATAL: SphericalSurfaceGrid needs ntheta>=3 and nphi>=3 (got "
              << ntheta << "," << nphi << ")" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // --- Safety cap on interpolation order based on ghost cells ---
  const int ng = pmy_pack->pmesh->mb_indcs.ng;
  const int max_stencil_pts = 2 * ng;
  const int max_order = max_stencil_pts - 1;

  if (interp_order_ > max_order) {
    std::cout << "### WARNING in SphericalSurfaceGrid [" << tag << "]:\n"
              << "    Requested interpolation order=" << interp_order_
              << " requires a stencil of " << interp_order_ + 1 << " points.\n"
              << "    With nghost=" << ng << ", the maximum available stencil is "
              << max_stencil_pts << " points (order " << max_order << ").\n"
              << "    Capping interpolation order to " << max_order << "." << std::endl;
    interp_order_ = max_order;
  }
  if (interp_order_ < 1) {
      std::cerr << "### FATAL: interpolation order must be 1 or greater." << std::endl;
      std::exit(EXIT_FAILURE);
  }


  // Store center
  center_[0] = center ? center[0] : 0.0;
  center_[1] = center ? center[1] : 0.0;
  center_[2] = center ? center[2] : 0.0;

  // Grid in parameter space
  n_th = ntheta; n_ph = nphi; npts = n_th * n_ph;
  dth = M_PI / n_th;  // θ ∈ (0,π)
  dph = (n_ph > 1) ? (2.0 * M_PI / n_ph) : 0.0;     // φ ∈ [0,2π)

  // Allocate geometry arrays
  Kokkos::realloc(theta,  npts);
  Kokkos::realloc(phi,    npts);
  Kokkos::realloc(radius, npts);
  Kokkos::realloc(coords, npts, 3);
  Kokkos::realloc(tan_th, npts, 3);
  Kokkos::realloc(tan_ph, npts, 3);
  Kokkos::realloc(weights, npts);
  Kokkos::realloc(geometric_weights_, npts);
  Kokkos::realloc(owners_, npts);
  Kokkos::realloc(stencil_reasons_, npts);
  Kokkos::realloc(local_spacing_, npts);
  Kokkos::realloc(motion_, npts, 3);
  Kokkos::deep_copy(motion_.d_view, 0.0);
  Kokkos::realloc(g_dd_surf_, npts, 6);
  Kokkos::realloc(gamma_dd_surf_, npts, 3); // gamma_th_th, gamma_th_ph, gamma_ph_ph
  Kokkos::realloc(proper_dA_, npts);       // scalar area element

  // Interpolation maps: allocate BEFORE building them
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  // Allocation size remains tied to ng, as it's the max possible stencil
  Kokkos::realloc(interp_indcs, npts, 4);
  Kokkos::realloc(interp_wghts, npts, 2 * indcs.ng, 3);

  // Initialize parameter grid + r(θ,φ) on host
  int p = 0;
  for (int it = 0; it < n_th; ++it) {
    const Real th = (it + 0.5) * dth;
    for (int ip = 0; ip < n_ph; ++ip, ++p) {
      const Real ph = ip * dph;
      theta.h_view(p) = th;
      phi.h_view(p)   = ph;
      radius.h_view(p) = r_of_thph(th, ph);
    }
  }

  // Build everything
  RebuildAll();
}

//----------------------------------------------------------------------------------------
// Public Methods

void SphericalSurfaceGrid::SetCenter(const Real new_center[3]) {
  center_[0] = new_center[0];
  center_[1] = new_center[1];
  center_[2] = new_center[2];
  RebuildAll();
}

void SphericalSurfaceGrid::SetRadiusFunction(RFunc f) {
  radius_function_ = f;
  RebuildFromRadius();
}

void SphericalSurfaceGrid::RebuildFromRadius() {
  for (int p=0; p<npts; ++p) {
    Real r = radius_function_(theta.h_view(p), phi.h_view(p));
    if (!(std::isfinite(r) && r > 0.0)) Kokkos::abort("Invalid surface radius");
    radius.h_view(p) = r;
  }
  radius.modify<HostMemSpace>();
  radius.sync<DevExeSpace>();
  RebuildAll();
}

void SphericalSurfaceGrid::SetMotion(MotionFunc f) {
  for (int p=0; p<npts; ++p) {
    Real v[3];
    f(theta.h_view(p), phi.h_view(p), v);
    for (int d=0; d<3; ++d) motion_.h_view(p,d)=v[d];
  }
  motion_.modify<HostMemSpace>();
  motion_.sync<DevExeSpace>();
}

DualArray2D<Real> SphericalSurfaceGrid::InterpolateToSurface(
    const DvceArray5D<Real> &source_array, int start_index, int end_index) {

  const int nvars = end_index - start_index;
  if (nvars <= 0) {
      std::cerr << "### FATAL: InterpolateToSurface called with invalid range: "
                << start_index << " to " << end_index << std::endl;
      std::exit(EXIT_FAILURE);
  }

  DualArray2D<Real> result("interp_vals_temp", npts, nvars);

  auto &iindcs = interp_indcs;
  auto &iwghts = interp_wghts;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;

  const int nsten = interp_order_ + 1;
  const int nleft = nsten / 2;

  par_for("int2surf_block", DevExeSpace(), 0, npts-1, 0, nvars-1,
    KOKKOS_LAMBDA(const int p, const int v) {
      const int mb_id = iindcs.d_view(p,0);
      if (mb_id == -1) {
        result.d_view(p,v) = 0.0;
        return;
      }
      const int i0 = iindcs.d_view(p,1), j0 = iindcs.d_view(p,2), k0 = iindcs.d_view(p,3);
      Real accum = 0.0;

      for (int k_sten = 0; k_sten < nsten; ++k_sten) {
        for (int j_sten = 0; j_sten < nsten; ++j_sten) {
          for (int i_sten = 0; i_sten < nsten; ++i_sten) {
            const Real w = iwghts.d_view(p, i_sten, 0) * iwghts.d_view(p, j_sten, 1) * iwghts.d_view(p, k_sten, 2);

            // --- FIXED: This index logic is now consistent with SetInterpolationWeights ---
            const int I = is + i0 - nleft + 1 + i_sten;
            const int J = js + j0 - nleft + 1 + j_sten;
            const int K = ks + k0 - nleft + 1 + k_sten;
            accum += w * source_array(mb_id, start_index + v, K, J, I);
          }
        }
      }
      result.d_view(p,v) = accum;
    });

  Kokkos::fence();

  result.template modify<DevExeSpace>();
  result.template sync<HostMemSpace>();
  return result;
}

void SphericalSurfaceGrid::InterpolateMetric() {
  const int start = adm::ADM::I_ADM_GXX;
  const int end = start + 6;
  DualArray2D<Real> interpolated_metric = InterpolateToSurface(pmy_pack->padm->u_adm, start, end);

  auto dst_view = g_dd_surf_.d_view;
  auto src_view = interpolated_metric.d_view;

  Kokkos::parallel_for("ManualMetricCopy",
                       Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {npts, 6}),
    KOKKOS_LAMBDA(const int p, const int v) {
      dst_view(p, v) = src_view(p, v);
    });
  g_dd_surf_.template modify<DevExeSpace>();
  metric_is_flat_ = false;

  CalculateDerivedGeometry();
}

void SphericalSurfaceGrid::BuildSurfaceCovectors(DualArray2D<Real>& dSigma) const {
  const int np = npts;
  Kokkos::realloc(dSigma, np, 3);
  auto &eTh = tan_th; auto &ePh = tan_ph; auto &wq = geometric_weights_;
  auto &g_surf = g_dd_surf_;

  Kokkos::parallel_for("surf_cov_curved", Kokkos::RangePolicy<DevExeSpace>(0, np),
    KOKKOS_LAMBDA(const int p) {
      const Real gxx = g_surf.d_view(p, 0), gxy = g_surf.d_view(p, 1), gxz = g_surf.d_view(p, 2);
      const Real gyy = g_surf.d_view(p, 3), gyz = g_surf.d_view(p, 4), gzz = g_surf.d_view(p, 5);
      const Real gamma = adm::SpatialDet(gxx, gxy, gxz, gyy, gyz, gzz);
      const Real sqrt_gamma = (gamma > 0.0) ? sqrt(gamma) : 0.0;
      const Real s = sqrt_gamma * wq.d_view(p);

      const Real e1x = eTh.d_view(p,0), e1y = eTh.d_view(p,1), e1z = eTh.d_view(p,2);
      const Real e2x = ePh.d_view(p,0), e2y = ePh.d_view(p,1), e2z = ePh.d_view(p,2);
      dSigma.d_view(p,0) = s * (e1y*e2z - e1z*e2y);
      dSigma.d_view(p,1) = s * (e1z*e2x - e1x*e2z);
      dSigma.d_view(p,2) = s * (e1x*e2y - e1y*e2x);
    }
  );

  Kokkos::fence();

  dSigma.template modify<DevExeSpace>();
}

//----------------------------------------------------------------------------------------
// Private Helper Methods

void SphericalSurfaceGrid::CalculateDerivedGeometry() {
    auto g_3D = g_dd_surf_.d_view;
    auto e_th = tan_th.d_view;
    auto e_ph = tan_ph.d_view;
    auto quad_weights = geometric_weights_.d_view;
    auto gamma_2D = gamma_dd_surf_.d_view;
    auto dA = proper_dA_.d_view;

    Kokkos::parallel_for("CalculateDerivedGeom", Kokkos::RangePolicy<DevExeSpace>(0, npts),
      KOKKOS_LAMBDA(const int p) {
        const Real gxx = g_3D(p, 0);
        const Real gxy = g_3D(p, 1);
        const Real gxz = g_3D(p, 2);
        const Real gyy = g_3D(p, 3);
        const Real gyz = g_3D(p, 4);
        const Real gzz = g_3D(p, 5);

        const Real e_th_x = e_th(p, 0), e_th_y = e_th(p, 1), e_th_z = e_th(p, 2);
        const Real e_ph_x = e_ph(p, 0), e_ph_y = e_ph(p, 1), e_ph_z = e_ph(p, 2);

        const Real gamma_th_th = gxx*e_th_x*e_th_x + gyy*e_th_y*e_th_y + gzz*e_th_z*e_th_z
                               + 2.0*(gxy*e_th_x*e_th_y + gxz*e_th_x*e_th_z + gyz*e_th_y*e_th_z);
        const Real gamma_ph_ph = gxx*e_ph_x*e_ph_x + gyy*e_ph_y*e_ph_y + gzz*e_ph_z*e_ph_z
                               + 2.0*(gxy*e_ph_x*e_ph_y + gxz*e_ph_x*e_ph_z + gyz*e_ph_y*e_ph_z);
        const Real gamma_th_ph = gxx*e_th_x*e_ph_x + gyy*e_th_y*e_ph_y + gzz*e_th_z*e_ph_z
                               + gxy*(e_th_x*e_ph_y + e_th_y*e_ph_x)
                               + gxz*(e_th_x*e_ph_z + e_th_z*e_ph_x)
                               + gyz*(e_th_y*e_ph_z + e_th_z*e_ph_y);

        gamma_2D(p, 0) = gamma_th_th;
        gamma_2D(p, 1) = gamma_th_ph;
        gamma_2D(p, 2) = gamma_ph_ph;

        const Real det_gamma_2D = gamma_th_th * gamma_ph_ph - SQR(gamma_th_ph);

        dA(p) = (det_gamma_2D > 0.0)
                ? sqrt(det_gamma_2D) * quad_weights(p)
                : 0.0;
    });

    gamma_dd_surf_.template modify<DevExeSpace>();
    proper_dA_.template modify<DevExeSpace>();
}

void SphericalSurfaceGrid::RebuildAll() {
  BuildCoordinates();
  SetInterpolationIndices();
  BuildTangentsFD();
  BuildQuadWeights();
  SetInterpolationWeights();
  InitializeFlatMetric();
}

void SphericalSurfaceGrid::InitializeFlatMetric() {
  auto h_g = g_dd_surf_.h_view;
  for (int p=0; p<npts; ++p) {
    h_g(p, 0) = 1.0; h_g(p, 1) = 0.0; h_g(p, 2) = 0.0;
    h_g(p, 3) = 1.0; h_g(p, 4) = 0.0;
    h_g(p, 5) = 1.0;
  }
  g_dd_surf_.template modify<HostMemSpace>();
  g_dd_surf_.template sync<DevExeSpace>();
  metric_is_flat_ = true;

  CalculateDerivedGeometry();
}

void SphericalSurfaceGrid::BuildCoordinates() {
  auto h_th = theta.h_view;
  auto h_ph = phi.h_view;
  auto h_r = radius.h_view;
  auto h_x = coords.h_view;
  for (int p = 0; p < npts; ++p) {
    Real Xl, Yl, Zl;
    const Real st = sin(h_th(p)), ct = cos(h_th(p));
    const Real sp = sin(h_ph(p)), cp = cos(h_ph(p));
    Xl = h_r(p) * st * cp; Yl = h_r(p) * st * sp; Zl = h_r(p) * ct;
    h_x(p,0) = Xl + center_[0];
    h_x(p,1) = Yl + center_[1];
    h_x(p,2) = Zl + center_[2];
  }
  coords.template modify<HostMemSpace>();
  coords.template sync<DevExeSpace>();
}

void SphericalSurfaceGrid::BuildQuadWeights() {
  auto h_w = weights.h_view;
  auto h_iind = interp_indcs.h_view;
  for (int p = 0; p < npts; ++p) {
    geometric_weights_.h_view(p) = dth*dph;
    if (h_iind(p, 0) != -1) {
      h_w(p) = dth * dph;
    } else {
      h_w(p) = 0.0;
    }
  }
  geometric_weights_.modify<HostMemSpace>();
  geometric_weights_.sync<DevExeSpace>();
  weights.template modify<HostMemSpace>();
  weights.template sync<DevExeSpace>();
}

void SphericalSurfaceGrid::SetInterpolationIndices() {
  auto &size = pmy_pack->pmb->mb_size;
  const int nmb = pmy_pack->nmb_thispack;
  auto h_coords = coords.h_view;
  auto h_iind = interp_indcs.h_view;
  for (int p = 0; p < npts; ++p) {
    owners_.h_view(p) = 0;
    h_iind(p,0) = -1; h_iind(p,1) = -1; h_iind(p,2) = -1; h_iind(p,3) = -1;
    for (int m = 0; m < nmb; ++m) {
      const auto &mb = size.h_view(m);
      const Real x1p = h_coords(p,0), x2p = h_coords(p,1), x3p = h_coords(p,2);
      if ((x1p >= mb.x1min && x1p < mb.x1max) && (x2p >= mb.x2min && x2p < mb.x2max) && (x3p >= mb.x3min && x3p < mb.x3max)) {
        h_iind(p,0) = m;
        h_iind(p,1) = static_cast<int>(std::floor((x1p - (mb.x1min + mb.dx1/2.0)) / mb.dx1));
        h_iind(p,2) = static_cast<int>(std::floor((x2p - (mb.x2min + mb.dx2/2.0)) / mb.dx2));
        // --- FIXED: Corrected typo "static_ofc" to "static_cast" ---
        h_iind(p,3) = static_cast<int>(std::floor((x3p - (mb.x3min + mb.dx3/2.0)) / mb.dx3));
        owners_.h_view(p) += 1;
      }
    }
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, owners_.h_view.data(), npts, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
#endif
  owners_.modify<HostMemSpace>(); owners_.sync<DevExeSpace>();
  interp_indcs.template modify<HostMemSpace>();
  interp_indcs.template sync<DevExeSpace>();
}

void SphericalSurfaceGrid::QualifyStencils() {
  // Check the actual interpolation support, including reset masks and frozen zoom data.
  auto &size = pmy_pack->pmb->mb_size;
  auto h_iind = interp_indcs.h_view;
  auto coord = pmy_pack->pcoord;
  auto mask = Kokkos::create_mirror_view_and_copy(HostMemSpace(), coord->excision_floor);
  auto flux_mask = Kokkos::create_mirror_view_and_copy(HostMemSpace(), coord->excision_flux);
  auto weight = Kokkos::create_mirror_view_and_copy(HostMemSpace(), coord->excision_weight);
  auto ind = pmy_pack->pmesh->mb_indcs;
  auto zoom = pmy_pack->pmesh->pzoom;
  for (int p=0; p<npts; ++p) {
    int reason = owners_.h_view(p) == 1 ? 0 : 1;
    int m = h_iind(p,0);
    Real dx = 0.0;
    if (m >= 0) {
      auto mb = size.h_view(m);
      dx = std::max({mb.dx1,mb.dx2,mb.dx3});
      int left = (interp_order_+1)/2;
      for (int c=0; c<interp_order_+1; ++c)
        for (int b=0; b<interp_order_+1; ++b)
          for (int a=0; a<interp_order_+1; ++a) {
            int I=ind.is+h_iind(p,1)-left+1+a;
            int J=ind.js+h_iind(p,2)-left+1+b;
            int K=ind.ks+h_iind(p,3)-left+1+c;
            if (I<0 || J<0 || K<0 || I>=ind.nx1+2*ind.ng ||
                J>=ind.nx2+2*ind.ng || K>=ind.nx3+2*ind.ng) { reason |= 1; continue; }
            if (coord->coord_data.bh_excise &&
                (mask(m,K,J,I) || flux_mask(m,K,J,I) ||
                 (coord->coord_data.smooth_excise && weight(m,K,J,I)>0.0))) reason |= 16;
            Real x=CellCenterX(I-ind.is,ind.nx1,mb.x1min,mb.x1max);
            Real y=CellCenterX(J-ind.js,ind.nx2,mb.x2min,mb.x2max);
            Real z=CellCenterX(K-ind.ks,ind.nx3,mb.x3min,mb.x3max);
            if (zoom && zoom->zstate.zone>0 && zoom->zregion.IsInRegion(x,y,z)) reason |= 32;
          }
      // A boundary stencil may use coarser AMR data: use a conservative spacing bound.
      if (pmy_pack->pmesh->multilevel &&
          (h_iind(p,1)<left || h_iind(p,1)+left>=ind.nx1 ||
           h_iind(p,2)<left || h_iind(p,2)+left>=ind.nx2 ||
           h_iind(p,3)<left || h_iind(p,3)+left>=ind.nx3)) dx*=2.0;
      if (min_cells_per_radius>0.0 && radius.h_view(p)/dx<min_cells_per_radius) reason |= 8;
    }
    local_spacing_.h_view(p)=dx;
    stencil_reasons_.h_view(p)=reason;
  }
  owners_.modify<HostMemSpace>(); owners_.sync<DevExeSpace>();
  stencil_reasons_.modify<HostMemSpace>(); stencil_reasons_.sync<DevExeSpace>();
  local_spacing_.modify<HostMemSpace>(); local_spacing_.sync<DevExeSpace>();
}

void SphericalSurfaceGrid::SetInterpolationWeights() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  const int ng = indcs.ng;
  auto h_iind = interp_indcs.h_view;
  auto h_iw = interp_wghts.h_view;
  auto h_coords = coords.h_view;

  const int nsten = interp_order_ + 1;
  const int nleft = nsten / 2;

  for (int p = 0; p < npts; ++p) {
    const int mb_id = h_iind(p,0);

    for (int i=0; i<2*ng; ++i) {
      h_iw(p,i,0) = 0.0; h_iw(p,i,1) = 0.0; h_iw(p,i,2) = 0.0;
    }

    if (mb_id != -1) {
      const int i0 = h_iind(p,1), j0 = h_iind(p,2), k0 = h_iind(p,3);
      const Real x0 = h_coords(p,0), y0 = h_coords(p,1), z0 = h_coords(p,2);
      const auto &mb = size.h_view(mb_id);

      // --- FIXED: Refactored into three separate loops for clarity and correctness ---

      // Calculate X-direction weights
      for (int i=0; i<nsten; ++i) {
        h_iw(p,i,0) = 1.0;
        for (int j=0; j<nsten; ++j) {
          if (j != i) {
            Real x1i = CellCenterX(i0 - nleft + 1 + i, indcs.nx1, mb.x1min, mb.x1max);
            Real x1j = CellCenterX(i0 - nleft + 1 + j, indcs.nx1, mb.x1min, mb.x1max);
            h_iw(p,i,0) *= (x0 - x1j) / (x1i - x1j);
          }
        }
      }

      // Calculate Y-direction weights
      for (int i=0; i<nsten; ++i) {
        h_iw(p,i,1) = 1.0;
        for (int j=0; j<nsten; ++j) {
          if (j != i) {
            Real x2i = CellCenterX(j0 - nleft + 1 + i, indcs.nx2, mb.x2min, mb.x2max);
            Real x2j = CellCenterX(j0 - nleft + 1 + j, indcs.nx2, mb.x2min, mb.x2max);
            h_iw(p,i,1) *= (y0 - x2j) / (x2i - x2j);
          }
        }
      }

      // Calculate Z-direction weights
      for (int i=0; i<nsten; ++i) {
        h_iw(p,i,2) = 1.0;
        for (int j=0; j<nsten; ++j) {
          if (j != i) {
            Real x3i = CellCenterX(k0 - nleft + 1 + i, indcs.nx3, mb.x3min, mb.x3max);
            Real x3j = CellCenterX(k0 - nleft + 1 + j, indcs.nx3, mb.x3min, mb.x3max);
            h_iw(p,i,2) *= (z0 - x3j) / (x3i - x3j);
          }
        }
      }
    }
  }
  interp_wghts.template modify<HostMemSpace>();
  interp_wghts.template sync<DevExeSpace>();
}

void SphericalSurfaceGrid::BuildTangentsFD() {
  const Real eps=1.0e-5;
  for (int p=0; p<npts; ++p) {
    Real th=theta.h_view(p), ph=phi.h_view(p), r=radius.h_view(p);
    Real rt=(radius_function_(th+eps,ph)-radius_function_(th-eps,ph))/(2*eps);
    Real rp=(radius_function_(th,ph+eps)-radius_function_(th,ph-eps))/(2*eps);
    Real n[3]={sin(th)*cos(ph),sin(th)*sin(ph),cos(th)};
    Real nt[3]={cos(th)*cos(ph),cos(th)*sin(ph),-sin(th)};
    Real np[3]={-sin(th)*sin(ph),sin(th)*cos(ph),0.0};
    for (int d=0; d<3; ++d) {
      tan_th.h_view(p,d)=rt*n[d]+r*nt[d];
      tan_ph.h_view(p,d)=rp*n[d]+r*np[d];
    }
  }
  tan_th.modify<HostMemSpace>(); tan_th.sync<DevExeSpace>();
  tan_ph.modify<HostMemSpace>(); tan_ph.sync<DevExeSpace>();
}
