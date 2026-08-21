//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file curvature_diagnostics.cpp
//! \brief MPI-global curvature extrema used by collapse stopping conditions.

#include <limits>
#include <type_traits>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "z4c/curvature_diagnostics.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_symmetry.hpp"

template <typename Centering, typename Symmetry, int NGHOST>
Z4cGlobalCurvatureMaxima ComputeZ4cGlobalCurvatureMaximaImpl(Mesh *pm) {
  auto *pmbp = pm->pmb_pack;
  const auto &layout = pmbp->pz4c->layout;
  const int active_nx1 = layout.ie - layout.is + 1;
  const int active_nx2 = layout.je - layout.js + 1;
  const int active_nx3 = layout.ke - layout.ks + 1;
  const int nmkji = pmbp->nmb_thispack * active_nx3 * active_nx2 * active_nx1;
  const int nkji = active_nx3 * active_nx2 * active_nx1;
  const int nji = active_nx2 * active_nx1;
  auto &u0 = pmbp->pz4c->u0;
  const auto g_dd = std::is_same_v<Centering, z4c::VertexCenteredZ4c>
                        ? pmbp->pz4c->adm.g_dd
                        : pmbp->padm->adm.g_dd;
  const auto vK_dd = std::is_same_v<Centering, z4c::VertexCenteredZ4c>
                         ? pmbp->pz4c->adm.vK_dd
                         : pmbp->padm->adm.vK_dd;
  auto &size = pmbp->pmb->mb_size;

  Real max_abs_k = 0.0;
  Kokkos::parallel_reduce(
      "GlobalCurvatureMaxAbsK",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int idx, Real &result) {
        const int m = idx / nkji;
        int k = (idx - m * nkji) / nji;
        int j = (idx - m * nkji - k * nji) / active_nx1;
        const int i = (idx - m * nkji - k * nji - j * active_nx1) + layout.is;
        k += layout.ks;
        j += layout.js;
        const Real trace_k =
            u0(m, z4c::Z4c::I_Z4C_KHAT, k, j, i) +
            2.0 * u0(m, z4c::Z4c::I_Z4C_THETA, k, j, i);
        result = fmax(result, fabs(trace_k));
      },
      Kokkos::Max<Real>(max_abs_k));

  Real max_abs_kretschmann = 0.0;
  Kokkos::parallel_reduce(
      "GlobalCurvatureMaxKretschmann",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int idx, Real &result) {
        const int m = idx / nkji;
        int k = (idx - m * nkji) / nji;
        int j = (idx - m * nkji - k * nji) / active_nx1;
        const int i = (idx - m * nkji - k * nji - j * active_nx1) + layout.is;
        k += layout.ks;
        j += layout.js;
        const Real inverse_spacing[3] = {
            1.0 / size.d_view(m).dx1,
            1.0 / size.d_view(m).dx2,
            1.0 / size.d_view(m).dx3};
        auto derivatives = z4c::MakeZ4cDerivativeProvider<Centering, Symmetry, NGHOST>(
            inverse_spacing, size.d_view, layout.nx1, layout.is, m, k, j, i);
        const auto diagnostic = ComputeZ4cCurvatureDiagnostics<NGHOST, false>(
            derivatives, g_dd, vK_dd, m, k, j, i);
        if (!diagnostic.valid) {
          result = std::numeric_limits<Real>::infinity();
        } else {
          result = fmax(result, fabs(diagnostic.kretschmann));
        }
      },
      Kokkos::Max<Real>(max_abs_kretschmann));

  Real values[2] = {max_abs_k, max_abs_kretschmann};
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, values, 2, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
#endif
  Z4cGlobalCurvatureMaxima maxima;
  maxima.max_abs_k = values[0];
  maxima.max_abs_kretschmann = values[1];
  maxima.finite =
      Kokkos::isfinite(values[0]) && Kokkos::isfinite(values[1]);
  return maxima;
}

Z4cGlobalCurvatureMaxima ComputeZ4cGlobalCurvatureMaxima(Mesh *pm) {
  const auto &config = pm->pmb_pack->z4c_symmetry;
  const bool cartoon = config.mode == z4c::Z4cSymmetryMode::cartoon_so2;
  const bool vertex = config.grid_centering == z4c::Z4cGridCentering::vertex;
  switch (config.stencil_width) {
    case 2:
      if (cartoon) {
        return vertex
                   ? ComputeZ4cGlobalCurvatureMaximaImpl<z4c::VertexCenteredZ4c,
                                                        z4c::CartoonSO2, 2>(pm)
                   : ComputeZ4cGlobalCurvatureMaximaImpl<z4c::CellCenteredZ4c,
                                                        z4c::CartoonSO2, 2>(pm);
      }
      return vertex
                 ? ComputeZ4cGlobalCurvatureMaximaImpl<z4c::VertexCenteredZ4c,
                                                      z4c::Cartesian3D, 2>(pm)
                 : ComputeZ4cGlobalCurvatureMaximaImpl<z4c::CellCenteredZ4c,
                                                      z4c::Cartesian3D, 2>(pm);
    case 3:
      if (cartoon) {
        return vertex
                   ? ComputeZ4cGlobalCurvatureMaximaImpl<z4c::VertexCenteredZ4c,
                                                        z4c::CartoonSO2, 3>(pm)
                   : ComputeZ4cGlobalCurvatureMaximaImpl<z4c::CellCenteredZ4c,
                                                        z4c::CartoonSO2, 3>(pm);
      }
      return vertex
                 ? ComputeZ4cGlobalCurvatureMaximaImpl<z4c::VertexCenteredZ4c,
                                                      z4c::Cartesian3D, 3>(pm)
                 : ComputeZ4cGlobalCurvatureMaximaImpl<z4c::CellCenteredZ4c,
                                                      z4c::Cartesian3D, 3>(pm);
    case 4:
      if (cartoon) {
        return vertex
                   ? ComputeZ4cGlobalCurvatureMaximaImpl<z4c::VertexCenteredZ4c,
                                                        z4c::CartoonSO2, 4>(pm)
                   : ComputeZ4cGlobalCurvatureMaximaImpl<z4c::CellCenteredZ4c,
                                                        z4c::CartoonSO2, 4>(pm);
      }
      return vertex
                 ? ComputeZ4cGlobalCurvatureMaximaImpl<z4c::VertexCenteredZ4c,
                                                      z4c::Cartesian3D, 4>(pm)
                 : ComputeZ4cGlobalCurvatureMaximaImpl<z4c::CellCenteredZ4c,
                                                      z4c::Cartesian3D, 4>(pm);
    default:
      return {0.0, 0.0, false};
  }
}
