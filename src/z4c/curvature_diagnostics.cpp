//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file curvature_diagnostics.cpp
//! \brief MPI-global curvature extrema used by collapse stopping conditions.

#include <limits>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "z4c/curvature_diagnostics.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_symmetry.hpp"

template <typename Symmetry, int NGHOST>
Z4cGlobalCurvatureMaxima ComputeZ4cGlobalCurvatureMaximaImpl(Mesh *pm) {
  auto *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int nx1 = indcs.nx1;
  const int nx2 = indcs.nx2;
  const int nx3 = indcs.nx3;
  const int is = indcs.is;
  const int js = indcs.js;
  const int ks = indcs.ks;
  const int nmkji = pmbp->nmb_thispack * nx3 * nx2 * nx1;
  const int nkji = nx3 * nx2 * nx1;
  const int nji = nx2 * nx1;
  auto &u0 = pmbp->pz4c->u0;
  auto &adm = pmbp->padm->adm;
  auto &size = pmbp->pmb->mb_size;

  Real max_abs_k = 0.0;
  Kokkos::parallel_reduce(
      "GlobalCurvatureMaxAbsK",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int idx, Real &result) {
        const int m = idx / nkji;
        int k = (idx - m * nkji) / nji;
        int j = (idx - m * nkji - k * nji) / nx1;
        const int i = (idx - m * nkji - k * nji - j * nx1) + is;
        k += ks;
        j += js;
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
        int j = (idx - m * nkji - k * nji) / nx1;
        const int i = (idx - m * nkji - k * nji - j * nx1) + is;
        k += ks;
        j += js;
        const Real inverse_spacing[3] = {
            1.0 / size.d_view(m).dx1,
            1.0 / size.d_view(m).dx2,
            1.0 / size.d_view(m).dx3};
        auto derivatives = z4c::MakeCellCenteredDerivativeProvider<Symmetry, NGHOST>(
            inverse_spacing, size.d_view, nx1, is, m, k, j, i);
        const auto diagnostic = ComputeZ4cCurvatureDiagnostics<NGHOST, false>(
            derivatives, adm.g_dd, adm.vK_dd, m, k, j, i);
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
  switch (config.stencil_width) {
    case 2:
      return cartoon
                 ? ComputeZ4cGlobalCurvatureMaximaImpl<z4c::CartoonSO2, 2>(pm)
                 : ComputeZ4cGlobalCurvatureMaximaImpl<z4c::Cartesian3D, 2>(pm);
    case 3:
      return cartoon
                 ? ComputeZ4cGlobalCurvatureMaximaImpl<z4c::CartoonSO2, 3>(pm)
                 : ComputeZ4cGlobalCurvatureMaximaImpl<z4c::Cartesian3D, 3>(pm);
    case 4:
      return cartoon
                 ? ComputeZ4cGlobalCurvatureMaximaImpl<z4c::CartoonSO2, 4>(pm)
                 : ComputeZ4cGlobalCurvatureMaximaImpl<z4c::Cartesian3D, 4>(pm);
    default:
      return {0.0, 0.0, false};
  }
}
