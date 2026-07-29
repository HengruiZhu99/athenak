//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file scalar_field_plane_wave.cpp
//! \brief Analytic real-scalar plane wave on a constant, nontrivial ADM background.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "scalar_field/scalar_field.hpp"

namespace {

constexpr Real kAlpha = 0.9;
constexpr Real kBetaX = 0.12;
constexpr Real kBetaY = -0.04;
constexpr Real kBetaZ = 0.0;
constexpr Real kGxx = 1.0;
constexpr Real kGxy = 0.25;
constexpr Real kGxz = 0.0;
constexpr Real kGyy = 1.0;
constexpr Real kGyz = 0.0;
constexpr Real kGzz = 1.0;
constexpr Real kMetricDeterminant = 0.9375;
constexpr Real kMass = 0.7;
constexpr Real kAmplitude = 1.0e-3;
constexpr Real kTwoPi = 6.283185307179586476925286766559;
constexpr Real kWaveX = kTwoPi;
constexpr Real kWaveY = kTwoPi;
constexpr Real kWaveZ = 0.0;

KOKKOS_INLINE_FUNCTION
Real NormalFrequency() {
  constexpr Real inverse_xx = 16.0/15.0;
  constexpr Real inverse_xy = -4.0/15.0;
  constexpr Real inverse_yy = 16.0/15.0;
  constexpr Real inverse_zz = 1.0;
  const Real wave_number_squared =
      inverse_xx*kWaveX*kWaveX + 2.0*inverse_xy*kWaveX*kWaveY +
      inverse_yy*kWaveY*kWaveY + inverse_zz*kWaveZ*kWaveZ;
  return sqrt(wave_number_squared + kMass*kMass);
}

KOKKOS_INLINE_FUNCTION
void AnalyticState(const Real x1, const Real x2, const Real x3, const Real time,
                   Real *phi, Real *pi) {
  const Real omega = NormalFrequency();
  const Real coordinate_frequency =
      -(kBetaX*kWaveX + kBetaY*kWaveY + kBetaZ*kWaveZ) + kAlpha*omega;
  const Real phase =
      kWaveX*x1 + kWaveY*x2 + kWaveZ*x3 - coordinate_frequency*time;
  *phi = kAmplitude*sin(phase);
  *pi = kAmplitude*omega*cos(phase);
}

void SetScalarFieldWaveADM(MeshBlockPack *pmbp) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const int n2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*indcs.ng : 1;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*indcs.ng : 1;
  const int nmb = pmbp->nmb_thispack;
  auto &adm = pmbp->padm->adm;

  par_for("pgen_scalar_wave_adm", DevExeSpace(), 0, nmb - 1, 0, n3 - 1,
  0, n2 - 1, 0, n1 - 1, KOKKOS_LAMBDA(int m, int k, int j, int i) {
    adm.alpha(m, k, j, i) = kAlpha;
    adm.beta_u(m, 0, k, j, i) = kBetaX;
    adm.beta_u(m, 1, k, j, i) = kBetaY;
    adm.beta_u(m, 2, k, j, i) = kBetaZ;
    adm.psi4(m, k, j, i) = pow(kMetricDeterminant, 1.0/3.0);

    adm.g_dd(m, 0, 0, k, j, i) = kGxx;
    adm.g_dd(m, 0, 1, k, j, i) = kGxy;
    adm.g_dd(m, 0, 2, k, j, i) = kGxz;
    adm.g_dd(m, 1, 1, k, j, i) = kGyy;
    adm.g_dd(m, 1, 2, k, j, i) = kGyz;
    adm.g_dd(m, 2, 2, k, j, i) = kGzz;

    adm.vK_dd(m, 0, 0, k, j, i) = 0.0;
    adm.vK_dd(m, 0, 1, k, j, i) = 0.0;
    adm.vK_dd(m, 0, 2, k, j, i) = 0.0;
    adm.vK_dd(m, 1, 1, k, j, i) = 0.0;
    adm.vK_dd(m, 1, 2, k, j, i) = 0.0;
    adm.vK_dd(m, 2, 2, k, j, i) = 0.0;
  });
}

void ScalarFieldWaveErrors(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int nx1 = indcs.nx1;
  const int nx2 = indcs.nx2;
  const int nx3 = indcs.nx3;
  const int is = indcs.is;
  const int js = indcs.js;
  const int ks = indcs.ks;
  const int nmkji = pmbp->nmb_thispack*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  const int i_phi = pmbp->pscalar->I_SF_PHI0;
  const int i_pi = pmbp->pscalar->I_SF_PI0;
  const Real time = pm->time;
  auto &size = pmbp->pmb->mb_size;
  auto &u0 = pmbp->pscalar->u0;

  array_sum::GlobalSum local_sum;
  Real local_linf = 0.0;
  Kokkos::parallel_reduce(
      "scalar_wave_errors", Kokkos::RangePolicy<DevExeSpace>(0, nmkji),
      KOKKOS_LAMBDA(const int idx, array_sum::GlobalSum &sum,
                    Real &max_error) {
        const int m = idx/nkji;
        const int k0 = (idx - m*nkji)/nji;
        const int j0 = (idx - m*nkji - k0*nji)/nx1;
        const int i0 = idx - m*nkji - k0*nji - j0*nx1;
        const int i = i0 + is;
        const int j = j0 + js;
        const int k = k0 + ks;

        const Real x1 = CellCenterX(
            i0, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
        const Real x2 = CellCenterX(
            j0, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
        const Real x3 = CellCenterX(
            k0, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
        const Real volume = size.d_view(m).dx1*size.d_view(m).dx2*
                            size.d_view(m).dx3*sqrt(kMetricDeterminant);
        Real exact_phi;
        Real exact_pi;
        AnalyticState(x1, x2, x3, time, &exact_phi, &exact_pi);
        const Real phi_error =
            fabs(u0(m, i_phi, k, j, i) - exact_phi);
        const Real pi_error =
            fabs(u0(m, i_pi, k, j, i) - exact_pi);

        array_sum::GlobalSum point;
        point.the_array[0] = volume*phi_error;
        point.the_array[1] = volume*pi_error;
        point.the_array[2] = volume;
        for (int n = 3; n < NREDUCTION_VARIABLES; ++n) {
          point.the_array[n] = 0.0;
        }
        sum += point;
        max_error = fmax(max_error, fmax(phi_error, pi_error));
      },
      Kokkos::Sum<array_sum::GlobalSum>(local_sum),
      Kokkos::Max<Real>(local_linf));

  Real reduced[4] = {
    local_sum.the_array[0], local_sum.the_array[1],
    local_sum.the_array[2], local_linf
  };
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, reduced, 3, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &reduced[3], 1, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
#endif
  const Real phi_l1 = reduced[0]/reduced[2];
  const Real pi_l1 = reduced[1]/reduced[2];
  const int ncreated = (pm->adaptive) ? pm->pmr->nmb_created : 0;
  const int ndeleted = (pm->adaptive) ? pm->pmr->nmb_deleted : 0;
  const int nlevels = pm->max_level - pm->root_level + 1;

  if (global_variable::my_rank == 0) {
    std::string filename = pin->GetString("job", "basename");
    filename.append("-errs.dat");
    FILE *file = std::fopen(filename.c_str(), "r");
    if (file != nullptr) {
      file = std::freopen(filename.c_str(), "a", file);
    } else {
      file = std::fopen(filename.c_str(), "w");
      if (file != nullptr) {
        std::fprintf(
            file, "# Nx1  Nx2  Nx3  Ncycle  time  phi_L1  Pi_L1  "
                  "max_Linf  Nmb  Ncreated  Ndeleted  Nlevels\n");
      }
    }
    if (file == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
                << __LINE__ << std::endl
                << "Scalar-field error file could not be opened" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    std::fprintf(file,
                 "%04d  %04d  %04d  %05d  %.17e  %.17e  %.17e  %.17e  "
                 "%05d  %05d  %05d  %03d\n",
                 pm->mesh_indcs.nx1, pm->mesh_indcs.nx2, pm->mesh_indcs.nx3,
                 pm->ncycle, time, phi_l1, pi_l1, reduced[3],
                 pm->nmb_total, ncreated, ndeleted, nlevels);
    std::fclose(file);
  }
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::ScalarFieldPlaneWave()
//! \brief Initialize a periodic scalar wave and its constant fixed-ADM background.

void ProblemGenerator::ScalarFieldPlaneWave(ParameterInput *pin, const bool restart) {
  pgen_final_func = ScalarFieldWaveErrors;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->padm == nullptr || pmbp->pscalar == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "Scalar plane wave requires <adm> and <scalar_field> blocks"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmbp->padm->is_dynamic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Scalar plane wave requires adm/dynamic=false"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  pmbp->padm->SetADMVariables = SetScalarFieldWaveADM;
  pmbp->padm->SetADMVariables(pmbp);
  if (restart) {
    return;
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is;
  const int js = indcs.js;
  const int ks = indcs.ks;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const int n2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*indcs.ng : 1;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*indcs.ng : 1;
  const int nmb = pmbp->nmb_thispack;
  const int i_phi = pmbp->pscalar->I_SF_PHI0;
  const int i_pi = pmbp->pscalar->I_SF_PI0;
  auto &size = pmbp->pmb->mb_size;
  auto &u0 = pmbp->pscalar->u0;

  par_for("pgen_scalar_wave", DevExeSpace(), 0, nmb - 1, 0, n3 - 1,
  0, n2 - 1, 0, n1 - 1, KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x1 = CellCenterX(
        i - is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    const Real x2 = CellCenterX(
        j - js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    const Real x3 = CellCenterX(
        k - ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    Real phi;
    Real pi;
    AnalyticState(x1, x2, x3, 0.0, &phi, &pi);
    u0(m, i_phi, k, j, i) = phi;
    u0(m, i_pi, k, j, i) = pi;
  });
}
