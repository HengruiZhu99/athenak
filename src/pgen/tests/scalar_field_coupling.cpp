//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file scalar_field_coupling.cpp
//! \brief One-step homogeneous scalar-field/Z4c backreaction regression.

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "scalar_field/scalar_field.hpp"
#include "z4c/tmunu.hpp"
#include "z4c/z4c.hpp"

namespace {

constexpr int kNumAverages = 12;
constexpr int kVolumeIndex = kNumAverages;
constexpr int kNumReductions = kNumAverages + 1;

void ScalarFieldCouplingDiagnostic(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  auto &z4c = pmbp->pz4c->z4c;
  const int nx1 = indcs.nx1;
  const int nx2 = indcs.nx2;
  const int nx3 = indcs.nx3;
  const int is = indcs.is;
  const int js = indcs.js;
  const int ks = indcs.ks;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  const int ncell = pmbp->nmb_thispack*nkji;
  const bool has_tmunu = pmbp->ptmunu != nullptr;
  DvceArray5D<Real> tmunu_data;
  if (has_tmunu) {
    tmunu_data = pmbp->ptmunu->u_tmunu;
  }

  array_sum::GlobalSum local_sum;
  Kokkos::parallel_reduce(
      "scalar coupling diagnostic",
      Kokkos::RangePolicy<DevExeSpace>(0, ncell),
      KOKKOS_LAMBDA(const int idx, array_sum::GlobalSum &sum) {
        const int m = idx/nkji;
        const int k0 = (idx - m*nkji)/nji;
        const int j0 = (idx - m*nkji - k0*nji)/nx1;
        const int i0 = idx - m*nkji - k0*nji - j0*nx1;
        const int k = ks + k0;
        const int j = js + j0;
        const int i = is + i0;
        const Real volume =
            size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;

        array_sum::GlobalSum point;
        point.the_array[0] = volume*z4c.vKhat(m, k, j, i);
        point.the_array[1] = volume*z4c.vTheta(m, k, j, i);
        for (int n = 0; n < Tmunu::N_Tmunu; ++n) {
          point.the_array[2 + n] =
              has_tmunu ? volume*tmunu_data(m, n, k, j, i) : 0.0;
        }
        point.the_array[kVolumeIndex] = volume;
        for (int n = kNumReductions; n < NREDUCTION_VARIABLES; ++n) {
          point.the_array[n] = 0.0;
        }
        sum += point;
      },
      Kokkos::Sum<array_sum::GlobalSum>(local_sum));

  Real reduced[kNumReductions];
  for (int n = 0; n < kNumReductions; ++n) {
    reduced[n] = local_sum.the_array[n];
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, reduced, kNumReductions, MPI_ATHENA_REAL,
                MPI_SUM, MPI_COMM_WORLD);
#endif
  for (int n = 0; n < kNumAverages; ++n) {
    reduced[n] /= reduced[kVolumeIndex];
  }

  if (global_variable::my_rank == 0) {
    std::string filename = pin->GetString("job", "basename");
    filename.append("-coupling.dat");
    FILE *file = std::fopen(filename.c_str(), "w");
    if (file == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
                << __LINE__ << std::endl
                << "Scalar coupling diagnostic file could not be opened"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    std::fprintf(
        file,
        "# cycle time dt backreaction has_tmunu Khat Theta "
        "Sxx Sxy Sxz Syy Syz Szz E Sx Sy Sz\n");
    std::fprintf(file, "%d %.17e %.17e %d %d", pm->ncycle, pm->time,
                 pm->dt_last_completed,
                 static_cast<int>(pmbp->pscalar->backreaction),
                 static_cast<int>(has_tmunu));
    for (int n = 0; n < kNumAverages; ++n) {
      std::fprintf(file, " %.17e", reduced[n]);
    }
    std::fprintf(file, "\n");
    std::fclose(file);
  }
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::ScalarFieldCoupling()
//! \brief Initialize homogeneous Minkowski Z4c and a constant real scalar.

void ProblemGenerator::ScalarFieldCoupling(ParameterInput *pin,
                                           const bool restart) {
  pgen_final_func = ScalarFieldCouplingDiagnostic;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pz4c == nullptr || pmbp->pscalar == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
              << __LINE__ << std::endl
              << "Scalar coupling test requires <z4c> and <scalar_field> blocks"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmbp->pscalar->ncomponents != 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
              << __LINE__ << std::endl
              << "Scalar coupling test requires a real scalar field"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmbp->pmhd != nullptr && pmbp->pdyngr == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
              << __LINE__ << std::endl
              << "Scalar coupling MHD case requires dynamical GRMHD"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (restart) {
    return;
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const int n2 = indcs.nx2 + 2*indcs.ng;
  const int n3 = indcs.nx3 + 2*indcs.ng;
  const int nmb = pmbp->nmb_thispack;
  const Real amplitude =
      pin->GetOrAddReal("problem", "amplitude", 0.2);
  auto &pz4c = pmbp->pz4c;
  auto &pscalar = pmbp->pscalar;
  auto &z4c = pz4c->z4c;
  auto &scalar = pscalar->u0;
  const int i_phi = pscalar->I_SF_PHI0;

  Kokkos::deep_copy(pz4c->u0, 0.0);
  Kokkos::deep_copy(pz4c->u1, 0.0);
  Kokkos::deep_copy(pz4c->u_rhs, 0.0);
  Kokkos::deep_copy(pscalar->u0, 0.0);
  Kokkos::deep_copy(pscalar->u1, 0.0);
  Kokkos::deep_copy(pscalar->u_rhs, 0.0);

  par_for(
      "pgen scalar coupling", DevExeSpace(), 0, nmb - 1,
      0, n3 - 1, 0, n2 - 1, 0, n1 - 1,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        z4c.chi(m, k, j, i) = 1.0;
        z4c.alpha(m, k, j, i) = 1.0;
        for (int direction = 0; direction < 3; ++direction) {
          z4c.g_dd(m, direction, direction, k, j, i) = 1.0;
        }
        scalar(m, i_phi, k, j, i) = amplitude;
      });
  pz4c->Z4cToADM(pmbp);

  if (pmbp->pmhd != nullptr) {
    const Real density =
        pin->GetOrAddReal("problem", "fluid_density", 1.25);
    const Real pressure =
        pin->GetOrAddReal("problem", "fluid_pressure", 0.3);
    const Real magnetic_x =
        pin->GetOrAddReal("problem", "magnetic_x", 0.4);
    const Real magnetic_y =
        pin->GetOrAddReal("problem", "magnetic_y", -0.2);
    const Real magnetic_z =
        pin->GetOrAddReal("problem", "magnetic_z", 0.1);
    auto &pmhd = pmbp->pmhd;
    auto &primitive = pmhd->w0;
    auto &bcc = pmhd->bcc0;

    Kokkos::deep_copy(pmhd->u0, 0.0);
    Kokkos::deep_copy(pmhd->u1, 0.0);
    Kokkos::deep_copy(primitive, 0.0);
    Kokkos::deep_copy(bcc, 0.0);
    Kokkos::deep_copy(pmhd->b0.x1f, magnetic_x);
    Kokkos::deep_copy(pmhd->b0.x2f, magnetic_y);
    Kokkos::deep_copy(pmhd->b0.x3f, magnetic_z);
    Kokkos::deep_copy(pmhd->b1.x1f, magnetic_x);
    Kokkos::deep_copy(pmhd->b1.x2f, magnetic_y);
    Kokkos::deep_copy(pmhd->b1.x3f, magnetic_z);

    par_for(
        "pgen scalar coupling mhd", DevExeSpace(), 0, nmb - 1,
        0, n3 - 1, 0, n2 - 1, 0, n1 - 1,
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
          primitive(m, IDN, k, j, i) = density;
          primitive(m, IPR, k, j, i) = pressure;
          bcc(m, IBX, k, j, i) = magnetic_x;
          bcc(m, IBY, k, j, i) = magnetic_y;
          bcc(m, IBZ, k, j, i) = magnetic_z;
        });
    pmbp->pdyngr->PrimToConInit(
        0, n1 - 1, 0, n2 - 1, 0, n3 - 1);
  }
}
