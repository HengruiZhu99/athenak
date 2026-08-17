//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file linear_wave.cpp
//! \brief Periodic transverse-traceless wave for vacuum FO-GH convergence tests.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "fo_gh/fo_gh.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {

struct WaveData {
  Real k[3];
  Real polarization[6];
  Real knorm;
  Real amp;
};

WaveData ReadWaveData(ParameterInput *pin, Mesh *pm) {
  const Real lengths[3] = {
      pm->mesh_size.x1max - pm->mesh_size.x1min,
      pm->mesh_size.x2max - pm->mesh_size.x2min,
      pm->mesh_size.x3max - pm->mesh_size.x3min};
  WaveData wave;
  wave.k[0] = pin->GetOrAddReal("problem", "kx1", 1.0/lengths[0]);
  wave.k[1] = pin->GetOrAddReal("problem", "kx2", 0.0);
  wave.k[2] = pin->GetOrAddReal("problem", "kx3", 0.0);
  wave.amp = pin->GetOrAddReal("problem", "amp", 1.0e-8);
  wave.knorm = std::sqrt(SQR(wave.k[0]) + SQR(wave.k[1]) + SQR(wave.k[2]));
  const Real theta = std::atan2(std::sqrt(SQR(wave.k[0]) + SQR(wave.k[1])),
                                wave.k[2]);
  const Real phi = std::atan2(wave.k[0], wave.k[1]);
  wave.polarization[0] = -SQR(std::cos(theta))*std::cos(2.0*phi)
                         - SQR(std::cos(phi))*SQR(std::sin(theta));
  wave.polarization[1] = -0.25*(3.0 + std::cos(2.0*theta))*std::sin(2.0*phi);
  wave.polarization[2] = -std::cos(theta)*std::sin(theta)*std::sin(phi);
  wave.polarization[3] = SQR(std::cos(theta))*std::cos(2.0*phi)
                         - SQR(std::sin(theta))*SQR(std::sin(phi));
  wave.polarization[4] = std::cos(theta)*std::sin(theta)*std::cos(phi);
  wave.polarization[5] = SQR(std::sin(theta));
  return wave;
}

void SetLinearWave(ParameterInput *pin, Mesh *pm, DvceArray5D<Real> state,
                   const Real time) {
  auto *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  const WaveData wave = ReadWaveData(pin, pm);
  par_for("fo_gh linear wave", DevExeSpace(), 0, pmbp->nmb_thispack - 1,
  indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                               size.d_view(m).x1min, size.d_view(m).x1max);
    const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                               size.d_view(m).x2min, size.d_view(m).x2max);
    const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                               size.d_view(m).x3min, size.d_view(m).x3max);
    const Real phase = 2.0*M_PI*(wave.k[0]*x + wave.k[1]*y
                                 + wave.k[2]*z - wave.knorm*time);
    const Real metric_factor = wave.amp*std::sin(phase);
    const Real curvature_factor = M_PI*wave.knorm*wave.amp*std::cos(phase);
    for (int n = 0; n < fo_gh::nvar; ++n) {
      state(m, n, k, j, i) = 0.0;
    }
    state(m, fo_gh::I_CHI, k, j, i) = 1.0;
    state(m, fo_gh::I_ALPHA, k, j, i) = 1.0;
    const int metric_index[6] = {fo_gh::I_TGXX, fo_gh::I_TGXY, fo_gh::I_TGXZ,
                                 fo_gh::I_TGYY, fo_gh::I_TGYZ, fo_gh::I_TGZZ};
    const int curvature_index[6] = {fo_gh::I_TAXX, fo_gh::I_TAXY,
                                    fo_gh::I_TAXZ, fo_gh::I_TAYY,
                                    fo_gh::I_TAYZ, fo_gh::I_TAZZ};
    for (int n = 0; n < 6; ++n) {
      const bool diagonal = (n == 0 || n == 3 || n == 5);
      state(m, metric_index[n], k, j, i) =
          (diagonal ? 1.0 : 0.0) + wave.polarization[n]*metric_factor;
      state(m, curvature_index[n], k, j, i) =
          wave.polarization[n]*curvature_factor;
      for (int p = 0; p < 3; ++p) {
        state(m, fo_gh::I_QXXX + 6*p + n, k, j, i) =
            wave.polarization[n]*wave.amp*2.0*M_PI*wave.k[p]*std::cos(phase);
      }
    }
  });
}

void FoGhLinearWaveErrors(ParameterInput *pin, Mesh *pm) {
  auto *pmbp = pm->pmb_pack;
  SetLinearWave(pin, pm, pmbp->pfogh->u1, pm->time);
  auto &indcs = pm->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  const auto numerical = pmbp->pfogh->u0;
  const auto exact = pmbp->pfogh->u1;
  const int metric_index[6] = {fo_gh::I_TGXX, fo_gh::I_TGXY, fo_gh::I_TGXZ,
                               fo_gh::I_TGYY, fo_gh::I_TGYZ, fo_gh::I_TGZZ};
  const int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  Real l1 = 0.0;
  Real linf = 0.0;
  Kokkos::parallel_reduce(
      "fo_gh linear wave L1", Kokkos::RangePolicy<>(DevExeSpace(),
      0, pmbp->nmb_thispack*6*ncells),
      KOKKOS_LAMBDA(const int idx, Real &sum) {
        int work = idx;
        const int i = work % indcs.nx1 + indcs.is;
        work /= indcs.nx1;
        const int j = work % indcs.nx2 + indcs.js;
        work /= indcs.nx2;
        const int k = work % indcs.nx3 + indcs.ks;
        work /= indcs.nx3;
        const int n = work % 6;
        const int m = work/6;
        const Real volume = size.d_view(m).dx1*size.d_view(m).dx2
                            *size.d_view(m).dx3;
        sum += volume*Kokkos::abs(numerical(m, metric_index[n], k, j, i)
                                  - exact(m, metric_index[n], k, j, i));
      }, l1);
  Kokkos::parallel_reduce(
      "fo_gh linear wave Linf", Kokkos::RangePolicy<>(DevExeSpace(),
      0, pmbp->nmb_thispack*6*ncells),
      KOKKOS_LAMBDA(const int idx, Real &maximum) {
        int work = idx;
        const int i = work % indcs.nx1 + indcs.is;
        work /= indcs.nx1;
        const int j = work % indcs.nx2 + indcs.js;
        work /= indcs.nx2;
        const int k = work % indcs.nx3 + indcs.ks;
        work /= indcs.nx3;
        const int n = work % 6;
        const int m = work/6;
        maximum = fmax(maximum,
            Kokkos::abs(numerical(m, metric_index[n], k, j, i)
                        - exact(m, metric_index[n], k, j, i)));
      }, Kokkos::Max<Real>(linf));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &l1, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &linf, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  const Real domain_volume =
      (pm->mesh_size.x1max - pm->mesh_size.x1min)
      *(pm->mesh_size.x2max - pm->mesh_size.x2min)
      *(pm->mesh_size.x3max - pm->mesh_size.x3min);
  l1 /= 6.0*domain_volume;
  if (global_variable::my_rank == 0) {
    const std::string filename = pin->GetString("job", "basename") + "-errors.dat";
    FILE *file = std::fopen(filename.c_str(), "w");
    if (file == nullptr) {
      std::cout << "Unable to open " << filename << std::endl;
      std::exit(EXIT_FAILURE);
    }
    std::fprintf(file, "# nx1 nx2 nx3 cycles metric_L1 metric_Linf\n");
    std::fprintf(file, "%d %d %d %d %.17e %.17e\n",
                 pm->mesh_indcs.nx1, pm->mesh_indcs.nx2, pm->mesh_indcs.nx3,
                 pm->ncycle, l1, linf);
    std::fclose(file);
    std::cout << "FO-GH linear-wave metric L1 = " << l1
              << ", Linf = " << linf << std::endl;
  }
}

} // namespace

void ProblemGenerator::FoGhLinearWave(ParameterInput *pin, const bool restart) {
  pgen_final_func = &FoGhLinearWaveErrors;
  if (restart) {
    return;
  }
  if (pmy_mesh_->pmb_pack->pfogh == nullptr) {
    std::cout << "FO-GH linear-wave data require an <fo_gh> block." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const WaveData wave = ReadWaveData(pin, pmy_mesh_);
  const Real periods = pin->GetReal("time", "tlim");
  pin->SetReal("time", "tlim", periods/wave.knorm);
  SetLinearWave(pin, pmy_mesh_, pmy_mesh_->pmb_pack->pfogh->u0, 0.0);
}
