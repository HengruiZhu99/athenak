//========================================================================================
//! \file linear_wave.cpp
//! \brief Weak periodic TT wave for reference-frame GH convergence tests.
//========================================================================================
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "ref_gh/ref_gh.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {

void SetRefGhLinearWave(ParameterInput *pin, Mesh *mesh, DvceArray5D<Real> state,
                        const Real time) {
  auto *pack = mesh->pmb_pack;
  auto &indcs = mesh->mb_indcs;
  auto &size = pack->pmb->mb_size;
  const Real amplitude = pin->GetOrAddReal("problem", "amp", 1.0e-8);
  const Real wave_number = 2.0*M_PI/(mesh->mesh_size.x1max - mesh->mesh_size.x1min);
  par_for("ref_gh linear wave", DevExeSpace(), 0, pack->nmb_thispack - 1,
  indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                               size.d_view(m).x1min, size.d_view(m).x1max);
    const Real phase = wave_number*(x - time);
    const Real perturbation = amplitude*std::sin(phase);
    const Real derivative = wave_number*amplitude*std::cos(phase);
    for (int n = 0; n < ref_gh::nvar; ++n) state(m, n, k, j, i) = 0.0;
    state(m, ref_gh::PsiIndex(0, 0), k, j, i) = -1.0;
    state(m, ref_gh::PsiIndex(1, 1), k, j, i) = 1.0;
    state(m, ref_gh::PsiIndex(2, 2), k, j, i) = 1.0 + perturbation;
    state(m, ref_gh::PsiIndex(3, 3), k, j, i) = 1.0 - perturbation;
    state(m, ref_gh::PiIndex(2, 2), k, j, i) = derivative;
    state(m, ref_gh::PiIndex(3, 3), k, j, i) = -derivative;
    state(m, ref_gh::PhiIndex(0, 2, 2), k, j, i) = derivative;
    state(m, ref_gh::PhiIndex(0, 3, 3), k, j, i) = -derivative;
  });
}

void RefGhLinearWaveErrors(ParameterInput *pin, Mesh *mesh) {
  auto *pack = mesh->pmb_pack;
  SetRefGhLinearWave(pin, mesh, pack->prefgh->u1, mesh->time);
  switch (pack->prefgh->opt.fd_order) {
    case 2: pack->prefgh->CalcConstraints<2>(); break;
    case 4: pack->prefgh->CalcConstraints<3>(); break;
    case 6: pack->prefgh->CalcConstraints<4>(); break;
  }
  auto &indcs = mesh->mb_indcs;
  auto &size = pack->pmb->mb_size;
  const auto numerical = pack->prefgh->u0;
  const auto exact = pack->prefgh->u1;
  const auto constraints = pack->prefgh->u_con;
  const int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  Real l1 = 0.0;
  Real linf = 0.0;
  Real constraint_linf = 0.0;
  Kokkos::parallel_reduce(
      "ref_gh wave L1", Kokkos::RangePolicy<>(DevExeSpace(),
      0, pack->nmb_thispack*2*ncells),
      KOKKOS_LAMBDA(const int idx, Real &sum) {
        int work = idx;
        const int i = work % indcs.nx1 + indcs.is; work /= indcs.nx1;
        const int j = work % indcs.nx2 + indcs.js; work /= indcs.nx2;
        const int k = work % indcs.nx3 + indcs.ks; work /= indcs.nx3;
        const int component = work % 2;
        const int m = work/2;
        const int n = component == 0 ? ref_gh::PsiIndex(2, 2)
                                     : ref_gh::PsiIndex(3, 3);
        const Real volume = size.d_view(m).dx1*size.d_view(m).dx2
                            *size.d_view(m).dx3;
        sum += volume*Kokkos::abs(numerical(m, n, k, j, i)
                                  - exact(m, n, k, j, i));
      }, l1);
  Kokkos::parallel_reduce(
      "ref_gh wave Linf", Kokkos::RangePolicy<>(DevExeSpace(),
      0, pack->nmb_thispack*ncells),
      KOKKOS_LAMBDA(const int idx, Real &maximum) {
        int work = idx;
        const int i = work % indcs.nx1 + indcs.is; work /= indcs.nx1;
        const int j = work % indcs.nx2 + indcs.js; work /= indcs.nx2;
        const int k = work % indcs.nx3 + indcs.ks;
        const int m = work/indcs.nx3;
        for (int n = 0; n < ref_gh::RefGh::kNativeConstraints; ++n) {
          maximum = fmax(maximum, Kokkos::abs(constraints(m, n, k, j, i)));
        }
      }, Kokkos::Max<Real>(constraint_linf));
  Kokkos::parallel_reduce(
      "ref_gh wave field Linf", Kokkos::RangePolicy<>(DevExeSpace(),
      0, pack->nmb_thispack*2*ncells),
      KOKKOS_LAMBDA(const int idx, Real &maximum) {
        int work = idx;
        const int i = work % indcs.nx1 + indcs.is; work /= indcs.nx1;
        const int j = work % indcs.nx2 + indcs.js; work /= indcs.nx2;
        const int k = work % indcs.nx3 + indcs.ks; work /= indcs.nx3;
        const int component = work % 2;
        const int m = work/2;
        const int n = component == 0 ? ref_gh::PsiIndex(2, 2)
                                     : ref_gh::PsiIndex(3, 3);
        maximum = fmax(maximum, Kokkos::abs(numerical(m, n, k, j, i)
                                            - exact(m, n, k, j, i)));
      }, Kokkos::Max<Real>(linf));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &l1, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &linf, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &constraint_linf, 1, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
#endif
  const Real volume = (mesh->mesh_size.x1max - mesh->mesh_size.x1min)
                      *(mesh->mesh_size.x2max - mesh->mesh_size.x2min)
                      *(mesh->mesh_size.x3max - mesh->mesh_size.x3min);
  l1 /= 2.0*volume;
  if (global_variable::my_rank == 0) {
    const std::string filename = pin->GetString("job", "basename") + "-errors.dat";
    FILE *file = std::fopen(filename.c_str(), "w");
    if (file == nullptr) std::exit(EXIT_FAILURE);
    std::fprintf(file, "# nx1 nx2 nx3 cycles metric_L1 metric_Linf constraint_Linf\n");
    std::fprintf(file, "%d %d %d %d %.17e %.17e %.17e\n",
                 mesh->mesh_indcs.nx1, mesh->mesh_indcs.nx2, mesh->mesh_indcs.nx3,
                 mesh->ncycle, l1, linf, constraint_linf);
    std::fclose(file);
    std::cout << "reference-GH linear-wave metric L1 = " << l1
              << ", Linf = " << linf
              << ", constraint Linf = " << constraint_linf << std::endl;
  }
}

}  // namespace

void ProblemGenerator::RefGhLinearWave(ParameterInput *pin, const bool restart) {
  pgen_final_func = &RefGhLinearWaveErrors;
  if (restart) return;
  if (pmy_mesh_->pmb_pack->prefgh == nullptr) {
    std::cout << "reference-GH linear-wave data require a <ref_gh> block." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  SetRefGhLinearWave(pin, pmy_mesh_, pmy_mesh_->pmb_pack->prefgh->u0, 0.0);
}
