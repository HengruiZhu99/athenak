//========================================================================================
//! \file stability.cpp
//! \brief Robust-stability noise test for reference-frame first-order GH.
//========================================================================================
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "ref_gh/ref_gh.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {
Real initial_l2 = 0.0;

KOKKOS_INLINE_FUNCTION
unsigned long long MixBits(unsigned long long value) {  // NOLINT(runtime/int)
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

Real PerturbationL2(Mesh *mesh) {
  auto *pack = mesh->pmb_pack;
  auto &indcs = mesh->mb_indcs;
  const auto state = pack->prefgh->u0;
  const int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  Real sum = 0.0;
  Kokkos::parallel_reduce(
      "ref_gh robust energy", Kokkos::RangePolicy<>(DevExeSpace(),
      0, pack->nmb_thispack*ref_gh::nvar*ncells),
      KOKKOS_LAMBDA(const int idx, Real &local_sum) {
        int work = idx;
        const int i = work % indcs.nx1 + indcs.is; work /= indcs.nx1;
        const int j = work % indcs.nx2 + indcs.js; work /= indcs.nx2;
        const int k = work % indcs.nx3 + indcs.ks; work /= indcs.nx3;
        const int n = work % ref_gh::nvar;
        const int m = work/ref_gh::nvar;
        Real background = 0.0;
        if (n == ref_gh::PsiIndex(0, 0)) background = -1.0;
        if (n == ref_gh::PsiIndex(1, 1) || n == ref_gh::PsiIndex(2, 2)
            || n == ref_gh::PsiIndex(3, 3)) background = 1.0;
        const Real difference = state(m, n, k, j, i) - background;
        local_sum += difference*difference;
      }, sum);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  return std::sqrt(sum/(mesh->nmb_total*ncells*ref_gh::nvar));
}

void RefGhStabilityErrors(ParameterInput *pin, Mesh *mesh) {
  auto *pack = mesh->pmb_pack;
  const Real final_l2 = PerturbationL2(mesh);
  auto &indcs = mesh->mb_indcs;
  const auto state = pack->prefgh->u0;
  const int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  Real linf = 0.0;
  Kokkos::parallel_reduce(
      "ref_gh robust Linf", Kokkos::RangePolicy<>(DevExeSpace(),
      0, pack->nmb_thispack*ref_gh::nvar*ncells),
      KOKKOS_LAMBDA(const int idx, Real &maximum) {
        int work = idx;
        const int i = work % indcs.nx1 + indcs.is; work /= indcs.nx1;
        const int j = work % indcs.nx2 + indcs.js; work /= indcs.nx2;
        const int k = work % indcs.nx3 + indcs.ks; work /= indcs.nx3;
        const int n = work % ref_gh::nvar;
        const int m = work/ref_gh::nvar;
        Real background = 0.0;
        if (n == ref_gh::PsiIndex(0, 0)) background = -1.0;
        if (n == ref_gh::PsiIndex(1, 1) || n == ref_gh::PsiIndex(2, 2)
            || n == ref_gh::PsiIndex(3, 3)) background = 1.0;
        maximum = fmax(maximum, Kokkos::abs(state(m, n, k, j, i) - background));
      }, Kokkos::Max<Real>(linf));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &linf, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  const Real growth = final_l2/initial_l2;
  const Real rate = std::log(growth)/mesh->time;
  if (global_variable::my_rank == 0) {
    const std::string filename = pin->GetString("job", "basename") + "-stability.dat";
    FILE *file = std::fopen(filename.c_str(), "w");
    if (file == nullptr) std::exit(EXIT_FAILURE);
    std::fprintf(file, "# nx1 cycles time initial_L2 final_L2 growth rate Linf\n");
    std::fprintf(file, "%d %d %.17e %.17e %.17e %.17e %.17e %.17e\n",
                 mesh->mesh_indcs.nx1, mesh->ncycle, mesh->time, initial_l2,
                 final_l2, growth, rate, linf);
    std::fclose(file);
    std::cout << "reference-GH robust stability: growth=" << growth
              << ", rate=" << rate << ", Linf=" << linf << std::endl;
  }
}

}  // namespace

void ProblemGenerator::RefGhStability(ParameterInput *pin, const bool restart) {
  pgen_final_func = &RefGhStabilityErrors;
  if (restart) return;
  auto *pack = pmy_mesh_->pmb_pack;
  if (pack->prefgh == nullptr) {
    std::cout << "reference-GH stability data require a <ref_gh> block." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const Real amplitude = pin->GetOrAddReal("problem", "amp", 1.0e-10);
  auto &indcs = pack->pmesh->mb_indcs;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const int n2 = indcs.nx2 + 2*indcs.ng;
  const int n3 = indcs.nx3 + 2*indcs.ng;
  const auto state = pack->prefgh->u0;
  const int gid0 = pack->gids;
  par_for("ref_gh robust noise", DevExeSpace(), 0, pack->nmb_thispack - 1,
  0, ref_gh::nvar - 1, 0, n3 - 1, 0, n2 - 1, 0, n1 - 1,
  KOKKOS_LAMBDA(const int m, const int n, const int k, const int j, const int i) {
    unsigned long long key = static_cast<unsigned long long>(gid0 + m + 1); // NOLINT
    key = MixBits(key ^ (static_cast<unsigned long long>(n + 3) << 12));
    key = MixBits(key ^ (static_cast<unsigned long long>(k + 5) << 24));
    key = MixBits(key ^ (static_cast<unsigned long long>(j + 7) << 36));
    key = MixBits(key ^ (static_cast<unsigned long long>(i + 11) << 48));
    const Real unit = static_cast<Real>(key & 0x1fffffffffffffULL)
                      /static_cast<Real>(0x1fffffffffffffULL);
    Real background = 0.0;
    if (n == ref_gh::PsiIndex(0, 0)) background = -1.0;
    if (n == ref_gh::PsiIndex(1, 1) || n == ref_gh::PsiIndex(2, 2)
        || n == ref_gh::PsiIndex(3, 3)) background = 1.0;
    state(m, n, k, j, i) = background + amplitude*(2.0*unit - 1.0);
  });
  initial_l2 = PerturbationL2(pmy_mesh_);
}
