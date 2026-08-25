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
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "ref_gh/ref_gh.hpp"
#include "utils/finite_diff.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {
Real initial_l2 = 0.0;
Real initial_reduction_l2 = 0.0;

KOKKOS_INLINE_FUNCTION
unsigned long long MixBits(unsigned long long value) {  // NOLINT(runtime/int)
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

Real ModifiedWaveNumber(const int fd_order, const Real wave_number,
                        const Real spacing) {
  const Real phase = wave_number*spacing;
  if (fd_order == 2) return std::sin(phase)/spacing;
  if (fd_order == 4) {
    return (8.0*std::sin(phase) - std::sin(2.0*phase))/(6.0*spacing);
  }
  return (45.0*std::sin(phase) - 9.0*std::sin(2.0*phase)
          + std::sin(3.0*phase))/(30.0*spacing);
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

template <int FDNG>
Real ReductionL2(Mesh *mesh) {
  auto *pack = mesh->pmb_pack;
  auto &indcs = mesh->mb_indcs;
  auto &size = pack->pmb->mb_size;
  const auto state = pack->prefgh->u0;
  const int radius = FDNG - 1;
  const int n1 = indcs.nx1 - 2*radius;
  const int n2 = indcs.nx2 - 2*radius;
  const int n3 = indcs.nx3 - 2*radius;
  const int ncells = n1*n2*n3;
  Real sum = 0.0;
  Kokkos::parallel_reduce(
      "ref_gh robust reduction", Kokkos::RangePolicy<>(DevExeSpace(), 0,
      pack->nmb_thispack*3*ref_gh::kSymmetric4Size*ncells),
      KOKKOS_LAMBDA(const int index, Real &local_sum) {
    int work = index;
    const int i = work % n1 + indcs.is + radius; work /= n1;
    const int j = work % n2 + indcs.js + radius; work /= n2;
    const int k = work % n3 + indcs.ks + radius; work /= n3;
    const int component = work % ref_gh::kSymmetric4Size;
    work /= ref_gh::kSymmetric4Size;
    const int direction = work % 3;
    const int m = work/3;
    const Real idx[3] = {1.0/size.d_view(m).dx1, 1.0/size.d_view(m).dx2,
                         1.0/size.d_view(m).dx3};
    const Real reduction = Dx<FDNG>(
        direction, idx, state, m, ref_gh::kPsiOffset + component, k, j, i)
        - state(m, ref_gh::kPhiOffset
                     + direction*ref_gh::kSymmetric4Size + component,
                k, j, i);
    local_sum += reduction*reduction;
  }, Kokkos::Sum<Real>(sum));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  const Real samples = static_cast<Real>(
      mesh->nmb_total*3*ref_gh::kSymmetric4Size*n1*n2*n3);
  return std::sqrt(sum/samples);
}

Real ReductionL2(Mesh *mesh) {
  const int order = mesh->pmb_pack->prefgh->opt.fd_order;
  if (order == 2) return ReductionL2<2>(mesh);
  if (order == 4) return ReductionL2<3>(mesh);
  return ReductionL2<4>(mesh);
}

void RefGhStabilityErrors(ParameterInput *pin, Mesh *mesh) {
  auto *pack = mesh->pmb_pack;
  const Real final_l2 = PerturbationL2(mesh);
  const Real final_reduction_l2 = ReductionL2(mesh);
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
    std::fprintf(file, "# nx1 cycles time initial_L2 final_L2 growth rate Linf "
                       "initial_reduction_L2 final_reduction_L2 "
                       "reduction_growth\n");
    std::fprintf(file, "%d %d %.17e %.17e %.17e %.17e %.17e %.17e %.17e "
                       "%.17e %.17e\n",
                 mesh->mesh_indcs.nx1, mesh->ncycle, mesh->time, initial_l2,
                 final_l2, growth, rate, linf, initial_reduction_l2,
                 final_reduction_l2, final_reduction_l2/initial_reduction_l2);
    std::fclose(file);
    std::cout << "reference-GH robust stability: growth=" << growth
              << ", rate=" << rate << ", Linf=" << linf
              << ", reduction_growth="
              << final_reduction_l2/initial_reduction_l2 << std::endl;
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
  const std::string perturbation =
      pin->GetOrAddString("problem", "perturbation", "random");
  auto &indcs = pack->pmesh->mb_indcs;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const int n2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*indcs.ng : 1;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*indcs.ng : 1;
  const auto state = pack->prefgh->u0;
  const int gid0 = pack->gids;
  if (perturbation == "gh_transverse") {
    const Real length = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;
    const Real spacing = length/static_cast<Real>(pmy_mesh_->mesh_indcs.nx1);
    const Real wave_number = 2.0*M_PI/length;
    const Real modified_wave_number = ModifiedWaveNumber(
        pack->prefgh->opt.fd_order, wave_number, spacing);
    const Real gamma0 = pack->prefgh->opt.gamma0;
    const Real omega2 = modified_wave_number*modified_wave_number
                        - 0.25*gamma0*gamma0;
    if (!(omega2 > 0.0) || pack->prefgh->opt.gauge_driver_enabled) {
      std::cout << "GH transverse stability mode requires an underdamped wave "
                   "and gauge_driver_enabled=false." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    const Real omega = std::sqrt(omega2);
    const Real x1min = pmy_mesh_->mesh_size.x1min;
    auto &size = pack->pmb->mb_size;
    par_for("ref_gh transverse GH constraint", DevExeSpace(),
    0, pack->nmb_thispack - 1, 0, n3 - 1, 0, n2 - 1, 0, n1 - 1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      for (int n = 0; n < ref_gh::nvar; ++n) state(m, n, k, j, i) = 0.0;
      state(m, ref_gh::PsiIndex(0, 0), k, j, i) = -1.0;
      state(m, ref_gh::PsiIndex(1, 1), k, j, i) = 1.0;
      state(m, ref_gh::PsiIndex(2, 2), k, j, i) = 1.0;
      state(m, ref_gh::PsiIndex(3, 3), k, j, i) = 1.0;
      const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                                 size.d_view(m).x1min, size.d_view(m).x1max);
      const Real phase = wave_number*(x - x1min);
      const Real cosine = Kokkos::cos(phase);
      const Real sine = Kokkos::sin(phase);
      const Real h0y = amplitude*(0.5*gamma0*cosine + omega*sine)
                       /(modified_wave_number*modified_wave_number);
      const Real dx_h0y = amplitude*(-0.5*gamma0*sine + omega*cosine)
                           /modified_wave_number;
      state(m, ref_gh::PsiIndex(0, 2), k, j, i) = h0y;
      state(m, ref_gh::PiIndex(0, 2), k, j, i) = amplitude*cosine;
      state(m, ref_gh::PhiIndex(0, 0, 2), k, j, i) = dx_h0y;
    });
    initial_l2 = PerturbationL2(pmy_mesh_);
    initial_reduction_l2 = ReductionL2(pmy_mesh_);
    return;
  }
  if (perturbation != "random") {
    std::cout << "Unknown Ref-GH stability perturbation: " << perturbation
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
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
  initial_reduction_l2 = ReductionL2(pmy_mesh_);
}
