//========================================================================================
//! \file stationary_trumpet.cpp
//! \brief Exact regular state for the stationary reference-frame trumpet gate.
//========================================================================================
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "ref_gh/gauge_driver.hpp"
#include "ref_gh/physical_gauge_target.hpp"
#include "ref_gh/puncture_exponent.hpp"
#include "ref_gh/ref_gh.hpp"
#include "ref_gh/ref_gh_geometry.hpp"
#include "ref_gh/stationary_gauge_data.hpp"
#include "utils/finite_diff.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {
Real initial_rhs_linf = 0.0;
constexpr int kFixedRadiusCount = 3;
constexpr Real kFixedMinimumRadius[kFixedRadiusCount] = {0.5, 1.0, 1.5};
constexpr Real kFixedMaximumRadius[kFixedRadiusCount] = {1.0, 1.5, 2.0};
Real initial_rhs_fixed_linf[kFixedRadiusCount] = {};
Real initial_reference_ricci_linf = 0.0;
Real initial_frame_ricci_linf = 0.0;
Real initial_spin_antisymmetry_linf = 0.0;
Real initial_structure_antisymmetry_linf = 0.0;
int initial_rhs_component = -1;
Real initial_rhs_radius = -1.0;
bool perturbed_trumpet = false;
Real perturbation_amplitude = 0.0;
Real perturbation_width = 0.0;

void CheckRefGhStationaryTrumpet(ParameterInput *pin, Mesh *mesh) {
  auto *pack = mesh->pmb_pack;
  switch (pack->prefgh->opt.fd_order) {
    case 2: pack->prefgh->CalcConstraints<2>(); break;
    case 4: pack->prefgh->CalcConstraints<3>(); break;
    case 6: pack->prefgh->CalcConstraints<4>(); break;
  }
  auto &indcs = mesh->mb_indcs;
  auto &size = pack->pmb->mb_size;
  const auto state = pack->prefgh->u0;
  const auto constraints = pack->prefgh->u_con;
  const Real center_x = pack->prefgh->opt.reference_center[0];
  const Real center_y = pack->prefgh->opt.reference_center[1];
  const Real center_z = pack->prefgh->opt.reference_center[2];
  const Real reference_mass = pack->prefgh->opt.reference_mass;
  const auto reference_table = pack->prefgh->reference_table;
  const bool compare_stationary_gauge =
      pack->prefgh->opt.gauge_driver_enabled && !perturbed_trumpet;
  const int stencil_radius = pack->prefgh->opt.fd_order/2;
  const int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  Real field_linf = 0.0;
  Real constraint_linf = 0.0;
  Real field_fixed_linf[kFixedRadiusCount] = {};
  Real gauge_fixed_linf[kFixedRadiusCount] = {};
  Real constraint_fixed_linf[kFixedRadiusCount] = {};
  Kokkos::parallel_reduce(
      "ref_gh stationary trumpet error", Kokkos::RangePolicy<>(DevExeSpace(),
      0, pack->nmb_thispack*ncells),
      KOKKOS_LAMBDA(const int idx, Real &maximum) {
        int work = idx;
        const int i = work % indcs.nx1 + indcs.is; work /= indcs.nx1;
        const int j = work % indcs.nx2 + indcs.js; work /= indcs.nx2;
        const int k = work % indcs.nx3 + indcs.ks;
        const int m = work/indcs.nx3;
        const Real displacement[3] = {
          CellCenterX(i - indcs.is, indcs.nx1, size.d_view(m).x1min,
                      size.d_view(m).x1max) - center_x,
          CellCenterX(j - indcs.js, indcs.nx2, size.d_view(m).x2min,
                      size.d_view(m).x2max) - center_y,
          CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(m).x3min,
                      size.d_view(m).x3max) - center_z};
        const Real spacing[3] = {
          size.d_view(m).dx1, size.d_view(m).dx2, size.d_view(m).dx3};
        if (!ref_gh::PunctureStencilIsClear(
                displacement, spacing, stencil_radius)) return;
        for (int n = 0; n < ref_gh::kHhatOffset; ++n) {
          Real expected = 0.0;
          if (n == ref_gh::PsiIndex(0, 0)) expected = -1.0;
          if (n == ref_gh::PsiIndex(1, 1) || n == ref_gh::PsiIndex(2, 2)
              || n == ref_gh::PsiIndex(3, 3)) expected = 1.0;
          maximum = fmax(maximum, Kokkos::abs(state(m, n, k, j, i) - expected));
        }
      }, Kokkos::Max<Real>(field_linf));
  Kokkos::parallel_reduce(
      "ref_gh stationary trumpet constraints", Kokkos::RangePolicy<>(DevExeSpace(),
      0, pack->nmb_thispack*ncells),
      KOKKOS_LAMBDA(const int idx, Real &maximum) {
        int work = idx;
        const int i = work % indcs.nx1 + indcs.is; work /= indcs.nx1;
        const int j = work % indcs.nx2 + indcs.js; work /= indcs.nx2;
        const int k = work % indcs.nx3 + indcs.ks;
        const int m = work/indcs.nx3;
        const Real displacement[3] = {
          CellCenterX(i - indcs.is, indcs.nx1, size.d_view(m).x1min,
                      size.d_view(m).x1max) - center_x,
          CellCenterX(j - indcs.js, indcs.nx2, size.d_view(m).x2min,
                      size.d_view(m).x2max) - center_y,
          CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(m).x3min,
                      size.d_view(m).x3max) - center_z};
        const Real spacing[3] = {
          size.d_view(m).dx1, size.d_view(m).dx2, size.d_view(m).dx3};
        if (!ref_gh::PunctureStencilIsClear(
                displacement, spacing, stencil_radius)) return;
        for (int n = 0; n < ref_gh::RefGh::kNativeConstraints; ++n) {
          maximum = fmax(maximum, Kokkos::abs(constraints(m, n, k, j, i)));
        }
      }, Kokkos::Max<Real>(constraint_linf));
  for (int region = 0; region < kFixedRadiusCount; ++region) {
    const Real minimum_radius = kFixedMinimumRadius[region];
    const Real maximum_radius = kFixedMaximumRadius[region];
    Kokkos::parallel_reduce(
        "ref_gh stationary trumpet fixed-radius field error",
        Kokkos::RangePolicy<>(DevExeSpace(),
        0, pack->nmb_thispack*ncells),
        KOKKOS_LAMBDA(const int idx, Real &maximum, Real &gauge_maximum) {
          int work = idx;
          const int i = work % indcs.nx1 + indcs.is; work /= indcs.nx1;
          const int j = work % indcs.nx2 + indcs.js; work /= indcs.nx2;
          const int k = work % indcs.nx3 + indcs.ks;
          const int m = work/indcs.nx3;
          const Real displacement[3] = {
            CellCenterX(i - indcs.is, indcs.nx1, size.d_view(m).x1min,
                        size.d_view(m).x1max) - center_x,
            CellCenterX(j - indcs.js, indcs.nx2, size.d_view(m).x2min,
                        size.d_view(m).x2max) - center_y,
            CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(m).x3min,
                        size.d_view(m).x3max) - center_z};
          const Real radius = Kokkos::sqrt(
              displacement[0]*displacement[0]
              + displacement[1]*displacement[1]
              + displacement[2]*displacement[2]);
          const Real spacing[3] = {
            size.d_view(m).dx1, size.d_view(m).dx2, size.d_view(m).dx3};
          if (radius < minimum_radius || radius >= maximum_radius
              || !ref_gh::PunctureStencilIsClear(
                  displacement, spacing, stencil_radius)) return;
          for (int n = 0; n < ref_gh::kHhatOffset; ++n) {
            Real expected = 0.0;
            if (n == ref_gh::PsiIndex(0, 0)) expected = -1.0;
            if (n == ref_gh::PsiIndex(1, 1) || n == ref_gh::PsiIndex(2, 2)
                || n == ref_gh::PsiIndex(3, 3)) expected = 1.0;
            maximum = fmax(
                maximum, Kokkos::abs(state(m, n, k, j, i) - expected));
          }
          if (compare_stationary_gauge) {
            const ref_gh::StationaryGaugeState expected =
                ref_gh::ComputeStationaryTrumpetGaugeState(
                    reference_table, reference_mass, center_x, center_y,
                    center_z, displacement[0] + center_x,
                    displacement[1] + center_y, displacement[2] + center_z);
            if (!expected.valid) {
              gauge_maximum = std::numeric_limits<Real>::infinity();
              return;
            }
            for (int A = 0; A < 4; ++A) {
              gauge_maximum = fmax(
                  gauge_maximum,
                  Kokkos::abs(state(m, ref_gh::kHhatOffset + A, k, j, i)
                              - expected.hhat[A]));
              gauge_maximum = fmax(
                  gauge_maximum,
                  Kokkos::abs(state(m, ref_gh::kThetaOffset + A, k, j, i)
                              - expected.theta[A]));
            }
            for (int p = 0; p < 3; ++p) {
              gauge_maximum = fmax(
                  gauge_maximum,
                  Kokkos::abs(state(m, ref_gh::kUpsilonOffset + p, k, j, i)));
            }
          }
        }, Kokkos::Max<Real>(field_fixed_linf[region]),
           Kokkos::Max<Real>(gauge_fixed_linf[region]));
    Kokkos::parallel_reduce(
        "ref_gh stationary trumpet fixed-radius constraints",
        Kokkos::RangePolicy<>(DevExeSpace(),
        0, pack->nmb_thispack*ncells),
        KOKKOS_LAMBDA(const int idx, Real &maximum) {
          int work = idx;
          const int i = work % indcs.nx1 + indcs.is; work /= indcs.nx1;
          const int j = work % indcs.nx2 + indcs.js; work /= indcs.nx2;
          const int k = work % indcs.nx3 + indcs.ks;
          const int m = work/indcs.nx3;
          const Real displacement[3] = {
            CellCenterX(i - indcs.is, indcs.nx1, size.d_view(m).x1min,
                        size.d_view(m).x1max) - center_x,
            CellCenterX(j - indcs.js, indcs.nx2, size.d_view(m).x2min,
                        size.d_view(m).x2max) - center_y,
            CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(m).x3min,
                        size.d_view(m).x3max) - center_z};
          const Real radius = Kokkos::sqrt(
              displacement[0]*displacement[0]
              + displacement[1]*displacement[1]
              + displacement[2]*displacement[2]);
          const Real spacing[3] = {
            size.d_view(m).dx1, size.d_view(m).dx2, size.d_view(m).dx3};
          if (radius < minimum_radius || radius >= maximum_radius
              || !ref_gh::PunctureStencilIsClear(
                  displacement, spacing, stencil_radius)) return;
          for (int n = 0; n < ref_gh::RefGh::kNativeConstraints; ++n) {
            maximum = fmax(
                maximum, Kokkos::abs(constraints(m, n, k, j, i)));
          }
        }, Kokkos::Max<Real>(constraint_fixed_linf[region]));
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &field_linf, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &constraint_linf, 1, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, field_fixed_linf, kFixedRadiusCount,
                MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, gauge_fixed_linf, kFixedRadiusCount,
                MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, constraint_fixed_linf, kFixedRadiusCount,
                MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  if (global_variable::my_rank == 0) {
    const std::string suffix =
        perturbed_trumpet ? "-perturbed-trumpet.dat" : "-trumpet.dat";
    const std::string filename = pin->GetString("job", "basename") + suffix;
    FILE *file = std::fopen(filename.c_str(), "w");
    if (file == nullptr) std::exit(EXIT_FAILURE);
    std::fprintf(file, "# nx1 cycles time field_Linf constraint_Linf "
                       "rhs_estimate coordinate_reference_Ricci_Linf "
                       "frame_reference_Ricci_Linf spin_antisymmetry_Linf "
                       "structure_antisymmetry_Linf rhs_component rhs_radius "
                       "rhs_r0p5to1 rhs_r1to1p5 rhs_r1p5to2 "
                       "field_r0p5to1 field_r1to1p5 field_r1p5to2 "
                       "gauge_r0p5to1 gauge_r1to1p5 gauge_r1p5to2 "
                       "constraint_r0p5to1 constraint_r1to1p5 "
                       "constraint_r1p5to2\n");
    const Real rhs_estimate = initial_rhs_linf;
    std::fprintf(file, "%d %d %.17e %.17e %.17e %.17e %.17e %.17e %.17e "
                       "%.17e %d %.17e %.17e %.17e %.17e %.17e %.17e %.17e "
                       "%.17e %.17e %.17e %.17e %.17e %.17e\n",
                 mesh->mesh_indcs.nx1, mesh->ncycle, mesh->time, field_linf,
                 constraint_linf, rhs_estimate, initial_reference_ricci_linf,
                 initial_frame_ricci_linf, initial_spin_antisymmetry_linf,
                 initial_structure_antisymmetry_linf,
                 initial_rhs_component, initial_rhs_radius,
                 initial_rhs_fixed_linf[0], initial_rhs_fixed_linf[1],
                 initial_rhs_fixed_linf[2], field_fixed_linf[0],
                 field_fixed_linf[1], field_fixed_linf[2],
                 gauge_fixed_linf[0], gauge_fixed_linf[1],
                 gauge_fixed_linf[2],
                 constraint_fixed_linf[0], constraint_fixed_linf[1],
                 constraint_fixed_linf[2]);
    std::fclose(file);
    std::cout << "reference-GH "
              << (perturbed_trumpet ? "perturbed" : "stationary")
              << " trumpet: field Linf=" << field_linf
              << ", constraint Linf=" << constraint_linf
              << ", RHS estimate=" << rhs_estimate << std::endl;
  }
}

}  // namespace

void ProblemGenerator::RefGhStationaryTrumpet(ParameterInput *pin, const bool restart) {
  perturbed_trumpet =
      pin->GetString("problem", "pgen_name") == "ref_gh_perturbed_trumpet";
  perturbation_amplitude = perturbed_trumpet
      ? pin->GetOrAddReal("problem", "perturb_amplitude", 1.0e-6) : 0.0;
  perturbation_width = perturbed_trumpet
      ? pin->GetOrAddReal("problem", "perturb_width", 0.5) : 0.0;
  if (perturbed_trumpet
      && (!(perturbation_amplitude > 0.0) || !(perturbation_width > 0.0))) {
    std::cout << "### FATAL ERROR: perturbed trumpet requires positive amplitude "
                 "and width." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  pgen_final_func = &CheckRefGhStationaryTrumpet;
  if (restart) return;
  auto *pack = pmy_mesh_->pmb_pack;
  if (pack->prefgh == nullptr || pack->prefgh->opt.reference_kind != 1) {
    std::cout << "stationary trumpet data require ref_gh/reference=trumpet."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto &indcs = pack->pmesh->mb_indcs;
  auto &size = pack->pmb->mb_size;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const int n2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*indcs.ng : 1;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*indcs.ng : 1;
  const auto state = pack->prefgh->u0;
  const Real cx = pack->prefgh->opt.reference_center[0];
  const Real cy = pack->prefgh->opt.reference_center[1];
  const Real cz = pack->prefgh->opt.reference_center[2];
  const Real mass = pack->prefgh->opt.reference_mass;
  const auto table = pack->prefgh->reference_table;
  const Real amplitude = perturbation_amplitude;
  const Real width = perturbation_width;
  Real minimum_radius = std::numeric_limits<Real>::max();
  Kokkos::parallel_reduce(
      "ref_gh minimum puncture radius", Kokkos::RangePolicy<>(DevExeSpace(),
      0, pack->nmb_thispack*indcs.nx1*indcs.nx2*indcs.nx3),
      KOKKOS_LAMBDA(const int idx, Real &minimum) {
        int work = idx;
        const int i = work % indcs.nx1; work /= indcs.nx1;
        const int j = work % indcs.nx2; work /= indcs.nx2;
        const int k = work % indcs.nx3;
        const int m = work/indcs.nx3;
        const Real x = CellCenterX(i, indcs.nx1, size.d_view(m).x1min,
                                   size.d_view(m).x1max);
        const Real y = CellCenterX(j, indcs.nx2, size.d_view(m).x2min,
                                   size.d_view(m).x2max);
        const Real z = CellCenterX(k, indcs.nx3, size.d_view(m).x3min,
                                   size.d_view(m).x3max);
        const Real radius = Kokkos::sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy)
                                         + (z-cz)*(z-cz));
        if (radius < minimum) minimum = radius;
      }, Kokkos::Min<Real>(minimum_radius));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &minimum_radius, 1, MPI_ATHENA_REAL, MPI_MIN,
                MPI_COMM_WORLD);
#endif
  if (!(minimum_radius > 0.0)) {
    std::cout << "### FATAL ERROR: the reference puncture lies on a cell center."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (global_variable::my_rank == 0) {
    std::cout << "reference-GH puncture minimum cell-center radius = "
              << minimum_radius << std::endl;
  }
  par_for("ref_gh stationary trumpet data", DevExeSpace(), 0,
  pack->nmb_thispack - 1, 0, n3 - 1, 0, n2 - 1, 0, n1 - 1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    for (int n = 0; n < ref_gh::nvar; ++n) state(m, n, k, j, i) = 0.0;
    state(m, ref_gh::PsiIndex(0, 0), k, j, i) = -1.0;
    state(m, ref_gh::PsiIndex(1, 1), k, j, i) = 1.0;
    state(m, ref_gh::PsiIndex(2, 2), k, j, i) = 1.0;
    state(m, ref_gh::PsiIndex(3, 3), k, j, i) = 1.0;
    if (amplitude > 0.0) {
      const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                                 size.d_view(m).x1min, size.d_view(m).x1max);
      const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                                 size.d_view(m).x2min, size.d_view(m).x2max);
      const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                                 size.d_view(m).x3min, size.d_view(m).x3max);
      const Real displacement[3] = {x - cx, y - cy, z - cz};
      const Real radius2 = displacement[0]*displacement[0]
                           + displacement[1]*displacement[1]
                           + displacement[2]*displacement[2];
      const Real width2 = width*width;
      const Real bump = amplitude*Kokkos::exp(-radius2/width2);
      state(m, ref_gh::PsiIndex(2, 2), k, j, i) += bump;
      state(m, ref_gh::PsiIndex(3, 3), k, j, i) -= bump;
      ref_gh::ReferencePsiKinematics reference;
      ref_gh::GetReferencePsiKinematics(
          1, table, mass, cx, cy, cz, 0.0, x, y, z, reference);
      for (int I = 0; I < 3; ++I) {
        const Real gradient = -2.0*bump*displacement[I]/width2;
        const Real frame_gradient =
            gradient/reference.spatial_coframe[I][I];
        state(m, ref_gh::PhiIndex(I, 2, 2), k, j, i) = frame_gradient;
        state(m, ref_gh::PhiIndex(I, 3, 3), k, j, i) = -frame_gradient;
      }
    }
  });
  if (pack->prefgh->opt.gauge_driver_enabled) {
    const Real shift_nu = pack->prefgh->opt.shift_nu;
    const Real shift_eta = pack->prefgh->opt.shift_eta;
    const bool is_perturbed = perturbed_trumpet;
    par_for("ref_gh stationary trumpet gauge data", DevExeSpace(), 0,
    pack->nmb_thispack - 1, 0, n3 - 1, 0, n2 - 1, 0, n1 - 1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                                 size.d_view(m).x1min, size.d_view(m).x1max);
      const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                                 size.d_view(m).x2min, size.d_view(m).x2max);
      const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                                 size.d_view(m).x3min, size.d_view(m).x3max);
      ref_gh::ReferenceGeometry reference;
      ref_gh::GetReferenceGeometry(
          1, table, mass, cx, cy, cz, 0.0, x, y, z, reference);
      Real psi[4][4], metric[4][4], pi[4][4], phi[3][4][4]; // NOLINT
      Real d_psi[4][4][4], d_metric[4][4][4]; // NOLINT
      ref_gh::CoordinateGhGeometry geometry;
      Real determinant = 0.0;
      if (!ref_gh::LoadPointGeometry(
              state, reference, m, k, j, i, psi, pi, phi, d_psi, metric,
              d_metric, geometry, determinant)) {
        for (int n = ref_gh::kHhatOffset; n < ref_gh::nvar; ++n) {
          state(m, n, k, j, i) = NAN;
        }
        return;
      }
      const Real upsilon[3] = {0.0, 0.0, 0.0};
      ref_gh::PhysicalGaugeTarget target;
      if (!ref_gh::ComputePhysicalGaugeTarget(
              metric, d_metric, geometry, reference, upsilon, shift_nu,
              shift_eta, target)) {
        for (int n = ref_gh::kHhatOffset; n < ref_gh::nvar; ++n) {
          state(m, n, k, j, i) = NAN;
        }
        return;
      }
      for (int A = 0; A < 4; ++A) {
        state(m, ref_gh::kHhatOffset + A, k, j, i) = target.frame[A];
      }
    });
    if (!is_perturbed) {
      const int fd_order = pack->prefgh->opt.fd_order;
      const int radius = fd_order/2;
      par_for("ref_gh stationary trumpet theta data", DevExeSpace(), 0,
      pack->nmb_thispack - 1, indcs.ks - radius, indcs.ke + radius,
      indcs.js - radius, indcs.je + radius,
      indcs.is - radius, indcs.ie + radius,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                                   size.d_view(m).x1min, size.d_view(m).x1max);
        const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                                   size.d_view(m).x2min, size.d_view(m).x2max);
        const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                                   size.d_view(m).x3min, size.d_view(m).x3max);
        ref_gh::ReferenceGeometry reference;
        ref_gh::GetReferenceGeometry(
            1, table, mass, cx, cy, cz, 0.0, x, y, z, reference);
        const Real idx[3] = {1.0/size.d_view(m).dx1, 1.0/size.d_view(m).dx2,
                             1.0/size.d_view(m).dx3};
        Real psi[4][4], metric[4][4], inverse[4][4]; // NOLINT
        ref_gh::LoadSymmetric(state, ref_gh::kPsiOffset, m, k, j, i, psi);
        for (int a = 0; a < 4; ++a) {
          for (int b = 0; b < 4; ++b) {
            metric[a][b] = 0.0;
            for (int A = 0; A < 4; ++A) {
              for (int B = 0; B < 4; ++B) {
                metric[a][b] += reference.coframe[A][a]
                                *reference.coframe[B][b]*psi[A][B];
              }
            }
          }
        }
        Real determinant = 0.0;
        if (!ref_gh::Invert4(metric, inverse, determinant)
            || !(inverse[0][0] < 0.0)) {
          for (int A = 0; A < 4; ++A) {
            state(m, ref_gh::kThetaOffset + A, k, j, i) = NAN;
          }
          return;
        }
        const Real lapse = 1.0/Kokkos::sqrt(-inverse[0][0]);
        Real shift[3];  // NOLINT(runtime/arrays)
        for (int p = 0; p < 3; ++p) {
          shift[p] = lapse*lapse*inverse[0][p + 1];
        }
        for (int A = 0; A < 4; ++A) {
          Real theta_frame = 0.0;
          for (int p = 0; p < 3; ++p) {
            Real derivative = 0.0;
            if (fd_order == 2) {
              derivative = Dx<2>(p, idx, state, m, ref_gh::kHhatOffset + A,
                                 k, j, i);
            } else if (fd_order == 4) {
              derivative = Dx<3>(p, idx, state, m, ref_gh::kHhatOffset + A,
                                 k, j, i);
            } else {
              derivative = Dx<4>(p, idx, state, m, ref_gh::kHhatOffset + A,
                                 k, j, i);
            }
            theta_frame -= shift[p]*derivative;
          }
          for (int B = 0; B < 4; ++B) {
            Real d0_frame = ref_gh::ReferenceFrameMotion(reference, A, 0, B);
            for (int p = 0; p < 3; ++p) {
              d0_frame -= shift[p]
                  *ref_gh::ReferenceFrameMotion(reference, A, p + 1, B);
            }
            theta_frame -= d0_frame
                *state(m, ref_gh::kHhatOffset + B, k, j, i);
          }
          state(m, ref_gh::kThetaOffset + A, k, j, i) = theta_frame;
        }
      });
    }
    Real target_baseline_linf = 0.0;
    Real conformal_gamma_linf = 0.0;
    const int active_cells = indcs.nx1*indcs.nx2*indcs.nx3;
    Kokkos::parallel_reduce(
        "ref_gh stationary gauge initialization audit",
        Kokkos::RangePolicy<>(DevExeSpace(),
            0, pack->nmb_thispack*active_cells),
        KOKKOS_LAMBDA(const int idx, Real &local_target,
                      Real &local_conformal_gamma) {
          int work = idx;
          const int i = work % indcs.nx1 + indcs.is; work /= indcs.nx1;
          const int j = work % indcs.nx2 + indcs.js; work /= indcs.nx2;
          const int k = work % indcs.nx3 + indcs.ks;
          const int m = work/indcs.nx3;
          const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                                     size.d_view(m).x1min, size.d_view(m).x1max);
          const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                                     size.d_view(m).x2min, size.d_view(m).x2max);
          const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                                     size.d_view(m).x3min, size.d_view(m).x3max);
          ref_gh::ReferenceGeometry reference;
          ref_gh::GetReferenceGeometry(
              1, table, mass, cx, cy, cz, 0.0, x, y, z, reference);
          ref_gh::CoordinateGhGeometry geometry;
          Real determinant = 0.0;
          if (!ref_gh::ComputeCoordinateGhGeometry(
                  reference.metric, reference.d_metric, reference, geometry,
                  determinant)) {
            local_target = fmax(local_target, 1.0);
            return;
          }
          const Real upsilon[3] = {0.0, 0.0, 0.0};
          ref_gh::PhysicalGaugeTarget target;
          if (!ref_gh::ComputePhysicalGaugeTarget(
                  reference.metric, reference.d_metric, geometry, reference,
                  upsilon, shift_nu, shift_eta, target)) {
            local_target = fmax(local_target, 1.0);
            return;
          }
          for (int A = 0; A < 4; ++A) {
            Real baseline = 0.0;
            for (int a = 0; a < 4; ++a) {
              baseline += reference.frame[A][a]*geometry.gauge_source[a];
            }
            local_target = fmax(
                local_target, Kokkos::abs(target.frame[A] - baseline));
          }
          for (int p = 0; p < 3; ++p) {
            local_conformal_gamma = fmax(
                local_conformal_gamma, Kokkos::abs(target.conformal_gamma[p]));
          }
        }, Kokkos::Max<Real>(target_baseline_linf),
           Kokkos::Max<Real>(conformal_gamma_linf));
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &target_baseline_linf, 1, MPI_ATHENA_REAL,
                  MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &conformal_gamma_linf, 1, MPI_ATHENA_REAL,
                  MPI_MAX, MPI_COMM_WORLD);
#endif
    if (global_variable::my_rank == 0) {
      std::cout << "reference-GH stationary physical-target audit: "
                << "|F-H_constraint|_Linf=" << target_baseline_linf
                << ", |tildeGamma|_Linf=" << conformal_gamma_linf
                << std::endl;
    }
  }
  switch (pack->prefgh->opt.fd_order) {
    case 2: (void)pack->prefgh->CalcRHS<2>(nullptr, 1); break;
    case 4: (void)pack->prefgh->CalcRHS<3>(nullptr, 1); break;
    case 6: (void)pack->prefgh->CalcRHS<4>(nullptr, 1); break;
  }
  if (pin->GetOrAddBoolean("problem", "debug_repeat_initial_rhs", false)) {
    switch (pack->prefgh->opt.fd_order) {
      case 2: (void)pack->prefgh->CalcRHS<2>(nullptr, 1); break;
      case 4: (void)pack->prefgh->CalcRHS<3>(nullptr, 1); break;
      case 6: (void)pack->prefgh->CalcRHS<4>(nullptr, 1); break;
    }
    Kokkos::fence("ref_gh repeated initial RHS");
    if (global_variable::my_rank == 0) {
      std::cout << "reference-GH repeated initial RHS completed" << std::endl;
    }
  }
  const auto rhs = pack->prefgh->u_rhs;
  const int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  const int rhs_stencil_radius = pack->prefgh->opt.fd_order/2;
  using MaxLoc = Kokkos::MaxLoc<Real, int>;
  MaxLoc::value_type rhs_maximum;
  Kokkos::parallel_reduce(
      perturbed_trumpet ? "ref_gh perturbed initial RHS"
                        : "ref_gh stationary initial RHS",
      Kokkos::RangePolicy<>(DevExeSpace(),
      0, pack->nmb_thispack*ref_gh::nvar*ncells),
      KOKKOS_LAMBDA(const int idx, MaxLoc::value_type &maximum) {
        int work = idx;
        const int i = work % indcs.nx1 + indcs.is; work /= indcs.nx1;
        const int j = work % indcs.nx2 + indcs.js; work /= indcs.nx2;
        const int k = work % indcs.nx3 + indcs.ks; work /= indcs.nx3;
        const int n = work % ref_gh::nvar;
        const int m = work/ref_gh::nvar;
        const Real displacement[3] = {
          CellCenterX(i - indcs.is, indcs.nx1, size.d_view(m).x1min,
                      size.d_view(m).x1max) - cx,
          CellCenterX(j - indcs.js, indcs.nx2, size.d_view(m).x2min,
                      size.d_view(m).x2max) - cy,
          CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(m).x3min,
                      size.d_view(m).x3max) - cz};
        const Real spacing[3] = {
          size.d_view(m).dx1, size.d_view(m).dx2, size.d_view(m).dx3};
        if (!ref_gh::PunctureStencilIsClear(
                displacement, spacing, rhs_stencil_radius)) return;
        const Real value = Kokkos::abs(rhs(m, n, k, j, i));
        if (value > maximum.val) {
          maximum.val = value;
          maximum.loc = idx;
        }
      }, MaxLoc(rhs_maximum));
  initial_rhs_linf = rhs_maximum.val;
  int rhs_work = rhs_maximum.loc;
  const int rhs_i = rhs_work % indcs.nx1 + indcs.is; rhs_work /= indcs.nx1;
  const int rhs_j = rhs_work % indcs.nx2 + indcs.js; rhs_work /= indcs.nx2;
  const int rhs_k = rhs_work % indcs.nx3 + indcs.ks; rhs_work /= indcs.nx3;
  initial_rhs_component = rhs_work % ref_gh::nvar;
  const int rhs_m = rhs_work/ref_gh::nvar;
  const auto &rhs_block = size.h_view(rhs_m);
  const Real rhs_x = CellCenterX(rhs_i - indcs.is, indcs.nx1,
                                 rhs_block.x1min, rhs_block.x1max);
  const Real rhs_y = CellCenterX(rhs_j - indcs.js, indcs.nx2,
                                 rhs_block.x2min, rhs_block.x2max);
  const Real rhs_z = CellCenterX(rhs_k - indcs.ks, indcs.nx3,
                                 rhs_block.x3min, rhs_block.x3max);
  initial_rhs_radius = std::sqrt((rhs_x-cx)*(rhs_x-cx) + (rhs_y-cy)*(rhs_y-cy)
                                 + (rhs_z-cz)*(rhs_z-cz));

  for (int region = 0; region < kFixedRadiusCount; ++region) {
    const Real minimum_radius = kFixedMinimumRadius[region];
    const Real maximum_radius = kFixedMaximumRadius[region];
    Kokkos::parallel_reduce(
        "ref_gh stationary fixed-radius initial RHS",
        Kokkos::RangePolicy<>(DevExeSpace(),
        0, pack->nmb_thispack*ref_gh::nvar*ncells),
        KOKKOS_LAMBDA(const int idx, Real &maximum) {
          int work = idx;
          const int i = work % indcs.nx1 + indcs.is; work /= indcs.nx1;
          const int j = work % indcs.nx2 + indcs.js; work /= indcs.nx2;
          const int k = work % indcs.nx3 + indcs.ks; work /= indcs.nx3;
          const int n = work % ref_gh::nvar;
          const int m = work/ref_gh::nvar;
          const Real displacement[3] = {
            CellCenterX(i - indcs.is, indcs.nx1, size.d_view(m).x1min,
                        size.d_view(m).x1max) - cx,
            CellCenterX(j - indcs.js, indcs.nx2, size.d_view(m).x2min,
                        size.d_view(m).x2max) - cy,
            CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(m).x3min,
                        size.d_view(m).x3max) - cz};
          const Real radius = Kokkos::sqrt(
              displacement[0]*displacement[0]
              + displacement[1]*displacement[1]
              + displacement[2]*displacement[2]);
          const Real spacing[3] = {
            size.d_view(m).dx1, size.d_view(m).dx2, size.d_view(m).dx3};
          if (radius < minimum_radius || radius >= maximum_radius
              || !ref_gh::PunctureStencilIsClear(
                  displacement, spacing, rhs_stencil_radius)) return;
          maximum = fmax(maximum, Kokkos::abs(rhs(m, n, k, j, i)));
        }, Kokkos::Max<Real>(initial_rhs_fixed_linf[region]));
  }

  Kokkos::parallel_reduce(
      "ref_gh stationary reference Ricci", Kokkos::RangePolicy<>(DevExeSpace(),
      0, pack->nmb_thispack*ncells),
      KOKKOS_LAMBDA(const int idx, Real &maximum) {
        int work = idx;
        const int i = work % indcs.nx1 + indcs.is; work /= indcs.nx1;
        const int j = work % indcs.nx2 + indcs.js; work /= indcs.nx2;
        const int k = work % indcs.nx3 + indcs.ks;
        const int m = work/indcs.nx3;
        const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                                   size.d_view(m).x1min, size.d_view(m).x1max);
        const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                                   size.d_view(m).x2min, size.d_view(m).x2max);
        const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                                   size.d_view(m).x3min, size.d_view(m).x3max);
        ref_gh::ReferenceGeometry reference;
        ref_gh::GetReferenceGeometry(1, table, mass, cx, cy, cz, 0.0, x, y, z,
                                     reference);
        for (int a = 0; a < 4; ++a) {
          for (int b = 0; b < 4; ++b) {
            Real ricci = 0.0;
            for (int c = 0; c < 4; ++c) {
              ricci += reference.d_christoffel[c][c][a][b]
                       - reference.d_christoffel[b][c][a][c];
              for (int d = 0; d < 4; ++d) {
                ricci += reference.christoffel[c][c][d]
                           *reference.christoffel[d][a][b]
                         - reference.christoffel[c][b][d]
                           *reference.christoffel[d][a][c];
              }
            }
            maximum = fmax(maximum, Kokkos::abs(ricci));
          }
        }
      }, Kokkos::Max<Real>(initial_reference_ricci_linf));
  Kokkos::parallel_reduce(
      "ref_gh stationary frame reference audits", Kokkos::RangePolicy<>(DevExeSpace(),
      0, pack->nmb_thispack*ncells),
      KOKKOS_LAMBDA(const int idx, Real &maximum) {
        int work = idx;
        const int i = work % indcs.nx1 + indcs.is; work /= indcs.nx1;
        const int j = work % indcs.nx2 + indcs.js; work /= indcs.nx2;
        const int k = work % indcs.nx3 + indcs.ks;
        const int m = work/indcs.nx3;
        const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                                   size.d_view(m).x1min, size.d_view(m).x1max);
        const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                                   size.d_view(m).x2min, size.d_view(m).x2max);
        const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                                   size.d_view(m).x3min, size.d_view(m).x3max);
        ref_gh::ReferenceGeometry reference;
        ref_gh::GetReferenceGeometry(1, table, mass, cx, cy, cz, 0.0, x, y, z,
                                     reference);
        for (int A = 0; A < 4; ++A) {
          for (int B = 0; B < 4; ++B) {
            maximum = fmax(maximum, Kokkos::abs(reference.ricci_frame[A][B]));
          }
        }
      }, Kokkos::Max<Real>(initial_frame_ricci_linf));
  const bool full_reference_audit =
      pin->GetOrAddBoolean("problem", "full_reference_audit", false);
  if (full_reference_audit) {
  Kokkos::parallel_reduce(
      "ref_gh stationary spin antisymmetry", Kokkos::RangePolicy<>(DevExeSpace(),
      0, pack->nmb_thispack*ncells),
      KOKKOS_LAMBDA(const int idx, Real &maximum) {
        int work = idx;
        const int i = work % indcs.nx1 + indcs.is; work /= indcs.nx1;
        const int j = work % indcs.nx2 + indcs.js; work /= indcs.nx2;
        const int k = work % indcs.nx3 + indcs.ks;
        const int m = work/indcs.nx3;
        ref_gh::ReferenceGeometry reference;
        ref_gh::GetReferenceGeometry(
            1, table, mass, cx, cy, cz, 0.0,
            CellCenterX(i - indcs.is, indcs.nx1, size.d_view(m).x1min,
                        size.d_view(m).x1max),
            CellCenterX(j - indcs.js, indcs.nx2, size.d_view(m).x2min,
                        size.d_view(m).x2max),
            CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(m).x3min,
                        size.d_view(m).x3max), reference);
        for (int A = 0; A < 4; ++A) {
          const Real eta_A = (A == 0) ? -1.0 : 1.0;
          for (int B = 0; B < 4; ++B) {
            const Real eta_B = (B == 0) ? -1.0 : 1.0;
            for (int C = 0; C < 4; ++C) {
              maximum = fmax(maximum, Kokkos::abs(
                  eta_A*reference.spin[A][B][C]
                  + eta_B*reference.spin[B][A][C]));
            }
          }
        }
      }, Kokkos::Max<Real>(initial_spin_antisymmetry_linf));
  Kokkos::parallel_reduce(
      "ref_gh stationary structure antisymmetry", Kokkos::RangePolicy<>(DevExeSpace(),
      0, pack->nmb_thispack*ncells),
      KOKKOS_LAMBDA(const int idx, Real &maximum) {
        int work = idx;
        const int i = work % indcs.nx1 + indcs.is; work /= indcs.nx1;
        const int j = work % indcs.nx2 + indcs.js; work /= indcs.nx2;
        const int k = work % indcs.nx3 + indcs.ks;
        const int m = work/indcs.nx3;
        ref_gh::ReferenceGeometry reference;
        ref_gh::GetReferenceGeometry(
            1, table, mass, cx, cy, cz, 0.0,
            CellCenterX(i - indcs.is, indcs.nx1, size.d_view(m).x1min,
                        size.d_view(m).x1max),
            CellCenterX(j - indcs.js, indcs.nx2, size.d_view(m).x2min,
                        size.d_view(m).x2max),
            CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(m).x3min,
                        size.d_view(m).x3max), reference);
        for (int A = 0; A < 4; ++A) {
          for (int B = 0; B < 4; ++B) {
            for (int C = 0; C < 4; ++C) {
              maximum = fmax(maximum, Kokkos::abs(
                  reference.structure4[A][B][C]
                  + reference.structure4[A][C][B]));
            }
          }
        }
      }, Kokkos::Max<Real>(initial_structure_antisymmetry_linf));
  } else {
    initial_spin_antisymmetry_linf = NAN;
    initial_structure_antisymmetry_linf = NAN;
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &initial_rhs_linf, 1, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, initial_rhs_fixed_linf, kFixedRadiusCount,
                MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &initial_reference_ricci_linf, 1, MPI_ATHENA_REAL,
                MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &initial_frame_ricci_linf, 1, MPI_ATHENA_REAL,
                MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &initial_spin_antisymmetry_linf, 1, MPI_ATHENA_REAL,
                MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &initial_structure_antisymmetry_linf, 1,
                MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  if (global_variable::my_rank == 0) {
    std::cout << "reference-GH "
              << (perturbed_trumpet ? "perturbed" : "stationary")
              << " initial RHS Linf = "
              << initial_rhs_linf << ", component=" << initial_rhs_component
              << ", radius=" << initial_rhs_radius
              << ", coordinate reference Ricci Linf="
              << initial_reference_ricci_linf
              << ", frame reference Ricci Linf=" << initial_frame_ricci_linf
              << std::endl;
  }
  const bool rhs_convergence_mode = pin->GetOrAddBoolean(
      "problem", "stationary_rhs_convergence_mode", false);
  if (!std::isfinite(initial_rhs_linf)
      || (!perturbed_trumpet && !rhs_convergence_mode
          && initial_rhs_linf > 1.0e-6)) {
    std::cout << "### FATAL ERROR: reference-GH initial RHS is nonfinite or "
                 "the stationary residual exceeds 1e-6." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!perturbed_trumpet && rhs_convergence_mode
      && initial_rhs_linf > 1.0e-6 && global_variable::my_rank == 0) {
    std::cout << "reference-GH stationary RHS retained as a convergence "
                 "observable; no single-resolution stationarity claim is made."
              << std::endl;
  }
}
