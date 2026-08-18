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
#include "ref_gh/ref_gh.hpp"
#include "ref_gh/ref_gh_geometry.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {
Real initial_rhs_linf = 0.0;
Real initial_reference_ricci_linf = 0.0;
Real initial_frame_ricci_linf = 0.0;
Real initial_spin_antisymmetry_linf = 0.0;
Real initial_structure_antisymmetry_linf = 0.0;
int initial_rhs_component = -1;
Real initial_rhs_radius = -1.0;

void CheckRefGhStationaryTrumpet(ParameterInput *pin, Mesh *mesh) {
  auto *pack = mesh->pmb_pack;
  switch (pack->prefgh->opt.fd_order) {
    case 2: pack->prefgh->CalcConstraints<2>(); break;
    case 4: pack->prefgh->CalcConstraints<3>(); break;
    case 6: pack->prefgh->CalcConstraints<4>(); break;
  }
  auto &indcs = mesh->mb_indcs;
  const auto state = pack->prefgh->u0;
  const auto constraints = pack->prefgh->u_con;
  const int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  Real field_linf = 0.0;
  Real constraint_linf = 0.0;
  Kokkos::parallel_reduce(
      "ref_gh stationary trumpet error", Kokkos::RangePolicy<>(DevExeSpace(),
      0, pack->nmb_thispack*ncells),
      KOKKOS_LAMBDA(const int idx, Real &maximum) {
        int work = idx;
        const int i = work % indcs.nx1 + indcs.is; work /= indcs.nx1;
        const int j = work % indcs.nx2 + indcs.js; work /= indcs.nx2;
        const int k = work % indcs.nx3 + indcs.ks;
        const int m = work/indcs.nx3;
        for (int n = 0; n < ref_gh::nvar; ++n) {
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
        for (int n = 0; n < ref_gh::RefGh::ncon; ++n) {
          maximum = fmax(maximum, Kokkos::abs(constraints(m, n, k, j, i)));
        }
      }, Kokkos::Max<Real>(constraint_linf));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &field_linf, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &constraint_linf, 1, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
#endif
  if (global_variable::my_rank == 0) {
    const std::string filename = pin->GetString("job", "basename") + "-trumpet.dat";
    FILE *file = std::fopen(filename.c_str(), "w");
    if (file == nullptr) std::exit(EXIT_FAILURE);
    std::fprintf(file, "# nx1 cycles time field_Linf constraint_Linf "
                       "rhs_estimate coordinate_reference_Ricci_Linf "
                       "frame_reference_Ricci_Linf spin_antisymmetry_Linf "
                       "structure_antisymmetry_Linf rhs_component rhs_radius\n");
    const Real rhs_estimate = initial_rhs_linf;
    std::fprintf(file, "%d %d %.17e %.17e %.17e %.17e %.17e %.17e %.17e "
                       "%.17e %d %.17e\n",
                 mesh->mesh_indcs.nx1, mesh->ncycle, mesh->time, field_linf,
                 constraint_linf, rhs_estimate, initial_reference_ricci_linf,
                 initial_frame_ricci_linf, initial_spin_antisymmetry_linf,
                 initial_structure_antisymmetry_linf,
                 initial_rhs_component, initial_rhs_radius);
    std::fclose(file);
    std::cout << "reference-GH stationary trumpet: field Linf=" << field_linf
              << ", constraint Linf=" << constraint_linf
              << ", RHS estimate=" << rhs_estimate << std::endl;
  }
}

}  // namespace

void ProblemGenerator::RefGhStationaryTrumpet(ParameterInput *, const bool restart) {
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
  const int n2 = indcs.nx2 + 2*indcs.ng;
  const int n3 = indcs.nx3 + 2*indcs.ng;
  const auto state = pack->prefgh->u0;
  const Real cx = pack->prefgh->opt.reference_center[0];
  const Real cy = pack->prefgh->opt.reference_center[1];
  const Real cz = pack->prefgh->opt.reference_center[2];
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
  });
  switch (pack->prefgh->opt.fd_order) {
    case 2: (void)pack->prefgh->CalcRHS<2>(nullptr, 1); break;
    case 4: (void)pack->prefgh->CalcRHS<3>(nullptr, 1); break;
    case 6: (void)pack->prefgh->CalcRHS<4>(nullptr, 1); break;
  }
  const auto rhs = pack->prefgh->u_rhs;
  const int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  using MaxLoc = Kokkos::MaxLoc<Real, int>;
  MaxLoc::value_type rhs_maximum;
  Kokkos::parallel_reduce(
      "ref_gh stationary initial RHS", Kokkos::RangePolicy<>(DevExeSpace(),
      0, pack->nmb_thispack*ref_gh::nvar*ncells),
      KOKKOS_LAMBDA(const int idx, MaxLoc::value_type &maximum) {
        int work = idx;
        const int i = work % indcs.nx1 + indcs.is; work /= indcs.nx1;
        const int j = work % indcs.nx2 + indcs.js; work /= indcs.nx2;
        const int k = work % indcs.nx3 + indcs.ks; work /= indcs.nx3;
        const int n = work % ref_gh::nvar;
        const int m = work/ref_gh::nvar;
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

  const auto table = pack->prefgh->reference_table;
  const Real mass = pack->prefgh->opt.reference_mass;
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
        const ref_gh::ReferenceGeometry reference = ref_gh::GetReferenceGeometry(
            1, table, mass, cx, cy, cz, 0.0, x, y, z);
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
        const ref_gh::ReferenceGeometry reference = ref_gh::GetReferenceGeometry(
            1, table, mass, cx, cy, cz, 0.0, x, y, z);
        for (int A = 0; A < 4; ++A) {
          for (int B = 0; B < 4; ++B) {
            maximum = fmax(maximum, Kokkos::abs(reference.ricci_frame[A][B]));
          }
        }
      }, Kokkos::Max<Real>(initial_frame_ricci_linf));
  Kokkos::parallel_reduce(
      "ref_gh stationary spin antisymmetry", Kokkos::RangePolicy<>(DevExeSpace(),
      0, pack->nmb_thispack*ncells),
      KOKKOS_LAMBDA(const int idx, Real &maximum) {
        int work = idx;
        const int i = work % indcs.nx1 + indcs.is; work /= indcs.nx1;
        const int j = work % indcs.nx2 + indcs.js; work /= indcs.nx2;
        const int k = work % indcs.nx3 + indcs.ks;
        const int m = work/indcs.nx3;
        const ref_gh::ReferenceGeometry reference = ref_gh::GetReferenceGeometry(
            1, table, mass, cx, cy, cz, 0.0,
            CellCenterX(i - indcs.is, indcs.nx1, size.d_view(m).x1min,
                        size.d_view(m).x1max),
            CellCenterX(j - indcs.js, indcs.nx2, size.d_view(m).x2min,
                        size.d_view(m).x2max),
            CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(m).x3min,
                        size.d_view(m).x3max));
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
        const ref_gh::ReferenceGeometry reference = ref_gh::GetReferenceGeometry(
            1, table, mass, cx, cy, cz, 0.0,
            CellCenterX(i - indcs.is, indcs.nx1, size.d_view(m).x1min,
                        size.d_view(m).x1max),
            CellCenterX(j - indcs.js, indcs.nx2, size.d_view(m).x2min,
                        size.d_view(m).x2max),
            CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(m).x3min,
                        size.d_view(m).x3max));
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
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &initial_rhs_linf, 1, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
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
    std::cout << "reference-GH stationary initial RHS Linf = "
              << initial_rhs_linf << ", component=" << initial_rhs_component
              << ", radius=" << initial_rhs_radius
              << ", coordinate reference Ricci Linf="
              << initial_reference_ricci_linf
              << ", frame reference Ricci Linf=" << initial_frame_ricci_linf
              << std::endl;
  }
  if (!std::isfinite(initial_rhs_linf) || initial_rhs_linf > 1.0e-6) {
    std::cout << "### FATAL ERROR: stationary reference RHS exceeds 1e-6."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}
