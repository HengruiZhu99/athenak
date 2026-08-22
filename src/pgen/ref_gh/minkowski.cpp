//========================================================================================
//! \file minkowski.cpp
//! \brief Exact Minkowski data for the 50-field reference-frame GH module.
//========================================================================================
#include <cstdlib>
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "ref_gh/ref_gh.hpp"

namespace {

void CheckRefGhMinkowski(ParameterInput *, Mesh *mesh) {
  auto *pack = mesh->pmb_pack;
  auto &indcs = mesh->mb_indcs;
  const auto state = pack->prefgh->u0;
  const int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  Real maximum = 0.0;
  Kokkos::parallel_reduce(
      "ref_gh Minkowski error", Kokkos::RangePolicy<>(DevExeSpace(),
      0, pack->nmb_thispack*ref_gh::nvar*ncells),
      KOKKOS_LAMBDA(const int idx, Real &local_maximum) {
        int work = idx;
        const int i = work % indcs.nx1 + indcs.is; work /= indcs.nx1;
        const int j = work % indcs.nx2 + indcs.js; work /= indcs.nx2;
        const int k = work % indcs.nx3 + indcs.ks; work /= indcs.nx3;
        const int n = work % ref_gh::nvar;
        const int m = work/ref_gh::nvar;
        Real expected = 0.0;
        if (n == ref_gh::PsiIndex(0, 0)) expected = -1.0;
        if (n == ref_gh::PsiIndex(1, 1) || n == ref_gh::PsiIndex(2, 2)
            || n == ref_gh::PsiIndex(3, 3)) expected = 1.0;
        const Real error = Kokkos::abs(state(m, n, k, j, i) - expected);
        if (!Kokkos::isfinite(error)) {
          local_maximum = std::numeric_limits<Real>::infinity();
        } else {
          local_maximum = fmax(local_maximum, error);
        }
      }, Kokkos::Max<Real>(maximum));
  if (maximum > 1.0e-14) {
    std::cout << "reference-GH Minkowski evolution failed: max error = "
              << maximum << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "reference-GH Minkowski evolution passed: max error = "
            << maximum << std::endl;
}

}  // namespace

void ProblemGenerator::RefGhMinkowski(ParameterInput *, const bool restart) {
  pgen_final_func = &CheckRefGhMinkowski;
  if (restart) return;
  auto *pack = pmy_mesh_->pmb_pack;
  if (pack->prefgh == nullptr) {
    std::cout << "reference-GH Minkowski data require a <ref_gh> block." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto &indcs = pack->pmesh->mb_indcs;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const int n2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*indcs.ng : 1;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*indcs.ng : 1;
  const auto state = pack->prefgh->u0;
  par_for("ref_gh Minkowski data", DevExeSpace(), 0, pack->nmb_thispack - 1,
  0, n3 - 1, 0, n2 - 1, 0, n1 - 1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    for (int n = 0; n < ref_gh::nvar; ++n) state(m, n, k, j, i) = 0.0;
    state(m, ref_gh::PsiIndex(0, 0), k, j, i) = -1.0;
    state(m, ref_gh::PsiIndex(1, 1), k, j, i) = 1.0;
    state(m, ref_gh::PsiIndex(2, 2), k, j, i) = 1.0;
    state(m, ref_gh::PsiIndex(3, 3), k, j, i) = 1.0;
  });
}
