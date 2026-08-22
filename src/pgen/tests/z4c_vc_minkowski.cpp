//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c_vc_minkowski.cpp
//! \brief Exact smooth vacuum carrier for native CC/VC Z4c lifecycle tests.

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "pgen/tests/z4c_deterministic_amr.hpp"
#include "z4c/z4c.hpp"

void ProblemGenerator::Z4cVcMinkowski(ParameterInput *pin,
                                      const bool restart) {
  if (z4c_test::ConfigureDeterministicRefinementSchedule(
          pin, pmy_mesh_, "z4c_vc_minkowski")) {
    user_ref_func = z4c_test::DeterministicRefinementSchedule;
  }
  if (restart) return;
  MeshBlockPack *pack = pmy_mesh_->pmb_pack;
  if (pack->pz4c == nullptr || pack->padm == nullptr ||
      pack->ptmunu != nullptr) {
    std::cerr << "### FATAL ERROR: z4c_vc_minkowski requires vacuum Z4c and ADM"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto state = pack->pz4c->u0;
  const auto bounds = pack->pz4c->layout;
  const int nmb = pack->nmb_thispack;
  Kokkos::deep_copy(state, 0.0);
  par_for("initialize exact native Z4c Minkowski", DevExeSpace(), 0, nmb - 1,
          0, bounds.n3 - 1, 0, bounds.n2 - 1, 0, bounds.n1 - 1,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        state(m, z4c::Z4c::I_Z4C_CHI, k, j, i) = 1.0;
        state(m, z4c::Z4c::I_Z4C_GXX, k, j, i) = 1.0;
        state(m, z4c::Z4c::I_Z4C_GYY, k, j, i) = 1.0;
        state(m, z4c::Z4c::I_Z4C_GZZ, k, j, i) = 1.0;
        state(m, z4c::Z4c::I_Z4C_ALPHA, k, j, i) = 1.0;
      });
  Kokkos::fence();
  pack->pz4c->ApplyVertexAxisRegularity(pack->pz4c->u0, 0, "initial_data");
  pack->pz4c->ReconstructAxisParityGhosts();
  pack->pz4c->Z4cToADM(pack);
  switch (pack->pz4c->opt.fd_stencil) {
    case 2: pack->pz4c->ADMConstraints<2>(pack); break;
    case 3: pack->pz4c->ADMConstraints<3>(pack); break;
    case 4: pack->pz4c->ADMConstraints<4>(pack); break;
    default:
      std::cerr << "### FATAL ERROR: invalid Z4c stencil in Minkowski pgen"
                << std::endl;
      std::exit(EXIT_FAILURE);
  }
}
