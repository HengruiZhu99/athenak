//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c_vc_minkowski.cpp
//! \brief Exact smooth vacuum carrier for native CC/VC Z4c lifecycle tests.

#include <cstdlib>
#include <cmath>
#include <stdexcept>
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
  // Optional smooth gauge pulse on flat initial geometry for temporal tests.
  // Default zero retains the exact Minkowski lifecycle carrier.
  const Real amplitude = pin->GetOrAddReal("problem", "lapse_gaussian_amplitude", 0.0);
  const Real width = pin->GetOrAddReal("problem", "lapse_gaussian_width", 1.0);
  if (!std::isfinite(amplitude) || std::fabs(amplitude) >= 1.0 ||
      !std::isfinite(width) || width <= 0.0) {
    throw std::runtime_error("invalid Minkowski gauge-pulse amplitude or width");
  }
  if (amplitude != 0.0 &&
      (bounds.centering != z4c::Z4cGridCentering::vertex ||
       pin->GetString("z4c", "coordinate_map") != "half_rho_z_suppressed_y_v2")) {
    throw std::runtime_error("Minkowski gauge pulse requires native VC Cartoon");
  }
  auto sizes = pack->pmb->mb_size.d_view;
  Kokkos::deep_copy(state, 0.0);
  par_for("initialize exact native Z4c Minkowski", DevExeSpace(), 0, nmb - 1,
          0, bounds.n3 - 1, 0, bounds.n2 - 1, 0, bounds.n1 - 1,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        state(m, z4c::Z4c::I_Z4C_CHI, k, j, i) = 1.0;
        state(m, z4c::Z4c::I_Z4C_GXX, k, j, i) = 1.0;
        state(m, z4c::Z4c::I_Z4C_GYY, k, j, i) = 1.0;
        state(m, z4c::Z4c::I_Z4C_GZZ, k, j, i) = 1.0;
        const Real rho = sizes(m).x1min + (i-bounds.is)*sizes(m).dx1;
        const Real z = sizes(m).x2min + (j-bounds.js)*sizes(m).dx2;
        state(m, z4c::Z4c::I_Z4C_ALPHA, k, j, i) =
            1.0 + amplitude*exp(-(rho*rho+z*z)/(width*width));
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
