//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c_vc_minkowski.cpp
//! \brief Exact smooth vacuum carrier for native CC/VC Z4c lifecycle tests.

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "boltzmann/boltzmann.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "pgen/z4c/kerr_puncture.hpp"
#include "pgen/tests/z4c_deterministic_amr.hpp"
#include "z4c/z4c.hpp"

namespace {
void SetFlatSlice(MeshBlockPack *pack, Real hubble) {
  auto state = pack->pz4c->u0;
  const auto bounds = pack->pz4c->layout;
  const int nmb = pack->nmb_thispack;
  Kokkos::deep_copy(state, 0.0);
  par_for(
      "initialize exact native Z4c Minkowski", DevExeSpace(), 0, nmb - 1, 0, bounds.n3 - 1, 0,
      bounds.n2 - 1, 0, bounds.n1 - 1,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        state(m, z4c::Z4c::I_Z4C_CHI, k, j, i) = 1.0;
        state(m, z4c::Z4c::I_Z4C_GXX, k, j, i) = 1.0;
        state(m, z4c::Z4c::I_Z4C_GYY, k, j, i) = 1.0;
        state(m, z4c::Z4c::I_Z4C_GZZ, k, j, i) = 1.0;
        state(m, z4c::Z4c::I_Z4C_KHAT, k, j, i) = -3 * hubble;
        state(m, z4c::Z4c::I_Z4C_ALPHA, k, j, i) = 1.0;
      });
  Kokkos::fence();
  pack->pz4c->ApplyVertexAxisRegularity(pack->pz4c->u0, 0, "initial_data");
  pack->pz4c->ReconstructAxisParityGhosts();
  pack->pz4c->Z4cToADM(pack);
}

void SetInitialConstraints(MeshBlockPack *pack) {
  switch (pack->pz4c->opt.fd_stencil) {
    case 2:
      pack->pz4c->ADMConstraints<2>(pack);
      break;
    case 3:
      pack->pz4c->ADMConstraints<3>(pack);
      break;
    case 4:
      pack->pz4c->ADMConstraints<4>(pack);
      break;
    default:
      std::cerr << "### FATAL ERROR: invalid Z4c stencil in Minkowski pgen" << std::endl;
      std::exit(EXIT_FAILURE);
  }
}
}  // namespace

void ProblemGenerator::Z4cVcMinkowski(ParameterInput *pin, const bool restart) {
  if (z4c_test::ConfigureDeterministicRefinementSchedule(pin, pmy_mesh_, "z4c_vc_minkowski"))
    user_ref_func = z4c_test::DeterministicRefinementSchedule;
  if (restart) return;
  auto pack = pmy_mesh_->pmb_pack;
  if (!pack->pz4c || !pack->padm || pack->ptmunu)
    Kokkos::abort("z4c_vc_minkowski requires vacuum Z4c and ADM");
  SetFlatSlice(pack, 0);
  SetInitialConstraints(pack);
}

void ProblemGenerator::BoltzmannPhotons(ParameterInput *pin, const bool restart) {
  auto pack = pmy_mesh_->pmb_pack;
  auto photons = pack->pboltzmann;
  if (!photons) Kokkos::abort("photon problem requires <boltzmann>");
  if (z4c_test::ConfigureDeterministicRefinementSchedule(pin, pmy_mesh_, "boltzmann"))
    user_ref_func = z4c_test::DeterministicRefinementSchedule;
  user_hist_func = boltzmann::RingHistory;
  if (restart) return;
  const auto name = pin->GetString("problem", "pgen_name");
  const bool flrw = name == "boltzmann_flrw";
  const Real H = flrw ? pin->GetOrAddReal("problem", "hubble", 0.2) : 0;
  if (name == "boltzmann_kerr_beam") {
    if (!photons->fixed_metric ||
        pin->GetOrAddString("problem", "initial_gauge", "stationary") != "stationary")
      Kokkos::abort("Kerr reference beam requires a fixed metric in stationary gauge");
    InitializeKerrPuncture(pmy_mesh_, pin, false);
  } else {
    SetFlatSlice(pack, H);
  }
  if (flrw) {
    if (photons->nmu != 2 * photons->nmu_half || !photons->backreaction)
      Kokkos::abort("radiation FLRW requires both hemispheres and backreaction");
    const auto u = photons->u;
    const auto sizes = pack->pmb->mb_size.d_view;
    const auto b = pmy_mesh_->mb_indcs;
    const Real intensity = 3 * H * H / (32 * M_PI * M_PI);
    par_for(
        "isotropic FLRW energy", DevExeSpace(), 0, pack->nmb_thispack - 1, 0, photons->nang - 1,
        b.js, b.je, b.is, b.ie, KOKKOS_LAMBDA(int m, int n, int j, int i) {
          u(m, n, 0, j, i) = intensity * (sizes(m).x1min + (i - b.is + 0.5) * sizes(m).dx1);
        });
  } else {
    photons->InitializeRing(pin);
  }
  photons->GeometryAndStress(nullptr, 0);
  SetInitialConstraints(pack);
}
