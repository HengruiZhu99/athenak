//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c_gauge_wave.cpp
//! \brief Built-in analytic harmonic gauge-wave initial data for native CC/VC Z4c.

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "z4c/z4c.hpp"

namespace {

template <typename AdmVars>
void InitializeGaugeWaveAdm(MeshBlockPack *pack, const z4c::Z4cGridLayout &layout,
                            AdmVars &adm, const Real amplitude,
                            const Real wavelength) {
  const auto sizes = pack->pmb->mb_size;
  const int nx1 = pack->pmesh->mb_indcs.nx1;
  const bool vertex = layout.centering == z4c::Z4cGridCentering::vertex;
  const Real wave_number = 2.0 * M_PI / wavelength;
  par_for("initialize analytic Z4c gauge wave ADM", DevExeSpace(),
          0, pack->nmb_thispack - 1, 0, layout.n3 - 1, 0, layout.n2 - 1,
          0, layout.n1 - 1,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        const Real x = vertex
            ? VertexX(i - layout.is, nx1, sizes.d_view(m).x1min,
                      sizes.d_view(m).x1max)
            : CellCenterX(i - layout.is, nx1, sizes.d_view(m).x1min,
                          sizes.d_view(m).x1max);
        const Real phase = wave_number * x;
        const Real h = 1.0 - amplitude * Kokkos::sin(phase);
        const Real alpha = Kokkos::sqrt(h);
        adm.g_dd(m, 0, 0, k, j, i) = h;
        adm.g_dd(m, 0, 1, k, j, i) = 0.0;
        adm.g_dd(m, 0, 2, k, j, i) = 0.0;
        adm.g_dd(m, 1, 1, k, j, i) = 1.0;
        adm.g_dd(m, 1, 2, k, j, i) = 0.0;
        adm.g_dd(m, 2, 2, k, j, i) = 1.0;
        adm.vK_dd(m, 0, 0, k, j, i) =
            -0.5 * amplitude * wave_number * Kokkos::cos(phase) / alpha;
        adm.vK_dd(m, 0, 1, k, j, i) = 0.0;
        adm.vK_dd(m, 0, 2, k, j, i) = 0.0;
        adm.vK_dd(m, 1, 1, k, j, i) = 0.0;
        adm.vK_dd(m, 1, 2, k, j, i) = 0.0;
        adm.vK_dd(m, 2, 2, k, j, i) = 0.0;
        adm.psi4(m, k, j, i) = 1.0;
        adm.alpha(m, k, j, i) = alpha;
        for (int component = 0; component < 3; ++component) {
          adm.beta_u(m, component, k, j, i) = 0.0;
        }
      });
  Kokkos::fence();
}

}  // namespace

void ProblemGenerator::Z4cGaugeWave(ParameterInput *pin,
                                    const bool restart) {
  if (restart) return;
  MeshBlockPack *pack = pmy_mesh_->pmb_pack;
  if (pack->pz4c == nullptr || pack->padm == nullptr ||
      pack->ptmunu != nullptr) {
    std::cerr << "### FATAL ERROR: z4c_gauge_wave requires vacuum Z4c and ADM"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const auto layout = pack->pz4c->layout;
  const Real amplitude = pin->GetOrAddReal("problem", "amp", 0.01);
  const Real wavelength =
      pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;
  if (!(amplitude > 0.0 && amplitude < 1.0) || !(wavelength > 0.0)) {
    std::cerr << "### FATAL ERROR: z4c_gauge_wave requires 0 < amp < 1 and "
                 "positive x1 extent" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (layout.centering == z4c::Z4cGridCentering::vertex) {
    InitializeGaugeWaveAdm(pack, layout, pack->pz4c->adm, amplitude, wavelength);
  } else {
    InitializeGaugeWaveAdm(pack, layout, pack->padm->adm, amplitude, wavelength);
  }
  switch (pack->pz4c->opt.fd_stencil) {
    case 2: pack->pz4c->ADMToZ4c<2>(pack, pin); break;
    case 3: pack->pz4c->ADMToZ4c<3>(pack, pin); break;
    case 4: pack->pz4c->ADMToZ4c<4>(pack, pin); break;
    default:
      std::cerr << "### FATAL ERROR: invalid Z4c stencil in gauge-wave pgen"
                << std::endl;
      std::exit(EXIT_FAILURE);
  }
  pack->pz4c->Z4cToADM(pack);
  switch (pack->pz4c->opt.fd_stencil) {
    case 2: pack->pz4c->ADMConstraints<2>(pack); break;
    case 3: pack->pz4c->ADMConstraints<3>(pack); break;
    case 4: pack->pz4c->ADMConstraints<4>(pack); break;
    default: break;
  }
}
