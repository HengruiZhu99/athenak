//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE for details
//========================================================================================
//! \file z4c_irisk_xcts.cpp
//! \brief Spectrally interpolate IrisK XCTS data onto an arbitrary AMR mesh.

#include <array>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "irisk_athenak_spectral_interpolator.h"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "z4c/z4c.hpp"

namespace {

[[noreturn]] void Fail(const std::string &message) {
  std::cout << "### FATAL ERROR in " << __FILE__ << std::endl
            << message << std::endl;
  std::exit(EXIT_FAILURE);
}

void FillAdmFromIrisSpectral(MeshBlockPack *pmbp,
                             IrisAthenakSpectralInterpolator *interpolator) {
  auto &u_adm = pmbp->padm->u_adm;
  HostArray5D<Real>::HostMirror host_u_adm = create_mirror(u_adm);
  HostArray5D<Real>::HostMirror host_u_z4c = create_mirror(pmbp->pz4c->u0);
  adm::ADM::ADMhost_vars host_adm;
  host_adm.g_dd.InitWithShallowSlice(host_u_adm, adm::ADM::I_ADM_GXX,
                                     adm::ADM::I_ADM_GZZ);
  host_adm.vK_dd.InitWithShallowSlice(host_u_adm, adm::ADM::I_ADM_KXX,
                                      adm::ADM::I_ADM_KZZ);
  host_adm.psi4.InitWithShallowSlice(host_u_adm, adm::ADM::I_ADM_PSI4);
  host_adm.alpha.InitWithShallowSlice(host_u_z4c, z4c::Z4c::I_Z4C_ALPHA);
  host_adm.beta_u.InitWithShallowSlice(host_u_z4c, z4c::Z4c::I_Z4C_BETAX,
                                       z4c::Z4c::I_Z4C_BETAZ);

  auto &indcs = pmbp->pmesh->mb_indcs;
  pmbp->pmb->mb_size.sync_host();
  auto size = pmbp->pmb->mb_size.h_view;
  const int isg = indcs.is - indcs.ng;
  const int ieg = indcs.ie + indcs.ng;
  const int jsg = indcs.js - indcs.ng;
  const int jeg = indcs.je + indcs.ng;
  const int ksg = indcs.ks - indcs.ng;
  const int keg = indcs.ke + indcs.ng;
  const std::size_t nx = static_cast<std::size_t>(ieg - isg + 1);
  const std::size_t ny = static_cast<std::size_t>(jeg - jsg + 1);
  const std::size_t nz = static_cast<std::size_t>(keg - ksg + 1);

  for (int m = 0; m < pmbp->nmb_thispack; ++m) {
    std::vector<double> x(nx), y(ny), z(nz);
    for (int i = isg; i <= ieg; ++i) {
      x[static_cast<std::size_t>(i - isg)] =
          CellCenterX(i - indcs.is, indcs.nx1, size(m).x1min, size(m).x1max);
    }
    for (int j = jsg; j <= jeg; ++j) {
      y[static_cast<std::size_t>(j - jsg)] =
          CellCenterX(j - indcs.js, indcs.nx2, size(m).x2min, size(m).x2max);
    }
    for (int k = ksg; k <= keg; ++k) {
      z[static_cast<std::size_t>(k - ksg)] =
          CellCenterX(k - indcs.ks, indcs.nx3, size(m).x3min, size(m).x3max);
    }
    std::vector<double> values(nx * ny * nz * IRISK_ATHENAK_ADM_VARIABLE_COUNT);
    std::array<char, 1024> error{};
    if (IrisAthenakSpectralInterpolateCartesian(
            interpolator, nx, ny, nz, x.data(), y.data(), z.data(),
            values.data(), error.data(), error.size()) != 0) {
      Fail(std::string("IrisK spectral interpolation failed: ") + error.data());
    }

    for (int k = ksg; k <= keg; ++k)
      for (int j = jsg; j <= jeg; ++j)
        for (int i = isg; i <= ieg; ++i) {
          const std::size_t point =
              static_cast<std::size_t>(i - isg) +
              nx * (static_cast<std::size_t>(j - jsg) +
                    ny * static_cast<std::size_t>(k - ksg));
          const double *value =
              values.data() + point * IRISK_ATHENAK_ADM_VARIABLE_COUNT;
          host_adm.g_dd(m, 0, 0, k, j, i) = value[0];
          host_adm.g_dd(m, 0, 1, k, j, i) = value[1];
          host_adm.g_dd(m, 0, 2, k, j, i) = value[2];
          host_adm.g_dd(m, 1, 1, k, j, i) = value[3];
          host_adm.g_dd(m, 1, 2, k, j, i) = value[4];
          host_adm.g_dd(m, 2, 2, k, j, i) = value[5];
          host_adm.vK_dd(m, 0, 0, k, j, i) = value[6];
          host_adm.vK_dd(m, 0, 1, k, j, i) = value[7];
          host_adm.vK_dd(m, 0, 2, k, j, i) = value[8];
          host_adm.vK_dd(m, 1, 1, k, j, i) = value[9];
          host_adm.vK_dd(m, 1, 2, k, j, i) = value[10];
          host_adm.vK_dd(m, 2, 2, k, j, i) = value[11];
          host_adm.psi4(m, k, j, i) = value[12];
          host_adm.alpha(m, k, j, i) = value[13];
          for (int component = 0; component < 3; ++component) {
            host_adm.beta_u(m, component, k, j, i) = value[14 + component];
          }
        }
  }
  Kokkos::deep_copy(u_adm, host_u_adm);
  Kokkos::deep_copy(pmbp->pz4c->u0, host_u_z4c);
}

} // namespace

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart)
    return;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pz4c == nullptr || pmbp->padm == nullptr) {
    Fail("z4c_irisk_xcts requires both <z4c> and <adm> blocks");
  }
  const std::string filename =
      pin->GetOrAddString("problem", "irisk_adm_spectral_file", "EMPTY");
  if (filename == "EMPTY" || filename.empty()) {
    Fail("z4c_irisk_xcts requires problem.irisk_adm_spectral_file");
  }
  IrisAthenakSpectralInterpolator *interpolator = nullptr;
  std::array<char, 1024> error{};
  if (IrisAthenakSpectralOpen(filename.c_str(), &interpolator, error.data(),
                              error.size()) != 0) {
    Fail(std::string("failed to open IrisK spectral data: ") + error.data());
  }
  FillAdmFromIrisSpectral(pmbp, interpolator);
  IrisAthenakSpectralClose(interpolator);

  // Match the established puncture import sequence, while preserving the
  // elliptically solved XCTS lapse and shift rather than imposing a puncture
  // pre-collapsed lapse.
  Z4cFinalizeImportedAdm(pin);
  std::cout << "Initialized Z4c from IrisK spectral XCTS data: " << filename
            << std::endl;
}
