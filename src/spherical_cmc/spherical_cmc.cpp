// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#include "spherical_cmc/spherical_cmc.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

#include "driver/driver.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"
#include "parameter_input.hpp"

namespace spherical_cmc {
const char *SphericalCMC::names[NVAR] = {
    "scmc_a",     "scmc_chi", "scmc_alpha",     "scmc_beta", "scmc_Khat",
    "scmc_Theta", "scmc_Arr", "scmc_Gamma",     "scmc_H",    "scmc_Krhs",
    "scmc_Arhs",  "scmc_Mr",  "scmc_mass_error"
};

SphericalCMC::SphericalCMC(MeshBlockPack *pack, ParameterInput *pin) : pmy_pack(pack) {
  const auto &ind = pack->pmesh->mb_indcs;
  const auto size = pack->pmb->mb_size.h_view(0);
  for (const char *block :
       {"z4c", "adm", "hydro", "mhd", "radiation", "gravity", "particles", "ion-neutral",
        "turb_driving", "cce", "fastflow"}) {
    if (pin->DoesBlockExist(block)) {
      throw std::invalid_argument("spherical_cmc is an exclusive vacuum physics module");
    }
  }
  if (sizeof(Real) != sizeof(double) || global_variable::nranks != 1 ||
      pack->nmb_thispack != 1 || pack->pmesh->nmb_total != 1 || pack->pmesh->multi_d ||
      pack->pmesh->multilevel || ind.ng < 2 || pack->pmesh->pzoom != nullptr ||
      size.x1max != 1 || pin->GetString("time", "integrator") != "rk3" ||
      pin->GetString("problem", "pgen_name") != "spherical_cmc") {
    throw std::invalid_argument(
        "spherical_cmc requires one uniform 1D block ending at r=1, "
        "double precision, ng>=2, rk3, and pgen_name=spherical_cmc; MPI and AMR are "
        "unsupported");
  }
  radial = std::make_unique<RadialSolver>(
      ind.nx1, pin->GetOrAddReal("spherical_cmc", "dirac_damping", 1),
      pin->GetOrAddReal("spherical_cmc", "mass", 0), size.x1min);
  const int ni = ind.nx1 + 2 * ind.ng;
  u0 = DvceArray5D<Real>("spherical_cmc state", 1, NVAR, 1, 1, ni);
  u1 = DvceArray5D<Real>("spherical_cmc RK base", 1, NVAR, 1, 1, ni);
  host = HostArray5D<Real>("spherical_cmc host", 1, NVAR, 1, 1, ni);
  saved = HostArray5D<Real>("spherical_cmc saved", 1, NVAR, 1, 1, ni);
  // Version lives in AthenaK's input/restart metadata; reject incompatible layouts.
  if (pin->GetOrAddInteger("spherical_cmc", "state_version", 1) != 1) {
    throw std::invalid_argument("unsupported spherical_cmc restart state version");
  }
}

void SphericalCMC::Initialize(ParameterInput *pin, bool restart) {
  if (!restart) {
    const double amplitude = pin->GetOrAddReal("spherical_cmc", "metric_pulse", 0.05);
    if (!std::isfinite(amplitude) || std::abs(amplitude) > 0.5) {
      throw std::invalid_argument(
          "spherical_cmc metric_pulse must be finite with |A|<=0.5");
    }
    for (int i = 0; i <= radial->n + 1; ++i) {
      const double r = radial->r[i];
      radial->a[i] = std::exp(amplitude * r * r * (1 - r * r) * (1 - r * r));
    }
    radial->Solve();
    StoreFields();
  }
  Refresh();
  NewTimeStep(nullptr, 0);
}

void SphericalCMC::LoadMetric() {
  Kokkos::deep_copy(host, u0);
  const int is = pmy_pack->pmesh->mb_indcs.is;
  // The nearest ghosts store the exact-boundary metric DOFs for restart.
  for (int i = 0; i <= radial->n + 1; ++i) radial->a[i] = host(0, AGR, 0, 0, is + i - 1);
}

void SphericalCMC::StoreFields() {
  const int is = pmy_pack->pmesh->mb_indcs.is;
  Kokkos::deep_copy(host, Real(0));
  for (int i = 0; i <= radial->n + 1; ++i) {
    const int cell = is + i - 1;
    host(0, AGR, 0, 0, cell) = radial->a[i];
    host(0, CHI, 0, 0, cell) = radial->chi[i];
    host(0, ALPHA, 0, 0, cell) = radial->alpha[i];
    host(0, BETA, 0, 0, cell) = radial->beta[i];
    host(0, KHAT, 0, 0, cell) = -3;
    host(0, ARR, 0, 0, cell) = radial->arr[i];
    host(0, GAMMA, 0, 0, cell) = radial->dirac[i];
    host(0, HAM, 0, 0, cell) = radial->hamiltonian[i];
    host(0, CMC, 0, 0, cell) = radial->cmc[i];
    host(0, ARHS, 0, 0, cell) = radial->a_rhs[i];
    host(0, MOM, 0, 0, cell) = radial->momentum[i];
    host(0, MASS_ERROR, 0, 0, cell) = radial->mass_error[i];
  }
  Kokkos::deep_copy(u0, host);
}

void SphericalCMC::Refresh() {
  LoadMetric();
  radial->Solve();
  radial->Diagnostics();
  StoreFields();
}

TaskStatus SphericalCMC::NewTimeStep(Driver *, int) {
  double speed = 1;
  for (int i = 0; i <= radial->n + 1; ++i) {
    speed =
        std::max(speed, std::abs(radial->beta[i]) +
                            radial->alpha[i] * std::sqrt(radial->chi[i] / radial->a[i]));
  }
  dtnew = std::min((1 - radial->inner_radius) / (radial->n * speed),
                   1. / (1 + radial->damping));
  return TaskStatus::complete;
}

void SphericalCMC::History(HistoryData *data) {
  data->nhist = 11;
  const char *labels[] = {"a_dev_max", "H_max",     "Krhs_max",      "Arhs_max",
                          "Dirac_max", "null_scri", "shift_defect",  "alpha_min",
                          "a_scri",    "Mr_max",    "mass_rle08_max"};
  for (int k = 0; k < 11; ++k) {
    data->label[k] = labels[k];
    data->hdata[k] = 0;
  }
  data->hdata[7] = radial->alpha[0];
  for (int i = 1; i <= radial->n; ++i) {
    data->hdata[0] = std::max(data->hdata[0], std::abs(radial->a[i] - 1));
    data->hdata[1] = std::max(data->hdata[1], std::abs(radial->hamiltonian[i]));
    data->hdata[2] = std::max(data->hdata[2], std::abs(radial->cmc[i]));
    data->hdata[3] = std::max(data->hdata[3], std::abs(radial->a_rhs[i]));
    data->hdata[4] = std::max(data->hdata[4], std::abs(radial->dirac[i]));
    data->hdata[7] = std::min(data->hdata[7], radial->alpha[i]);
    data->hdata[9] = std::max(data->hdata[9], std::abs(radial->momentum[i]));
    if (radial->r[i] <= 0.8) {
      data->hdata[10] = std::max(data->hdata[10], std::abs(radial->mass_error[i]));
    }
  }
  const int s = radial->n + 1;
  data->hdata[5] =
      radial->chi[s] / radial->a[s] - std::pow(radial->beta[s] / radial->alpha[s], 2);
  data->hdata[6] = radial->shift_residual;
  data->hdata[8] = radial->a[s];
}
}  // namespace spherical_cmc
