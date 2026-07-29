//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file scalar_field_newdt.cpp
//! \brief Conservative scalar-field light-cone and potential timestep.

#include <algorithm>
#include <cmath>
#include <limits>

#include <Kokkos_Core.hpp>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/coordinates.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "scalar_field/scalar_field.hpp"

namespace scalar_field {

TaskStatus ScalarField::NewTimeStep(Driver *pdriver, int stage) {
  if (stage != pdriver->nexp_stages) {
    return TaskStatus::complete;
  }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  auto &adm_vars = pmy_pack->padm->adm;
  auto &excision_mask = pmy_pack->pcoord->excision_floor;
  auto &state = u0;
  const int nx1 = indcs.nx1;
  const int nx2 = indcs.nx2;
  const int nx3 = indcs.nx3;
  const int is = indcs.is;
  const int js = indcs.js;
  const int ks = indcs.ks;
  const int ndim = 1 + static_cast<int>(pmy_pack->pmesh->multi_d) +
                   static_cast<int>(pmy_pack->pmesh->three_d);
  const int ncomp = ncomponents;
  const int nkji = nx1*nx2*nx3;
  const int ncell = pmy_pack->nmb_thispack*nkji;
  const PotentialData pot = potential;
  const bool use_excision = excision;
  const Real damping_time = excision_tdamp;
  const Real epsilon = 16.0*std::numeric_limits<Real>::epsilon();
  Real minimum_dt = std::numeric_limits<float>::max();

  Kokkos::parallel_reduce(
      "scalar field dt",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, ncell),
      KOKKOS_LAMBDA(const int index, Real &local_minimum) {
        const int m = index/nkji;
        const int cell = index - m*nkji;
        const int k = ks + cell/(nx1*nx2);
        const int row = cell - (k - ks)*nx1*nx2;
        const int j = js + row/nx1;
        const int i = is + row - (j - js)*nx1;

        if (use_excision && excision_mask(m, k, j, i)) {
          local_minimum = fmin(local_minimum, damping_time);
          return;
        }

        const Real gxx = adm_vars.g_dd(m, 0, 0, k, j, i);
        const Real gxy = adm_vars.g_dd(m, 0, 1, k, j, i);
        const Real gxz = adm_vars.g_dd(m, 0, 2, k, j, i);
        const Real gyy = adm_vars.g_dd(m, 1, 1, k, j, i);
        const Real gyz = adm_vars.g_dd(m, 1, 2, k, j, i);
        const Real gzz = adm_vars.g_dd(m, 2, 2, k, j, i);
        const Real determinant =
            adm::SpatialDet(gxx, gxy, gxz, gyy, gyz, gzz);
        Real guu[6];
        adm::SpatialInv(1.0/determinant, gxx, gxy, gxz, gyy, gyz, gzz,
                        &guu[0], &guu[1], &guu[2],
                        &guu[3], &guu[4], &guu[5]);

        const Real alpha = adm_vars.alpha(m, k, j, i);
        const Real absolute_alpha = fabs(alpha);
        const Real dx[3] = {
          size.d_view(m).dx1,
          size.d_view(m).dx2,
          size.d_view(m).dx3
        };
        const int diagonal[3] = {0, 3, 5};
        for (int direction = 0; direction < ndim; ++direction) {
          const Real speed =
              fabs(adm_vars.beta_u(m, direction, k, j, i)) +
              absolute_alpha*sqrt(fmax(guu[diagonal[direction]], 0.0));
          local_minimum =
              fmin(local_minimum, dx[direction]/fmax(speed, epsilon));
        }

        Real phi[2] = {0.0, 0.0};
        for (int component = 0; component < ncomp; ++component) {
          phi[component] = state(m, 2*component, k, j, i);
        }
        const Real frequency_squared =
            pot.FrequencySquared(FieldInvariant(ncomp, phi));
        if (frequency_squared > 0.0) {
          const Real coordinate_frequency =
              absolute_alpha*sqrt(frequency_squared);
          local_minimum = fmin(
              local_minimum, 1.0/fmax(coordinate_frequency, epsilon));
        }
      },
      Kokkos::Min<Real>(minimum_dt));

  if (use_excision) {
    minimum_dt = std::min(minimum_dt, damping_time);
  }
  dtnew = minimum_dt;
  return TaskStatus::complete;
}

} // namespace scalar_field
