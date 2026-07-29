//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file scalar_field_update.cpp
//! \brief Low-storage explicit Runge-Kutta update for canonical scalar fields.

#include "athena.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "scalar_field/scalar_field.hpp"

namespace scalar_field {

//----------------------------------------------------------------------------------------
//! \fn TaskStatus ScalarField::ExpRKUpdate(Driver*, int)
//! \brief Update the active scalar cells using the driver's low-storage RK coefficients.

TaskStatus ScalarField::ExpRKUpdate(Driver *driver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is;
  const int ie = indcs.ie;
  const int js = indcs.js;
  const int je = indcs.je;
  const int ks = indcs.ks;
  const int ke = indcs.ke;
  const int nmb1 = pmy_pack->nmb_thispack - 1;

  const Real gam0 = driver->gam0[stage - 1];
  const Real gam1 = driver->gam1[stage - 1];
  const Real beta_dt = driver->beta[stage - 1]*pmy_pack->pmesh->dt;

  par_for(
      "scalar field RK update", DevExeSpace(), 0, nmb1, 0, nvar - 1,
      ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(const int m, const int n, const int k, const int j,
                    const int i) {
        u0(m, n, k, j, i) = gam0*u0(m, n, k, j, i) +
                            gam1*u1(m, n, k, j, i) +
                            beta_dt*u_rhs(m, n, k, j, i);
      });

  return TaskStatus::complete;
}

}  // namespace scalar_field
