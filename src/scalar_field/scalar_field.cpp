//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file scalar_field.cpp
//! \brief Constructor and storage for canonical scalar fields.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include <Kokkos_Core.hpp>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "mesh/mesh.hpp"
#include "scalar_field/scalar_field.hpp"
#include "z4c/z4c.hpp"

namespace scalar_field {
namespace {

[[noreturn]] void FatalInput(const std::string &message) {
  std::cout << "### FATAL ERROR in " << __FILE__ << std::endl
            << message << std::endl;
  std::exit(EXIT_FAILURE);
}

} // namespace

ScalarField::ScalarField(MeshBlockPack *ppack, ParameterInput *pin)
    : ncomponents(1),
      nvar(2),
      spatial_order(2),
      fd_stencil(2),
      extrap_order(2),
      backreaction(true),
      excision(false),
      diss(0.0),
      dtnew(std::numeric_limits<float>::max()),
      excision_phi(0.0),
      excision_pi(0.0),
      excision_tdamp(1.0),
      potential(PotentialType::free, 0.0, 0.0),
      u0("u0 scalar field", 1, 1, 1, 1, 1),
      u1("u1 scalar field", 1, 1, 1, 1, 1),
      u_rhs("u_rhs scalar field", 1, 1, 1, 1, 1),
      coarse_u0("coarse u0 scalar field", 1, 1, 1, 1, 1),
      pbval_u(nullptr),
      pmy_pack(ppack) {
  if (ppack->padm == nullptr) {
    FatalInput("<scalar_field> requires either an <adm> or <z4c> block.");
  }
  if (ppack->pmesh->mesh_bcs[BoundaryFace::inner_x1] ==
          BoundaryFlag::shear_periodic ||
      ppack->pmesh->mesh_bcs[BoundaryFace::outer_x1] ==
          BoundaryFlag::shear_periodic) {
    FatalInput("<scalar_field> does not support shear-periodic boundaries.");
  }

  const std::string field_type =
      pin->GetOrAddString("scalar_field", "field_type", "real");
  if (field_type == "real") {
    ncomponents = 1;
  } else if (field_type == "complex") {
    ncomponents = 2;
  } else {
    FatalInput("<scalar_field>/field_type must be real or complex, but is " +
               field_type + ".");
  }
  nvar = 2*ncomponents;

  const std::string potential_name =
      pin->GetOrAddString("scalar_field", "potential", "free");
  PotentialType potential_type;
  if (potential_name == "free") {
    potential_type = PotentialType::free;
  } else if (potential_name == "mass_quartic") {
    potential_type = PotentialType::mass_quartic;
  } else {
    FatalInput("<scalar_field>/potential must be free or mass_quartic, but is " +
               potential_name + ".");
  }

  const Real mass = pin->GetOrAddReal("scalar_field", "mass", 0.0);
  const Real quartic = pin->GetOrAddReal("scalar_field", "lambda", 0.0);
  if (mass < 0.0) {
    FatalInput("<scalar_field>/mass must be nonnegative.");
  }
  if (quartic < 0.0) {
    FatalInput("<scalar_field>/lambda must be nonnegative.");
  }
  potential = PotentialData(potential_type, mass, quartic);
  backreaction = pin->GetOrAddBoolean(
      "scalar_field", "backreaction", ppack->pz4c != nullptr);
  if (backreaction && ppack->pz4c == nullptr) {
    FatalInput("<scalar_field>/backreaction=true requires Z4c.");
  }

  excision = ppack->pcoord->coord_data.bh_excise;
  if (excision) {
    if (!ppack->pcoord->coord_data.smooth_excision) {
      FatalInput("Scalar black-hole excision requires "
                 "<coord>/smooth_excision=true.");
    }
    excision_phi =
        pin->GetOrAddReal("scalar_field", "excision_phi", 0.0);
    excision_pi =
        pin->GetOrAddReal("scalar_field", "excision_pi", 0.0);
    excision_tdamp = pin->GetOrAddReal(
        "scalar_field", "excision_tdamp",
        ppack->pcoord->coord_data.tdamp);
    if (!std::isfinite(excision_tdamp) || excision_tdamp <= 0.0) {
      FatalInput("<scalar_field>/excision_tdamp must be finite and positive.");
    }
  }

  auto &indcs = ppack->pmesh->mb_indcs;
  if (ppack->pz4c != nullptr) {
    spatial_order = ppack->pz4c->opt.spatial_order;
    fd_stencil = ppack->pz4c->opt.fd_stencil;
    if (pin->DoesParameterExist("scalar_field", "spatial_order")) {
      const int requested =
          pin->GetInteger("scalar_field", "spatial_order");
      if (requested != spatial_order) {
        FatalInput("<scalar_field>/spatial_order must match <z4c>/spatial_order.");
      }
    }
  } else {
    spatial_order =
        pin->GetOrAddInteger("scalar_field", "spatial_order", 2);
    if (spatial_order != 2 && spatial_order != 4 && spatial_order != 6) {
      FatalInput("<scalar_field>/spatial_order must be 2, 4, or 6.");
    }
    fd_stencil = spatial_order/2 + 1;
  }
  if (indcs.ng < fd_stencil) {
    FatalInput("<scalar_field>/spatial_order requires at least " +
               std::to_string(fd_stencil) + " ghost cells.");
  }
  if (ppack->pmesh->multilevel && indcs.ng != 2 && indcs.ng != 4) {
    FatalInput("High-order multilevel scalar evolution requires nghost=2 or 4.");
  }
  if (ppack->pmesh->multilevel &&
      (indcs.nx1 < 2*indcs.ng ||
       (ppack->pmesh->multi_d && indcs.nx2 < 2*indcs.ng) ||
       (ppack->pmesh->three_d && indcs.nx3 < 2*indcs.ng))) {
    FatalInput("Multilevel scalar evolution requires at least 2*nghost cells "
               "per active MeshBlock dimension.");
  }
  extrap_order = std::max(2, std::min(indcs.ng, std::min(
      4, pin->GetOrAddInteger("scalar_field", "extrap_order", 2))));

  const Real input_diss =
      pin->GetOrAddReal("scalar_field", "diss", 0.0);
  const Real diss_sign = (fd_stencil % 2 == 0) ? -1.0 : 1.0;
  diss = input_diss*std::pow(2.0, -2.0*fd_stencil)*diss_sign;

  const int nmb = std::max(ppack->nmb_thispack,
                           ppack->pmesh->nmb_maxperrank);
  const int ncells1 = indcs.nx1 + 2*indcs.ng;
  const int ncells2 =
      (indcs.nx2 > 1) ? indcs.nx2 + 2*indcs.ng : 1;
  const int ncells3 =
      (indcs.nx3 > 1) ? indcs.nx3 + 2*indcs.ng : 1;
  Kokkos::realloc(u0, nmb, nvar, ncells3, ncells2, ncells1);
  Kokkos::realloc(u1, nmb, nvar, ncells3, ncells2, ncells1);
  Kokkos::realloc(u_rhs, nmb, nvar, ncells3, ncells2, ncells1);
  Kokkos::deep_copy(u0, 0.0);
  Kokkos::deep_copy(u1, 0.0);
  Kokkos::deep_copy(u_rhs, 0.0);

  if (ppack->pmesh->multilevel) {
    const int nccells1 = indcs.cnx1 + 2*indcs.ng;
    const int nccells2 =
        (indcs.cnx2 > 1) ? indcs.cnx2 + 2*indcs.ng : 1;
    const int nccells3 =
        (indcs.cnx3 > 1) ? indcs.cnx3 + 2*indcs.ng : 1;
    Kokkos::realloc(coarse_u0, nmb, nvar, nccells3, nccells2, nccells1);
    Kokkos::deep_copy(coarse_u0, 0.0);
  }

  pbval_u = new MeshBoundaryValuesCC(ppack, pin, true);
  pbval_u->InitializeBuffers(nvar);
  if (!ppack->pmesh->strictly_periodic) {
    Kokkos::deep_copy(pbval_u->u_in.d_view, 0.0);
  }
}

ScalarField::~ScalarField() {
  delete pbval_u;
}

} // namespace scalar_field
