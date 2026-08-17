#ifndef FO_GH_FO_GH_HPP_
#define FO_GH_FO_GH_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file fo_gh.hpp
//! \brief Definitions for the vacuum regularized first-order GH module.

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "fo_gh/fo_gh_state.hpp"
#include "parameter_input.hpp"

class MeshBlockPack;

namespace fo_gh {

class FoGh {
 public:
  FoGh(MeshBlockPack *ppack, ParameterInput *pin);
  ~FoGh() = default;

  static constexpr int nfo_gh = nvar;
  static char const * const StateNames[nfo_gh];

  struct Variables {
    AthenaTensor<Real, TensorSymm::SYM2, 3, 2> gtilde;
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> chi;
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> alpha;
    AthenaTensor<Real, TensorSymm::NONE, 3, 1> beta;
    AthenaTensor<Real, TensorSymm::SYM2, 3, 2> Atilde;
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> K;
    AthenaTensor<Real, TensorSymm::NONE, 3, 1> Lambda;
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> pi;
    AthenaTensor<Real, TensorSymm::SYM2, 3, 2> Q[3]; // NOLINT(runtime/arrays)
    AthenaTensor<Real, TensorSymm::NONE, 3, 1> X;
    AthenaTensor<Real, TensorSymm::NONE, 3, 1> a;
    AthenaTensor<Real, TensorSymm::NONE, 3, 2> B;
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> h_perp;
    AthenaTensor<Real, TensorSymm::NONE, 3, 1> h;
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> vartheta_perp;
    AthenaTensor<Real, TensorSymm::NONE, 3, 1> vartheta;
  };

  struct Options {
    Real kappa;
    Real mu_H;
    Real eta_H;
    Real eta_beta;
    Real diss;
  } opt;

  DvceArray5D<Real> u0;
  DvceArray5D<Real> u1;
  DvceArray5D<Real> u_rhs;
  DvceArray5D<Real> coarse_u0;
  Variables u;
  Variables rhs;
  Real dtnew;

 private:
  void BindVariables(DvceArray5D<Real> data, Variables &vars);
  MeshBlockPack *pmy_pack;
};

} // namespace fo_gh

#endif // FO_GH_FO_GH_HPP_
