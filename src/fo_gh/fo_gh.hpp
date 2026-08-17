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
#include "tasklist/task_list.hpp"

class MeshBlockPack;
class Driver;
class MeshBoundaryValuesCC;

namespace fo_gh {

class FoGh {
 public:
  FoGh(MeshBlockPack *ppack, ParameterInput *pin);
  ~FoGh();

  static constexpr int nfo_gh = nvar;
  static char const * const StateNames[nfo_gh];
  enum ConstraintIndex {
    I_CON_H, I_CON_MX, I_CON_MY, I_CON_MZ,
    I_CON_GH_PERP, I_CON_GHX, I_CON_GHY, I_CON_GHZ,
    I_CON_RQ, I_CON_RX, I_CON_RA, I_CON_RB, ncon
  };
  static char const * const ConstraintNames[ncon];

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
    int fd_order;
    int extrap_order;
    Real kappa;
    Real mu_H;
    Real eta_H;
    Real eta_beta;
    Real diss;
    Real excise_lapse;
  } opt;

  DvceArray5D<Real> u0;
  DvceArray5D<Real> u1;
  DvceArray5D<Real> u_rhs;
  DvceArray5D<Real> u_con;
  DvceArray5D<Real> coarse_u0;
  Variables u;
  Variables rhs;
  Real dtnew;
  Real max_char_speed;
  MeshBoundaryValuesCC *pbval_u;

  template <int FDNG>
  TaskStatus CalcRHS(Driver *d, int stage);
  template <int FDNG>
  void CalcConstraints();
  void QueueTasks();
  TaskStatus InitRecv(Driver *d, int stage);
  TaskStatus ClearRecv(Driver *d, int stage);
  TaskStatus ClearSend(Driver *d, int stage);
  TaskStatus CopyU(Driver *d, int stage);
  TaskStatus ExpRKUpdate(Driver *d, int stage);
  TaskStatus RestrictU(Driver *d, int stage);
  TaskStatus SendU(Driver *d, int stage);
  TaskStatus RecvU(Driver *d, int stage);
  TaskStatus Prolongate(Driver *d, int stage);
  TaskStatus ApplyPhysicalBCs(Driver *d, int stage);
  TaskStatus NewTimeStep(Driver *d, int stage);
  void RepairGradients(const DualArray1D<int> &repair);
  void FoGhToADM();
  void UpdateDiagnostics();

 private:
  void BindVariables(DvceArray5D<Real> data, Variables &vars);
  MeshBlockPack *pmy_pack;
};

KOKKOS_INLINE_FUNCTION
void LoadPoint(const FoGh::Variables &v, const int m, const int k,
               const int j, const int i, RegularPointState &u) {
  u.chi = v.chi(m, k, j, i);
  u.alpha = v.alpha(m, k, j, i);
  u.K = v.K(m, k, j, i);
  u.pi = v.pi(m, k, j, i);
  u.h_perp = v.h_perp(m, k, j, i);
  u.vartheta_perp = v.vartheta_perp(m, k, j, i);
  for (int a = 0; a < 3; ++a) {
    u.beta(a) = v.beta(m, a, k, j, i);
    u.Lambda(a) = v.Lambda(m, a, k, j, i);
    u.X(a) = v.X(m, a, k, j, i);
    u.a(a) = v.a(m, a, k, j, i);
    u.h(a) = v.h(m, a, k, j, i);
    u.vartheta(a) = v.vartheta(m, a, k, j, i);
    for (int b = 0; b < 3; ++b) {
      u.B(a, b) = v.B(m, a, b, k, j, i);
    }
    for (int b = 0; b < 3; ++b) {
      for (int c = b; c < 3; ++c) {
        u.Q(a, b, c) = v.Q[a](m, b, c, k, j, i);
      }
    }
  }
  for (int a = 0; a < 3; ++a) {
    for (int b = a; b < 3; ++b) {
      u.gtilde(a, b) = v.gtilde(m, a, b, k, j, i);
      u.Atilde(a, b) = v.Atilde(m, a, b, k, j, i);
    }
  }
}

} // namespace fo_gh

#endif // FO_GH_FO_GH_HPP_
