#ifndef PC_GH_PC_GH_HPP_
#define PC_GH_PC_GH_HPP_

//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pc_gh.hpp
//! \brief puncture-conformal first-order generalized-harmonic state and interfaces

#include <string>

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "bvals/bvals.hpp"
#include "parameter_input.hpp"

class MeshBlockPack;
class Driver;

namespace pc_gh {

class PcGh {
 public:
  PcGh(MeshBlockPack *ppack, ParameterInput *pin);
  ~PcGh();

  // The ordering is a restart/output ABI. Q is stored as three consecutive
  // symmetric tensors Q_kij; B is stored row-major as B_i^j.
  enum : int {
    I_CHI,
    I_GTXX, I_GTXY, I_GTXZ, I_GTYY, I_GTYZ, I_GTZZ,
    I_K,
    I_ATXX, I_ATXY, I_ATXZ, I_ATYY, I_ATYZ, I_ATZZ,
    I_LAMX, I_LAMY, I_LAMZ,
    I_PI,
    I_A,
    I_BETAX, I_BETAY, I_BETAZ,
    I_X1, I_X2, I_X3,
    I_Q1XX, I_Q1XY, I_Q1XZ, I_Q1YY, I_Q1YZ, I_Q1ZZ,
    I_Q2XX, I_Q2XY, I_Q2XZ, I_Q2YY, I_Q2YZ, I_Q2ZZ,
    I_Q3XX, I_Q3XY, I_Q3XZ, I_Q3YY, I_Q3YZ, I_Q3ZZ,
    I_Y1, I_Y2, I_Y3,
    I_B11, I_B12, I_B13,
    I_B21, I_B22, I_B23,
    I_B31, I_B32, I_B33,
    npcgh
  };

  static_assert(npcgh == 55, "PC-GH state ABI must contain exactly 55 fields");
  static char const * const PcGhNames[npcgh];

  KOKKOS_INLINE_FUNCTION
  static constexpr int SymmetricIndex(int i, int j) {
    return (i > j) ? SymmetricIndex(j, i)
                   : ((i == 0) ? j : ((i == 1) ? j + 2 : 5));
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr int QIndex(int k, int i, int j) {
    return I_Q1XX + 6*k + SymmetricIndex(i, j);
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr int BIndex(int i, int j) {
    return I_B11 + 3*i + j;
  }

  template <int FD_STENCIL>
  void ADMToPcGh(MeshBlockPack *pmbp);
  template <int FD_STENCIL>
  TaskStatus CalcRHS(Driver *pdriver, int stage);
  void PcGhToADM(MeshBlockPack *pmbp);
  void ProjectAlgebraic(MeshBlockPack *pmbp);

  TaskStatus CopyU(Driver *pdriver, int stage);
  void QueuePcGhTasks();
  TaskStatus InitRecv(Driver *pdriver, int stage);
  TaskStatus ClearRecv(Driver *pdriver, int stage);
  TaskStatus ClearSend(Driver *pdriver, int stage);
  TaskStatus SendU(Driver *pdriver, int stage);
  TaskStatus RecvU(Driver *pdriver, int stage);
  TaskStatus RestrictU(Driver *pdriver, int stage);
  TaskStatus ApplyPhysicalBCs(Driver *pdriver, int stage);
  TaskStatus Prolongate(Driver *pdriver, int stage);
  TaskStatus EnforceAlgebraicConstraints(Driver *pdriver, int stage);
  TaskStatus ConvertToADM(Driver *pdriver, int stage);
  TaskStatus BoundaryRHS(Driver *pdriver, int stage);
  TaskStatus ExpRKUpdate(Driver *pdriver, int stage);
  TaskStatus NewTimeStep(Driver *pdriver, int stage);

  struct Variables {
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> chi;
    AthenaTensor<Real, TensorSymm::SYM2, 3, 2> gtilde;
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> K;
    AthenaTensor<Real, TensorSymm::SYM2, 3, 2> Atilde;
    AthenaTensor<Real, TensorSymm::NONE, 3, 1> Lambda;
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> pi;
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> A;
    AthenaTensor<Real, TensorSymm::NONE, 3, 1> beta;
    AthenaTensor<Real, TensorSymm::NONE, 3, 1> X;
    AthenaTensor<Real, TensorSymm::NONE, 3, 1> Y;
  };

  struct Options {
    int spatial_order;
    int fd_stencil;
    std::string gauge;
    Real kappa;
    Real dissipation;
  } opt;

  DvceArray5D<Real> u0;
  DvceArray5D<Real> u1;
  DvceArray5D<Real> u_rhs;
  DvceArray5D<Real> coarse_u0;
  Variables u;
  Variables rhs;
  MeshBoundaryValuesCC *pbval_u;
  Real dtnew;

 private:
  void BindVariables(DvceArray5D<Real> state, Variables &vars);

  MeshBlockPack *pmy_pack;
};

}  // namespace pc_gh

#endif  // PC_GH_PC_GH_HPP_
