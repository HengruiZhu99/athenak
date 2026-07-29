#ifndef SCALAR_FIELD_SCALAR_FIELD_HPP_
#define SCALAR_FIELD_SCALAR_FIELD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file scalar_field.hpp
//! \brief Canonical real and complex scalar-field evolution on an ADM spacetime.

#include "athena.hpp"
#include "bvals/bvals.hpp"
#include "parameter_input.hpp"
#include "scalar_field/scalar_field_utils.hpp"
#include "tasklist/task_list.hpp"

class Driver;
class MeshBlockPack;

namespace scalar_field {

//! \class ScalarField
//! \brief Evolves one real component or the two real components of a complex field.
class ScalarField {
 public:
  enum VariableIndex {
    I_SF_PHI0 = 0,
    I_SF_PI0 = 1,
    I_SF_PHI1 = 2,
    I_SF_PI1 = 3
  };

  ScalarField(MeshBlockPack *ppack, ParameterInput *pin);
  ~ScalarField();

  int ncomponents;
  int nvar;
  int spatial_order;
  int fd_stencil;
  int extrap_order;
  bool backreaction;
  bool excision;
  Real diss;
  Real dtnew;
  Real excision_phi;
  Real excision_pi;
  Real excision_tdamp;
  PotentialData potential;

  DvceArray5D<Real> u0;
  DvceArray5D<Real> u1;
  DvceArray5D<Real> u_rhs;
  DvceArray5D<Real> coarse_u0;
  MeshBoundaryValuesCC *pbval_u;

  void QueueScalarFieldTasks();

  TaskStatus InitRecv(Driver *driver, int stage);
  TaskStatus ClearRecv(Driver *driver, int stage);
  TaskStatus ClearSend(Driver *driver, int stage);
  TaskStatus SetADM(Driver *driver, int stage);
  TaskStatus SetADMFinal(Driver *driver, int stage);
  TaskStatus CopyU(Driver *driver, int stage);
  template <int NGHOST>
  TaskStatus CalcRHS(Driver *driver, int stage);
  template <int NGHOST>
  TaskStatus AddTmunu(Driver *driver, int stage);
  TaskStatus AddTmunu(Driver *driver, int stage);
  TaskStatus AddTmunuFinal(Driver *driver, int stage);
  TaskStatus ExpRKUpdate(Driver *driver, int stage);
  TaskStatus RestrictU(Driver *driver, int stage);
  TaskStatus SendU(Driver *driver, int stage);
  TaskStatus RecvU(Driver *driver, int stage);
  TaskStatus ApplyPhysicalBCs(Driver *driver, int stage);
  TaskStatus Prolongate(Driver *driver, int stage);
  TaskStatus NewTimeStep(Driver *driver, int stage);

 private:
  MeshBlockPack *pmy_pack;
};

} // namespace scalar_field

#endif // SCALAR_FIELD_SCALAR_FIELD_HPP_
