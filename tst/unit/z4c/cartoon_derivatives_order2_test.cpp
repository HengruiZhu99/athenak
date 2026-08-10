//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cartoon_derivatives_order2_test.cpp
//! \brief Second-order analytic SO(2) manufactured-field checks.

#include "cartoon_derivatives_test_common.hpp"

bool RunCartoonDerivativeOrder2() {
  return CheckParity() && CheckIndependentPiRotation() && CheckOrder<2>();
}
