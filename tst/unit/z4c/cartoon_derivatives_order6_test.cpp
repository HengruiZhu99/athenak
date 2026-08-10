//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cartoon_derivatives_order6_test.cpp
//! \brief Sixth-order analytic SO(2) manufactured-field checks.

#include "cartoon_derivatives_test_common.hpp"

bool RunCartoonDerivativeOrder6() { return CheckOrder<4>(); }
