//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c_cartoon_derivatives.hpp
//! \brief Non-templated host entry for the Cartoon derivative manufactured check.

#ifndef PGEN_UNIT_TESTS_Z4C_CARTOON_DERIVATIVES_HPP_
#define PGEN_UNIT_TESTS_Z4C_CARTOON_DERIVATIVES_HPP_

class Mesh;
class ParameterInput;

namespace z4c_mms {

void RunCartoonDerivativeMms(ParameterInput *pin, Mesh *mesh);

}  // namespace z4c_mms

#endif  // PGEN_UNIT_TESTS_Z4C_CARTOON_DERIVATIVES_HPP_
