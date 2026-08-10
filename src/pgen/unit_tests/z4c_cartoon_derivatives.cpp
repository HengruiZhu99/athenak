//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c_cartoon_derivatives.cpp
//! \brief Input-selected manufactured-solution pgen for the production SO(2) provider.

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <string>

#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/z4c_cartoon_derivatives.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_symmetry.hpp"

void ProblemGenerator::Z4cCartoonDerivatives(ParameterInput *pin,
                                              const bool restart) {
  if (restart) {
    std::cerr << "z4c_cartoon_derivatives is check_only and rejects restart\n";
    std::exit(EXIT_FAILURE);
  }
  std::string raw_check_only = pin->DoesParameterExist("problem", "check_only")
                                   ? pin->GetString("problem", "check_only")
                                   : "";
  std::transform(raw_check_only.begin(), raw_check_only.end(),
                 raw_check_only.begin(), [](const unsigned char character) {
                   return static_cast<char>(std::tolower(character));
                 });
  if (raw_check_only != "true" && raw_check_only != "1") {
    std::cerr << "z4c_cartoon_derivatives requires <problem>/check_only=true\n";
    std::exit(EXIT_FAILURE);
  }
  MeshBlockPack *pack = pmy_mesh_->pmb_pack;
  if (pack->pz4c == nullptr ||
      pack->z4c_symmetry.mode != z4c::Z4cSymmetryMode::cartoon_so2) {
    std::cerr << "z4c_cartoon_derivatives requires staged cartoon_so2 Z4c\n";
    std::exit(EXIT_FAILURE);
  }
  z4c_mms::RunCartoonDerivativeMms(pin, pmy_mesh_);
}
