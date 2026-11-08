//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

#include <stdexcept>

#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "z4c_hyperboloidal/z4c_hyperboloidal.hpp"

void ProblemGenerator::Z4cHyperboloidalPuncture(ParameterInput *, const bool restart) {
  auto *physics=pmy_mesh_->pmb_pack->pz4c_hyperboloidal;
  if (physics==nullptr) {
    throw std::invalid_argument("Hyperboloidal puncture requires z4c_hyperboloidal");
  }
  if (restart) {
    throw std::runtime_error("Hyperboloidal checkpoint state loading is not yet integrated");
  }
  physics->InitializeReference();
}
