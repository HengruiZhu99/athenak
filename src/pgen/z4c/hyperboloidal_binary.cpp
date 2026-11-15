// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <stdexcept>
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "z4c_hyperboloidal/z4c_hyperboloidal.hpp"
void ProblemGenerator::Z4cHyperboloidalBinary(ParameterInput *pin,const bool restart) {
  auto *p=pmy_mesh_->pmb_pack->pz4c_hyperboloidal;
  if(!p || !p->parameters.residual_weight.Regular())
    throw std::invalid_argument("Binary pgen requires native hyperboloidal regular_c6 weight");
  if(!pin->GetOrAddBoolean("problem","binary_prepare",false))p->InitializeBinaryTracking();
  if(pmy_mesh_->adaptive) {
    if(pin->GetOrAddString("z4c_amr","method","chi")!="chi")
      throw std::invalid_argument("Binary AMR adapter supports the Cartesian chi criterion");
    user_ref_func=[](MeshBlockPack *pack){pack->pz4c_hyperboloidal->RefineBinaryChi();};
  }
  if(restart) {
    if(!p->IsInitialized()) throw std::runtime_error("Binary restart was not restored");
    return;
  }
  p->InitializeBinary(pin);
}
