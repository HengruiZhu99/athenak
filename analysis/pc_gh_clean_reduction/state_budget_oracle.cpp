// Test adapter for the actual production snapshot/writer, including ghost cells.
#include <stdexcept>
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "pc_gh/pc_gh.hpp"
#include "pgen/pgen.hpp"

namespace {
void Final(ParameterInput *, Mesh *mesh) {
  auto *pc=mesh->pmb_pack->ppcgh;
  auto state=pc->u0;
  int nv=state.extent_int(1), nk=state.extent_int(2);
  int nj=state.extent_int(3), ni=state.extent_int(4);
  par_for("state budget seed",DevExeSpace(),0,0,0,nv-1,0,nk-1,0,nj-1,0,ni-1,
    KOKKOS_LAMBDA(int m,int n,int k,int j,int i) {
      state(m,n,k,j,i)=0.01*(n+1)+0.001*i;
    });
  Kokkos::fence();
  pc->BeginStateBudget(1001);
  par_for("known signed state budget increments",DevExeSpace(),
    0,0,0,nv-1,0,nk-1,0,nj-1,0,ni-1,
    KOKKOS_LAMBDA(int m,int n,int k,int j,int i) {
      state(m,n,k,j,i)+=0.1*((i+2*j+3*k+5*n)%7-3);
    });
  Kokkos::fence();
  pc->EndStateBudget(1001);
  pc->BeginStateBudget(1002);
  pc->EndStateBudget(1002);
}
}

void ProblemGenerator::UserProblem(ParameterInput *pin,bool restart) {
  if (restart || pmy_mesh_->pmb_pack->nmb_thispack!=1
      || !pmy_mesh_->pmb_pack->ppcgh->opt.state_budget
      || pin->GetInteger("time","nlim")!=0) {
    throw std::runtime_error("state budget oracle requires one block, zero steps, budget on");
  }
  PcGhMinkowski(pin,restart);
  pgen_final_func=Final;
}
