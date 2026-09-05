// Independent nonlinear off-reduction jets, sampled from the production RHS.
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "pc_gh/pc_gh.hpp"
#include "pgen/pgen.hpp"

namespace {
using PC = pc_gh::PcGh;

void Audit(ParameterInput *, Mesh *pm) {
  auto *pc = pm->pmb_pack->ppcgh;
  auto state = pc->u0;
  auto size = pm->pmb_pack->pmb->mb_size;
  auto ind = pm->mb_indcs;
  int const ni=state.extent_int(4), nj=state.extent_int(3), nk=state.extent_int(2);
  std::ofstream out("nonlinear-jets.csv");
  out << "case,axis,offset,h,var,state,rhs\n" << std::setprecision(17);
  for (int c=0; c<3; ++c) {
    Real const w0 = c == 0 ? .25 : (c == 1 ? .5 : .9);
    par_for("nonlinear off-reduction probe", DevExeSpace(),
    0, 0, 0, nk-1, 0, nj-1, 0, ni-1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real const x[3] = {(i-ind.is-ind.nx1/2)*size.d_view(m).dx1,
                         (j-ind.js-ind.nx2/2)*size.d_view(m).dx2,
                         (k-ind.ks-ind.nx3/2)*size.d_view(m).dx3};
      for (int n=0; n<PC::npcgh; ++n) {
        Real value = .003*((n*7)%13-6);
        for (int d=0; d<3; ++d) value += .002*((n*3+d*5)%11-5)*x[d];
        state(m,n,k,j,i) = value;
      }
      state(m,PC::I_W,k,j,i) = w0+.025*x[0]-.04*x[1]+.015*x[2];
      state(m,PC::I_RHO,k,j,i) = 1.4-.03*x[0]+.02*x[1]+.01*x[2];
      for (int n=PC::I_GTXX; n<=PC::I_GTZZ; ++n) state(m,n,k,j,i)=0.;
      state(m,PC::I_GTXX,k,j,i)=state(m,PC::I_GTYY,k,j,i)=state(m,PC::I_GTZZ,k,j,i)=1.;
      state(m,PC::I_ATZZ,k,j,i)=-state(m,PC::I_ATXX,k,j,i)-state(m,PC::I_ATYY,k,j,i);
      for (int d=0; d<3; ++d) {
        state(m,PC::QIndex(d,2,2),k,j,i)=-state(m,PC::QIndex(d,0,0),k,j,i)
                                        -state(m,PC::QIndex(d,1,1),k,j,i);
        state(m,PC::I_BETAX+d,k,j,i) += .1*(d+1);
      }
    });
    (void)pc->CalcRHS<2>(nullptr, 0);
    auto u = Kokkos::create_mirror_view_and_copy(HostMemSpace(), state);
    auto f = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pc->u_rhs);
    auto sz = Kokkos::create_mirror_view_and_copy(HostMemSpace(), size.d_view);
    Real const dx[3] = {sz(0).dx1,sz(0).dx2,sz(0).dx3};
    for (int d=0; d<3; ++d) for (int offset=-1; offset<=1; ++offset) {
      int const i=ind.is+ind.nx1/2+(d==0 ? offset : 0);
      int const j=ind.js+ind.nx2/2+(d==1 ? offset : 0);
      int const k=ind.ks+ind.nx3/2+(d==2 ? offset : 0);
      for (int n=0; n<PC::npcgh; ++n) {
        out << c << ',' << d << ',' << offset << ',' << dx[d] << ',' << n << ','
            << u(0,n,k,j,i) << ',' << f(0,n,k,j,i) << '\n';
      }
    }
  }
  std::cout << "Recorded three nonlinear, algebraically tangent, off-reduction jets\n";
}
}  // namespace

void ProblemGenerator::UserProblem(ParameterInput *pin, bool restart) {
  auto *pm=pmy_mesh_; auto *pc=pm->pmb_pack->ppcgh;
  if (restart || pc==nullptr || pm->multilevel || global_variable::nranks!=1
      || pm->pmb_pack->nmb_thispack!=1 || pc->opt.fd_stencil!=2
      || pin->GetInteger("time","nlim")!=0 || pc->opt.dissipation!=0.
      || pc->opt.project_reduction_constraints || pc->opt.reduction_system!="advective") {
    std::cerr << "Nonlinear oracle needs one uniform block, zero steps, order 2, advective, no KO/projection\n";
    std::exit(EXIT_FAILURE);
  }
  std::cout << "Research qualification execution backend: " << DevExeSpace::name() << '\n';
  if (pin->GetOrAddBoolean("problem","require_cuda",false)
      && std::string(DevExeSpace::name())!="Cuda") std::exit(EXIT_FAILURE);
  Kokkos::deep_copy(pc->u0,0.);
  auto state=pc->u0;
  par_for("flat nonlinear oracle initialization", DevExeSpace(),0,0,
    0,state.extent_int(2)-1,0,state.extent_int(3)-1,0,state.extent_int(4)-1,
    KOKKOS_LAMBDA(int m,int k,int j,int i) {
      state(m,PC::I_W,k,j,i)=state(m,PC::I_RHO,k,j,i)=1.;
      state(m,PC::I_GTXX,k,j,i)=state(m,PC::I_GTYY,k,j,i)=state(m,PC::I_GTZZ,k,j,i)=1.;
    });
  pgen_final_func=Audit;
}
