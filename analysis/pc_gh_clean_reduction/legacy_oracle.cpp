// Identical test adapter compiled against collision and current production source.
// Samples arbitrary smooth off-constraint data; it is not a physical solution test.
#include <cmath>
#include <fstream>
#include <iomanip>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "pc_gh/pc_gh.hpp"
#include "pgen/pgen.hpp"

namespace {
using PC = pc_gh::PcGh;

void Fill(Mesh *mesh, int sample) {
  auto state = mesh->pmb_pack->ppcgh->u0;
  auto ind = mesh->mb_indcs;
  auto size = mesh->pmb_pack->pmb->mb_size;
  bool const three = mesh->three_d;
  par_for("legacy qualification jets", DevExeSpace(), 0, 0,
      0, state.extent_int(2)-1, 0, state.extent_int(3)-1,
      0, state.extent_int(4)-1,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x = (i-ind.is+0.5)*size.d_view(m).dx1;
    Real y = (j-ind.js+0.5)*size.d_view(m).dx2;
    Real z = three ? (k-ind.ks+0.5)*size.d_view(m).dx3 : 0.0;
    Real phase = 6.283185307179586*(x+y/1.3+z/1.7);
    for (int n=0; n<PC::npcgh; ++n) {
      state(m,n,k,j,i) = 0.01*std::sin(phase+0.37*n+0.13*sample);
    }
    Real w = 0.3+0.1*(sample%7)+0.03*std::sin(phase);
    state(m,PC::I_W,k,j,i) = w;
    state(m,PC::I_RHO,k,j,i) = 0.8+0.05*std::cos(phase+0.2);
    Real a = 0.04*std::sin(phase+0.3), c = 0.03*std::cos(phase);
    Real b = 0.02*std::sin(phase-0.1), d=0.01*std::cos(phase+0.1);
    Real e = 0.03*std::sin(phase+0.2);
    Real ta = std::exp(a), tc=std::exp(c), tf=std::exp(-a-c);
    state(m,PC::I_GTXX,k,j,i)=ta*ta;
    state(m,PC::I_GTXY,k,j,i)=ta*b;
    state(m,PC::I_GTXZ,k,j,i)=ta*d;
    state(m,PC::I_GTYY,k,j,i)=b*b+tc*tc;
    state(m,PC::I_GTYZ,k,j,i)=b*d+tc*e;
    state(m,PC::I_GTZZ,k,j,i)=d*d+e*e+tf*tf;
  });
  Kokkos::fence();
}

void Dump(Mesh *mesh, std::ofstream &out, int sample, int operation,
          DvceArray5D<Real> view) {
  auto h = Kokkos::create_mirror_view_and_copy(HostMemSpace(),view);
  auto ind=mesh->mb_indcs;
  for (int k=ind.ks; k<=ind.ke; ++k) {
    for (int j=ind.js; j<=ind.je; ++j) {
      for (int i=ind.is; i<=ind.ie; ++i) {
        for (int n=0; n<PC::npcgh; ++n) {
          out << sample << ',' << operation << ',' << k << ',' << j << ',' << i
              << ',' << n << ',' << h(0,n,k,j,i) << '\n';
        }
      }
    }
  }
}

void Final(ParameterInput *pin, Mesh *mesh) {
  auto *pc=mesh->pmb_pack->ppcgh;
  std::ofstream out("legacy-oracle.csv");
  out << "sample,operation,k,j,i,var,value\n" << std::setprecision(17);
  if (pin->GetInteger("time","nlim")>0) {
    Dump(mesh,out,0,3,pc->u0);
    return;
  }
  for (int sample=0; sample<7; ++sample) {
    Fill(mesh,sample);
    switch (pc->opt.fd_stencil) {
      case 2: pc->CalcRHS<2>(nullptr,0); break;
      case 3: pc->CalcRHS<3>(nullptr,0); break;
      case 4: pc->CalcRHS<4>(nullptr,0); break;
    }
    Dump(mesh,out,sample,0,pc->u_rhs);
    pc->ProjectAlgebraic(mesh->pmb_pack);
    pc->ProjectGaugeConstraints(mesh->pmb_pack);
    Dump(mesh,out,sample,1,pc->u0);
    switch (pc->opt.fd_stencil) {
      case 2: pc->ProjectReduction<2>(mesh->pmb_pack); break;
      case 3: pc->ProjectReduction<3>(mesh->pmb_pack); break;
      case 4: pc->ProjectReduction<4>(mesh->pmb_pack); break;
    }
    Dump(mesh,out,sample,2,pc->u0);
  }
}
}  // namespace

void ProblemGenerator::UserProblem(ParameterInput *pin, bool restart) {
  if (restart) return;  // Resume the saved state without refilling the oracle fixture.
  if (global_variable::nranks!=1 || pmy_mesh_->multilevel
      || pmy_mesh_->pmb_pack->nmb_thispack!=1) {
    throw std::runtime_error("legacy oracle initialization requires one uniform block");
  }
  pgen_final_func=Final;
  Fill(pmy_mesh_,3);
}
