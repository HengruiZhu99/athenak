// Actual mesh transfer fixture, not an Einstein evolution or manufactured solution.
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <iomanip>
#include "globals.hpp"
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "pc_gh/pc_gh.hpp"
#include "pgen/pgen.hpp"

namespace {
using PC = pc_gh::PcGh;
void Ordinary(Mesh *mesh) {
  auto *pc = mesh->pmb_pack->ppcgh;
  while (pc->pbval_u->ClearSend() != TaskStatus::complete) {}
  while (pc->pbval_u->ClearRecv() != TaskStatus::complete) {}
  if (mesh->multilevel) mesh->pmr->RestrictCC(pc->u0,pc->coarse_u0,true);
  pc->pbval_u->InitRecv(PC::npcgh);
  pc->pbval_u->PackAndSendCC(pc->u0,pc->coarse_u0);
  while (pc->pbval_u->RecvAndUnpackCC(pc->u0,pc->coarse_u0)
         != TaskStatus::complete) {}
  if (mesh->multilevel) pc->pbval_u->ProlongateCC(pc->u0,pc->coarse_u0,true);
  while (pc->pbval_u->ClearSend() != TaskStatus::complete) {}
  while (pc->pbval_u->ClearRecv() != TaskStatus::complete) {}
}
void Final(ParameterInput *pin, Mesh *mesh) {
  auto *pc = mesh->pmb_pack->ppcgh;
  auto state = pc->u0;
  auto ind = mesh->mb_indcs;
  auto size = mesh->pmb_pack->pmb->mb_size;
  int nmb = mesh->pmb_pack->nmb_thispack;
  bool three = mesh->three_d;
  const auto profile=pin->GetOrAddString("problem","residual_profile","constant");
  if (profile != "constant" && profile != "smooth") {
    throw std::runtime_error("unknown residual fixture profile");
  }
  const bool smooth=profile == "smooth";
  par_for("transfer primary seed",DevExeSpace(),0,nmb-1,0,state.extent_int(2)-1,
  0,state.extent_int(3)-1,0,state.extent_int(4)-1,
  KOKKOS_LAMBDA(int m,int k,int j,int i) {
    Real x = size.d_view(m).x1min+(i-ind.is+0.5)*size.d_view(m).dx1;
    Real y = size.d_view(m).x2min+(j-ind.js+0.5)*size.d_view(m).dx2;
    Real z = three ? size.d_view(m).x3min+(k-ind.ks+0.5)*size.d_view(m).dx3 : 0;
    Real phase = 6.283185307179586*(x+y/1.3+z/1.7);
    for (int n=0; n<PC::npcgh; ++n) state(m,n,k,j,i)=.01*std::sin(phase+.13*n);
    state(m,PC::I_W,k,j,i)=.7+.03*std::sin(phase);
    state(m,PC::I_RHO,k,j,i)=.8+.02*std::cos(phase+.2);
    state(m,PC::I_GTXX,k,j,i)=1+.01*std::sin(phase);
    state(m,PC::I_GTYY,k,j,i)=1+.01*std::cos(phase);
    state(m,PC::I_GTZZ,k,j,i)=1;
  });
  Kokkos::fence();
  Ordinary(mesh);
  // Independent existing centered derivative kernel supplies active targets.
  switch (pc->opt.fd_stencil) {
    case 2: pc->ProjectReduction<2>(mesh->pmb_pack); break;
    case 3: pc->ProjectReduction<3>(mesh->pmb_pack); break;
    case 4: pc->ProjectReduction<4>(mesh->pmb_pack); break;
  }
  par_for("constant independent reductions",DevExeSpace(),0,nmb-1,PC::I_P1,
  PC::npcgh-1,ind.ks,ind.ke,ind.js,ind.je,ind.is,ind.ie,
  KOKKOS_LAMBDA(int m,int n,int k,int j,int i) {
    Real x=size.d_view(m).x1min+(i-ind.is+.5)*size.d_view(m).dx1;
    Real y=size.d_view(m).x2min+(j-ind.js+.5)*size.d_view(m).dx2;
    Real z=three?size.d_view(m).x3min+(k-ind.ks+.5)*size.d_view(m).dx3:0;
    Real phase=6.283185307179586*(x+y/1.3+z/1.7);
    state(m,n,k,j,i) += .001*(n-PC::I_P1+1)*(smooth?std::sin(phase+.13*n):1);
  });
  Kokkos::fence();
  // Compute shifted differentiation weights independently from Lagrange products.
  int order = pc->opt.spatial_order;
  long double weights[7][7] = {};
  for (int point=0; point<=order; ++point) for (int node=0; node<=order; ++node) {
    for (int excluded=0; excluded<=order; ++excluded) if (excluded != node) {
      long double value = 1.0L/(node-excluded);
      for (int q=0; q<=order; ++q) if (q != node && q != excluded) {
        value *= static_cast<long double>(point-q)/(node-q);
      }
      weights[point][node] += value;
    }
  }
  std::ofstream out("transfer-mesh-rank"+std::to_string(global_variable::my_rank)+".jsonl");
  out << std::setprecision(17);
  for (int repeat=0; repeat<3; ++repeat) {
    Ordinary(mesh);
    auto before = Kokkos::create_mirror(state);
    Kokkos::deep_copy(before,state);
    pc->opt.state_budget = pin->GetOrAddBoolean("problem","raw_budget",false);
    pc->CompleteCoherentTransfer(1101+repeat);
    auto after = Kokkos::create_mirror_view_and_copy(HostMemSpace(),state);
    for (int m=0; m<nmb; ++m) {
      Real error_before=0, error_after=0, fixed_change=0;
      int ghosts=0;
      Real ghost_square_before=0, ghost_square_after=0;
      auto sz=size.h_view(m);
      Real volume=sz.dx1*sz.dx2*(three?sz.dx3:1);
      auto phase_at = [&](int k,int j,int i) {
        Real x=sz.x1min+(i-ind.is+.5)*sz.dx1;
        Real y=sz.x2min+(j-ind.js+.5)*sz.dx2;
        Real z=three?sz.x3min+(k-ind.ks+.5)*sz.dx3:0;
        return 6.283185307179586*(x+y/1.3+z/1.7);
      };
      for (int k=0; k<state.extent_int(2); ++k) {
        for (int j=0; j<state.extent_int(3); ++j) {
          for (int i=0; i<state.extent_int(4); ++i) {
            bool active=i>=ind.is && i<=ind.ie && j>=ind.js && j<=ind.je
                        && k>=ind.ks && k<=ind.ke;
            for (int n=0; n<PC::npcgh; ++n) {
              if (!std::isfinite(before(m,n,k,j,i)) || !std::isfinite(after(m,n,k,j,i))) {
                throw std::runtime_error("nonfinite transfer fixture state");
              }
              if (active || n<PC::I_P1) {
                fixed_change=std::max(fixed_change,std::abs(after(m,n,k,j,i)-before(m,n,k,j,i)));
                continue;
              }
              ++ghosts;
              int direction, primary;
              bool lapse=false;
              if (n<PC::I_Q1XX) {direction=n-PC::I_P1;primary=PC::I_W;}
              else if (n<PC::I_L1) {direction=(n-PC::I_Q1XX)/6;primary=PC::I_GTXX+(n-PC::I_Q1XX)%6;}
              else if (n<PC::I_B11) {direction=n-PC::I_L1;primary=PC::I_W;lapse=true;}
              else {direction=(n-PC::I_B11)/3;primary=PC::I_BETAX+(n-PC::I_B11)%3;}
              int position=direction==0?i:(direction==1?j:k);
              int extent=state.extent_int(4-direction);
              long double target=0;
              if (extent>1) {
                int start=std::max(0,std::min(position-order/2,extent-order-1));
                for (int node=0; node<=order; ++node) {
                  int offset=start+node-position;
                  int ii=i+(direction==0?offset:0), jj=j+(direction==1?offset:0);
                  int kk=k+(direction==2?offset:0);
                  long double value=before(m,primary,kk,jj,ii);
                  if (lapse) {
                    long double rho=before(m,PC::I_RHO,kk,jj,ii);
                    value=pc->opt.lapse_projection_target=="collision_factorized"
                        ? 2*(before(m,PC::I_W,k,j,i)*rho+before(m,PC::I_RHO,k,j,i)*value)
                        : 2*rho*value;
                  }
                  target+=weights[position-start][node]*value;
                }
                auto sz=size.h_view(m);
                target/=direction==0?sz.dx1:(direction==1?sz.dx2:sz.dx3);
              }
              target+=.001*(n-PC::I_P1+1)*(smooth?std::sin(phase_at(k,j,i)+.13*n):1);
              Real delta_before=before(m,n,k,j,i)-target;
              Real delta_after=after(m,n,k,j,i)-target;
              ghost_square_before+=delta_before*delta_before*volume;
              ghost_square_after+=delta_after*delta_after*volume;
              error_before=std::max(error_before,static_cast<Real>(std::abs(before(m,n,k,j,i)-target)));
              error_after=std::max(error_after,static_cast<Real>(std::abs(after(m,n,k,j,i)-target)));
            }
          }
        }
      }
      // Curl uses independently generated centered weights on active cells.
      // The reference is the analytic curl of the seeded E. The legacy
      // factorized lapse target may itself have a discrete product-rule curl.
      Real curl_before[11]={}, curl_after[11]={};
      Real curl_square_before[11]={}, curl_square_after[11]={};
      auto index=[](int family,int direction) {
        if (family==0) return PC::I_P1+direction;
        if (family<7) return PC::I_Q1XX+6*direction+family-1;
        if (family==7) return PC::I_L1+direction;
        return PC::I_B11+3*direction+family-8;
      };
      Real spacing[3]={sz.dx1,sz.dx2,sz.dx3};
      Real frequency[3]={6.283185307179586,6.283185307179586/1.3,
                        three?6.283185307179586/1.7:0};
      auto derivative=[&](const auto &data,int n,int direction,int k,int j,int i) {
        if (direction==2 && !three) return 0.0;
        long double value=0;
        for (int node=0; node<=order; ++node) {
          int offset=node-order/2;
          value+=weights[order/2][node]*data(m,n,k+(direction==2?offset:0),
                          j+(direction==1?offset:0),i+(direction==0?offset:0));
        }
        return static_cast<Real>(value/spacing[direction]);
      };
      for (int k=ind.ks; k<=ind.ke; ++k) for (int j=ind.js; j<=ind.je; ++j) {
        for (int i=ind.is; i<=ind.ie; ++i) for (int family=0; family<11; ++family) {
          for (int a=0; a<3; ++a) for (int b=a+1; b<3; ++b) {
            int na=index(family,a), nb=index(family,b);
            Real expected=smooth?.001*(
                frequency[a]*(nb-PC::I_P1+1)*std::cos(phase_at(k,j,i)+.13*nb)
               -frequency[b]*(na-PC::I_P1+1)*std::cos(phase_at(k,j,i)+.13*na)):0;
            Real cb=derivative(before,nb,a,k,j,i)-derivative(before,na,b,k,j,i)-expected;
            Real ca=derivative(after,nb,a,k,j,i)-derivative(after,na,b,k,j,i)-expected;
            curl_before[family]=std::max(curl_before[family],std::abs(cb));
            curl_after[family]=std::max(curl_after[family],std::abs(ca));
            curl_square_before[family]+=cb*cb*volume;
            curl_square_after[family]+=ca*ca*volume;
          }
        }
      }
      out << "{\"repeat\":" << repeat << ",\"block\":" << mesh->pmb_pack->pmb->mb_gid.h_view(m)
          << ",\"level\":" << mesh->pmb_pack->pmb->mb_lev.h_view(m)
          << ",\"ghost_components\":" << ghosts << ",\"fixed_change\":" << fixed_change
          << ",\"profile\":\"" << profile << "\""
          << ",\"before_reference_residual_error\":" << error_before
          << ",\"after_reference_residual_error\":" << error_after
          << ",\"ghost_component_volume\":" << ghosts*volume
          << ",\"ghost_error_square_before\":" << ghost_square_before
          << ",\"ghost_error_square_after\":" << ghost_square_after
          << ",\"active_volume\":" << ind.nx1*ind.nx2*ind.nx3*volume;
      if (!smooth) out << ",\"before_constant_residual_error\":" << error_before
                       << ",\"after_constant_residual_error\":" << error_after;
      auto array=[&](const char *name,const Real values[11]) {
        out << ",\"" << name << "\":[";
        for (int q=0; q<11; ++q) {if (q) out << ',';out << values[q];}
        out << ']';
      };
      array("curl_error_before_max",curl_before);
      array("curl_error_after_max",curl_after);
      array("curl_error_before_square",curl_square_before);
      array("curl_error_after_square",curl_square_after);
      out << "}\n";
    }
  }
}
}

void ProblemGenerator::UserProblem(ParameterInput *pin,bool restart) {
  if (restart || pin->GetInteger("time","nlim")!=0
      || pmy_mesh_->pmb_pack->ppcgh->opt.coherent_transfer != "residual_shifted") {
    throw std::runtime_error("transfer oracle requires residual_shifted, zero steps, no restart");
  }
  PcGhMinkowski(pin,restart);
  pgen_final_func=Final;
}
