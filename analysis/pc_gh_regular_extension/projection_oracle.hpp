// Zero-step production projection oracle. Python independently differentiates
// the preserved ghost-inclusive pre-state; this code never constructs targets.
#ifndef PC_GH_PROJECTION_ORACLE_HPP_
#define PC_GH_PROJECTION_ORACLE_HPP_
namespace projection_oracle {
using PC = pc_gh::PcGh;
void Dump(Mesh *pm, const char *phase) {
  auto *pc=pm->pmb_pack->ppcgh;
  auto host=Kokkos::create_mirror_view_and_copy(HostMemSpace(),pc->u0);
  auto ind=pm->mb_indcs;
  std::ofstream out(std::string("projection-")+phase+"-rank"
                    +std::to_string(global_variable::my_rank)+".csv");
  if (!out) { std::cerr << "Cannot write projection oracle\n"; std::exit(EXIT_FAILURE); }
  out << "block,i,j,k,active,x,y,z,dx,dy,dz";
  for (int n=0;n<PC::npcgh;++n) out << ",u" << n;
  out << '\n' << std::setprecision(17);
  for (int m=0;m<pm->pmb_pack->nmb_thispack;++m) {
    auto size=pm->pmb_pack->pmb->mb_size.h_view(m);
    for (int k=0;k<host.extent_int(2);++k) for (int j=0;j<host.extent_int(3);++j)
      for (int i=0;i<host.extent_int(4);++i) {
        bool active=i>=ind.is && i<=ind.ie && j>=ind.js && j<=ind.je && k>=ind.ks && k<=ind.ke;
        out << pm->pmb_pack->pmb->mb_gid.h_view(m) << ',' << i << ',' << j << ',' << k << ',' << active
            << ',' << CellCenterX(i-ind.is,ind.nx1,size.x1min,size.x1max)
            << ',' << CellCenterX(j-ind.js,ind.nx2,size.x2min,size.x2max)
            << ',' << CellCenterX(k-ind.ks,ind.nx3,size.x3min,size.x3max)
            << ',' << size.dx1 << ',' << size.dx2 << ',' << size.dx3;
        for (int n=0;n<PC::npcgh;++n) out << ',' << host(m,n,k,j,i);
        out << '\n';
      }
  }
}
void Final(ParameterInput *pin, Mesh *pm) {
  auto *pc=pm->pmb_pack->ppcgh;
  // Exercise completed-step center refresh independently of the cached RHS view.
  if (pc->opt.reduction_follow_trackers) {
    for (std::size_t n=0;n<pc->ptracker.size();++n) {
      Real pos[3];
      for (int a=0;a<3;++a) pos[a]=pc->ptracker[n]->GetPos(a)+(a==0 ? 0.125 : 0.0);
      pc->ptracker[n]->SetPos(pos);
    }
  }
  Dump(pm,"before");
  switch (pc->opt.fd_stencil) {
    case 2: pc->ProjectReduction<2>(pm->pmb_pack); break;
    case 3: pc->ProjectReduction<3>(pm->pmb_pack); break;
    case 4: pc->ProjectReduction<4>(pm->pmb_pack); break;
    default: std::abort();
  }
  Dump(pm,"after");
}
void Initialize(ParameterInput *pin, Mesh *pm, bool restart) {
  if (restart || pin->GetInteger("time","nlim")!=0) {
    std::cerr << "Projection oracle requires zero steps, no restart\n"; std::exit(EXIT_FAILURE);
  }
  auto state=pm->pmb_pack->ppcgh->u0;
  auto sizes=pm->pmb_pack->pmb->mb_size.d_view;
  auto ind=pm->mb_indcs;
  par_for("projection oracle smooth seed",DevExeSpace(),0,pm->pmb_pack->nmb_thispack-1,
          0,state.extent_int(2)-1,0,state.extent_int(3)-1,0,state.extent_int(4)-1,
  KOKKOS_LAMBDA(int m,int k,int j,int i) {
    Real x=CellCenterX(i-ind.is,ind.nx1,sizes(m).x1min,sizes(m).x1max);
    Real y=CellCenterX(j-ind.js,ind.nx2,sizes(m).x2min,sizes(m).x2max);
    Real z=CellCenterX(k-ind.ks,ind.nx3,sizes(m).x3min,sizes(m).x3max);
    for (int n=0;n<PC::npcgh;++n) {
      state(m,n,k,j,i)=0.003*std::sin((n%5+1)*x+2*y-3*z+0.13*n);
    }
    state(m,PC::I_W,k,j,i)+=1.; state(m,PC::I_RHO,k,j,i)+=1.;
    state(m,PC::I_GTXX,k,j,i)+=1.; state(m,PC::I_GTYY,k,j,i)+=1.;
    state(m,PC::I_GTZZ,k,j,i)+=1.;
  });
}
}  // namespace projection_oracle
#endif
