// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <ostream>
#include <string>
#include "mesh/mesh.hpp"
#include "mesh/cubed_sphere/extension.hpp"
#include "z4c/z4c.hpp"

namespace cubed_sphere {
// Final-state diagnostic only: no field or interpolation plan is modified.
// Subsets overlap deliberately. Radial bins partition each region; shell seam,
// inner-interface and outer-boundary collars do not form a disjoint partition.
void Extension::ReportRegions(MeshBlockPack *pack,const char *region,
                              std::ostream &out) const {
  using Z=z4c::Z4c;
  struct Sum {
    int cells=0;
    Real volume=0,h2=0,m2=0,z2=0,w2=0,hmax=0,x[3]{};
    Real alpha=std::numeric_limits<Real>::infinity();
    Real chi=std::numeric_limits<Real>::infinity();
  };
  std::map<std::string,Sum> sums;
  auto con=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),pack->pz4c->u_con);
  auto weyl=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),pack->pz4c->u_weyl);
  auto u=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),pack->pz4c->u0);
  auto geometry=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                pack->pcoord->z4c_geometry);
  const auto ind=pack->pmesh->mb_indcs;
  const auto sizes=pack->pmb->mb_size.h_view;
  const bool shell=geometry.extent(0)!=0;
  for (int m=0;m<pack->nmb_thispack;++m)
  for (int k=ind.ks;k<=ind.ke;++k) for (int j=ind.js;j<=ind.je;++j)
  for (int i=ind.is;i<=ind.ie;++i) {
    auto s=sizes(m);
    Real x[3]={s.x1min+(i-ind.is+0.5)*s.dx1,s.x2min+(j-ind.js+0.5)*s.dx2,
               s.x3min+(k-ind.ks+0.5)*s.dx3};
    Real volume=s.dx1*s.dx2*s.dx3;
    if (shell) {
      auto p=geometry(m,k,j,i);
      for (int a=0;a<3;++a) x[a]=p.x[a];
      const Real det=p.J[0][0]*(p.J[1][1]*p.J[2][2]-p.J[1][2]*p.J[2][1])
          -p.J[0][1]*(p.J[1][0]*p.J[2][2]-p.J[1][2]*p.J[2][0])
          +p.J[0][2]*(p.J[1][0]*p.J[2][1]-p.J[1][1]*p.J[2][0]);
      volume/=fabs(det);
    }
    auto add=[&](const std::string &name) {
      auto &sum=sums[name]; ++sum.cells; sum.volume+=volume;
      const Real h=con(m,Z::I_CON_H,k,j,i);
      sum.h2+=volume*h*h;
      sum.m2+=volume*con(m,Z::I_CON_M,k,j,i);
      sum.z2+=volume*con(m,Z::I_CON_Z,k,j,i);
      sum.w2+=volume*(SQR(weyl(m,0,k,j,i))+SQR(weyl(m,1,k,j,i)));
      sum.alpha=std::min(sum.alpha,u(m,Z::I_Z4C_ALPHA,k,j,i));
      sum.chi=std::min(sum.chi,u(m,Z::I_Z4C_CHI,k,j,i));
      if (fabs(h)>sum.hmax) {
        sum.hmax=fabs(h); for (int a=0;a<3;++a) sum.x[a]=x[a];
      }
    };
    add("all");
    const Real r=sqrt(SQR(x[0])+SQR(x[1])+SQR(x[2]));
    const int bin=std::min(15,static_cast<int>(16*r/map_.R));
    add("radial_"+std::to_string(bin));
    if (shell) {
      if (i<ind.is+2) add("inner_interface");
      if (i>ind.ie-2) add("outer_boundary");
      if (j<ind.js+2 || j>ind.je-2 || k<ind.ks+2 || k>ind.ke-2)
        add("patch_seam");
      else if (i>=ind.is+2 && i<=ind.ie-2) add("interior");
    } else {
      const Real margin=std::min({map_.L-fabs(x[0]),map_.L-fabs(x[1]),
                                  map_.L-fabs(x[2])});
      add(margin<2*std::max({s.dx1,s.dx2,s.dx3}) ? "outer_interface" : "interior");
    }
  }
  for (const auto &entry:sums) {
    const auto &v=entry.second;
    out<<region<<","<<entry.first<<","<<bulk_->pmesh->time<<","<<v.cells<<","<<v.volume
       <<","<<sqrt(v.h2/v.volume)<<","<<v.hmax<<","<<sqrt(v.m2/v.volume)
       <<","<<sqrt(v.z2/v.volume)<<","<<sqrt(v.w2/v.volume)
       <<","<<v.x[0]<<","<<v.x[1]<<","<<v.x[2]<<","<<v.alpha<<","<<v.chi<<"\n";
  }
}
}  // namespace cubed_sphere
