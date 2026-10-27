// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef TST_INJECT_REPAIRED_INTERFACE_HPP_
#define TST_INJECT_REPAIRED_INTERFACE_HPP_
#include "crossface_ray_interface_value.hpp"
namespace interface_probe {
// Standalone experiment only. Prepare and validate every replacement before
// committing any row. No active value, scri boundary row or coarse MG is changed.
// Derivative-amplification qualification is a separate required experiment.
inline size_t InjectRepairedInterface(cubed_sphere::CompositeDifference&g){
  if(g.order!=4)throw std::invalid_argument("repair probe expects fourth order");
  const size_t blocksize=size_t(g.padded)*g.padded*g.padded;
  std::set<int> stencil;
  for(int center:g.active){
    stencil.insert(center);
    for(int a=0;a<3;++a)for(int i=-2;i<=2;++i)stencil.insert(center+g.Offset(a,i));
    for(int a=0;a<3;++a)for(int b=a+1;b<3;++b)
      for(int i=-2;i<=2;++i)for(int j=-2;j<=2;++j)stencil.insert(center+g.Offset(a,i)+g.Offset(b,j));
  }
  CartesianPlaneAtlas atlas(g);
  std::map<size_t,cubed_sphere::CompositeDifference::Row> replacements;
  for(size_t rowid=0;rowid<g.halos.size();++rowid){const auto&row=g.halos[rowid];
    if(!stencil.count(row.target))continue;
    const bool receiver=g.mesh.blocks[row.target/blocksize].face<0;bool cross=false;
    for(const auto&e:row.entries)cross=cross||((g.mesh.blocks[e.point/blocksize].face<0)!=receiver);
    if(!cross)continue;
    if(row.boundary_weight!=0)throw std::runtime_error("interface replacement reached scri");
    const int m=row.target/blocksize,i=row.target%g.padded-g.ng,
       j=row.target/g.padded%g.padded-g.ng,k=row.target/(g.padded*g.padded)%g.padded-g.ng;
    const Real q[3]={(i+.5)/g.n,(j+.5)/g.n,(k+.5)/g.n};Real x[3];g.mesh.Forward(m,q,x);
    CrossFaceRayValue plan;
    if(!plan.Build(g,x,16,&atlas))throw std::runtime_error("unqualified interface row: "+plan.diagnostic.status);
    auto replacement=row;replacement.entries.clear();
    for(const auto&e:plan.weights)replacement.entries.push_back({g.active.at(e.first),e.second});
    replacements.emplace(rowid,std::move(replacement));
  }
  for(auto&entry:replacements)g.halos[entry.first]=std::move(entry.second);
  return replacements.size();
}
}
#endif
