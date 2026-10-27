// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef TST_CARTESIAN_PLANE_ATLAS_HPP_
#define TST_CARTESIAN_PLANE_ATLAS_HPP_
#include <map>
#include "interface_planar_weights.hpp"
#include "mesh/cubed_sphere/composite_difference.hpp"
namespace interface_probe {
// A geometry-only catalog of actual coplanar Cartesian leaf nodes. Neighboring
// blocks participate without recursive halo dependencies or invented AMR nodes.
// Rebuild with the grid after regridding; cache ownership is one grid instance.
class CartesianPlaneAtlas {
 public:
  using Grid=cubed_sphere::CompositeDifference;
  struct Row {std::map<size_t,Real> weights;Real condition=0,l1=0,reproduction=0;};
  const Grid& grid;
  std::array<Real,3> spacing;
  std::array<std::map<long long,std::vector<size_t>>,3> planes;
  mutable std::map<std::array<Real,4>,Row> cache;
  explicit CartesianPlaneAtlas(const Grid&g):grid(g){
    spacing.fill(1e9);
    for(const auto&b:g.mesh.blocks)if(b.face<0)for(int a=0;a<3;++a)
      spacing[a]=std::min(spacing[a],(b.hi[a]-b.lo[a])/g.n);
    const size_t per=size_t(g.n)*g.n*g.n;
    for(size_t p=0;p<g.active.size();++p)if(g.mesh.blocks[p/per].face<0)
      for(int a=0;a<3;++a)planes[a][Key(a,g.geometry[p].x[a])].push_back(p);
  }
  long long Key(int axis,Real coordinate)const{
    const Real key=(coordinate+grid.mesh.half_width)*2/spacing[axis];
    const auto integer=std::llround(key);
    if(std::abs(key-integer)>1e-8)throw std::runtime_error("noncommensurate Cartesian plane atlas");
    return integer;
  }
  const Row& Build(int axis,const Real*x)const{
    const int c=(axis+1)%3,d=(axis+2)%3;const auto normal=Key(axis,x[axis]);
    const std::array<Real,4> key{Real(axis),Real(normal),x[c],x[d]};
    const auto cached=cache.find(key);if(cached!=cache.end())return cached->second;
    const auto&ids=planes[axis].at(normal);std::vector<std::array<Real,3>> points;
    for(size_t p:ids)points.push_back({grid.geometry[p].x[c],grid.geometry[p].x[d],0});
    const Real query[3]={x[c],x[d],0};std::string failure;
    for(int count:{36,49,64,96})if(ids.size()>=size_t(count)){
      try{
        cubed_sphere::InterfacePlanarWeights fit(points,count);const auto plan=fit.Build(query);
        Row row;row.condition=plan.condition;row.l1=plan.l1;row.reproduction=plan.reproduction;
        for(size_t j=0;j<plan.ids.size();++j)row.weights[ids[plan.ids[j]]]+=plan.weights[j];
        return cache.emplace(key,std::move(row)).first->second;
      }catch(const std::exception&e){failure=e.what();}
    }
    throw std::runtime_error("Cartesian plane atlas supports exhausted: "+failure);
  }
};
}
#endif
