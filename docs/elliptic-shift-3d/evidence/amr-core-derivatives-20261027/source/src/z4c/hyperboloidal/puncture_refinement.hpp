// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_PUNCTURE_REFINEMENT_HPP_
#define Z4C_HYPERBOLOIDAL_PUNCTURE_REFINEMENT_HPP_
#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>
#include "mesh/cubed_sphere/composite.hpp"
#include "z4c/hyperboloidal/puncture_position.hpp"

namespace z4c { namespace hyperboloidal {
struct PunctureRefinementResult {
  cubed_sphere::CompositeMesh mesh;
  size_t initial_active_cells, active_cells;
  size_t initial_cartesian_leaves, cartesian_leaves, shell_leaves;
  size_t targeted_refinements, balance_refinements, containing_leaves;
  Real minimum_spacing;
  unsigned long old_revision, new_revision;
  PuncturePosition::Stamp position_stamp;
};
namespace puncture_refinement_detail {
inline bool Contains(const cubed_sphere::CompositeBlock &b,
                     const std::array<Real,3> &x){
  if(b.face>=0)return false;
  for(int a=0;a<3;++a)if(x[a]<b.lo[a]||x[a]>b.hi[a])return false;
  return true;
}
inline bool Touches(const cubed_sphere::CompositeBlock &a,
                    const cubed_sphere::CompositeBlock &b){
  if(a.face>=0||b.face>=0)return false;
  bool boundary=false;
  for(int d=0;d<3;++d){
    if(a.hi[d]<b.lo[d]||b.hi[d]<a.lo[d])return false;
    boundary=boundary||a.hi[d]==b.lo[d]||b.hi[d]==a.lo[d];
  }
  return boundary;
}
}  // namespace puncture_refinement_detail

// Geometry-only, deterministic host operation. All failure paths affect only
// this by-value descriptor. Active fields, references, puncture kinematics and
// solver/halo plans are neither transferred nor modified. The returned revision
// requires their explicit rebuild/transfer before any evolution can resume.
inline PunctureRefinementResult RefinePunctureNeighborhood(
    cubed_sphere::CompositeMesh mesh,const PuncturePosition &point,
    int target_refinement,size_t maximum_active_cells=50000){
  using namespace puncture_refinement_detail;
  point.Validate();
  // A bounded absolute level contract also avoids integer depth overflow.
  if(target_refinement<0||target_refinement>30)
    throw std::invalid_argument("puncture refinement target must lie in [0,30]");
  if(mesh.cells<2||mesh.ghosts<1||!(mesh.half_width>0)||!std::isfinite(mesh.half_width))
    throw std::invalid_argument("invalid Cartesian refinement descriptor");
  const size_t n=static_cast<size_t>(mesh.cells),limit=std::numeric_limits<size_t>::max();
  if(n>limit/n||n*n>limit/n)throw std::overflow_error("active block cell count overflow");
  const size_t cells_per_block=n*n*n;
  if(mesh.blocks.size()>maximum_active_cells/cells_per_block)
    throw std::length_error("initial mesh exceeds puncture refinement cell budget");
  const size_t initial_cells=mesh.blocks.size()*cells_per_block;
  const unsigned long old_revision=mesh.revision;
  size_t initial_cartesian=0,containing=0,shells=0;
  for(const auto &b:mesh.blocks){
    if(b.face>=0){++shells;continue;}
    ++initial_cartesian;
    if(b.refinement<0||b.refinement>30)
      throw std::invalid_argument("invalid Cartesian refinement level");
    for(int a=0;a<3;++a)
      if(!std::isfinite(b.lo[a])||!std::isfinite(b.hi[a])||!(b.lo[a]<b.hi[a])
         ||b.lo[a]<-mesh.half_width||b.hi[a]>mesh.half_width)
        throw std::invalid_argument("invalid Cartesian refinement map bounds");
    containing+=Contains(b,point.position)?1:0;
  }
  if(!containing)throw std::invalid_argument("tracked puncture lies outside Cartesian leaves");
  // Reject overlapping Cartesian interiors instead of treating them as contacts.
  for(size_t a=0;a<mesh.blocks.size();++a)if(mesh.blocks[a].face<0)
  for(size_t b=a+1;b<mesh.blocks.size();++b)if(mesh.blocks[b].face<0){
    bool overlap=true;for(int d=0;d<3;++d)
      overlap=overlap&&std::max(mesh.blocks[a].lo[d],mesh.blocks[b].lo[d])
        <std::min(mesh.blocks[a].hi[d],mesh.blocks[b].hi[d]);
    if(overlap)throw std::invalid_argument("overlapping active Cartesian leaves");
  }
  auto refine=[&](size_t id){
    if(mesh.blocks.size()>maximum_active_cells/cells_per_block
       ||maximum_active_cells/cells_per_block-mesh.blocks.size()<7)
      throw std::length_error("puncture refinement exceeds active cell budget");
    if(mesh.revision==std::numeric_limits<unsigned long>::max())
      throw std::overflow_error("composite mesh revision exhausted");
    const auto &b=mesh.blocks[id];
    if(b.refinement>=30)throw std::invalid_argument("Cartesian balancing exceeds level30");
    for(int a=0;a<3;++a){const Real mid=b.lo[a]+(b.hi[a]-b.lo[a])/2;
      // RefineCartesian uses (lo+hi)/2, so validate that precise map too.
      const Real native_mid=(b.lo[a]+b.hi[a])/2;
      if(!std::isfinite(mid)||!std::isfinite(native_mid)||!(b.lo[a]<native_mid&&native_mid<b.hi[a]))
        throw std::overflow_error("Cartesian refinement midpoint is not representable");}
    if(id>static_cast<size_t>(std::numeric_limits<int>::max()))
      throw std::overflow_error("Cartesian refinement block index overflow");
    mesh.RefineCartesian(static_cast<int>(id));
  };
  size_t targeted=0,balanced=0;
  for(;;){size_t id=mesh.blocks.size();
    for(size_t b=0;b<mesh.blocks.size();++b)
      if(Contains(mesh.blocks[b],point.position)&&mesh.blocks[b].refinement<target_refinement){id=b;break;}
    if(id==mesh.blocks.size())break;refine(id);++targeted;
  }
  for(;;){std::vector<size_t> coarser;
    for(size_t a=0;a<mesh.blocks.size();++a)if(mesh.blocks[a].face<0)
    for(size_t b=a+1;b<mesh.blocks.size();++b)
      if(Touches(mesh.blocks[a],mesh.blocks[b])){
        const int delta=mesh.blocks[a].refinement-mesh.blocks[b].refinement;
        if(delta>1)coarser.push_back(b);else if(delta< -1)coarser.push_back(a);
      }
    if(coarser.empty())break;
    std::sort(coarser.begin(),coarser.end());coarser.erase(std::unique(coarser.begin(),coarser.end()),coarser.end());
    // Reverse original indices remain valid when a parent is erased and its
    // children appended. Sorting also makes ties reproducible.
    for(auto id=coarser.rbegin();id!=coarser.rend();++id){refine(*id);++balanced;}
  }
  size_t cartesian=0;containing=0;Real spacing=std::numeric_limits<Real>::infinity();
  for(const auto &b:mesh.blocks)if(b.face<0){++cartesian;
    for(int a=0;a<3;++a)spacing=std::min(spacing,(b.hi[a]-b.lo[a])/mesh.cells);
    if(Contains(b,point.position)){++containing;
      if(b.refinement<target_refinement)throw std::runtime_error("puncture target not reached");}}
  if(!(spacing>0)||!std::isfinite(spacing))throw std::runtime_error("invalid refined Cartesian spacing");
  const size_t active=mesh.blocks.size()*cells_per_block;const auto revision=mesh.revision;
  return {std::move(mesh),initial_cells,active,initial_cartesian,cartesian,shells,
    targeted,balanced,containing,spacing,old_revision,revision,point.Capture()};
}
}}  // namespace z4c::hyperboloidal
#endif
