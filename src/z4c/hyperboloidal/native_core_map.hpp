// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_NATIVE_CORE_MAP_HPP_
#define Z4C_HYPERBOLOIDAL_NATIVE_CORE_MAP_HPP_

#include <limits>
#include "mesh/cubed_sphere/composite.hpp"

namespace z4c { namespace hyperboloidal {

// Runtime-independent adapter input: callers read actual native MeshBlock
// bounds and levels. Field storage, task scheduling and shell ownership remain
// separate. IDs refer to native pack slots, not Cartesian lexicographic order.
struct NativeCartesianBlock {
  int native=0, refinement=0;
  std::array<Real,3> lo{},hi{};
};

class NativeCartesianCoreMap {
 public:
  struct Match {int native,composite;};
  std::vector<Match> matches;

  NativeCartesianCoreMap(const cubed_sphere::CompositeMesh &mesh,
                        std::vector<NativeCartesianBlock> native,
                        int active_cells,int ghosts)
      :saved_mesh_(mesh),saved_native_(Canonical(std::move(native))) {
    if(active_cells!=mesh.cells||ghosts!=mesh.ghosts)
      throw std::invalid_argument("native/composite logical shape or halo mismatch");
    ValidateCorePartition(mesh);
    size_t core=0;for(const auto &block:mesh.blocks)core+=block.face<0;
    if(saved_native_.size()!=core)
      throw std::invalid_argument("native/composite Cartesian leaf count mismatch");
    std::vector<bool> used(mesh.blocks.size(),false);
    for(const auto &block:saved_native_){
      int owner=-1;
      for(size_t m=0;m<mesh.blocks.size();++m){const auto &candidate=mesh.blocks[m];
        if(candidate.face>=0||candidate.refinement!=block.refinement)continue;
        bool equal=true;for(int a=0;a<3;++a)
          equal=equal&&Near(block.lo[a],candidate.lo[a],mesh.half_width)
                     &&Near(block.hi[a],candidate.hi[a],mesh.half_width);
        if(equal){if(owner>=0)throw std::invalid_argument("ambiguous Cartesian leaf map");owner=m;}
      }
      if(owner<0||used[owner])throw std::invalid_argument("native physical bounds/level do not match unique Cartesian leaf");
      used[owner]=true;matches.push_back({block.native,owner});
    }
    for(size_t m=0;m<mesh.blocks.size();++m)
      if(mesh.blocks[m].face<0&&!used[m])throw std::invalid_argument("unmapped Cartesian leaf");
  }

  // Construct a STATIC leaf descriptor from actual native records. Relative
  // levels are measured from AthenaK root_level, not from pack order/count.
  // Every leaf must lie on the root-subdivision-anchored dyadic lattice. This
  // changes neither native fields nor shell ownership and never drops a leaf.
  static cubed_sphere::CompositeMesh FromNativeLeaves(Real half,
      const std::vector<Real> &radii,int face_subdivisions,int cells,int ghosts,
      int root_subdivisions,std::vector<NativeCartesianBlock> native,
      size_t maximum_cells=50000){
    if(root_subdivisions<1||root_subdivisions>64||cells<2||ghosts<1)
      throw std::invalid_argument("native static Cartesian root/logical shape");
    const auto records=Canonical(std::move(native));
    // Seed only the shell maps; avoid allocating a potentially huge fictitious
    // uniform root mesh before the actual leaf/cell budget is checked.
    auto mesh=cubed_sphere::CompositeMesh(half,radii,face_subdivisions,cells,ghosts);
    const size_t shell_blocks=mesh.blocks.size()-1;
    if(records.size()+shell_blocks>maximum_cells/size_t(cells)/cells/cells)
      throw std::invalid_argument("native static composite active-cell budget");
    std::vector<cubed_sphere::CompositeBlock> blocks;blocks.reserve(records.size()+shell_blocks);
    const Real root_width=2*half/root_subdivisions;
    for(const auto &record:records){
      if(record.refinement>20)throw std::invalid_argument("native static relative level exceeds20");
      cubed_sphere::CompositeBlock leaf;leaf.refinement=record.refinement;
      const Real width=std::ldexp(root_width,-record.refinement);
      for(int a=0;a<3;++a){
        const Real index=(record.lo[a]+half)/width;
        if(!Near(record.hi[a]-record.lo[a],width,half)||
            !std::isfinite(index)||std::abs(index-std::round(index))>2e-10)
          throw std::invalid_argument("native static leaf bounds/relative level violate root dyadic lattice");
        leaf.lo[a]=record.lo[a];leaf.hi[a]=record.hi[a];
      }
      blocks.push_back(leaf);
    }
    for(const auto &b:mesh.blocks)if(b.face>=0)blocks.push_back(b);
    mesh.blocks=std::move(blocks);
    // Reuse the full complete-volume/nonoverlap/bijection verification before
    // any hierarchy or field array is constructed.
    NativeCartesianCoreMap verified(mesh,records,cells,ghosts);
    return mesh;
  }

  // A native reorder/regrid/map change requires a newly constructed map before
  // any fields are exchanged. Reordering the input records alone is harmless;
  // changing a pack slot's physical block is not.
  void ValidateCurrent(const cubed_sphere::CompositeMesh &mesh,
                       std::vector<NativeCartesianBlock> native) const {
    const auto current=Canonical(std::move(native));
    if(mesh.cells!=saved_mesh_.cells||mesh.ghosts!=saved_mesh_.ghosts
        ||mesh.half_width!=saved_mesh_.half_width||mesh.radii!=saved_mesh_.radii
        ||mesh.face_subdivisions!=saved_mesh_.face_subdivisions
        ||mesh.revision!=saved_mesh_.revision||mesh.blocks.size()!=saved_mesh_.blocks.size()
        ||current.size()!=saved_native_.size())
      throw std::runtime_error("stale native/composite topology map");
    for(size_t m=0;m<mesh.blocks.size();++m){const auto &a=mesh.blocks[m],&b=saved_mesh_.blocks[m];
      if(a.face!=b.face||a.layer!=b.layer||a.refinement!=b.refinement)
        throw std::runtime_error("stale native/composite patch identity");
      for(int d=0;d<3;++d)if(a.lo[d]!=b.lo[d]||a.hi[d]!=b.hi[d])
        throw std::runtime_error("stale native/composite coordinate map");
    }
    for(size_t m=0;m<current.size();++m){const auto &a=current[m],&b=saved_native_[m];
      if(a.native!=b.native||a.refinement!=b.refinement||a.lo!=b.lo||a.hi!=b.hi)
        throw std::runtime_error("native pack block moved or reordered; rebuild Cartesian map");
    }
  }

 private:
  cubed_sphere::CompositeMesh saved_mesh_;
  std::vector<NativeCartesianBlock> saved_native_;
  static bool Near(Real a,Real b,Real scale){
    return std::isfinite(a)&&std::isfinite(b)
      &&std::abs(a-b)<=128*std::numeric_limits<Real>::epsilon()*scale;
  }
  static std::vector<NativeCartesianBlock> Canonical(std::vector<NativeCartesianBlock> blocks){
    if(blocks.empty())throw std::invalid_argument("empty native Cartesian pack");
    std::sort(blocks.begin(),blocks.end(),[](const auto &a,const auto &b){return a.native<b.native;});
    for(size_t m=0;m<blocks.size();++m){const auto &b=blocks[m];
      if(b.native!=static_cast<int>(m)||b.refinement<0)
        throw std::invalid_argument("native pack IDs must be complete unique slots with nonnegative relative levels");
      for(int a=0;a<3;++a)if(!std::isfinite(b.lo[a])||!std::isfinite(b.hi[a])||!(b.hi[a]>b.lo[a]))
        throw std::invalid_argument("invalid native Cartesian bounds");
    }
    return blocks;
  }
  static void ValidateCorePartition(const cubed_sphere::CompositeMesh &mesh){
    if(!(mesh.half_width>0)||!std::isfinite(mesh.half_width))
      throw std::invalid_argument("invalid composite Cartesian half width");
    long double volume=0;size_t core=0;
    for(size_t m=0;m<mesh.blocks.size();++m){const auto &a=mesh.blocks[m];if(a.face>=0)continue;
      ++core;long double v=1;
      if(a.layer!=-1||a.refinement<0)throw std::invalid_argument("invalid Cartesian patch identity");
      for(int d=0;d<3;++d){
        if(!std::isfinite(a.lo[d])||!std::isfinite(a.hi[d])||!(a.hi[d]>a.lo[d])
            ||a.lo[d]<-mesh.half_width||a.hi[d]>mesh.half_width)
          throw std::invalid_argument("invalid composite Cartesian leaf bounds");
        v*=a.hi[d]-a.lo[d];
      }
      volume+=v;
      for(size_t n=0;n<m;++n){const auto &b=mesh.blocks[n];if(b.face>=0)continue;
        bool overlap=true;for(int d=0;d<3;++d)
          overlap=overlap&&std::min(a.hi[d],b.hi[d])>std::max(a.lo[d],b.lo[d]);
        if(overlap)throw std::invalid_argument("overlapping composite Cartesian interiors");
      }
    }
    const long double side=2*mesh.half_width,expected=side*side*side;
    if(!core||std::abs(volume-expected)>256*std::numeric_limits<Real>::epsilon()*expected)
      throw std::invalid_argument("Cartesian leaves do not cover the complete inner box");
  }
};
}} // namespace z4c::hyperboloidal
#endif
