#ifndef DRIVER_VERTEX_PARENT_STATES_HPP_
#define DRIVER_VERTEX_PARENT_STATES_HPP_
#include <limits>
#include <algorithm>
#include <cstdint>
#include <map>
#include <stdexcept>
#include "athena.hpp"
#include "driver/subcycle_hierarchy.hpp"
#include "driver/block_batches.hpp"
#include "z4c/z4c_grid.hpp"
#include "z4c/cartoon_axis_boundary.hpp"
#include "z4c/physical_extrapolation.hpp"

namespace subcycling {
// Single-pack VC Cartoon parent storage. Initialize only from a common-time
// leaf state whose coincident vertices have already been reconciled. Active
// parent values use point injection; ghosts stay NaN until boundary filling.
// Covered parents are kept outside the physical leaf arrays and diagnostics.
class VertexParentStates {
 public:
#ifdef ATHENA_SUBCYCLE_DIAGNOSTICS
  int test_restriction_margin=0;
#endif
  void Initialize(const Hierarchy &hierarchy, const DvceArray5D<Real> &leaves,
                  const z4c::Z4cGridLayout &layout) {
    if (hierarchy.Dimension() != 2 || layout.centering != z4c::Z4cGridCentering::vertex || layout.nx3 != 1 || layout.n3 != 1 || layout.ks != layout.ke ||
        layout.nx1%2 || layout.nx2%2 || layout.nx1<2 || layout.nx2<2) {
      throw std::invalid_argument("parent injection requires even-interval VC Cartoon blocks");
    }
    if (leaves.extent_int(2)!=layout.n3 || leaves.extent_int(3)!=layout.n2 ||
        leaves.extent_int(4)!=layout.n1 || leaves.extent_int(1)<=0) {
      throw std::invalid_argument("source leaf storage does not match parent layout");
    }
    all_nodes_=false;
    restriction_levels_.clear();
    layout_key_=LayoutKey(layout);
    const auto &nodes=hierarchy.Nodes();
    keys_.clear(); leaf_ids_.clear();
    for (const auto &node : nodes) { keys_.push_back(node.key); leaf_ids_.push_back(node.source_leaf); }
    std::vector<int> source(nodes.size());
    std::map<int,std::vector<int>> levels;
    parent_nodes_.clear();
    for (int n=0; n<static_cast<int>(nodes.size()); ++n) {
      if (nodes[n].Covered()) {
        const int p=static_cast<int>(parent_nodes_.size());
        source[n]=-1-p; parent_nodes_.push_back(n);
        levels[nodes[n].key[0]].push_back(p);
      } else {
        if (nodes[n].source_leaf>=static_cast<int>(leaves.extent(0))) {
          throw std::invalid_argument("parent hierarchy requires all source leaves in pack");
        }
        source[n]=nodes[n].source_leaf;
      }
    }
    const int np=static_cast<int>(parent_nodes_.size());
    const int nv=leaves.extent_int(1);
    Kokkos::realloc(values_,np,nv,layout.n3,layout.n2,layout.n1);
    Kokkos::deep_copy(values_,std::numeric_limits<Real>::quiet_NaN());
    DvceArray2D<int> children("parent child sources",np,4);
    auto host=Kokkos::create_mirror_view(children);
    for (int p=0; p<np; ++p) {
      for (int c=0; c<4; ++c) host(p,c)=source[nodes[parent_nodes_[p]].children[c]];
    }
    Kokkos::deep_copy(children,host);
    auto values=values_;
    // Parents are ordered by level, so each level is a contiguous parent range.
    // Fine parent states must be populated before restricting into their parents.
    for (auto level=levels.rbegin(); level!=levels.rend(); ++level) {
      const int first=level->second.front(), last=level->second.back();
      const int nx=layout.nx1, ny=layout.nx2;
      const int is=layout.is, js=layout.js, ks=layout.ks;
      par_for("initialize covered VC parent",DevExeSpace(),first,last,0,nv-1,
          ks,ks,js,layout.je,is,layout.ie,
          KOKKOS_LAMBDA(int p,int v,int k,int j,int i) {
        // Choose the lower child on an exact shared face. Reconciled child
        // vertices agree; this tie rule makes the copy deterministic.
        const int cx=(i-is)>nx/2, cy=(j-js)>ny/2;
        const int fi=2*(i-is)-cx*nx+is, fj=2*(j-js)-cy*ny+js;
        const int child=children(p,cx+2*cy);
        values(p,v,k,j,i)=child>=0 ? leaves(child,v,ks,fj,fi)
                                  : values(-1-child,v,ks,fj,fi);
      });
    }
    Kokkos::fence("covered VC parent initialization complete");
  }
  // Unified populated hierarchy for the recursive stepper. Original leaf arrays
  // remain untouched until CopyLeavesTo is explicitly called at synchronization.
  // The default Initialize API above continues to allocate covered parents only.
  void InitializeAll(const Hierarchy &hierarchy,const DvceArray5D<Real> &leaves,
                     const z4c::Z4cGridLayout &layout) {
    VertexParentStates covered;
    covered.Initialize(hierarchy,leaves,layout);
    const auto &nodes=hierarchy.Nodes();const int count=nodes.size();
    std::vector<int> parent_source(count,-1);
    for(int p=0;p<static_cast<int>(covered.ParentNodes().size());++p)
      parent_source[covered.ParentNodes()[p]]=p;
    keys_.clear();leaf_ids_.clear();parent_nodes_.clear();max_leaf_id_=-1;
    std::map<int,std::vector<int>> restriction;
    for(int n=0;n<count;++n) {
      keys_.push_back(nodes[n].key);leaf_ids_.push_back(nodes[n].source_leaf);
      parent_nodes_.push_back(n);max_leaf_id_=std::max(max_leaf_id_,nodes[n].source_leaf);
      if(nodes[n].Covered()) restriction[nodes[n].key[0]].push_back(n);
    }
    restriction_levels_.clear();
    for(const auto &group : restriction) {
      DvceArray1D<int> ids("covered parent level indices",group.second.size());
      auto host=Kokkos::create_mirror_view(ids);
      for(int n=0;n<static_cast<int>(group.second.size());++n) host(n)=group.second[n];
      Kokkos::deep_copy(ids,host);restriction_levels_.emplace(group.first,ids);
    }
    layout_key_=LayoutKey(layout);
    Kokkos::realloc(values_,count,leaves.extent(1),layout.n3,layout.n2,layout.n1);
    Kokkos::deep_copy(values_,std::numeric_limits<Real>::quiet_NaN());
    all_levels_=DualArray1D<int>("stored hierarchy levels",count);
    Kokkos::realloc(all_children_,count,4);Kokkos::realloc(all_leaf_ids_,count);
    DvceArray1D<int> sources("stored hierarchy initialization sources",count);
    auto hs=Kokkos::create_mirror_view(sources);
    auto hc=Kokkos::create_mirror_view(all_children_);
    auto hl=Kokkos::create_mirror_view(all_leaf_ids_);
    for(int n=0;n<count;++n) {
      hs(n)=nodes[n].Covered() ? -1-parent_source[n] : nodes[n].source_leaf;
      hl(n)=nodes[n].source_leaf;all_levels_.h_view(n)=nodes[n].key[0];
      for(int c=0;c<4;++c) hc(n,c)=nodes[n].children[c];
    }
    all_levels_.modify_host();all_levels_.sync_device();
    Kokkos::deep_copy(sources,hs);Kokkos::deep_copy(all_children_,hc);Kokkos::deep_copy(all_leaf_ids_,hl);
    const auto values=values_,parents=covered.Values();
    par_for("initialize populated VC hierarchy",DevExeSpace(),0,count-1,
        0,leaves.extent_int(1)-1,layout.ks,layout.ke,layout.js,layout.je,layout.is,layout.ie,
        KOKKOS_LAMBDA(int n,int v,int k,int j,int i) {
      const int src=sources(n);
      values(n,v,k,j,i)=src>=0 ? leaves(src,v,k,j,i) : parents(-1-src,v,k,j,i);
    });
    Kokkos::fence("populated hierarchy initialization complete");
    all_nodes_=true;
  }
  // Call only after children and this parent reach the same physical time.
  // No allocation, reinitialization or ghost writes occur during restriction.
  void RestrictLevel(const z4c::Z4cGridLayout &layout,int level) {
    RestrictField(layout,level,values_);
  }
  void RestrictField(const z4c::Z4cGridLayout &layout,int level,const DvceArray5D<Real> &field) {
    for(int d=0;d<5;++d) if(field.extent(d)!=values_.extent(d))
      throw std::invalid_argument("restriction field shape mismatch");
    if(!all_nodes_ || LayoutKey(layout)!=layout_key_ || level<0)
      throw std::invalid_argument("restriction requires a populated common-time hierarchy");
    const auto found=restriction_levels_.find(level);
    if(found==restriction_levels_.end()) return;
    const auto ids=found->second;
    const auto values=field;const auto children=all_children_;
    const int nx=layout.nx1,ny=layout.nx2,is=layout.is,js=layout.js;
    int margin=0;
#ifdef ATHENA_SUBCYCLE_DIAGNOSTICS
    margin=test_restriction_margin;
#endif
    par_for("restrict synchronized covered parents",DevExeSpace(),0,ids.extent_int(0)-1,
        0,values.extent_int(1)-1,
        layout.ks,layout.ke,js,layout.je,is,layout.ie,
        KOKKOS_LAMBDA(int row,int v,int k,int j,int i) {
      if(i<is+margin || i>is+nx-margin || j<js+margin || j>js+ny-margin) return;
      const int n=ids(row);
      const int cx=(i-is)>nx/2,cy=(j-js)>ny/2;
      const int child=children(n,cx+2*cy);
      values(n,v,k,j,i)=values(child,v,k,2*(j-js)-cy*ny+js,2*(i-is)-cx*nx+is);
    });
  }
  void CopyLeavesTo(const z4c::Z4cGridLayout &layout,const DvceArray5D<Real> &leaves) const {
    if(!all_nodes_ || LayoutKey(layout)!=layout_key_ || leaves.extent_int(0)<=max_leaf_id_)
      throw std::invalid_argument("invalid synchronized leaf destination");
    for(int d=1;d<5;++d) if(leaves.extent(d)!=values_.extent(d))
      throw std::invalid_argument("changed synchronized leaf layout");
    const auto values=values_;const auto ids=all_leaf_ids_;
    par_for("copy synchronized hierarchy leaves",DevExeSpace(),0,values.extent_int(0)-1,
        0,values.extent_int(1)-1,layout.ks,layout.ke,layout.js,layout.je,layout.is,layout.ie,
        KOKKOS_LAMBDA(int n,int v,int k,int j,int i) {
      if(ids(n)>=0) leaves(ids(n),v,k,j,i)=values(n,v,k,j,i);
    });
  }
  void CopyLeavesFrom(const z4c::Z4cGridLayout &layout,const DvceArray5D<Real> &leaves) {
    if(!all_nodes_ || LayoutKey(layout)!=layout_key_ || leaves.extent_int(0)<=max_leaf_id_)
      throw std::invalid_argument("invalid synchronized leaf source");
    for(int d=1;d<5;++d) if(leaves.extent(d)!=values_.extent(d))
      throw std::invalid_argument("changed synchronized leaf source layout");
    const auto values=values_;const auto ids=all_leaf_ids_;
    par_for("import synchronized hierarchy leaves",DevExeSpace(),0,values.extent_int(0)-1,
        0,values.extent_int(1)-1,layout.ks,layout.ke,layout.js,layout.je,layout.is,layout.ie,
        KOKKOS_LAMBDA(int n,int v,int k,int j,int i) {
      if(ids(n)>=0) values(n,v,k,j,i)=leaves(ids(n),v,k,j,i);
    });
  }
  const DualArray1D<int> &AllLevels() const {
    if(!all_nodes_) throw std::logic_error("populated hierarchy not initialized");
    return all_levels_;
  }
  const std::vector<int> &StoredNodes() const { return parent_nodes_; }
  struct GhostCoverage { int copied=0, unavailable=0; };
  // Copies only coincident same-level ACTIVE donors. Unavailable physical or
  // coarse/fine ghosts remain NaN; they need separate boundary providers.
  GhostCoverage FillSameLevelGhosts(const Hierarchy &hierarchy,
                                    const DvceArray5D<Real> &leaves,
                                    const z4c::Z4cGridLayout &layout,
                                    int minimum_level=0, int maximum_level=std::numeric_limits<int>::max()) {
    if(minimum_level<0 || maximum_level<minimum_level) throw std::invalid_argument("invalid ghost level range");
    const auto &nodes=hierarchy.Nodes();
    if (LayoutKey(layout)!=layout_key_ || leaves.extent_int(1)!=values_.extent_int(1) ||
        leaves.extent_int(2)!=layout.n3 || leaves.extent_int(3)!=layout.n2 ||
        leaves.extent_int(4)!=layout.n1) throw std::invalid_argument("changed parent layout");
    if (nodes.size()!=keys_.size()) throw std::invalid_argument("changed parent hierarchy");
    std::vector<int> sources(nodes.size());
    for (int n=0; n<static_cast<int>(nodes.size()); ++n) {
      if (nodes[n].key!=keys_[n] || nodes[n].source_leaf!=leaf_ids_[n] ||
          nodes[n].source_leaf>=leaves.extent_int(0)) {
        throw std::invalid_argument("changed parent hierarchy");
      }
      sources[n]=nodes[n].source_leaf;
    }
    for (int p=0; p<static_cast<int>(parent_nodes_.size()); ++p) sources[parent_nodes_[p]]=-1-p;
    std::vector<std::array<int,6>> copies;
    GhostCoverage coverage;
    const int nx=layout.nx1, ny=layout.nx2;
    for (int p=0; p<static_cast<int>(parent_nodes_.size()); ++p) {
      const auto key=nodes[parent_nodes_[p]].key;
      if(key[0]<minimum_level || key[0]>maximum_level) continue;
      for (int j=0; j<layout.n2; ++j) for (int i=0; i<layout.n1; ++i) {
        if (i>=layout.is && i<=layout.ie && j>=layout.js && j<=layout.je) continue;
        const std::int64_t gx=std::int64_t(key[1])*nx+i-layout.is;
        const std::int64_t gy=std::int64_t(key[2])*ny+j-layout.js;
        int donor=-1, di=0, dj=0;
        if (gx>=0 && gy>=0) {
          const auto bx=gx/nx, by=gy/ny;
          // At a shared vertex prefer lower logical coordinates if present.
          for (auto x=bx-(gx%nx==0); x<=bx && donor<0; ++x) {
            for (auto y=by-(gy%ny==0); y<=by && donor<0; ++y) {
              if (x<0 || y<0 || x>std::numeric_limits<int>::max() ||
                  y>std::numeric_limits<int>::max()) continue;
              donor=hierarchy.Find({key[0],static_cast<int>(x),static_cast<int>(y),0});
              if (donor>=0) { di=gx-x*nx+layout.is; dj=gy-y*ny+layout.js; }
            }
          }
        }
        if (donor<0) { ++coverage.unavailable; continue; }
        copies.push_back({p,j,i,sources[donor],dj,di}); ++coverage.copied;
      }
    }
    DvceArray2D<int> transfers("same-level parent ghost copies",copies.size(),6);
    auto host=Kokkos::create_mirror_view(transfers);
    for (int c=0; c<static_cast<int>(copies.size()); ++c) {
      for (int q=0; q<6; ++q) host(c,q)=copies[c][q];
    }
    Kokkos::deep_copy(transfers,host);
    const auto values=values_; const int ks=layout.ks;
    if (!copies.empty()) {
      par_for("fill same-level parent ghosts",DevExeSpace(),0,copies.size()-1,
          0,leaves.extent_int(1)-1,KOKKOS_LAMBDA(int c,int v) {
        const int src=transfers(c,3), j=transfers(c,4), i=transfers(c,5);
        values(transfers(c,0),v,ks,transfers(c,1),transfers(c,2))=
            src>=0 ? leaves(src,v,ks,j,i) : values(-1-src,v,ks,j,i);
      });
    }
    Kokkos::fence("same-level parent ghost copies complete");
    return coverage;
  }
  // Caller must confirm that logical x=0 is the physical Cartoon axis.
  // Unavailable transverse ghost donors remain NaN after reflection.
  int FillAxisAtLogicalZero(const z4c::Z4cGridLayout &layout,
                            const std::vector<int> &parities,
                            int minimum_level=0,int maximum_level=std::numeric_limits<int>::max()) {
    if(minimum_level<0 || maximum_level<minimum_level) throw std::invalid_argument("invalid axis level range");
    if (LayoutKey(layout)!=layout_key_ || layout.is>layout.nx1 ||
        static_cast<int>(parities.size())!=values_.extent_int(1)) {
      throw std::invalid_argument("invalid parent axis layout or parity count");
    }
    for (int sign : parities) if (sign!=-1 && sign!=1) {
      throw std::invalid_argument("invalid parent axis parity sign");
    }
    std::vector<int> axis;
    for (int p=0; p<static_cast<int>(parent_nodes_.size()); ++p) {
      const auto &key=keys_[parent_nodes_[p]];
      if (key[1]==0 && key[0]>=minimum_level && key[0]<=maximum_level) axis.push_back(p);
    }
    DvceArray1D<int> parents("axis parent indices",axis.size());
    DvceArray1D<int> signs("parent axis parity",parities.size());
    auto hp=Kokkos::create_mirror_view(parents), hs=Kokkos::create_mirror_view(signs);
    for (int p=0; p<static_cast<int>(axis.size()); ++p) hp(p)=axis[p];
    for (int v=0; v<static_cast<int>(parities.size()); ++v) hs(v)=parities[v];
    Kokkos::deep_copy(parents,hp); Kokkos::deep_copy(signs,hs);
    const auto values=values_; const int start=layout.is, ks=layout.ks;
    if (!axis.empty()) {
      par_for("fill parent Cartoon axis",DevExeSpace(),0,axis.size()-1,
          0,parities.size()-1,0,layout.n2-1,KOKKOS_LAMBDA(int p,int v,int j) {
        z4c::FillCenteredAxisGhostLine<z4c::VertexCenteredZ4c>(
            values,parents(p),v,ks,j,start,start,signs(v));
      });
    }
    Kokkos::fence("parent axis ghosts complete");
    return axis.size()*layout.is*layout.n2;
  }
  // Same physical ghost extrapolation as the leaf path. Flags correspond to
  // inner/outer x1 then inner/outer x2; enable only outflow/diode/vacuum faces.
  // Run after same-level fill and axis reflection. x1 precedes x2 so physical
  // corners use already-filled radial ghosts. Missing interlevel donors stay NaN.
  template<int ORDER>
  int FillPhysicalGhosts(const z4c::Z4cGridLayout &layout, int root_level,
                         int root_x, int root_y, const std::array<bool,4> &faces,
                         int minimum_level=0,int maximum_level=std::numeric_limits<int>::max()) {
    if(minimum_level<0 || maximum_level<minimum_level) throw std::invalid_argument("invalid physical boundary level range");
    static_assert(ORDER>=2 && ORDER<=4,"supported physical extrapolation order");
    if (LayoutKey(layout)!=layout_key_ || root_level<0 || root_x<=0 || root_y<=0 ||
        layout.nx1+1<ORDER || layout.nx2+1<ORDER)
      throw std::invalid_argument("invalid parent physical boundary layout");
    std::vector<std::array<int,2>> work[2];
    int targets=0;
    for (int p=0;p<static_cast<int>(parent_nodes_.size());++p) {
      const auto &key=keys_[parent_nodes_[p]];const int depth=key[0]-root_level;
      if(key[0]<minimum_level || key[0]>maximum_level) continue;
      if (depth<0 || depth>30 || std::int64_t(root_x)*(std::int64_t(1)<<depth)>
          std::numeric_limits<int>::max() || std::int64_t(root_y)*(std::int64_t(1)<<depth)>
          std::numeric_limits<int>::max())
        throw std::invalid_argument("invalid parent physical domain extent");
      const int extents[2]={root_x*(1<<depth),root_y*(1<<depth)};
      for (int d=0;d<2;++d) {
        if (key[d+1]>=extents[d]) throw std::invalid_argument("parent outside physical domain");
        for(int side=0;side<2;++side) if (faces[2*d+side] &&
            key[d+1]==(side==0 ? 0 : extents[d]-1)) {
          work[d].push_back({p,side});
          const int start=d==0 ? layout.is : layout.js;
          const int end=d==0 ? layout.ie : layout.je;
          const int length=d==0 ? layout.n1 : layout.n2;
          targets+=(side==0 ? start : length-end-1)*(d==0 ? layout.n2 : layout.n1);
        }
      }
    }
    const auto values=values_;const int ks=layout.ks;
    for(int d=0;d<2;++d) {
      if(work[d].empty()) continue;
      DvceArray2D<int> entries("physical parent ghost work",work[d].size(),2);
      auto host=Kokkos::create_mirror_view(entries);
      for(int r=0;r<static_cast<int>(work[d].size());++r)
        for(int q=0;q<2;++q) host(r,q)=work[d][r][q];
      Kokkos::deep_copy(entries,host);
      const int start=d==0 ? layout.is : layout.js;
      const int end=d==0 ? layout.ie : layout.je;
      const int length=d==0 ? layout.n1 : layout.n2;
      const int transverse=d==0 ? layout.n2 : layout.n1;
      par_for("fill physical parent ghosts",DevExeSpace(),0,work[d].size()-1,
          0,values.extent_int(1)-1,0,transverse-1,KOKKOS_LAMBDA(int r,int v,int t) {
        const int p=entries(r,0),side=entries(r,1),edge=side==0 ? start : end;
        const int inward=side==0 ? 1 : -1;
        const int ng=side==0 ? start : length-end-1;
        for(int n=1;n<=ng;++n) {
          const int i=d==0 ? edge : t,j=d==0 ? t : edge;
          const int target=edge-inward*n;
          values(p,v,ks,d==0 ? t : target,d==0 ? target : t)=
              z4c::Extrapolate<ORDER>(values,p,v,ks,j,i,0,d==1 ? inward : 0,
                                      d==0 ? inward : 0,n);
        }
      });
      Kokkos::fence("parent physical boundary direction complete");
    }
    return targets;
  }
  const DvceArray5D<Real> &Values() const { return values_; }
  // Legacy name: in InitializeAll mode this list also contains active leaves.
  const std::vector<int> &ParentNodes() const { return parent_nodes_; }
 private:
  static std::array<int,13> LayoutKey(const z4c::Z4cGridLayout &l) {
    return {l.nx1,l.nx2,l.nx3,l.is,l.ie,l.js,l.je,l.ks,l.ke,l.n1,l.n2,l.n3,
            static_cast<int>(l.centering)};
  }
  std::map<int,DvceArray1D<int>> restriction_levels_;
  bool all_nodes_=false;
  int max_leaf_id_=-1;
  DualArray1D<int> all_levels_;
  DvceArray2D<int> all_children_;
  DvceArray1D<int> all_leaf_ids_;
  std::array<int,13> layout_key_{};
  DvceArray5D<Real> values_;
  std::vector<int> parent_nodes_;
  std::vector<BlockKey> keys_;
  std::vector<int> leaf_ids_;
};
}  // namespace subcycling
#endif  // DRIVER_VERTEX_PARENT_STATES_HPP_
