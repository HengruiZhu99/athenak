#ifndef DRIVER_HIERARCHY_VERTEX_EXCHANGE_HPP_
#define DRIVER_HIERARCHY_VERTEX_EXCHANGE_HPP_
#include <cstdint>
#include <limits>
#include <map>
#include <vector>
#include "athena.hpp"
#include "driver/subcycle_hierarchy.hpp"
#include "z4c/z4c_grid.hpp"

namespace subcycling {
// Rebuild only after topology/layout changes. Each transfer reads a canonical
// ACTIVE same-level vertex that is never a destination of another transfer.
// Thus active reconciliation and ghost filling can share one device phase,
// with no race or temporary field allocation. Coarse/fine and physical ghosts
// are deliberately left for their separate stage-aware boundary providers.
class HierarchyVertexExchange {
 public:
  struct Coverage { int shared=0, ghosts=0, unavailable=0; };
  Coverage Build(const Hierarchy &tree,const z4c::Z4cGridLayout &l) {
    ready_=false; // A failed topology rebuild must not leave a usable stale plan.
    if(tree.Dimension()!=2 || l.centering!=z4c::Z4cGridCentering::vertex ||
       l.nx1<=0 || l.nx2<=0 || l.nx3!=1 || l.n3!=1 || l.ks!=0 || l.ke!=0 ||
       l.ie-l.is!=l.nx1 || l.je-l.js!=l.nx2 || l.is<0 || l.js<0 ||
       l.ie>=l.n1 || l.je>=l.n2)
      throw std::invalid_argument("invalid hierarchy vertex exchange layout");
    std::map<int,std::vector<std::array<int,6>>> records;
    Coverage coverage;
    for(int n=0;n<static_cast<int>(tree.Nodes().size());++n) {
      const auto key=tree.Nodes()[n].key;
      for(int j=0;j<l.n2;++j) for(int i=0;i<l.n1;++i) {
        // Strict interiors have no duplicate vertex and need no lookup.
        if(i>l.is && i<l.ie && j>l.js && j<l.je) continue;
        const bool active=i>=l.is && i<=l.ie && j>=l.js && j<=l.je;
        const std::int64_t gx=std::int64_t(key[1])*l.nx1+i-l.is;
        const std::int64_t gy=std::int64_t(key[2])*l.nx2+j-l.js;
        int donor=-1,di=0,dj=0;
        if(gx>=0 && gy>=0) {
          const auto bx=gx/l.nx1,by=gy/l.nx2;
          for(auto x=bx-(gx%l.nx1==0);x<=bx && donor<0;++x)
            for(auto y=by-(gy%l.nx2==0);y<=by && donor<0;++y) {
              if(x<0 || y<0 || x>std::numeric_limits<int>::max() ||
                 y>std::numeric_limits<int>::max()) continue;
              donor=tree.Find({key[0],static_cast<int>(x),static_cast<int>(y),0});
              if(donor>=0) {di=gx-x*l.nx1+l.is;dj=gy-y*l.nx2+l.js;}
            }
        }
        if(donor<0) {++coverage.unavailable;continue;}
        if(donor==n && di==i && dj==j) continue;
        records[key[0]].push_back({n,j,i,donor,dj,di});
        if(active) ++coverage.shared;else ++coverage.ghosts;
      }
    }
    transfers_.clear();
    for(const auto &entry:records) {
      DvceArray2D<int> list("hierarchy vertex transfers",entry.second.size(),6);
      auto h=Kokkos::create_mirror_view(list);
      for(int r=0;r<static_cast<int>(entry.second.size());++r)
        for(int q=0;q<6;++q) h(r,q)=entry.second[r][q];
      Kokkos::deep_copy(list,h);transfers_.emplace(entry.first,list);
    }
    nodes_=tree.Nodes().size();n1_=l.n1;n2_=l.n2;ready_=true;
    return coverage;
  }
  // All nodes within each requested level must have the same stage/time.
  // No exchange between levels occurs. Allows a synchronous coarse group.
  void Apply(const DvceArray5D<Real> &state,int minimum,int maximum) const {
    if(!ready_ || minimum<0 || maximum<minimum || state.extent_int(0)!=nodes_ ||
       state.extent_int(1)<=0 || state.extent_int(2)!=1 ||
       state.extent_int(3)!=n2_ || state.extent_int(4)!=n1_)
      throw std::invalid_argument("invalid hierarchy vertex exchange state/range");
    for(const auto &entry:transfers_) {
      if(entry.first<minimum || entry.first>maximum) continue;
      const auto list=entry.second;
      par_for("reconcile hierarchy same-level vertices",DevExeSpace(),0,list.extent_int(0)-1,
          0,state.extent_int(1)-1,KOKKOS_LAMBDA(int r,int v) {
        state(list(r,0),v,0,list(r,1),list(r,2))=
            state(list(r,3),v,0,list(r,4),list(r,5));
      });
    }
  }
 private:
  std::map<int,DvceArray2D<int>> transfers_;
  int nodes_=0,n1_=0,n2_=0;
  bool ready_=false;
};
} // namespace subcycling
#endif
