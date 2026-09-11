#ifndef DRIVER_VERTEX_PARENT_STATES_HPP_
#define DRIVER_VERTEX_PARENT_STATES_HPP_
#include <limits>
#include <map>
#include <stdexcept>
#include "athena.hpp"
#include "driver/subcycle_hierarchy.hpp"
#include "z4c/z4c_grid.hpp"

namespace subcycling {
// Single-pack VC Cartoon parent storage. Initialize only from a common-time
// leaf state whose coincident vertices have already been reconciled. Active
// parent values use point injection; ghosts stay NaN until boundary filling.
// Covered parents are kept outside the physical leaf arrays and diagnostics.
class VertexParentStates {
 public:
  void Initialize(const Hierarchy &hierarchy, const DvceArray5D<Real> &leaves,
                  const z4c::Z4cGridLayout &layout) {
    if (hierarchy.Dimension() != 2 || layout.centering != z4c::Z4cGridCentering::vertex || layout.nx3 != 1 ||
        layout.nx1%2 || layout.nx2%2 || layout.nx1<2 || layout.nx2<2) {
      throw std::invalid_argument("parent injection requires even-interval VC Cartoon blocks");
    }
    const auto &nodes=hierarchy.Nodes();
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
  const DvceArray5D<Real> &Values() const { return values_; }
  const std::vector<int> &ParentNodes() const { return parent_nodes_; }
 private:
  DvceArray5D<Real> values_;
  std::vector<int> parent_nodes_;
};
}  // namespace subcycling
#endif  // DRIVER_VERTEX_PARENT_STATES_HPP_
