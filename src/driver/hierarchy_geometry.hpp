#ifndef DRIVER_HIERARCHY_GEOMETRY_HPP_
#define DRIVER_HIERARCHY_GEOMETRY_HPP_
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "driver/subcycle_hierarchy.hpp"
#include "z4c/z4c_grid.hpp"

namespace subcycling {
// Geometry for populated leaves and covered predictors in Hierarchy node order.
// Internal faces remain block faces, including coarse/fine interfaces. Boundary
// flags describe the physical domain only; temporal ownership is separate.
struct HierarchyGeometry {
  DualArray1D<RegionSize> sizes;
  DualArray2D<BoundaryFlag> boundaries;
  void Initialize(const Hierarchy &tree, const RegionSize &domain,
                  int root, int root_x, int root_y,
                  const z4c::Z4cGridLayout &layout,
                  const std::array<BoundaryFlag,6> &faces) {
    if(tree.Dimension()!=2 || root<0 || root_x<=0 || root_y<=0 ||
       layout.centering!=z4c::Z4cGridCentering::vertex || layout.nx1<=0 ||
       layout.nx2<=0 || layout.nx3!=1 ||
       !(domain.x1max>domain.x1min) || !(domain.x2max>domain.x2min) ||
       !(domain.x3max>domain.x3min) ||
       !std::isfinite(domain.x1max-domain.x1min) ||
       !std::isfinite(domain.x2max-domain.x2min) ||
       !std::isfinite(domain.x3max-domain.x3min))
      throw std::invalid_argument("invalid VC Cartoon hierarchy geometry");
    if(faces[0]==BoundaryFlag::axis && domain.x1min!=0)
      throw std::invalid_argument("Cartoon axis must be at rho=0");
    // Validate before publishing any replacement metadata.
    for(const auto &node:tree.Nodes()) {
      const int depth=node.key[0]-root;
      if(depth<0 || depth>30) throw std::invalid_argument("unsupported geometry depth");
      const std::int64_t nx=static_cast<std::int64_t>(root_x)<<depth;
      const std::int64_t ny=static_cast<std::int64_t>(root_y)<<depth;
      if(node.key[1]>=nx || node.key[2]>=ny || node.key[3]!=0)
        throw std::invalid_argument("hierarchy node outside physical domain");
    }
    sizes=DualArray1D<RegionSize>("hierarchy geometry",tree.Nodes().size());
    boundaries=DualArray2D<BoundaryFlag>("hierarchy boundary flags",tree.Nodes().size(),6);
    for(int n=0;n<static_cast<int>(tree.Nodes().size());++n) {
      const auto key=tree.Nodes()[n].key;const int depth=key[0]-root;
      const std::int64_t nx=static_cast<std::int64_t>(root_x)<<depth;
      const std::int64_t ny=static_cast<std::int64_t>(root_y)<<depth;
      auto &s=sizes.h_view(n);
      const Real wx=(domain.x1max-domain.x1min)/nx;
      const Real wy=(domain.x2max-domain.x2min)/ny;
      s.x1min=domain.x1min+wx*key[1];
      s.x1max=key[1]+1==nx ? domain.x1max : domain.x1min+wx*(key[1]+1);
      s.x2min=domain.x2min+wy*key[2];
      s.x2max=key[2]+1==ny ? domain.x2max : domain.x2min+wy*(key[2]+1);
      s.x3min=domain.x3min;s.x3max=domain.x3max;
      s.dx1=wx/layout.nx1;s.dx2=wy/layout.nx2;s.dx3=domain.x3max-domain.x3min;
      for(int f=0;f<6;++f) boundaries.h_view(n,f)=BoundaryFlag::block;
      if(key[1]==0) boundaries.h_view(n,0)=faces[0];
      if(key[1]+1==nx) boundaries.h_view(n,1)=faces[1];
      if(key[2]==0) boundaries.h_view(n,2)=faces[2];
      if(key[2]+1==ny) boundaries.h_view(n,3)=faces[3];
      boundaries.h_view(n,4)=faces[4];boundaries.h_view(n,5)=faces[5];
    }
    sizes.modify_host();sizes.sync_device();
    boundaries.modify_host();boundaries.sync_device();
  }
};
} // namespace subcycling
#endif
