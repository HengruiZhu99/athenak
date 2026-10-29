#ifndef HYPERBOLIC_REPRODUCTION_FIXED_COMPOSITE_MESH_HPP_
#define HYPERBOLIC_REPRODUCTION_FIXED_COMPOSITE_MESH_HPP_
#include "mesh/cubed_sphere/composite.hpp"
namespace hyperbolic_reproduction {
// Static spatial-resolution control. The shell map and shell cell count stay
// fixed; core2 divides the same Cartesian cube into eight equal leaf blocks.
// This descriptor does not implement dynamic AMR or field transfer.
inline cubed_sphere::CompositeMesh FixedMesh(int n,int core_subdivisions=1,int overlap=0) {
  if(core_subdivisions!=1&&core_subdivisions!=2)
    throw std::invalid_argument("fixed core control supports subdivisions1 or2");
  auto mesh=cubed_sphere::CompositeMesh(.125,{.45,.75,1},1,n,3,core_subdivisions);
  mesh.evolved_overlap=overlap;return mesh;
}
}
#endif
