#ifndef HYPERBOLIC_REPRODUCTION_FIXED_COMPOSITE_MESH_HPP_
#define HYPERBOLIC_REPRODUCTION_FIXED_COMPOSITE_MESH_HPP_
#include "mesh/cubed_sphere/composite.hpp"
#ifndef HYPERBOLIC_SIX_SHELLS
#define HYPERBOLIC_SIX_SHELLS 0
#endif
#ifndef HYPERBOLIC_REFINED_CORE
#define HYPERBOLIC_REFINED_CORE 0
#endif
namespace hyperbolic_reproduction {
inline constexpr bool refined_core=HYPERBOLIC_REFINED_CORE;
inline constexpr int continuous_core_subdivisions=refined_core?4:2;
inline constexpr bool six_shells=HYPERBOLIC_SIX_SHELLS;
inline constexpr int shell_layers=six_shells?6:3;
// Static spatial-resolution control. The shell map and shell cell count stay
// fixed; core2 divides the same Cartesian cube into eight equal leaf blocks.
// This descriptor does not implement dynamic AMR or field transfer.
inline cubed_sphere::CompositeMesh FixedMesh(int n,int core_subdivisions=1,int overlap=0) {
  if(core_subdivisions!=1&&core_subdivisions!=2)
    throw std::invalid_argument("fixed core control supports subdivisions1 or2");
  auto mesh=cubed_sphere::CompositeMesh(.125,{.45,.75,1},1,n,3,core_subdivisions);
  mesh.evolved_overlap=overlap;return mesh;
}
// V10 research geometry: the core sets the initial characteristic CFL at n16.
inline cubed_sphere::CompositeMesh SphericalOverlapMesh(int n) {
  // Bisect each original spherical layer; keep the original radial interfaces.
  auto mesh=six_shells?cubed_sphere::CompositeMesh(.18,{.305,.45,.6,.75,.875,1},1,n,3,continuous_core_subdivisions,.16)
                       :cubed_sphere::CompositeMesh(.18,{.45,.75,1},1,n,3,2);
  mesh.evolved_overlap=3;mesh.spherical_inner_boundary=true;
  mesh.equiangular=true;mesh.inner_radius=.16;return mesh;
}
inline cubed_sphere::CompositeMesh ContinuousCoreMesh(int n) {
  auto mesh=SphericalOverlapMesh(n);mesh.continuous_cartesian_core=true;return mesh;
}
}
#endif
