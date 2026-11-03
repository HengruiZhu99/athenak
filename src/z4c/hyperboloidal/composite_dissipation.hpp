// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_COMPOSITE_DISSIPATION_HPP_
#define Z4C_HYPERBOLOIDAL_COMPOSITE_DISSIPATION_HPP_
#include <Kokkos_Core.hpp>
#include <vector>
#ifndef HYPERBOLIC_CONSISTENT_KO
#define HYPERBOLIC_CONSISTENT_KO 0
#endif
namespace z4c::hyperboloidal {
// Q=-epsilon*n/(2^(2p))*V^-1 B_p^T W B_p, with positive line weights.
// Truncating complete lines at patch edges preserves weighted nonpositive
// energy. With p=4 it leaves O(h^3) at edges; p=5 instead gives O(h^4)
// for smooth mapped fields even where the transpose stencil is truncated.
// The taper/physical metric weights are unchanged; this is KO10 in interiors.
// Choosing p=5 does not claim interface stability of the full coupled PDE.
struct CompositeDissipation {
  static constexpr bool consistent=HYPERBOLIC_CONSISTENT_KO;
  static constexpr int order=consistent?5:4;
  static constexpr int points=order+1;
  static constexpr int normalization=1<<(2*order);
  KOKKOS_INLINE_FUNCTION static int Coefficient(int i){
    if constexpr(consistent){const int c[6]={-1,5,-10,10,-5,1};return c[i];}
    else {const int c[5]={1,-4,6,-4,1};return c[i];}
  }
  struct Term {int point[points];int target_coefficient;Real weight=0;};
  // Continue every KO8 difference through coarse/fine halos at the target
  // spacing. This agrees with the uniform-core B^T W B formula there, but
  // interpolation at a mixed-level interface is not a global adjoint: no
  // discrete energy estimate is asserted for the experimental AMR closure.
  template<class Grid> static std::vector<Term> CartesianTerms(const Grid&g,int p){
    if(g.ng<order)throw std::invalid_argument("AMR KO requires complete radius-four halos");
    std::vector<Term> out;const auto c=g.cells[p];const auto&b=g.mesh.blocks[c.block];
    for(int axis=0;axis<3;++axis)for(int target=0;target<points;++target){
      const Real h=(b.hi[axis]-b.lo[axis])/g.n;
      const Real start=g.geometry[p].x[axis]-target*h;
      const Real limit=g.mesh.half_width+(g.mesh.evolved_overlap-.5)*h;
      if(start < -limit-1e-13 || start+order*h > limit+1e-13)continue;
      Term term{};term.target_coefficient=Coefficient(target);
      for(int a=0;a<points;++a){int ijk[3]={c.i,c.j,c.k};ijk[axis]+=a-target;
        term.point[a]=g.Index(c.block,ijk[0],ijk[1],ijk[2]);
        Real r2=0;for(int d=0;d<3;++d){const Real x=g.geometry[p].x[d]+(d==axis?(a-target)*h:0);r2+=x*x;}
        const Real r=std::sqrt(r2),taper=std::pow(std::max(Real(0),1-r*r/.64),6);
        term.weight+=g.volumes[p]/(g.n*h)*taper/points;
      }out.push_back(term);
    }return out;
  }
};
}
#endif
