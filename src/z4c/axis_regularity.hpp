#ifndef Z4C_AXIS_REGULARITY_HPP_
#define Z4C_AXIS_REGULARITY_HPP_
#include <cmath>
#include <stdexcept>
#include "driver/block_batches.hpp"
#include "z4c/cartoon_vertex_axis.hpp"
#include "z4c/z4c_grid.hpp"
namespace z4c {
// Selected-block form of the existing lean axis gate. Excessive corrections
// abort before this RHS can be consumed by an integrator; no tolerance fallback.
inline void EnforceLocalVertexAxis(const Z4cGridLayout &layout,
    const DualArray2D<BoundaryFlag> &mb_bcs,const subcycling::BlockBatches &blocks,
    const DvceArray5D<Real> &state,Real tolerance) {
  if(!std::isfinite(tolerance) || tolerance<0)
    throw std::invalid_argument("invalid axis correction tolerance");
  const int is=layout.is;
  blocks.For4("enforce evolved vertex axis regularity lean",layout.ks,layout.ke,
      layout.js,layout.je,is,is,KOKKOS_LAMBDA(int m,int k,int j,int i) {
    if(mb_bcs.d_view(m,BoundaryFace::inner_x1)!=BoundaryFlag::axis) return;
    const VertexAxisCorrection correction=EnforceVertexAxisZ4cPoint(state,m,k,j,i);
    if(correction.nonfinite!=0 || correction.max_rel>tolerance)
      Kokkos::abort("VC axis regularity correction rejected in lean runtime");
  });
}
}  // namespace z4c
#endif  // Z4C_AXIS_REGULARITY_HPP_
