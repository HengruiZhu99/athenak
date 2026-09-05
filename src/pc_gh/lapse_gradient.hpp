// Shared discrete lapse-gradient target for relaxation, projection and diagnostics.
#ifndef PC_GH_LAPSE_GRADIENT_HPP_
#define PC_GH_LAPSE_GRADIENT_HPP_
#include "utils/finite_diff.hpp"

namespace pc_gh {
// Evaluate the product at each stencil point; do not invoke a discrete Leibniz rule.
template <typename Rho, typename W>
struct LapseProduct {
  Rho rho;
  W w;
  KOKKOS_INLINE_FUNCTION
  Real operator()(int m, int k, int j, int i) const {
    return rho(m,k,j,i)*w(m,k,j,i);
  }
};

template <int FD_STENCIL, typename Rho, typename W>
KOKKOS_INLINE_FUNCTION
Real DirectLapseGradient(int d, const Real idx[], Rho rho, W w,
                         int m, int k, int j, int i) {
  LapseProduct<Rho,W> product{rho,w};
  return 2.0*Dx<FD_STENCIL>(d,idx,product,m,k,j,i);
}
}  // namespace pc_gh
#endif  // PC_GH_LAPSE_GRADIENT_HPP_
