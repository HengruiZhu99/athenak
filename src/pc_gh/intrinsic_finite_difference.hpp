#ifndef PC_GH_INTRINSIC_FINITE_DIFFERENCE_HPP_
#define PC_GH_INTRINSIC_FINITE_DIFFERENCE_HPP_

#include "athena.hpp"
#include "pc_gh/intrinsic_rhs.hpp"
#include "utils/finite_diff.hpp"

namespace pc_gh::intrinsic {
// The caller supplies synchronized stencil data and a nonnegative KO amplitude.
// Only active spatial directions are read. No projection is performed here.
template <int Stencil, typename State>
KOKKOS_INLINE_FUNCTION
void FiniteDifferenceRHS(State state, int m, int k, int j, int i,
                         const Real idx[3], int dimensions, double rate,
                         double eta, double kappa, double ko_amplitude,
                         double rhs[50]) {
  static_assert(Stencil >= 2 && Stencil <= 4, "Supported orders are 2, 4, 6");
  double u[50], du[3][50];
  for (int n = 0; n < 50; ++n) {
    u[n] = state(m, n, k, j, i);
    for (int d = 0; d < 3; ++d) {
      du[d][n] = d < dimensions ? Dx<Stencil>(d, idx, state, m, n, k, j, i) : 0;
    }
  }
  PointRHS(u, du, rate, eta, kappa, rhs);
  constexpr double normalization = (Stencil % 2 ? 1.0 : -1.0)/(1 << (2*Stencil));
  if (ko_amplitude != 0) {
    for (int n = 0; n < 50; ++n) {
      for (int d = 0; d < dimensions; ++d) {
        rhs[n] += ko_amplitude*normalization
                  *Diss<Stencil>(d, idx, state, m, n, k, j, i);
      }
    }
  }
}
}  // namespace pc_gh::intrinsic
#endif  // PC_GH_INTRINSIC_FINITE_DIFFERENCE_HPP_
