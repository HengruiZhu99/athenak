// Available-halo derivative target for the explicit residual_shifted experiment.
#ifndef PC_GH_TRANSFER_TARGET_HPP_
#define PC_GH_TRANSFER_TARGET_HPP_
#include "athena.hpp"
#include "pc_gh/pc_gh.hpp"
#include "pc_gh/halo_weights.hpp"

namespace pc_gh {
// Centered on every active cell. At ghost cells, shift only as needed to remain
// inside allocated, already synchronized primary ghosts. No out-of-halo reads.
template <int ORDER>
KOKKOS_INLINE_FUNCTION
Real LegacyTransferTarget(const DvceArray5D<Real> &state, int m, int n,
                          int k, int j, int i, const Real idx[3],
                          bool collision_lapse) {
  int direction, primary;
  bool lapse = false;
  if (n < PcGh::I_Q1XX) {
    direction = n-PcGh::I_P1;
    primary = PcGh::I_W;
  } else if (n < PcGh::I_L1) {
    direction = (n-PcGh::I_Q1XX)/6;
    primary = PcGh::I_GTXX+(n-PcGh::I_Q1XX)%6;
  } else if (n < PcGh::I_B11) {
    direction = n-PcGh::I_L1;
    primary = PcGh::I_W;
    lapse = true;
  } else {
    direction = (n-PcGh::I_B11)/3;
    primary = PcGh::I_BETAX+(n-PcGh::I_B11)%3;
  }
  int position = direction == 0 ? i : (direction == 1 ? j : k);
  int extent = state.extent_int(4-direction);
  if (extent == 1) return 0.0;
  int start = position-ORDER/2;
  if (start < 0) start = 0;
  if (start+ORDER >= extent) start = extent-ORDER-1;
  Real answer = 0.0;
  for (int node = 0; node <= ORDER; ++node) {
    int offset = start+node-position;
    int ii = i+(direction == 0 ? offset : 0);
    int jj = j+(direction == 1 ? offset : 0);
    int kk = k+(direction == 2 ? offset : 0);
    Real value = state(m,primary,kk,jj,ii);
    if (lapse) {
      Real rho = state(m,PcGh::I_RHO,kk,jj,ii);
      value = collision_lapse
          ? 2.0*(state(m,PcGh::I_W,k,j,i)*rho
                 + state(m,PcGh::I_RHO,k,j,i)*value)
          : 2.0*rho*value;
    }
    answer += HaloDerivativeWeight<ORDER>(position-start,node)*value;
  }
  return answer*idx[direction];
}
}  // namespace pc_gh
#endif
