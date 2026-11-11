//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2026 the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
#ifndef UTILS_CIRCUMBINARY_ACCUMULATOR_HPP_
#define UTILS_CIRCUMBINARY_ACCUMULATOR_HPP_
#include <cmath>
#include "athena.hpp"

namespace circumbinary {
// This branch's MHD CopyCons holds u1 fixed at the beginning of the step.
// Hence the auxiliary scalar follows gam0*a + beta*dt*rate. Never sum beta.
struct QualifiedMass {
  Real mass = 0, start = 0, committed_time = 0, covered = 0, unavailable = 0;
  int committed_cycle = -1;
  Real trial = 0;
  bool interval_valid = true;
  int reason = 0;
  void Begin() {
    trial = 0;
    interval_valid = true;
    reason = 0;
  }
  void Stage(Real gam0, Real beta_dt, Real rate, bool valid, int why) {
    interval_valid &= valid && std::isfinite(rate);
    reason |= why;
    trial = gam0 * trial + (valid && std::isfinite(rate) ? beta_dt * rate : 0.0);
  }
  void Commit(int cycle, Real end, Real dt) {
    if (cycle <= committed_cycle) {
      return;
    }
    // Count elapsed mesh-clock time, rather than independently summing dt at a
    // large absolute epoch (which can drift enough to reject a valid restart).
    const Real elapsed = end - committed_time;
    if (interval_valid) {
      mass += trial;
      covered += elapsed;
    } else {
      unavailable += elapsed;
    }
    (void)dt;  // The flux quadrature used the integrator dt; duration uses its clock.
    committed_cycle = cycle;
    committed_time = end;
  }
};
}  // namespace circumbinary
#endif  // UTILS_CIRCUMBINARY_ACCUMULATOR_HPP_
