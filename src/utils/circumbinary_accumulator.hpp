#ifndef UTILS_CIRCUMBINARY_ACCUMULATOR_HPP_
#define UTILS_CIRCUMBINARY_ACCUMULATOR_HPP_
#include <cmath>
#include "athena.hpp"

namespace circumbinary {
// This branch's MHD CopyCons holds u1 fixed at the beginning of the step.
// Hence the auxiliary scalar follows gam0*a + beta*dt*rate. Never sum beta.
struct QualifiedMass {
  Real mass=0, start=0, committed_time=0, covered=0, unavailable=0;
  int committed_cycle=-1;
  Real trial=0;
  bool interval_valid=true;
  int reason=0;
  void Begin() { trial=0; interval_valid=true; reason=0; }
  void Stage(Real gam0, Real beta_dt, Real rate, bool valid, int why) {
    interval_valid &= valid && std::isfinite(rate);
    reason |= why;
    trial=gam0*trial+(valid && std::isfinite(rate) ? beta_dt*rate : 0.0);
  }
  void Commit(int cycle, Real end, Real dt) {
    if (cycle<=committed_cycle) return;
    if (interval_valid) {mass+=trial; covered+=dt;} else {unavailable+=dt;}
    committed_cycle=cycle; committed_time=end;
  }
};
}  // namespace circumbinary
#endif
