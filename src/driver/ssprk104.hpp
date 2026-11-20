#ifndef DRIVER_SSPRK104_HPP_
#define DRIVER_SSPRK104_HPP_
#include "athena.hpp"

// Ketcheson's optimal SSPRK(10,4), in the two-register form used by Z4c
// and Boltzmann. Before stage six, old += delta[5]*state. Its contribution
// at the final stage is 0.9*u5 - 0.5*u0, a positive sum of Euler stages.
inline void ConfigureSSPRK104(Real (&current)[10], Real (&old)[10], Real (&rhs)[10],
                              Real (&delta)[10]) {
  for (int k = 0; k < 10; ++k) {
    current[k] = 1;
    old[k] = delta[k] = 0;
    rhs[k] = 1.0 / 6;
  }
  current[4] = 0.4;
  old[4] = 0.6;
  rhs[4] = 1.0 / 15;
  delta[5] = -1.8;
  current[9] = 0.6;
  old[9] = -0.5;
  rhs[9] = 0.1;
}
#endif
