#include <cmath>
#include <iostream>
#include <stdexcept>
#include "driver/classical_rk4.hpp"
#include "z4c/timestep_contract.hpp"

void Check(bool ok) { if (!ok) throw std::runtime_error("classical RK4 regression"); }
// Nonautonomous nonlinear equation with exact y(t)=exp(t):
// y' = y*y - exp(2t) + exp(t). Exercises both stage times and states.
double Error(int steps) {
  const double dt = 0.5/steps;
  double y = 1.0;
  for (int n=0; n<steps; ++n) {
    const double initial = y;
    double sum = 0;
    for (int stage=1; stage<=4; ++stage) {
      double t = n*dt+dt*classical_rk4::StageTime(stage);
      double rhs = y*y-std::exp(2*t)+std::exp(t);
      sum += classical_rk4::Weight(stage)*rhs;
      y = initial+dt*(stage == 4 ? sum : classical_rk4::NextFraction(stage)*rhs);
    }
  }
  return std::abs(y-std::exp(0.5));
}
int main() {
  double old = Error(8);
  for (int steps : {16,32,64}) {
    double error = Error(steps), ratio = old/error;
    Check(ratio > 14 && ratio < 18);
    std::cout << steps << " error=" << error << " ratio=" << ratio << '\n';
    old = error;
  }
  z4c::ExplicitRKMethod method;
  method.stages = 4;
  method.classical_rk4 = true;
  Check(std::abs(z4c::ExplicitRKNegativeRealStabilityRadius(method)-
                 2.785293563405282) < 1.e-12);
  std::cout << "PASS: nonlinear nonautonomous order and source stability radius\n";
}
