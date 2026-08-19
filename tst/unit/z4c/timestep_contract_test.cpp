#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "z4c/timestep_contract.hpp"

namespace {

void Require(const bool condition, const char *message) {
  if (!condition) {
    std::cerr << "timestep contract test: " << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

z4c::ExplicitRKMethod Method(const int stages, const std::initializer_list<Real> gam0,
                             const std::initializer_list<Real> gam1,
                             const std::initializer_list<Real> beta,
                             const std::initializer_list<Real> delta = {}) {
  z4c::ExplicitRKMethod method;
  method.stages = stages;
  int index = 0;
  for (const Real value : gam0) method.gam0[index++] = value;
  index = 0;
  for (const Real value : gam1) method.gam1[index++] = value;
  index = 0;
  for (const Real value : beta) method.beta[index++] = value;
  index = 0;
  for (const Real value : delta) method.delta[index++] = value;
  return method;
}

void Near(const Real actual, const Real expected, const Real tolerance, const char *message) {
  Require(std::fabs(actual - expected) <= tolerance, message);
}

}  // namespace

int main() {
  const auto rk1 = Method(1, {0.0}, {1.0}, {1.0});
  const auto rk2 = Method(2, {0.0, 0.5}, {1.0, 0.5}, {1.0, 0.5});
  const auto rk3 = Method(3, {0.0, 0.25, 2.0 / 3.0}, {1.0, 0.75, 1.0 / 3.0},
                          {1.0, 0.25, 2.0 / 3.0});
  const auto rk4 = Method(4,
      {0.0, 0.121098479554482, -3.843833699660025, 0.546370891121863},
      {1.0, 0.721781678111411, 2.121209265338722, 0.198653035682705},
      {1.193743905974738, 0.099279895495783, 1.131678018054042, 0.310665766509336},
      {1.0, 0.217683334308543, 1.065841341361089, 0.0});
  Near(z4c::ExplicitRKNegativeRealStabilityRadius(rk1), 2.0, 1.0e-12,
       "RK1 negative-real radius mismatch");
  Near(z4c::ExplicitRKNegativeRealStabilityRadius(rk2), 2.0, 1.0e-12,
       "RK2 negative-real radius mismatch");
  Near(z4c::ExplicitRKNegativeRealStabilityRadius(rk3), 2.5127453266183286, 1.0e-12,
       "RK3 negative-real radius mismatch");
  Near(z4c::ExplicitRKNegativeRealStabilityRadius(rk4), 3.489292716341045, 1.0e-12,
       "implemented RK4 negative-real radius mismatch");

  const Real source_dt = z4c::SourceTimestepCeiling(0.8, 2.0, 4.0);
  Near(source_dt, 0.4, 1.0e-15, "source ceiling mismatch");
  Near(z4c::SourceTimestepCeiling(0.8, 2.0, 20.0), source_dt / 5.0, 1.0e-15,
       "source ceiling does not scale inversely with damping rate");
  Near(z4c::SelectZ4cTimestep(0.2, 10.0, source_dt), source_dt, 1.0e-15,
       "source cap was multiplied by spatial CFL");
  Near(z4c::SelectZ4cTimestep(0.1, 1.0, 10.0), 0.1, 1.0e-15,
       "spatial cap does not scale with ordinary CFL");

  Near(z4c::BonaMassoCoordinateSpeed(1.0, 2.0, 1.0), std::sqrt(2.0), 1.0e-15,
       "flat 1+log gauge speed mismatch");
  Near(z4c::TelegraphCoordinateSpeed(1.0, 4.0, 1.0), 2.0, 1.0e-15,
       "flat telegraph speed mismatch");
  Near(z4c::GammaDriverCoordinateSpeed(3.0, 1.0), 2.0, 1.0e-15,
       "flat Gamma-driver speed mismatch");
  const Real no_shift_speed = z4c::CoordinateCharacteristicSpeed(0.0, 1.0, 1.2, 0.0, 0.0);
  const Real advected_speed = z4c::CoordinateCharacteristicSpeed(0.8, 1.0, 1.2, 0.0, 0.0);
  Near(no_shift_speed, 1.2, 1.0e-15, "baseline coordinate speed mismatch");
  Near(advected_speed, 2.0, 1.0e-15, "shift advection not included in speed");
  Require(1.0 / advected_speed < 1.0 / no_shift_speed,
          "increasing beta did not decrease spatial timestep");
  const Real without_telegraph =
      z4c::CoordinateCharacteristicSpeed(0.0, 1.0, 1.0, 0.0, 0.0);
  const Real with_telegraph =
      z4c::CoordinateCharacteristicSpeed(0.0, 1.0, 1.0, 2.0, 0.0);
  Near(without_telegraph, 1.0, 1.0e-15, "disabled telegraph changed speed");
  Near(with_telegraph, 2.0, 1.0e-15, "telegraph wave speed was not included");
  const Real prescribed_zero =
      z4c::CoordinateCharacteristicSpeed(0.0, 1.0, 1.0, 0.0, 0.0);
  Near(prescribed_zero, 1.0, 1.0e-15,
       "prescribed-zero shift retained a Gamma-driver speed contribution");
  Require(!std::isfinite(z4c::SourceTimestepCeiling(0.9, 2.0,
              std::numeric_limits<Real>::quiet_NaN())),
          "NaN source rate was accepted");
  Require(!std::isfinite(z4c::BonaMassoCoordinateSpeed(1.0,
              std::numeric_limits<Real>::quiet_NaN(), 1.0)),
          "NaN gauge speed was accepted");

  std::cout << "Z4C_TIMESTEP_CONTRACT_UNIT_PASS\n";
  return EXIT_SUCCESS;
}
