#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "z4c/chi_parent_provenance.hpp"

namespace {

bool Same(const Real left, const Real right) {
  return left == right && std::signbit(left) == std::signbit(right);
}

void Require(const bool condition, const char *message) {
  if (!condition) {
    std::cerr << "chi pre-update diagnostic test: " << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

}  // namespace

int main() {
  constexpr std::array<Real, 4> gamma0 = {
      0.0, 0.121098479554482, -3.843833699660025, 0.546370891121863};
  constexpr std::array<Real, 4> gamma1 = {
      1.0, 0.721781678111411, 2.121209265338722, 0.198653035682705};
  constexpr std::array<Real, 4> beta = {
      1.193743905974738, 0.099279895495783, 1.131678018054042,
      0.310665766509336};
  constexpr std::array<Real, 4> delta = {
      1.0, 0.217683334308543, 1.065841341361089, 0.0};
  constexpr Real dt = 0.0005859375;
  constexpr Real old = 0.35710902688047425;
  constexpr Real accumulator = 0.645;
  constexpr Real rhs = -12.5;

  for (int stage = 0; stage < 4; ++stage) {
    const auto value = z4c::EvaluateChiRKCandidate(
        gamma0[stage], gamma1[stage], beta[stage] * dt, old, accumulator, rhs);
    const Real expected_base = gamma0[stage] * old + gamma1[stage] * accumulator;
    const Real expected_increment = beta[stage] * dt * rhs;
    Require(Same(value.affine_base, expected_base),
            "affine base did not preserve production expression order");
    Require(Same(value.rhs_increment, expected_increment),
            "RHS increment did not preserve production expression order");
    Require(Same(value.candidate, expected_base + expected_increment),
            "candidate did not preserve production expression order");
  }

  Real u1 = 0.125;
  for (int stage = 1; stage < 4; ++stage) {
    const Real before = u1;
    u1 += delta[stage] * old;
    Require(Same(u1, before + delta[stage] * old),
            "CopyU accumulator identity failed");
  }

  constexpr Real adv_rho = 0.4;
  constexpr Real adv_z = -0.2;
  constexpr Real adv_y = 0.0;
  constexpr Real lie_divergence = 0.05;
  constexpr Real curvature = -0.3;
  constexpr Real ko_rho = 0.02;
  constexpr Real ko_z = -0.01;
  constexpr Real ko_y = 0.0;
  const Real production_rhs =
      ((((adv_rho + adv_z) + adv_y) + lie_divergence) + curvature + ko_rho) +
      ko_z + ko_y;
  Require(std::fabs(production_rhs + 0.04) < 8.0 * std::numeric_limits<Real>::epsilon(),
          "term-sum fixture arithmetic changed unexpectedly");

  const auto smooth_positive = z4c::EvaluateChiRKCandidate(
      0.0, 1.0, dt, 0.9, 0.9, -0.1);
  Require(smooth_positive.candidate > 0.0,
          "smooth positive fixture became nonpositive");

  const auto nonconvex = z4c::EvaluateChiRKCandidate(
      gamma0[2], gamma1[2], beta[2] * dt, 0.9, 0.5, 0.0);
  Require(old > 0.0 && accumulator > 0.0,
          "positive-input fixture is invalid");
  Require(nonconvex.affine_base < 0.0 && nonconvex.candidate < 0.0,
          "stage-3 nonconvex affine failure fixture did not cross zero");

  std::cout << "CHI_PREUPDATE_DIAGNOSTIC_UNIT_PASS\n";
  return EXIT_SUCCESS;
}
