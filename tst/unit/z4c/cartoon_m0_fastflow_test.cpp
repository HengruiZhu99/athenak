//========================================================================================
//! Focused analytic tests for the theta-only Cartoon FastFlow adapter.
//========================================================================================
#include <cassert>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "z4c/cartoon_m0_fastflow.hpp"
#include "z4c/cartoon_derivatives.hpp"
#include "z4c/cartoon_meridional_sampler.hpp"

namespace {
constexpr double kPi = 3.141592653589793238462643383279502884;

void Close(const double actual, const double expected, const double tolerance) {
  assert(std::isfinite(actual));
  assert(std::abs(actual - expected) <= tolerance);
}
}  // namespace

int main() {
  z4c::M0AdmSample flat;
  flat.valid = true;
  flat.metric = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0};
  const double radius = 2.5;
  const auto sphere = z4c::EvaluateM0SurfacePoint(
      0.7, radius, 0.0, 0.0, flat);
  assert(sphere.valid);
  Close(sphere.expansion, 2.0 / radius, 2.0e-14);
  Close(sphere.area_factor, radius * radius * std::sin(0.7), 2.0e-14);
  Close(sphere.spin_integrand_z, 0.0, 0.0);
  auto spinning = flat;
  spinning.curvature[1] = 1.0;  // K_XY=K_YX
  const auto spin_point = z4c::EvaluateM0SurfacePoint(
      0.7, radius, 0.0, 0.0, spinning);
  Close(spin_point.spin_integrand_z,
        radius * std::sin(0.7) * std::sin(0.7), 2.0e-14);
  double sphere_area = 0.0;
  for (const double mu : {-1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0)}) {
    const double node = std::acos(mu);
    const auto point = z4c::EvaluateM0SurfacePoint(node, radius, 0.0, 0.0, flat);
    sphere_area += z4c::kCartoonTwoPi * point.area_factor / std::sin(node);
  }
  Close(sphere_area, 4.0 * kPi * radius * radius, 2.0e-13);
  Close(z4c::M0HorizonMass(16.0 * kPi * 4.0, 0.0), 2.0, 2.0e-15);
  Close(z4c::M0HorizonMass(16.0 * kPi * 4.0, 3.0),
        std::sqrt(4.0 + 9.0 / 16.0), 2.0e-15);

  // Time-symmetric isotropic Schwarzschild has its minimal surface at r=M/2.
  const double mass = 1.0;
  const double isotropic = 0.5 * mass;
  const double theta = 1.1;
  const double x = isotropic * std::sin(theta);
  const double z = isotropic * std::cos(theta);
  const double psi = 1.0 + mass / (2.0 * isotropic);
  const double psi4 = std::pow(psi, 4);
  const double radial_derivative = -2.0 * mass * std::pow(psi, 3) /
                                   (isotropic * isotropic);
  z4c::M0AdmSample schwarzschild;
  schwarzschild.valid = true;
  schwarzschild.metric = {psi4, 0.0, 0.0, psi4, 0.0, psi4};
  for (int d = 0; d < 3; ++d) {
    const double direction = d == 0 ? x / isotropic :
                             d == 2 ? z / isotropic : 0.0;
    schwarzschild.metric_derivative[d * 6 + 0] = radial_derivative * direction;
    schwarzschild.metric_derivative[d * 6 + 3] = radial_derivative * direction;
    schwarzschild.metric_derivative[d * 6 + 5] = radial_derivative * direction;
  }
  const auto horizon = z4c::EvaluateM0SurfacePoint(
      theta, isotropic, 0.0, 0.0, schwarzschild);
  assert(horizon.valid);
  Close(horizon.expansion, 0.0, 3.0e-14);

  z4c::M0AdmSample anisotropic;
  anisotropic.valid = true;
  anisotropic.metric = {2.0, 0.2, 0.3, 3.0, 0.4, 4.0};
  anisotropic.curvature = {5.0, 0.6, 0.7, 6.0, 0.8, 7.0};
  const auto quarter_turn = z4c::RotateM0AdmSample(anisotropic, 0.5 * kPi);
  Close(quarter_turn.metric[0], 3.0, 2.0e-15);
  Close(quarter_turn.metric[3], 2.0, 2.0e-15);
  Close(quarter_turn.metric[5], 4.0, 2.0e-15);
  Close(quarter_turn.metric[1], -0.2, 2.0e-15);
  assert((z4c::DerivativeProvider<z4c::CartoonSO2, 2>::TensorParity(0, 1) == -1));
  assert((z4c::DerivativeProvider<z4c::CartoonSO2, 2>::TensorParity(0, 2) == 1));
  const auto opposite = z4c::RotateM0AdmSample(anisotropic, kPi);
  Close(opposite.metric[0], anisotropic.metric[0], 2.0e-15);
  Close(opposite.metric[1], anisotropic.metric[1], 2.0e-15);
  Close(opposite.metric[2], -anisotropic.metric[2], 2.0e-15);
  Close(opposite.metric[4], -anisotropic.metric[4], 2.0e-15);

  std::vector<z4c::M0CandidateSummary> candidates(3);
  for (auto &candidate : candidates) {
    candidate.converged = true;
    candidate.failure = "none";
    candidate.area = 9.0;
    candidate.irreducible_mass = 0.8;
    candidate.mass = 0.9;
    candidate.mean_radius = 0.8;
    candidate.minimum_radius = 0.7;
    candidate.flow_residual = 1.0e-5;
  }
  candidates[0].branch = "origin";
  candidates[0].area = 10.0;
  candidates[0].irreducible_mass = 0.9;
  candidates[0].mass = 1.0;
  candidates[0].mean_radius = 1.0;
  candidates[0].minimum_radius = 0.9;
  candidates[0].direct_residual = 2.0e-5;
  candidates[0].coefficients = {1.0, 0.0};
  candidates[1].branch = "plus";
  candidates[1].center_z = 2.0;
  candidates[1].direct_residual = 1.0e-5;
  candidates[1].coefficients = {1.0, 0.1};
  candidates[2].branch = "minus";
  candidates[2].center_z = -2.0;
  candidates[2].direct_residual = 1.0e-5;
  candidates[2].coefficients = {1.0, -0.1};
  assert(z4c::SelectM0Single(candidates) == 1);
  auto nonfinite_spin = candidates;
  nonfinite_spin[0].spin_z = std::numeric_limits<double>::quiet_NaN();
  nonfinite_spin[1].spin_z = std::numeric_limits<double>::quiet_NaN();
  nonfinite_spin[2].spin_z = std::numeric_limits<double>::quiet_NaN();
  assert(z4c::SelectM0Single(nonfinite_spin) == -1);
  auto nonfinite_coefficient = candidates;
  for (auto &candidate : nonfinite_coefficient)
    candidate.coefficients[0] = std::numeric_limits<double>::infinity();
  assert(z4c::SelectM0Single(nonfinite_coefficient) == -1);
  int plus = -1;
  int minus = -1;
  assert(z4c::SelectM0MirrorPair(candidates, 1.0e-12, &plus, &minus));
  assert(plus == 1 && minus == 2);
  candidates[2].converged = false;
  assert(!z4c::SelectM0MirrorPair(candidates, 1.0e-12, &plus, &minus));

  const std::vector<z4c::M0AxisSample> lapse_samples = {
      {true, -3.0, 0.4}, {true, 1.0, 0.5}, {true, -1.0, 0.2},
      {true, 2.0, 0.1}, {true, 3.0, 0.1}};
  double center = 0.0;
  double lapse = 0.0;
  assert(z4c::SelectM0AxisLapseMinimum(
      lapse_samples, 1, &center, &lapse));
  Close(center, 2.0, 0.0);  // equal-lapse tie chooses the nearer point
  Close(lapse, 0.1, 0.0);
  assert(z4c::SelectM0AxisLapseMinimum(
      lapse_samples, -1, &center, &lapse));
  Close(center, -1.0, 0.0);
  const std::vector<z4c::M0AxisSample> symmetric_lapse = {
      {true, -2.0, 0.1}, {true, -1.0, 0.3},
      {true, 1.0, 0.3}, {true, 2.0, 0.1}};
  assert(z4c::SelectM0AxisLapseMinimum(
      symmetric_lapse, 1, &center, &lapse));
  Close(center, 2.0, 0.0);
  assert(z4c::SelectM0AxisLapseMinimum(
      symmetric_lapse, -1, &center, &lapse));
  Close(center, -2.0, 0.0);
  assert(!z4c::SelectM0AxisLapseMinimum(
      {{false, -1.0, 0.2}, {true, 1.0, 0.2}}, -1, &center, &lapse));

  z4c::Z4cM0FastFlowRestartState restart;
  restart.surface_mode = "mirror_pair";
  restart.selected_branch = "plus_minus";
  restart.center_count = 2;
  restart.center_z0 = 2.0;
  restart.center_z1 = -2.0;
  restart.status = "accepted";
  restart.coefficients = {1.0, 0.1, 0.0, 1.0, -0.1, 0.0};
  restart.last_search_cycle = 8;
  restart.last_search_time = 9.0;
  restart.time_first_found = 7.0;
  restart.converged = true;
  std::string reason;
  assert(z4c::ValidateM0RestartState(restart, 2, &reason));
  restart.coefficients = {2.0, 0.0, 0.0, 3.0, 0.0, 0.0};
  const auto restored = z4c::RestoreM0Candidates(
      restart, 2, {1.0, 1.0}, {1.0, 0.0, 0.0, 1.0, 0.0, 0.0});
  assert(restored.size() == 2);
  Close(restored[0].center_z, 2.0, 0.0);
  Close(restored[1].center_z, -2.0, 0.0);
  Close(restored[0].minimum_radius, 2.0, 0.0);
  Close(restored[1].minimum_radius, 3.0, 0.0);
  Close(z4c::MinimumM0SelectedRadius(restored, {0, 1}), 2.0, 0.0);
  Close(z4c::M0SelectedCenterZ(restored, {0, 1}), 0.0, 0.0);
  const auto valid_restart = restart;
  restart.time_first_found = std::numeric_limits<double>::quiet_NaN();
  assert(!z4c::ValidateM0RestartState(restart, 2, &reason));
  restart = valid_restart;
  restart.time_first_found = -2.0;
  assert(!z4c::ValidateM0RestartState(restart, 2, &reason));
  restart = valid_restart;
  restart.time_first_found = -1.0;
  assert(!z4c::ValidateM0RestartState(restart, 2, &reason));
  restart = valid_restart;
  restart.time_first_found = 10.0;
  assert(!z4c::ValidateM0RestartState(restart, 2, &reason));
  restart = valid_restart;
  restart.converged = false;
  restart.center_count = 0;
  restart.selected_branch = "none";
  restart.status = "failed";
  restart.failure_code = "no_candidate";
  restart.coefficients.clear();
  assert(z4c::ValidateM0RestartState(restart, 2, &reason));
  restart = valid_restart;
  restart.coefficients.pop_back();
  assert(!z4c::ValidateM0RestartState(restart, 2, &reason));
  return 0;
}
