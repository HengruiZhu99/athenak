//========================================================================================
//! Focused analytic tests for the theta-only Cartoon FastFlow adapter.
//========================================================================================
// These are executable numerical checks, including in Release CTest builds.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <chrono>
#include <cstdio>
#include "pgen/z4c/kerr_puncture.hpp"
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
// Direct meridional solve on the analytic puncture exterior. The singular
// second asymptotic end is outside every interpolation stencil used here.
void CheckReflectedKerr(double spin) {
  using namespace kerr_puncture;
  const Parameters<double> parameters{1, spin, 0};
  const auto fields = [&](const std::array<double, 3> &x) {
    return Evaluate<CoordinateMap::cartesian_xyz, GaugeChoice::pre_collapsed>(
        x[0], x[1], x[2], parameters);
  };
  const double rplus = 1 + std::sqrt(1-spin*spin);
  const double radius = rplus / 4;
  for (bool folded : {false, true}) {
    auto sampler = [&](const z4c::M0Locations &points) {
      std::vector<z4c::M0AdmSample> out;
      for (const auto &p : points) {
        std::array<double, 3> x{p[0], 0, folded ? std::abs(p[1]) : p[1]};
        const auto f = fields(x);
        z4c::M0AdmSample sample;
        sample.valid = f.valid && f.physical_adm_available;
        int k = 0;
        for (int a = 0; a < 3; ++a) for (int b = a; b < 3; ++b, ++k) {
          sample.metric[k] = f.spatial_metric(a,b);
          sample.curvature[k] = f.extrinsic_curvature(a,b);
        }
        for (int d = 0; d < 3; ++d) {
          auto left = x, right = x;
          const double h = 1e-6 * radius;
          left[d] -= h;
          right[d] += h;
          const auto gl = fields(left).spatial_metric;
          const auto gr = fields(right).spatial_metric;
          k = 0;
          for (int a = 0; a < 3; ++a) for (int b = a; b < 3; ++b, ++k)
            sample.metric_derivative[6*d+k] = (gr(a,b)-gl(a,b))/(2*h);
        }
        if (folded && p[1] < 0) z4c::ReflectM0AdmSample(sample);
        out.push_back(sample);
      }
      return out;
    };
    for (double ratio : {0.8, 1.0, 1.2}) {
      std::vector<double> seed(9);
      seed[0] = std::sqrt(4*kPi)*radius*ratio;
      if (folded) seed[1] = .03;  // Restart or guess can contain an odd mode.
      const auto start = std::chrono::steady_clock::now();
      const auto result = z4c::RefineM0Surface(seed, 0, ratio == 1 ? 33 : 32, 20, 0, 2e-6, 2e-6,
                                               1e-7, sampler, folded);
      const double ms = std::chrono::duration<double,std::milli>(
          std::chrono::steady_clock::now()-start).count();
      std::printf("Kerr chi=%g folded=%d seed/rH=%g accepted=%d residual=%.3g "
                  "area=%.10g mass=%.10g J=%.10g ms=%.3f failure=%s\n",
                  spin, folded, ratio, result.converged, result.direct_residual,
                  result.area, result.mass, result.spin_z, ms, result.failure.c_str());
      assert(result.converged);
      if (folded) for (int l=1; l<9; l+=2) assert(result.coefficients[l] == 0);
      Close(result.area, 8*kPi*rplus, 2.e-6);
      Close(result.mass, 1, 1.e-6);
      Close(result.spin_z, spin, 1.e-6);
    }
  }
}
}  // namespace

int main() {
  for (double spin : {0., .95, .995}) CheckReflectedKerr(spin);
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
  for (const double pole : {0., kPi}) {
    const double hpp = .12;
    const auto tip = z4c::EvaluateM0SurfacePoint(pole, radius, 0, hpp, flat);
    assert(tip.valid);
    Close(tip.expansion, 2/radius-2*hpp/(radius*radius), 3e-14);
    Close(tip.area_factor, 0, 0);
    assert(!z4c::EvaluateM0SurfacePoint(pole, radius, .1, hpp, flat).valid);
  }
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

  // Flat spatial slices of ingoing Painleve-Gullstrand Schwarzschild.
  // r*Theta+ = 2*(1-sqrt(2M/r)). Check signed targets, area, a displaced
  // coordinate center, and independent rejection of an unresolved polar cap.
  const double center_z = 0.37;
  auto pg = [&](double rho, double world_z) {
    auto field = flat;
    const double zloc = world_z - center_z;
    const double r = std::hypot(rho, zloc);
    const double k = std::sqrt(2.0*mass/(r*r*r));
    const double nx = rho/r, nz = zloc/r;
    field.curvature = {k*(1 - 1.5*nx*nx), 0, -1.5*k*nx*nz, k, 0, k*(1 - 1.5*nz*nz)};
    return field;
  };
  auto batch = [](auto sample) {
    return [sample](const z4c::M0Locations &locations) {
      std::vector<z4c::M0AdmSample> fields;
      for (const auto &xz : locations) fields.push_back(sample(xz[0], xz[1]));
      return fields;
    };
  };
  std::vector<Real> seed(7, 0);
  seed[0] = 3.4*std::sqrt(4*kPi);
  seed[2] = 0.05;
  for (double target : {0.0, -0.02, 0.02}) {
    const auto surface = z4c::RefineM0Surface(seed, center_z, 16, 20, target,
                                               1.e-6, 1.e-6, 1.e-6, batch(pg));
    assert(surface.converged);
    const double r = 2*mass/std::pow(1 - target/2, 2);
    Close(surface.irreducible_mass, 0.5*r, 2.e-7);
    Close(surface.minimum_radius, r, 5.e-7);
  }
  // Disjoint analytic PG neighborhoods form a mirror pair. Offset the two
  // expansion centers from the holes so the local dipole is indispensable.
  auto pair_field = [&](double rho, double z) {
    auto field = pg(rho, std::abs(z) - 5 + center_z);
    if (z < 0) z4c::ReflectM0AdmSample(field);
    return field;
  };
  std::vector<Real> pair_seed(9);
  pair_seed[0] = 2*std::sqrt(4*kPi);
  for (double center : {-4.92, 4.92}) {
    const auto component = z4c::RefineM0Surface(pair_seed,center,32,20,0,
                           1e-6,1e-6,1e-7,batch(pair_field),true);
    assert(component.converged);
    assert(component.coefficients[1]*center > 0);
    Close(component.area,16*kPi,3e-6);
  }
  auto flat_result = z4c::RefineM0Surface(seed, 0, 16, 20, 0, .03, .03, 1.e-4,
                                          batch([&](double, double) { return flat; }));
  assert(!flat_result.converged);
  auto cap = [&](double rho, double zpos) {
    auto field = pg(rho, zpos);
    if (rho < 0.02*std::abs(zpos - center_z)) field.curvature.fill(0);
    return field;
  };
  assert(!z4c::RefineM0Surface(seed, center_z, 16, 20, 0, .03, .03, 1.e-4, batch(cap)).converged);
  auto unresolved = [&](double x, double zpos) {
    auto field = pg(x, zpos);
    field.spacing = 1.1;
    return field;
  };
  assert(!z4c::RefineM0Surface(seed, center_z, 16, 20, 0, .03, .03, 1.e-4,
                               batch(unresolved)).converged);
  auto invalid = flat;
  invalid.metric[0] = -1;
  invalid.metric[3] = -1;
  assert(!z4c::EvaluateM0SurfacePoint(theta, radius, 0, 0, invalid).valid);
  Close(sphere.ingoing_expansion, -2.0/radius, 2.e-14);

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
  auto nested = candidates;
  nested[0].coefficients = {5*std::sqrt(4*kPi), 0.0};
  assert(z4c::SelectM0Single(nested) == 0);  // Inner roots have smaller residuals.
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
  auto overlapping = candidates;
  overlapping[1].center_z = 0.1;
  overlapping[2].center_z = -0.1;
  assert(!z4c::SelectM0MirrorPair(overlapping, 1.e-12, &plus, &minus));
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

  // Fresh mirror candidates must be disjoint, while the origin candidate
  // encloses both lapse minima by the configured factor and radius floor.
  const double plus_center = 8.0 / 65.0;
  const double minus_center = -plus_center;
  const double pair_radius = z4c::M0DisjointPairInitialRadius(
      1.0, 0.8, plus_center, minus_center);
  Close(pair_radius, 0.8 * plus_center, 1.0e-15);
  assert(2.0 * pair_radius < plus_center - minus_center);
  Close(z4c::M0OriginInitialRadius(
            1.0, 3.0, plus_center, minus_center),
        1.0, 0.0);
  Close(z4c::M0OriginInitialRadius(1.0, 3.0, 0.6, -0.6), 1.8, 1.0e-15);
  assert(std::isnan(z4c::M0DisjointPairInitialRadius(
      1.0, 1.0, plus_center, minus_center)));
  assert(std::isnan(z4c::M0OriginInitialRadius(1.0, 0.0,
                                               plus_center, minus_center)));

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
