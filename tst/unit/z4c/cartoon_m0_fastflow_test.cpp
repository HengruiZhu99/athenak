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
  for (int l : {0, 2, 8, 16, 32, 64, 128}) {
    const double norm = std::sqrt((2*l+1)/(4*kPi));
    double equator = 1;
    for (int n=2; n<=l; n+=2) equator *= -(n-1.0)/n;
    auto h = z4c::M0Harmonic(l, kPi/2);
    Close(h[0], norm*equator, 1.e-13);
    Close(h[1], 0, 1.e-10);
    Close(h[2], -l*(l+1)*h[0], 1.e-9);
    h = z4c::M0Harmonic(l, 0);
    Close(h[0], norm, 1.e-13);
    Close(h[1], 0, 1.e-13);
    Close(h[2], -0.5*l*(l+1)*norm, 1.e-8);
    for (double theta : {0.01, 0.4, 1.2, 2.5, 3.13}) {
      h = z4c::M0Harmonic(l, theta);
      Close(h[2] + std::cos(theta)/std::sin(theta)*h[1] + l*(l+1)*h[0], 0, 2.e-8);
    }
  }

  z4c::M0AdmSample flat;
  flat.valid = true;
  flat.metric = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0};
  const double radius = 2.5;
  const auto sphere = z4c::EvaluateM0SurfacePoint(
      0.7, radius, 0.0, 0.0, flat);
  assert(sphere.valid);
  Close(sphere.expansion, 2.0 / radius, 2.0e-14);
  for(double pole:{0.0,kPi}) {
    const auto point=z4c::EvaluateM0SurfacePoint(pole,radius,0.0,0.0,flat);
    assert(point.valid);Close(point.expansion,2.0/radius,2.e-14);
    Close(point.ingoing_expansion,-2.0/radius,2.e-14);
  }
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
  // End-to-end production optimizer on exact data, including scales for
  // which the old fixed flow parameter made a negative radius in one step.
  for(double m : {1.0,0.1,0.01}) for(double translation : {0.0,2.3}) {
    auto exact=[m,translation](const std::vector<std::array<Real,2>> &points) {
      std::vector<z4c::M0AdmSample> samples;
      for(auto p:points) {
        const double x=p[0],z=p[1]-translation*m,r=std::hypot(x,z);
        const double psi=1+m/(2*r),value=std::pow(psi,4);
        z4c::M0AdmSample sample;sample.valid=true;
        sample.metric={value,0,0,value,0,value};
        const double dr=-2*m*std::pow(psi,3)/(r*r);
        for(int d=0;d<3;++d)for(int c:{0,3,5})
          sample.metric_derivative[d*6+c]=dr*(d==0?x/r:d==2?z/r:0);
        samples.push_back(sample);
      }
      return samples;
    };
    z4c::M0SolveOptions opt;
    for(double factor:{0.9,1.1,1.5}) {
      const auto solved=z4c::SolveM0Surface(exact,opt,"analytic",translation*m,.5*m*factor);
      assert(solved.verified);
      Close(solved.mean_radius,.5*m,1.e-5*m);
      Close(solved.area,16*kPi*m*m,1.e-7*m*m);
    }
    auto flow_only=opt;
    flow_only.newton_switch=1.e-12;
    flow_only.iterations=1000;
    const auto flowed=z4c::SolveM0Surface(exact,flow_only,"flow",translation*m,.55*m);
    assert(flowed.verified);
    opt.iterations=1;
    const auto stopped=z4c::SolveM0Surface(exact,opt,"analytic",translation*m,.8*m);
    assert(!stopped.verified);
    // Re-evaluate precisely the returned coefficients: metadata is the same iterate.
    auto again=z4c::AssessM0Surface(exact,opt.ntheta,stopped);
    Close(again.direct_residual,stopped.direct_residual,0);
    Close(again.area,stopped.area,0);
    auto good=stopped;good.verified=true;good.converged=true;
    auto failed=stopped;failed.center_z+=m;failed.coefficients[0]*=0.8;
    auto retained=z4c::PreferM0Recentered(good,failed);
    assert(retained.verified && retained.coefficients==good.coefficients);
    Close(retained.center_z,good.center_z,0);
  }
  // A coordinate stretch of Schwarzschild has an ellipsoidal, non-spherical MOTS.
  auto stretched=[](const std::vector<std::array<Real,2>> &points) {
    std::vector<z4c::M0AdmSample> values;
    const double stretch=1.3;
    for(auto p:points) {
      const double r=std::hypot(p[0],stretch*p[1]),psi=1+0.5/r;
      z4c::M0AdmSample v;v.valid=true;
      v.metric={std::pow(psi,4),0,0,std::pow(psi,4),0,stretch*stretch*std::pow(psi,4)};
      for(int d:{0,2})for(int c:{0,3,5})
        v.metric_derivative[d*6+c]=-2*std::pow(psi,3)/(r*r*r)*
          (d==0?p[0]:stretch*stretch*p[1])*(c==5?stretch*stretch:1);
      values.push_back(v);
    }
    return values;
  };
  z4c::M0SolveOptions distorted_options;
  distorted_options.lmax=20;distorted_options.ntheta=44;
  const auto distorted=z4c::SolveM0Surface(stretched,distorted_options,"ellipsoid",0,0.55);
  assert(distorted.verified);
  Close(distorted.area,16*kPi,1.e-6);

  z4c::M0SolveOptions flat_opt;
  auto flat_sampler=[flat](const std::vector<std::array<Real,2>> &p) {
    return std::vector<z4c::M0AdmSample>(p.size(),flat);
  };
  assert(!z4c::SolveM0Surface(flat_sampler,flat_opt,"flat",0,1).verified);
  auto unequal=candidates;
  unequal[1].direct_residual=1.e-8;unequal[2].direct_residual=2.e-8;
  assert(z4c::SelectM0MirrorPair(unequal,1.e-3,&plus,&minus));
  unequal[1].center_z=unequal[2].center_z=0;
  assert(!z4c::SelectM0MirrorPair(unequal,1.e-3,&plus,&minus));
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
  const auto seeds=z4c::RestoreM0Seeds(restart,4);
  assert(seeds.size()==2 && seeds[0].coefficients.size()==5);
  Close(seeds[0].center_z,2.0,0);Close(seeds[1].center_z,-2.0,0);
  Close(seeds[0].coefficients[0],2.0,0);Close(seeds[1].coefficients[0],3.0,0);
  assert(!seeds[0].converged && !seeds[0].verified);
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
