//========================================================================================
//! \file cartoon_m0_fastflow.cpp
//! \brief Theta-only SO(2) FastFlow adapter.
//========================================================================================
#include "z4c/cartoon_m0_fastflow.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "coordinates/adm.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "utils/spherical_harm.hpp"
#include "z4c/cartoon_derivatives.hpp"
#include "z4c/cartoon_meridional_sampler.hpp"
#include "z4c/z4c.hpp"

namespace z4c {
namespace {

constexpr Real kPi = 3.141592653589793238462643383279502884;

KOKKOS_INLINE_FUNCTION
constexpr int PackedIndex(int a, int b) {
  if (a > b) {
    const int temporary = a;
    a = b;
    b = temporary;
  }
  return a == 0 ? b : (a == 1 ? b + 2 : 5);
}

#ifndef ATHENA_CARTOON_M0_MATH_ONLY
std::vector<std::pair<Real, Real>> GaussLegendre(const int count) {
  std::vector<std::pair<Real, Real>> result(count);
  for (int n = 0; n < (count + 1) / 2; ++n) {
    Real x = std::cos(kPi * (n + 0.75) / (count + 0.5));
    Real derivative = 0.0;
    for (int iteration = 0; iteration < 32; ++iteration) {
      Real previous = 1.0;
      Real current = x;
      for (int l = 2; l <= count; ++l) {
        const Real next = ((2 * l - 1) * x * current - (l - 1) * previous) / l;
        previous = current;
        current = next;
      }
      derivative = count * (previous - x * current) / (1.0 - x * x);
      const Real update = current / derivative;
      x -= update;
      if (std::abs(update) < 4.0 * std::numeric_limits<Real>::epsilon()) break;
    }
    const Real weight = 2.0 / ((1.0 - x * x) * derivative * derivative);
    result[n] = {-x, weight};
    result[count - 1 - n] = {x, weight};
  }
  return result;
}
#endif

bool FiniteSummary(const M0CandidateSummary &value) {
  return std::isfinite(value.center_z) &&
         std::isfinite(value.axis_extremum_z) &&
         std::isfinite(value.center_lapse) && std::isfinite(value.area) &&
         std::isfinite(value.irreducible_mass) && std::isfinite(value.mass) &&
         std::isfinite(value.spin_z) && std::isfinite(value.mean_radius) &&
         std::isfinite(value.minimum_radius) &&
         std::isfinite(value.direct_residual) &&
         std::isfinite(value.flow_residual) &&
         value.area > 0.0 && value.irreducible_mass > 0.0 && value.mass > 0.0 &&
         value.mean_radius > 0.0 && value.minimum_radius > 0.0 &&
         value.center_lapse >= 0.0 && value.direct_residual >= 0.0 &&
         value.flow_residual >= 0.0 &&
         !value.coefficients.empty() &&
         std::all_of(value.coefficients.begin(), value.coefficients.end(),
                     [](const Real coefficient) {
                       return std::isfinite(coefficient);
                     });
}

Real RelativeDifference(const Real left, const Real right) {
  return std::abs(left - right) /
         std::max({std::abs(left), std::abs(right), Real{1.0e-300}});
}

}  // namespace

M0AdmSample RotateM0AdmSample(const M0AdmSample &input, const Real phi) {
  if (!input.valid) return input;
  const Real c = std::cos(phi);
  const Real s = std::sin(phi);
  const Real rotation[3][3] = {{c, -s, 0.0}, {s, c, 0.0}, {0.0, 0.0, 1.0}};
  M0AdmSample output;
  output.valid = true;
  for (int a = 0; a < 3; ++a) {
    for (int b = a; b < 3; ++b) {
      Real metric = 0.0;
      Real curvature = 0.0;
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          metric += rotation[a][i] * rotation[b][j] *
                    input.metric[PackedIndex(i, j)];
          curvature += rotation[a][i] * rotation[b][j] *
                       input.curvature[PackedIndex(i, j)];
        }
      }
      output.metric[PackedIndex(a, b)] = metric;
      output.curvature[PackedIndex(a, b)] = curvature;
      for (int d = 0; d < 3; ++d) {
        Real derivative = 0.0;
        for (int k = 0; k < 3; ++k) {
          for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
              derivative += rotation[d][k] * rotation[a][i] * rotation[b][j] *
                            input.metric_derivative[k * 6 + PackedIndex(i, j)];
            }
          }
        }
        output.metric_derivative[d * 6 + PackedIndex(a, b)] = derivative;
      }
    }
  }
  return output;
}

M0SurfacePoint EvaluateM0SurfacePoint(
    const Real theta, const Real radius, const Real radius_theta,
    const Real radius_theta_theta, const M0AdmSample &sample) {
  M0SurfacePoint result;
  if (!sample.valid || !std::isfinite(radius) || radius <= 0.0) return result;
  const Real st = std::sin(theta);
  const Real ct = std::cos(theta);
  if (!(st > 0.0)) return result;
  Real g[3][3], curvature[3][3], inverse[3][3];
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      g[a][b] = sample.metric[PackedIndex(a, b)];
      curvature[a][b] = sample.curvature[PackedIndex(a, b)];
    }
  }
  const Real det = adm::SpatialDet(g[0][0], g[0][1], g[0][2], g[1][1],
                                   g[1][2], g[2][2]);
  if (!std::isfinite(det) || det <= 0.0) return result;
  adm::SpatialInv(1.0 / det, g[0][0], g[0][1], g[0][2], g[1][1], g[1][2],
                  g[2][2], &inverse[0][0], &inverse[0][1], &inverse[0][2],
                  &inverse[1][1], &inverse[1][2], &inverse[2][2]);
  inverse[1][0] = inverse[0][1];
  inverse[2][0] = inverse[0][2];
  inverse[2][1] = inverse[1][2];

  const Real x = radius * st;
  const Real z = radius * ct;
  const Real invr = 1.0 / radius;
  const Real dr[3] = {st, 0.0, ct};
  const Real dtheta[3] = {ct * invr, 0.0, -st * invr};
  Real ddr[3][3]{};
  Real ddtheta[3][3]{};
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      ddr[a][b] = ((a == b ? 1.0 : 0.0) - dr[a] * dr[b]) * invr;
    }
  }
  ddtheta[0][0] = -2.0 * x * z / std::pow(radius, 4);
  ddtheta[0][2] = ddtheta[2][0] = (x * x - z * z) / std::pow(radius, 4);
  ddtheta[1][1] = z / (x * radius * radius);
  ddtheta[2][2] = 2.0 * x * z / std::pow(radius, 4);
  Real dF[3], ddF[3][3], upper[3]{};
  for (int a = 0; a < 3; ++a) {
    dF[a] = dr[a] - radius_theta * dtheta[a];
    for (int b = 0; b < 3; ++b) {
      ddF[a][b] = ddr[a][b] - radius_theta * ddtheta[a][b] -
                  radius_theta_theta * dtheta[a] * dtheta[b];
    }
  }
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) upper[a] += inverse[a][b] * dF[b];
  Real norm = 0.0;
  for (int a = 0; a < 3; ++a) norm += dF[a] * upper[a];
  if (!std::isfinite(norm) || norm <= 0.0) return result;
  const Real invu = 1.0 / std::sqrt(norm);
  Real trace_k = 0.0, laplacian = 0.0, normal_hessian = 0.0, normal_k = 0.0;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      Real covariant_hessian = ddF[a][b];
      for (int cidx = 0; cidx < 3; ++cidx) {
        Real gamma = 0.0;
        for (int d = 0; d < 3; ++d) {
          gamma += 0.5 * inverse[cidx][d] *
              (sample.metric_derivative[a * 6 + PackedIndex(d, b)] +
               sample.metric_derivative[b * 6 + PackedIndex(d, a)] -
               sample.metric_derivative[d * 6 + PackedIndex(a, b)]);
        }
        covariant_hessian -= gamma * dF[cidx];
      }
      laplacian += inverse[a][b] * covariant_hessian;
      normal_hessian += upper[a] * upper[b] * covariant_hessian;
      normal_k += upper[a] * upper[b] * curvature[a][b];
      trace_k += inverse[a][b] * curvature[a][b];
    }
  }
  result.expansion = laplacian * invu - normal_hessian * invu * invu * invu +
                     normal_k * invu * invu - trace_k;
  result.flow_residual = result.expansion / invu;
  // At phi=0 the axial rotational Killing vector is (0,x,0).
  for (int b = 0; b < 3; ++b) {
    result.spin_integrand_z += x * upper[b] * invu * curvature[1][b];
  }
  const Real tangent_theta[3] = {radius_theta * st + radius * ct, 0.0,
                                 radius_theta * ct - radius * st};
  const Real tangent_phi[3] = {0.0, radius * st, 0.0};
  Real h11 = 0.0, h12 = 0.0, h22 = 0.0;
  for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) {
    h11 += tangent_theta[a] * tangent_theta[b] * g[a][b];
    h12 += tangent_theta[a] * tangent_phi[b] * g[a][b];
    h22 += tangent_phi[a] * tangent_phi[b] * g[a][b];
  }
  const Real det_h = h11 * h22 - h12 * h12;
  if (!std::isfinite(det_h) || det_h <= 0.0 || !std::isfinite(result.expansion)) {
    return M0SurfacePoint{};
  }
  result.area_factor = std::sqrt(det_h);
  result.valid = true;
  return result;
}

Real M0HorizonMass(const Real area, const Real spin_z) {
  if (!std::isfinite(area) || area <= 0.0 || !std::isfinite(spin_z))
    return std::numeric_limits<Real>::quiet_NaN();
  const Real irreducible_mass = std::sqrt(area / (16.0 * kPi));
  return std::sqrt(irreducible_mass * irreducible_mass +
                   0.25 * spin_z * spin_z /
                       (irreducible_mass * irreducible_mass));
}

bool SelectM0AxisLapseMinimum(const std::vector<M0AxisSample> &samples,
                              const int sign, Real *center_z, Real *lapse) {
  if ((sign != -1 && sign != 1) || center_z == nullptr || lapse == nullptr)
    return false;
  int selected = -1;
  for (int index = 0; index < static_cast<int>(samples.size()); ++index) {
    const auto &sample = samples[index];
    if (!sample.valid || !std::isfinite(sample.z) ||
        !std::isfinite(sample.lapse) || sample.lapse < 0.0 ||
        sign * sample.z <= 0.0) continue;
    if (selected < 0 || sample.lapse < samples[selected].lapse ||
        (sample.lapse == samples[selected].lapse &&
         (std::abs(sample.z) < std::abs(samples[selected].z) ||
          (std::abs(sample.z) == std::abs(samples[selected].z) &&
           index < selected)))) selected = index;
  }
  if (selected < 0) return false;
  *center_z = samples[selected].z;
  *lapse = samples[selected].lapse;
  return true;
}

Real M0OriginInitialRadius(const Real configured_radius,
                           const Real lapse_radius_factor,
                           const Real plus_center_z,
                           const Real minus_center_z) {
  if (!std::isfinite(configured_radius) || !(configured_radius > 0.0) ||
      !std::isfinite(lapse_radius_factor) || !(lapse_radius_factor > 0.0) ||
      !std::isfinite(plus_center_z) || !std::isfinite(minus_center_z) ||
      !(plus_center_z > 0.0) || !(minus_center_z < 0.0)) {
    return std::numeric_limits<Real>::quiet_NaN();
  }
  const Real lapse_radius = std::max(std::abs(plus_center_z),
                                     std::abs(minus_center_z));
  return std::max(configured_radius, lapse_radius_factor * lapse_radius);
}

Real M0DisjointPairInitialRadius(const Real configured_radius,
                                 const Real pair_fraction,
                                 const Real plus_center_z,
                                 const Real minus_center_z) {
  if (!std::isfinite(configured_radius) || !(configured_radius > 0.0) ||
      !std::isfinite(pair_fraction) || !(pair_fraction > 0.0) ||
      !(pair_fraction < 1.0) || !std::isfinite(plus_center_z) ||
      !std::isfinite(minus_center_z) || !(plus_center_z > minus_center_z)) {
    return std::numeric_limits<Real>::quiet_NaN();
  }
  const Real half_separation = 0.5 * (plus_center_z - minus_center_z);
  return std::min(configured_radius, pair_fraction * half_separation);
}

int SelectM0Single(const std::vector<M0CandidateSummary> &candidates) {
  int selected = -1;
  for (int index = 0; index < static_cast<int>(candidates.size()); ++index) {
    const auto &candidate = candidates[index];
    if (!candidate.converged || !FiniteSummary(candidate)) continue;
    if (selected < 0 || candidate.direct_residual < candidates[selected].direct_residual ||
        (candidate.direct_residual == candidates[selected].direct_residual &&
         index < selected)) selected = index;
  }
  return selected;
}

bool SelectM0MirrorPair(const std::vector<M0CandidateSummary> &candidates,
                        const Real tolerance, int *plus, int *minus) {
  *plus = -1;
  *minus = -1;
  for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
    if (!candidates[i].converged || !FiniteSummary(candidates[i])) continue;
    if (candidates[i].branch == "plus" && (*plus < 0 ||
        candidates[i].direct_residual < candidates[*plus].direct_residual)) *plus = i;
    if (candidates[i].branch == "minus" && (*minus < 0 ||
        candidates[i].direct_residual < candidates[*minus].direct_residual)) *minus = i;
  }
  if (*plus < 0 || *minus < 0) return false;
  const auto &p = candidates[*plus];
  const auto &m = candidates[*minus];
  return RelativeDifference(p.center_z, -m.center_z) <= tolerance &&
         RelativeDifference(p.area, m.area) <= tolerance &&
         RelativeDifference(p.irreducible_mass, m.irreducible_mass) <= tolerance &&
         RelativeDifference(p.mass, m.mass) <= tolerance &&
         RelativeDifference(p.mean_radius, m.mean_radius) <= tolerance &&
         RelativeDifference(p.direct_residual, m.direct_residual) <= tolerance;
}

bool ValidateM0RestartState(const Z4cM0FastFlowRestartState &state,
                            const int lmax, std::string *reason) {
  auto fail = [reason](const char *message) {
    if (reason != nullptr) *reason = message;
    return false;
  };
  if (state.schema != Z4cM0FastFlowRestartState::kCurrentSchema)
    return fail("schema");
  if (state.last_search_cycle < -1 ||
      !std::isfinite(state.last_search_time) || state.last_search_time < 0.0 ||
      !std::isfinite(state.time_first_found) ||
      (state.time_first_found != -1.0 && state.time_first_found < 0.0) ||
      (state.time_first_found >= 0.0 &&
       (state.last_search_cycle < 0 ||
        state.time_first_found > state.last_search_time)) ||
      (state.converged && state.time_first_found < 0.0))
    return fail("time metadata");
  if (state.surface_mode == "none") {
    if (!state.coefficients.empty() || state.center_count != 0 ||
        state.selected_branch != "none" || state.center_z0 != 0.0 ||
        state.center_z1 != 0.0 || state.status != "not_started" ||
        state.failure_code != "none" || state.last_search_cycle != -1 ||
        state.last_search_time != 0.0 || state.time_first_found != -1.0 ||
        state.converged)
      return fail("nonempty none state");
    return true;
  }
  if (state.surface_mode != "single" && state.surface_mode != "mirror_pair")
    return fail("surface mode");
  const int expected_centers = state.surface_mode == "single" ? 1 : 2;
  if (state.converged && state.center_count != expected_centers)
    return fail("center count");
  if (!state.converged && state.center_count != 0)
    return fail("failed state centers");
  const std::size_t expected = state.converged
      ? static_cast<std::size_t>(expected_centers * (lmax + 1)) : 0;
  if (state.coefficients.size() != expected) return fail("coefficient count");
  for (const double coefficient : state.coefficients)
    if (!std::isfinite(coefficient)) return fail("nonfinite coefficient");
  if (!std::isfinite(state.center_z0) || !std::isfinite(state.center_z1) ||
      !std::isfinite(state.last_search_time)) return fail("nonfinite metadata");
  if (state.converged && state.surface_mode == "single" &&
      state.selected_branch != "origin" && state.selected_branch != "plus" &&
      state.selected_branch != "minus") return fail("single branch");
  if (state.converged && state.surface_mode == "mirror_pair" &&
      state.selected_branch != "plus_minus") return fail("pair branch");
  return true;
}

std::vector<M0CandidateSummary> RestoreM0Candidates(
    const Z4cM0FastFlowRestartState &state, const int lmax,
    const std::vector<Real> &weights, const std::vector<Real> &y0) {
  std::string reason;
  if (!ValidateM0RestartState(state, lmax, &reason) || !state.converged)
    return {};
  const int stride = lmax + 1;
  if (y0.size() != weights.size() * static_cast<std::size_t>(stride)) return {};
  std::vector<M0CandidateSummary> candidates;
  for (int candidate_index = 0; candidate_index < state.center_count;
       ++candidate_index) {
    M0CandidateSummary candidate;
    candidate.converged = true;
    candidate.failure = "none";
    candidate.branch = state.center_count == 2
        ? (candidate_index == 0 ? "plus" : "minus") : state.selected_branch;
    candidate.center_z = candidate_index == 0 ? state.center_z0 : state.center_z1;
    const auto begin = state.coefficients.begin() + candidate_index * stride;
    candidate.coefficients.assign(begin, begin + stride);
    candidate.minimum_radius = std::numeric_limits<Real>::infinity();
    for (int n = 0; n < static_cast<int>(weights.size()); ++n) {
      Real radius = 0.0;
      for (int l = 0; l <= lmax; ++l)
        radius += candidate.coefficients[l] * y0[n * stride + l];
      candidate.minimum_radius = std::min(candidate.minimum_radius, radius);
      candidate.mean_radius += 0.5 * weights[n] * radius;
    }
    if (!std::isfinite(candidate.minimum_radius) ||
        candidate.minimum_radius <= 0.0) return {};
    candidates.push_back(std::move(candidate));
  }
  return candidates;
}

Real MinimumM0SelectedRadius(const std::vector<M0CandidateSummary> &candidates,
                             const std::vector<int> &selected) {
  Real minimum = std::numeric_limits<Real>::infinity();
  for (const int index : selected) {
    if (index < 0 || index >= static_cast<int>(candidates.size()) ||
        !std::isfinite(candidates[index].minimum_radius) ||
        candidates[index].minimum_radius <= 0.0) return -1.0;
    minimum = std::min(minimum, candidates[index].minimum_radius);
  }
  return std::isfinite(minimum) ? minimum : -1.0;
}

Real M0SelectedCenterZ(const std::vector<M0CandidateSummary> &candidates,
                       const std::vector<int> &selected) {
  if (selected.empty()) return 0.0;
  Real center = 0.0;
  for (const int index : selected) {
    if (index < 0 || index >= static_cast<int>(candidates.size()) ||
        !std::isfinite(candidates[index].center_z)) return 0.0;
    center += candidates[index].center_z;
  }
  return center / selected.size();
}

#ifndef ATHENA_CARTOON_M0_MATH_ONLY
CartoonM0FastFlow::CartoonM0FastFlow(MeshBlockPack *pack, ParameterInput *pin,
                                     const int horizon)
    : pack_(pack), pin_(pin), horizon_(horizon) {
  const std::string suffix = std::to_string(horizon);
  if (horizon != 0 || pin->GetOrAddInteger("fastflow", "num_horizons", 1) != 1) {
    throw std::runtime_error(
        "Cartoon m=0 FastFlow uses one finder containing origin/+/- candidates");
  }
  lmax_ = pin->GetOrAddInteger("fastflow", "lmax", 4);
  ntheta_ = pin->GetOrAddInteger("fastflow", "ntheta", 12);
  iterations_ = pin->GetOrAddInteger("fastflow", "flow_iterations_" + suffix, 100);
  find_interval_ = pin->GetOrAddInteger("fastflow", "find_interval_" + suffix, 1);
  start_time_ = pin->GetOrAddReal("fastflow", "start_time_" + suffix, 0.0);
  stop_time_ = pin->GetOrAddReal("fastflow", "stop_time_" + suffix, -1.0);
  initial_radius_ = pin->GetOrAddReal("fastflow", "initial_radius_" + suffix, 1.0);
  flow_scale_ = pin->GetOrAddReal("fastflow", "flow_alpha_beta_const_" + suffix, 1.0);
  hrms_tolerance_ = pin->GetOrAddReal("fastflow", "dimensionless_hrms_tol_" + suffix,
                                      3.0e-2);
  mass_tolerance_ = pin->GetOrAddReal("fastflow", "mass_relative_tol_" + suffix, 1e-4);
  direct_tolerance_ = pin->GetOrAddReal("fastflow", "cartoon_direct_residual_tol_" +
                                        suffix, 3.0e-2);
  pair_tolerance_ = pin->GetOrAddReal("fastflow", "cartoon_pair_relative_tol_" +
                                      suffix, 1e-3);
  adaptive_initial_radius_ = pin->GetOrAddBoolean(
      "fastflow", "cartoon_adaptive_initial_radius_" + suffix, true);
  origin_lapse_radius_factor_ = pin->GetOrAddReal(
      "fastflow", "cartoon_origin_lapse_radius_factor_" + suffix, 3.0);
  pair_disjoint_fraction_ = pin->GetOrAddReal(
      "fastflow", "cartoon_pair_disjoint_fraction_" + suffix, 0.8);
  center_seed_ = std::abs(pin->GetOrAddReal("fastflow", "cartoon_center_z_" + suffix,
                                            initial_radius_));
  axis_search_bound_ = pin->GetOrAddReal(
      "fastflow", "cartoon_axis_search_bound_" + suffix,
      center_seed_ > 0.0 ? center_seed_ : initial_radius_);
  axis_search_samples_ = pin->GetOrAddInteger(
      "fastflow", "cartoon_axis_search_samples_" + suffix, 33);
  mode_ = pin->GetOrAddString("fastflow", "cartoon_surface_mode_" + suffix, "single");
  if (lmax_ < 1 || ntheta_ < 2 || iterations_ < 1 || find_interval_ < 1 ||
      !(initial_radius_ > 0.0) || !(flow_scale_ > 0.0) ||
      !(hrms_tolerance_ > 0.0) || !(mass_tolerance_ > 0.0) ||
      !(direct_tolerance_ > 0.0) || !(pair_tolerance_ >= 0.0) ||
      !std::isfinite(origin_lapse_radius_factor_) ||
      !(origin_lapse_radius_factor_ > 0.0) ||
      !std::isfinite(pair_disjoint_fraction_) ||
      !(pair_disjoint_fraction_ > 0.0) || !(pair_disjoint_fraction_ < 1.0) ||
      !std::isfinite(center_seed_) || !std::isfinite(axis_search_bound_) ||
      !(axis_search_bound_ > 0.0) || axis_search_samples_ < 2 ||
      !std::isfinite(start_time_) ||
      !std::isfinite(stop_time_) ||
      (mode_ != "single" && mode_ != "mirror_pair")) {
    throw std::runtime_error("invalid Cartoon m=0 FastFlow configuration");
  }
  const auto quadrature = GaussLegendre(ntheta_);
  theta_.resize(ntheta_);
  weights_.resize(ntheta_);
  y0_.resize(ntheta_ * (lmax_ + 1));
  dy0_.resize(y0_.size());
  ddy0_.resize(y0_.size());
  for (int n = 0; n < ntheta_; ++n) {
    theta_[n] = std::acos(quadrature[n].first);
    weights_[n] = quadrature[n].second;
    for (int l = 0; l <= lmax_; ++l) {
      Real yi, dyi, dphir, dphii, ddyi, ddphir, ddphii, mixedr, mixedi;
      SphericalHarmSecondDerivs(&y0_[n * (lmax_ + 1) + l], &yi,
          &dy0_[n * (lmax_ + 1) + l], &dyi, &dphir, &dphii,
          &ddy0_[n * (lmax_ + 1) + l], &ddyi, &ddphir, &ddphii,
          &mixedr, &mixedi, l, 0, theta_[n], 0.0);
    }
  }
  Restore();
}

CartoonM0FastFlow::~CartoonM0FastFlow() {
  if (output_ != nullptr) std::fclose(output_);
}

bool CartoonM0FastFlow::ShouldSearch(const int cycle, const Real time) const {
  return cycle >= 0 && cycle % find_interval_ == 0 && time >= start_time_ &&
         (stop_time_ < 0.0 || time <= stop_time_);
}

M0AdmSample CartoonM0FastFlow::SampleAdm(const Real rho, const Real z) const {
  M0AdmSample result;
  const auto stencil = LocateCartoonMeridionalPoint(pack_->pmesh, rho, z);
  if (!stencil.valid) return result;
  const int fd_stencil = pack_->z4c_symmetry.stencil_width;
  if (fd_stencil < 2 || fd_stencil > 4 || pack_->pmesh->mb_indcs.ng < fd_stencil)
    return result;
  Kokkos::View<Real *> values("Cartoon m0 ADM sample", 42);
  Kokkos::deep_copy(values, 0.0);
  if (stencil.owner_rank == global_variable::my_rank) {
    auto metric = pack_->padm->adm.g_dd;
    auto curvature = pack_->padm->adm.vK_dd;
    auto size = pack_->pmb->mb_size.d_view;
    const auto indices = pack_->pmesh->mb_indcs;
    Kokkos::parallel_for("Cartoon m0 ADM interpolate",
        Kokkos::RangePolicy<DevExeSpace>(0, 1), KOKKOS_LAMBDA(const int) {
      const int physical_to_code[3] = {0, 2, 1};
      for (int dj = 0; dj <= 1; ++dj) for (int di = 0; di <= 1; ++di) {
        const int i = stencil.i0 + di;
        const int j = stencil.j0 + dj;
        const Real weight = (di ? stencil.wi : 1.0 - stencil.wi) *
                            (dj ? stencil.wj : 1.0 - stencil.wj);
        const Real inverse_spacing[3] = {
            1.0 / size(stencil.local_block).dx1,
            1.0 / size(stencil.local_block).dx2,
            1.0 / size(stencil.local_block).dx3};
        for (int a = 0; a < 3; ++a) for (int b = a; b < 3; ++b) {
          const int packed = PackedIndex(a, b);
          const int ca = physical_to_code[a];
          const int cb = physical_to_code[b];
          values(packed) += weight * metric(stencil.local_block, ca, cb,
                                             stencil.k, j, i);
          values(6 + packed) += weight * curvature(stencil.local_block, ca, cb,
                                                    stencil.k, j, i);
          for (int d = 0; d < 3; ++d) {
            Real derivative_value = 0.0;
            if (fd_stencil == 2) {
              auto derivative = MakeCellCenteredDerivativeProvider<CartoonSO2, 2>(
                  inverse_spacing, size, indices.nx1, indices.is,
                  stencil.local_block, stencil.k, j, i);
              derivative_value = derivative.template TensorFirst<
                  TensorVariance::all_lower>(physical_to_code[d], ca, cb, metric);
            } else if (fd_stencil == 3) {
              auto derivative = MakeCellCenteredDerivativeProvider<CartoonSO2, 3>(
                  inverse_spacing, size, indices.nx1, indices.is,
                  stencil.local_block, stencil.k, j, i);
              derivative_value = derivative.template TensorFirst<
                  TensorVariance::all_lower>(physical_to_code[d], ca, cb, metric);
            } else {
              auto derivative = MakeCellCenteredDerivativeProvider<CartoonSO2, 4>(
                  inverse_spacing, size, indices.nx1, indices.is,
                  stencil.local_block, stencil.k, j, i);
              derivative_value = derivative.template TensorFirst<
                  TensorVariance::all_lower>(physical_to_code[d], ca, cb, metric);
            }
            values(12 + d * 6 + packed) += weight * derivative_value;
          }
        }
      }
    });
    Kokkos::fence();
  }
  auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), values);
  Real reduced[42];
  for (int n = 0; n < 42; ++n) reduced[n] = host(n);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, reduced, 42, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  for (int n = 0; n < 6; ++n) {
    result.metric[n] = reduced[n];
    result.curvature[n] = reduced[6 + n];
  }
  for (int n = 0; n < 18; ++n) result.metric_derivative[n] = reduced[12 + n];
  result.valid = true;
  for (const Real value : reduced) result.valid = result.valid && std::isfinite(value);
  return result;
}

M0AxisSample CartoonM0FastFlow::SampleAxisLapse(const Real z) const {
  M0AxisSample result;
  result.z = z;
  // Lapse is an evolved Z4c field.  In VC mode sample the authoritative
  // rho=0 nodal line; the ADM metric path above deliberately uses the explicit
  // cell-centred adapter instead.
  const auto stencil =
      LocateNativeCartoonMeridionalPoint(pack_->pmesh, 0.0, z);
  if (!stencil.valid) return result;
  Kokkos::View<Real *> values("Cartoon axis lapse sample", 2);
  Kokkos::deep_copy(values, 0.0);
  if (stencil.owner_rank == global_variable::my_rank) {
    auto u0 = pack_->pz4c->u0;
    const int alpha = pack_->pz4c->I_Z4C_ALPHA;
    Kokkos::parallel_for("Cartoon axis lapse interpolate",
        Kokkos::RangePolicy<DevExeSpace>(0, 1), KOKKOS_LAMBDA(const int) {
          values(0) = SampleCartoonMeridionalScalar(u0, alpha, stencil);
          values(1) = 1.0;
        });
    Kokkos::fence();
  }
  auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), values);
  Real reduced[2] = {host(0), host(1)};
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, reduced, 2, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
#endif
  result.lapse = reduced[0];
  result.valid = reduced[1] == 1.0 && std::isfinite(result.lapse) &&
                 result.lapse >= 0.0;
  return result;
}

M0CandidateSummary CartoonM0FastFlow::SearchCandidate(
    const std::string &branch, const Real center_z,
    const Real fresh_initial_radius,
    const std::vector<Real> &warm_start) {
  M0CandidateSummary summary;
  summary.branch = branch;
  summary.center_z = center_z;
  summary.fresh_initial_radius = fresh_initial_radius;
  summary.coefficients.assign(lmax_ + 1, 0.0);
  if (warm_start.size() == summary.coefficients.size()) {
    summary.coefficients = warm_start;
  } else {
    if (!std::isfinite(fresh_initial_radius) || !(fresh_initial_radius > 0.0)) {
      summary.failure = "invalid_initial_radius";
      return summary;
    }
    summary.coefficients[0] = fresh_initial_radius * std::sqrt(4.0 * kPi);
  }
  Real previous_mass = 0.0;
  for (int iteration = 0; iteration < iterations_; ++iteration) {
    if (!std::all_of(summary.coefficients.begin(), summary.coefficients.end(),
                     [](const Real coefficient) {
                       return std::isfinite(coefficient);
                     })) {
      summary.failure = "nonfinite_coefficient";
      return summary;
    }
    Real area = 0.0, direct2 = 0.0, flow2 = 0.0, mean = 0.0, spin_z = 0.0;
    Real minimum = std::numeric_limits<Real>::infinity();
    std::vector<Real> projection(lmax_ + 1, 0.0);
    bool covered = true;
    for (int n = 0; n < ntheta_; ++n) {
      Real radius = 0.0, first = 0.0, second = 0.0;
      for (int l = 0; l <= lmax_; ++l) {
        const int index = n * (lmax_ + 1) + l;
        radius += summary.coefficients[l] * y0_[index];
        first += summary.coefficients[l] * dy0_[index];
        second += summary.coefficients[l] * ddy0_[index];
      }
      if (!std::isfinite(radius) || radius <= 0.0) { covered = false; break; }
      minimum = std::min(minimum, radius);
      mean += weights_[n] * radius * 0.5;
      const auto point = EvaluateM0SurfacePoint(theta_[n], radius, first, second,
          SampleAdm(radius * std::sin(theta_[n]), center_z + radius * std::cos(theta_[n])));
      if (!point.valid) { covered = false; break; }
      const Real darea = kCartoonTwoPi * weights_[n] * point.area_factor /
                         std::sin(theta_[n]);
      area += darea;
      spin_z += darea * point.spin_integrand_z;
      direct2 += darea * point.expansion * point.expansion;
      flow2 += weights_[n] * point.flow_residual * point.flow_residual * 0.5;
      for (int l = 0; l <= lmax_; ++l) {
        projection[l] += kCartoonTwoPi * weights_[n] * point.flow_residual *
                         y0_[n * (lmax_ + 1) + l];
      }
    }
    if (!covered || !std::isfinite(area) || area <= 0.0 ||
        !std::isfinite(spin_z) || !std::isfinite(direct2) ||
        !std::isfinite(flow2) || !std::isfinite(mean) || mean <= 0.0 ||
        !std::isfinite(minimum) || minimum <= 0.0) {
      summary.failure = "coverage_or_radius";
      return summary;
    }
    const Real irreducible_mass = std::sqrt(area / (16.0 * kPi));
    const Real quasilocal_spin = spin_z / (8.0 * kPi);
    const Real mass = M0HorizonMass(area, quasilocal_spin);
    const Real direct = std::sqrt(direct2 / area) * mean;
    const Real flow = std::sqrt(flow2) * mean;
    summary.area = area;
    summary.irreducible_mass = irreducible_mass;
    summary.mass = mass;
    summary.spin_z = quasilocal_spin;
    summary.mean_radius = mean;
    summary.minimum_radius = minimum;
    summary.direct_residual = direct;
    summary.flow_residual = flow;
    if (!FiniteSummary(summary)) {
      summary.failure = "nonfinite_integral";
      return summary;
    }
    // Preserve the established flow convergence policy: the plateau check is
    // on irreducible mass, while `mass` reports the Christodoulou mass.
    const Real relative_mass = iteration == 0 ? std::numeric_limits<Real>::infinity() :
        std::abs(irreducible_mass - previous_mass) / irreducible_mass;
    if (iteration > 0 && relative_mass < mass_tolerance_ &&
        flow < hrms_tolerance_ && direct < direct_tolerance_) {
      summary.converged = true;
      summary.failure = "none";
      return summary;
    }
    previous_mass = irreducible_mass;
    const Real alpha = flow_scale_;
    const Real beta = 0.5 * flow_scale_;
    for (int l = 0; l <= lmax_; ++l) {
      const Real factor = (alpha / (lmax_ * (lmax_ + 1.0)) + beta) /
                          (1.0 + (beta / alpha) * l * (l + 1.0));
      summary.coefficients[l] -= factor * projection[l];
    }
  }
  summary.failure = "iteration_limit";
  return summary;
}

void CartoonM0FastFlow::Find(const int cycle, const Real time) {
  if (!ShouldSearch(cycle, time)) return;
  last_search_cycle_ = cycle;
  last_search_time_ = time;
  candidates_.clear();
  const M0AxisSample origin_sample = SampleAxisLapse(0.0);
  std::vector<M0AxisSample> axis_samples;
  axis_samples.reserve(2 * axis_search_samples_);
  for (int index = 1; index <= axis_search_samples_; ++index) {
    const Real magnitude = axis_search_bound_ * index / axis_search_samples_;
    axis_samples.push_back(SampleAxisLapse(-magnitude));
    axis_samples.push_back(SampleAxisLapse(magnitude));
  }
  Real plus_center = 0.0, minus_center = 0.0;
  Real plus_lapse = 0.0, minus_lapse = 0.0;
  const bool plus_valid = SelectM0AxisLapseMinimum(
      axis_samples, 1, &plus_center, &plus_lapse);
  const bool minus_valid = SelectM0AxisLapseMinimum(
      axis_samples, -1, &minus_center, &minus_lapse);
  if (!origin_sample.valid || !plus_valid || !minus_valid) {
    for (const auto &branch : {"origin", "plus", "minus"}) {
      M0CandidateSummary failed;
      failed.branch = branch;
      failed.failure = "axis_lapse_scan_coverage";
      candidates_.push_back(std::move(failed));
    }
    selected_.clear();
    found_ = false;
    Capture();
    return;
  }
  const Real origin_initial_radius = adaptive_initial_radius_
      ? M0OriginInitialRadius(initial_radius_, origin_lapse_radius_factor_,
                              plus_center, minus_center)
      : initial_radius_;
  const Real pair_initial_radius = adaptive_initial_radius_
      ? M0DisjointPairInitialRadius(initial_radius_, pair_disjoint_fraction_,
                                    plus_center, minus_center)
      : initial_radius_;
  for (const auto &entry : std::vector<std::pair<std::string, Real>>{
           {"origin", 0.0}, {"plus", plus_center}, {"minus", minus_center}}) {
    std::vector<Real> warm;
    const auto &saved = pack_->z4c_restart_state.fastflow;
    if (saved.selected_branch == entry.first && saved.center_count == 1)
      warm.assign(saved.coefficients.begin(), saved.coefficients.end());
    if (saved.selected_branch == "plus_minus" && saved.center_count == 2) {
      const int offset = entry.first == "plus" ? 0 :
                         entry.first == "minus" ? lmax_ + 1 : -1;
      if (offset >= 0) warm.assign(saved.coefficients.begin() + offset,
                                   saved.coefficients.begin() + offset + lmax_ + 1);
    }
    const Real fresh_initial_radius = entry.first == "origin"
        ? origin_initial_radius : pair_initial_radius;
    auto candidate = SearchCandidate(entry.first, entry.second,
                                     fresh_initial_radius, warm);
    candidate.axis_extremum_z = entry.second;
    candidate.center_lapse = entry.first == "plus" ? plus_lapse :
                              entry.first == "minus" ? minus_lapse :
                              origin_sample.lapse;
    // Absorb the dipole into the axial center, then re-solve so persisted
    // coefficients and center describe the same coordinate system.
    for (int recenter = 0; recenter < 2 && candidate.converged && lmax_ >= 1;
         ++recenter) {
      const Real shift = candidate.coefficients[1] *
                         std::sqrt(3.0 / (4.0 * kPi));
      if (std::abs(shift) <= 1.0e-10 * candidate.mean_radius) break;
      auto recentered = candidate.coefficients;
      recentered[1] = 0.0;
      const Real center_lapse = candidate.center_lapse;
      const Real axis_extremum_z = candidate.axis_extremum_z;
      candidate = SearchCandidate(entry.first, candidate.center_z + shift,
                                  fresh_initial_radius,
                                  recentered);
      candidate.axis_extremum_z = axis_extremum_z;
      candidate.center_lapse = center_lapse;
    }
    candidates_.push_back(std::move(candidate));
  }
  selected_.clear();
  if (mode_ == "single") {
    const int choice = SelectM0Single(candidates_);
    if (choice >= 0) selected_.push_back(choice);
  } else {
    int plus = -1, minus = -1;
    if (SelectM0MirrorPair(candidates_, pair_tolerance_, &plus, &minus)) {
      selected_ = {plus, minus};
    }
  }
  found_ = !selected_.empty();
  if (found_ && time_first_found_ < 0.0) time_first_found_ = time;
  Capture();
}

void CartoonM0FastFlow::Restore() {
  const auto &state = pack_->z4c_restart_state.fastflow;
  if (state.surface_mode == "none") return;
  std::string reason;
  if (!ValidateM0RestartState(state, lmax_, &reason))
    throw std::runtime_error("invalid Cartoon m=0 restart state: " + reason);
  if (mode_ != state.surface_mode)
    throw std::runtime_error("Cartoon m=0 surface mode conflicts with restart");
  found_ = state.converged;
  last_search_cycle_ = state.last_search_cycle;
  last_search_time_ = state.last_search_time;
  time_first_found_ = state.time_first_found;
  if (!found_) return;
  candidates_ = RestoreM0Candidates(state, lmax_, weights_, y0_);
  if (candidates_.size() != static_cast<std::size_t>(state.center_count))
    throw std::runtime_error("Cartoon m=0 restart has nonpositive surface radius");
  for (int index = 0; index < state.center_count; ++index) selected_.push_back(index);
}

void CartoonM0FastFlow::Capture() {
  auto &state = pack_->z4c_restart_state.fastflow;
  state = {};
  state.surface_mode = mode_;
  state.last_search_cycle = last_search_cycle_;
  state.last_search_time = last_search_time_;
  state.time_first_found = time_first_found_;
  state.converged = found_;
  state.status = found_ ? "accepted" : "failed";
  if (!found_) {
    state.failure_code = !candidates_.empty() &&
                         candidates_[0].failure == "axis_lapse_scan_coverage"
        ? "axis_lapse_scan_coverage"
        : (mode_ == "mirror_pair" ? "pair_incomplete" : "no_candidate");
    return;
  }
  state.failure_code = "none";
  state.center_count = static_cast<int>(selected_.size());
  state.selected_branch = selected_.size() == 2 ? "plus_minus" :
                          candidates_[selected_[0]].branch;
  state.center_z0 = candidates_[selected_[0]].center_z;
  if (selected_.size() == 2) state.center_z1 = candidates_[selected_[1]].center_z;
  for (const int index : selected_) {
    state.coefficients.insert(state.coefficients.end(),
                              candidates_[index].coefficients.begin(),
                              candidates_[index].coefficients.end());
  }
}

void CartoonM0FastFlow::Write(const int cycle, const Real time) {
  if (global_variable::my_rank != 0) return;
  if (output_ == nullptr) {
    const std::string path = pin_->GetString("job", "basename") +
        ".cartoon_m0_horizon_" + std::to_string(horizon_) + ".txt";
    output_ = std::fopen(path.c_str(), "a");
    if (output_ == nullptr) throw std::runtime_error("cannot open Cartoon m=0 output");
    std::fseek(output_, 0, SEEK_END);
    if (std::ftell(output_) == 0) {
      std::fprintf(output_, "# cycle time branch accepted center_z axis_extremum_z "
                            "center_lapse area irreducible_mass horizon_mass "
                            "spin_z mean_radius "
                            "minimum_radius direct_residual flow_residual failure "
                            "fresh_initial_radius a_l0...\n");
    }
  }
  for (const auto &candidate : candidates_) {
    std::fprintf(output_, "%d %.17e %s %d %.17e %.17e %.17e %.17e %.17e %.17e "
                          "%.17e %.17e %.17e %.17e %.17e %s",
                 cycle, time, candidate.branch.c_str(), candidate.converged,
                 candidate.center_z, candidate.axis_extremum_z,
                 candidate.center_lapse, candidate.area,
                 candidate.irreducible_mass, candidate.mass, candidate.spin_z,
                 candidate.mean_radius, candidate.minimum_radius,
                 candidate.direct_residual, candidate.flow_residual,
                 candidate.failure.c_str());
    std::fprintf(output_, " %.17e", candidate.fresh_initial_radius);
    for (const Real coefficient : candidate.coefficients)
      std::fprintf(output_, " %.17e", coefficient);
    std::fputc('\n', output_);
  }
  std::fflush(output_);
}

Real CartoonM0FastFlow::MinimumRadius() const {
  return MinimumM0SelectedRadius(candidates_, selected_);
}

Real CartoonM0FastFlow::SelectedCenterZ() const {
  return M0SelectedCenterZ(candidates_, selected_);
}
#endif

}  // namespace z4c
