//========================================================================================
//! \file cartoon_m0_fastflow.cpp
//! \brief Theta-only SO(2) FastFlow adapter.
//========================================================================================
#include "z4c/cartoon_m0_fastflow.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
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
std::array<Real, 3> M0Harmonic(int l, Real theta) {
  const Real x = std::cos(theta), dx = -std::sin(theta), ddx = -x;
  std::array<Real, 3> previous{1, 0, 0}, current{x, dx, ddx};
  if (l == 0) current = previous;
  for (int n = 2; n <= l; ++n) {
    std::array<Real, 3> next{
        ((2 * n - 1) * x * current[0] - (n - 1) * previous[0]) / n,
        ((2 * n - 1) * (dx * current[0] + x * current[1]) - (n - 1) * previous[1]) / n,
        ((2 * n - 1) * (ddx * current[0] + 2 * dx * current[1] + x * current[2]) -
         (n - 1) * previous[2]) /
            n};
    previous = current;
    current = next;
  }
  const Real normalization =
      std::sqrt((2 * l + 1) / (4 * 3.141592653589793238462643383279502884));
  for (auto& value : current) value *= normalization;
  return current;
}

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

bool FiniteSummary(const M0CandidateSummary& value) {
  return std::isfinite(value.center_z) && std::isfinite(value.axis_extremum_z) &&
         std::isfinite(value.center_lapse) && std::isfinite(value.area) &&
         std::isfinite(value.irreducible_mass) && std::isfinite(value.mass) &&
         std::isfinite(value.spin_z) && std::isfinite(value.mean_radius) &&
         std::isfinite(value.minimum_radius) && std::isfinite(value.direct_residual) &&
         std::isfinite(value.flow_residual) && value.area > 0.0 &&
         value.irreducible_mass > 0.0 && value.mass > 0.0 && value.mean_radius > 0.0 &&
         value.minimum_radius > 0.0 && value.center_lapse >= 0.0 &&
         value.direct_residual >= 0.0 && value.flow_residual >= 0.0 &&
         !value.coefficients.empty() &&
         std::all_of(value.coefficients.begin(), value.coefficients.end(),
                     [](const Real coefficient) { return std::isfinite(coefficient); });
}

Real RelativeDifference(const Real left, const Real right) {
  return std::abs(left - right) /
         std::max({std::abs(left), std::abs(right), Real{1.0e-300}});
}

}  // namespace

M0AdmSample RotateM0AdmSample(const M0AdmSample& input, const Real phi) {
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
          metric += rotation[a][i] * rotation[b][j] * input.metric[PackedIndex(i, j)];
          curvature +=
              rotation[a][i] * rotation[b][j] * input.curvature[PackedIndex(i, j)];
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

M0SurfacePoint EvaluateM0SurfacePoint(const Real theta, const Real radius,
                                      const Real radius_theta,
                                      const Real radius_theta_theta,
                                      const M0AdmSample& sample) {
  M0SurfacePoint result;
  if (!sample.valid || !std::isfinite(radius) || radius <= 0.0) return result;
  const bool pole = theta == 0.0 || theta == kPi;
  const Real st = pole ? 0.0 : std::sin(theta);
  const Real ct = std::cos(theta);
  if ((!pole && !(st > 0.0)) || (pole && std::abs(radius_theta) > 1.e-12 * radius))
    return result;
  Real g[3][3], curvature[3][3], inverse[3][3];
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      g[a][b] = sample.metric[PackedIndex(a, b)];
      curvature[a][b] = sample.curvature[PackedIndex(a, b)];
    }
  }
  const Real det = adm::SpatialDet(g[0][0], g[0][1], g[0][2], g[1][1], g[1][2], g[2][2]);
  if (!std::isfinite(det) || det <= 0.0 || !(g[0][0] > 0) ||
      !(g[0][0] * g[1][1] - g[0][1] * g[0][1] > 0))
    return result;
  adm::SpatialInv(1.0 / det, g[0][0], g[0][1], g[0][2], g[1][1], g[1][2], g[2][2],
                  &inverse[0][0], &inverse[0][1], &inverse[0][2], &inverse[1][1],
                  &inverse[1][2], &inverse[2][2]);
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
  ddtheta[1][1] = pole ? 0.0 : z / (x * radius * radius);
  ddtheta[2][2] = 2.0 * x * z / std::pow(radius, 4);
  Real dF[3], ddF[3][3], upper[3]{};
  for (int a = 0; a < 3; ++a) {
    dF[a] = dr[a] - radius_theta * dtheta[a];
    for (int b = 0; b < 3; ++b) {
      ddF[a][b] = ddr[a][b] - radius_theta * ddtheta[a][b] -
                  radius_theta_theta * dtheta[a] * dtheta[b];
    }
  }
  if (pole) {
    // Smooth m=0 graph: h'=0 and both transverse Hessian components
    // approach (r-h'')/r^2. Avoid the 0 * cot(theta) indeterminacy.
    dF[0] = dF[1] = 0.0;
    dF[2] = ct;
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) ddF[a][b] = 0.0;
    ddF[0][0] = ddF[1][1] = (radius - radius_theta_theta) * invr * invr;
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
  result.ingoing_expansion = -result.expansion + 2.0 * (normal_k * invu * invu - trace_k);
  result.flow_residual = result.expansion / invu;
  // At phi=0 the axial rotational Killing vector is (0,x,0).
  for (int b = 0; b < 3; ++b) {
    result.spin_integrand_z += x * upper[b] * invu * curvature[1][b];
  }
  const Real tangent_theta[3] = {radius_theta * st + radius * ct, 0.0,
                                 radius_theta * ct - radius * st};
  const Real tangent_phi[3] = {0.0, radius * st, 0.0};
  Real h11 = 0.0, h12 = 0.0, h22 = 0.0;
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) {
      h11 += tangent_theta[a] * tangent_theta[b] * g[a][b];
      h12 += tangent_theta[a] * tangent_phi[b] * g[a][b];
      h22 += tangent_phi[a] * tangent_phi[b] * g[a][b];
    }
  const Real det_h = h11 * h22 - h12 * h12;
  if (!std::isfinite(det_h) || (!pole && det_h <= 0.0) ||
      !std::isfinite(result.expansion)) {
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
                   0.25 * spin_z * spin_z / (irreducible_mass * irreducible_mass));
}

bool SelectM0AxisLapseMinimum(const std::vector<M0AxisSample>& samples, const int sign,
                              Real* center_z, Real* lapse) {
  if ((sign != -1 && sign != 1) || center_z == nullptr || lapse == nullptr) return false;
  int selected = -1;
  for (int index = 0; index < static_cast<int>(samples.size()); ++index) {
    const auto& sample = samples[index];
    if (!sample.valid || !std::isfinite(sample.z) || !std::isfinite(sample.lapse) ||
        sample.lapse < 0.0 || sign * sample.z <= 0.0)
      continue;
    if (selected < 0 || sample.lapse < samples[selected].lapse ||
        (sample.lapse == samples[selected].lapse &&
         (std::abs(sample.z) < std::abs(samples[selected].z) ||
          (std::abs(sample.z) == std::abs(samples[selected].z) && index < selected))))
      selected = index;
  }
  if (selected < 0) return false;
  *center_z = samples[selected].z;
  *lapse = samples[selected].lapse;
  return true;
}

Real M0OriginInitialRadius(const Real configured_radius, const Real lapse_radius_factor,
                           const Real plus_center_z, const Real minus_center_z) {
  if (!std::isfinite(configured_radius) || !(configured_radius > 0.0) ||
      !std::isfinite(lapse_radius_factor) || !(lapse_radius_factor > 0.0) ||
      !std::isfinite(plus_center_z) || !std::isfinite(minus_center_z) ||
      !(plus_center_z > 0.0) || !(minus_center_z < 0.0)) {
    return std::numeric_limits<Real>::quiet_NaN();
  }
  const Real lapse_radius = std::max(std::abs(plus_center_z), std::abs(minus_center_z));
  return std::max(configured_radius, lapse_radius_factor * lapse_radius);
}

Real M0DisjointPairInitialRadius(const Real configured_radius, const Real pair_fraction,
                                 const Real plus_center_z, const Real minus_center_z) {
  if (!std::isfinite(configured_radius) || !(configured_radius > 0.0) ||
      !std::isfinite(pair_fraction) || !(pair_fraction > 0.0) || !(pair_fraction < 1.0) ||
      !std::isfinite(plus_center_z) || !std::isfinite(minus_center_z) ||
      !(plus_center_z > minus_center_z)) {
    return std::numeric_limits<Real>::quiet_NaN();
  }
  const Real half_separation = 0.5 * (plus_center_z - minus_center_z);
  return std::min(configured_radius, pair_fraction * half_separation);
}

// Select an enclosing member of the discovered set, not merely largest area.
// Dense meridional containment includes both poles and translated centers.
// This does not certify that the search discovered the global outermost MOTS.
int SelectM0Outermost(const std::vector<M0CandidateSummary>& candidates) {
  auto radius = [](const M0CandidateSummary& c, Real theta) {
    Real value = 0;
    for (std::size_t l = 0; l < c.coefficients.size(); ++l)
      value += c.coefficients[l] * M0Harmonic(l, theta)[0];
    return value;
  };
  int selected = -1;
  for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
    const auto& outer = candidates[i];
    if (!outer.converged || !FiniteSummary(outer) || outer.coefficients.empty()) continue;
    bool encloses = true;
    for (int j = 0; j < static_cast<int>(candidates.size()) && encloses; ++j) {
      const auto& inner = candidates[j];
      if (i == j || !inner.converged || !FiniteSummary(inner)) continue;
      const int points = std::max(1061, 16 * static_cast<int>(
          std::max(outer.coefficients.size(), inner.coefficients.size())) + 1);
      for (int k = 0; k < points; ++k) {
        const Real theta = kPi * k / (points - 1);
        const Real ri = radius(inner, theta);
        const Real rho = ri * std::sin(theta);
        const Real z = inner.center_z + ri * std::cos(theta) - outer.center_z;
        const Real distance = std::hypot(rho, z);
        const Real ro = radius(outer, std::atan2(rho, z));
        if (!(ri > 0) || !(ro > 0) || !std::isfinite(distance) ||
            distance > ro + 1.e-8 * std::max(Real(1), ro)) {
          encloses = false;
          break;
        }
      }
    }
    if (encloses && (selected < 0 || outer.area > candidates[selected].area)) selected = i;
  }
  return selected;  // -1 also signals non-nested/crossing accepted candidates.
}

int SelectM0Single(const std::vector<M0CandidateSummary>& candidates) {
  int selected = -1;
  for (int index = 0; index < static_cast<int>(candidates.size()); ++index) {
    const auto& candidate = candidates[index];
    if (!candidate.converged || !FiniteSummary(candidate)) continue;
    if (selected < 0 ||
        candidate.direct_residual < candidates[selected].direct_residual ||
        (candidate.direct_residual == candidates[selected].direct_residual &&
         index < selected))
      selected = index;
  }
  return selected;
}

bool SelectM0MirrorPair(const std::vector<M0CandidateSummary>& candidates,
                        const Real tolerance, int* plus, int* minus) {
  *plus = -1;
  *minus = -1;
  for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
    if (!candidates[i].converged || !FiniteSummary(candidates[i])) continue;
    if (candidates[i].branch == "plus" &&
        (*plus < 0 || candidates[i].direct_residual < candidates[*plus].direct_residual))
      *plus = i;
    if (candidates[i].branch == "minus" &&
        (*minus < 0 ||
         candidates[i].direct_residual < candidates[*minus].direct_residual))
      *minus = i;
  }
  if (*plus < 0 || *minus < 0) return false;
  const auto& p = candidates[*plus];
  const auto& m = candidates[*minus];
  if (!(p.center_z > 0.0 && m.center_z < 0.0) ||
      p.coefficients.size() != m.coefficients.size())
    return false;
  // A conservative separating plane certifies disjointness. Failure means
  // unqualified pair, never rejection of either independently solved MOTS.
  Real extent_p = 0.0, extent_m = 0.0;
  for (std::size_t l = 0; l < p.coefficients.size(); ++l) {
    const Real bound = std::sqrt((2 * l + 1) / (4 * kPi));
    extent_p += std::abs(p.coefficients[l]) * bound;
    extent_m += std::abs(m.coefficients[l]) * bound;
    if (std::abs(p.coefficients[l] - (l % 2 ? -1 : 1) * m.coefficients[l]) >
        tolerance * std::sqrt(4 * kPi) * std::max(p.mean_radius, m.mean_radius))
      return false;
  }
  if (p.center_z - extent_p <= m.center_z + extent_m) return false;
  return RelativeDifference(p.center_z, -m.center_z) <= tolerance &&
         RelativeDifference(p.area, m.area) <= tolerance &&
         RelativeDifference(p.irreducible_mass, m.irreducible_mass) <= tolerance &&
         RelativeDifference(p.mass, m.mass) <= tolerance &&
         RelativeDifference(p.mean_radius, m.mean_radius) <= tolerance;
}

bool ValidateM0RestartState(const Z4cM0FastFlowRestartState& state, const int lmax,
                            std::string* reason) {
  auto fail = [reason](const char* message) {
    if (reason != nullptr) *reason = message;
    return false;
  };
  if (state.schema != Z4cM0FastFlowRestartState::kCurrentSchema) return fail("schema");
  if (state.last_search_cycle < -1 || !std::isfinite(state.last_search_time) ||
      state.last_search_time < 0.0 || !std::isfinite(state.time_first_found) ||
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
  if (!state.converged && state.center_count != 0) return fail("failed state centers");
  const std::size_t expected =
      state.converged ? static_cast<std::size_t>(expected_centers * (lmax + 1)) : 0;
  if (state.coefficients.size() != expected) return fail("coefficient count");
  for (const double coefficient : state.coefficients)
    if (!std::isfinite(coefficient)) return fail("nonfinite coefficient");
  if (!std::isfinite(state.center_z0) || !std::isfinite(state.center_z1) ||
      !std::isfinite(state.last_search_time))
    return fail("nonfinite metadata");
  if (state.converged && state.surface_mode == "single" &&
      state.selected_branch != "origin" && state.selected_branch != "plus" &&
      state.selected_branch != "minus")
    return fail("single branch");
  if (state.converged && state.surface_mode == "mirror_pair" &&
      state.selected_branch != "plus_minus")
    return fail("pair branch");
  return true;
}

std::vector<M0CandidateSummary> RestoreM0Candidates(
    const Z4cM0FastFlowRestartState& state, const int lmax,
    const std::vector<Real>& weights, const std::vector<Real>& y0) {
  std::string reason;
  if (!ValidateM0RestartState(state, lmax, &reason) || !state.converged) return {};
  const int stride = lmax + 1;
  if (y0.size() != weights.size() * static_cast<std::size_t>(stride)) return {};
  std::vector<M0CandidateSummary> candidates;
  for (int candidate_index = 0; candidate_index < state.center_count; ++candidate_index) {
    M0CandidateSummary candidate;
    candidate.converged = true;
    candidate.failure = "none";
    candidate.branch = state.center_count == 2 ? (candidate_index == 0 ? "plus" : "minus")
                                               : state.selected_branch;
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
    if (!std::isfinite(candidate.minimum_radius) || candidate.minimum_radius <= 0.0)
      return {};
    candidates.push_back(std::move(candidate));
  }
  return candidates;
}

Real MinimumM0SelectedRadius(const std::vector<M0CandidateSummary>& candidates,
                             const std::vector<int>& selected) {
  Real minimum = std::numeric_limits<Real>::infinity();
  for (const int index : selected) {
    if (index < 0 || index >= static_cast<int>(candidates.size()) ||
        !std::isfinite(candidates[index].minimum_radius) ||
        candidates[index].minimum_radius <= 0.0)
      return -1.0;
    minimum = std::min(minimum, candidates[index].minimum_radius);
  }
  return std::isfinite(minimum) ? minimum : -1.0;
}

Real M0SelectedCenterZ(const std::vector<M0CandidateSummary>& candidates,
                       const std::vector<int>& selected) {
  if (selected.empty()) return 0.0;
  Real center = 0.0;
  for (const int index : selected) {
    if (index < 0 || index >= static_cast<int>(candidates.size()) ||
        !std::isfinite(candidates[index].center_z))
      return 0.0;
    center += candidates[index].center_z;
  }
  return center / selected.size();
}

namespace {
// Geometry-independent tables are reused for every Jacobian column and time slice.
// Never cache sampled spacetime: positions and fields change during evolution.
struct M0AngularTable {
  std::vector<std::pair<Real, Real>> quadrature;
  std::vector<std::array<Real, 3>> harmonic;
  std::vector<Real> radius_basis;
};
const M0AngularTable& AngularTable(int count, int dim) {
  static std::map<std::pair<int, int>, M0AngularTable> cache;
  const auto key = std::make_pair(count, dim);
  auto found = cache.find(key);
  if (found != cache.end()) return found->second;
  // Bound host memory for parameter scans. No caller retains tables across evaluations.
  if (cache.size() >= 16) cache.clear();
  M0AngularTable table;
  table.quadrature = GaussLegendre(count);
  for (int n = 0; n < count; ++n)
    for (int l = 0; l < dim; ++l)
      table.harmonic.push_back(M0Harmonic(l, std::acos(table.quadrature[n].first)));
  table.radius_basis.resize((4 * count + 1) * dim);
  for (int n = 0; n <= 4 * count; ++n) {
    const Real mu = std::cos(kPi * n / (4 * count));
    Real previous = 1, current = mu;
    table.radius_basis[n * dim] = 1 / std::sqrt(4 * kPi);
    for (int l = 1; l < dim; ++l) {
      const Real value =
          l == 1 ? current : ((2 * l - 1) * mu * current - (l - 1) * previous) / l;
      if (l > 1) {
        previous = current;
        current = value;
      }
      table.radius_basis[n * dim + l] = value * std::sqrt((2 * l + 1) / (4 * kPi));
    }
  }
  return cache.emplace(key, std::move(table)).first->second;
}

struct M0Evaluation {
  M0CandidateSummary surface;
  std::vector<Real> projection;
  bool valid = false;
};
M0Evaluation EvaluateM0(const M0GeometrySampler& sample, int count,
                        const M0CandidateSummary& seed) {
  M0Evaluation out;
  out.surface = seed;
  auto& s = out.surface;
  auto failure = [&](const std::string& reason) {
    s.failure = reason;
    s.area = 0.0;  // Never expose a partially integrated surface as a full area.
    s.direct_residual = s.epsilon_inf = std::numeric_limits<Real>::infinity();
    return out;
  };
  s.converged = s.verified = s.angular_candidate = false;
  s.area = s.direct_residual = s.flow_residual = s.mean_radius = s.spin_z = 0;
  s.epsilon_inf = s.spacing = 0;
  s.minimum_radius = s.ingoing_min = std::numeric_limits<Real>::infinity();
  s.ingoing_max = -s.ingoing_min;
  const int dim = seed.coefficients.size();
  out.projection.assign(dim, 0.0);
  const auto& table = AngularTable(count, dim);
  const auto& q = table.quadrature;
  std::vector<std::array<Real, 2>> positions;
  std::vector<std::array<Real, 3>> shapes;
  std::vector<Real> basis(count * dim);
  for (int n = 0; n < count; ++n) {
    const Real theta = std::acos(q[n].first);
    std::array<Real, 3> shape{};
    for (int l = 0; l < dim; ++l) {
      const auto& harmonic = table.harmonic[n * dim + l];
      const Real y = harmonic[0], dy = harmonic[1], ddy = harmonic[2];
      basis[n * dim + l] = y;
      shape[0] += seed.coefficients[l] * y;
      shape[1] += seed.coefficients[l] * dy;
      shape[2] += seed.coefficients[l] * ddy;
    }
    if (!(shape[0] > 0) || !std::isfinite(shape[0])) {
      return failure("invalid_radius");
    }
    shapes.push_back(shape);
    positions.push_back(
        {shape[0] * std::sin(theta), seed.center_z + shape[0] * std::cos(theta)});
  }
  // Harmonic m=0 surfaces are pole-regular; also check radius on a separate
  // uniform angular grid including both poles (quadrature omits them).
  for (int n = 0; n <= 4 * count; ++n) {
    Real r = 0;
    for (int l = 0; l < dim; ++l)
      r += seed.coefficients[l] * table.radius_basis[n * dim + l];
    s.minimum_radius = std::min(s.minimum_radius, r);
    if (!(r > 0) || !std::isfinite(r)) {
      return failure("invalid_radius");
    }
  }
  // Include the analytic pole limits in epsilon_infinity, even though
  // Gauss quadrature has no endpoint weights.
  for (int pole = 0; pole < 2; ++pole) {
    std::array<Real, 3> h{};
    for (int l = 0; l < dim; ++l) {
      const Real y = std::sqrt((2 * l + 1) / (4 * kPi)) * ((pole && l % 2) ? -1 : 1);
      h[0] += seed.coefficients[l] * y;
      h[2] -= 0.5 * l * (l + 1) * seed.coefficients[l] * y;
    }
    shapes.push_back(h);
    positions.push_back({0.0, seed.center_z + (pole ? -1 : 1) * h[0]});
  }
  const auto geometry = sample(positions);
  if (geometry.size() != positions.size()) {
    return failure("unavailable_stencil");
  }
  for (int n = 0; n < count; ++n) {
    if (!geometry[n].valid) {
      return failure(geometry[n].error == 2 ? "nonfinite_residual"
                                            : "unavailable_stencil");
    }
    const Real theta = std::acos(q[n].first);
    const auto& h = shapes[n];
    const auto point = EvaluateM0SurfacePoint(theta, h[0], h[1], h[2], geometry[n]);
    if (!point.valid) {
      return failure("invalid_geometry");
    }
    const Real da = 2 * kPi * q[n].second * point.area_factor / std::sin(theta);
    s.area += da;
    s.direct_residual += da * point.expansion * point.expansion;
    s.flow_residual += 0.5 * q[n].second * point.flow_residual * point.flow_residual;
    s.epsilon_inf = std::max(s.epsilon_inf, std::abs(point.expansion));
    s.ingoing_min = std::min(s.ingoing_min, point.ingoing_expansion);
    s.ingoing_max = std::max(s.ingoing_max, point.ingoing_expansion);
    s.spacing = std::max(s.spacing, geometry[n].spacing);
    s.mean_radius += 0.5 * q[n].second * h[0];
    s.minimum_radius = std::min(s.minimum_radius, h[0]);
    s.spin_z += da * point.spin_integrand_z / (8 * kPi);
    for (int l = 0; l < dim; ++l)
      out.projection[l] +=
          2 * kPi * q[n].second * point.flow_residual * basis[n * dim + l];
  }
  for (int pole = 0; pole < 2; ++pole) {
    const auto& h = shapes[count + pole];
    const auto point =
        EvaluateM0SurfacePoint(pole ? kPi : 0.0, h[0], 0.0, h[2], geometry[count + pole]);
    if (!point.valid) {
      return failure("invalid_pole_geometry");
    }
    s.epsilon_inf = std::max(s.epsilon_inf, std::abs(point.expansion));
    s.ingoing_min = std::min(s.ingoing_min, point.ingoing_expansion);
    s.ingoing_max = std::max(s.ingoing_max, point.ingoing_expansion);
    s.spacing = std::max(s.spacing, geometry[count + pole].spacing);
  }
  const Real ra = std::sqrt(s.area / (4 * kPi));
  s.direct_residual = ra * std::sqrt(s.direct_residual / s.area);
  s.epsilon_inf *= ra;
  s.flow_residual = s.mean_radius * std::sqrt(s.flow_residual);
  s.irreducible_mass = ra / 2;
  s.mass = M0HorizonMass(s.area, s.spin_z);
  out.valid = FiniteSummary(s) && std::isfinite(s.epsilon_inf);
  s.failure = out.valid ? "none" : "nonfinite_residual";
  return out;
}
bool M0LinearSolve(std::vector<Real> a, std::vector<Real>& b) {
  const int n = b.size();
  Real norm = 0;
  for (Real x : a) norm = std::max(norm, std::abs(x));
  for (int k = 0; k < n; ++k) {
    int p = k;
    for (int i = k + 1; i < n; ++i)
      if (std::abs(a[i * n + k]) > std::abs(a[p * n + k])) p = i;
    if (!std::isfinite(a[p * n + k]) || std::abs(a[p * n + k]) <= 1.e-12 * norm)
      return false;
    for (int j = k; j < n; ++j) std::swap(a[k * n + j], a[p * n + j]);
    std::swap(b[k], b[p]);
    for (int i = k + 1; i < n; ++i) {
      const Real f = a[i * n + k] / a[k * n + k];
      for (int j = k; j < n; ++j) a[i * n + j] -= f * a[k * n + j];
      b[i] -= f * b[k];
    }
  }
  for (int i = n - 1; i >= 0; --i) {
    for (int j = i + 1; j < n; ++j) b[i] -= a[i * n + j] * b[j];
    b[i] /= a[i * n + i];
  }
  return std::all_of(b.begin(), b.end(), [](Real x) { return std::isfinite(x); });
}
}  // namespace

M0CandidateSummary AssessM0Surface(const M0GeometrySampler& sample, int count,
                                   const M0CandidateSummary& surface) {
  return EvaluateM0(sample, count, surface).surface;
}
M0CandidateSummary PreferM0Recentered(const M0CandidateSummary& original,
                                      const M0CandidateSummary& recentered) {
  if (!recentered.verified) return original;
  auto result = recentered;
  result.axis_extremum_z = original.axis_extremum_z;
  result.center_lapse = original.center_lapse;
  return result;
}

M0CandidateSummary SolveM0Surface(const M0GeometrySampler& sample,
                                  const M0SolveOptions& opt, const std::string& branch,
                                  Real center, Real radius,
                                  const std::vector<Real>& seed) {
  if (opt.lmax < 1 || opt.ntheta < 2 * opt.lmax + 4 || opt.iterations < 1)
    throw std::runtime_error("invalid M0 solver dimensions");
  M0CandidateSummary initial;
  initial.branch = branch;
  initial.center_z = center;
  initial.fresh_initial_radius = radius;
  initial.coefficients.assign(opt.lmax + 1, 0);
  initial.coefficients[0] = radius * std::sqrt(4 * kPi);
  if (seed.size() == initial.coefficients.size()) initial.coefficients = seed;
  auto current = EvaluateM0(sample, opt.ntheta, initial);
  if (!current.valid) return current.surface;
  const int dim = opt.lmax + 1;
  Real window_residual = current.surface.direct_residual;
  for (int it = 0; it < opt.iterations; ++it) {
    current.surface.iterations = it;
    if (current.surface.direct_residual <= opt.epsilon2 &&
        current.surface.epsilon_inf <= opt.epsilon_inf) {
      auto dense = EvaluateM0(sample, 4 * opt.ntheta + 3, current.surface);
      current.surface.converged = true;
      if (dense.valid && dense.surface.direct_residual <= opt.epsilon2 &&
          dense.surface.epsilon_inf <= opt.epsilon_inf) {
        dense.surface.converged = dense.surface.verified = true;
        return dense.surface;
      }
      current.surface.failure = "angular_verification_failed";
      return current.surface;
    }
    if (opt.candidate_policy && it > 0 && it % 8 == 0) {
      if (current.surface.direct_residual > window_residual * (1 - 1.e-3)) {
        current.surface.failure = "residual_plateau";
        return current.surface;
      }
      window_residual = current.surface.direct_residual;
    }
    bool advanced = false;
    for (int method = 0; method < 2 && !advanced; ++method) {
      std::vector<Real> step(dim);
      const bool use_newton = current.surface.direct_residual <= opt.newton_switch
                                  ? method == 0
                                  : method == 1;
      if (use_newton) {
        std::vector<Real> jac(dim * dim);
        bool good = true;
        std::vector<M0CandidateSummary> perturbed;
        std::vector<Real> deltas;
        std::vector<std::array<Real, 2>> all_points;
        std::vector<std::size_t> offsets{0};
        for (int j = 0; j < dim; ++j) {
          const Real delta = std::cbrt(std::numeric_limits<Real>::epsilon()) *
                             std::max(current.surface.mean_radius,
                                      std::abs(current.surface.coefficients[j]));
          deltas.push_back(delta);
          for (Real sign : {1.0, -1.0}) {
            auto trial = current.surface;
            trial.coefficients[j] += sign * delta;
            perturbed.push_back(trial);
            if (opt.batch_newton) {
              EvaluateM0(
                  [&](const std::vector<std::array<Real, 2>>& points) {
                    all_points.insert(all_points.end(), points.begin(), points.end());
                    return std::vector<M0AdmSample>{};
                  },
                  opt.ntheta, trial);
              good = good && all_points.size() > offsets.back();
              offsets.push_back(all_points.size());
            }
          }
        }
        std::vector<M0AdmSample> geometry;
        if (opt.batch_newton && good) {
          geometry = sample(all_points);
          good = geometry.size() == all_points.size();
        }
        for (int j = 0; j < dim && good; ++j) {
          auto evaluate = [&](int k) {
            if (!opt.batch_newton) return EvaluateM0(sample, opt.ntheta, perturbed[k]);
            return EvaluateM0(
                [&](const std::vector<std::array<Real, 2>>& points) {
                  if (points.size() != offsets[k + 1] - offsets[k])
                    return std::vector<M0AdmSample>{};
                  return std::vector<M0AdmSample>(geometry.begin() + offsets[k],
                                                  geometry.begin() + offsets[k + 1]);
                },
                opt.ntheta, perturbed[k]);
          };
          const auto plus = evaluate(2 * j), minus = evaluate(2 * j + 1);
          good = plus.valid && minus.valid;
          if (good)
            for (int i = 0; i < dim; ++i)
              jac[i * dim + j] =
                  (plus.projection[i] - minus.projection[i]) / (2 * deltas[j]);
        }
        for (int i = 0; i < dim; ++i) step[i] = -current.projection[i];
        if (!good || !M0LinearSolve(jac, step)) continue;
      } else {
        const Real scale =
            opt.flow_scale * current.surface.mean_radius * current.surface.mean_radius;
        for (int l = 0; l < dim; ++l)
          step[l] = -scale * (1.0 / (opt.lmax * (opt.lmax + 1.0)) + 0.5) /
                    (1 + 0.5 * l * (l + 1.0)) * current.projection[l];
      }
      // |Y_l0| <= sqrt((2l+1)/4pi), a conservative all-angle displacement bound.
      Real bound = 0;
      for (int l = 0; l < dim; ++l)
        bound += std::abs(step[l]) * std::sqrt((2 * l + 1) / (4 * kPi));
      Real fraction =
          std::min(Real(1), opt.displacement * current.surface.minimum_radius /
                                std::max(bound, std::numeric_limits<Real>::min()));
      for (int back = 0; back < opt.backtracks; ++back, fraction *= 0.5) {
        auto trial = current.surface;
        for (int l = 0; l < dim; ++l) trial.coefficients[l] += fraction * step[l];
        auto evaluated = EvaluateM0(sample, opt.ntheta, trial);
        if (evaluated.valid &&
            evaluated.surface.direct_residual <
                current.surface.direct_residual * (1 - 1.e-4 * fraction)) {
          current = std::move(evaluated);
          advanced = true;
          break;
        }
      }
    }
    if (!advanced) {
      current.surface.failure = "line_search_failure";
      return current.surface;
    }
  }
  current.surface.failure = "iteration_limit";
  return current.surface;
}

M0CandidateSummary SolveM0Refined(const M0GeometrySampler& sample,
                                  const M0SolveOptions& options, int start_l,
                                  const std::string& branch, Real center, Real radius,
                                  const std::vector<Real>& seed) {
  if (start_l < 1 || start_l > options.lmax)
    throw std::runtime_error("invalid angular refinement range");
  if (options.candidate_l32_window &&
      (!options.candidate_policy || start_l != 8 || options.lmax != 64))
    throw std::runtime_error("L32 candidate window requires L8 start and L64 cap");
  bool coarse_window_passed = false;
  std::vector<Real> coefficients = seed;
  int level = seed.empty() ? start_l : std::max(start_l, int(seed.size()) - 1);
  if (level > options.lmax) throw std::runtime_error("seed exceeds angular cap");
  // Candidate evidence is rebuilt on each slice, including for tracked surfaces.
  // Truncate a good tracked shape to the coarse basis, then recheck angular decrease.
  if (options.candidate_policy && !seed.empty()) {
    level = start_l;
    coefficients.resize(level + 1);
  }
  std::vector<M0AngularStage> stages;
  std::vector<Real> residuals;
  M0CandidateSummary result, previous;
  int consecutive_decreases = 0;
  for (;;) {
    auto opt = options;
    opt.lmax = level;
    opt.ntheta =
        std::max(2 * level + 4,
                 int(std::ceil(Real(options.ntheta - 4) * level / options.lmax)) + 4);
    if (options.candidate_policy)
      opt.iterations = std::min(opt.iterations, opt.coarse_iterations);
    if (!coefficients.empty()) coefficients.resize(level + 1, 0);
    result = SolveM0Surface(sample, opt, branch, center, radius, coefficients);
    result.solved_lmax = level;
    if (options.candidate_policy && result.area > 0 &&
        std::isfinite(result.direct_residual)) {
      const bool strict = result.verified;
      const int iterations = result.iterations;
      const auto failure = result.failure;
      result = AssessM0Surface(sample, options.candidate_points, result);
      result.verified = strict && result.direct_residual <= options.epsilon2 &&
                        result.epsilon_inf <= options.epsilon_inf;
      result.converged = result.verified;
      result.iterations = iterations;
      result.solved_lmax = level;
      if (result.area > 0) result.failure = failure;
    }
    stages.push_back({level, opt.ntheta, result.iterations, result.direct_residual,
                      result.epsilon_inf, result.area, result.verified, result.failure});
    if (options.candidate_policy) {
      Real shape = std::numeric_limits<Real>::infinity(), area = shape;
      if (previous.area > 0 && result.area > 0) {
        Real sum = 0;
        for (std::size_t l = 0; l < result.coefficients.size(); ++l) {
          const Real diff =
              result.coefficients[l] -
              (l < previous.coefficients.size() ? previous.coefficients[l] : 0);
          sum += diff * diff;
        }
        shape = std::sqrt(sum / (4 * kPi)) / result.mean_radius;
        area = std::abs(result.area - previous.area) / result.area;
        consecutive_decreases =
            (options.candidate_l32_window
                 ? result.direct_residual < previous.direct_residual
                 : result.direct_residual <= options.angular_ratio * previous.direct_residual)
                ? consecutive_decreases + 1
                : 0;
      }
      if (options.candidate_l32_window && level == 32)
        coarse_window_passed = consecutive_decreases >= 2;
      result.angular_candidate = options.candidate_l32_window
          ? (level == 64 && coarse_window_passed && result.area > 0 &&
             std::isfinite(result.direct_residual) &&
             result.direct_residual <= options.candidate_bound)
          : (result.verified ||
             (consecutive_decreases >= 2 && result.area > 0 &&
              result.direct_residual <= options.candidate_bound &&
              shape <= options.shape_change && area <= options.area_change));
      // The relaxed policy measures at L64, but only the L8/16/32 sequence
      // supplies convergence evidence. A failed coarse window never promotes.
      if (options.candidate_l32_window && level == 32 && !coarse_window_passed) {
        result.failure = "coarse_window_failed";
        break;
      }
      if (result.angular_candidate) {
        result.failure = result.verified ? "none" : "angular_candidate";
        break;
      }
      if (!(result.area > 0) || !std::isfinite(result.direct_residual) ||
          result.direct_residual > options.promotion_max ||
          (previous.area > 0 &&
           result.direct_residual > 1.25 * previous.direct_residual)) {
        result.failure = "refinement_gated";
        break;
      }
    }
    if (level == options.lmax) break;
    if (!(result.area > 0) || !std::isfinite(result.direct_residual)) break;
    previous = result;
    coefficients = result.coefficients;
    level = std::min(2 * level, options.lmax);
  }
  result.angular_stages = std::move(stages);
  result.coefficients.resize(options.lmax + 1, 0);  // fixed-width restart/history carrier

  return result;
}

std::vector<M0CandidateSummary> RestoreM0Seeds(const Z4cM0FastFlowRestartState& state,
                                               const int lmax) {
  std::vector<M0CandidateSummary> seeds;
  if (!state.converged || state.center_count < 1 || state.center_count > 2 || lmax < 1 ||
      state.coefficients.empty() || state.coefficients.size() % state.center_count)
    return seeds;
  const int old_dim = state.coefficients.size() / state.center_count;
  std::string reason;
  if (!ValidateM0RestartState(state, old_dim - 1, &reason)) return seeds;
  for (int n = 0; n < state.center_count; ++n) {
    M0CandidateSummary seed;
    seed.branch =
        state.center_count == 1 ? state.selected_branch : (n == 0 ? "plus" : "minus");
    seed.center_z = n == 0 ? state.center_z0 : state.center_z1;
    seed.coefficients.assign(lmax + 1, 0.0);
    for (int l = 0; l < std::min(old_dim, lmax + 1); ++l)
      seed.coefficients[l] = state.coefficients[n * old_dim + l];
    seed.fresh_initial_radius = std::abs(seed.coefficients[0]) / std::sqrt(4 * kPi);
    // These are guesses, never current-slice detection or verification state.
    seeds.push_back(std::move(seed));
  }
  return seeds;
}

#ifndef ATHENA_CARTOON_M0_MATH_ONLY
CartoonM0FastFlow::CartoonM0FastFlow(MeshBlockPack* pack, ParameterInput* pin,
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
  hrms_tolerance_ =
      pin->GetOrAddReal("fastflow", "dimensionless_hrms_tol_" + suffix, 3.0e-2);
  mass_tolerance_ = pin->GetOrAddReal("fastflow", "mass_relative_tol_" + suffix, 1e-4);
  direct_tolerance_ =
      pin->GetOrAddReal("fastflow", "cartoon_direct_residual_tol_" + suffix, 3.0e-2);
  pair_tolerance_ =
      pin->GetOrAddReal("fastflow", "cartoon_pair_relative_tol_" + suffix, 1e-3);
  adaptive_initial_radius_ =
      pin->GetOrAddBoolean("fastflow", "cartoon_adaptive_initial_radius_" + suffix, true);
  origin_lapse_radius_factor_ =
      pin->GetOrAddReal("fastflow", "cartoon_origin_lapse_radius_factor_" + suffix, 3.0);
  pair_disjoint_fraction_ =
      pin->GetOrAddReal("fastflow", "cartoon_pair_disjoint_fraction_" + suffix, 0.8);
  center_seed_ = std::abs(
      pin->GetOrAddReal("fastflow", "cartoon_center_z_" + suffix, initial_radius_));
  axis_search_bound_ =
      pin->GetOrAddReal("fastflow", "cartoon_axis_search_bound_" + suffix,
                        center_seed_ > 0.0 ? center_seed_ : initial_radius_);
  axis_search_samples_ =
      pin->GetOrAddInteger("fastflow", "cartoon_axis_search_samples_" + suffix, 33);
  mode_ = pin->GetOrAddString("fastflow", "cartoon_surface_mode_" + suffix, "single");
  if (lmax_ < 1 || ntheta_ < 2 || iterations_ < 1 || find_interval_ < 1 ||
      !(initial_radius_ > 0.0) || !(flow_scale_ > 0.0) || !(hrms_tolerance_ > 0.0) ||
      !(mass_tolerance_ > 0.0) || !(direct_tolerance_ > 0.0) ||
      !(pair_tolerance_ >= 0.0) || !std::isfinite(origin_lapse_radius_factor_) ||
      !(origin_lapse_radius_factor_ > 0.0) || !std::isfinite(pair_disjoint_fraction_) ||
      !(pair_disjoint_fraction_ > 0.0) || !(pair_disjoint_fraction_ < 1.0) ||
      !std::isfinite(center_seed_) || !std::isfinite(axis_search_bound_) ||
      !(axis_search_bound_ > 0.0) || axis_search_samples_ < 2 ||
      !std::isfinite(start_time_) || !std::isfinite(stop_time_) ||
      (mode_ != "single" && mode_ != "mirror_pair")) {
    throw std::runtime_error("invalid Cartoon m=0 FastFlow configuration");
  }
  ntheta_ = std::max(ntheta_, 2 * lmax_ + 4);
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
      const auto harmonic = M0Harmonic(l, theta_[n]);
      y0_[n * (lmax_ + 1) + l] = harmonic[0];
      dy0_[n * (lmax_ + 1) + l] = harmonic[1];
      ddy0_[n * (lmax_ + 1) + l] = harmonic[2];
    }
  }
  solve_options_.lmax = lmax_;
  solve_options_.ntheta = std::max(ntheta_, 2 * lmax_ + 4);
  pin->SetInteger("fastflow", "ntheta", solve_options_.ntheta);
  solve_options_.iterations = iterations_;
  solve_options_.flow_scale = flow_scale_;
  solve_options_.epsilon2 = pin->GetOrAddReal("fastflow", "mots_epsilon2", 1.e-6);
  solve_options_.epsilon_inf = pin->GetOrAddReal("fastflow", "mots_epsilon_inf", 1.e-5);
  solve_options_.newton_switch = pin->GetOrAddReal("fastflow", "mots_newton_switch", 0.1);
  solve_options_.displacement = pin->GetOrAddReal("fastflow", "mots_displacement", 0.1);
  solve_options_.backtracks = pin->GetOrAddInteger("fastflow", "mots_backtracks", 24);
  l_start_ = pin->GetOrAddInteger("fastflow", "mots_l_start", std::min(8, lmax_));
  const auto policy =
      pin->GetOrAddString("fastflow", "mots_detection", "angular_candidate");
  if (policy != "strict" && policy != "angular_candidate" && policy != "angular_l32")
    throw std::runtime_error("unknown MOTS detection policy");
  solve_options_.candidate_policy = policy != "strict";
  solve_options_.candidate_l32_window = policy == "angular_l32";
  if (solve_options_.candidate_l32_window && (l_start_ != 8 || lmax_ != 64))
    throw std::runtime_error("angular_l32 requires mots_l_start=8 and lmax=64");
  solve_options_.coarse_iterations =
      pin->GetOrAddInteger("fastflow", "mots_level_iterations", 96);
  solve_options_.candidate_points =
      pin->GetOrAddInteger("fastflow", "mots_candidate_points", 1061);
  solve_options_.promotion_max = pin->GetOrAddReal("fastflow", "mots_promotion_max", 0.5);
  solve_options_.candidate_bound =
      pin->GetOrAddReal("fastflow", "mots_candidate_bound", 0.05);
  solve_options_.angular_ratio = pin->GetOrAddReal("fastflow", "mots_angular_ratio", 0.8);
  solve_options_.shape_change = pin->GetOrAddReal("fastflow", "mots_shape_change", 0.02);
  solve_options_.area_change = pin->GetOrAddReal("fastflow", "mots_area_change", 0.01);
  if (solve_options_.coarse_iterations < 1 ||
      solve_options_.candidate_points < 2 * lmax_ + 4 ||
      !(solve_options_.promotion_max > 0) || !(solve_options_.candidate_bound > 0) ||
      solve_options_.candidate_bound > 0.05 || !(solve_options_.angular_ratio > 0) ||
      !(solve_options_.angular_ratio < 1) || !(solve_options_.shape_change > 0) ||
      !(solve_options_.area_change > 0))
    throw std::runtime_error("invalid candidate controls");
  discovery_interval_ = pin->GetOrAddInteger("fastflow", "mots_discovery_interval", 8);
  tracking_residual_ = pin->GetOrAddReal("fastflow", "mots_tracking_residual", 0.01);
  if (l_start_ < 1 || l_start_ > lmax_ || discovery_interval_ < 1 ||
      !(tracking_residual_ > 0) || !std::isfinite(tracking_residual_))
    throw std::runtime_error("invalid MOTS runtime continuation controls");
  const auto selection = pin->GetOrAddString("fastflow", "mots_selection", "outermost");
  if (selection != "outermost" && selection != "residual")
    throw std::runtime_error("unknown MOTS selection policy");
  outermost_selection_ = selection == "outermost";
  radius_count_ = pin->GetOrAddInteger("fastflow", "mots_radius_count", 8);
  radius_min_ = pin->GetOrAddReal("fastflow", "mots_radius_min", 0.0);
  if (!std::isfinite(radius_min_) || !(solve_options_.displacement > 0) ||
      !(solve_options_.displacement <= 0.1) || radius_count_ < 1 || radius_min_ < 0 ||
      solve_options_.epsilon2 <= 0 || solve_options_.epsilon_inf <= 0 ||
      solve_options_.newton_switch <= 0 || solve_options_.backtracks < 1)
    throw std::runtime_error("invalid MOTS controls");
  if (pin->GetOrAddBoolean("fastflow", "horizon_only", false)) {
    last_good_ = RestoreM0Seeds(pack_->z4c_restart_state.fastflow, lmax_);
    // Analysis seeds are guesses only, including unconverged lower-order surfaces.
    const auto seed_path = pin->GetOrAddString("fastflow", "mots_seed_file", "");
    if (!seed_path.empty()) {
      std::ifstream file(seed_path);
      M0CandidateSummary seed;
      int count = 0;
      if (!(file >> seed.center_z >> count) || !std::isfinite(seed.center_z) ||
          count < 1 || count > lmax_ + 1)
        throw std::runtime_error("invalid MOTS analysis seed header");
      seed.branch = seed.center_z == 0 ? "origin" : seed.center_z > 0 ? "plus" : "minus";
      seed.coefficients.assign(lmax_ + 1, 0.0);
      for (int l = 0; l < count; ++l)
        if (!(file >> seed.coefficients[l]) || !std::isfinite(seed.coefficients[l]))
          throw std::runtime_error("invalid MOTS analysis seed coefficient");
      std::string extra;
      if (file >> extra) throw std::runtime_error("trailing MOTS analysis seed data");
      seed.fresh_initial_radius = seed.coefficients[0] / std::sqrt(4 * kPi);
      last_good_ = {seed};
    }
  } else {
    if (pack_->z4c_restart_state.fastflow.status == "angular_candidate")
      last_good_ = RestoreM0Seeds(pack_->z4c_restart_state.fastflow, lmax_);
    else
      Restore();
    for (int i : selected_) last_good_.push_back(candidates_[i]);
  }
}

CartoonM0FastFlow::~CartoonM0FastFlow() {
  if (output_ != nullptr) std::fclose(output_);
}

bool CartoonM0FastFlow::ShouldSearch(const int cycle, const Real time) const {
  return cycle >= 0 && cycle % find_interval_ == 0 && time >= start_time_ &&
         (stop_time_ < 0.0 || time <= stop_time_);
}

std::vector<M0AdmSample> CartoonM0FastFlow::SampleAdmBatch(
    const std::vector<std::array<Real, 2>>& points) const {
  std::vector<M0AdmSample> result(points.size());
  if (pack_->pz4c->layout.centering != Z4cGridCentering::vertex) {
    for (std::size_t n = 0; n < points.size(); ++n)
      result[n] = SampleAdm(points[n][0], points[n][1]);
    return result;
  }
  const int count = points.size();
  if (count == 0) return result;
  const auto layout = pack_->pz4c->layout;
  const int fd = pack_->pz4c->opt.fd_stencil;
  if (fd < 2 || fd > 4 || pack_->pmesh->mb_indcs.ng < fd - 1 || layout.nx1 < 3 ||
      layout.nx2 < 3)
    return result;
  Kokkos::View<CartoonMeridionalStencil*> stencils("MOTS locations", count);
  auto locations = Kokkos::create_mirror_view(stencils);
  for (int n = 0; n < count; ++n) {
    auto p = LocateNativeCartoonMeridionalPoint(pack_->pmesh, points[n][0], points[n][1]);
    if (p.valid && p.owner_rank == global_variable::my_rank) {
      // Keep all derivative evaluation nodes active. Cubic stencils become
      // one-sided near a face, without reducing polynomial degree.
      const int i = std::max(layout.is, std::min(layout.ie - 3, p.i0 - 1));
      const int j = std::max(layout.js, std::min(layout.je - 3, p.j0 - 1));
      p.wi += p.i0 - i;
      p.wj += p.j0 - j;
      p.i0 = i;
      p.j0 = j;
      const int reach = fd - 1;
      if (i - reach < 0 || i + 3 + reach >= layout.n1 || j - reach < 0 ||
          j + 3 + reach >= layout.n2)
        p.valid = false;
    }
    locations(n) = p;
  }
  Kokkos::deep_copy(stencils, locations);
  Kokkos::View<Real**> values("MOTS geometry batch", count, 32);
  Kokkos::deep_copy(values, 0.0);
  const auto metric = pack_->pz4c->adm.g_dd;
  const auto curvature = pack_->pz4c->adm.vK_dd;
  const auto size = pack_->pmb->mb_size.d_view;
  const int rank = global_variable::my_rank;
  Kokkos::parallel_for(
      "MOTS native cubic samples", Kokkos::RangePolicy<DevExeSpace>(0, count),
      KOKKOS_LAMBDA(const int n) {
        const auto s = stencils(n);
        if (!s.valid || s.owner_rank != rank) return;
        Real wi[4], wj[4];
        for (int a = 0; a < 4; ++a) {
          wi[a] = wj[a] = 1;
          for (int b = 0; b < 4; ++b)
            if (a != b) {
              wi[a] *= (s.wi - b) / (a - b);
              wj[a] *= (s.wj - b) / (a - b);
            }
        }
        const int map[3] = {0, 2, 1};
        const Real inv[3] = {1 / size(s.local_block).dx1, 1 / size(s.local_block).dx2,
                             1 / size(s.local_block).dx3};
        for (int dj = 0; dj < 4; ++dj)
          for (int di = 0; di < 4; ++di) {
            const int i = s.i0 + di, j = s.j0 + dj, m = s.local_block;
            const Real w = wi[di] * wj[dj];
            for (int a = 0; a < 3; ++a)
              for (int b = a; b < 3; ++b) {
                const int v = PackedIndex(a, b), ca = map[a], cb = map[b];
                values(n, v) += w * metric(m, ca, cb, s.k, j, i);
                values(n, 6 + v) += w * curvature(m, ca, cb, s.k, j, i);
                for (int d = 0; d < 3; ++d) {
                  Real value = 0;
                  if (fd == 2) {
                    auto deriv = MakeVertexCenteredDerivativeProvider<CartoonSO2, 2>(
                        inv, size, layout.nx1, layout.is, m, s.k, j, i);
                    value = deriv.template TensorFirst<TensorVariance::all_lower>(
                        map[d], ca, cb, metric);
                  } else if (fd == 3) {
                    auto deriv = MakeVertexCenteredDerivativeProvider<CartoonSO2, 3>(
                        inv, size, layout.nx1, layout.is, m, s.k, j, i);
                    value = deriv.template TensorFirst<TensorVariance::all_lower>(
                        map[d], ca, cb, metric);
                  } else {
                    auto deriv = MakeVertexCenteredDerivativeProvider<CartoonSO2, 4>(
                        inv, size, layout.nx1, layout.is, m, s.k, j, i);
                    value = deriv.template TensorFirst<TensorVariance::all_lower>(
                        map[d], ca, cb, metric);
                  }
                  values(n, 12 + 6 * d + v) += w * value;
                }
              }
          }
        values(n, 30) = 1;
        values(n, 31) = fmax(size(s.local_block).dx1, size(s.local_block).dx2);
      });
  auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), values);
  // Pack explicitly: device/host layouts need not have the same strides.
  std::vector<Real> packed(count * 32);
  for (int n = 0; n < count; ++n)
    for (int v = 0; v < 32; ++v) packed[n * 32 + v] = host(n, v);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, packed.data(), count * 32, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
#endif
  for (int n = 0; n < count; ++n) {
    auto& r = result[n];
    r.valid = packed[n * 32 + 30] == 1;
    r.spacing = packed[n * 32 + 31];
    for (int v = 0; v < 6; ++v) {
      r.metric[v] = packed[n * 32 + v];
      r.curvature[v] = packed[n * 32 + 6 + v];
    }
    for (int v = 0; v < 18; ++v) r.metric_derivative[v] = packed[n * 32 + 12 + v];
    r.error = r.valid ? 0 : 1;
    for (int v = 0; v < 30; ++v)
      if (!std::isfinite(packed[n * 32 + v])) {
        r.valid = false;
        r.error = 2;
      }
  }
  return result;
}

M0AdmSample CartoonM0FastFlow::SampleAdm(const Real rho, const Real z) const {
  if (pack_->pz4c->layout.centering == Z4cGridCentering::vertex)
    return SampleAdmBatch({{rho, z}})[0];
  M0AdmSample result;
  const auto stencil = LocateCartoonMeridionalPoint(pack_->pmesh, rho, z);
  if (!stencil.valid) return result;
  const int fd_stencil = pack_->z4c_symmetry.stencil_width;
  if (fd_stencil < 2 || fd_stencil > 4 || pack_->pmesh->mb_indcs.ng < fd_stencil)
    return result;
  Kokkos::View<Real*> values("Cartoon m0 ADM sample", 42);
  Kokkos::deep_copy(values, 0.0);
  if (stencil.owner_rank == global_variable::my_rank) {
    auto metric = pack_->padm->adm.g_dd;
    auto curvature = pack_->padm->adm.vK_dd;
    auto size = pack_->pmb->mb_size.d_view;
    const auto indices = pack_->pmesh->mb_indcs;
    Kokkos::parallel_for(
        "Cartoon m0 ADM interpolate", Kokkos::RangePolicy<DevExeSpace>(0, 1),
        KOKKOS_LAMBDA(const int) {
          const int physical_to_code[3] = {0, 2, 1};
          for (int dj = 0; dj <= 1; ++dj)
            for (int di = 0; di <= 1; ++di) {
              const int i = stencil.i0 + di;
              const int j = stencil.j0 + dj;
              const Real weight = (di ? stencil.wi : 1.0 - stencil.wi) *
                                  (dj ? stencil.wj : 1.0 - stencil.wj);
              const Real inverse_spacing[3] = {1.0 / size(stencil.local_block).dx1,
                                               1.0 / size(stencil.local_block).dx2,
                                               1.0 / size(stencil.local_block).dx3};
              for (int a = 0; a < 3; ++a)
                for (int b = a; b < 3; ++b) {
                  const int packed = PackedIndex(a, b);
                  const int ca = physical_to_code[a];
                  const int cb = physical_to_code[b];
                  values(packed) +=
                      weight * metric(stencil.local_block, ca, cb, stencil.k, j, i);
                  values(6 + packed) +=
                      weight * curvature(stencil.local_block, ca, cb, stencil.k, j, i);
                  for (int d = 0; d < 3; ++d) {
                    Real derivative_value = 0.0;
                    if (fd_stencil == 2) {
                      auto derivative = MakeCellCenteredDerivativeProvider<CartoonSO2, 2>(
                          inverse_spacing, size, indices.nx1, indices.is,
                          stencil.local_block, stencil.k, j, i);
                      derivative_value =
                          derivative.template TensorFirst<TensorVariance::all_lower>(
                              physical_to_code[d], ca, cb, metric);
                    } else if (fd_stencil == 3) {
                      auto derivative = MakeCellCenteredDerivativeProvider<CartoonSO2, 3>(
                          inverse_spacing, size, indices.nx1, indices.is,
                          stencil.local_block, stencil.k, j, i);
                      derivative_value =
                          derivative.template TensorFirst<TensorVariance::all_lower>(
                              physical_to_code[d], ca, cb, metric);
                    } else {
                      auto derivative = MakeCellCenteredDerivativeProvider<CartoonSO2, 4>(
                          inverse_spacing, size, indices.nx1, indices.is,
                          stencil.local_block, stencil.k, j, i);
                      derivative_value =
                          derivative.template TensorFirst<TensorVariance::all_lower>(
                              physical_to_code[d], ca, cb, metric);
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
  return SampleAxisLapseBatch({z})[0];
}
std::vector<M0AxisSample> CartoonM0FastFlow::SampleAxisLapseBatch(
    const std::vector<Real>& z) const {
  const int count = z.size();
  std::vector<M0AxisSample> result(count);
  Kokkos::View<CartoonMeridionalStencil*> stencils("MOTS axis stencils", count);
  auto host_stencils = Kokkos::create_mirror_view(stencils);
  for (int n = 0; n < count; ++n)
    host_stencils(n) = LocateNativeCartoonMeridionalPoint(pack_->pmesh, 0, z[n]);
  Kokkos::deep_copy(stencils, host_stencils);
  Kokkos::View<Real*> values("MOTS axis lapse batch", 2 * count);
  Kokkos::deep_copy(values, 0.0);
  auto u0 = pack_->pz4c->u0;
  const int alpha = pack_->pz4c->I_Z4C_ALPHA, rank = global_variable::my_rank;
  Kokkos::parallel_for(
      "MOTS axis lapse batch", Kokkos::RangePolicy<DevExeSpace>(0, count),
      KOKKOS_LAMBDA(const int n) {
        const auto s = stencils(n);
        if (s.valid && s.owner_rank == rank) {
          values(2 * n) = SampleCartoonMeridionalScalar(u0, alpha, s);
          values(2 * n + 1) = 1;
        }
      });
  auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), values);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, host.data(), 2 * count, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
#endif
  for (int n = 0; n < count; ++n) {
    result[n].z = z[n];
    result[n].lapse = host(2 * n);
    result[n].valid =
        host(2 * n + 1) == 1 && std::isfinite(host(2 * n)) && host(2 * n) >= 0;
  }
  return result;
}

M0CandidateSummary CartoonM0FastFlow::SearchCandidate(const std::string& branch,
                                                      const Real center_z,
                                                      const Real radius,
                                                      const std::vector<Real>& seed) {
  return SolveM0Refined(
      [this](const std::vector<std::array<Real, 2>>& points) {
        return SampleAdmBatch(points);
      },
      solve_options_, l_start_, branch, center_z, radius, seed);
}

void CartoonM0FastFlow::Find(const int cycle, const Real time, const bool force) {
  if (!force && !ShouldSearch(cycle, time)) return;
  last_search_cycle_ = cycle;
  last_search_time_ = time;
  candidates_.clear();
  selected_.clear();
  const auto& tracking = last_good_.empty() ? last_trial_ : last_good_;
  for (const auto& good : tracking)
    candidates_.push_back(SearchCandidate(good.branch, good.center_z,
                                          good.fresh_initial_radius, good.coefficients));
  const bool seed_only = pin_->GetOrAddBoolean("fastflow", "horizon_only", false) &&
                         pin_->GetOrAddBoolean("fastflow", "mots_seed_only", false);
  if (seed_only && candidates_.empty())
    throw std::runtime_error("mots_seed_only requires an analysis seed");
  bool tracking_ok = false;
  for (const auto& candidate : candidates_)
    tracking_ok = tracking_ok ||
                  (candidate.area > 0 && candidate.direct_residual < tracking_residual_);
  // Outermost selection requires the full seed bank at each scheduled search.
  // Tracking is an additional guess, never a substitute for discovery.
  const bool discover = outermost_selection_ || force || !tracking_ok ||
                        search_count_ % discovery_interval_ == 0;
  ++search_count_;
  if (!seed_only && discover) {
    std::vector<Real> axis_z{0.0};
    if (pack_->pz4c->layout.centering == Z4cGridCentering::vertex) {
      // Sample every active native vertex on the whole symmetry axis, not a
      // uniform coarse scan which can miss a narrow displaced lapse minimum.
      const auto* mesh = pack_->pmesh;
      for (int gid = 0; gid < mesh->nmb_total; ++gid) {
        const auto& loc = mesh->lloc_eachmb[gid];
        if (loc.lx1 != 0) continue;
        Real r0, r1, z0, z1;
        meridional_detail::LogicalEdges(*mesh, loc, &r0, &r1, &z0, &z1);
        for (int j = 0; j <= mesh->mb_indcs.nx2; ++j)
          axis_z.push_back(z0 + (z1 - z0) * j / mesh->mb_indcs.nx2);
      }
    } else {
      for (int i = -axis_search_samples_; i <= axis_search_samples_; ++i)
        axis_z.push_back(axis_search_bound_ * i / axis_search_samples_);
    }
    std::sort(axis_z.begin(), axis_z.end());
    axis_z.erase(std::unique(axis_z.begin(), axis_z.end()), axis_z.end());
    const auto axis = SampleAxisLapseBatch(axis_z);
    std::vector<Real> centers{0.0};
    Real minimum = std::numeric_limits<Real>::infinity(), origin = minimum;
    for (const auto& a : axis)
      if (a.valid) {
        minimum = std::min(minimum, a.lapse);
        if (a.z == 0) origin = a.lapse;
      }
    const Real tie = std::max(Real(1.e-14), minimum * 1.e-8);
    if (origin > minimum + tie) {
      for (std::size_t n = 0; n < axis.size() && centers.size() < 3; ++n) {
        const auto& a = axis[n];
        if (!a.valid || a.lapse > minimum + tie || a.z == 0) continue;
        if (n && axis[n - 1].valid && axis[n - 1].lapse <= a.lapse) continue;
        if (n + 1 < axis.size() && axis[n + 1].valid && axis[n + 1].lapse < a.lapse)
          continue;
        centers.push_back(a.z);
      }
    }
    if (global_variable::my_rank == 0) {
      std::ofstream seeds(pin_->GetString("job", "basename") + ".mots_centers.csv",
                          std::ios::app);
      if (seeds.tellp() == 0)
        seeds << "cycle,time,seed_center_z,seed_lapse,global_axis_min_lapse,kind\n";
      seeds << std::setprecision(17);
      for (Real center : centers) {
        const auto at =
            std::find_if(axis.begin(), axis.end(),
                         [&](const M0AxisSample& a) { return a.z == center; });
        seeds << cycle << ',' << time << ',' << center << ',' << at->lapse << ','
              << minimum << ',' << (center == 0 ? "origin" : "axial_minimum") << '\n';
      }
    }
    for (Real center : centers) {
      const auto geometry = SampleAdm(0, center);
      const auto lapse = SampleAxisLapse(center);
      Real lower = radius_min_ > 0 ? radius_min_ : 4 * geometry.spacing;
      if (!(lower > 0)) lower = initial_radius_ / 128;
      const Real upper =
          center == 0
              ? std::max(
                    initial_radius_,
                    origin_lapse_radius_factor_ *
                        std::abs(*std::max_element(
                            centers.begin(), centers.end(),
                            [](Real a, Real b) { return std::abs(a) < std::abs(b); })))
              : initial_radius_;
      const Real maximum = std::max(lower, std::abs(upper));
      for (int r = 0; r < radius_count_; ++r) {
        // Discovery asks for existence, not an enumeration of all radial seeds.
        // Start with the outer seed, which avoids costly outward drift from tiny
        // spheres. Still visit every distinct center independently.
        const int seed_index = solve_options_.candidate_policy ? radius_count_ - 1 - r : r;
        const Real radius =
            radius_count_ == 1
                ? maximum
                : lower * std::pow(maximum / lower, Real(seed_index) / (radius_count_ - 1));
        const std::string branch = center == 0 ? "origin" : center > 0 ? "plus" : "minus";
        auto candidate = SearchCandidate(branch, center, radius, {});
        candidate.axis_extremum_z = center;
        candidate.center_lapse = lapse.lapse;
        if (candidate.verified && lmax_ >= 1) {
          const Real shift = candidate.coefficients[1] * std::sqrt(3 / (4 * kPi));
          auto shape = candidate.coefficients;
          shape[1] = 0;
          auto recentered = SearchCandidate(branch, center + shift, radius, shape);
          candidate = PreferM0Recentered(candidate, recentered);
        }
        candidates_.push_back(std::move(candidate));
        if (solve_options_.candidate_policy && !outermost_selection_ &&
            (candidates_.back().verified || candidates_.back().angular_candidate))
          break;
      }
    }
  }
  // An individual verified MOTS is sufficient for this finder milestone.
  // The optional pair label cannot veto a successful component.
  auto eligible = candidates_;
  for (auto& candidate : eligible)
    candidate.converged = candidate.verified || (solve_options_.candidate_policy &&
                                                 candidate.angular_candidate);
  int plus = -1, minus = -1;
  if (!outermost_selection_ && mode_ == "mirror_pair" &&
      SelectM0MirrorPair(eligible, pair_tolerance_, &plus, &minus))
    selected_ = {plus, minus};
  else {
    int best = outermost_selection_ ? SelectM0Outermost(eligible) : SelectM0Single(eligible);
    if (best >= 0) selected_.push_back(best);
  }
  if (outermost_selection_ && global_variable::my_rank == 0) {
    int count = 0;
    for (const auto& c : eligible) if (c.converged) ++count;
    std::ofstream selection(pin_->GetString("job", "basename") + ".mots_selection.csv", std::ios::app);
    if (selection.tellp() == 0) selection << "cycle,time,accepted_trials,selected,status\n";
    selection << std::setprecision(17) << cycle << ',' << time << ',' << count << ','
              << (selected_.empty() ? -1 : selected_[0]) << ','
              << (count && selected_.empty() ? "ambiguous_enclosure" :
                  count ? "outermost_discovered" : "no_candidate") << '\n';
  }
  found_ = !selected_.empty();
  if (found_) {
    if (time_first_found_ < 0) time_first_found_ = time;
    last_good_.clear();
    for (int n : selected_) last_good_.push_back(candidates_[n]);
  }
  last_trial_.clear();
  for (const auto& candidate : candidates_)
    if (candidate.area > 0 && std::isfinite(candidate.direct_residual) &&
        (last_trial_.empty() ||
         candidate.direct_residual < last_trial_[0].direct_residual))
      last_trial_ = {candidate};
  Capture();
}

void CartoonM0FastFlow::Restore() {
  const auto& state = pack_->z4c_restart_state.fastflow;
  if (state.surface_mode == "none") return;
  std::string reason;
  if (!ValidateM0RestartState(state, lmax_, &reason))
    throw std::runtime_error("invalid Cartoon m=0 restart state: " + reason);
  if (mode_ != state.surface_mode &&
      !(mode_ == "mirror_pair" && state.surface_mode == "single" && state.converged))
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
  auto& state = pack_->z4c_restart_state.fastflow;
  state = {};
  state.surface_mode = found_ && selected_.size() == 1 ? "single" : mode_;
  state.last_search_cycle = last_search_cycle_;
  state.last_search_time = last_search_time_;
  state.time_first_found = time_first_found_;
  state.converged = found_;
  state.status =
      found_ ? (StrictlyVerified() ? "accepted" : "angular_candidate") : "failed";
  if (!found_) {
    state.failure_code =
        !candidates_.empty() && candidates_[0].failure == "axis_lapse_scan_coverage"
            ? "axis_lapse_scan_coverage"
            : (mode_ == "mirror_pair" ? "pair_incomplete" : "no_candidate");
    return;
  }
  state.failure_code = "none";
  state.center_count = static_cast<int>(selected_.size());
  state.selected_branch =
      selected_.size() == 2 ? "plus_minus" : candidates_[selected_[0]].branch;
  state.center_z0 = candidates_[selected_[0]].center_z;
  if (selected_.size() == 2) state.center_z1 = candidates_[selected_[1]].center_z;
  for (const int index : selected_) {
    state.coefficients.insert(state.coefficients.end(),
                              candidates_[index].coefficients.begin(),
                              candidates_[index].coefficients.end());
  }
}

void CartoonM0FastFlow::Write(const int cycle, const Real time) {
  if (pin_->GetOrAddBoolean("fastflow", "horizon_only", false) ||
      pin_->GetOrAddBoolean("fastflow", "mots_write_profiles", false)) {
    // Every rank participates in geometry sampling; only rank zero writes.
    for (std::size_t id = 0; id < candidates_.size(); ++id) {
      const auto& s = candidates_[id];
      if (s.coefficients.size() != static_cast<std::size_t>(lmax_ + 1)) continue;
      const int count =
          std::max(4 * solve_options_.ntheta + 3,
                   pin_->GetOrAddInteger("fastflow", "mots_profile_points", 0));
      const auto quad = GaussLegendre(count);
      std::vector<std::array<Real, 2>> points;
      std::vector<std::array<Real, 3>> shapes;
      for (int n = 0; n < count; ++n) {
        const Real theta = std::acos(quad[n].first);
        std::array<Real, 3> h{};
        for (int l = 0; l <= lmax_; ++l) {
          const auto harmonic = M0Harmonic(l, theta);
          const Real y = harmonic[0], dy = harmonic[1], ddy = harmonic[2];
          h[0] += s.coefficients[l] * y;
          h[1] += s.coefficients[l] * dy;
          h[2] += s.coefficients[l] * ddy;
        }
        shapes.push_back(h);
        points.push_back({h[0] * std::sin(theta), s.center_z + h[0] * std::cos(theta)});
      }
      const auto geometry = SampleAdmBatch(points);
      const auto dense = EvaluateM0(
          [this](const std::vector<std::array<Real, 2>>& p) { return SampleAdmBatch(p); },
          count, s);
      if (global_variable::my_rank == 0) {
        std::ofstream check(pin_->GetString("job", "basename") + ".mots_dense_" +
                            std::to_string(cycle) + "_" + std::to_string(id) + ".json");
        check << std::setprecision(17) << "{\"points\":" << count
              << ",\"valid\":" << (dense.valid ? "true" : "false") << ",\"area\":";
        if (dense.valid) {
          check << dense.surface.area << ",\"epsilon2\":" << dense.surface.direct_residual
                << ",\"epsilon_inf\":" << dense.surface.epsilon_inf;
        } else {
          check << "null,\"epsilon2\":null,\"epsilon_inf\":null";
        }
        check << "}\n";
        const std::string name = pin_->GetString("job", "basename") + ".mots_surface_" +
                                 std::to_string(cycle) + "_" + std::to_string(id) +
                                 ".csv";
        std::ofstream file(name);
        file << "theta,rho,z,radius,theta_plus,theta_minus,spacing,valid\n"
             << std::setprecision(17);
        for (int n = 0; n < count; ++n) {
          const Real theta = std::acos(quad[n].first);
          const auto& h = shapes[n];
          const auto p = EvaluateM0SurfacePoint(theta, h[0], h[1], h[2], geometry[n]);
          file << theta << ',' << points[n][0] << ',' << points[n][1] << ',' << h[0]
               << ',' << p.expansion << ',' << p.ingoing_expansion << ','
               << geometry[n].spacing << ',' << p.valid << '\n';
        }
      }
    }
  }
  if (global_variable::my_rank != 0) return;
  {
    const std::string name =
        pin_->GetString("job", "basename") + ".mots_angular_stages.csv";
    std::ofstream stages(name, std::ios::app);
    if (stages.tellp() == 0)
      stages << "cycle,time,candidate,lmax,ntheta,iterations,epsilon2,epsilon_inf,area,"
                "verified,failure\n";
    stages << std::setprecision(17);
    for (std::size_t i = 0; i < candidates_.size(); ++i)
      for (const auto& stage : candidates_[i].angular_stages)
        stages << cycle << ',' << time << ',' << i << ',' << stage.lmax << ','
               << stage.ntheta << ',' << stage.iterations << ',' << stage.epsilon2 << ','
               << stage.epsilon_inf << ',' << stage.area << ',' << stage.verified << ','
               << stage.failure << '\n';
  }
  {
    std::ofstream events(pin_->GetString("job", "basename") + ".mots_candidates.csv",
                         std::ios::app);
    if (events.tellp() == 0)
      events << "cycle,time,id,center_z,solved_lmax,angular_candidate,strict_verified,"
                "policy_accepted,epsilon2,epsilon_inf,area,failure\n";
    events << std::setprecision(17);
    for (std::size_t i = 0; i < candidates_.size(); ++i) {
      const auto& s = candidates_[i];
      events << cycle << ',' << time << ',' << i << ',' << s.center_z << ','
             << s.solved_lmax << ',' << s.angular_candidate << ',' << s.verified << ','
             << (std::find(selected_.begin(), selected_.end(), int(i)) != selected_.end())
             << ',' << s.direct_residual << ',' << s.epsilon_inf << ',' << s.area << ','
             << s.failure << '\n';
    }
  }
  if (output_ == nullptr) {
    const std::string path = pin_->GetString("job", "basename") + ".cartoon_m0_horizon_" +
                             std::to_string(horizon_) + ".txt";
    output_ = std::fopen(path.c_str(), "a");
    if (output_ == nullptr) throw std::runtime_error("cannot open Cartoon m=0 output");
    std::fseek(output_, 0, SEEK_END);
    if (std::ftell(output_) == 0) {
      std::fprintf(output_,
                   "# cycle time branch solver_converged center_z axis_extremum_z "
                   "center_lapse area irreducible_mass horizon_mass "
                   "spin_z mean_radius "
                   "minimum_radius direct_residual flow_residual failure "
                   "fresh_initial_radius a_l0... verified epsilon_inf ingoing_min "
                   "ingoing_max spacing iterations\n");
    }
  }
  for (const auto& candidate : candidates_) {
    std::fprintf(output_,
                 "%d %.17e %s %d %.17e %.17e %.17e %.17e %.17e %.17e "
                 "%.17e %.17e %.17e %.17e %.17e %s",
                 cycle, time, candidate.branch.c_str(), candidate.converged,
                 candidate.center_z, candidate.axis_extremum_z, candidate.center_lapse,
                 candidate.area, candidate.irreducible_mass, candidate.mass,
                 candidate.spin_z, candidate.mean_radius, candidate.minimum_radius,
                 candidate.direct_residual, candidate.flow_residual,
                 candidate.failure.c_str());
    std::fprintf(output_, " %.17e", candidate.fresh_initial_radius);
    for (const Real coefficient : candidate.coefficients)
      std::fprintf(output_, " %.17e", coefficient);
    std::fprintf(output_, " %d %.17e %.17e %.17e %.17e %d\n", candidate.verified,
                 candidate.epsilon_inf, candidate.ingoing_min, candidate.ingoing_max,
                 candidate.spacing, candidate.iterations);
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
