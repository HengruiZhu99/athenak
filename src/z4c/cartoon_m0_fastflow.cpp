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

void ReflectM0AdmSample(M0AdmSample &s) {
  // Sampler returns physical Cartesian (rho, azimuth, axial) components.
  for (int k = 0; k < 6; ++k) {
    const int sign = k == 2 || k == 4 ? -1 : 1;
    s.metric[k] *= sign;
    s.curvature[k] *= sign;
    for (int d = 0; d < 3; ++d)
      s.metric_derivative[6*d+k] *= d == 2 ? -sign : sign;
  }
}

M0AdmSample RotateM0AdmSample(const M0AdmSample &input, const Real phi) {
  if (!input.valid) return input;
  const Real c = std::cos(phi);
  const Real s = std::sin(phi);
  const Real rotation[3][3] = {{c, -s, 0.0}, {s, c, 0.0}, {0.0, 0.0, 1.0}};
  M0AdmSample output;
  output.valid = true;
  output.spacing = input.spacing;
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
  if (!(theta >= 0 && theta <= kPi)) return result;
  const bool pole = theta == 0 || theta == kPi;
  if (pole && std::abs(radius_theta) > 128*std::numeric_limits<Real>::epsilon()*
      (radius + std::abs(radius_theta_theta))) return result;
  const Real st = pole ? 0 : std::sin(theta);
  const Real ct = std::cos(theta);
  Real g[3][3], curvature[3][3], inverse[3][3];
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      g[a][b] = sample.metric[PackedIndex(a, b)];
      curvature[a][b] = sample.curvature[PackedIndex(a, b)];
    }
  }
  const Real det = adm::SpatialDet(g[0][0], g[0][1], g[0][2], g[1][1],
                                   g[1][2], g[2][2]);
  if (!(g[0][0] > 0.0 && g[0][0]*g[1][1] > g[0][1]*g[0][1]) ||
      !std::isfinite(det) || det <= 0.0) return result;
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
  ddtheta[1][1] = pole ? 0 : z / (x * radius * radius);
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
    // In the regular chart, F = r - h(theta) has equal transverse Hessians:
    // F_xx = F_yy = 1/h - h''/h^2. The mixed axial derivatives vanish.
    dF[0] = 0;
    ddF[0][0] = ddF[1][1] = invr - radius_theta_theta*invr*invr;
    ddF[0][2] = ddF[2][0] = 0;
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
  const Real mean_curvature = (laplacian - normal_hessian * invu * invu) * invu;
  const Real projected_k = normal_k * invu * invu - trace_k;
  result.expansion = projected_k + mean_curvature;
  result.ingoing_expansion = projected_k - mean_curvature;
  result.gradient_norm = 1.0 / invu;
  result.flow_residual = result.expansion * result.gradient_norm;
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
  if (!std::isfinite(det_h) || (!pole && det_h <= 0.0) || !std::isfinite(result.expansion)) {
    return M0SurfacePoint{};
  }
  result.area_factor = std::sqrt(det_h);
  result.valid = std::isfinite(result.ingoing_expansion) &&
                 std::isfinite(result.flow_residual) && std::isfinite(result.spin_integrand_z);
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

namespace {
// Differentiate the Legendre recurrence in theta. This also gives regular
// first and second derivatives near either symmetry pole without dividing by sin(theta).
struct M0Node {
  Real theta, weight;
  std::vector<std::array<Real, 3>> basis;
  M0Node(Real angle, Real w, int count) : theta(angle), weight(w), basis(count) {
    const Real x = std::cos(theta), s = std::sin(theta);
    std::array<Real, 3> prev{}, cur{1, 0, 0};
    for (int l = 0; l < count; ++l) {
      const Real norm = std::sqrt((2*l + 1)/(4*kPi));
      for (int k = 0; k < 3; ++k) basis[l][k] = norm * cur[k];
      const Real a = (2*l + 1.0)/(l + 1.0), b = l/(l + 1.0);
      std::array<Real, 3> next = {a*x*cur[0] - b*prev[0],
          a*(x*cur[1] - s*cur[0]) - b*prev[1],
          a*(x*cur[2] - 2*s*cur[1] - x*cur[0]) - b*prev[2]};
      prev = cur;
      cur = next;
    }
  }

  std::array<Real, 3> profile(const std::vector<Real> &c) const {
    std::array<Real, 3> h{};
    for (std::size_t l = 0; l < c.size(); ++l)
      for (int k = 0; k < 3; ++k) h[k] += c[l]*basis[l][k];
    return h;
  }
};

// Fold each weighted row into an upper triangular least-squares system with
// Givens rotations. Only lmax+1 unknowns, independent of the mesh and rank count.
void InsertRow(std::vector<std::vector<Real>> &qr, std::vector<Real> &row) {
  const int count = qr.size();
  for (int j = 0; j < count; ++j) {
    const Real norm = std::hypot(qr[j][j], row[j]);
    if (norm == 0) continue;
    const Real c = qr[j][j]/norm, s = row[j]/norm;
    for (int k = j; k <= count; ++k) {
      const Real top = qr[j][k], bottom = row[k];
      qr[j][k] = c*top + s*bottom;
      row[k] = c*bottom - s*top;
    }
  }
}

const char *M0ExpansionFailure(const std::vector<Real> &coeff, Real center, int ntheta,
                       Real target, Real tolerance, Real area_radius, const M0Sampler &sample) {
  // Uniform meridional angles interleave the quadrature, include the equator
  // and approach both regular poles. Angular-average convergence alone can
  // hide an untrapped cap. Allow the time-symmetric bifurcation sphere.
  const int intervals = 2*std::max(ntheta, static_cast<int>(coeff.size()));
  const Real cap = 1.e-4;
  std::vector<Real> angles{0., kPi};
  for (const auto &point : GaussLegendre(ntheta)) angles.push_back(std::acos(point.first));
  for (int i = 0; i <= intervals; ++i) angles.push_back(cap + (kPi - 2*cap)*i/intervals);
  std::vector<std::array<Real, 3>> profiles;
  M0Locations locations;
  for (const Real theta : angles) {
    auto h = M0Node(theta, 0, coeff.size()).profile(coeff);
    if (!(h[0] > 0) || !std::isfinite(h[0])) return "nonpositive_radius";
    profiles.push_back(h);
    locations.push_back({theta == kPi ? 0 : h[0]*std::sin(theta), center + h[0]*std::cos(theta)});
  }
  const auto adm = sample(locations);
  for (std::size_t i = 0; i < angles.size(); ++i) {
    const auto &h = profiles[i];
    const auto p = EvaluateM0SurfacePoint(angles[i], h[0], h[1], h[2], adm[i]);
    const Real length = std::max(area_radius, h[0]);
    if (!p.valid) return "invalid_surface_geometry";
    if (h[0] < 2*adm[i].spacing) return "underresolved_surface";
    if (length*std::abs(p.expansion - target/h[0]) > tolerance ||
        (target < 0 && p.expansion >= 0)) return "pointwise_expansion";
    if (target < 0 ? p.ingoing_expansion >= 0 : length*p.ingoing_expansion > tolerance)
      return "ingoing_expansion";
  }
  return nullptr;
}
}  // namespace

M0CandidateSummary RefineM0Surface(const std::vector<Real> &seed, Real center,
    int ntheta, int iterations, Real target, Real direct_tol, Real flow_tol,
    Real mass_tol, const M0Sampler &sample, bool equatorial) {
  M0CandidateSummary result;
  result.coefficients = seed;
  result.center_z = center;
  const int count = seed.size();
  const int stride = equatorial && center == 0 ? 2 : 1;
  const int unknowns = (count + stride - 1)/stride;
  if (stride == 2) {
    for (int l = 1; l < count; l += 2) result.coefficients[l] = 0;
  }
  if (count == 0 || ntheta < count || ntheta < 2 || iterations < 2) {
    result.failure = "insufficient_collocation";
    return result;
  }
  std::vector<M0Node> nodes;
  const auto quadrature = GaussLegendre(ntheta);
  for (int n = stride == 2 ? ntheta/2 : 0; n < ntheta; ++n) {
    const auto &point = quadrature[n];
    const bool has_partner = stride == 2 && n != ntheta-1-n;
    nodes.emplace_back(std::acos(point.first), point.second*(has_partner ? 2 : 1), count);
  }
  const Real diff = std::cbrt(std::numeric_limits<Real>::epsilon());
  using Matrix = std::vector<std::vector<Real>>;
  auto evaluate = [&](M0CandidateSummary &candidate, Matrix *qr) {
    Real cost = 0, area = 0, spin = 0, mean = 0, direct2 = 0, flow2 = 0;
    Real minimum = std::numeric_limits<Real>::infinity();
    std::vector<std::array<Real, 3>> profiles;
    M0Locations locations;
    for (const auto &node : nodes) {
      const auto h = node.profile(candidate.coefficients);
      const Real r = h[0];
      if (!(r > 0) || !std::isfinite(r)) return std::numeric_limits<Real>::infinity();
      profiles.push_back(h);
      locations.push_back({r*std::sin(node.theta), center + r*std::cos(node.theta)});
      if (qr) {
        const Real trial = r + diff*r;
        locations.push_back({trial*std::sin(node.theta), center + trial*std::cos(node.theta)});
      }
    }
    const auto samples = sample(locations);
    for (size_t n = 0; n < nodes.size(); ++n) {
      const auto &node = nodes[n];
      const auto &h = profiles[n];
      const Real r = h[0], theta = node.theta, w = node.weight;
      const auto &adm = samples[qr ? 2*n : n];
      const auto p = EvaluateM0SurfacePoint(theta, r, h[1], h[2], adm);
      if (!p.valid) return std::numeric_limits<Real>::infinity();
      const Real f = r*p.expansion - target;
      const Real da = 2*kPi*w*p.area_factor/std::sin(theta);
      cost += w*f*f;
      area += da;
      spin += da*p.spin_integrand_z;
      mean += 0.5*w*r;
      minimum = std::min(minimum, r);
      direct2 += da*(f/r)*(f/r);
      const Real flow = (f/r)*p.gradient_norm;
      flow2 += 0.5*w*flow*flow;
      if (!qr) continue;
      std::array<Real, 3> derivative{};
      for (int k = 0; k < 3; ++k) {
        auto trial = h;
        const Real step = diff*std::max(r, std::abs(h[k]));
        trial[k] += step;
        const auto perturbed = EvaluateM0SurfacePoint(theta, trial[0], trial[1], trial[2],
                                                      k == 0 ? samples[2*n+1] : adm);
        if (!perturbed.valid) return std::numeric_limits<Real>::infinity();
        derivative[k] = (trial[0]*perturbed.expansion - r*p.expansion)/step;
      }
      std::vector<Real> row(unknowns + 1, 0);
      for (int l = 0; l < count; l += stride)
        for (int k = 0; k < 3; ++k) row[l/stride] += std::sqrt(w)*derivative[k]*node.basis[l][k];
      row[unknowns] = -std::sqrt(w)*f;
      InsertRow(*qr, row);
    }
    candidate.minimum_radius = minimum;
    candidate.mean_radius = mean;
    candidate.area = area;
    candidate.spin_z = spin/(8*kPi);
    candidate.irreducible_mass = std::sqrt(area/(16*kPi));
    candidate.mass = M0HorizonMass(area, candidate.spin_z);
    candidate.direct_residual = mean*std::sqrt(direct2/area);
    candidate.flow_residual = mean*std::sqrt(flow2);
    return FiniteSummary(candidate) ? cost : std::numeric_limits<Real>::infinity();
  };

  Real previous_mass = 0;
  for (int iter = 0; iter < iterations; ++iter) {
    Matrix qr(unknowns, std::vector<Real>(unknowns + 1, 0));
    const Real cost = evaluate(result, &qr);
    if (!std::isfinite(cost)) { result.failure = "invalid_meridional_sample"; break; }
    const char *validation_failure = nullptr;
    if (iter && RelativeDifference(previous_mass, result.irreducible_mass) < mass_tol &&
        result.direct_residual < direct_tol && result.flow_residual < flow_tol) {
      validation_failure = M0ExpansionFailure(result.coefficients, center, ntheta, target,
                                              direct_tol, 2*result.irreducible_mass, sample);
      if (!validation_failure) {
        result.converged = true;
        result.failure = "none";
        return result;
      }
      // Further optimization of a sub-cell surface cannot resolve its geometry.
      if (std::string(validation_failure) == "underresolved_surface") {
        result.failure = validation_failure;
        return result;
      }
    }
    previous_mass = result.irreducible_mass;
    // Give an initially exact seed a second mass observation.
    if (iter == 0 && cost < std::numeric_limits<Real>::epsilon()) continue;
    std::vector<Real> step(count, 0);
    Real diagonal = 0;
    for (int j = 0; j < unknowns; ++j) diagonal = std::max(diagonal, std::abs(qr[j][j]));
    for (int j = unknowns-1; j >= 0; --j) {
      if (std::abs(qr[j][j]) <= unknowns*std::numeric_limits<Real>::epsilon()*diagonal) {
        result.failure = "singular_meridional_step";
        return result;
      }
      Real rhs = qr[j][unknowns];
      for (int k = j+1; k < unknowns; ++k) rhs -= qr[j][k]*step[stride*k];
      step[stride*j] = rhs/qr[j][j];
    }
    Real fraction = 1;
    for (const auto &node : nodes) {
      const Real delta = std::abs(node.profile(step)[0]);
      if (delta > 0) fraction = std::min(fraction, 0.5*node.profile(result.coefficients)[0]/delta);
    }
    bool advanced = false;
    for (int backtrack = 0; backtrack < 8; ++backtrack, fraction *= 0.5) {
      auto trial = result;
      for (int l = 0; l < count; ++l) trial.coefficients[l] += fraction*step[l];
      if (evaluate(trial, nullptr) < cost) {
        result = std::move(trial);
        advanced = true;
        break;
      }
    }
    if (!advanced) {
      result.failure = validation_failure ? validation_failure : "meridional_stagnation";
      break;
    }
    result.failure = validation_failure ? validation_failure : "iteration_limit";
  }
  return result;
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

// |P_l| <= 1 gives conservative coordinate-radius bounds, including the
// unsampled caps. Only discard an inner candidate when containment is certain.
static std::pair<Real, Real> M0RadiusBounds(const std::vector<Real> &c) {
  Real variation = 0.0;
  for (std::size_t l = 1; l < c.size(); ++l)
    variation += std::abs(c[l])*std::sqrt((2*l+1)/(4*kPi));
  const Real mean = c[0]/std::sqrt(4*kPi);
  return {mean - variation, mean + variation};
}

int SelectM0Single(const std::vector<M0CandidateSummary> &candidates) {
  int selected = -1;
  for (int index = 0; index < static_cast<int>(candidates.size()); ++index) {
    const auto &candidate = candidates[index];
    if (!candidate.converged || !FiniteSummary(candidate)) continue;
    const Real upper = M0RadiusBounds(candidate.coefficients).second;
    const bool enclosed = std::any_of(candidates.begin(), candidates.end(), [&](const auto &outer) {
      return &outer != &candidate && outer.converged && FiniteSummary(outer) &&
          upper + std::abs(candidate.center_z - outer.center_z) <
          M0RadiusBounds(outer.coefficients).first*(1-64*std::numeric_limits<Real>::epsilon());
    });
    if (enclosed) continue;
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
  // A mirror pair must occupy opposite half-spaces. Two centers can
  // otherwise converge onto the same common horizon and pass the mass tests.
  auto in_half_space = [](const M0CandidateSummary &candidate, int sign) {
    const auto &c = candidate.coefficients;
    Real slope_bound = 0;
    for (std::size_t l = 0; l < c.size(); ++l) {
      // |d(mu P_l)/dmu| <= 1 + l(l+1)/2 on [-1,1].
      slope_bound += std::abs(c[l])*std::sqrt((2*l+1)/(4*kPi))*(1 + 0.5*l*(l+1));
    }
    const int intervals = std::max(128, 16*static_cast<int>(c.size()));
    for (int i = 0; i < intervals; ++i) {
      const Real mu = -1 + (2*i+1.0)/intervals;
      const Real radius = M0Node(std::acos(mu), 0, c.size()).profile(c)[0];
      if (sign*(candidate.center_z + mu*radius) <= slope_bound/intervals) return false;
    }
    return true;
  };
  if (!in_half_space(p, 1) || !in_half_space(m, -1)) return false;
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
  equatorial_ = pack_->pmesh->mesh_bcs[BoundaryFace::inner_x2] == BoundaryFlag::reflect &&
                pack_->pmesh->mesh_size.x2min == 0;
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
  const Real inner_target = pin->GetOrAddReal("fastflow", "cartoon_inner_target_" + suffix, 0.0);
  if (!std::isfinite(inner_target) || inner_target > 0.0)
    throw std::runtime_error("cartoon_inner_target must be zero or negative");
  mode_ = pin->GetOrAddString("fastflow", "cartoon_surface_mode_" + suffix, "single");
  if (!pin->GetOrAddBoolean("fastflow", "cartoon_search_offcenter_" + suffix, true) &&
      mode_ == "mirror_pair") {
    throw std::runtime_error("Cartoon mirror_pair requires off-center searches");
  }
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
    const M0Node node(theta_[n], weights_[n], lmax_ + 1);
    for (int l = 0; l <= lmax_; ++l) {
      const int i = n * (lmax_ + 1) + l;
      y0_[i] = node.basis[l][0];
      dy0_[i] = node.basis[l][1];
      ddy0_[i] = node.basis[l][2];
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

namespace {
std::vector<M0AdmSample> SampleNativeAdm(MeshBlockPack *pack_, const M0Locations &points) {
  std::vector<M0AdmSample> result(points.size());
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
    for (int v = 0; v < 30; ++v)
      if (!std::isfinite(packed[n * 32 + v])) {
        r.valid = false;
      }
  }
  return result;
}
}  // namespace

std::vector<M0AdmSample> CartoonM0FastFlow::SampleAdm(const M0Locations &locations) const {
  if (!equatorial_) return SampleStoredAdm(locations);
  M0Locations folded = locations;
  for (auto &point : folded) point[1] = std::abs(point[1]);
  auto samples = SampleStoredAdm(folded);
  for (size_t n = 0; n < locations.size(); ++n) {
    if (locations[n][1] >= 0) continue;
    auto &s = samples[n];
    ReflectM0AdmSample(s);
  }
  return samples;
}

std::vector<M0AdmSample> CartoonM0FastFlow::SampleStoredAdm(const M0Locations &locations) const {
  if (pack_->pz4c->layout.centering == Z4cGridCentering::vertex &&
      pin_->GetOrAddBoolean("fastflow", "cartoon_native_geometry", false))
    return SampleNativeAdm(pack_, locations);
  const int count = locations.size();
  std::vector<M0AdmSample> result(count);
  const int fd_stencil = pack_->z4c_symmetry.stencil_width;
  if (count == 0 || fd_stencil < 2 || fd_stencil > 4 ||
      pack_->pmesh->mb_indcs.ng < fd_stencil) return result;
  Kokkos::View<CartoonMeridionalStencil *> stencils("m0 surface stencils", count);
  auto host_stencils = Kokkos::create_mirror_view(stencils);
  for (int n = 0; n < count; ++n) {
    host_stencils(n) = LocateCartoonMeridionalPoint(pack_->pmesh, locations[n][0], locations[n][1]);
  }
  Kokkos::deep_copy(stencils, host_stencils);
  // One launch and one collective for an entire meridian. The last value
  // verifies exactly one contributing rank, also for points on AMR interfaces.
  Kokkos::View<Real *[32], Kokkos::LayoutRight> values("m0 ADM samples", count);
  const int rank = global_variable::my_rank;
  auto metric = pack_->padm->adm.g_dd;
  auto curvature = pack_->padm->adm.vK_dd;
  auto size = pack_->pmb->mb_size.d_view;
  const auto indices = pack_->pmesh->mb_indcs;
  Kokkos::parallel_for("Cartoon m0 ADM interpolate",
      Kokkos::RangePolicy<DevExeSpace>(0, count), KOKKOS_LAMBDA(const int n) {
    const auto stencil = stencils(n);
    if (!stencil.valid || stencil.owner_rank != rank) return;
    values(n, 30) = Kokkos::max(size(stencil.local_block).dx1, size(stencil.local_block).dx2);
    values(n, 31) = 1.0;
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
        values(n, packed) += weight * metric(stencil.local_block, ca, cb,
                                           stencil.k, j, i);
        values(n, 6 + packed) += weight * curvature(stencil.local_block, ca, cb,
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
          values(n, 12 + d * 6 + packed) += weight * derivative_value;
        }
      }
    }
  });
  Kokkos::fence();
  auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), values);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, host.data(), 32*count, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  for (int n = 0; n < count; ++n) {
    auto &point = result[n];
    point.spacing = host(n, 30);
    point.valid = host(n, 31) == 1.0 && point.spacing > 0;
    for (int k = 0; k < 30; ++k) point.valid = point.valid && std::isfinite(host(n, k));
    for (int k = 0; k < 6; ++k) {
      point.metric[k] = host(n, k);
      point.curvature[k] = host(n, 6+k);
    }
    for (int k = 0; k < 18; ++k) point.metric_derivative[k] = host(n, 12+k);
  }
  return result;
}

std::vector<M0AxisSample> CartoonM0FastFlow::SampleAxisLapse(const std::vector<Real> &z) const {
  const int count = z.size();
  Kokkos::View<CartoonMeridionalStencil *> stencils("m0 axis stencils", count);
  auto on_host = Kokkos::create_mirror_view(stencils);
  // The lapse uses the native Z4c layout, including the rho=0 vertex.

  for (int n = 0; n < count; ++n)
    on_host(n) = LocateNativeCartoonMeridionalPoint(pack_->pmesh, 0.0,
                                                    equatorial_ ? std::abs(z[n]) : z[n]);
  Kokkos::deep_copy(stencils, on_host);
  Kokkos::View<Real *[2], Kokkos::LayoutRight> values("m0 axis lapse samples", count);
  const int rank = global_variable::my_rank;
  const auto u0 = pack_->pz4c->u0;
  const int alpha = pack_->pz4c->I_Z4C_ALPHA;
  Kokkos::parallel_for("Cartoon axis lapse interpolate",
      Kokkos::RangePolicy<DevExeSpace>(0, count), KOKKOS_LAMBDA(const int n) {
        const auto stencil = stencils(n);
        if (!stencil.valid || stencil.owner_rank != rank) return;
        values(n, 0) = SampleCartoonMeridionalScalar(u0, alpha, stencil);
        values(n, 1) = 1.0;
      });
  Kokkos::fence();
  auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), values);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, host.data(), 2*count, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  std::vector<M0AxisSample> samples(count);
  for (int n = 0; n < count; ++n) {
    const Real lapse = host(n, 0);
    samples[n] = {host(n, 1) == 1.0 && std::isfinite(lapse) && lapse >= 0.0, z[n], lapse};
  }
  return samples;
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
  const bool even_shape = equatorial_ && branch == "origin" && center_z == 0;
  if (even_shape) {
    for (int l = 1; l <= lmax_; l += 2) summary.coefficients[l] = 0;
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
    std::vector<std::array<Real, 3>> profiles(ntheta_);
    M0Locations locations;
    for (int n = 0; n < ntheta_; ++n) {
      Real radius = 0.0, first = 0.0, second = 0.0;
      for (int l = 0; l <= lmax_; ++l) {
        const int index = n * (lmax_ + 1) + l;
        radius += summary.coefficients[l] * y0_[index];
        first += summary.coefficients[l] * dy0_[index];
        second += summary.coefficients[l] * ddy0_[index];
      }
      if (!std::isfinite(radius) || radius <= 0.0) {
        summary.failure = "nonpositive_radius";
        return summary;
      }
      profiles[n] = {radius, first, second};
      locations.push_back({radius*std::sin(theta_[n]), center_z + radius*std::cos(theta_[n])});
    }
    const auto samples = SampleAdm(locations);
    for (int n = 0; n < ntheta_; ++n) {
      const auto [radius, first, second] = profiles[n];
      minimum = std::min(minimum, radius);
      mean += weights_[n] * radius * 0.5;
      const auto &adm = samples[n];
      const auto point = EvaluateM0SurfacePoint(theta_[n], radius, first, second, adm);
      if (!point.valid) {
        summary.failure = adm.valid ? "invalid_surface_geometry" : "invalid_adm_sample";
        return summary;
      }
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
    if (!std::isfinite(area) || area <= 0.0 ||
        !std::isfinite(spin_z) || !std::isfinite(direct2) ||
        !std::isfinite(flow2) || !std::isfinite(mean) || mean <= 0.0 ||
        !std::isfinite(minimum) || minimum <= 0.0) {
      summary.failure = "nonfinite_integral";
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
      if (const char *failure = M0ExpansionFailure(summary.coefficients, center_z, ntheta_,
              0.0, direct_tolerance_, 2*irreducible_mass,
              [this](const M0Locations &points) { return SampleAdm(points); })) {
        summary.failure = failure;
        return summary;
      }
      summary.converged = true;
      summary.failure = "none";
      return summary;
    }
    if (iteration + 1 == iterations_) break;  // Keep diagnostics and coefficients on one iterate.
    previous_mass = irreducible_mass;
    const Real alpha = flow_scale_;
    const Real beta = 0.5 * flow_scale_;
    for (int l = 0; l <= lmax_; ++l) {
      const Real factor = (alpha / (lmax_ * (lmax_ + 1.0)) + beta) /
                          (1.0 + (beta / alpha) * l * (l + 1.0));
      projection[l] = even_shape && l % 2 ? 0 : -factor * projection[l];
    }
    // Do not advance to a nonpositive trial radius. Hand an excessive step
    // to the safeguarded meridional correction at the last admissible surface.
    for (int n = 0; n < ntheta_; ++n) {
      Real radius = 0.0, delta = 0.0;
      for (int l = 0; l <= lmax_; ++l) {
        const Real y = y0_[n*(lmax_ + 1) + l];
        radius += y*summary.coefficients[l];
        delta += y*projection[l];
      }
      if (delta < -0.5*radius) {
        summary.failure = "excessive_radial_step";
        return summary;
      }
    }
    for (int l = 0; l <= lmax_; ++l) summary.coefficients[l] += projection[l];
  }
  summary.failure = "iteration_limit";
  return summary;
}

void CartoonM0FastFlow::Find(const int cycle, const Real time) {
  if (!ShouldSearch(cycle, time)) return;
  last_search_cycle_ = cycle;
  last_search_time_ = time;
  candidates_.clear();
  const bool offcenter = pin_->GetBoolean("fastflow", "cartoon_search_offcenter_0");
  std::vector<Real> positions{0.0};
  const int axis_samples_per_side = offcenter ? axis_search_samples_ : 0;
  for (int index = 1; index <= axis_samples_per_side; ++index) {
    const Real magnitude = axis_search_bound_ * index / axis_search_samples_;
    positions.push_back(-magnitude);
    positions.push_back(magnitude);
  }
  const auto axis_samples = SampleAxisLapse(positions);
  const auto &origin_sample = axis_samples.front();
  Real plus_center = 0.0, minus_center = 0.0;
  Real plus_lapse = 0.0, minus_lapse = 0.0;
  const bool plus_valid = !offcenter || SelectM0AxisLapseMinimum(
      axis_samples, 1, &plus_center, &plus_lapse);
  const bool minus_valid = !offcenter || SelectM0AxisLapseMinimum(
      axis_samples, -1, &minus_center, &minus_lapse);
  std::vector<std::pair<std::string, Real>> centers{{"origin", 0.0}};
  if (offcenter) {
    centers.emplace_back("plus", plus_center);
    centers.emplace_back("minus", minus_center);
  }
  if (!origin_sample.valid || !plus_valid || !minus_valid) {
    for (const auto &entry : centers) {
      M0CandidateSummary failed;
      failed.branch = entry.first;
      failed.failure = "axis_lapse_scan_coverage";
      candidates_.push_back(std::move(failed));
    }
    selected_.clear();
    found_ = false;
    Capture();
    return;
  }
  const Real origin_initial_radius = offcenter && adaptive_initial_radius_
      ? M0OriginInitialRadius(initial_radius_, origin_lapse_radius_factor_,
                              plus_center, minus_center)
      : initial_radius_;
  const Real pair_initial_radius = offcenter && adaptive_initial_radius_
      ? M0DisjointPairInitialRadius(initial_radius_, pair_disjoint_fraction_,
                                    plus_center, minus_center)
      : initial_radius_;
  for (const auto &entry : centers) {
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
    const auto sample = [this](const M0Locations &points) { return SampleAdm(points); };
    if (!candidate.converged && iterations_ > 1 && ntheta_ >= lmax_ + 1) {
      // A small sphere can lie inside the basin of the inner root. A larger
      // origin trial also covers common surfaces enclosing both axial centers.
      // Keep mirror seeds disjoint. All trials use the same acceptance checks.
      const std::vector<std::array<Real, 2>> seed_axes = entry.first == "origin"
          ? std::vector<std::array<Real, 2>>{{4.0, 4.0}, {1.0, 2.0}, {1.0, 1.0}}
          : std::vector<std::array<Real, 2>>{{1.0, 1.0}};
      for (const auto &axes : seed_axes) {
        const Real polar = axes[0] * fresh_initial_radius;
        const Real equatorial = axes[1] * fresh_initial_radius;
        std::vector<Real> seed(lmax_ + 1, 0.0);
        if (polar == equatorial) {
          seed[0] = std::sqrt(4*kPi) * polar;
        } else {
          // The outer near-critical root can lie outside the basin reached
          // by either sphere. Project a regular oblate trial onto the same basis.
          for (int n = 0; n < ntheta_; ++n) {
            const Real radius = 1.0 / std::hypot(std::cos(theta_[n])/polar,
                                                std::sin(theta_[n])/equatorial);
            for (int l = 0; l <= lmax_; ++l)
              seed[l] += 2*kPi * weights_[n] * radius * y0_[n*(lmax_ + 1) + l];
          }
        }
        auto trial = RefineM0Surface(seed, entry.second, ntheta_, std::min(iterations_, 20),
                                     0.0, direct_tolerance_, hrms_tolerance_, mass_tolerance_,
                                     sample, equatorial_ && entry.first == "origin");
        trial.branch = entry.first;
        trial.fresh_initial_radius = polar;
        if (trial.converged || (trial.area > 0 &&
            (candidate.area <= 0 || trial.direct_residual < candidate.direct_residual))) {
          candidate = std::move(trial);
        }
        if (candidate.converged) break;
      }
    }
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
      auto moved = SearchCandidate(entry.first, candidate.center_z + shift,
                                  fresh_initial_radius,
                                  recentered);
      if (!moved.converged) break;
      candidate = std::move(moved);
      candidate.axis_extremum_z = axis_extremum_z;
      candidate.center_lapse = center_lapse;
    }
    const Real inner_target = pin_->GetReal("fastflow", "cartoon_inner_target_0");
    if (inner_target < 0 && candidate.area > 0 && ntheta_ >= lmax_ + 1) {
      const Real tol = std::min(direct_tolerance_, -0.25*inner_target);
      auto inner = RefineM0Surface(candidate.coefficients, candidate.center_z, ntheta_,
                                   std::min(iterations_, 20), inner_target, tol,
                                   hrms_tolerance_, mass_tolerance_, sample,
                                   equatorial_ && entry.first == "origin");
      // Do not promote a strictly trapped surface to a MOTS or feed it to the
      // tracker/restart. Save its shape and mass for independent verification.
      if (global_variable::my_rank == 0) {
        const auto path = pin_->GetString("job", "basename") + ".cartoon_m0_inner_0.txt";
        FILE *fp = std::fopen(path.c_str(), "a");
        if (!fp) throw std::runtime_error("cannot open Cartoon inner-surface history");
        std::fseek(fp, 0, SEEK_END);
        if (std::ftell(fp) == 0)
          std::fprintf(fp, "# cycle time branch target_rThetaPlus trapped center_z area "
                           "irreducible_mass direct_residual status a_l0...\n");
        std::fprintf(fp, "%d %.17e %s %.17e %d %.17e %.17e %.17e %.17e %s", cycle,
                     time, entry.first.c_str(), inner_target, inner.converged, inner.center_z,
                     inner.area, inner.irreducible_mass, inner.direct_residual, inner.failure.c_str());
        for (const Real c : inner.coefficients) std::fprintf(fp, " %.17e", c);
        std::fputc('\n', fp);
        std::fclose(fp);
      }
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
  if (!pin_->GetBoolean("fastflow", "cartoon_search_offcenter_0") &&
      state.converged && state.selected_branch != "origin") {
    pack_->z4c_restart_state.fastflow = {};
    return;
  }
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
