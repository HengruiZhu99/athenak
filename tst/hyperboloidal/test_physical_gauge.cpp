// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
// Extract lower-order residues independently of the principal-symbol tests.
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "z4c/hyperboloidal/cartesian_patch.hpp"

namespace hyp = z4c::hyperboloidal;
using Jet = hyp::Z4cJet<double>;

void Close(double x, double y, double tolerance, const char* label) {
  if (!std::isfinite(x) || !std::isfinite(y) || std::abs(x - y) > tolerance) {
    throw std::runtime_error(label);
  }
}

hyp::LayerGaugeParameters Gauge(double xi) {
  hyp::LayerGaugeParameters gauge;
  gauge.physical_trace_lapse = true;
  gauge.preferred_source = false;
  gauge.scri_lapse_damping = xi;
  gauge.Validate(1);
  return gauge;
}

// Hypothetical instantaneous preferred-source projection, deliberately excluded
// from production: it restores a positive continuum pole. Audit the candidate
// rather than enabling it. The live source must include F0's singular part.
hyp::GaugeRHSParts<double> HypotheticalProjection(const hyp::LayerPoint<double>& p,
                                                  const Jet& u, double xi) {
  const auto g = Gauge(xi);
  auto out = hyp::InteriorLayerGauge(p, u, g);
  const auto geo = hyp::Geometry(u.metric);
  const double alpha = u.alpha.value, chi = u.chi.value;
  const auto coefficients = hyp::LayerCoefficients(p.radius, alpha, g);
  if (coefficients.weight != 1)
    throw std::runtime_error("projection audit requires harmonic collar");
  double ref_adv_alpha = 0, bomega = 0, norm = 0, fbase[3]{}, hessian4 = 0,
         contraction = 0;
  for (int i = 0; i < 3; ++i) {
    ref_adv_alpha += p.beta[i] * p.dalpha[i];
    bomega += u.beta.value[i] * p.domega[i];
    norm += p.domega[i] * p.domega[i];
  }
  const double wn = -bomega / alpha;
  const double f0reg =
      (ref_adv_alpha + alpha * coefficients.nu * std::log(alpha / p.alpha)) /
      (alpha * alpha * alpha);
  const double f0pole =
      -out.pole.alpha / (alpha * alpha * alpha) - (u.trace.value - 3 * wn) / alpha;
  for (int i = 0; i < 3; ++i) {
    double refadv = 0;
    for (int j = 0; j < 3; ++j) refadv += p.beta[j] * p.state.beta.d[j][i];
    fbase[i] =
        chi * p.state.lambda.value[i] +
        (refadv + coefficients.eta * (u.beta.value[i] - p.beta[i])) / (alpha * alpha) -
        u.beta.value[i] * f0reg;
    for (int j = 0; j < 3; ++j) {
      fbase[i] += chi * geo.inverse[i][j] *
                  (p.state.chi.d[j] / (2 * p.state.chi.value) - p.dalpha[j] / p.alpha);
      hessian4 += (chi * geo.inverse[i][j] -
                   u.beta.value[i] * u.beta.value[j] / (alpha * alpha)) *
                  p.omega_hessian[i][j];
    }
    contraction += p.domega[i] * fbase[i];
  }
  const double delta_reg = hessian4 - p.omega * p.w_omega - contraction;
  const double delta_pole = bomega * f0pole;
  for (int i = 0; i < 3; ++i) {
    out.regular.beta[i] -= alpha * alpha * p.domega[i] * delta_reg / norm;
    out.pole.beta[i] -= alpha * alpha * p.domega[i] * delta_pole / norm;
  }
  return out;
}

// Independent 4D Christoffel contraction using metric derivatives in all four
// directions. Geometry time derivatives come from the actual tensor RHS.
void IndependentFourGamma(const Jet& u, const hyp::Z4cRHS<double>& geom,
                          const hyp::GaugeRHS<double>& gauge, double gamma[4]) {
  const auto b = hyp::PenroseMetric(u.metric, u.chi);
  const auto gb = hyp::Geometry(b);
  const double alpha = u.alpha.value;
  double g[4][4]{}, inv[4][4]{}, dg[4][4][4]{};
  double bd[4][3][3]{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      g[i + 1][j + 1] = b.g[i][j];
      inv[i + 1][j + 1] =
          gb.inverse[i][j] - u.beta.value[i] * u.beta.value[j] / (alpha * alpha);
      bd[0][i][j] = geom.metric[i][j] / u.chi.value -
                    u.metric.g[i][j] * geom.chi / (u.chi.value * u.chi.value);
      for (int d = 0; d < 3; ++d) bd[d + 1][i][j] = b.dg[d][i][j];
    }
  g[0][0] = -alpha * alpha;
  inv[0][0] = -1 / (alpha * alpha);
  for (int i = 0; i < 3; ++i) {
    inv[0][i + 1] = inv[i + 1][0] = u.beta.value[i] / (alpha * alpha);
    for (int j = 0; j < 3; ++j) {
      g[0][i + 1] += b.g[i][j] * u.beta.value[j];
      g[0][0] += b.g[i][j] * u.beta.value[i] * u.beta.value[j];
    }
    g[i + 1][0] = g[0][i + 1];
  }
  for (int d = 0; d < 4; ++d) {
    const double da = d == 0 ? gauge.alpha : u.alpha.d[d - 1];
    double db[3];
    for (int i = 0; i < 3; ++i) db[i] = d == 0 ? gauge.beta[i] : u.beta.d[d - 1][i];
    dg[d][0][0] = -2 * alpha * da;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        dg[d][i + 1][j + 1] = bd[d][i][j];
        dg[d][0][i + 1] += bd[d][i][j] * u.beta.value[j] + b.g[i][j] * db[j];
        dg[d][0][0] += bd[d][i][j] * u.beta.value[i] * u.beta.value[j] +
                       2 * b.g[i][j] * u.beta.value[i] * db[j];
      }
    for (int i = 0; i < 3; ++i) dg[d][i + 1][0] = dg[d][0][i + 1];
  }
  for (int a = 0; a < 4; ++a)
    for (int b = 0; b < 4; ++b)
      for (int c = 0; c < 4; ++c)
        for (int l = 0; l < 4; ++l)
          gamma[a] +=
              .5 * inv[b][c] * inv[a][l] * (dg[b][l][c] + dg[c][l][b] - dg[l][b][c]);
}

void PreferredProjectionIdentityAudit() {
  const hyp::LayerReference<double> ref(1., 1., {true, .35, .75});
  double error = 0, boxerror = 0;
  for (double radius : {.86, .93, .99, .999}) {
    const auto p = ref.At(.36 * radius, -.48 * radius, .8 * radius);
    auto u = p.state;
    u.alpha.value *= 1.1;
    u.chi.value *= .94;
    u.beta.value[0] += .03;
    u.beta.value[1] -= .02;
    u.beta.value[2] += .015;
    u.metric.g[0][0] = 1.1;
    u.metric.g[1][1] = 1 / 1.1;
    u.a.k[0][1] = u.a.k[1][0] = .002;
    u.trace.value += .04;
    u.theta.value = .006;
    u.lambda.value[0] += .002;
    u.lambda.value[1] -= .005;
    u.alpha.d[0] += .017;
    u.chi.d[1] -= .013;
    const auto o = hyp::CartesianOmega(u, p);
    hyp::Z4cRHS<double> geom{};
    hyp::GaugeRHS<double> gauge{};
    if (!hyp::AssembleInterior(hyp::ConformalRHS(u, o, 5 / u.alpha.value, 0.), p.omega,
                               geom) ||
        !hyp::AssembleGaugeInterior(HypotheticalProjection(p, u, 1.5), p.omega, gauge))
      throw std::runtime_error("invalid identity inputs");
    const auto projected_parts = HypotheticalProjection(p, u, 1.5);
    // Existing production gauges have no shift pole; their common assembler
    // handles the lapse pole only. Assemble this hypothetical shift pole here.
    for (int i = 0; i < 3; ++i) gauge.beta[i] += projected_parts.pole.beta[i] / p.omega;
    double gamma[4]{};
    IndependentFourGamma(u, geom, gauge, gamma);
    const auto gt = hyp::Geometry(u.metric);
    double dtalpha = gauge.alpha, bomega = 0, hessian4 = 0, projected = 0,
           actualbox = 0;
    for (int i = 0; i < 3; ++i) {
      dtalpha -= u.beta.value[i] * u.alpha.d[i];
      bomega += u.beta.value[i] * p.domega[i];
    }
    const double f0 = -dtalpha / std::pow(u.alpha.value, 3) -
                      (u.trace.value - 3 * o.normal) / (u.alpha.value * p.omega);
    error = std::max(
        error, std::abs(gamma[0] + 2 * u.theta.value / (u.alpha.value * p.omega) - f0));
    for (int i = 0; i < 3; ++i) {
      double dtbeta = gauge.beta[i];
      for (int j = 0; j < 3; ++j) dtbeta -= u.beta.value[j] * u.beta.d[j][i];
      double source = u.chi.value * u.lambda.value[i] -
                      dtbeta / (u.alpha.value * u.alpha.value) - u.beta.value[i] * f0;
      for (int j = 0; j < 3; ++j) {
        source += u.chi.value * gt.inverse[i][j] *
                  (u.chi.d[j] / (2 * u.chi.value) - u.alpha.d[j] / u.alpha.value);
        hessian4 +=
            (u.chi.value * gt.inverse[i][j] -
             u.beta.value[i] * u.beta.value[j] / (u.alpha.value * u.alpha.value)) *
            p.omega_hessian[i][j];
      }
      const double z4i = .5 * u.chi.value * (u.lambda.value[i] - gt.contracted[i]) -
                         u.beta.value[i] * u.theta.value / (u.alpha.value * p.omega);
      error = std::max(error, std::abs(gamma[i + 1] + 2 * z4i - source));
      projected += p.domega[i] * source;
      actualbox -= p.domega[i] * gamma[i + 1];
    }
    actualbox += hessian4;
    double target = p.omega * p.w_omega + 2 * o.normal * u.theta.value / p.omega;
    for (int i = 0; i < 3; ++i)
      target += u.chi.value * (u.lambda.value[i] - gt.contracted[i]) * p.domega[i];
    boxerror = std::max(boxerror, std::abs(actualbox - target));
    Close(projected, hessian4 - p.omega * p.w_omega, 2e-12,
          "preferred source contraction");
  }
  Close(error, 0, 2e-12, "independent 4D gamma/source identity");
  Close(boxerror, 0, 2e-12, "off-constraint BoxOmega identity");
  std::cout << "PASS hypothetical projection 4D source identity error=" << error
            << " off-constraint BoxOmega error=" << boxerror << '\n';
}

double ReferenceDerivative(const Jet& u, int field, int direction) {
  if (field == z4c::Z4c::I_Z4C_CHI) return u.chi.d[direction];
  if (field == z4c::Z4c::I_Z4C_KHAT) return u.trace.d[direction];
  if (field == z4c::Z4c::I_Z4C_ALPHA) return u.alpha.d[direction];
  if (field >= z4c::Z4c::I_Z4C_BETAX && field <= z4c::Z4c::I_Z4C_BETAZ) {
    return u.beta.d[direction][field - z4c::Z4c::I_Z4C_BETAX];
  }
  if (field >= z4c::Z4c::I_Z4C_GAMX && field <= z4c::Z4c::I_Z4C_GAMZ) {
    return u.lambda.d[direction][field - z4c::Z4c::I_Z4C_GAMX];
  }
  const int ti[6] = {0, 0, 0, 1, 1, 2}, tj[6] = {0, 1, 2, 1, 2, 2};
  if (field >= z4c::Z4c::I_Z4C_GXX && field <= z4c::Z4c::I_Z4C_GZZ) {
    const int index = field - z4c::Z4c::I_Z4C_GXX;
    return u.metric.dg[direction][ti[index]][tj[index]];
  }
  if (field >= z4c::Z4c::I_Z4C_AXX && field <= z4c::Z4c::I_Z4C_AZZ) {
    const int index = field - z4c::Z4c::I_Z4C_AXX;
    return u.a.dk[direction][ti[index]][tj[index]];
  }
  return 0;
}

double ReferenceDerivativeAudit(const hyp::LayerReference<double>& ref,
                                const hyp::LayerPoint<double>& p,
                                const std::array<double, 3>& position) {
  constexpr double step = 1e-4;
  double maximum = 0;
  for (int direction = 0; direction < 3; ++direction) {
    hyp::LayerPoint<double> values[4];
    const int offsets[4] = {-2, -1, 1, 2};
    for (int i = 0; i < 4; ++i) {
      auto shifted = position;
      shifted[direction] += offsets[i] * step;
      values[i] = ref.At(shifted[0], shifted[1], shifted[2]);
    }
    for (int field = 0; field < z4c::Z4c::nz4c; ++field) {
      const double exact = ReferenceDerivative(p.state, field, direction);
      const double fd = (hyp::ReferenceComponent(field, values[0]) -
                         8 * hyp::ReferenceComponent(field, values[1]) +
                         8 * hyp::ReferenceComponent(field, values[2]) -
                         hyp::ReferenceComponent(field, values[3])) /
                        (12 * step);
      maximum = std::max(maximum, std::abs(fd - exact) / (1 + std::abs(exact)));
    }
  }
  Close(maximum, 0, 2e-7, "reference finite-difference derivative");
  return maximum;
}

double GeometryAudit(const hyp::LayerPoint<double>& p) {
  const auto& u = p.state;
  hyp::Z4cRHS<double> rhs{};
  if (!hyp::AssembleInterior(
          hyp::ConformalRHS(u, hyp::CartesianOmega(u, p), 5 / u.alpha.value, 0.),
          p.omega, rhs)) {
    throw std::runtime_error("invalid reference geometry RHS");
  }
  double maximum =
      std::max({std::abs(rhs.chi), std::abs(rhs.trace), std::abs(rhs.theta)});
  for (int i = 0; i < 3; ++i) {
    maximum = std::max(maximum, std::abs(rhs.lambda[i]));
    for (int j = 0; j < 3; ++j) {
      maximum = std::max({maximum, std::abs(rhs.metric[i][j]), std::abs(rhs.a[i][j])});
    }
  }
  Close(maximum, 0, 2e-7, "reference geometric fixed point");
  return maximum;
}

void ReferenceAudit() {
  double maximum = 0;
  for (double a : {0.5, 0.75, 1., 2.}) {
    double fd_maximum = 0, geometry_maximum = 0, gauge_maximum = 0;
    for (bool layer : {false, true}) {
      const hyp::LayerReference<double> ref(1, a, {layer, 0.35, 0.75});
      ref.Validate();
      for (double xi : {0., 0.5, 1.5, 3.}) {
        const auto gauge = Gauge(xi);
        for (double r :
             {0., 0.1, 0.35, 0.4, 0.45, 0.5, 0.6, 0.75, 0.85, 0.99, 0.9999}) {
          for (const auto& direction : {std::array<double, 3>{1, 0, 0},
                                        std::array<double, 3>{0.36, -0.48, 0.8}}) {
            const std::array<double, 3> position{r * direction[0], r * direction[1],
                                                 r * direction[2]};
            const auto p = ref.At(position[0], position[1], position[2]);
            const auto parts = hyp::InteriorLayerGauge(p, p.state, gauge);
            hyp::GaugeRHS<double> rhs{};
            if (!hyp::AssembleGaugeInterior(parts, p.omega, rhs)) {
              throw std::runtime_error("invalid physical reference gauge");
            }
            gauge_maximum = std::max(gauge_maximum, std::abs(rhs.alpha));
            for (double value : rhs.beta) {
              gauge_maximum = std::max(gauge_maximum, std::abs(value));
            }
            Close(parts.pole.alpha, 0, 0, "reference lapse pole");
            if (xi == 1.5) {
              fd_maximum =
                  std::max(fd_maximum, ReferenceDerivativeAudit(ref, p, position));
              geometry_maximum = std::max(geometry_maximum, GeometryAudit(p));
            }
          }
        }
        const auto p = ref.At(1., 0., 0.);
        for (double factor : {0.5, 0.9, 1.1, 1.5, 2.}) {
          auto u = p.state;
          u.alpha.value *= factor;
          const auto parts = hyp::InteriorLayerGauge(p, u, gauge);
          if (!parts.valid || parts.pole.alpha * (factor - 1) >= 0) {
            throw std::runtime_error("physical lapse pole must restore lapse");
          }
        }
        hyp::GaugeRHS<double> rhs{};
        if (hyp::AssembleGaugeInterior(hyp::InteriorLayerGauge(p, p.state, gauge), 0.,
                                       rhs)) {
          throw std::runtime_error("physical gauge must reject exact scri assembly");
        }
      }
    }
    maximum = std::max(maximum, gauge_maximum);
    std::cout << "curvature_radius=" << a << " gauge_fixed_point=" << gauge_maximum
              << " geometry_fixed_point=" << geometry_maximum
              << " reference_derivative_scaled_error=" << fd_maximum << '\n';
  }
  Close(maximum, 0, 2e-10, "physical reference fixed point");
  std::cout << "PASS physical lapse fixed points and finite-amplitude restoring signs: "
            << maximum << '\n';
}

void SmallPositiveLapseAudit() {
  const hyp::LayerReference<double> ref(1, 1, {true, .35, .75});
  const auto p = ref.At(.1, 0., 0.);
  for (bool physical : {false, true}) {
    for (double nu : {0., 1.5}) {
      auto gauge = Gauge(1.5);
      gauge.physical_trace_lapse = physical;
      gauge.lapse_inner = nu;
      for (double alpha : {1e-12, 1e-18, 1e-100, 1e-300}) {
        auto u = p.state;
        u.alpha.value = alpha;
        u.trace.value = .3;
        hyp::GaugeRHS<double> rhs{};
        if (!hyp::AssembleGaugeInterior(hyp::InteriorLayerGauge(p, u, gauge),
                                       p.omega, rhs)) {
          throw std::runtime_error("positive collapsed lapse rejected");
        }
        const double scaled = -nu * std::log(alpha) - (alpha + 2) * .3;
        Close(rhs.alpha / alpha, scaled, 1e-10, "collapsed lapse scaled RHS");
      }
    }
  }
  std::cout << "PASS positive collapsed lapse down to 1e-300 without floors\n";
}

// Five symmetric tensor directions tangent to det(g)=1 and tr(A)=0 at g=I.
void TensorDirection(int column, double delta, double tensor[3][3]) {
  if (column == 0) {
    tensor[0][0] += delta;
    tensor[1][1] -= delta / 2;
    tensor[2][2] -= delta / 2;
  }
  if (column == 1) tensor[0][1] = tensor[1][0] += delta;
  if (column == 2) tensor[0][2] = tensor[2][0] += delta;
  if (column == 3) {
    tensor[1][1] += delta;
    tensor[2][2] -= delta;
  }
  if (column == 4) tensor[1][2] = tensor[2][1] += delta;
}

void Perturb(int column, double delta, Jet& u) {
  if (column == 0) u.alpha.value += delta;
  if (column == 1) u.chi.value += delta;
  if (column == 2) u.trace.value += delta;
  if (column == 3) u.theta.value += delta;
  if (column >= 4 && column < 7) u.beta.value[column - 4] += delta;
  if (column >= 7 && column < 12) TensorDirection(column - 7, delta, u.metric.g);
  if (column >= 12 && column < 17) TensorDirection(column - 12, delta, u.a.k);
  if (column >= 17) u.lambda.value[column - 17] += delta;
}

void Pole(const hyp::LayerPoint<double>& p, const Jet& u, double xi, double kappa,
          bool physical, double output[20]) {
  auto gauge = Gauge(xi);
  gauge.physical_trace_lapse = physical;
  const auto parts =
      hyp::ConformalRHS(u, hyp::CartesianOmega(u, p), kappa / u.alpha.value, 0.);
  const auto g = hyp::InteriorLayerGauge(p, u, gauge);
  if (!parts.valid || !g.valid) throw std::runtime_error("invalid pole extraction");
  output[0] = g.pole.alpha;
  output[1] = parts.pole.chi;
  output[2] = parts.pole.trace;
  output[3] = parts.pole.theta;
  for (int i = 0; i < 3; ++i) output[4 + i] = g.pole.beta[i];
  const auto& h = parts.pole.metric;
  const auto& a = parts.pole.a;
  output[7] = h[0][0];
  output[8] = h[0][1];
  output[9] = h[0][2];
  output[10] = (h[1][1] - h[2][2]) / 2;
  output[11] = h[1][2];
  output[12] = a[0][0];
  output[13] = a[0][1];
  output[14] = a[0][2];
  output[15] = (a[1][1] - a[2][2]) / 2;
  output[16] = a[1][2];
  for (int i = 0; i < 3; ++i) output[17 + i] = parts.pole.lambda[i];
}

void Extract(double xi, double kappa, bool physical, double a, double matrix[20][20],
             bool projection = false) {
  const hyp::LayerReference<double> ref(1, a, {true, 0.35, 0.75});
  const auto p = ref.At(1., 0., 0.);
  constexpr double epsilon = 1e-6;
  for (int column = 0; column < 20; ++column) {
    double output[2][20];
    for (int sign = 0; sign < 2; ++sign) {
      auto u = p.state;
      Perturb(column, sign ? epsilon : -epsilon, u);
      Pole(p, u, xi, kappa, physical, output[sign]);
      if (projection) {
        const auto projected = HypotheticalProjection(p, u, xi);
        for (int i = 0; i < 3; ++i) output[sign][4 + i] = projected.pole.beta[i];
      }
    }
    for (int row = 0; row < 20; ++row) {
      matrix[row][column] = (output[1][row] - output[0][row]) / (2 * epsilon);
    }
  }
}

void PrintPole(double xi, double kappa, bool physical, double a,
               bool projection = false) {
  double matrix[20][20]{};
  Extract(xi, kappa, physical, a, matrix, projection);
  std::cout << "{\"xi\":" << xi << ",\"kappa\":" << kappa << ",\"a\":" << a
            << ",\"physical\":" << (physical ? "true" : "false") << ",\"M\":[";
  for (int row = 0; row < 20; ++row) {
    if (row) std::cout << ',';
    std::cout << '[';
    for (int column = 0; column < 20; ++column) {
      if (column) std::cout << ',';
      std::cout << matrix[row][column];
    }
    std::cout << ']';
  }
  std::cout << "]}";
}

void PrintPoles() {
  std::cout << std::setprecision(17) << '[';
  bool first = true;
  for (double xi : {0., 0.5, 1., 1.5, 3.}) {
    for (double kappa : {0., 0.5, 1., 1.5, 2., 3., 5., 10.}) {
      for (bool physical : {false, true}) {
        if (!first) std::cout << ',';
        first = false;
        PrintPole(xi, kappa, physical, 1);
      }
    }
  }
  for (double a : {0.5, 0.75, 2.}) {
    std::cout << ',';
    PrintPole(1.5, 5, true, a);
  }
  std::cout << "]\n";
}

int main(int argc, char* argv[]) {
  if (argc == 2 && std::string(argv[1]) == "--pole") {
    PrintPoles();
  } else if (argc == 2 && std::string(argv[1]) == "--projection-pole") {
    std::cout << std::setprecision(17) << '[';
    bool first = true;
    for (double xi : {0., 0.5, 1.5, 3.}) {
      for (double kappa : {0., 1., 1.5, 5., 10.}) {
        if (!first) std::cout << ',';
        first = false;
        PrintPole(xi, kappa, true, 1., true);
      }
    }
    std::cout << "]\n";
  } else {
    ReferenceAudit();
    SmallPositiveLapseAudit();
    PreferredProjectionIdentityAudit();
  }
}
