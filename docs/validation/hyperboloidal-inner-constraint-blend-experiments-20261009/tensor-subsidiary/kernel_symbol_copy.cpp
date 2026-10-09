// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
// Extract the constrained 20-field symbol from the actual tensor and gauge RHS.
#include <cmath>
#include <iomanip>
#include <iostream>
#include "z4c/hyperboloidal/layer_gauge.hpp"

namespace hyp = z4c::hyperboloidal;
using Jet = hyp::Z4cJet<double>;
using RHS = hyp::Z4cRHS<double>;

void Invert(const double e[3][3], double inv[3][3]) {
  const double det = e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1]) -
                     e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0]) +
                     e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0]);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      inv[i][j] = (e[(j + 1) % 3][(i + 1) % 3] * e[(j + 2) % 3][(i + 2) % 3] -
                   e[(j + 1) % 3][(i + 2) % 3] * e[(j + 2) % 3][(i + 1) % 3]) /
                  det;
    }
}

double TensorFrame(const double v[3][3], const double inv[3][3], int i, int j) {
  double out = 0;
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
      out += inv[a][i] * inv[b][j] * v[a][b];
  return out;
}

void TensorInput(int column, double h[3][3], double a[3][3]) {
  if (column == 2) {
    h[0][0] = 1;
    h[1][1] = h[2][2] = -0.5;
  }
  if (column == 5) {
    a[0][0] = 1;
    a[1][1] = a[2][2] = -0.5;
  }
  for (int v = 0; v < 2; ++v) {
    if (column == 8 + 4 * v)
      h[0][1 + v] = h[1 + v][0] = 1;
    if (column == 9 + 4 * v)
      a[0][1 + v] = a[1 + v][0] = 1;
  }
  if (column == 16) {
    h[1][1] = 1;
    h[2][2] = -1;
  }
  if (column == 17) {
    a[1][1] = 1;
    a[2][2] = -1;
  }
  if (column == 18)
    h[1][2] = h[2][1] = 1;
  if (column == 19)
    a[1][2] = a[2][1] = 1;
}

void Extract(double alpha, double chi, double radius, bool oblique,
             bool physical_lapse, double matrix[20][20]) {
  bulk_radius=radius; // prescribed scratch C(r) only
  const double b[3][3] = {{1.3, 0.2, -0.1}, {0, 0.8, 0.12}, {0, 0, 1 / 1.04}};
  const double q[3][3] = {
      {0.36, -0.48, 0.8}, {0.8, 0.6, 0}, {-0.48, 0.64, 0.6}};
  double e[3][3]{}, inv[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k)
        e[i][j] += (oblique ? q[i][k] : (i == k ? 1 : 0)) *
                   (oblique ? b[k][j] : (k == j ? 1 : 0)) / std::sqrt(chi);
    }
  Invert(e, inv);
  Jet base{};
  base.alpha.value = alpha;
  base.chi.value = chi;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        base.metric.g[i][j] += chi * e[k][i] * e[k][j];
  hyp::LayerPoint<double> p{};
  p.state = base;
  p.alpha = alpha;
  p.radius = radius;
  p.omega = 0.37;
  hyp::LayerGaugeParameters gauge;
  gauge.preferred_source = false; // algebraic projection has no principal part
  gauge.physical_trace_lapse = physical_lapse;
  hyp::OmegaJet<double> omega{};
  omega.omega = p.omega;
  // Separate derivative orders to exclude lower-order physical-trace/Z4 terms.
  // The reduction enforces det(gtilde)=1 and trace(A)=0 before extraction.
  for (int column = 0; column < 20; ++column) {
    double h[3][3]{}, a[3][3]{}, lambda[3]{}, beta[3]{};
    TensorInput(column, h, a);
    if (column == 6)
      lambda[0] = 1;
    if (column == 7)
      beta[0] = 1;
    for (int v = 0; v < 2; ++v) {
      if (column == 10 + 4 * v)
        lambda[1 + v] = 1;
      if (column == 11 + 4 * v)
        beta[1 + v] = 1;
    }
    double hcoord[3][3]{}, acoord[3][3]{}, lcoord[3]{}, bcoord[3]{};
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        lcoord[i] += inv[i][j] * lambda[j] / chi;
        bcoord[i] += alpha * inv[i][j] * beta[j];
        for (int k = 0; k < 3; ++k)
          for (int l = 0; l < 3; ++l) {
            hcoord[i][j] += chi * e[k][i] * e[l][j] * h[k][l];
            acoord[i][j] += chi * e[k][i] * e[l][j] * a[k][l];
          }
      }
    }
    Jet first = base, second = base, connection = base, slicing = base,
        shift = base;
    first.trace.value = column == 3 ? omega.omega : 0;
    first.theta.value = column == 4 ? omega.omega : 0;
    slicing.trace.value = first.trace.value;
    for (int i = 0; i < 3; ++i) {
      connection.trace.d[i] = first.trace.value * e[0][i];
      connection.theta.d[i] = first.theta.value * e[0][i];
      shift.lambda.value[i] = lcoord[i];
      shift.alpha.d[i] = column == 0 ? alpha * e[0][i] : 0;
      shift.chi.d[i] = column == 1 ? chi * e[0][i] : 0;
      for (int j = 0; j < 3; ++j) {
        first.a.k[i][j] = acoord[i][j];
        first.beta.d[j][i] = bcoord[i] * e[0][j];
        second.lambda.d[j][i] = lcoord[i] * e[0][j];
        second.alpha.dd[i][j] = column == 0 ? alpha * e[0][i] * e[0][j] : 0;
        second.chi.dd[i][j] = column == 1 ? chi * e[0][i] * e[0][j] : 0;
        for (int k = 0; k < 3; ++k) {
          connection.beta.dd[j][k][i] = bcoord[i] * e[0][j] * e[0][k];
          for (int l = 0; l < 3; ++l) {
            second.metric.ddg[k][l][i][j] = hcoord[i][j] * e[0][k] * e[0][l];
          }
        }
      }
    }
    RHS r1{}, r2{}, r3{};
    if (!hyp::AssembleInterior(hyp::ConformalRHS(first, omega, 0., 0.),
                               omega.omega, r1) ||
        !hyp::AssembleInterior(hyp::ConformalRHS(second, omega, 0., 0.),
                               omega.omega, r2) ||
        !hyp::AssembleInterior(hyp::ConformalRHS(connection, omega, 0., 0.),
                               omega.omega, r3)) {
      throw std::runtime_error("kernel symbol extraction failed");
    }
    hyp::GaugeRHS<double> gl{}, gs{}, gbase{};
    if (!hyp::AssembleGaugeInterior(hyp::InteriorLayerGauge(p, slicing, gauge),
                                    omega.omega, gl) ||
        !hyp::AssembleGaugeInterior(hyp::InteriorLayerGauge(p, shift, gauge),
                                    omega.omega, gs) ||
        !hyp::AssembleGaugeInterior(hyp::InteriorLayerGauge(p, base, gauge),
                                    omega.omega, gbase)) {
      throw std::runtime_error("gauge symbol extraction failed");
    }
    matrix[0][column] = (gl.alpha - gbase.alpha) / (alpha * alpha);
    matrix[1][column] = r1.chi / (alpha * chi);
    matrix[2][column] = TensorFrame(r1.metric, inv, 0, 0) / (alpha * chi);
    matrix[3][column] = r2.trace / (omega.omega * alpha);
    matrix[4][column] = r2.theta / (omega.omega * alpha);
    matrix[5][column] = TensorFrame(r2.a, inv, 0, 0) / (alpha * chi);
    double lframe[3]{}, bframe[3]{};
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        lframe[i] += chi * e[i][j] * r3.lambda[j] / alpha;
        bframe[i] += e[i][j] * (gs.beta[j] - gbase.beta[j]) / (alpha * alpha);
      }
    matrix[6][column] = lframe[0];
    matrix[7][column] = bframe[0];
    for (int v = 0; v < 2; ++v) {
      matrix[8 + 4 * v][column] =
          TensorFrame(r1.metric, inv, 0, 1 + v) / (alpha * chi);
      matrix[9 + 4 * v][column] =
          TensorFrame(r2.a, inv, 0, 1 + v) / (alpha * chi);
      matrix[10 + 4 * v][column] = lframe[1 + v];
      matrix[11 + 4 * v][column] = bframe[1 + v];
    }
    matrix[16][column] = TensorFrame(r1.metric, inv, 1, 1) / (alpha * chi);
    matrix[17][column] = TensorFrame(r2.a, inv, 1, 1) / (alpha * chi);
    matrix[18][column] = TensorFrame(r1.metric, inv, 1, 2) / (alpha * chi);
    matrix[19][column] = TensorFrame(r2.a, inv, 1, 2) / (alpha * chi);
    // Plus tensor polarization is the transverse difference, removing scalar.
    matrix[16][column] -= TensorFrame(r1.metric, inv, 2, 2) / (alpha * chi);
    matrix[17][column] -= TensorFrame(r2.a, inv, 2, 2) / (alpha * chi);
    matrix[16][column] /= 2;
    matrix[17][column] /= 2;
  }
}

int main() {
  std::cout << std::setprecision(17) << '[';
  bool first = true;
  for (double alpha : {0.2, 1., 3.})
    for (double chi : {0.4, 1., 2.})
      for (double r : {0., 0.45, 0.5, 0.65, 0.8, 0.83, 0.84, 0.849, 0.85, 1.})
        for (bool oblique : {false, true})
          for (bool physical_lapse : {false, true}) {
            if (!first)
              std::cout << ',';
            first = false;
            double matrix[20][20]{};
            Extract(alpha, chi, r, oblique, physical_lapse, matrix);
            const auto c =
                hyp::LayerCoefficients(r, alpha, hyp::LayerGaugeParameters{});
            std::cout << "{\"alpha\":" << alpha << ",\"chi\":" << chi
                      << ",\"r\":" << r << ",\"oblique\":" << oblique
                      << ",\"W\":" << c.weight
                      << ",\"physical_trace_lapse\":" << physical_lapse
                      << ",\"f\":" << c.f << ",\"mu\":" << c.mu
                      << ",\"q\":" << c.q << ",\"M\":[";
            for (int i = 0; i < 20; ++i) {
              if (i)
                std::cout << ',';
              std::cout << '[';
              for (int j = 0; j < 20; ++j) {
                if (j)
                  std::cout << ',';
                std::cout << matrix[i][j];
              }
              std::cout << ']';
            }
            std::cout << "]}";
          }
  std::cout << "]\n";
}
