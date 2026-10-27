// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include "z4c/hyperboloidal/elliptic_shift.hpp"
#include "z4c/hyperboloidal/spherical_tensor.hpp"

namespace h = z4c::hyperboloidal;
void Check(bool okay, const char *message) {
  if (!okay) throw std::runtime_error(message);
}
int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  try {
    std::mt19937 random(7419);
    std::uniform_real_distribution<double> sample(-0.1, 0.1);
    double variation_error = 0, flat_error = 0, rhs_error = 0, radial_error = 0;
    for (int trial = 0; trial < 32; ++trial) {
      h::Z4cJet<double> u{};
      u.alpha.value = 0.8;
      u.chi.value = 1.1;
      for (int i = 0; i < 3; ++i) {
        u.metric.g[i][i] = 1;
        u.alpha.d[i] = sample(random);
        for (int j = i; j < 3; ++j) {
          u.metric.g[i][j] += sample(random);
          u.metric.g[j][i] = u.metric.g[i][j];
          u.a.k[i][j] = u.a.k[j][i] = sample(random);
          for (int d = 0; d < 3; ++d) {
            u.metric.dg[d][i][j] = u.metric.dg[d][j][i] = sample(random);
            u.a.dk[d][i][j] = u.a.dk[d][j][i] = sample(random);
            for (int e = d; e < 3; ++e) {
              const double value = sample(random);
              u.metric.ddg[d][e][i][j] = u.metric.ddg[d][e][j][i] = value;
              u.metric.ddg[e][d][i][j] = u.metric.ddg[e][d][j][i] = value;
            }
          }
        }
        u.beta.value[i] = sample(random);
        for (int d = 0; d < 3; ++d) {
          u.beta.d[d][i] = sample(random);
          for (int e = d; e < 3; ++e)
            u.beta.dd[d][e][i] = u.beta.dd[e][d][i] = sample(random);
        }
      }
      const auto geometry = h::Geometry(u.metric);
      Check(geometry.valid, "test metric must be positive definite");
      const auto rate = h::ShiftMetricRate(u.metric, u.beta);
      double got[3], source[3];
      h::ApplyEllipticShift(u.metric, geometry, u.beta, got);
      h::EllipticShiftSource(u, geometry, 0.4, source);
      auto plus = u.metric, minus = u.metric;
      constexpr double dt = 1e-5;
      for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        plus.g[i][j] += dt*rate.h[i][j];
        minus.g[i][j] -= dt*rate.h[i][j];
        for (int d = 0; d < 3; ++d) {
          plus.dg[d][i][j] += dt*rate.dh[d][i][j];
          minus.dg[d][i][j] -= dt*rate.dh[d][i][j];
        }
      }
      const auto gp = h::Geometry(plus), gm = h::Geometry(minus);
      for (int i = 0; i < 3; ++i)
        variation_error = std::max(variation_error,
            std::abs(got[i]-(gp.contracted[i]-gm.contracted[i])/(2*dt)));
      h::OmegaJet<double> omega{};
      omega.omega = 0.2;
      const auto rhs = h::ConformalRHS(u, omega, 4., -0.99);
      Check(rhs.valid, "tensor RHS invalid");
      for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        rhs_error = std::max(rhs_error, std::abs(rhs.regular.metric[i][j]
                            -(rate.h[i][j]-2*u.alpha.value*u.a.k[i][j])));
      // Lambda is arbitrary and remains unrelated to the metric connection.
      auto independent = u;
      for (int i = 0; i < 3; ++i) independent.lambda.value[i] = 1e6*(i+1);
      double source2[3];
      h::EllipticShiftSource(independent, geometry, 0.4, source2);
      for (int i = 0; i < 3; ++i) Check(source[i] == source2[i], "Lambda entered gauge");
      h::MetricJet<double> flat{};
      for (int i = 0; i < 3; ++i) flat.g[i][i] = 1;
      h::ApplyEllipticShift(flat, h::Geometry(flat), u.beta, got);
      for (int i = 0; i < 3; ++i) {
        double exact = 0;
        for (int j = 0; j < 3; ++j)
          exact += u.beta.dd[j][j][i]+u.beta.dd[i][j][j]/3;
        flat_error = std::max(flat_error, std::abs(got[i]-exact));
      }
    }
    // Compare with the independently derived expanded 1D equation, with damping
    // and nonflat spherical metric. This is a pointwise tensor-reduction check.
    for (double r : {0.03, 0.14, 0.45, 0.81, 1.0}) {
      const double g = std::exp(.3*r*r), dg = .6*r*g;
      const double ddg = (.6+.36*r*r)*g, b = 1/std::sqrt(g);
      const double db = -.3*r*b, ddb = (-.3+.09*r*r)*b;
      h::Z4cJet<double> u{};
      u.metric = h::RadialTensor(g, dg, ddg, b, db, ddb, r);
      u.alpha = h::RadialScalar((1+r*r)/2, r, 1., r);
      const double a = .02*r*r, da = .04*r;
      const double transverse = -a*b/(2*g);
      const auto curvature = h::RadialTensor(a, da, 0., transverse,
          -(da*b+a*db-a*b*dg/g)/(2*g), 0., r);
      for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        u.a.k[i][j] = curvature.g[i][j];
        for (int d = 0; d < 3; ++d) u.a.dk[d][i][j] = curvature.dg[d][i][j];
      }
      const double beta = -r+.2*r*r*r, d = -1+.6*r*r, dd = 1.2*r;
      u.beta = h::RadialVector(beta, d, dd, r);
      const auto geometry = h::Geometry(u.metric);
      double applied[3], source[3];
      h::ApplyEllipticShift(u.metric, geometry, u.beta, applied);
      h::EllipticShiftSource(u, geometry, .7, source);
      const double P = -2*dg/g+(std::pow(g, 1.5)+2)/r;
      const double C = 4*g/3, D = dg-4*g/(3*r);
      const double dC = 4*dg/3, dD = ddg-4*dg/(3*r)+4*g/(3*r*r);
      const double V = dg/(g*g)+2*(std::sqrt(g)-1/g)/r;
      const double residual = C*dd+(dC+D+P*C)*d+(dD+P*D)*beta
          -2*(r*a+u.alpha.value*da)-2*P*u.alpha.value*a+.7*g*g*V;
      radial_error = std::max(radial_error, std::abs(applied[0]-source[0]-residual/(g*g)));
      for (int i = 1; i < 3; ++i) Check(std::abs(applied[i]-source[i])<1e-14,
                                                "radial transverse residual");
    }
    // Non-diagonal conformal metric at scri: metric-normal and coordinate-radial
    // tangential conditions differ. Test the chosen fixed-angular prescription.
    h::MetricJet<double> metric{};
    metric.g[0][0] = 1.2; metric.g[1][1] = .9; metric.g[2][2] = 1.1;
    metric.g[0][1] = metric.g[1][0] = .17;
    const auto geometry = h::Geometry(metric);
    const double x[3] = {.6, 0., .8};
    double beta[3], normal = 0, norm = 0, tangential = 0;
    Check(h::FixedAngleScriShift(geometry, 1., .8, x, beta), "scri boundary invalid");
    for (int i = 0; i < 3; ++i) {
      normal += beta[i]*x[i];
      for (int j = 0; j < 3; ++j) norm += .8*geometry.inverse[i][j]*x[i]*x[j];
    }
    for (int i = 0; i < 3; ++i)
      tangential = std::max(tangential, std::abs(beta[i]-normal*x[i]));
    Check(normal < 0, "wrong future-scri branch");
    Check(std::abs(normal*normal-norm)<1e-14 && tangential<1e-14,
          "scri null or angular boundary residual");
    h::Z4cJet<double> u{};
    const double r = .2, B = .03, dB = -.04, w = .7, dw = -.2;
    u.alpha = h::RadialScalar(.6, .3, 0., r);
    u.chi.value = .8;
    u.beta = h::RadialVector(-.2, -.4, 0., r);
    const double da[3] = {.2, 0, 0}, window_gradient[3] = {dw, 0, 0};
    const auto telegraph = h::ReferenceTelegraph(u, h::RadialVector(B, dB, 0., r),
        da, w, window_gradient, .02, .005);
    Check(std::abs(telegraph.lapse-.8*(w*dB+(dw+2*w/r)*B))<1e-15,
          "telegraph radial divergence mismatch");
    Check(std::abs(telegraph.flux[0]-(-.2*dB+(.005*w*.1-B)/.02))<1e-14,
          "telegraph radial transport mismatch");
    Check(variation_error<2e-10 && flat_error<1e-14 && rhs_error<1e-14
          && radial_error<2e-12, "exact metric-connection operator test failed");
    std::cout << std::setprecision(17) << "{\"variation_error\":" << variation_error
              << ",\"flat_error\":" << flat_error << ",\"metric_rhs_error\":" << rhs_error
              << ",\"radial_error\":" << radial_error << ",\"scri_null_error\":"
              << std::abs(normal*normal-norm) << ",\"scri_angular_error\":" << tangential
              << ",\"cases\":32,\"status\":\"passed\",\"selected_telegraph_support\":[";
    for(int i=0;i<=100;++i){const double r=i/100.;const auto support=h::ReferenceTelegraphSupport(r);
      Check(support.valid,"invalid selected support");if(i)std::cout<<',';
      std::cout<<'['<<r<<','<<support.value<<','<<support.radial_derivative<<']';}
    std::cout<<"]}\n";
  } catch (...) {
    Kokkos::finalize();
    throw;
  }
  Kokkos::finalize();
}
