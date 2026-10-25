// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#ifndef SPHERICAL_CMC_RADIAL_SOLVER_HPP_
#define SPHERICAL_CMC_RADIAL_SOLVER_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "z4c/hyperboloidal/cmc_trumpet.hpp"
#include "z4c/hyperboloidal/spherical_tensor.hpp"

namespace spherical_cmc {
// Host radial solve for vacuum CMC data. S=a_curv=1, K=-3.
// Endpoints are actual boundary nodes; interior nodes are AthenaK cell centers.
// Regular-center data select M=C=0. An excised Schwarzschild trumpet instead
// fixes M>0 and the critical C supplied by the existing trumpet initializer.
// This is not a matter solver or the unconstrained Z4c system.
class RadialSolver {
 public:
  explicit RadialSolver(int cells, double eta, double bh_mass = 0, double rmin = 0)
      : n(CheckedCells(cells)),
        damping(eta),
        mass(bh_mass),
        inner_radius(rmin),
        r(n + 2),
        a(n + 2, 1),
        chi(n + 2),
        alpha(n + 2),
        beta(n + 2),
        arr(n + 2),
        rhs(n + 2),
        dirac(n + 2),
        hamiltonian(n + 2),
        momentum(n + 2),
        cmc(n + 2),
        a_rhs(n + 2),
        chi_rhs(n + 2),
        mass_error(n + 2) {
    if (n < 8 || !std::isfinite(eta) || eta < 0) {
      throw std::invalid_argument("spherical_cmc: nx1>=8 and finite eta>=0 required");
    }
    if (!std::isfinite(mass) || mass < 0 || !std::isfinite(rmin) || rmin < 0 ||
        rmin >= 1 || (mass == 0 && rmin != 0) || (mass > 0 && rmin == 0)) {
      throw std::invalid_argument(
          "spherical_cmc: M=0 requires rmin=0; M>0 requires 0<rmin<1");
    }
    if (mass > 0) trumpet = std::make_unique<z4c::hyperboloidal::CMCTrumpet>(mass);
    r[0] = rmin;
    r[n + 1] = 1;
    for (int i = 1; i <= n; ++i) r[i] = rmin + (i - 0.5) * (1 - rmin) / n;
    for (int i = 0; i <= n + 1; ++i) {
      even_stencil.push_back(MakeStencil(i, 1));
      odd_stencil.push_back(MakeStencil(i, -1));
    }
  }

  struct Stencil {
    int count = 0;
    int index[6]{};
    double d[6]{}, dd[6]{};
  };

  // Fourth-order differentiation, with even/odd reflection at a regular center
  // and six-point closures on the unequal endpoint intervals. Cached once.
  Stencil MakeStencil(int i, int parity) const {
    Stencil out;
    double x[6]{}, sign[6]{1, 1, 1, 1, 1, 1};
    if (inner_radius == 0 && i >= 1 && i <= 2) {
      out.count = 5;
      for (int k = 0; k < 5; ++k) {
        const int j = i + k - 2;
        out.index[k] = j >= 1 ? j : 1 - j;
        x[k] = j >= 1 ? r[j] : -r[1 - j];
        sign[k] = j >= 1 ? 1 : parity;
      }
    } else {
      out.count = (i <= 2 || i >= n - 1) ? 6 : 5;
      const int start = std::max(0, std::min(i - 2, n + 2 - out.count));
      for (int k = 0; k < out.count; ++k) {
        out.index[k] = start + k;
        x[k] = r[start + k];
      }
    }
    for (int j = 0; j < out.count; ++j) {
      double den = 1, first = 0, second = 0;
      for (int k = 0; k < out.count; ++k)
        if (k != j) den *= x[j] - x[k];
      for (int k = 0; k < out.count; ++k) {
        if (k == j) continue;
        double product = 1;
        for (int l = 0; l < out.count; ++l)
          if (l != j && l != k) product *= r[i] - x[l];
        first += product;
        for (int l = k + 1; l < out.count; ++l) {
          if (l == j) continue;
          double p = 1;
          for (int m = 0; m < out.count; ++m)
            if (m != j && m != k && m != l) p *= r[i] - x[m];
          second += 2 * p;
        }
      }
      out.d[j] = sign[j] * first / den;
      out.dd[j] = sign[j] * second / den;
    }
    return out;
  }

  void Derivatives(const std::vector<double> &v, int i, double &d, double &dd,
                   int parity = 1) const {
    const auto &st = parity == 1 ? even_stencil[i] : odd_stencil[i];
    d = 0;
    dd = 0;
    // Difference form preserves constants exactly for even fields.
    for (int k = 0; k < st.count; ++k) {
      const double value = v[st.index[k]] - (parity == 1 ? v[i] : 0);
      d += st.d[k] * value;
      dd += st.dd[k] * value;
    }
  }

  void Solve() {
    for (double v : a) {
      if (!std::isfinite(v) || v <= 0) {
        throw std::runtime_error("spherical_cmc: nonpositive/nonfinite radial metric");
      }
    }
    if (inner_radius == 0 && std::abs(a[0] - 1) > 1e-12) {
      throw std::runtime_error("spherical_cmc: irregular center metric");
    }
    // Hamiltonian first integral: asinh(1/R)=log(1/r)+J,
    // J=int_r^1 (a^(3/4)-1) dr/r. Regular center selects mass=0, A_ij=0.
    std::vector<double> f(n + 2), integral(n + 2, 0), ap(n + 2), app(n + 2);
    for (int i = 0; i <= n + 1; ++i) {
      f[i] = r[i] == 0 ? 0 : std::expm1(0.75 * std::log(a[i])) / r[i];
    }
    for (int i = n; i >= 0; --i) {
      // High-order quadrature is essential here. A trapezoidal J error of O(h^3)
      // next to scri becomes O(h^2) in chi after division by Omega and an O(1)
      // error in its second derivative against the exact endpoint value.
      // Quintic interpolation with 3-point Gauss quadrature also controls the
      // amplification of derivative errors by 1/Omega in tensor diagnostics.
      const int start = std::max(0, std::min(i - 2, n - 4));
      const double mid = (r[i] + r[i + 1]) / 2, half = (r[i + 1] - r[i]) / 2;
      double sum = 0;
      for (int point = -1; point <= 1; ++point) {
        const double x = mid + point * half * std::sqrt(3. / 5);
        const double gauss_weight = point == 0 ? 8. / 9 : 5. / 9;
        for (int j = 0; j < 6; ++j) {
          double weight = 1;
          for (int k = 0; k < 6; ++k) {
            if (k != j) weight *= (x - r[start + k]) / (r[start + j] - r[start + k]);
          }
          sum += gauss_weight * weight * f[start + j];
        }
      }
      integral[i] = integral[i + 1] + half * sum;
    }
    const double clock = std::pow(a[n + 1], 0.75);  // alpha_scri = 1
    std::vector<double> inverse_areal(n + 2, 0), static_shift(n + 2, 0);
    if (trumpet) {
      // Reuse the factored trumpet first integral. RK4 quadrature in conformal
      // radial distance; no finite-radius approximation to the scri initial value.
      for (int i = n; i >= 0; --i) {
        const double interval = std::log(r[i + 1] / r[i]) + integral[i] - integral[i + 1];
        if (!(interval > 0) || !std::isfinite(interval)) {
          throw std::runtime_error(
              "spherical_cmc: unresolved conformal radial quadrature");
        }
        const int steps = std::max(1, static_cast<int>(std::ceil(interval / 0.002)));
        const double dt = interval / steps;
        double u = inverse_areal[i + 1];
        for (int j = 0; j < steps; ++j) {
          const double k1 = trumpet->InverseArealDerivative(u);
          const double k2 = trumpet->InverseArealDerivative(u + dt * k1 / 2);
          const double k3 = trumpet->InverseArealDerivative(u + dt * k2 / 2);
          const double k4 = trumpet->InverseArealDerivative(u + dt * k3);
          u += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6;
        }
        inverse_areal[i] = u;
      }
      if (!(1 / inverse_areal[0] < 2 * mass)) {
        throw std::runtime_error(
            "spherical_cmc inner boundary must lie inside the horizon");
      }
    }
    for (int i = 0; i <= n + 1; ++i) {
      const double b = 1 / std::sqrt(a[i]), omega = (1 - r[i] * r[i]) / 2;
      if (i == n + 1) {
        chi[i] = a[i];  // exact Hamiltonian limit, not 0/0 or a floor
        alpha[i] = 1;
        arr[i] = 0;
        static_shift[i] = -1;
      } else if (trumpet) {
        const double u = inverse_areal[i], c = trumpet->integration_constant();
        chi[i] = b * std::pow(r[i] * u / omega, 2);
        alpha[i] = clock * omega * trumpet->InverseArealDerivative(u) / u;
        arr[i] = -2 * a[i] * c * u * u * u / omega;
        static_shift[i] = clock * r[i] * std::pow(a[i], -0.75) * (-1 + c * u * u * u);
      } else {
        const double ratio = std::exp(-integral[i]) *
                             (1 + std::expm1(2 * integral[i]) / (1 - r[i] * r[i]));
        chi[i] = b * ratio * ratio;
        const double ref = (1 + r[i] * r[i]) / 2;
        const double rad = omega * omega + r[i] * r[i] * b / chi[i];
        // Rationalized deviation from the analytic reference lapse.
        alpha[i] =
            clock * (ref + r[i] * r[i] * (b / chi[i] - 1) / (std::sqrt(rad) + ref));
        arr[i] = 0;
        static_shift[i] = -r[i] * std::pow(a[n + 1] / a[i], 0.75);
      }
      if (!(chi[i] > 0) || !(alpha[i] > 0) || !std::isfinite(chi[i]) ||
          !std::isfinite(alpha[i])) {
        throw std::runtime_error("spherical_cmc: invalid elliptic solution");
      }
      Derivatives(a, i, ap[i], app[i]);
    }
    if (inner_radius == 0) ap[0] = 0;
    // Metric-derived Dirac connection. Solve for the nonhomogeneous shift;
    // beta_static preserves the reconstructed stationary spherical geometry.
    // Unknown y=(beta-beta_static)/r is even at a regular center. The
    // transformed elliptic operator has leading part y''+4y'/r at a=1.
    // Its fixed-width band matrix requires O(N) storage and work.
    constexpr int band = 5;
    using Row = std::array<double, 2 * band + 1>;
    std::vector<Row> matrix(n);
    std::vector<double> source(n), v(n + 2, 0);
    dirac[0] = 0;
    for (int i = (inner_radius == 0 ? 1 : 0); i <= n + 1; ++i) {
      const double u = 1 / a[i], up = -ap[i] * u * u;
      dirac[i] = -up + 2 * u * std::expm1(1.5 * std::log(a[i])) / r[i];
      if (i == 0 || i == n + 1) continue;
      const double upp = 2 * ap[i] * ap[i] * u * u * u - app[i] * u * u;
      const double c = (2 + std::pow(u, -1.5)) / r[i];
      const double p = up / (4 * u) + 1 / r[i] + c;
      const double q = -0.75 * (upp + up * (1 / r[i] + c)) / u;
      const auto &st = even_stencil[i];
      for (int k = 0; k < st.count; ++k) {
        const int j = st.index[k];
        if (j >= 1 && j <= n) matrix[i - 1][band + j - i] += st.dd[k] + p * st.d[k];
      }
      matrix[i - 1][band] += q;
      source[i - 1] = -0.75 * damping * dirac[i] / (u * r[i]);
    }
    const auto original = matrix;
    const auto source0 = source;
    for (int i = 0; i < n; ++i) {
      double scale = 0;
      for (double x : matrix[i]) scale += std::abs(x);
      const double pivot = matrix[i][band];
      if (!std::isfinite(pivot) || std::abs(pivot) < 1e-14 * scale) {
        throw std::runtime_error("spherical_cmc: singular shift elimination pivot");
      }
      for (int j = i + 1; j <= std::min(n - 1, i + band); ++j) {
        const double factor = matrix[j][band + i - j] / pivot;
        for (int k = i + 1; k <= std::min(n - 1, i + band); ++k) {
          matrix[j][band + k - j] -= factor * matrix[i][band + k - i];
        }
        matrix[j][band + i - j] = 0;
        source[j] -= factor * source[i];
      }
    }
    for (int i = n - 1; i >= 0; --i) {
      double value = source[i];
      for (int j = i + 1; j <= std::min(n - 1, i + band); ++j)
        value -= matrix[i][band + j - i] * v[j + 1];
      v[i + 1] = value / matrix[i][band];
      if (!std::isfinite(v[i + 1])) {
        throw std::runtime_error("spherical_cmc: nonfinite elliptic shift");
      }
    }
    shift_residual = 0;
    for (int i = 0; i < n; ++i) {
      double defect = -source0[i];
      for (int j = std::max(0, i - band); j <= std::min(n - 1, i + band); ++j) {
        defect += original[i][band + j - i] * v[j + 1];
      }
      shift_residual =
          std::max(shift_residual, std::abs(defect) / (1 + std::abs(source0[i])));
    }
    if (!std::isfinite(shift_residual) || shift_residual > 1e-8) {
      throw std::runtime_error("spherical_cmc: shift residual exceeds tolerance");
    }
    rhs[0] = 0;
    beta[0] = 0;
    for (int i = (inner_radius == 0 ? 1 : 0); i <= n + 1; ++i) {
      beta[i] = static_shift[i] + r[i] * v[i];
      double vp, vpp;
      Derivatives(v, i, vp, vpp);
      rhs[i] = r[i] * (v[i] * ap[i] + (4. / 3) * a[i] * vp);
      if (!std::isfinite(rhs[i]) || !std::isfinite(beta[i])) {
        throw std::runtime_error("spherical_cmc: nonfinite metric RHS");
      }
    }
    beta[n + 1] = -1;  // exact beta=-alpha*sqrt(chi/a) at scri
    if (trumpet && !(-beta[0] + alpha[0] * std::sqrt(chi[0] / a[0]) < 0)) {
      throw std::runtime_error("spherical_cmc inner boundary is not physical outflow");
    }
  }

  // Independent differential diagnostics through the existing full tensor
  // kernel. No analytic reference RHS is subtracted, and scri is never divided by Omega.
  void Diagnostics() {
    namespace hyp = z4c::hyperboloidal;
    const hyp::CMCReference<double> ref{1, 1};
    std::vector<double> dc(n + 2), da(n + 2), db(n + 2), dg(n + 2);
    for (int i = 0; i <= n + 1; ++i) {
      dc[i] = chi[i] - 1;
      da[i] = alpha[i] - (1 + r[i] * r[i]) / 2;
      db[i] = beta[i] + r[i];
      dg[i] = a[i] - 1;
    }
    for (int i = 1; i <= n; ++i) {
      double val[hyp::NFIELDS]{}, d[hyp::NFIELDS]{}, dd[hyp::NFIELDS]{};
      val[hyp::DCHI] = dc[i];
      val[hyp::DGRR] = dg[i];
      val[hyp::DALPHA] = da[i];
      val[hyp::DBETA] = db[i];
      val[hyp::ARR] = arr[i];
      Derivatives(dc, i, d[hyp::DCHI], dd[hyp::DCHI]);
      Derivatives(dg, i, d[hyp::DGRR], dd[hyp::DGRR]);
      Derivatives(da, i, d[hyp::DALPHA], dd[hyp::DALPHA]);
      Derivatives(db, i, d[hyp::DBETA], dd[hyp::DBETA], -1);
      Derivatives(arr, i, d[hyp::ARR], dd[hyp::ARR]);
      auto u = hyp::SphericalJet(r[i], val, d, dd, ref);
      const auto geo = hyp::Geometry(u.metric);
      for (int j = 0; j < 3; ++j) {
        u.lambda.value[j] = geo.contracted[j];
        for (int k = 0; k < 3; ++k) u.lambda.d[k][j] = geo.dcontracted[k][j];
      }
      hyp::OmegaJet<double> o{};
      o.omega = (1 - r[i] * r[i]) / 2;
      o.gradient[0] = -r[i];
      for (int j = 0; j < 3; ++j) o.hessian[j][j] = -1;
      hyp::SetStationaryOmegaNormal(u.alpha.value, u.beta.value, u.alpha.d, u.beta.d, o);
      const auto cons = hyp::EvolvedConstraints(u, o);
      hyp::Z4cRHS<double> full{};
      if (!cons.valid ||
          !hyp::AssembleInterior(hyp::ConformalRHS(u, o, 0., 0.), o.omega, full)) {
        throw std::runtime_error("spherical_cmc: invalid tensor diagnostic");
      }
      hamiltonian[i] = cons.hamiltonian;
      momentum[i] = cons.momentum[0];
      cmc[i] = full.trace;
      a_rhs[i] = full.a[0][0];
      chi_rhs[i] = full.chi;
      mass_error[i] = hyp::SphereDiagnostics(r[i], u, o).mass - mass;
    }
  }

  const int n;
  const double damping;
  const double mass, inner_radius;
  std::vector<double> r, a, chi, alpha, beta, arr, rhs, dirac, hamiltonian, momentum, cmc,
      a_rhs, chi_rhs, mass_error;
  double shift_residual = 0;

 private:
  static int CheckedCells(int cells) {
    if (cells < 8 || cells > std::numeric_limits<int>::max() - 2) {
      throw std::invalid_argument("spherical_cmc: nx1>=8 required");
    }
    return cells;
  }
  std::unique_ptr<z4c::hyperboloidal::CMCTrumpet> trumpet;
  std::vector<Stencil> even_stencil, odd_stencil;
};
}  // namespace spherical_cmc
#endif  // SPHERICAL_CMC_RADIAL_SOLVER_HPP_
