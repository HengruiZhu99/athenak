#ifndef PGEN_AE93_SHELL_ADM_HPP_
#define PGEN_AE93_SHELL_ADM_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace z4c_irisk {

// Meridional plane of equator-symmetric, maximal initial data from IrisK.
// Scalar order is psi, lapse, eta, K^r_r, K^phi_phi, and
// (K^r_theta/r)/(sin(theta) cos(theta)). The sixth field is regular on
// both reflection planes. Coordinates and returned tensors are physical XYZ.
class Ae93ShellAdm {
 public:
  explicit Ae93ShellAdm(const std::string& path) {
    std::ifstream in(path);
    std::string magic;
    if (!(in >> magic >> n_ >> nt_ >> inner_ >> outer_) ||
        magic != "IRISK_AE93_SHELL_ADM_V1" || n_ < 4 || n_ > 512 || nt_ < 2 ||
        nt_ > 128 || !(inner_ > 0.0 && outer_ > inner_) ||
        !std::isfinite(outer_)) {
      throw std::runtime_error("Invalid IrisK AE93 shell header in " + path);
    }
    coefficients_.resize(3 * 6 * n_ * nt_);
    for (auto& v : coefficients_) {
      if (!(in >> v) || !std::isfinite(v)) {
        throw std::runtime_error("Invalid IrisK AE93 shell coefficient in " +
                                 path);
      }
    }
    std::string trailing;
    if (in >> trailing)
      throw std::runtime_error("Trailing IrisK AE93 shell data");
  }

  void Interpolate(const std::vector<double>& rho, const std::vector<double>& z,
                   std::vector<double>& output) const {
    if (output.size() != 17 * rho.size() * z.size())
      throw std::invalid_argument("Invalid AE93 output buffer extent");
    for (std::size_t j = 0; j < z.size(); ++j)
      for (std::size_t i = 0; i < rho.size(); ++i) {
        const double r = std::hypot(rho[i], z[j]);
        const double s = r > 0 ? rho[i] / r : 0;
        const double c = r > 0 ? z[j] / r : 1;
        const int d = r < inner_ ? 0 : (r < outer_ ? 1 : 2);
        const double x = d == 0   ? 2 * r / inner_ - 1
                         : d == 1 ? 2 * (r - inner_) / (outer_ - inner_) - 1
                                  : 1 - 2 * outer_ / r;
        const double angular = 1 - 2 * c * c;
        std::array<double, 6> field;
        for (int f = 0; f < 6; ++f) {
          const auto* coeff = coefficients_.data() + (6 * d + f) * n_ * nt_;
          double b1 = 0, b2 = 0;
          for (int m = n_ - 1; m > 0; --m) {
            const double b =
                2 * x * b1 - b2 + Chebyshev(coeff + m * nt_, nt_, angular);
            b2 = b1;
            b1 = b;
          }
          field[f] = x * b1 - b2 + Chebyshev(coeff, nt_, angular);
        }
        const auto [psi, alpha, q, kr, kphi, koff] = field;
        if (!(psi > 0 && alpha > 0))
          throw std::runtime_error("Invalid AE93 psi or lapse");
        const double e = std::exp(2 * q);
        const double p = std::pow(psi, 4);
        const double t = -kr - kphi;
        const double h = koff * s * c;
        auto* v = output.data() + 17 * (j * rho.size() + i);
        std::fill(v, v + 17, 0.0);
        v[0] = v[5] = p * e;
        v[3] = p;
        v[6] = p * e * (kr * s * s + t * c * c + 2 * h * s * c);
        v[8] = p * e * ((kr - t) * s * c + h * (c * c - s * s));
        v[9] = p * kphi;
        v[11] = p * e * (kr * c * c + t * s * s - 2 * h * s * c);
        v[12] = p * std::exp(4 * q / 3);
        v[13] = alpha;
      }
  }

 private:
  static double Chebyshev(const double* c, int n, double x) {
    double b1 = 0, b2 = 0;
    for (int l = n - 1; l > 0; --l) {
      const double b = 2 * x * b1 - b2 + c[l];
      b2 = b1;
      b1 = b;
    }
    return x * b1 - b2 + c[0];
  }
  int n_, nt_;
  double inner_, outer_;
  std::vector<double> coefficients_;
};
}  // namespace z4c_irisk
#endif  // PGEN_AE93_SHELL_ADM_HPP_
