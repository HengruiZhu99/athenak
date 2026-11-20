// Integrate over colatitude to remove the integrable 1/sin(theta) in dpsi/dt.
// A cubic in theta is constrained by the four finite-volume means in dmu.
#ifndef BOLTZMANN_POLAR_QUADRATURE_HPP_
#define BOLTZMANN_POLAR_QUADRATURE_HPP_
#include <algorithm>
#include <cmath>

#include "athena.hpp"

namespace boltzmann {
struct PolarQuadrature {
  int offset;
  Real mu[4], weight[4], basis[4][4];
};
inline PolarQuadrature MakePolarQuadrature(int band, int hemisphere, Real start,
                                           Real dm) {
  constexpr double nodes[] = {-0.8611363115940526, -0.3399810435848563,
                              0.3399810435848563, 0.8611363115940526};
  constexpr double weights[] = {0.3478548451374539, 0.6521451548625461,
                                0.6521451548625461, 0.3478548451374539};
  auto edge = [&](int k) {
    return std::acos(std::clamp(start + k * dm, -1., 1.));
  };
  PolarQuadrature rule{};
  const int k = band % hemisphere;
  const int left = start + (band + 0.5) * dm < 0 ? 2 : 1;
  rule.offset = std::clamp(k - left, 0, hemisphere - 4) - k;
  const double mid = (edge(band) + edge(band + 1)) / 2,
               width = edge(band) - edge(band + 1);
  // Transposed moment matrix, augmented with basis values at the four nodes.
  double matrix[4][8]{};
  for (int s = 0; s < 4; ++s) {
    const int b = band + rule.offset + s;
    const double lo = edge(b + 1), hi = edge(b);
    double norm = 0;
    for (int q = 0; q < 4; ++q) {
      double theta = (lo + hi + (hi - lo) * nodes[q]) / 2;
      double w = weights[q] * std::sin(theta), x = (theta - mid) / width;
      norm += w;
      for (int l = 0; l < 4; ++l) matrix[l][s] += w * std::pow(x, l);
    }
    for (int l = 0; l < 4; ++l) matrix[l][s] /= norm;
  }
  double total = 0;
  for (int q = 0; q < 4; ++q) {
    const double theta = mid + width * nodes[q] / 2;
    rule.mu[q] = std::cos(theta);
    total += (rule.weight[q] = weights[q] * std::sin(theta));
    for (int l = 0; l < 4; ++l) matrix[l][4 + q] = std::pow(nodes[q] / 2, l);
  }
  for (int l = 0; l < 4; ++l) {
    int pivot = l;
    for (int i = l + 1; i < 4; ++i)
      if (std::abs(matrix[i][l]) > std::abs(matrix[pivot][l])) pivot = i;
    for (int j = 0; j < 8; ++j) std::swap(matrix[l][j], matrix[pivot][j]);
    double scale = matrix[l][l];
    for (int j = l; j < 8; ++j) matrix[l][j] /= scale;
    for (int i = 0; i < 4; ++i)
      if (i != l) {
        double factor = matrix[i][l];
        for (int j = l; j < 8; ++j) matrix[i][j] -= factor * matrix[l][j];
      }
  }
  for (int q = 0; q < 4; ++q) {
    rule.weight[q] /= total;
    for (int s = 0; s < 4; ++s) rule.basis[q][s] = matrix[s][4 + q];
  }
  return rule;
}
}  // namespace boltzmann
#endif
