// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_COMPOSITE_SHIFT_HPP_
#define Z4C_HYPERBOLOIDAL_COMPOSITE_SHIFT_HPP_

#include <array>
#include <chrono>
#include "multigrid/composite_poisson.hpp"
#include "multigrid/flexible_gmres.hpp"
#include "z4c/hyperboloidal/elliptic_shift.hpp"

namespace z4c {
namespace hyperboloidal {

// Stage-owned frozen geometry: setting this does not modify evolved fields or
// project algebraic/differential constraints. Caller rebuilds after any regrid.
class CompositeShiftStage {
 public:
  using Boundary = std::function<std::array<Real, 3>(const std::array<Real, 3> &)>;
  cubed_sphere::CompositeDifference grid;
  std::vector<MetricJet<Real>> metric;
  std::vector<SpatialGeometry<Real>> geometry;
  std::vector<Real> norm_volume;
  multigrid::CompositePoisson preconditioner;

  explicit CompositeShiftStage(cubed_sphere::CompositeMesh mesh, int order = 4)
      : grid(mesh, order), preconditioner(std::move(mesh)) {}

  void SetGeometry(const std::vector<MetricJet<Real>> &g, const std::vector<Real> &chi = {}) {
    if (g.size() != grid.active.size()) throw std::invalid_argument("stage metric shape");
    if (!chi.empty() && chi.size() != g.size()) throw std::invalid_argument("stage chi shape");
    metric = g; geometry.resize(g.size()); norm_volume.resize(g.size());
    for (size_t p = 0; p < g.size(); ++p) {
      geometry[p] = Geometry(g[p]);
      if (!geometry[p].valid) throw std::runtime_error("invalid frozen shift geometry");
      const Real conformal_chi = chi.empty() ? 1 : chi[p];
      if (!(conformal_chi > 0) || !std::isfinite(conformal_chi))
        throw std::runtime_error("invalid conformal-volume coefficient");
      norm_volume[p] = grid.volumes[p]*std::sqrt(geometry[p].determinant)/std::pow(conformal_chi, 1.5);
    }
  }
  void Apply(const multigrid::Vector &beta, multigrid::Vector &out,
              const Boundary &boundary = {}) const {
    const size_t size = grid.active.size();
    if (geometry.size() != size || beta.size() != 3*size)
      throw std::invalid_argument("shift stage geometry/vector shape");
    std::vector<Real> field(3*grid.count, 0);
    for (size_t p = 0; p < size; ++p)
      for (int v = 0; v < 3; ++v) field[v*grid.count+grid.active[p]] = beta[3*p+v];
    grid.Fill(field, 3, boundary ? std::function<Real(int, const std::array<Real, 3> &)>(
        [&](int v, const std::array<Real, 3> &x) {return boundary(x)[v];}) : nullptr);
    out.resize(beta.size());
    for (size_t p = 0; p < size; ++p) {
      VectorJet<Real> jet{};
      for (int v = 0; v < 3; ++v) {
        jet.value[v] = beta[3*p+v];
        Real d[3], dd[3][3]; grid.Derivatives(field, v, p, d, dd);
        for (int a = 0; a < 3; ++a) {
          jet.d[a][v] = d[a];
          for (int b = 0; b < 3; ++b) jet.dd[a][b][v] = dd[a][b];
        }
      }
      Real result[3]; ApplyEllipticShift(metric[p], geometry[p], jet, result);
      for (int v = 0; v < 3; ++v) out[3*p+v] = -result[v];
    }
  }
  double Dot(const multigrid::Vector &a, const multigrid::Vector &b) const {
    if (a.size() != 3*grid.active.size() || b.size() != a.size())
      throw std::invalid_argument("vector norm shape");
    double result = 0, volume = 0;
    for (size_t p = 0; p < grid.active.size(); ++p) {
      for (int v = 0; v < 3; ++v) result += norm_volume[p]*a[3*p+v]*b[3*p+v];
      volume += norm_volume[p];
    }
    return result/volume;
  }
  // Source means the ORIGINAL equation L_g beta = source. Inhomogeneous vector
  // boundary data enter the affine ghost lift; no potential boundary is assumed.
  multigrid::KrylovReport Solve(const multigrid::Vector &source,
      const Boundary &boundary, multigrid::Vector &beta, double atol = 1e-10,
      double rtol = 1e-10, int max_iterations = 160) const {
    multigrid::Vector affine, zero(source.size(), 0), rhs(source.size());
    Apply(zero, affine, boundary);
    for (size_t p = 0; p < rhs.size(); ++p) rhs[p] = -source[p]-affine[p];
    const auto report = multigrid::FlexibleGMRES(
        [&](const auto &a, auto &b) {Apply(a, b);},
        [&](const auto &a, auto &b) {
          b.resize(a.size());
          for (int v = 0; v < 3; ++v) {
            multigrid::Vector scalar(grid.active.size());
            for (size_t p = 0; p < scalar.size(); ++p) scalar[p] = a[3*p+v];
            const auto correction = preconditioner.Cycle(scalar);
            for (size_t p = 0; p < scalar.size(); ++p) b[3*p+v] = correction[p];
          }
        }, [&](const auto &a, const auto &b) {return Dot(a, b);},
        rhs, beta, std::max(atol, rtol*std::sqrt(Dot(source, source))), 0., max_iterations);
    return report;
  }
};

}  // namespace hyperboloidal
}  // namespace z4c
#endif  // Z4C_HYPERBOLOIDAL_COMPOSITE_SHIFT_HPP_
