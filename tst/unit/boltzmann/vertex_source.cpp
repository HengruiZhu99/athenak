// Manufactured volume integrals for native finite-ordinate vertex sources.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>

#include "boltzmann/source_projection.hpp"

namespace {
using namespace boltzmann;
using Policy = Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<2>>;
constexpr double k = 6.283185307179586;
// A regular, anisotropic radiation tensor in an analytic, nondiagonal coframe.
void Solution(double r, double z, double v[10], double c[3][3]) {
  double e = 1 + .12 * cos(k * r) * sin(k * z + .37) +
             .08 * cos(1.7 * k * r) * cos(1.2 * k * z);
  c[0][0] = 1 + .07 * cos(k * z);
  c[1][1] = 1.05 + .04 * cos(k * r) * sin(k * z);
  c[2][2] = c[0][0] + .03 * r * r;
  c[0][1] = .03 * r * sin(k * z);
  c[0][2] = .015 * r * r;
  c[1][2] = .025 * r;
  v[6] = e;
  v[7] = .10 * r * e;
  v[8] = -.11 * e * cos(k * z);
  v[9] = .04 * r * e;
  v[0] = e * (.27 + .03 * r * r);
  v[5] = .27 * e;
  v[3] = e - v[0] - v[5];
  v[1] = .02 * r * e;
  v[2] = .01 * r * r * e;
  v[4] = .015 * r * e;
}

std::array<Real, 3> Patch(int n, bool fine) {
  const Real dx = (fine ? .5 : 1.) / n, x0 = fine ? .5 : 0.;
  const int nx = fine ? n : n / 2, ny = fine ? 2 * n : n;
  DvceArray5D<Real> means("weighted cell means", 1, 10, 1, ny, nx);
  DvceArray5D<Real> adm("vertex metric", 1, 6, 1, ny + 1, nx + 1);
  DvceArray5D<Real> result("covariant vertex source", 1, 10, 1, ny + 1, nx + 1);
  auto u = Kokkos::create_mirror_view(means),
       g = Kokkos::create_mirror_view(adm);
  constexpr int sym[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
  constexpr double pg[4] = {-.8611363115940526, -.3399810435848563,
                            .3399810435848563, .8611363115940526};
  constexpr double wg[4] = {.3478548451374539, .6521451548625461,
                            .6521451548625461, .3478548451374539};
  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
      const double rc = x0 + (i + .5) * dx;
      for (int l = 0; l < 10; ++l) u(0, l, 0, j, i) = 0;
      for (int b = 0; b < 4; ++b)
        for (int a = 0; a < 4; ++a) {
          double c[3][3]{}, v[10], r = rc + .5 * dx * pg[a],
                                   z = (j + .5 + .5 * pg[b]) * dx;
          Solution(r, z, v, c);
          const double weight =
              .25 * wg[a] * wg[b] * r / rc * c[0][0] * c[1][1] * c[2][2];
          for (int l = 0; l < 10; ++l) u(0, l, 0, j, i) += weight * v[l];
        }
    }
  auto exact = Kokkos::create_mirror(result);
  for (int j = 0; j <= ny; ++j)
    for (int i = 0; i <= nx; ++i) {
      double c[3][3]{}, v[10];
      Solution(x0 + i * dx, j * dx, v, c);
      exact(0, 6, 0, j, i) = v[6];
      for (int a = 0; a < 3; ++a) {
        double s = 0;
        for (int p = 0; p < 3; ++p) s += c[a][p] * v[7 + p];
        exact(0, 7 + a, 0, j, i) = s;
        for (int b = a; b < 3; ++b) {
          double metric = 0, stress = 0;
          for (int p = 0; p < 3; ++p) {
            metric += c[a][p] * c[b][p];
            for (int q = 0; q < 3; ++q)
              stress += c[a][p] * c[b][q] * v[sym[p][q]];
          }
          g(0, sym[a][b], 0, j, i) = metric;
          exact(0, sym[a][b], 0, j, i) = stress;
        }
      }
    }

  Kokkos::deep_copy(means, u);
  Kokkos::deep_copy(adm, g);
  Kokkos::parallel_for(
      "project native source", Policy({0, 0}, {ny + 1, nx + 1}),
      KOKKOS_LAMBDA(int j, int i) {
        auto radial = RadialSourceStencil(i, 0, nx - 1, x0 / dx);
        auto vertical = VerticalSourceStencil(j, 0, ny - 1);
        Real v[10];
        if (!ProjectSourceMoments(means, 0, j, i, radial, vertical, v))
          Kokkos::abort("smooth source triggered realizability fallback");
        constexpr int index[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
        Real metric[3][3];
        for (int a = 0; a < 3; ++a)
          for (int b = 0; b < 3; ++b)
            metric[a][b] = adm(0, index[a][b], 0, j, i);
        Frame f{};
        f.Set(metric);
        StoreVertexSource(v, f, result, 0, j, i, x0 == 0 && i == 0);
      });
  auto actual =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);
  std::array<Real, 3> errors{};
  for (int j = 0; j <= ny; ++j)
    for (int i = 0; i <= nx; ++i)
      for (int l = 0; l < 10; ++l) {
        const Real error =
            std::abs(actual(0, l, 0, j, i) - exact(0, l, 0, j, i));
        errors[0] = std::max(errors[0], error);
        if (!fine && i == 0) errors[1] = std::max(errors[1], error);
        if ((fine && i == 0) || (!fine && i == nx))
          errors[2] = std::max(errors[2], error);
      }
  return errors;
}

bool Sharp() {
  DvceArray5D<Real> means("realizable sharp cell moments", 1, 10, 1, 4, 4);
  Kokkos::deep_copy(means, 0.);
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), means);
  // The extrapolating boundary stencil samples a bright second cell with
  // negative weight. Its positive adjacent cell supplies the fallback.
  for (int j = 0; j < 4; ++j)
    for (int i = 0; i < 2; ++i) {
      h(0, 6, 0, j, i) = h(0, 0, 0, j, i) = h(0, 7, 0, j, i) = i ? 100 : 1;
    }
  Kokkos::deep_copy(means, h);
  int errors = 0;
  Kokkos::parallel_reduce(
      "verify realizability and vacuum", Kokkos::RangePolicy<DevExeSpace>(0, 1),
      KOKKOS_LAMBDA(int, int& e) {
        Real v[10];
        auto z = VerticalSourceStencil(2, 0, 3);
        if (ProjectSourceMoments(means, 0, 2, 0,
                                 RadialSourceStencil(0, 0, 3, 0), z, v) ||
            v[6] != 1 || !RealizableSource(v))
          ++e;
        ProjectSourceMoments(means, 0, 2, 4, RadialSourceStencil(4, 0, 3, 0), z,
                             v);
        if (v[6] != 0) ++e;
        v[6] = 1;
        v[0] = v[3] = v[5] = 1. / 3;
        v[7] = .9;
        if (RealizableSource(v)) ++e;
      },
      errors);
  return errors == 0;
}
}  // namespace
int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  bool ok = Sharp();
  std::array<Real, 3> previous{};
  std::puts(
      "n,all_Linf,axis_Linf,interface_Linf,all_order,axis_order,interface_"
      "order");
  for (int n : {24, 48, 96, 192, 384}) {
    auto e = Patch(n, false), f = Patch(n, true);
    std::array<double, 3> order{};
    for (int c = 0; c < 3; ++c) {
      e[c] = std::max(e[c], f[c]);
      order[c] = previous[c] ? log2(previous[c] / e[c]) : 0;
      if (!std::isfinite(e[c]) || (n >= 192 && order[c] < 3.7)) ok = false;
    }
    std::printf("%d,%.12e,%.12e,%.12e,%.5f,%.5f,%.5f\n", n, e[0], e[1], e[2],
                order[0], order[1], order[2]);
    previous = e;
  }
  return ok ? 0 : 1;
}
