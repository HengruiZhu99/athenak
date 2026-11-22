// Independent exact phase-cell averages of a smooth variable-coefficient flux.
// Exercise each of the four production face operators, including the periodic
// seam and both one-sided latitude reconstructions, on its actual Kokkos
// backend.
#include <cmath>
#include <cstdio>

#include "boltzmann/fourth_order.hpp"
#include "boltzmann/polar_quadrature.hpp"
#include "driver/ssprk104.hpp"

namespace {
constexpr Real pi2 = 6.28318530717958647692528676656;
KOKKOS_INLINE_FUNCTION Real sinc(Real x) { return x == 0 ? 1 : sin(x) / x; }
KOKKOS_INLINE_FUNCTION Real Density(Real x, Real h) {
  return 1 + 0.3 * sinc(pi2 * h / 2) * sin(pi2 * x + 0.21);
}
KOKKOS_INLINE_FUNCTION Real Velocity(Real x) {
  return 1 + 0.4 * cos(pi2 * x - 0.37);
}
// Mean of (1 + 0.3 sin(kx+p))(1 + 0.4 cos(kx-q)) on a width-h interval.
KOKKOS_INLINE_FUNCTION Real ExactProduct(Real x, Real h) {
  return Density(x, h) + 0.4 * sinc(pi2 * h / 2) * cos(pi2 * x - 0.37) +
         0.06 * (sin(0.58) + sinc(pi2 * h) * sin(2 * pi2 * x - 0.16));
}
Real Error(int n, int d) {
  const bool sky = d >= 2;
  const int side = sky ? 8 : n + 8, bands = sky ? n : 6, sectors = sky ? n : 6;
  const Real h = 1. / n;
  DvceArray5D<Real> u("means", 1, bands * sectors, 1, side, side);
  DvceArray5D<Real> a("point speed", 1, bands * sectors, 1, side, side);
  DvceArray1D<boltzmann::PolarQuadrature> rules("polar integration", bands);
  auto host = Kokkos::create_mirror_view(rules);
  Kokkos::Array<DvceArray5D<Real>, 3> coeff;
  if (d == 3) {
    for (int k = 0; k < bands; ++k)
      host(k) = boltzmann::MakePolarQuadrature(k, bands, 0, h);
    Kokkos::deep_copy(rules, host);
    for (auto& v : coeff)
      v = DvceArray5D<Real>("polar coefficients", 1, bands * sectors, 1, side,
                            side);
  }
  const int axis = d % 2;
  using Policy = Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "manufactured phase averages",
      Policy({0, 0, 0}, {bands * sectors, side, side}),
      KOKKOS_LAMBDA(int o, int j, int i) {
        Real x = (sky ? o / sectors + 0.5 : i - 4 + 0.5) * h;
        Real y = (sky ? o % sectors + 0.5 : j - 4 + 0.5) * h;
        u(0, o, 0, j, i) = Density(x, h) * Density(y, h);
        Real normal = (axis ? y : x) - h / 2;
        // The cosine-coordinate face speed vanishes at the hemisphere
        // endpoints.
        Real speed =
            d == 2 ? sin(pi2 * normal) : (axis ? -1 : 1) * Velocity(normal);
        a(0, o, 0, j, i) = speed * Velocity(axis ? x : y);
        if (d == 3) {
          auto& rule = rules(o / sectors);
          Real av = 0, c[4]{};
          for (int q = 0; q < 4; ++q) {
            Real v = rule.weight[q] * speed * Velocity(rule.mu[q]);
            av += v;
            for (int k = 0; k < 4; ++k) c[k] += v * rule.basis[q][k];
          }
          a(0, o, 0, j, i) = av;
          int ind = 0;
          for (int k = 0; k < 4; ++k)
            if (k + rule.offset != 0) coeff[ind++](0, o, 0, j, i) = c[k];
        }
      });
  Real error = 0;
  Kokkos::parallel_reduce(
      "manufactured divergence", Kokkos::RangePolicy<DevExeSpace>(0, n * n),
      KOKKOS_LAMBDA(int k, Real& sum) {
        int x = k % n, y = k / n;
        boltzmann::PhaseCell p{sky ? 4 : x + 4, sky ? 4 : y + 4, sky ? x : 2,
                               sky ? y : 2};
        boltzmann::PhaseFace face{d, 0, bands, sectors, u, a};
        Real left = (d == 2 && x == 0) ? 0 : face.Product(p);
        Real right =
            (d == 2 && x == n - 1) ? 0 : face.Product(p.Shift(d, 1, sectors));
        if (d == 3) {
          left = face.PolarFlux(p, rules(p.band).offset, coeff);
          right = face.PolarFlux(p.Shift(d, 1, sectors), rules(p.band).offset,
                                 coeff);
        }
        Real lo = (axis ? y : x) * h, hi = lo + h,
             transverse = ((axis ? x : y) + 0.5) * h;
        Real gl = d == 2 ? sin(pi2 * lo) : (axis ? -1 : 1) * Velocity(lo);
        Real gr = d == 2 ? sin(pi2 * hi) : (axis ? -1 : 1) * Velocity(hi);
        Real exact = (gr * Density(hi, 0) - gl * Density(lo, 0)) *
                     ExactProduct(transverse, h);
        sum += fabs(right - left - exact) / h;
      },
      error);
  return error / (n * n);
}
// Nonlinear autonomous order test, y'=-y^2 from y(0)=1 to y(1)=1/2.
Real TimeError(int n) {
  Real g0[10], g1[10], beta[10], delta[10], y = 1;
  ConfigureSSPRK104(g0, g1, beta, delta);
  for (int i = 0; i < n; ++i) {
    Real old = y;
    for (int s = 0; s < 10; ++s) {
      old += delta[s] * y;
      y = g0[s] * y + g1[s] * old - beta[s] * y * y / n;
    }
  }
  return fabs(y - 0.5);
}
}  // namespace
int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  bool ok = true;
  // Exercise both sides of the periodic seam, including more than one turn.
  for (int sectors : {1, 6, 48}) {
    for (int sector = 0; sector < sectors; ++sector) {
      boltzmann::PhaseCell cell{0, 0, 0, sector};
      for (int step = -2 * sectors; step <= 2 * sectors; ++step) {
        const int expected = ((sector + step) % sectors + sectors) % sectors;
        if (cell.Shift(3, step, sectors).sector != expected) ok = false;
      }
    }
  }
  std::puts("direction,n,L1,order");
  for (int d = 0; d <= 4; ++d) {
    Real prev = 0;
    for (int n : {12, 24, 48, 96}) {
      Real e = d == 4 ? TimeError(n) : Error(n, d),
           order = prev ? log2(prev / e) : 0;
      std::printf("%d,%d,%.12e,%.6f\n", d, n, e, order);
      if (n >= 48 && !(order > 3.85 && std::isfinite(e))) ok = false;
      prev = e;
    }
  }
  Kokkos::finalize();
  return ok ? 0 : 1;
}
