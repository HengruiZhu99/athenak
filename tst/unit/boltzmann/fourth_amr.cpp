// Exact volume averages and face integrals on a composite, two-level FV mesh.
#include <cmath>
#include <cstdio>
#include <limits>

#include "boltzmann/fourth_order.hpp"
#include "boltzmann/prolongation.hpp"
#include "mesh/restriction.hpp"

namespace {
using Policy = Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<3>>;
constexpr Real k = 6.2831853071795864769;
KOKKOS_INLINE_FUNCTION Real Sinc(Real x) { return x == 0 ? 1 : sin(x) / x; }
KOKKOS_INLINE_FUNCTION Real Mean(Real x, Real h) {
  return 1 + .3 * Sinc(k * h / 2) * sin(k * x + .21);
}
KOKKOS_INLINE_FUNCTION Real Speed(Real x) { return 1 + .4 * cos(k * x - .37); }
KOKKOS_INLINE_FUNCTION Real MeanProduct(Real x, Real h) {
  return Mean(x, h) + .4 * Sinc(k * h / 2) * cos(k * x - .37) +
         .06 * (sin(.58) + Sinc(k * h) * sin(2 * k * x - .16));
}

Real TransferError(int n, int tile, bool pulse) {
  const Real h = 1. / n;
  DvceArray5D<Real> u("parent means", 1, 1, 1, n, n),
      v("child means", 1, 1, 1, 2 * n, 2 * n);
  Kokkos::parallel_for(
      "fill transfer data", Policy({0, 0, 0}, {1, n, n}),
      KOKKOS_LAMBDA(int, int j, int i) {
        u(0, 0, 0, j, i) =
            pulse ? ((j + i) % 7 == 0 ? 100. : ((i + j) % 7 == 1 ? .1 : 0.))
                  : Mean((i + .5) * h, h) * Mean((j + .5) * h, h);
      });
  Kokkos::parallel_for(
      "subdivide volume means", Policy({0, 0, 0}, {1, n, n}),
      KOKKOS_LAMBDA(int, int j, int i) {
        int il = i - i % tile, jl = j - j % tile;
        boltzmann::ProlongOrdinatesFourth(0, 0, j, i, 2 * j, 2 * i, il,
                                          il + tile - 1, jl, jl + tile - 1, u,
                                          v);
      });
  Real error = 0, roundtrip = 0, minimum = 0;
  Kokkos::parallel_reduce(
      "check integrated child means", Policy({0, 0, 0}, {1, n, n}),
      KOKKOS_LAMBDA(int, int j, int i, Real& e, Real& conservation, Real& low) {
        conservation = fmax(
            conservation, fabs(RestrictCellAverage2D(v, 0, 0, 0, 2 * j, 2 * i) -
                               u(0, 0, 0, j, i)));
        for (int z = 2 * j; z < 2 * j + 2; ++z)
          for (int x = 2 * i; x < 2 * i + 2; ++x) {
            Real val = v(0, 0, 0, z, x);
            low = fmin(low, val);
            e = fmax(e, fabs(val - Mean((x + .5) * h / 2, h / 2) *
                                       Mean((z + .5) * h / 2, h / 2)));
          }
      },
      Kokkos::Max<Real>(error), Kokkos::Max<Real>(roundtrip),
      Kokkos::Min<Real>(minimum));
  if (!(roundtrip < 200 * std::numeric_limits<Real>::epsilon() && minimum >= 0))
    return INFINITY;
  return pulse ? roundtrip : error;
}

struct Level {
  DvceArray5D<Real> state, speed;
  Real h, x0;
  int nx, ny;
  Level(int xcells, int ycells, Real dx, Real left)
      : state("composite ordinates", 1, 6, 1, ycells + 8, xcells + 8),
        speed("composite face speeds", 1, 6, 1, ycells + 8, xcells + 8),
        h(dx),
        x0(left),
        nx(xcells),
        ny(ycells) {}
  void Fill() const {
    const Real left = x0, dx = h;
    const int xcells = nx, ycells = ny;
    const auto u = state, a = speed;
    Kokkos::parallel_for(
        "exact composite data", Policy({0, 0, 0}, {6, ycells + 8, xcells + 8}),
        KOKKOS_LAMBDA(int b, int j, int i) {
          const Real x = left + (i - 4) * dx, y = (j - 3.5) * dx;
          u(0, b, 0, j, i) = Mean(x + dx / 2, dx) * Mean(y, dx);
          a(0, b, 0, j, i) = Speed(x) * Speed(y);
        });
  }
};

Real CompositeDivergence(int n) {
  Level coarse(n / 2, n, 1. / n, 0), fine(n, 2 * n, .5 / n, .5);
  coarse.Fill();
  fine.Fill();
  const auto u = coarse.state, v = fine.state;
  const int edge = 4 + n / 2;
  // The last four coarse ghosts receive exact restrictions of the fine data.
  Kokkos::parallel_for(
      "coarse boundary restriction", Policy({0, 0, 0}, {6, n, 4}),
      KOKKOS_LAMBDA(int b, int j, int i) {
        u(0, b, 0, 4 + j, edge + i) =
            RestrictCellAverage2D(v, 0, b, 0, 4 + 2 * j, 4 + 2 * i);
      });
  // This is the same normal/tangential parent rectangle as a production face.
  Kokkos::parallel_for(
      "fine boundary prolongation", Policy({0, 0, 0}, {6, n + 4, 2}),
      KOKKOS_LAMBDA(int b, int j, int i) {
        boltzmann::ProlongOrdinatesFourth(0, b, j + 2, edge - 2 + i, 2 * j,
                                          2 * i, edge - 4, edge + 3, 2, n + 5,
                                          u, v);
      });
  const boltzmann::PhaseFace child_face{0, 0, 6, 1, fine.state, fine.speed};
  Real l1 = 0;
  for (const auto& level : {coarse, fine}) {
    const bool is_coarse = level.x0 == 0;
    const Real dx = level.h, left = level.x0;
    const boltzmann::PhaseFace face{0, 0, 6, 1, level.state, level.speed};
    Real sum = 0;
    Kokkos::parallel_reduce(
        "composite finite volume divergence",
        Policy({0, 0, 0}, {1, level.ny, level.nx}),
        KOKKOS_LAMBDA(int, int j, int i, Real& err) {
          const boltzmann::PhaseCell p{i + 4, j + 4, 2, 0};
          Real right = face.Product(p.Shift(0, 1, 1));
          if (is_coarse && p.i + 1 == edge) {
            const boltzmann::PhaseCell first{4, 4 + 2 * j, 2, 0};
            right = .5 * (child_face.Product(first) +
                          child_face.Product(first.Shift(1, 1, 1)));
          }
          Real x = left + i * dx, y = (j + .5) * dx;
          Real integral =
              (Speed(x + dx) * Mean(x + dx, 0) - Speed(x) * Mean(x, 0)) *
              MeanProduct(y, dx);
          err += fabs(right - face.Product(p) - integral) * dx;
        },
        sum);
    l1 += sum;
  }
  return l1;
}
bool CheckFourthTransfers() {
  bool ok = true;
  std::puts("AMR operator,n,error,order");
  for (int tile : {4, 6, 0}) {
    Real prev = 0;
    for (int n : {24, 48, 96, 192}) {
      Real error =
          tile == 0 ? CompositeDivergence(n) : TransferError(n, tile, false);
      Real order = prev ? log2(prev / error) : 0;
      std::printf("%s,%d,%.12e,%.5f\n",
                  tile == 0 ? "composite L1"
                            : (tile == 4 ? "cubic Linf" : "quartic Linf"),
                  n, error, order);
      if (!std::isfinite(error) || (n >= 96 && order < (tile == 6 ? 4.7 : 3.8)))
        ok = false;
      prev = error;
    }
  }
  return std::isfinite(TransferError(24, 4, true)) &&
         std::isfinite(TransferError(24, 6, true)) && ok;
}
}  // namespace

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  return CheckFourthTransfers() ? 0 : 1;
}
