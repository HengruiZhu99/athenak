// Exact volume averages and face integrals on a composite, two-level FV mesh.
#include <cmath>
#include <cstdio>
#include <limits>

#include "boltzmann/fourth_order.hpp"
#include "boltzmann/prolongation.hpp"
#include "mesh/restriction.hpp"
#include "driver/ssprk104.hpp"
#include "mesh/mesh.hpp"

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
          const Real x = left + (i - 4) * dx;
          const Real y = (b < 3 ? 1 : -1) * (j - 3.5) * dx;
          u(0, b, 0, j, i) = Mean(x + dx / 2, dx) * Mean(y, dx);
          a(0, b, 0, j, i) = Speed(x) * Speed(y);
        });
    // Mirrored directional populations F_-(x,z)=F_+(x,-z). Their sum
    // is smooth and even, their difference is smooth and odd. Use the
    // opposite population to supply the equatorial ghosts read by FV4.
    Kokkos::parallel_for(
        "equatorial manufactured ghost fill", Policy({0, 0, 0}, {6, 4, xcells + 8}),
        KOKKOS_LAMBDA(int b, int j, int i) {
          u(0, b, 0, j, i) = u(0, 5-b, 0, 7-j, i);
          a(0, b, 0, j, i) = a(0, 5-b, 0, 7-j, i);
        });
  }
};

Real CompositeDivergence(int n, Real &junction) {
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
  const auto zface = boltzmann::PhaseFace{0, 0, 6, 1, fine.state, fine.speed};
  Real flux_error = 0;
  Kokkos::parallel_reduce("equator and refinement face intersection",
      Kokkos::RangePolicy<DevExeSpace>(0, 2), KOKKOS_LAMBDA(int j, Real &e) {
        const boltzmann::PhaseCell p{4, 4+j, 1, 0};
        const Real dx = .5/n;
        const Real exact = Speed(.5)*Mean(.5, 0)*MeanProduct((j+.5)*dx, dx);
        e = fmax(e, fabs(zface.Product(p)-exact));
      }, Kokkos::Max<Real>(flux_error));
  junction = flux_error;
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
          const boltzmann::PhaseCell p{i + 4, j + 4, 1, 0};
          Real right = face.Product(p.Shift(0, 1, 1));
          if (is_coarse && p.i + 1 == edge) {
            const boltzmann::PhaseCell first{4, 4 + 2 * j, 1, 0};
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
// Two counterstreaming ordinates with exact solution U_+(z,t)=f(z-t),
// U_-(z,t)=f(-z-t). The normal flux of each bin at z=0 is nonzero.
Real EquatorialStreamingError(int n) {
  const Real dx = 1./n, time = .13;
  Level level(1,n,dx,0);
  const auto u = level.state, a = level.speed;
  Kokkos::parallel_for("counterstreaming FV means", Policy({0,4,0},{6,n+8,9}),
      KOKKOS_LAMBDA(int b, int j, int i) {
        const Real sign = b < 3 ? 1 : -1;
        u(0,b,0,j,i) = Mean(sign*(j-3.5)*dx-time,dx);
        a(0,b,0,j,i) = sign;
      });
  Kokkos::parallel_for("counterstreaming equatorial ghosts", Policy({0,0,0},{6,4,9}),
      KOKKOS_LAMBDA(int b, int j, int i) {
        u(0,b,0,j,i) = u(0,5-b,0,7-j,i);
        a(0,b,0,j,i) = -a(0,5-b,0,7-j,i);
      });
  const boltzmann::PhaseFace face{1,0,6,1,u,a};
  Real error = 0, defect = 0;
  Kokkos::parallel_reduce("streaming divergence and equatorial budget",
      Kokkos::RangePolicy<DevExeSpace>(0,2*n),
      KOKKOS_LAMBDA(int index, Real &e, Real &balance) {
        const int j = index % n;
        const int band = index < n ? 1 : 4;
        const Real sign = index < n ? 1 : -1;
        boltzmann::PhaseCell p{4,j+4,band,0};
        const Real exact = sign*(Mean(sign*(j+1)*dx-time,0)-Mean(sign*j*dx-time,0))/dx;
        const Real left = face.Product(p);
        if (!j) {
          const boltzmann::PhaseCell partner{4,4,5-band,0};
          balance = fmax(balance,fabs(left+face.Product(partner)));
        }
        e = fmax(e, fabs((face.Product(p.Shift(1,1,1))-left)/dx-exact));
      }, Kokkos::Max<Real>(error), Kokkos::Max<Real>(defect));
  return defect <= 16*std::numeric_limits<Real>::epsilon() ? error : INFINITY;
}

// Advance both directional cell averages through the symmetry boundary for
// a quarter period. The period-one profile also reflects at the outer face.
Real EvolvedStreamingError(int n, Real &budget) {
  const Real dx = 1./n, final_time = .25;
  const int steps = n/4;
  const Real dt = final_time/steps;
  DvceArray5D<Real> u("streaming means",1,2,1,n+8,1), old("saved means",1,2,1,n+8,1);
  DvceArray2D<Real> flux("shared streaming flux",2,n+1);
  Real curr[10], saved[10], rhs[10], delta[10];
  ConfigureSSPRK104(curr,saved,rhs,delta);
  using Range = Kokkos::RangePolicy<DevExeSpace>;
  Kokkos::parallel_for("initialize both means",Range(0,2*n),KOKKOS_LAMBDA(int index) {
    const int b = index/n, j = index%n;
    u(0,b,0,4+j,0) = Mean((1-2*b)*(j+.5)*dx,dx);
  });
  for (int step=0; step<steps; ++step) {
    Kokkos::deep_copy(old,u);
    for (int stage=0; stage<10; ++stage) {
      Kokkos::parallel_for("reflect streaming means",Range(0,8),KOKKOS_LAMBDA(int index) {
        const int b = index/4, g = index%4;
        u(0,b,0,3-g,0) = u(0,1-b,0,4+g,0);
        u(0,b,0,4+n+g,0) = u(0,1-b,0,3+n-g,0);
      });
      // For constant streaming speed, the production flux reduces exactly
      // to velocity times its one-dimensional finite-volume reconstruction.
      const boltzmann::PhaseFace face{1,0,1,2,u,u};
      Kokkos::parallel_for("shared streaming faces",Range(0,2*(n+1)),KOKKOS_LAMBDA(int index) {
        const int b = index/(n+1), j = index%(n+1);
        const Real sign = 1-2*b;
        flux(b,j) = sign*face.Reconstruct({0,4+j,0,b},sign);
      });
      const Real c = curr[stage], o = saved[stage], d = delta[stage], g = rhs[stage]*dt/dx;
      Kokkos::parallel_for("advance reflected means",Range(0,2*n),KOKKOS_LAMBDA(int index) {
        const int b = index/n, j = index%n;
        auto &v = u(0,b,0,4+j,0), &base = old(0,b,0,4+j,0);
        base += d*v;
        v = c*v + o*base - g*(flux(b,j+1)-flux(b,j));
      });
    }
  }
  Real error = 0, total = 0;
  Kokkos::parallel_reduce("final streaming solution",Range(0,2*n),
      KOKKOS_LAMBDA(int index, Real &max_error, Real &sum) {
        const int b = index/n, j = index%n;
        const Real ufinal = u(0,b,0,4+j,0);
        const Real expected = Mean((1-2*b)*(j+.5)*dx-final_time,dx);
        max_error = fmax(max_error, fabs(ufinal-expected));
        sum += dx*ufinal;
      },Kokkos::Max<Real>(error),Kokkos::Sum<Real>(total));
  budget = fabs(total/2-1);
  return error;
}

bool CheckCoarseNeighborSlots() {
  DvceArray2D<NeighborBlock> neighbors("meridional test neighbors", 1, 16);
  auto host = Kokkos::create_mirror_view(neighbors);
  for (int n = 0; n < 16; ++n) {
    host(0,n).gid = -1;
    host(0,n).lev = -1;
  }
  // The two inner faces meet coarser blocks through opposite child slots.
  // The outer faces meet a same-level block and a pair of finer blocks.
  for (int slot : {0, 4, 9, 12, 13}) host(0,slot).gid = slot;
  host(0,0).lev = host(0,9).lev = 2;
  host(0,4).lev = 3;
  host(0,12).lev = host(0,13).lev = 4;
  Kokkos::deep_copy(neighbors, host);
  int errors = 0;
  Kokkos::parallel_reduce("coarse neighbor slots", Kokkos::RangePolicy<DevExeSpace>(0,4),
      KOKKOS_LAMBDA(const int side, int& sum) {
        if (boltzmann::HasCoarserFace(neighbors, 0, side, 3) != (side % 2 == 0)) ++sum;
      }, errors);
  return errors == 0;
}

bool CheckFourthTransfers() {
  bool ok = true;
  std::puts("AMR operator,n,error,order");
  for (int tile : {4, 6, 0}) {
    Real prev = 0, prev_junction = 0;
    for (int n : {24, 48, 96, 192}) {
      Real junction = 0;
      Real error =
          tile == 0 ? CompositeDivergence(n, junction) : TransferError(n, tile, false);
      Real order = prev ? log2(prev / error) : 0;
      std::printf("%s,%d,%.12e,%.5f\n",
                  tile == 0 ? "composite L1"
                            : (tile == 4 ? "cubic Linf" : "quartic Linf"),
                  n, error, order);
      if (!std::isfinite(error) || (n >= 96 && order < (tile == 6 ? 4.7 : 3.8)))
        ok = false;
      if (tile == 0) {
        const Real observed = prev_junction ? log2(prev_junction/junction) : 0;
        std::printf("equator junction flux Linf,%d,%.12e,%.5f\n", n, junction, observed);
        if (!std::isfinite(junction) || (n >= 96 && observed < 3.8)) ok = false;
        prev_junction = junction;
      }
      prev = error;
    }
  }
  Real previous = 0;
  for (int n : {24,48,96,192}) {
    const Real error = EquatorialStreamingError(n);
    const Real order = previous ? log2(previous/error) : 0;
    std::printf("reflected streaming Linf,%d,%.12e,%.5f\n",n,error,order);
    if (!std::isfinite(error) || (n >= 96 && order < 3.8)) ok = false;
    previous = error;
  }
  previous = 0;
  for (int n : {24,48,96,192}) {
    Real budget = 0;
    const Real error = EvolvedStreamingError(n,budget);
    const Real order = previous ? log2(previous/error) : 0;
    std::printf("evolved reflected pulse Linf,%d,%.12e,%.5f,relative budget %.3e\n",
                n,error,order,budget);
    if (!std::isfinite(error) || budget > 3.e-13 || (n >= 96 && order < 3.8)) ok = false;
    previous = error;
  }
  return std::isfinite(TransferError(24, 4, true)) &&
         std::isfinite(TransferError(24, 6, true)) && ok;
}
}  // namespace

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  return CheckCoarseNeighborSlots() && CheckFourthTransfers() ? 0 : 1;
}
