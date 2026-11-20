// Direct tests of the production reconstruction, subdivision and geodesic rates.
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "boltzmann/geometry.hpp"
#include "boltzmann/prolongation.hpp"
#include "boltzmann/moments.hpp"
#include "boltzmann/reconstruction.hpp"
#include "coordinates/cartesian_ks.hpp"

namespace {
void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
using boltzmann::Characteristics;
using boltzmann::Geometry;
struct ADMPoint {
  Real lapse, shift[3], metric[3][3], curvature[3][3];
};
struct Stencil {
  ADMPoint point[3][3]{};
  Real alpha(int, int, int j, int i) const { return point[j][i].lapse; }
  Real beta_u(int, int a, int, int j, int i) const { return point[j][i].shift[a]; }
  Real g_dd(int, int a, int b, int, int j, int i) const { return point[j][i].metric[a][b]; }
  Real vK_dd(int, int a, int b, int, int j, int i) const { return point[j][i].curvature[a][b]; }
};
ADMPoint Kerr(Real r, Real z, Real spin) {
  ADMPoint p{};
  Real beta[3], g[6], k[6], psi;
  ComputeADMDecomposition(r, 0, z, false, spin, &p.lapse, beta, beta + 1, beta + 2, &psi, g,
                          g + 1, g + 2, g + 3, g + 4, g + 5, k, k + 1, k + 2, k + 3, k + 4,
                          k + 5);
  const int perm[3] = {0, 2, 1}, sym[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
  for (int a = 0; a < 3; ++a) {
    p.shift[a] = beta[perm[a]];
    for (int b = 0; b < 3; ++b) {
      p.metric[a][b] = g[sym[perm[a]][perm[b]]];
      p.curvature[a][b] = k[sym[perm[a]][perm[b]]];
    }
  }
  return p;
}
using Ray = std::array<Real, 5>;  // rho,z,mu,psi,log(epsilon)
constexpr Real spin = 0.6;
Geometry RayGeometry(const Ray& r) {
  Real h = 0.0002;
  Stencil s;
  for (int j = 0; j < 3; ++j)
    for (int i = 0; i < 3; ++i)
      s.point[j][i] = Kerr(r[0] + (i - 1) * h, r[1] + (j - 1) * h, spin);
  Geometry g{};
  g.Set(s, 0, 1, 1, r[0], h, h);
  return g;
}
Ray Derivative(const Ray& r) {
  auto v = Characteristics(RayGeometry(r), r[2], r[3]);
  return {v.speed[0], v.speed[1], v.mu, v.psi, v.redshift};
}
Ray Shift(Ray r, const Ray& v, Real h) {
  for (int i = 0; i < 5; ++i) r[i] += h * v[i];
  return r;
}
void KerrTurn() {
  Ray ray{10, 1., 0.60, 2.85, 0};
  Real energy = 0, angular = 0, min_radius = ray[0];
  for (int it = 0; it < 4000; ++it) {
    auto g = RayGeometry(ray);
    Real mu = ray[2], s = sqrt(1 - mu * mu), n[3] = {s * cos(ray[3]), s * sin(ray[3]), mu};
    Real eps = exp(ray[4]), e = g.alpha;
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) e -= g.beta[a] * g.frame.co[a][b] * n[b];
    e *= eps;
    Real l = eps * g.frame.co[2][2] * mu;
    if (it == 0) {
      energy = e;
      angular = l;
    }
    Require(fabs(e - energy) < 1e-7 && fabs(l - angular) < 1e-6,
            "stationary Kerr Killing charges");
    // Independent radial potential in Boyer-Lindquist r. Compute Carter Q
    // from the angular covector (KS -> BL time and phi shifts depend only on r).
    Real x = ray[0], z = ray[1], rr = x * x + z * z - spin * spin;
    Real bl = sqrt(0.5 * (rr + sqrt(rr * rr + 4 * spin * spin * z * z)));
    Real ct = z / bl, st = sqrt(1 - ct * ct), sigma = bl * bl + spin * spin * ct * ct;
    Real p[2]{};
    for (int i = 0; i < 2; ++i)
      for (int b = 0; b < 3; ++b) p[i] += eps * g.frame.co[i][b] * n[b];
    Real ptheta = p[0] * sqrt(bl * bl + spin * spin) * ct - p[1] * bl * st;
    Real carter = ptheta * ptheta + ct * ct * (l * l / (st * st) - spin * spin * e * e);
    static Real q0;
    if (it == 0) q0 = carter;
    Require(fabs(carter - q0) < 3.e-5, "Kerr Carter invariant");
    auto v = Derivative(ray);
    Real drdt = (bl * x * v[0] + (bl + spin * spin / bl) * z * v[1]) / sigma;
    Real potential = pow(e * (bl * bl + spin * spin) - spin * l, 2) -
                     (bl * bl - 2 * bl + spin * spin) * (q0 + pow(l - spin * e, 2));
    Real radial = sigma * drdt * eps / g.alpha;
    Require(fabs(radial * radial - potential) < 3e-4, "Kerr radial potential");
    min_radius = fmin(min_radius, x);
    if (it > 10 && x > 10.) {
      Require(min_radius < 8, "Kerr ray must turn inside launching cylinder");
      return;
    }
    const Real dt = 0.01;
    auto k2 = Derivative(Shift(ray, v, dt / 2)), k3 = Derivative(Shift(ray, k2, dt / 2)),
         k4 = Derivative(Shift(ray, k3, dt));
    for (int a = 0; a < 5; ++a) ray[a] += dt * (v[a] + 2 * k2[a] + 2 * k3[a] + k4[a]) / 6;
  }
  throw std::runtime_error("Kerr ray did not escape");
}
void Redshift() {
  Stencil s;
  const Real H = 0.3;
  for (auto& row : s.point)
    for (auto& p : row) {
      p.lapse = 1;
      for (int i = 0; i < 3; ++i) {
        p.metric[i][i] = 4;
        p.curvature[i][i] = -4 * H;
      }
    }
  Geometry g{};
  g.Set(s, 0, 1, 1, 3, 0.1, 0.1);
  for (int k = 0; k < 20; ++k) {
    auto rate = Characteristics(g, -0.95 + 0.1 * k, 0.75 * k);
    Require(fabs(rate.redshift + H) < 1e-13, "FLRW redshift source");
    // The evolved energy is densitized by a^3. Test the Eulerian energy law.
    Require(fabs(rate.redshift - 3 * H + 4 * H) < 1e-13, "FLRW energy exponent");
  }
}
Real Advection(int N, bool angular, bool vacuum_front = false) {
  Real dx = 2 * M_PI / N, dt = 0.25 * dx, speed = -1;
  DvceArray5D<Real> u("u", 1, N, 1, 1, 1);
  auto sample = [&](int i, Real t) {
    if (vacuum_front) {
      const Real phase = fmod((i + 0.5) * dx - speed * t + 2 * M_PI, 2 * M_PI);
      return phase < M_PI ? Real{2} : Real{0};
    }
    return 1 + 0.8 * sin((i + 0.5) * dx - speed * t) * sin(dx / 2) / (dx / 2);
  };
  std::vector<Real> old(N), flux(N);
  for (int i = 0; i < N; ++i) u(0, i, 0, 0, 0) = sample(i, 0);
  for (int it = 0; it < N; ++it) {
    for (int i = 0; i < N; ++i) old[i] = u(0, i, 0, 0, 0);
    for (int stage = 0; stage < 2; ++stage) {
      for (int i = 0; i < N; ++i) {
        flux[i] = angular ? boltzmann::AngularFlux(u, 0, 0, 0, 0, i, 1, N, false, speed)
                          : speed * boltzmann::FaceState(u(0, (i + N - 1) % N, 0, 0, 0),
                                                         u(0, i, 0, 0, 0),
                                                         u(0, (i + 1) % N, 0, 0, 0), -1);
      }
      for (int i = 0; i < N; ++i) {
        Real next = u(0, i, 0, 0, 0) - dt / dx * (flux[(i + 1) % N] - flux[i]);
        u(0, i, 0, 0, 0) = stage == 0 ? next : 0.5 * (old[i] + next);
        Require(u(0, i, 0, 0, 0) >= 0, "nonnegative transport stage");
      }
    }
  }
  Real err = 0, sum = 0;
  for (int i = 0; i < N; ++i) {
    Real val = u(0, i, 0, 0, 0);
    sum += val;
    err += fabs(val - sample(i, N * dt));
  }
  Require(fabs(sum / N - 1) < 32 * std::numeric_limits<Real>::epsilon(),
          "transport conservation");
  return err / N;
}
Real LatitudeAdvection(int bands) {
  const Real dx = 1.0 / bands, dt = 0.5 * dx;
  DvceArray5D<Real> u("latitude", 1, bands, 1, 1, 1);
  std::vector<Real> old(bands), flux(bands + 1, 0);
  // Conservative logistic flow dmu/dt = mu(1-mu). Backtrace both
  // faces of each cell and integrate the initial density 1+mu exactly.
  auto primitive = [](Real x, Real t) {
    Real y = x / (x + (1 - x) * exp(t));
    return y + 0.5 * y * y;
  };
  for (int i = 0; i < bands; ++i) u(0, i, 0, 0, 0) = 1 + (i + 0.5) * dx;
  for (int step = 0; step < bands; ++step) {
    for (int i = 0; i < bands; ++i) old[i] = u(0, i, 0, 0, 0);
    for (int stage = 0; stage < 2; ++stage) {
      for (int i = 1; i < bands; ++i) {
        Real x = i * dx;
        flux[i] = boltzmann::AngularFlux(u, 0, 0, 0, i, 0, bands, 1, true, x * (1 - x));
      }
      for (int i = 0; i < bands; ++i) {
        Real next = u(0, i, 0, 0, 0) - dt / dx * (flux[i + 1] - flux[i]);
        u(0, i, 0, 0, 0) = stage ? 0.5 * (old[i] + next) : next;
      }
    }
  }
  Real error = 0, sum = 0;
  for (int i = 0; i < bands; ++i) {
    Real val = u(0, i, 0, 0, 0);
    Require(val >= 0, "latitude positivity");
    error += fabs(dx * val - (primitive((i + 1) * dx, 0.5) - primitive(i * dx, 0.5)));
    sum += dx * val;
  }
  Require(fabs(sum - 1.5) < 64 * std::numeric_limits<Real>::epsilon(), "latitude conservation");
  return error;
}

Real VertexError(Real h, Real radius, bool coarse_neighbor) {
  DvceArray5D<Real> centers("centers", 1, 10, 1, 2, 2), vertex("vertex", 1, 10, 1, 2, 2);
  DvceArray5D<Real> parents("parents", 1, 10, 1, 3, 3), children("children", 1, 10, 1, 2, 2);
  auto field = [](int n, Real r, Real z) {
    // A regular rotating covector in the plane and a symmetric spatial stress.
    const Real scalar = 1 + z * z + r * r;
    switch (n) {
      case 0:
        return scalar + r * r;
      case 5:
        return scalar;
      case 1:
      case 4:
      case 7:
      case 9:
        return r * (1 + z * z);
      case 2:
        return r * r;
      default:
        return scalar;
    }
  };
  for (int n = 0; n < 10; ++n) {
    for (int j = 0; j < 2; ++j)
      for (int i = 0; i < 2; ++i)
        centers(0, n, 0, j, i) = field(n, radius + (i - 0.5) * h, (j - 0.5) * h);
    if (coarse_neighbor) {
      // Adjacent coarse parent covers [radius-2h,radius] x [-h,h].
      // Subdivide densitized data and undensitize on the fine ghost centers.
      for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i) {
          Real r = radius + (2 * i - 3) * h, z = (2 * j - 2) * h;
          parents(0, n, 0, j, i) = r * field(n, r, z);
        }
      boltzmann::ProlongOrdinates(0, n, 1, 1, 0, 0, parents, children);
      for (int j = 0; j < 2; ++j)
        centers(0, n, 0, j, 0) = children(0, n, 0, j, 1) / (radius - 0.5 * h);
    }
  }
  boltzmann::TransferVertex(centers, vertex, 0, 1, 1, radius == 0);
  Real error = 0;
  for (int n = 0; n < 10; ++n)
    error = fmax(error, fabs(vertex(0, n, 0, 1, 1) - field(n, radius, 0)));
  if (radius == 0) {
    Require(vertex(0, 0, 0, 1, 1) == vertex(0, 5, 0, 1, 1), "transverse stress on axis");
    for (int n : {1, 2, 4, 7, 9}) Require(vertex(0, n, 0, 1, 1) == 0, "axis tensor parity");
  }
  return error;
}

void Subdivision() {
  DvceArray5D<Real> parent("parent", 1, 1, 1, 3, 3), child("child", 1, 1, 1, 2, 2);
  Kokkos::deep_copy(parent, 1.);
  parent(0, 0, 0, 1, 2) = 50;
  boltzmann::ProlongOrdinates(0, 0, 1, 1, 0, 0, parent, child);
  Real sum = 0, low = 2;
  for (int j = 0; j < 2; ++j)
    for (int i = 0; i < 2; ++i) {
      low = fmin(low, child(0, 0, 0, j, i));
      sum += child(0, 0, 0, j, i);
    }
  Require(low >= 0 && low < 1e-15 && fabs(sum - 4) < 1e-14,
          "active positivity limiter preserves parent");
}
}  // namespace
int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  Subdivision();
  for (bool angular : {false, true}) {
    Advection(48, angular, true);
    Real coarse = Advection(48, angular), fine = Advection(96, angular);
    std::cout << (angular ? "angular" : "spatial") << " L1 order " << log2(coarse / fine)
              << "\n";
    Require(coarse / fine > 3.4, "smooth PLM convergence");
  }
  Real latitude_order = log2(LatitudeAdvection(48) / LatitudeAdvection(96));
  std::cout << "latitude L1 order " << latitude_order << "\n";
  Require(latitude_order > 1.8, "latitude convergence");
  for (int test = 0; test < 3; ++test) {
    Real radius = test == 0 ? 0 : 1;
    Real order =
        log2(VertexError(0.1, radius, test == 2) / VertexError(0.05, radius, test == 2));
    std::cout << "vertex order " << order << "\n";
    Require(order > 1.8, "vertex convergence");
  }
  Redshift();
  KerrTurn();
}
