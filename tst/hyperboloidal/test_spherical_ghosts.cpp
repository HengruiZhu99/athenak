// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "athena.hpp"  // NOLINT(build/include_subdir): root AthenaK types header
#include "utils/finite_diff.hpp"
#include "z4c/hyperboloidal/spherical_ghosts.hpp"

namespace hyp = z4c::hyperboloidal;
using View = Kokkos::View<double *>;
using Plans = Kokkos::View<hyp::SphericalGhostStencil *>;

struct FlatScalar {
  View q;
  int nx, ny;
  KOKKOS_INLINE_FUNCTION
  Real operator()(int, int k, int j, int i) const { return q(i+nx*(j+ny*k)); }
};
struct RadialShift {
  double origin, h;
  KOKKOS_INLINE_FUNCTION
  Real operator()(int, int a, int k, int j, int i) const {
    const int index[3] = {i,j,k};
    return -(origin+index[a]*h);
  }
};

void Check(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}

hyp::SphericalGhostGrid Grid(int n, bool anisotropic = false) {
  hyp::SphericalGhostGrid g{};
  g.radius = 1;
  for (int a = 0; a < 3; ++a) {
    g.h[a] = 2.*(anisotropic ? 0.8+0.2*a : 1)/n;
    g.n[a] = static_cast<int>(std::ceil(2/g.h[a]))+8;
    g.first[a] = -1-3.5*g.h[a];
  }
  return g;
}

Plans Upload(const std::vector<hyp::SphericalGhostStencil> &stencils) {
  Plans plans("ghost plans", stencils.size());
  auto host = Kokkos::create_mirror_view(plans);
  for (size_t p = 0; p < stencils.size(); ++p) host(p) = stencils[p];
  Kokkos::deep_copy(plans, host);
  return plans;
}

void PolynomialAndCoverage() {
  for (int degree : {2, 3, 4, 5}) {
    const auto g = Grid(degree == 5 ? 48 : 24, true);
    const int count = g.n[0]*g.n[1]*g.n[2];
    const auto stencils = hyp::PlanSphericalGhosts(g, 3, degree);
    std::vector<int> coverage(count, 0);
    View q("polynomial", count);
    auto host = Kokkos::create_mirror_view(q);
    const auto value = [g,degree](int index) {
      const double x = g.first[0]+(index%g.n[0])*g.h[0];
      const double y = g.first[1]+(index/g.n[0]%g.n[1])*g.h[1];
      const double z = g.first[2]+(index/(g.n[0]*g.n[1]))*g.h[2];
      return 1+0.3*x*y+std::pow(0.7*x-0.4*y+0.3*z, degree);
    };
    for (int k = 0; k < g.n[2]; ++k)
    for (int j = 0; j < g.n[1]; ++j)
    for (int i = 0; i < g.n[0]; ++i) {
      const int p = g.Index(i,j,k);
      host(p) = g.Interior(i,j,k) ? value(p) :
          std::numeric_limits<double>::quiet_NaN();
    }
    for (const auto &s : stencils) {
      Check(coverage[s.target] == 0, "duplicate ghost target");
      coverage[s.target] = 1;
      double sum = 0;
      for (int l = 0; l < s.count; ++l) {
        const int d = s.donors[l];
        Check(d >= 0 && d < count, "donor allocation");
        Check(g.Interior(d%g.n[0], d/g.n[0]%g.n[1], d/(g.n[0]*g.n[1])),
              "exterior/recursive donor");
        sum += s.weights[l];
      }
      Check(std::abs(sum-1) < 1e-12, "constant preservation");
    }
    Kokkos::deep_copy(q, host);
    hyp::FillSphericalGhosts(q, Upload(stencils));
    Kokkos::deep_copy(host, q);
    for (const auto &s : stencils) {
      Check(std::abs(host(s.target)-value(s.target)) < 2e-11,
            "mixed polynomial reproduction");
    }
    int mixed = 0;
    for (int k = 3; k < g.n[2]-3; ++k)
    for (int j = 3; j < g.n[1]-3; ++j)
    for (int i = 3; i < g.n[0]-3; ++i) {
      if (!g.Interior(i,j,k)) continue;
      for (int c = -3; c <= 3; ++c)
      for (int b = -3; b <= 3; ++b)
      for (int a = -3; a <= 3; ++a) {
        const int axes = (a != 0)+(b != 0)+(c != 0);
        if (axes > 2 || (axes == 2 &&
            std::max({std::abs(a),std::abs(b),std::abs(c)}) == 3)) continue;
        if (!g.Interior(i+a,j+b,k+c)) {
          Check(coverage[g.Index(i+a,j+b,k+c)] == 1, "missing Cartesian corner");
          if (axes == 2) ++mixed;
        }
      }
    }
    Check(mixed > 0, "test did not cross sphere at mixed corners");
    std::cout << "polynomial degree=" << degree << " ghosts=" << stencils.size()
              << " mixed_corner_requests=" << mixed << '\n';
  }
  bool rejected = false;
  try {
    auto bad = Grid(24);
    bad.first[0] = 0;
    hyp::PlanSphericalGhosts(bad, 3, 4);
  } catch (const std::invalid_argument &) { rejected = true; }
  Check(rejected, "truncated sphere allocation accepted");
  rejected = false;
  try {
    hyp::PlanSphericalGhosts(Grid(24, true), 3, 5);
  } catch (const std::invalid_argument &) { rejected = true; }
  Check(rejected, "unsupported coarse normal stencil was silently degraded");
}

double BoundaryDerivatives(int n) {
  const auto g = Grid(n);
  const int nx = g.n[0], ny = g.n[1], cells = nx*ny*g.n[2];
  const double h = g.h[0], origin = g.first[0];
  View q("smooth Cartesian field", cells);
  Kokkos::deep_copy(q, std::numeric_limits<double>::quiet_NaN());
  Kokkos::parallel_for("smooth interior", cells, KOKKOS_LAMBDA(const int s) {
    const double x = origin+(s%nx)*h, y = origin+(s/nx%ny)*h;
    const double z = origin+(s/(nx*ny))*h;
    if (x*x+y*y+z*z < 1) q(s) = Kokkos::exp(0.7*x-0.4*y+0.3*z);
  });
  hyp::FillSphericalGhosts(q, Upload(hyp::PlanSphericalGhosts(g, 3, 4)));
  double error = 0;
  Kokkos::parallel_reduce("boundary Hessian", cells,
      KOKKOS_LAMBDA(const int s, double &maximum) {
    const int i = s%nx, j = s/nx%ny, k = s/(nx*ny);
    const double x = origin+i*h, y = origin+j*h, z = origin+k*h;
    const double r2 = x*x+y*y+z*z;
    if (r2 >= 1 || r2 < (1-3*h)*(1-3*h)) return;
    const Real idx[3] = {1/h,1/h,1/h}, wave[3] = {0.7,-0.4,0.3};
    FlatScalar scalar{q,nx,ny};
    for (int a = 0; a < 3; ++a) for (int b = a; b < 3; ++b) {
      const double derivative = a == b ? Dxx<3>(a,idx,scalar,0,k,j,i) :
          Dxy<3>(a,b,idx,scalar,0,k,j,i);
      const double diff = Kokkos::isfinite(derivative) ?
          Kokkos::abs(derivative-wave[a]*wave[b]*q(s)) : 1e100;
      if (diff > maximum) maximum = diff;
    }
  }, Kokkos::Max<double>(error));
  std::cout << "boundary Hessian n=" << n << " Linf=" << error << '\n';
  return error;
}

// Exact off-axis smooth solution of q_t + x^i d_i q = 0. Normal speed is
// positive on the entire sphere: no boundary data may be prescribed there.
KOKKOS_INLINE_FUNCTION
double Exact(double x, double y, double z, double t) {
  const double e = Kokkos::exp(-t);
  x = e*x-0.3; y = e*y+0.15; z = e*z-0.2;
  return Kokkos::exp(-30*(x*x+y*y+z*z));
}

struct EvolutionError { double l2, maximum; };

EvolutionError Evolve(int n, int degree, double end, double dissipation = 0.1,
                      bool centered = false) {
  Check(n >= 8 && std::isfinite(end) && end > 0 && std::isfinite(dissipation)
      && dissipation >= 0, "invalid transport options");
  const auto g = Grid(n);
  const auto plans = Upload(hyp::PlanSphericalGhosts(g, 3, degree));
  const int cells = g.n[0]*g.n[1]*g.n[2];
  std::vector<int> ids;
  for (int k = 0; k < g.n[2]; ++k)
  for (int j = 0; j < g.n[1]; ++j)
  for (int i = 0; i < g.n[0]; ++i) {
    if (g.Interior(i,j,k)) ids.push_back(g.Index(i,j,k));
  }
  Kokkos::View<int *> active("interior nodes", ids.size());
  auto ah = Kokkos::create_mirror_view(active);
  for (size_t p = 0; p < ids.size(); ++p) ah(p) = ids[p];
  Kokkos::deep_copy(active, ah);
  View q("transport", cells), stage("RK stage", cells), rhs("RHS", cells);
  View total("RK sum", cells);
  Kokkos::deep_copy(q, std::numeric_limits<double>::quiet_NaN());
  Kokkos::deep_copy(stage, std::numeric_limits<double>::quiet_NaN());
  const int nx = g.n[0], ny = g.n[1];
  const double h = g.h[0], origin = g.first[0];
  Kokkos::parallel_for("initialize off-axis pulse", active.extent(0),
      KOKKOS_LAMBDA(const int p) {
    const int s = active(p), i = s%nx, j = s/nx%ny, k = s/(nx*ny);
    q(s) = Exact(origin+i*h, origin+j*h, origin+k*h, 0);
  });
  double time = 0, max_error = 0;
  int steps = 0;
  while (time < end) {
    const double dt = std::min(0.15*h, end-time);
    Kokkos::parallel_for("RK start", active.extent(0), KOKKOS_LAMBDA(const int p) {
      const int s = active(p);
      stage(s) = q(s);
      total(s) = 0;
    });
    for (int rk = 0; rk < 4; ++rk) {
      hyp::FillSphericalGhosts(stage, plans);
      Kokkos::parallel_for("Cartesian transport RHS", active.extent(0),
          KOKKOS_LAMBDA(const int p) {
        const int s = active(p), i = s%nx, j = s/nx%ny, k = s/(nx*ny);
        const int stride[3] = {1, nx, nx*ny};
        const Real inverse_h[3] = {1/h, 1/h, 1/h};
        FlatScalar scalar{stage, nx, ny};
        const RadialShift beta{origin,h};
        double value = 0;
        for (int a = 0; a < 3; ++a) {
          const int d = stride[a];
          const double advection = centered ?
              beta(0,a,k,j,i)*Dx<3>(a,inverse_h,scalar,0,k,j,i) :
              Lx<3>(a, inverse_h, beta, scalar, 0,a,k,j,i);
          const double ko = (stage(s-3*d)-6*stage(s-2*d)+15*stage(s-d)
              -20*stage(s)+15*stage(s+d)-6*stage(s+2*d)+stage(s+3*d))/(64*h);
          value += advection+dissipation*ko;
        }
        rhs(s) = value;
      });
      const double weight = rk == 0 || rk == 3 ? 1./6 : 1./3;
      const double fraction = rk < 2 ? 0.5 : 1;
      Kokkos::parallel_for("RK stage update", active.extent(0),
          KOKKOS_LAMBDA(const int p) {
        const int s = active(p);
        total(s) += weight*rhs(s);
        stage(s) = q(s)+fraction*dt*rhs(s);
        if (rk == 3) q(s) += dt*total(s);
      });
    }
    time += dt;
    ++steps;
    double error = 0;
    Kokkos::parallel_reduce("transport max error", active.extent(0),
        KOKKOS_LAMBDA(const int p, double &maximum) {
      const int s = active(p), i = s%nx, j = s/nx%ny, k = s/(nx*ny);
      const double expected = Exact(origin+i*h, origin+j*h, origin+k*h, time);
      const double diff = Kokkos::isfinite(q(s)) ? Kokkos::abs(q(s)-expected) : 1e100;
      if (diff > maximum) maximum = diff;
    }, Kokkos::Max<double>(error));
    max_error = std::max(max_error, error);
    if (!(error < 1e3)) {
      std::cerr << "transport failure n=" << n << " degree=" << degree
                << " t=" << time << " max_error=" << error << '\n';
      throw std::runtime_error("Cartesian sphere transport became unstable");
    }
  }
  double sum = 0;
  Kokkos::parallel_reduce("transport L2 error", active.extent(0),
      KOKKOS_LAMBDA(const int p, double &norm) {
    const int s = active(p), i = s%nx, j = s/nx%ny, k = s/(nx*ny);
    const double diff = q(s)-Exact(origin+i*h, origin+j*h, origin+k*h, time);
    norm += diff*diff;
  }, sum);
  const double l2 = std::sqrt(sum/ids.size());
  std::cout << "transport n=" << n << " degree=" << degree << " t=" << time
            << " centered=" << centered << " steps=" << steps << " L2=" << l2
            << " max_time_Linf="
            << max_error << '\n';
  return {l2,max_error};
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  try {
    if (argc >= 4 && argc <= 6) {
      Check(argc != 6 || std::string(argv[5]) == "centered", "unknown advection option");
      Evolve(std::stoi(argv[1]), std::stoi(argv[2]), std::stod(argv[3]),
             argc >= 5 ? std::stod(argv[4]) : 0.1, argc == 6);
      Kokkos::finalize();
      return 0;
    }
    PolynomialAndCoverage();
    const double coarse_hessian = BoundaryDerivatives(24);
    const double fine_hessian = BoundaryDerivatives(48);
    Check(coarse_hessian/fine_hessian > 4, "boundary Hessian does not converge");
    for (int degree : {3, 4, 5}) {
      const int coarse_n = degree == 5 ? 32 : 24;
      const auto coarse = Evolve(coarse_n, degree, 1.5);
      const auto fine = Evolve(2*coarse_n, degree, 1.5);
      Check(coarse.l2/fine.l2 > 5 && coarse.maximum/fine.maximum > 4,
            "Cartesian sphere transport does not converge in space/time norms");
    }
    const auto longer = Evolve(48, 4, 8);
    Check(longer.maximum < 0.02 && longer.l2 < 1e-5, "longer transport bound");
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    status = 1;
  }
  Kokkos::finalize();
  return status;
}
