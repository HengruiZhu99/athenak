// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>
#include "athena.hpp"  // NOLINT(build/include_subdir): root AthenaK types header
#include "utils/finite_diff.hpp"
#include "z4c/hyperboloidal/spherical_ghosts.hpp"

namespace hyp = z4c::hyperboloidal;
using View = Kokkos::View<double *>;
using Plans = Kokkos::View<hyp::SphericalGhostStencil *>;

void Check(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}

hyp::SphericalGhostGrid Grid(int n) {
  hyp::SphericalGhostGrid g;
  g.radius = 1;
  for (int a = 0; a < 3; ++a) {
    g.n[a] = n+6;
    g.h[a] = 2.1/n;
    g.first[a] = -1.05-2.5*g.h[a];
  }
  return g;
}

int Transform(int s, const hyp::SphericalGhostGrid &g,
              const std::array<int,3> &perm, int reflect) {
  const int index[3] = {s%g.n[0],s/g.n[0]%g.n[1],s/(g.n[0]*g.n[1])};
  int out[3];
  for (int a = 0; a < 3; ++a) {
    out[a] = reflect & (1 << a) ? g.n[a]-1-index[perm[a]] : index[perm[a]];
  }
  return g.Index(out[0],out[1],out[2]);
}

double Polynomial(int s, const hyp::SphericalGhostGrid &g, int a, int b, int c) {
  const double x = g.first[0]+(s%g.n[0])*g.h[0];
  const double y = g.first[1]+(s/g.n[0]%g.n[1])*g.h[1];
  const double z = g.first[2]+(s/(g.n[0]*g.n[1]))*g.h[2];
  return std::pow(x,a)*std::pow(y,b)*std::pow(z,c);
}

Plans Upload(const std::vector<hyp::SphericalGhostStencil> &stencils) {
  Plans plans("symmetric ghost plans",stencils.size());
  auto host = Kokkos::create_mirror_view(plans);
  for (size_t p = 0; p < stencils.size(); ++p) host(p) = stencils[p];
  Kokkos::deep_copy(plans,host);
  return plans;
}

void Audit(int n, int degree) {
  const auto g = Grid(n);
  const auto plans = hyp::PlanSymmetricSphericalGhosts(g,3,degree);
  const auto old = hyp::PlanSphericalGhosts(g,3,degree);
  Check(plans.size() == old.size(), "changed stencil target coverage");
  const int cells = g.n[0]*g.n[1]*g.n[2];
  std::map<int,const hyp::SphericalGhostStencil *> lookup;
  std::vector<int> coverage(cells,0);
  double maximum_weight = 0, maximum_symmetry = 0, maximum_polynomial = 0;
  int maximum_support = 0;
  for (size_t p = 0; p < plans.size(); ++p) {
    const auto &s = plans[p];
    Check(s.target == old[p].target && !coverage[s.target], "changed/duplicate target");
    lookup[s.target] = &s;
    coverage[s.target] = 1;
    maximum_support = std::max(maximum_support,s.count);
    double norm = 0;
    for (int l = 0; l < s.count; ++l) {
      const int d = s.donors[l];
      Check(d >= 0 && d < cells && g.Interior(d%g.n[0],d/g.n[0]%g.n[1],
            d/(g.n[0]*g.n[1])), "exterior/recursive donor");
      norm += std::abs(s.weights[l]);
    }
    maximum_weight = std::max(maximum_weight,norm);
    for (int a = 0; a <= degree; ++a)
    for (int b = 0; b <= degree-a; ++b)
    for (int c = 0; c <= degree-a-b; ++c) {
      double value = 0;
      for (int l = 0; l < s.count; ++l) {
        value += s.weights[l]*Polynomial(s.donors[l],g,a,b,c);
      }
      maximum_polynomial = std::max(maximum_polynomial,
          std::abs(value-Polynomial(s.target,g,a,b,c)));
    }
  }
  // These reflections and swaps generate the full 48-element cube group.
  const std::array<int,3> identity = {0,1,2}, xy = {1,0,2}, yz = {0,2,1};
  for (const auto &p : plans) {
    for (const auto &tr : {std::make_pair(identity,1),std::make_pair(identity,2),
                          std::make_pair(identity,4),std::make_pair(xy,0),
                          std::make_pair(yz,0)}) {
      const auto found = lookup.find(Transform(p.target,g,tr.first,tr.second));
      Check(found != lookup.end(), "non-invariant target mask");
      std::map<int,double> difference;
      for (int l = 0; l < p.count; ++l) {
        difference[Transform(p.donors[l],g,tr.first,tr.second)] += p.weights[l];
      }
      const auto &other = *found->second;
      for (int l = 0; l < other.count; ++l) {
        difference[other.donors[l]] -= other.weights[l];
      }
      double mismatch = 0;
      for (const auto &item : difference) mismatch += std::abs(item.second);
      maximum_symmetry = std::max(maximum_symmetry,mismatch);
    }
  }
  for (int k = 3; k < g.n[2]-3; ++k)
  for (int j = 3; j < g.n[1]-3; ++j)
  for (int i = 3; i < g.n[0]-3; ++i) {
    if (!g.Interior(i,j,k)) continue;
    for (int c = -3; c <= 3; ++c)
    for (int b = -3; b <= 3; ++b)
    for (int a = -3; a <= 3; ++a) {
      const int axes = (a != 0)+(b != 0)+(c != 0);
      if (axes > 2 || (axes == 2
          && std::max({std::abs(a),std::abs(b),std::abs(c)}) == 3)) {
        continue;
      }
      if (!g.Interior(i+a,j+b,k+c)) {
        Check(coverage[g.Index(i+a,j+b,k+c)] == 1, "missing mixed/upwind target");
      }
    }
  }
  View q("poisoned exterior scalar",cells);
  auto host = Kokkos::create_mirror_view(q);
  for (int s = 0; s < cells; ++s) {
    const int i = s%g.n[0], j = s/g.n[0]%g.n[1], k = s/(g.n[0]*g.n[1]);
    host(s) = g.Interior(i,j,k) ? Polynomial(s,g,degree-1,1,0)
        : std::numeric_limits<double>::quiet_NaN();
  }
  Kokkos::deep_copy(q,host);
  hyp::FillSphericalGhosts(q,Upload(plans));
  Kokkos::deep_copy(host,q);
  for (const auto &p : plans) {
    Check(std::isfinite(host(p.target)), "fill used poisoned exterior/recursive donor");
    Check(std::abs(host(p.target)-Polynomial(p.target,g,degree-1,1,0)) < 2e-12,
          "filled mixed polynomial mismatch");
  }
  Check(maximum_symmetry < 2e-13, "cube-equivariant donor maps differ");
  Check(maximum_polynomial < 2e-12, "total-degree polynomial reproduction");
  std::cout << "symmetric ghosts N=" << n << " degree=" << degree
            << " support_max=" << maximum_support << " weight_L1=" << maximum_weight
            << " group_generator_L1=" << maximum_symmetry
            << " polynomial_Linf=" << maximum_polynomial << '\n';
}

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
struct DerivativeError { double hessian, upwind; };

DerivativeError Derivatives(int n) {
  const auto g = Grid(n);
  const int nx = g.n[0], ny = g.n[1], cells = nx*ny*g.n[2];
  const double h = g.h[0], origin = g.first[0];
  View q("smooth boundary scalar",cells);
  Kokkos::deep_copy(q,std::numeric_limits<double>::quiet_NaN());
  Kokkos::parallel_for("smooth interior",cells,KOKKOS_LAMBDA(const int s) {
    const double x = origin+(s%nx)*h, y = origin+(s/nx%ny)*h;
    const double z = origin+(s/(nx*ny))*h;
    if (x*x+y*y+z*z < 1) q(s) = Kokkos::exp(0.7*x-0.4*y+0.3*z);
  });
  hyp::FillSphericalGhosts(q,Upload(hyp::PlanSymmetricSphericalGhosts(g,3,4)));
  double hessian = 0, upwind = 0;
  Kokkos::parallel_reduce("symmetric ghost derivative audit",cells,
      KOKKOS_LAMBDA(const int s, double &max_hessian, double &max_upwind) {
    const int i = s%nx, j = s/nx%ny, k = s/(nx*ny);
    const double x = origin+i*h, y = origin+j*h, z = origin+k*h;
    const double r2 = x*x+y*y+z*z;
    if (r2 >= 1 || r2 < (1-3*h)*(1-3*h)) return;
    const Real idx[3] = {1/h,1/h,1/h}, wave[3] = {0.7,-0.4,0.3};
    FlatScalar scalar{q,nx,ny};
    const RadialShift beta{origin,h};
    for (int a = 0; a < 3; ++a) {
      const double derivative = Lx<3>(a,idx,beta,scalar,0,a,k,j,i);
      max_upwind = Kokkos::fmax(max_upwind,
          Kokkos::abs(derivative-beta(0,a,k,j,i)*wave[a]*q(s)));
      for (int b = a; b < 3; ++b) {
        const double second = a == b ? Dxx<3>(a,idx,scalar,0,k,j,i)
            : Dxy<3>(a,b,idx,scalar,0,k,j,i);
        max_hessian = Kokkos::fmax(max_hessian,
            Kokkos::abs(second-wave[a]*wave[b]*q(s)));
      }
    }
  },Kokkos::Max<double>(hessian),Kokkos::Max<double>(upwind));
  Check(std::isfinite(hessian) && std::isfinite(upwind),
        "nonfinite boundary derivatives");
  std::cout << "symmetric derivatives N=" << n << " Hessian_Linf=" << hessian
            << " upwind_Linf=" << upwind << '\n';
  return {hessian,upwind};
}

void Rejections() {
  for (int mode = 0; mode < 3; ++mode) {
    auto g = Grid(24);
    if (mode == 0) g.h[1] *= 1.01;
    if (mode == 1) g.first[0] += 0.1*g.h[0];
    if (mode == 2) ++g.n[2];
    bool rejected = false;
    try {
      hyp::PlanSymmetricSphericalGhosts(g,3,3);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    Check(rejected, "nonsymmetric grid accepted");
  }
}

int main(int argc, char **argv) {
  Kokkos::ScopeGuard guard(argc,argv);
  try {
    Rejections();
    for (int n : {24,36,48}) for (int degree : {2,3,4}) Audit(n,degree);
    Audit(25,3);  // Exercise stabilizers containing reflection across zero.
    const auto coarse = Derivatives(24), fine = Derivatives(48);
    Check(coarse.hessian/fine.hessian > 4,
          "symmetric boundary Hessian does not converge");
    Check(coarse.upwind/fine.upwind > 4, "symmetric boundary upwind does not converge");
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
