// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include "z4c/hyperboloidal/interior_dissipation.hpp"

namespace hyp = z4c::hyperboloidal;
void Audit(bool polynomial) {
  constexpr int n = 24, cells = n*n*n;
  constexpr double h = 0.125, first = -1-3.5*h;
  Kokkos::View<double *> q("field",cells), result("dissipation",cells);
  Kokkos::View<unsigned char *> mask("active sphere",cells);
  auto host = Kokkos::create_mirror_view(q);
  auto mh = Kokkos::create_mirror_view(mask);
  for (int s = 0; s < cells; ++s) {
    const double x = first+(s%n)*h, y = first+(s/n%n)*h, z = first+(s/(n*n))*h;
    mh(s) = x*x+y*y+z*z < 1;
    if (s == n/2+n*(n/2+n*(n/2))) mh(s) = 0;  // an interior hole
    host(s) = !mh(s) ? std::numeric_limits<double>::quiet_NaN() :
        (polynomial ? 1+0.3*x-0.4*y*z+0.2*x*x : std::sin(1.37*s)+0.3*std::cos(0.19*s));
  }
  Kokkos::deep_copy(q,host);
  Kokkos::deep_copy(mask,mh);
  Kokkos::deep_copy(result,-999.);
  Kokkos::parallel_for("masked negative dissipation",cells,KOKKOS_LAMBDA(const int s) {
    if (!mask(s)) return;
    const int stride[3] = {1,n,n*n};
    const double spacing[3] = {h,h,h};
    result(s) = hyp::InteriorKOSixth(q,mask,s,stride,spacing);
  });
  const auto rh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),result);
  double dot = 0, sum = 0, squares = 0, max_value = 0;
  const int stride[3] = {1,n,n*n};
  const double c[4] = {-1,3,-3,1};
  for (int s = 0; s < cells; ++s) {
    if (!mh(s)) {
      if (rh(s) != -999.) throw std::runtime_error("inactive dissipation write");
      continue;
    }
    if (!std::isfinite(rh(s))) throw std::runtime_error("read inactive poison");
    dot += host(s)*rh(s); sum += rh(s);
    max_value = std::fmax(max_value,std::abs(rh(s)));
    // Independent row-wise D3 norm, versus the column-wise implementation.
    for (int d = 0; d < 3; ++d) {
      bool valid = true;
      for (int b = 0; b < 4; ++b) {
        if (!mh(s+b*stride[d])) valid = false;
      }
      if (!valid) continue;
      double third = 0;
      for (int b = 0; b < 4; ++b) third += c[b]*host(s+b*stride[d]);
      squares += third*third/(64*h);
    }
  }
  if (std::abs(dot+squares) > 1e-11*(1+squares) || std::abs(sum) > 1e-9
      || (polynomial && max_value > 1e-12) || (!polynomial && dot >= -1)) {
    throw std::runtime_error("masked KO energy/polynomial identity");
  }
  std::cout << "polynomial=" << polynomial << " qQq=" << dot
            << " negative_D3_norm=" << -squares << " sumQ=" << sum << '\n';
}
int main(int argc, char **argv) {
  Kokkos::ScopeGuard guard(argc,argv);
  try {
    Audit(false);
    Audit(true);
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
