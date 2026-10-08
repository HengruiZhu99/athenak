// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include "z4c/hyperboloidal/cartesian_radial.hpp"
#include "z4c/hyperboloidal/cmc_trumpet.hpp"

namespace hyp = z4c::hyperboloidal;
hyp::Z4cJet<double> Sample(const double x[3], bool trumpet) {
  const double r = std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
  const double n[3] = {x[0]/r,x[1]/r,x[2]/r};
  if (trumpet) return hyp::CartesianRadialJet(hyp::CMCTrumpet(0.05).Jet(r),r,n);
  double v[hyp::NFIELDS]{}, d[hyp::NFIELDS]{}, dd[hyp::NFIELDS]{};
  for (int f = 0; f < hyp::NFIELDS; ++f) {
    const int power = f == hyp::LAMBDA || f == hyp::DBETA ? 3 : 2;
    const double a = 0.01*(f+1);
    v[f] = a*std::pow(r,power);
    d[f] = power*a*std::pow(r,power-1);
    dd[f] = power*(power-1)*a*std::pow(r,power-2);
  }
  const auto axis = hyp::SphericalJet(r,v,d,dd,hyp::CMCReference<double>{1,1});
  return hyp::CartesianRadialJet(axis,r,n);
}

// Independently finite-difference component values at displaced Cartesian
// points, including mixed corners. No derivative formula enters the oracle.
void Derivatives(bool trumpet) {
  const double x[3] = {0.21,-0.17,0.26};
  const auto u = Sample(x,trumpet);
  double previous = 0;
  for (double h : {0.02,0.01,0.005}) {
    double error = 0;
    for (int d = 0; d < 3; ++d) {
      double xp[3] = {x[0],x[1],x[2]}, xm[3] = {x[0],x[1],x[2]};
      xp[d] += h; xm[d] -= h;
      const auto up = Sample(xp,trumpet), um = Sample(xm,trumpet);
      const hyp::ScalarJet<double> a[4] = {u.chi,u.alpha,u.trace,u.theta};
      const hyp::ScalarJet<double> p[4] = {up.chi,up.alpha,up.trace,up.theta};
      const hyp::ScalarJet<double> m[4] = {um.chi,um.alpha,um.trace,um.theta};
      for (int f = 0; f < 4; ++f) {
        error = std::max(error,std::abs((p[f].value-m[f].value)/(2*h)-a[f].d[d]));
        error = std::max(error,std::abs((p[f].value-2*a[f].value+m[f].value)/(h*h)
                                       -a[f].dd[d][d]));
      }
      for (int i = 0; i < 3; ++i) {
        error = std::max(error,std::abs((up.beta.value[i]-um.beta.value[i])/(2*h)
                                       -u.beta.d[d][i]));
        for (int j = 0; j < 3; ++j) {
          error = std::max(error,std::abs((up.metric.g[i][j]-um.metric.g[i][j])/(2*h)
                                         -u.metric.dg[d][i][j]));
          error = std::max(error,std::abs((up.a.k[i][j]-um.a.k[i][j])/(2*h)
                                         -u.a.dk[d][i][j]));
        }
      }
      for (int e = d+1; e < 3; ++e) {
        double pp[3] = {x[0],x[1],x[2]}, pm[3] = {x[0],x[1],x[2]};
        double mp[3] = {x[0],x[1],x[2]}, mm[3] = {x[0],x[1],x[2]};
        pp[d] += h; pp[e] += h; pm[d] += h; pm[e] -= h;
        mp[d] -= h; mp[e] += h; mm[d] -= h; mm[e] -= h;
        const auto a = Sample(pp,trumpet), b = Sample(pm,trumpet);
        const auto c = Sample(mp,trumpet), f = Sample(mm,trumpet);
        error = std::max(error,std::abs((a.chi.value-b.chi.value-c.chi.value+f.chi.value)
                                       /(4*h*h)-u.chi.dd[d][e]));
        for (int i = 0; i < 3; ++i) {
          error = std::max(error,std::abs((a.beta.value[i]-b.beta.value[i]
              -c.beta.value[i]+f.beta.value[i])/(4*h*h)-u.beta.dd[d][e][i]));
          for (int j = 0; j < 3; ++j) {
            error = std::max(error,std::abs((a.metric.g[i][j]-b.metric.g[i][j]
                -c.metric.g[i][j]+f.metric.g[i][j])/(4*h*h)-u.metric.ddg[d][e][i][j]));
          }
        }
      }
    }
    std::cout << "trumpet=" << trumpet << " h=" << h << " derivative_error=" << error
              << '\n';
    if (!std::isfinite(error) || (previous > 0 && previous/error < 3.7)) {
      throw std::runtime_error("Cartesian radial derivatives fail convergence");
    }
    previous = error;
  }
}

void Constraints() {
  const double directions[3][3] = {{0.3,0.4,std::sqrt(0.75)},
                                    {-1,0,0},{0,-0.8,0.6}};
  double max_h = 0, max_m = 0;
  for (double mass : {0.05,0.5})
  for (double r : {0.02,0.15,0.6,0.95}) {
    const auto axis = hyp::CMCTrumpet(mass).Jet(r);
    for (const auto &n : directions) {
      const auto u = hyp::CartesianRadialJet(axis,r,n);
      hyp::OmegaJet<double> omega{};
      omega.omega = (1-r*r)/2;
      for (int a = 0; a < 3; ++a) {
        omega.gradient[a] = -r*n[a];
        omega.hessian[a][a] = -1;
      }
      hyp::SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,omega);
      const auto c = hyp::EvolvedConstraints(u,omega);
      if (!c.valid || !c.z4.valid || u.alpha.value <= 0 || u.chi.value <= 0) {
        throw std::runtime_error("invalid off-axis trumpet");
      }
      max_h = std::max(max_h,std::abs(c.hamiltonian));
      max_m = std::max(max_m,std::sqrt(c.momentum_conformal_norm2));
    }
  }
  std::cout << "off-axis trumpet max_H=" << max_h << " max_M=" << max_m << '\n';
  if (max_h > 1e-8 || max_m > 1e-8) {
    throw std::runtime_error("Cartesian trumpet constraints");
  }
}

int main(int argc, char **argv) {
  Kokkos::ScopeGuard guard(argc,argv);
  try {
    Derivatives(false);
    Derivatives(true);
    Constraints();
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
