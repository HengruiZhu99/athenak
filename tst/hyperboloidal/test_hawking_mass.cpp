// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include "z4c/hyperboloidal/cartesian_hawking.hpp"
#include "z4c/hyperboloidal/cartesian_trumpet.hpp"

namespace hyp = z4c::hyperboloidal;

void ExactSpheres() {
  const hyp::CMCReference<double> ref{1,1};
  for (double mass : {0.0,0.05,0.5}) {
    const hyp::CMCTrumpet trumpet(mass == 0 ? 0.5 : mass);
    for (double radius : {0.2,0.4,0.7,0.9}) {
      const auto radial = trumpet.Jet(radius);
      const auto result = hyp::IntegrateHawkingMass(radius,16,[&](const double x[3]) {
        const auto p = ref.At(x[0],x[1],x[2]);
        hyp::Z4cJet<double> u{};
        if (mass == 0) {
          hyp::AddReferenceJet(u,p,ref);
        } else {
          const double n[3] = {x[0]/radius,x[1]/radius,x[2]/radius};
          u = hyp::CartesianRadialJet(radial,radius,n);
        }
        return hyp::CoordinateSphereDensity(u,hyp::CartesianOmega(u,p),x);
      });
      std::cout << "exact mass=" << mass << " r=" << radius
                << " measured=" << result.mass << '\n';
      if (std::abs(result.mass-mass) > 2e-11) {
        throw std::runtime_error("exact Hawking mass mismatch");
      }
    }
  }
  // Nonzero physical Theta must be included in K=P+2*Theta.
  const double radius = 0.7, k = 0.2;
  for (bool trace_in_a : {false,true}) {
    const auto result = hyp::IntegrateHawkingMass(radius,12,[&](const double x[3]) {
      hyp::Z4cJet<double> u{};
      u.chi.value = 1; u.theta.value = 0.07;
      u.trace.value = (trace_in_a ? 0 : 3*k)-2*u.theta.value;
      for (int i = 0; i < 3; ++i) {
        u.metric.g[i][i] = 1;
        if (trace_in_a) u.a.k[i][i] = k;
      }
      hyp::OmegaJet<double> o{}; o.omega = 1;
      return hyp::CoordinateSphereDensity(u,o,x);
    });
    if (std::abs(result.mass-k*k*std::pow(radius,3)/2) > 1e-13) {
      throw std::runtime_error("Hawking physical trace normalization");
    }
  }
}

void ShearedFlatOracle() {
  // Physical Euclidean coordinates X=L*x, with non-diagonal constant metric.
  const double l[3][3] = {{1.3,0.2,0.1},{0,0.9,-0.1},{0,0,1.1}};
  const double inv[3][3] = {{1/1.3,-0.2/(1.3*0.9),-0.11/(1.3*0.9*1.1)},
                           {0,1/0.9,0.1/(0.9*1.1)},{0,0,1/1.1}};
  hyp::Z4cJet<double> u{}; u.chi.value = 1;
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j)
  for (int k = 0; k < 3; ++k) u.metric.g[i][j] += l[k][i]*l[k][j];
  hyp::OmegaJet<double> o{}; o.omega = 1;
  const double x[3] = {0.31,-0.27,0.42};
  const auto density = hyp::CoordinateSphereDensity(u,o,x);
  const auto normal = [&](const double y[3], int component) {
    double v[3]{}, norm = 0;
    for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) v[i] += inv[j][i]*y[j];
    for (double a : v) norm += a*a;
    return v[component]/std::sqrt(norm);
  };
  double previous = 0;
  for (double h : {0.002,0.001,0.0005}) {
    double divergence = 0;
    for (int d = 0; d < 3; ++d) {
      double plus[3], minus[3];
      for (int j = 0; j < 3; ++j) {
        plus[j] = x[j]+h*inv[j][d]; minus[j] = x[j]-h*inv[j][d];
      }
      divergence += (normal(plus,d)-normal(minus,d))/(2*h);
    }
    const double error = std::abs(density.correction-4
                                  +divergence*divergence*density.area);
    std::cout << "sheared h=" << h << " error=" << error << '\n';
    if (previous && previous/error < 3.8) {
      throw std::runtime_error("sheared surface curvature oracle convergence");
    }
    previous = error;
  }
  if (!density.valid || previous > 1e-5) {
    throw std::runtime_error("sheared surface curvature oracle");
  }
}

void CurvilinearFlatSpheres() {
  // Euclidean space in nonlinear coordinates Y=(1+a*r^2)*x. Coordinate
  // spheres remain round, but the metric and its off-diagonal derivatives vary.
  for (double radius : {0.3,0.7}) {
    const double a = 0.3;
    const auto result = hyp::IntegrateHawkingMass(radius,16,[&](const double x[3]) {
      double q = 0;
      for (int i = 0; i < 3; ++i) q += x[i]*x[i];
      const double f = 1+a*q, d = 4*a*(1+2*a*q);
      hyp::Z4cJet<double> u{}; u.chi.value = 1;
      for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        u.metric.g[i][j] = (i == j ? f*f : 0)+d*x[i]*x[j];
        for (int k = 0; k < 3; ++k) {
          u.metric.dg[k][i][j] = (i == j ? 4*a*f*x[k] : 0)
              +16*a*a*x[k]*x[i]*x[j]
              +d*((i == k ? x[j] : 0)+(j == k ? x[i] : 0));
        }
      }
      // The surface integral only needs first spatial derivatives.
      hyp::OmegaJet<double> o{}; o.omega = 1;
      return hyp::CoordinateSphereDensity(u,o,x);
    });
    if (std::abs(result.mass) > 1e-12
        || std::abs(result.areal_radius-radius*(1+a*radius*radius)) > 1e-12) {
      throw std::runtime_error("curvilinear flat Hawking sphere");
    }
  }
}

void GridRefinement() {
  double previous = 0;
  for (int n : {24,36,48}) {
    hyp::SphericalGhostGrid grid;
    grid.radius = 1;
    for (int d = 0; d < 3; ++d) {
      grid.n[d] = n+6; grid.h[d] = 2.1/n;
      grid.first[d] = -1.05-2.5*grid.h[d];
    }
    hyp::CartesianConformalPatch patch(grid);
    auto q = patch.Allocate("mass oracle");
    patch.InitializeReference(q);
    for (const auto &a : hyp::CartesianHawkingMasses(patch,q,{0.3,0.5,0.7})) {
      if (std::abs(a.mass) > 1e-11) throw std::runtime_error("grid CMC mass nonzero");
    }
    hyp::InitializeCartesianTrumpet(patch,q,0.5);
    double maximum = 0;
    const auto fine_quad = hyp::CartesianHawkingMasses(patch,q,{0.3,0.5,0.7},64);
    size_t index = 0;
    for (const auto &a : hyp::CartesianHawkingMasses(patch,q,{0.3,0.5,0.7})) {
      std::cout << "grid n=" << n << " r=" << a.coordinate_radius
                << " mass=" << a.mass << " areal=" << a.areal_radius << '\n';
      const double quadrature_error = std::abs(a.mass-fine_quad[index++].mass);
      std::cout << "quadrature difference=" << quadrature_error << '\n';
      if (quadrature_error > 0.15*std::abs(a.mass-0.5)+1e-11) {
        throw std::runtime_error("quadrature error exceeds interpolation budget");
      }
      maximum = std::max(maximum,std::abs(a.mass-0.5));
    }
    if (previous && previous/maximum < 2) {
      throw std::runtime_error("grid mass interpolation does not converge");
    }
    previous = maximum;
    bool rejected = false;
    try {
      hyp::CartesianHawkingMasses(patch,q,{0.99});
    } catch (const std::invalid_argument &) { rejected = true; }
    if (!rejected) throw std::runtime_error("mass accepted exterior donors");
  }
}

int main(int argc, char **argv) {
  Kokkos::ScopeGuard guard(argc,argv);
  try {
    std::cout << std::setprecision(14);
    ExactSpheres(); ShearedFlatOracle(); CurvilinearFlatSpheres(); GridRefinement();
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n'; return 1;
  }
}
