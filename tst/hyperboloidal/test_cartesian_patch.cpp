// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include "z4c/hyperboloidal/cartesian_patch.hpp"

namespace hyp = z4c::hyperboloidal;
using z4c::Z4c;

// The continuum Hamiltonian constraint must be tangent to zero under a
// lapse-only perturbation of flat CMC data. This checks the full tensor RHS
// independently of finite-difference convergence or a stationary fixed point.
void AuditGaugeConstraintTangent() {
  const hyp::CMCReference<Real> ref{1,1};
  for (double x : {0.1,0.4,0.8}) {
    const auto p = ref.At(x,0.07,-0.11);
    hyp::Z4cJet<Real> u{};
    hyp::AddReferenceJet(u,p,ref);
    const double xyz[3] = {x,0.07,-0.11};
    const double f = 1e-4*std::exp(-(x*x+0.07*0.07+0.11*0.11)/0.09);
    double lapf = 0, dot = 0, grad2 = 0;
    u.alpha.value += f;
    for (int a = 0; a < 3; ++a) {
      const double df = -2*xyz[a]*f/0.09;
      u.alpha.d[a] += df;
      dot += df*p.domega[a];
      grad2 += p.domega[a]*p.domega[a];
      for (int b = 0; b < 3; ++b) {
        const double ddf = (4*xyz[a]*xyz[b]/(0.09*0.09)
            -(a == b ? 2/0.09 : 0))*f;
        u.alpha.dd[a][b] += ddf;
        if (a == b) lapf += ddf;
      }
    }
    hyp::Z4cRHS<Real> rhs{};
    if (!hyp::AssembleInterior(hyp::ConformalRHS(u,hyp::CartesianOmega(u,p),
        5/u.alpha.value,Real(0)),p.omega,rhs)) {
      throw std::runtime_error("invalid analytic lapse RHS");
    }
    const double o = p.omega, lapo = 3*p.hessian_omega;
    const double chi_dot = -2*f/o;
    const double dot_chi_omega = -2*(dot/o-f*grad2/(o*o));
    const double lap_chi = -2*(lapf/o-2*dot/(o*o)-f*lapo/(o*o)
        +2*f*grad2/(o*o*o));
    const double h_dot = 2*o*o*lap_chi-4*(rhs.trace+2*rhs.theta)
        +4*o*chi_dot*lapo-2*o*dot_chi_omega-6*chi_dot*grad2;
    std::cout << "analytic lapse x=" << x << " H_dot=" << h_dot
              << " chi_dot_error=" << rhs.chi-chi_dot << '\n';
    if (std::abs(h_dot) > 1e-11 || std::abs(rhs.chi-chi_dot) > 1e-11) {
      throw std::runtime_error("continuum lapse constraint tangent failed");
    }
  }
}

void AuditInterfaces(hyp::CartesianConformalPatch &patch) {
  auto q = patch.Allocate("interface audit"), rhs = patch.Allocate("audit RHS");
  patch.InitializeReference(q);
  bool rejected = false;
  try {
    patch.RHS(q,q);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  if (!rejected) throw std::runtime_error("accepted aliased RHS");
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),q);
  const auto g = patch.grid;
  for (int k = 0; k < g.n[2]; ++k)
  for (int j = 0; j < g.n[1]; ++j)
  for (int i = 0; i < g.n[0]; ++i) {
    if (g.Interior(i,j,k)) continue;
    for (int f = 0; f < Z4c::nz4c; ++f) {
      host(0,f,k,j,i) = std::numeric_limits<double>::quiet_NaN();
    }
  }
  Kokkos::deep_copy(q,host);
  patch.RHS(q,rhs);
  const auto rh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),rhs);
  double maximum = 0;
  for (size_t s = 0; s < rh.size(); ++s) {
    if (!std::isfinite(rh.data()[s])) throw std::runtime_error("inactive poison read");
    maximum = std::max(maximum,std::abs(rh.data()[s]));
  }
  if (maximum > 1e-10) throw std::runtime_error("reference RHS not zero");
  const auto diagnostics = patch.Diagnose(q);
  if (diagnostics.h_l2 > 1e-12 || diagnostics.m_l2 > 1e-12) {
    throw std::runtime_error("poisoned reference constraints");
  }
  // A finite, positive determinant alone would admit two negative eigenvalues.
  patch.InitializeReference(q);
  host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),q);
  const int i = g.n[0]/2, j = g.n[1]/2, k = g.n[2]/2;
  host(0,Z4c::I_Z4C_GXX,k,j,i) = -1;
  host(0,Z4c::I_Z4C_GYY,k,j,i) = -1;
  Kokkos::deep_copy(q,host);
  rejected = false;
  try {
    patch.RHS(q,rhs);
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  if (!rejected) throw std::runtime_error("accepted indefinite spatial metric");
}

hyp::CartesianDiagnostics Run(int n, double end, double amplitude,
                              bool smooth = false) {
  hyp::SphericalGhostGrid grid;
  grid.radius = 1;
  for (int d = 0; d < 3; ++d) {
    grid.n[d] = n+6;
    grid.h[d] = 2.1/n;
    grid.first[d] = -1.05-2.5*grid.h[d];
  }
  hyp::CartesianConformalPatch patch(grid);
  if (amplitude == 0) AuditInterfaces(patch);
  auto q = patch.Allocate("state"), initial = patch.Allocate("RK initial");
  auto stage = patch.Allocate("RK stage"), rhs = patch.Allocate("RHS");
  patch.InitializeReference(q);
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),q);
  for (int k = 0; k < grid.n[2]; ++k)
  for (int j = 0; j < grid.n[1]; ++j)
  for (int i = 0; i < grid.n[0]; ++i) {
    if (!grid.Interior(i,j,k)) continue;
    const double x = grid.first[0]+i*grid.h[0];
    const double y = grid.first[1]+j*grid.h[1];
    const double z = grid.first[2]+k*grid.h[2];
    const double r2 = x*x+y*y+z*z;
    // Smooth compact, nonspherical gauge pulse. Spatial initial data remain
    // exactly Minkowski; this lapse change introduces no initial constraints.
    const double bump = smooth ? std::pow(1-r2,4)*std::exp(-r2/0.25)
        : (r2 < 0.36 ? std::exp(1-1/(1-r2/0.36)) : 0);
    host(0,Z4c::I_Z4C_ALPHA,k,j,i) += amplitude*bump*(1+0.2*x+0.3*y*z);
  }
  Kokkos::deep_copy(q,host);
  const auto first = patch.Diagnose(q);
  if (first.h_l2 > 1e-10 || first.m_l2 > 1e-10 || first.z_l2 > 1e-10) {
    throw std::runtime_error("nonzero initial Minkowski constraints");
  }
  const double dt_limit = std::min(0.025*grid.h[0],0.04*patch.min_omega);
  double t = 0;
  int steps = 0;
  double next_report = end/10;
  while (t < end) {
    const double dt = std::min(dt_limit,end-t);
    Kokkos::deep_copy(initial,q);
    // SSPRK3, every RHS reconstructs the spherical ghost fringe.
    for (int rk = 0; rk < 3; ++rk) {
      patch.RHS(q,rhs);
      const double a = rk == 0 ? 0 : (rk == 1 ? 0.75 : 1.0/3);
      Kokkos::parallel_for("conformal RK stage",q.size(),KOKKOS_LAMBDA(const int s) {
        stage.data()[s] = a*initial.data()[s]+(1-a)*(q.data()[s]+dt*rhs.data()[s]);
      });
      Kokkos::deep_copy(q,stage);
    }
    t += dt;
    ++steps;
    if (end > 0.1 && t >= next_report) {
      const auto d = patch.Diagnose(q);
      std::cout << "progress n=" << n << " t=" << t << " H=" << d.h_l2
                << " M=" << d.m_l2 << " deviation=" << d.max_deviation
                << " shell_pole_delta=" << d.shell_max_pole_deviation << std::endl;
      next_report += end/10;
    }
  }
  const auto last = patch.Diagnose(q);
  std::cout << std::setprecision(12) << "n=" << n << " t=" << t
            << " steps=" << steps << " min_omega=" << patch.min_omega
            << " amplitude=" << amplitude << " smooth=" << smooth
            << " H=" << last.h_l2 << " M=" << last.m_l2
            << " Z=" << last.z_l2 << " Theta=" << last.theta_l2
            << " det=" << last.max_det << " trace=" << last.max_trace
            << " min_alpha=" << last.min_alpha << " min_chi=" << last.min_chi
            << " deviation=" << last.max_deviation
            << " shell_pole=" << last.shell_max_pole
            << " shell_pole_delta=" << last.shell_max_pole_deviation
            << " shell_null_deviation=" << last.shell_max_null_deviation << std::endl;
  if (!(last.min_alpha > 0 && last.min_chi > 0)
      || last.max_deviation > std::max(1e-10,20*std::abs(amplitude))
      || (amplitude == 0 && last.max_deviation > 1e-10)) {
    throw std::runtime_error("Cartesian evolution audit failed");
  }
  return last;
}

int main(int argc, char **argv) {
  Kokkos::ScopeGuard guard(argc,argv);
  try {
    AuditGaugeConstraintTangent();
    if (argc > 1) {
      if (argc != 4 && argc != 5) {
        throw std::invalid_argument("usage: N end amplitude [compact|smooth]");
      }
      const int n = std::atoi(argv[1]);
      const double end = std::atof(argv[2]), amplitude = std::atof(argv[3]);
      if (n < 24 || n%2 || !std::isfinite(end) || end <= 0
          || !std::isfinite(amplitude)) throw std::invalid_argument("invalid parameters");
      const std::string profile = argc == 5 ? argv[4] : "compact";
      if (profile != "compact" && profile != "smooth") {
        throw std::invalid_argument("unknown pulse profile");
      }
      Run(n,end,amplitude,profile == "smooth");
    } else {
      Run(24,0.01,0);
      Run(24,0.01,1e-4);
      const auto coarse = Run(24,0.01,1e-4,true);
      const auto fine = Run(36,0.01,1e-4,true);
      if (coarse.h_l2 < 4*fine.h_l2 || coarse.m_l2 < 3*fine.m_l2
          || coarse.z_l2 < 2*fine.z_l2) {
        throw std::runtime_error("Cartesian pulse constraints fail to converge");
      }
    }
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
