// Physical-geometry reduction and genuine FastFlow tests on hyperboloidal slices.
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "globals.hpp"
#include "parameter_input.hpp"
#include "z4c/fastflow.hpp"
#include "z4c/hyperboloidal/horizon_geometry.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace hh = z4c::hyperboloidal;

void Require(bool ok, const std::string &message) {
  if (!ok) throw std::runtime_error(message);
}

void Near(double actual, double expected, double tol, const std::string &message) {
  if (!(std::isfinite(actual) && std::abs(actual-expected) <= tol*(1+std::abs(expected)))) {
    std::ostringstream out;
    out << message << ": " << actual << " expected " << expected;
    throw std::runtime_error(out.str());
  }
}

// Exact CMC Schwarzschild slice, R=r/(1-r), K=-3. The integration constant C
// selects the future black-hole branch when C>(2M)^3. The data obey
// gamma_RR=1/F, F=1-2M/R+(-R+C/R^2)^2,
// K^R_R=-1-2C/R^3, K^theta_theta=-1+C/R^3.
// The outer horizon has R=2M, physical area 16 pi M^2 on every such slice.
// M=C=0 is a future Minkowski hyperboloid with no apparent horizon.
struct CMCData {
  double mass, c;
  double origin[3]{};
  hh::Z4cJet<Real> jet{};
  hh::OmegaJet<Real> omega{};
  z4c::HorizonGeometry<Real> exact{};

  void Evaluate(const Real x[3]) {
    double n[3], r = 0;
    for (int i = 0; i < 3; ++i) { n[i] = x[i]-origin[i]; r += n[i]*n[i]; }
    r = std::sqrt(r);
    jet = {}; omega = {}; exact = {};
    if (!(r > 0 && r < 1)) return;
    for (double &ni : n) ni /= r;
    const double o = 1-r, R = r/o, Rp = 1/(o*o);
    const double w = -R+c/(R*R);
    const double f = 1-2*mass/R+w*w;
    if (!(f > 0)) return;
    const double fp = 2*mass/(R*R)+2*w*(-1-2*c/(R*R*R));
    const double gr = Rp*Rp/f, gt = 1/(o*o);
    const double dgr = gr*(4/o-fp*Rp/f), dgt = 2/(o*o*o);
    const double chi = std::pow(gr*gt*gt, -1./3.)/(o*o);
    const double dlogchi = -(dgr/gr+2*dgt/gt)/3+2/o;
    const double scale = chi*o*o;
    const double tr = scale*gr, tt = scale*gt;
    const double dtr = scale*(dgr+gr*(dlogchi-2/o));
    const double dtt = scale*(dgt+gt*(dlogchi-2/o));
    const double kr = -1-2*c/(R*R*R), kt = -1+c/(R*R*R);
    const double ar = chi*o*gr*(kr+1), at = chi*o*gt*(kt+1);
    jet.chi.value = chi;
    jet.theta.value = 0.037;  // exercise P+2Theta, independently of CMC K=-3
    jet.trace.value = -3-2*jet.theta.value;
    omega.omega = o;
    for (int d = 0; d < 3; ++d) {
      jet.chi.d[d] = chi*dlogchi*n[d];
      omega.gradient[d] = -n[d];
    }
    for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      const double delta = i == j;
      jet.metric.g[i][j] = tt*delta+(tr-tt)*n[i]*n[j];
      jet.a.k[i][j] = at*delta+(ar-at)*n[i]*n[j];
      exact.metric[i][j] = gt*delta+(gr-gt)*n[i]*n[j];
      exact.curvature[i][j] = gt*kt*delta+(gr*kr-gt*kt)*n[i]*n[j];
      for (int d = 0; d < 3; ++d) {
        const double dn = ((d == i)*n[j]+(d == j)*n[i]-2*n[d]*n[i]*n[j])/r;
        jet.metric.dg[d][i][j] = dtt*n[d]*delta+(dtr-dtt)*n[d]*n[i]*n[j]
            +(tr-tt)*dn;
        exact.derivative[d][i][j] = dgt*n[d]*delta+(dgr-dgt)*n[d]*n[i]*n[j]
            +(gr-gt)*dn;
      }
    }
    exact.valid = true;
  }
};

void TestReconstruction() {
  CMCData data{1, 16};
  const Real x[3] = {0.36, -0.48, 0.30};
  data.Evaluate(x);
  const auto q = hh::PhysicalHorizonGeometry(data.jet, data.omega);
  Require(q.valid, "CMC physical geometry valid");
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) {
    Near(q.metric[i][j], data.exact.metric[i][j], 3e-14, "physical metric");
    Near(q.curvature[i][j], data.exact.curvature[i][j], 3e-14, "physical K");
    for (int d = 0; d < 3; ++d) {
      Near(q.derivative[d][i][j], data.exact.derivative[d][i][j], 3e-13,
           "physical metric derivative, including Omega gradient");
    }
  }
  // Omega=1 reduces exactly to the ordinary Z4c physical ADM conversion.
  data.omega = {}; data.omega.omega = 1;
  const auto flat_omega = hh::PhysicalHorizonGeometry(data.jet, data.omega);
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) {
    const auto &u = data.jet;
    Near(flat_omega.metric[i][j], u.metric.g[i][j]/u.chi.value, 2e-14, "Omega=1 metric");
    Near(flat_omega.curvature[i][j], (u.a.k[i][j]-u.metric.g[i][j])/u.chi.value,
         2e-14, "Omega=1 extrinsic curvature");
  }
  data.omega.omega = 0;
  Require(!hh::PhysicalHorizonGeometry(data.jet, data.omega).valid, "scri rejected");
  data.omega.omega = 1; data.jet.chi.value = 0;
  Require(!hh::PhysicalHorizonGeometry(data.jet, data.omega).valid, "chi=0 rejected");
  data.jet.chi.value = 1; data.jet.metric.g[0][0] = -1;
  Require(!hh::PhysicalHorizonGeometry(data.jet, data.omega).valid, "nonpositive metric rejected");
}

void Input(ParameterInput &pin, const std::string &name, double radius) {
  std::ostringstream text;
  text << "<job>\nbasename = " << name << "\n<fastflow>\n"
       << "ntheta = 12\nlmax = 6\nflow = standard\n"
       << "flow_iterations_0 = 800\nflow_alpha_beta_const_0 = 0.5\n"
       << "start_time_0 = 0\nstop_time_0 = 1\n"
       << "mass_tol_0 = 1e-10\nexpansion_rms_tol_0 = 1e-9\n"
       << "hmean_tol_0 = 1e8\ninitial_radius_0 = " << radius << "\n";
  std::istringstream stream(text.str());
  pin.LoadFromStream(stream);
}

bool Own(const Real x[3]) {
  #if MPI_PARALLEL_ENABLED
    // Test actual distributed angular ownership, including empty ranks.
    const int owner = x[0] < 0 ? 0 : global_variable::nranks-1;
    return global_variable::my_rank == owner;
  #else
    (void)x;
    return true;
  #endif
}

FastFlow::Result Solve(const std::string &name, bool converted, bool shifted) {
  ParameterInput pin;
  Input(pin, name, 0.70);
  CMCData data{1, 16};
  if (shifted) {
    data.origin[0] = 0.03; data.origin[1] = -0.02; data.origin[2] = 0.01;
  }
  FastFlow finder(nullptr, &pin, 0, [&](const Real x[3], z4c::HorizonGeometry<Real> &q) {
    if (!Own(x)) return false;
    data.Evaluate(x);
    q = converted ? hh::PhysicalHorizonGeometry(data.jet, data.omega) : data.exact;
    return true;
  });
  // A displaced black hole relative to the search center exercises nonzero
  // angular modes, rather than solving only the radial l=0 equation.
  finder.Find(0, 0);
  const auto result = finder.GetResult();
  if (global_variable::my_rank == 0) {
    std::cout << name << " found=" << result.found << " iterations=" << result.iterations
              << " R=" << result.areal_radius << " mean_r=" << result.mean_radius
              << " H_rms=" << result.expansion_rms << '\n';
  }
  Require(result.found && result.complete_surface, name+" found complete horizon");
  Near(result.areal_radius, 2, 2e-8, name+" horizon areal radius");
  Near(result.mass, 1, 2e-8, name+" horizon mass");
  Require(result.expansion_rms < 1e-9, name+" outgoing expansion residual");
  return result;
}

void TestNoFalseHorizon() {
  ParameterInput pin;
  Input(pin, "fastflow_minkowski", 0.5);
  pin.SetReal("fastflow", "mass_tol_0", 1e20);
  pin.SetReal("fastflow", "flow_alpha_beta_const_0", 1e-8);
  pin.SetInteger("fastflow", "flow_iterations_0", 4);
  CMCData data{0, 0};
  FastFlow finder(nullptr, &pin, 0, [&](const Real x[3], z4c::HorizonGeometry<Real> &q) {
    if (!Own(x)) return false;
    data.Evaluate(x);
    q = hh::PhysicalHorizonGeometry(data.jet, data.omega);
    return true;
  });
  finder.Find(0, 0);
  Require(!finder.ah_found && finder.GetResult().complete_surface,
          "Minkowski hyperboloid cannot pass mass-only false convergence");
}

void TestIsotropicWormhole() {
  ParameterInput pin;
  Input(pin, "fastflow_isotropic_wormhole", 0.18);
  constexpr double mass = 1./3.;
  FastFlow finder(nullptr, &pin, 0, [&](const Real x[3], z4c::HorizonGeometry<Real> &q) {
    if (!Own(x)) return false;
    const double r = std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
    hh::Z4cJet<Real> u{};
    hh::OmegaJet<Real> o{};
    o.omega = (1-r*r)/2;
    if (!(r > 0 && o.omega > 0)) return true;
    const double psi = 1+mass/(2*r), dpsi = -mass/(2*r*r);
    u.chi.value = 1/(o.omega*o.omega*std::pow(psi, 4));
    u.theta.value = 0.037;
    u.trace.value = -2*u.theta.value;
    for (int i = 0; i < 3; ++i) {
      u.metric.g[i][i] = 1;
      o.gradient[i] = -x[i];
      u.chi.d[i] = u.chi.value*(-2*o.gradient[i]/o.omega-4*dpsi/psi*x[i]/r);
    }
    q = hh::PhysicalHorizonGeometry(u, o);
    return true;
  });
  finder.Find(0, 0);
  auto result = finder.GetResult();
  Require(result.found, "isotropic wormhole horizon found");
  Near(result.mean_radius, mass/2, 3e-10, "isotropic wormhole coordinate horizon");
  Near(result.mass, mass, 3e-10, "isotropic wormhole horizon mass");
  Near(result.areal_radius, 2*mass, 3e-10, "isotropic wormhole areal horizon");
  if (global_variable::my_rank == 0) {
    std::cout << "fastflow_isotropic_wormhole found=1 iterations=" << result.iterations
              << " R=" << result.areal_radius << " r=" << result.mean_radius
              << " H_rms=" << result.expansion_rms << '\n';
  }
}

void TestMissingSurface() {
  ParameterInput pin;
  Input(pin, "fastflow_incomplete", 0.7);
  CMCData data{1, 16};
  FastFlow finder(nullptr, &pin, 0, [&](const Real x[3], z4c::HorizonGeometry<Real> &q) {
    if (!Own(x) || x[2] > 0) return false;
    data.Evaluate(x); q = hh::PhysicalHorizonGeometry(data.jet, data.omega);
    return true;
  });
  finder.Find(0, 0);
  Require(!finder.ah_found && !finder.GetResult().complete_surface,
          "incomplete quadrature surface rejected");
  Require(std::isnan(finder.GetResult().area), "failed horizon has no stale area");
}

void TestInvalidOwnedPoint() {
  ParameterInput pin;
  Input(pin, "fastflow_invalid", 0.7);
  FastFlow finder(nullptr, &pin, 0, [&](const Real x[3], z4c::HorizonGeometry<Real> &q) {
    if (!Own(x)) return false;
    q = {};  // owned but invalid; no division by zero or rank-local exception
    return true;
  });
  finder.Find(0, 0);
  Require(!finder.ah_found && !finder.GetResult().complete_surface,
          "owned invalid geometry rejected collectively");
}

void TestDuplicateOwners() {
  #if MPI_PARALLEL_ENABLED
    if (global_variable::nranks < 2) return;
    ParameterInput pin;
    Input(pin, "fastflow_duplicate", 0.7);
    CMCData data{1, 16};
    FastFlow finder(nullptr, &pin, 0, [&](const Real x[3], z4c::HorizonGeometry<Real> &q) {
      data.Evaluate(x); q = hh::PhysicalHorizonGeometry(data.jet, data.omega);
      return true;  // deliberately claimed by every rank
    });
    finder.Find(0, 0);
    Require(!finder.ah_found && !finder.GetResult().complete_surface,
            "duplicate MPI ownership rejected");
  #endif
}

int main(int argc, char **argv) {
  #if MPI_PARALLEL_ENABLED
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &global_variable::my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &global_variable::nranks);
  #endif
  Kokkos::initialize(argc, argv);
  int status = 0;
  try {
    TestReconstruction();
    const auto physical = Solve("fastflow_cmc_physical", false, false);
    const auto conformal = Solve("fastflow_cmc_hyperboloidal", true, false);
    Near(conformal.area, physical.area, 2e-13, "physical/conformal area reduction");
    Solve("fastflow_cmc_displaced", true, true);
    TestIsotropicWormhole();
    TestNoFalseHorizon();
    TestMissingSurface();
    TestInvalidOwnedPoint();
    TestDuplicateOwners();
    if (global_variable::my_rank == 0) std::cout << "Hyperboloidal FastFlow tests passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n'; status = 1;
  }
  Kokkos::finalize();
  #if MPI_PARALLEL_ENABLED
    MPI_Finalize();
  #endif
  return status;
}
