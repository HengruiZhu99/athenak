// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Analytic geometry, ownership failures, full-harmonic masks; no evolved fields.
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include "globals.hpp"
#include "parameter_input.hpp"
#include "z4c/fastflow.hpp"

namespace {
int owned_samples = 0;
void Require(bool yes, const char *message) {
  int failed = yes ? 0 : 1;
  MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (failed) throw std::runtime_error(message);
}
void Pass(const std::string &name) {
  if (!global_variable::my_rank) std::cout << "PASS " << name << '\n';
}
void Input(ParameterInput &pin, const std::string &name, Real initial = 0.5,
           int iterations = 300) {
  std::ostringstream text;
  text << "<job>\nbasename=" << name << "\n<fastflow>\nntheta=12\nlmax=8\n"
       << "flow_iterations_0=" << iterations << "\nflow_alpha_beta_const_0=1\n"
       << "mass_tol_0=1e-12\nexpansion_rms_tol_0=1e-9\nhmean_tol_0=100\n"
       << "initial_radius_0=" << initial << "\nstart_time_0=0\nstop_time_0=10\n";
  std::istringstream stream(text.str());
  pin.LoadFromStream(stream);
}
int Owner(const Real *x) { return x[0] >= 0 ? 0 : 1; }
z4c::HorizonGeometry<Real> Schwarzschild(const Real *x) {
  z4c::HorizonGeometry<Real> q{};
  const Real r = std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
  const Real psi = 1+0.5/r;
  for (int a = 0; a < 3; ++a) {
    q.metric[a][a] = std::pow(psi, 4);
    for (int d = 0; d < 3; ++d)
      q.derivative[d][a][a] = -2*std::pow(psi, 3)*x[d]/(r*r*r);
  }
  q.valid = true;
  return q;
}
bool Scalar(const Real *x, z4c::HorizonGeometry<Real> &q) {
  if (Owner(x) != global_variable::my_rank) return false;
  q = Schwarzschild(x);
  ++owned_samples;
  return true;
}
void Batch(int count, const Real *x, z4c::HorizonGeometry<Real> *q, int *owners) {
  for (int p = 0; p < count; ++p) owners[p] = Scalar(&x[3*p], q[p]) ? 1 : 0;
}
void Analytic() {
  ParameterInput p, b;
  Input(p, "scalar"); Input(b, "batch");
  FastFlow scalar(nullptr, &p, 0, Scalar), batch(nullptr, &b, 0, Scalar);
  batch.SetBatchGeometrySampler(Batch);
  scalar.Find(0, 0); batch.Find(0, 0);
  const auto a = scalar.GetResult(), c = batch.GetResult();
  Require(a.found && c.found && a.complete_surface && c.complete_surface,
          "analytic horizon not found");
  Require(a.area == c.area && a.mass == c.mass && a.expansion_rms == c.expansion_rms
          && a.mean_radius == c.mean_radius && a.iterations == c.iterations,
          "scalar and batch searches differ");
  Require(std::abs(a.area-16*M_PI) < 2e-12 && std::abs(a.mass-1) < 3e-14
          && std::abs(a.mean_radius-0.5) < 3e-14 && a.expansion_rms < 2e-13,
          "isotropic Schwarzschild analytic solution disagrees");
  auto surface = scalar.GetSurface();
  Require(surface.Valid() && std::abs(surface.Radius(.7,1.1)-.5) < 3e-14,
          "accepted harmonic surface unavailable");
  scalar.Write(0, 0); batch.Write(0, 0);
  if (!global_variable::my_rank) {
    std::cout << std::setprecision(17) << "SCHWARZSCHILD area=" << a.area
              << " mass=" << a.mass << " rms=" << a.expansion_rms
              << " iterations=" << a.iterations << '\n';
  }
  int zero_owners = owned_samples == 0 ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &zero_owners, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  Require(zero_owners == global_variable::nranks-2, "zero-owner rank fixture changed");
  if (!global_variable::my_rank) std::cout << "ZERO_OWNER_RANKS " << zero_owners << '\n';
  Pass("schwarzschild_scalar_batch");
  scalar.Find(1, 11);
  const auto skipped = scalar.GetResult();
  Require(!skipped.found && !skipped.complete_surface && skipped.iterations == 0
          && skipped.status == FastFlow::SearchStatus::not_searched
          && !scalar.GetSurface().Valid() && std::isnan(skipped.area),
          "skipped search exposed stale accepted surface");
  Pass("skipped_search");

  // A genuine flow, starting away from the analytic horizon, exercises updates.
  ParameterInput f; Input(f, "flow", .65, 1000);
  FastFlow flow(nullptr, &f, 0, Scalar); flow.SetBatchGeometrySampler(Batch);
  flow.Find(0, 0);
  const auto r = flow.GetResult();
  Require(r.found && r.iterations > 2 && std::abs(r.mean_radius-.5) < 2e-8
          && r.expansion_rms <= 1e-9 && std::abs(r.mass-1) < 2e-13,
          "nontrivial Schwarzschild flow failed");
  if (!global_variable::my_rank) std::cout << "FLOW iterations=" << r.iterations
      << " radius=" << r.mean_radius << " rms=" << r.expansion_rms << '\n';
  Pass("schwarzschild_nontrivial_flow");
}
void FailureCases() {
  const std::string names[] = {"missing", "duplicate", "invalid_geometry", "bad_flag",
                             "batch_exception", "scalar_exception", "invalid_radius"};
  for (const auto &name : names) {
    ParameterInput pin; Input(pin, name);
    int calls = 0;
    auto sampler = [&](const Real *x, z4c::HorizonGeometry<Real> &q) {
      if (name == "scalar_exception" && global_variable::my_rank == 0)
        throw std::runtime_error("synthetic rank-local callback failure");
      return Scalar(x,q);
    };
    FastFlow flow(nullptr, &pin, 0, sampler);
    if (name != "scalar_exception") {
      flow.SetBatchGeometrySampler([&](int count, const Real *x,
          z4c::HorizonGeometry<Real> *q, int *owners) {
        ++calls; Batch(count,x,q,owners);
        if (name == "batch_exception" && global_variable::my_rank == 0)
          throw std::runtime_error("synthetic rank-local callback failure");
        for (int p = 0; p < count; ++p) {
          if (name == "missing" && global_variable::my_rank == 0) owners[p] = 0;
          if (name == "duplicate" && global_variable::my_rank == 1) {
            owners[p] = 1; q[p] = Schwarzschild(&x[3*p]);
          }
          if (name == "invalid_geometry" && owners[p]) q[p].metric[0][0] = -1;
          if (name == "bad_flag" && owners[p]) owners[p] = 2;
        }
      });
    }
    if (name == "invalid_radius" && global_variable::my_rank == 0)
      flow.initial_radius = -1;
    flow.Find(0,0);
    const auto r = flow.GetResult();
    Require(!r.found && !r.complete_surface
            && r.status == FastFlow::SearchStatus::incomplete_surface
            && !flow.GetSurface().Valid() && std::isnan(r.area),
            "sampling failure did not fail coherently");
    if (name == "invalid_radius") Require(calls == 0,
        "rank-local invalid radius entered communicating batch callback");
    Pass(name);
  }
  ParameterInput pin; Input(pin, "stale");
  bool bad = false;
  FastFlow flow(nullptr, &pin, 0, [&](const Real *x, z4c::HorizonGeometry<Real> &q) {
    const bool owner = Scalar(x,q); if (bad && owner) q.valid = false; return owner;
  });
  flow.Find(0,0); Require(flow.GetResult().found,"stale fixture initial search failed");
  bad = true; flow.Find(1,1);
  Require(!flow.GetResult().found && std::isnan(flow.GetResult().mass)
          && !flow.GetSurface().Valid(),"failed search returned previous horizon");
  Pass("failed_after_success");
  ParameterInput p; Input(p,"nonconverged",.65,1);
  FastFlow limited(nullptr,&p,0,Scalar); limited.Find(0,0);
  Require(!limited.GetResult().found && limited.GetResult().complete_surface
          && limited.GetResult().status == FastFlow::SearchStatus::failed
          && std::isfinite(limited.GetResult().last_expansion_rms),
          "nonconvergence status wrong");
  Pass("nonconverged");
  ParameterInput io; Input(io,"nonexistent-directory-for-fastflow-test/output");
  bool caught = false;
  try { FastFlow bad_io(nullptr,&io,0,Scalar); } catch (const std::runtime_error &) {
    caught = true;
  }
  Require(caught,"rank-local output open failure was not collective");
  Pass("collective_output_failure");
}
void Harmonics() {
  // Independent closed form for l=2,m=0 and l=2,m=2 (both real components).
  FastFlowSurface s; s.lmax = 2; s.center = {.3,-.2,.1};
  s.a0.resize(3); s.ac.resize(9); s.as.resize(9);
  s.a0[0] = 2*std::sqrt(4*M_PI); s.a0[2] = .17;
  s.ac[lmindex(2,2,2)] = .21; s.as[lmindex(2,2,2)] = -.13; s.Bound();
  int masks = 0;
  for (int t = 1; t < 31; ++t) for (int p = 0; p < 64; ++p) {
    const Real theta = M_PI*t/31, phi = 2*M_PI*p/64;
    const Real c = std::cos(theta), sn = std::sin(theta);
    const Real radius = 2+.17*std::sqrt(5/(16*M_PI))*(3*c*c-1)
        +std::sqrt(15/(16*M_PI))*sn*sn*(.21*std::cos(2*phi)-.13*std::sin(2*phi));
    Require(std::abs(s.Radius(theta,phi)-radius) < 2e-15
            && radius >= s.lower && radius <= s.upper,"full harmonic radius or bound wrong");
    for (const Real factor : {.99,1.01}) {
      Real x[3] = {s.center[0]+factor*radius*sn*std::cos(phi),
                   s.center[1]+factor*radius*sn*std::sin(phi),
                   s.center[2]+factor*radius*c};
      Require(s.Outside(x) == (factor>1),"nonspherical exterior classification wrong");
      ++masks;
    }
  }
  Require(!s.Outside(s.center.data()),"surface center classified exterior");
  if (!global_variable::my_rank) std::cout << "HARMONICS masks=" << masks << '\n';
  Pass("full_harmonic_mask");
}
}  // namespace
int main(int argc, char **argv) {
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);
  Kokkos::initialize(argc,argv);
  int result = 0;
  try {
    Require(global_variable::nranks >= 2,"test requires at least two ranks");
    Analytic(); FailureCases(); Harmonics();
    if (!global_variable::my_rank) std::cout << "PASS fastflow_external_all\n";
  } catch (const std::exception &e) {
    std::cerr << "rank " << global_variable::my_rank << ": " << e.what() << '\n';
    result = 1;
  }
  Kokkos::finalize(); MPI_Finalize(); return result;
}
