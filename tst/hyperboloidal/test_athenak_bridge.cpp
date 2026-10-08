// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include "z4c/hyperboloidal/athenak_bridge.hpp"

namespace hyp = z4c::hyperboloidal;
using Z = z4c::Z4c;

void Check(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}

Z::Z4c_vars Bind(DvceArray5D<Real> data) {
  Z::Z4c_vars q;
  q.chi.InitWithShallowSlice(data, Z::I_Z4C_CHI);
  q.vKhat.InitWithShallowSlice(data, Z::I_Z4C_KHAT);
  q.vTheta.InitWithShallowSlice(data, Z::I_Z4C_THETA);
  q.alpha.InitWithShallowSlice(data, Z::I_Z4C_ALPHA);
  q.g_dd.InitWithShallowSlice(data, Z::I_Z4C_GXX, Z::I_Z4C_GZZ);
  q.vA_dd.InitWithShallowSlice(data, Z::I_Z4C_AXX, Z::I_Z4C_AZZ);
  q.vGam_u.InitWithShallowSlice(data, Z::I_Z4C_GAMX, Z::I_Z4C_GAMZ);
  q.beta_u.InitWithShallowSlice(data, Z::I_Z4C_BETAX, Z::I_Z4C_BETAZ);
  q.vB_d.InitWithShallowSlice(data, Z::I_Z4C_BX, Z::I_Z4C_BZ);
  return q;
}

// All components have distinct amplitudes, and all Cartesian mixed derivatives
// are nonzero. Refinement checks AthenaK's actual 2nd/4th/6th order FD operators.
template <int NGHOST>
void Derivatives() {
  DvceArray5D<Real> data("polynomial state", 1, Z::nz4c, 9, 9, 9);
  const auto q = Bind(data);
  double last = 0;
  for (double h : {0.4, 0.2, 0.1}) {
    auto host = Kokkos::create_mirror_view(data);
    for (int f = 0; f < Z::nz4c; ++f)
    for (int k = 0; k < 9; ++k)
    for (int j = 0; j < 9; ++j)
    for (int i = 0; i < 9; ++i) {
      const double phase = 0.37+h*(0.7*(i-4)-0.4*(j-4)+0.3*(k-4));
      host(0,f,k,j,i) = 0.1*f+(1+f)*std::sin(phase);
    }
    Kokkos::deep_copy(data, host);
    Real error = 0;
    Kokkos::parallel_reduce("mesh derivative audit", 1,
        KOKKOS_LAMBDA(const int, Real &err) {
      const Real idx[3] = {1/h, 1/h, 1/h}, wave[3] = {0.7, -0.4, 0.3};
      const auto u = hyp::LoadMeshJet<NGHOST>(q, idx, 0,4,4,4);
      const hyp::ScalarJet<Real> scalars[4] = {u.chi, u.trace, u.theta, u.alpha};
      const int fields[4] = {Z::I_Z4C_CHI, Z::I_Z4C_KHAT, Z::I_Z4C_THETA, Z::I_Z4C_ALPHA};
      for (int s = 0; s < 4; ++s) {
        const int f = fields[s];
        err += Kokkos::fabs(scalars[s].value-(0.1*f+(1+f)*Kokkos::sin(0.37)));
        for (int a = 0; a < 3; ++a) {
          err += Kokkos::fabs(scalars[s].d[a]-(1+f)*wave[a]*Kokkos::cos(0.37));
          for (int b = 0; b < 3; ++b) {
            err += Kokkos::fabs(scalars[s].dd[a][b]
                +(1+f)*wave[a]*wave[b]*Kokkos::sin(0.37));
          }
        }
      }
      const int tensor_index[3][3] = {{0,1,2}, {1,3,4}, {2,4,5}};
      for (int a = 0; a < 3; ++a) {
        const int bf = Z::I_Z4C_BETAX+a, lf = Z::I_Z4C_GAMX+a;
        err += Kokkos::fabs(u.beta.value[a]-(0.1*bf+(1+bf)*Kokkos::sin(0.37)));
        err += Kokkos::fabs(u.lambda.value[a]-(0.1*lf+(1+lf)*Kokkos::sin(0.37)));
        for (int b = 0; b < 3; ++b) {
          const int f = Z::I_Z4C_GXX+tensor_index[a][b];
          const int af = Z::I_Z4C_AXX+tensor_index[a][b];
          err += Kokkos::fabs(u.metric.g[a][b]-(0.1*f+(1+f)*Kokkos::sin(0.37)));
          err += Kokkos::fabs(u.a.k[a][b]-(0.1*af+(1+af)*Kokkos::sin(0.37)));
        }
      }
      for (int a = 0; a < 3; ++a)
      for (int d = 0; d < 3; ++d) {
        err += Kokkos::fabs(u.beta.d[d][a]
            -(1+Z::I_Z4C_BETAX+a)*wave[d]*Kokkos::cos(0.37));
        err += Kokkos::fabs(u.lambda.d[d][a]
            -(1+Z::I_Z4C_GAMX+a)*wave[d]*Kokkos::cos(0.37));
        for (int e = 0; e < 3; ++e) {
          err += Kokkos::fabs(u.beta.dd[d][e][a]
              +(1+Z::I_Z4C_BETAX+a)*wave[d]*wave[e]*Kokkos::sin(0.37));
        }
        for (int b = 0; b < 3; ++b) {
          const int f = Z::I_Z4C_GXX+tensor_index[a][b];
          const int af = Z::I_Z4C_AXX+tensor_index[a][b];
          err += Kokkos::fabs(u.metric.dg[d][a][b]-(1+f)*wave[d]*Kokkos::cos(0.37));
          err += Kokkos::fabs(u.a.dk[d][a][b]-(1+af)*wave[d]*Kokkos::cos(0.37));
          for (int e = 0; e < 3; ++e) {
            err += Kokkos::fabs(u.metric.ddg[d][e][a][b]
                +(1+f)*wave[d]*wave[e]*Kokkos::sin(0.37));
          }
        }
      }
    }, error);
    std::cout << "mesh FD order=" << 2*(NGHOST-1) << " h=" << h
              << " error=" << error << '\n';
    if (last > 0) {
      Check(last/error > std::pow(2., 2*(NGHOST-1)-0.2), "mesh FD convergence");
    }
    last = error;
  }
}

void MinkowskiMesh() {
  DvceArray5D<Real> data("CMC mesh", 1, Z::nz4c, 12, 12, 12);
  DvceArray5D<Real> output("RHS mesh", 1, Z::nz4c, 12, 12, 12);
  Kokkos::deep_copy(output, -999.);
  const auto q = Bind(data), rhs = Bind(output);
  const hyp::CMCReference<Real> ref{1, 1};
  Kokkos::parallel_for("CMC fill", 12*12*12, KOKKOS_LAMBDA(const int index) {
    const int i = index%12, j = (index/12)%12, k = index/144;
    const auto p = ref.At((i-5.5)*0.035, (j-5.5)*0.035, (k-5.5)*0.035);
    q.chi(0,k,j,i) = 1;
    q.alpha(0,k,j,i) = p.alpha;
    q.vKhat(0,k,j,i) = p.k_physical;
    for (int a = 0; a < 3; ++a) {
      q.g_dd(0,a,a,k,j,i) = 1;
      q.beta_u(0,a,k,j,i) = p.beta[a];
    }
  });
  int failures = 0;
  Kokkos::parallel_reduce("Cartesian CMC RHS", 6*6*6,
      KOKKOS_LAMBDA(const int index, int &bad) {
    const int i = 3+index%6, j = 3+(index/6)%6, k = 3+index/36;
    const Real spacing[3] = {0.035, 0.035, 0.035};
    const Real xyz[3] = {(i-5.5)*0.035, (j-5.5)*0.035, (k-5.5)*0.035};
    hyp::Z4cJet<Real> u{};
    if (!hyp::LoadInteriorMeshJet<3>(q, ref, xyz, spacing, 0,k,j,i, u)) {
      ++bad;
      return;
    }
    const auto p = ref.At((i-5.5)*0.035, (j-5.5)*0.035, (k-5.5)*0.035);
    hyp::OmegaJet<Real> o{};
    o.omega = p.omega;
    for (int a = 0; a < 3; ++a) {
      o.gradient[a] = p.domega[a];
      o.hessian[a][a] = p.hessian_omega;
    }
    hyp::SetStationaryOmegaNormal(u.alpha.value, u.beta.value, u.alpha.d, u.beta.d, o);
    hyp::Z4cRHS<Real> evolution{};
    hyp::GaugeRHS<Real> gauge{};
    const auto gauge_parts = hyp::UnfactoredReferenceGauge(ref, p, u,
        hyp::GaugeParameters<Real>{2, 0.1, 1.5, 1}, true, true);
    if (!hyp::AssembleInterior(hyp::ConformalRHS(u, o, Real(5)/u.alpha.value, Real(0)),
                              o.omega, evolution)
        || !hyp::AssembleGaugeInterior(gauge_parts, o.omega, gauge)) ++bad;
    hyp::StoreMeshRHS(rhs, 0,k,j,i, evolution, gauge);
    const auto adm = hyp::ToPhysicalADM(u, o.omega);
    if (!adm.valid || Kokkos::fabs(adm.alpha-u.alpha.value/o.omega) > 1e-12) ++bad;
    for (int a = 0; a < 3; ++a) {
      if (Kokkos::fabs(adm.curvature[a][a]+adm.metric[a][a]) > 1e-11) ++bad;
    }
    const auto constraints = hyp::EvolvedConstraints(u, o);
    if (!constraints.valid || Kokkos::fabs(constraints.hamiltonian) > 1e-10) ++bad;
  }, failures);
  Check(failures == 0, "Cartesian CMC mesh calculation");
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output);
  for (int f = 0; f < Z::nz4c; ++f)
  for (int k = 0; k < 12; ++k)
  for (int j = 0; j < 12; ++j)
  for (int i = 0; i < 12; ++i) {
    const bool inside = i >= 3 && i < 9 && j >= 3 && j < 9 && k >= 3 && k < 9;
    Check(inside ? std::abs(host(0,f,k,j,i)) < 1e-10 : host(0,f,k,j,i) == -999.,
          "mesh RHS writes or reference stationarity");
  }
}

void PhysicalConversion() {
  hyp::Z4cJet<Real> u{};
  u.chi.value = 0.8;
  u.alpha.value = 0.9;
  u.trace.value = -2.3;
  u.theta.value = 0.07;
  u.metric.g[0][0] = 1.25;
  u.metric.g[1][1] = u.metric.g[2][2] = 1;
  u.metric.g[0][1] = u.metric.g[1][0] = 0.5;
  const auto geometry = hyp::Geometry(u.metric);
  Real trace = 0;
  for (int a = 0; a < 3; ++a)
  for (int b = 0; b < 3; ++b) {
    u.a.k[a][b] = 0.03*(1+a+b);
    trace += geometry.inverse[a][b]*u.a.k[a][b];
  }
  for (int a = 0; a < 3; ++a)
  for (int b = 0; b < 3; ++b) u.a.k[a][b] -= u.metric.g[a][b]*trace/3;
  for (Real omega : {Real(0.2), Real(0.7), Real(1)}) {
    const auto p = hyp::ToPhysicalADM(u, omega);
    Check(p.valid, "physical ADM conversion rejected");
    Real physical_trace = 0;
    const Real chi = omega*omega*u.chi.value;
    for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) {
      physical_trace += chi*geometry.inverse[a][b]*p.curvature[a][b];
      const Real recovered_a = (chi*p.curvature[a][b]
          -(u.trace.value+2*u.theta.value)*u.metric.g[a][b]/3)/omega;
      Check(std::abs(recovered_a-u.a.k[a][b]) < 1e-13, "physical tracefree conversion");
      Check(std::abs(chi*p.metric[a][b]-u.metric.g[a][b]) < 1e-13, "metric conversion");
    }
    Check(std::abs(physical_trace-u.trace.value-2*u.theta.value) < 1e-13,
          "physical trace must include Theta");
    Check(std::abs(omega*p.alpha-u.alpha.value) < 1e-13, "physical lapse conversion");
  }
  Check(!hyp::ToPhysicalADM(u, Real(0)).valid, "physical ADM allowed at scri");
  Check(!hyp::ToPhysicalADM(u, Real(-0.1)).valid, "physical ADM allowed outside scri");
  Check(!hyp::ToPhysicalADM(u, std::numeric_limits<Real>::quiet_NaN()).valid,
        "nonfinite compactifier allowed");
  Check(!hyp::ToPhysicalADM(u, Real(1e200)).valid, "underflowed physical metric accepted");
  u.chi.value = std::numeric_limits<Real>::infinity();
  Check(!hyp::ToPhysicalADM(u, Real(1)).valid, "infinite chi accepted");
  u.chi.value = 0;
  Check(!hyp::ToPhysicalADM(u, Real(1)).valid, "zero chi accepted");
}

void GaugeEquivalence() {
  const hyp::CMCReference<Real> ref{2, 3};
  const hyp::GaugeParameters<Real> params{0.4, 0.1, 1.5, 1};
  for (Real radius : {Real(0.2), Real(1), Real(1.99)}) {
    const auto p = ref.At(radius/3, 2*radius/3, 2*radius/3);
    hyp::GaugeDeviation<Real> d{};
    d.lapse = 0.013;
    d.trace = -0.027;
    d.chi = 0.91;
    hyp::Z4cJet<Real> u{};
    u.alpha.value = p.alpha+p.omega*d.lapse;
    u.trace.value = p.k_physical+p.omega*d.trace;
    u.chi.value = d.chi;
    for (int a = 0; a < 3; ++a) {
      d.shift[a] = 0.01*(a-1);
      d.lambda[a] = -0.003*(a+1);
      d.dlapse[a] = 0.007*(a-1);
      u.lambda.value[a] = d.lambda[a];
      u.beta.value[a] = p.beta[a]+p.omega*d.shift[a];
      u.alpha.d[a] = p.dalpha[a]+p.domega[a]*d.lapse+p.omega*d.dlapse[a];
      for (int b = 0; b < 3; ++b) {
        d.dshift[a][b] = 0.002*(a+b-1);
        u.beta.d[b][a] = (a == b ? -1/ref.curvature_radius : 0)
            +p.domega[b]*d.shift[a]+p.omega*d.dshift[a][b];
      }
    }
    const auto expected = hyp::ReferenceGauge(ref, p, d, params);
    const auto parts = hyp::UnfactoredReferenceGauge(ref, p, u, params);
    hyp::GaugeRHS<Real> actual{};
    Check(hyp::AssembleGaugeInterior(parts, p.omega, actual), "gauge assembly");
    Check(std::abs(actual.alpha-expected.alpha) < 1e-12, "factored gauge equivalence");
    for (int a = 0; a < 3; ++a) {
      Check(std::abs(actual.beta[a]-expected.beta[a]) < 1e-12, "Cartesian shift gauge");
    }
    const auto modified = hyp::UnfactoredReferenceGauge(ref, p, u, params, true, true);
    const Real radial = 2*ref.curvature_radius*ref.scri_radius*p.omega;
    const Real change = -params.slicing*radial*radial*(u.alpha.value-1)
        *(u.trace.value-p.k_physical)
        +params.lapse_damping*p.alpha*(u.alpha.value-p.alpha);
    Check(std::abs(modified.pole.alpha-parts.pole.alpha-change) < 1e-13,
          "puncture gauge modifications");
    Check(!hyp::AssembleGaugeInterior(parts, Real(0), actual),
          "gauge pole divided at scri");
  }
}

void StencilAdmission() {
  const hyp::CMCReference<Real> ref{1, 1};
  const Real xyz[3] = {0.69, 0.69, 0}, spacing[3] = {0.012, 0.012, 0.012};
  // Every axial sample is inside; a mixed derivative corner is outside.
  for (int a = 0; a < 3; ++a)
  for (int sign : {-1, 1}) {
    Real x[3] = {xyz[0], xyz[1], xyz[2]};
    x[a] += sign*2*spacing[a];
    Check(ref.At(x[0], x[1], x[2]).omega > 0, "admission counterexample axes");
  }
  const Z::Z4c_vars unallocated{};
  hyp::Z4cJet<Real> u{};
  Check(!hyp::LoadInteriorMeshJet<3>(unallocated, ref, xyz, spacing, 0,0,0,0, u),
        "mixed derivative reached outside scri");
  const Real invalid[3] = {0, 0.1, 0.1};
  Check(!hyp::LoadInteriorMeshJet<3>(unallocated, ref, xyz, invalid, 0,0,0,0, u),
        "invalid spacing accepted");
}

void NonzeroPacking() {
  DvceArray5D<Real> data("packing", 1, Z::nz4c, 1, 1, 1);
  const auto q = Bind(data);
  Kokkos::parallel_for("nonzero RHS packing", 1, KOKKOS_LAMBDA(const int) {
    hyp::Z4cRHS<Real> rhs{};
    hyp::GaugeRHS<Real> gauge{};
    rhs.chi = 100+Z::I_Z4C_CHI;
    rhs.trace = 100+Z::I_Z4C_KHAT;
    rhs.theta = 100+Z::I_Z4C_THETA;
    gauge.alpha = 100+Z::I_Z4C_ALPHA;
    const int indices[3][3] = {{0,1,2}, {1,3,4}, {2,4,5}};
    for (int a = 0; a < 3; ++a) {
      rhs.lambda[a] = 100+Z::I_Z4C_GAMX+a;
      gauge.beta[a] = 100+Z::I_Z4C_BETAX+a;
      for (int b = 0; b < 3; ++b) {
        rhs.metric[a][b] = 100+Z::I_Z4C_GXX+indices[a][b];
        rhs.a[a][b] = 100+Z::I_Z4C_AXX+indices[a][b];
      }
    }
    hyp::StoreMeshRHS(q, 0,0,0,0, rhs, gauge);
  });
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data);
  for (int f = 0; f < Z::nz4c; ++f) {
    Check(host(0,f,0,0,0) == (f >= Z::I_Z4C_BX ? 0 : 100+f), "nonzero RHS packing");
  }
}

int main(int argc, char **argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  try {
    Derivatives<2>();
    Derivatives<3>();
    Derivatives<4>();
    MinkowskiMesh();
    PhysicalConversion();
    GaugeEquivalence();
    NonzeroPacking();
    StencilAdmission();
    std::cout << "PASS AthenaK Cartesian mesh adapter\n";
  } catch (const std::exception &e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
