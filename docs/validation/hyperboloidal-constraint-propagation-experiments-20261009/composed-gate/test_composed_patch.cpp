#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include "z4c/hyperboloidal/cartesian_patch.hpp"

namespace h = z4c::hyperboloidal;
using z4c::Z4c;

void Require(bool test, const char *message) {
  if (!test) throw std::runtime_error(message);
}

h::SphericalGhostGrid Grid(int n) {
  h::SphericalGhostGrid g{};
  g.radius = 1;
  for (int d = 0; d < 3; ++d) {
    g.n[d] = n+8;
    g.h[d] = 2.1/n;
    g.first[d] = -1.05-3.5*g.h[d];
  }
  return g;
}

std::array<int,3> Unflatten(int s, const h::SphericalGhostGrid &g) {
  return {s%g.n[0],s/g.n[0]%g.n[1],s/(g.n[0]*g.n[1])};
}

double Polynomial(const double x[3]) {
  return .125+.03*x[0]+.05*x[1]-.04*x[2]+.02*x[0]*x[0]
      -.01*x[1]*x[1]+.015*x[2]*x[2]+.012*x[0]*x[1]
      +.013*x[0]*x[2]-.014*x[1]*x[2];
}
const double Hessian[3][3] = {{.04,.012,.013},{.012,-.02,-.014},{.013,-.014,.03}};
double Gradient(int d, const double x[3]) {
  const double linear[3] = {.03,.05,-.04};
  double value = linear[d];
  for (int b = 0; b < 3; ++b) value += Hessian[d][b]*x[b];
  return value;
}

void Coordinates(const h::SphericalGhostGrid &g, const std::array<int,3> &v,
                 double x[3]) {
  for (int d = 0; d < 3; ++d) x[d] = g.first[d]+v[d]*g.h[d];
}

void AuditPlan(h::CartesianConformalPatch &patch) {
  const auto &g = patch.grid;
  const auto ghosts = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),patch.ghosts);
  const auto active = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),patch.active);
  const auto mask = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),patch.mask);
  std::map<int,int> lookup;
  int max_count = 0;
  double sum_abs_max = 0;
  for (size_t q = 0; q < ghosts.extent(0); ++q) {
    const auto &p = ghosts(q);
    Require(lookup.emplace(p.target,q).second,"duplicate ghost target");
    Require(p.target >= 0 && p.target < g.n[0]*g.n[1]*g.n[2],"ghost target bounds");
    Require(!mask(p.target),"active ghost target");
    Require(p.count > 0 && p.count <= 216,"donor capacity");
    max_count = std::max(max_count,p.count);
    double absolute = 0;
    for (int j = 0; j < p.count; ++j) {
      Require(p.donors[j] >= 0 && p.donors[j] < g.n[0]*g.n[1]*g.n[2],"donor bounds");
      const auto v = Unflatten(p.donors[j],g);
      Require(g.Interior(v[0],v[1],v[2]),"exterior or recursive donor");
      Require(mask(p.donors[j]) && std::isfinite(p.weights[j]),"invalid donor mask/weight");
      absolute += std::abs(p.weights[j]);
    }
    sum_abs_max = std::max(sum_abs_max,absolute);
  }
  long axis_checks = 0, mixed_checks = 0, old_axis_checks = 0;
  const auto target = [&](const std::array<int,3> &v) {
    Require(g.Allocated(v[0],v[1],v[2]),"unallocated stencil access");
    const int s = g.Index(v[0],v[1],v[2]);
    Require(mask(s) || lookup.count(s),"required exterior target absent");
  };
  for (size_t q = 0; q < active.extent(0); ++q) {
    const auto v = Unflatten(active(q),g);
    for (int a = 0; a < 3; ++a) {
      for (int off = -4; off <= 4; ++off) {
        auto w = v; w[a] += off; target(w); ++axis_checks;
        if (std::abs(off) <= 3) ++old_axis_checks;
      }
      for (int b = a+1; b < 3; ++b)
      for (int da = -2; da <= 2; ++da)
      for (int db = -2; db <= 2; ++db) {
        auto w = v; w[a] += da; w[b] += db; target(w); ++mixed_checks;
      }
    }
  }
  const std::array<std::array<int,3>,5> permutations = {{{0,1,2},{0,1,2},{0,1,2},{1,0,2},{0,2,1}}};
  const int reflections[5] = {1,2,4,0,0};
  const auto transform = [&](int s, int generator) {
    const auto v = Unflatten(s,g);
    int w[3];
    for (int d = 0; d < 3; ++d)
      w[d] = reflections[generator] & (1 << d)
          ? g.n[d]-1-v[permutations[generator][d]] : v[permutations[generator][d]];
    return g.Index(w[0],w[1],w[2]);
  };
  double symmetry_error = 0;
  for (size_t q = 0; q < ghosts.extent(0); ++q) {
    const auto &p = ghosts(q);
    for (int tr = 0; tr < 5; ++tr) {
      const auto found = lookup.find(transform(p.target,tr));
      Require(found != lookup.end(),"non-invariant ghost target set");
      const auto &other = ghosts(found->second);
      std::map<int,double> difference;
      for (int j = 0; j < p.count; ++j) difference[transform(p.donors[j],tr)] += p.weights[j];
      for (int j = 0; j < other.count; ++j) difference[other.donors[j]] -= other.weights[j];
      for (const auto &entry : difference) symmetry_error = std::max(symmetry_error,std::abs(entry.second));
    }
  }
  Require(symmetry_error < 2e-12,"non-equivariant donor weights");
  auto scalar = DvceArray1D<Real>("quadratic ghost oracle",g.n[0]*g.n[1]*g.n[2]);
  auto sh = Kokkos::create_mirror_view(scalar);
  for (size_t s = 0; s < sh.extent(0); ++s) {
    double x[3]; Coordinates(g,Unflatten(s,g),x);
    sh(s) = mask(s) ? Polynomial(x) : std::numeric_limits<double>::quiet_NaN();
  }
  Kokkos::deep_copy(scalar,sh);
  h::FillSphericalGhosts(scalar,patch.ghosts);
  sh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),scalar);
  double quadratic_error = 0;
  for (size_t q = 0; q < ghosts.extent(0); ++q) {
    const int s = ghosts(q).target;
    double x[3]; Coordinates(g,Unflatten(s,g),x);
    Require(std::isfinite(sh(s)),"quadratic ghost poison");
    quadratic_error = std::max(quadratic_error,std::abs(sh(s)-Polynomial(x)));
  }
  Require(quadratic_error < 1e-12,"quadratic ghost consistency");
  std::cout << "{\"check\":\"halo\",\"degree\":" << patch.ghost_degree
      << ",\"active\":" << active.extent(0) << ",\"ghosts\":" << ghosts.extent(0)
      << ",\"axis_radius4_checks\":" << axis_checks << ",\"mixed_radius2_checks\":" << mixed_checks
      << ",\"upwind_KO_radius3_checks\":" << old_axis_checks << ",\"max_donors\":" << max_count
      << ",\"max_weight_l1\":" << sum_abs_max << ",\"cube_generator_weight_error\":" << symmetry_error
      << ",\"quadratic_fill_error\":" << quadratic_error << "}\n";
}

void Compare(double actual, double expected, double &error) {
  Require(std::isfinite(actual),"nonfinite consumed jet");
  error = std::max(error,std::abs(actual-expected));
}

void CompareScalar(const h::ScalarJet<Real> &j, int f, const double x[3], double &error) {
  const double scale = 1e-5*(f+1);
  Compare(j.value,scale*Polynomial(x),error);
  for (int a = 0; a < 3; ++a) {
    Compare(j.d[a],scale*Gradient(a,x),error);
    for (int b = 0; b < 3; ++b) Compare(j.dd[a][b],scale*Hessian[a][b],error);
  }
}

void AuditActualPatch(h::CartesianConformalPatch &patch) {
  const auto &g = patch.grid;
  auto data = patch.Allocate("gate state"), rhs = patch.Allocate("gate RHS");
  const auto active = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),patch.active);
  const auto mask = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),patch.mask);
  patch.InitializeReference(data);
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),data);
  for (int s = 0; s < g.n[0]*g.n[1]*g.n[2]; ++s) {
    if (mask(s)) continue;
    const auto v = Unflatten(s,g);
    for (int f = 0; f < Z4c::nz4c; ++f) host(0,f,v[2],v[1],v[0]) = std::numeric_limits<double>::quiet_NaN();
  }
  Kokkos::deep_copy(data,host);
  patch.RHS(data,rhs);
  auto rh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),rhs);
  double reference_rhs = 0;
  for (size_t s = 0; s < rh.size(); ++s) {
    Require(std::isfinite(rh.data()[s]),"reference RHS inactive poison or bounds");
    reference_rhs = std::max(reference_rhs,std::abs(rh.data()[s]));
  }
  Require(reference_rhs < 1e-10,"reference RHS above roundoff");
  const auto diagnostic = patch.Diagnose(data);
  Require(diagnostic.h_l2 < 1e-11 && diagnostic.m_l2 < 1e-11 && diagnostic.z_l2 < 1e-11,
          "original diagnostic reference mismatch");
  patch.InitializeReference(data);
  host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),data);
  for (int s = 0; s < g.n[0]*g.n[1]*g.n[2]; ++s) {
    const auto v = Unflatten(s,g);
    double x[3]; Coordinates(g,v,x);
    for (int f = 0; f < Z4c::nz4c; ++f) {
      if (mask(s)) host(0,f,v[2],v[1],v[0]) += 1e-5*(f+1)*Polynomial(x);
      else host(0,f,v[2],v[1],v[0]) = std::numeric_limits<double>::quiet_NaN();
    }
  }
  Kokkos::deep_copy(data,host);
  patch.Prepare(data);
  const auto q = h::BindCartesianFields(patch.deviations);
  const double inv[3] = {1/g.h[0],1/g.h[1],1/g.h[2]};
  const int fi[6] = {0,0,0,1,1,2}, fj[6] = {0,1,2,1,2,2};
  double jet_error = 0;
  for (size_t point = 0; point < active.extent(0); ++point) {
    const auto v = Unflatten(active(point),g);
    double x[3]; Coordinates(g,v,x);
    const auto u = h::LoadComposedMeshJet<3>(q,inv,0,v[2],v[1],v[0]);
    CompareScalar(u.chi,Z4c::I_Z4C_CHI,x,jet_error);
    CompareScalar(u.alpha,Z4c::I_Z4C_ALPHA,x,jet_error);
    CompareScalar(u.trace,Z4c::I_Z4C_KHAT,x,jet_error);
    CompareScalar(u.theta,Z4c::I_Z4C_THETA,x,jet_error);
    for (int a = 0; a < 3; ++a) {
      const int fields[2] = {Z4c::I_Z4C_BETAX+a,Z4c::I_Z4C_GAMX+a};
      for (int f = 0; f < 2; ++f) {
        const auto &b = f == 0 ? u.beta : u.lambda;
        const double scale = 1e-5*(fields[f]+1);
        Compare(b.value[a],scale*Polynomial(x),jet_error);
        for (int d = 0; d < 3; ++d) {
          Compare(b.d[d][a],scale*Gradient(d,x),jet_error);
          for (int e = 0; e < 3; ++e) Compare(b.dd[d][e][a],scale*Hessian[d][e],jet_error);
        }
      }
    }
    for (int f = 0; f < 6; ++f) {
      const int a = fi[f], b = fj[f];
      const double sg = 1e-5*(Z4c::I_Z4C_GXX+f+1), sa = 1e-5*(Z4c::I_Z4C_AXX+f+1);
      Compare(u.metric.g[a][b],sg*Polynomial(x),jet_error);
      Compare(u.a.k[a][b],sa*Polynomial(x),jet_error);
      for (int d = 0; d < 3; ++d) {
        Compare(u.metric.dg[d][a][b],sg*Gradient(d,x),jet_error);
        Compare(u.a.dk[d][a][b],sa*Gradient(d,x),jet_error);
        for (int e = 0; e < 3; ++e) Compare(u.metric.ddg[d][e][a][b],sg*Hessian[d][e],jet_error);
      }
    }
  }
  Require(jet_error < 2e-10,"quadratic full consumed jet mismatch");
  patch.RHS(data,rhs);
  rh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),rhs);
  double finite_rhs = 0;
  for (size_t s = 0; s < rh.size(); ++s) {
    Require(std::isfinite(rh.data()[s]),"perturbed RHS nonfinite or bounds");
    finite_rhs = std::max(finite_rhs,std::abs(rh.data()[s]));
  }
  Require(finite_rhs > 1e-8,"nontrivial manufactured RHS unexpectedly zero");
  std::cout << "{\"check\":\"actual_patch\",\"degree\":" << patch.ghost_degree
      << ",\"reference_RHS_max\":" << reference_rhs << ",\"reference_H_rms\":" << diagnostic.h_l2
      << ",\"reference_M_rms\":" << diagnostic.m_l2 << ",\"reference_Z_rms\":" << diagnostic.z_l2
      << ",\"all_active_consumed_quadratic_jet_error\":" << jet_error
      << ",\"manufactured_RHS_max\":" << finite_rhs << "}\n";
}

void AuditSin() {
  std::vector<double> errors;
  double convolution_error = 0, mixed_composition_error = 0;
  for (int n : {16,32,64}) {
    const auto g = Grid(n);
    auto data = DvceArray5D<Real>("analytic sine",1,25,g.n[2],g.n[1],g.n[0]);
    auto host = Kokkos::create_mirror_view(data);
    const double wave[3] = {1.3,1.7,2.1};
    for (int k = 0; k < g.n[2]; ++k)
    for (int j = 0; j < g.n[1]; ++j)
    for (int i = 0; i < g.n[0]; ++i) {
      double x[3]; Coordinates(g,{i,j,k},x);
      const double value = std::sin(.17+wave[0]*x[0]+wave[1]*x[1]+wave[2]*x[2]);
      for (int f = 0; f < 25; ++f) host(0,f,k,j,i) = value;
    }
    Kokkos::deep_copy(data,host);
    const auto q = h::BindCartesianFields(data);
    const double inv[3] = {1/g.h[0],1/g.h[1],1/g.h[2]};
    const double coefficients[9] = {1,-16,64,16,-130,16,64,-16,1};
    double error = 0;
    for (int k = 4; k < g.n[2]-4; ++k)
    for (int j = 4; j < g.n[1]-4; ++j)
    for (int i = 4; i < g.n[0]-4; ++i) {
      if (!g.Interior(i,j,k)) continue;
      const std::array<int,3> v = {i,j,k};
      for (int d = 0; d < 3; ++d) {
        const double composed = h::ComposedDxx<3>(d,inv,q.chi,0,k,j,i);
        double convolution = 0;
        for (int off = -4; off <= 4; ++off) {
          auto w = v; w[d] += off;
          convolution += coefficients[off+4]*host(0,0,w[2],w[1],w[0])/(144*g.h[d]*g.h[d]);
        }
        convolution_error = std::max(convolution_error,std::abs(composed-convolution));
        error = std::max(error,std::abs(composed+wave[d]*wave[d]*host(0,0,k,j,i)));
        for (int e = d+1; e < 3; ++e) {
          auto first = h::FirstDerivativeField<3,decltype(q.chi)>(e,inv,q.chi);
          const double mixed = Dx<3>(d,inv,first,0,k,j,i);
          mixed_composition_error = std::max(mixed_composition_error,
              std::abs(mixed-Dxy<3>(d,e,inv,q.chi,0,k,j,i)));
        }
      }
    }
    errors.push_back(error);
  }
  Require(convolution_error < 1e-10,"nine-point diagonal convolution mismatch");
  Require(mixed_composition_error < 1e-10,"old mixed derivative no longer composed first derivatives");
  const double order1 = std::log(errors[0]/errors[1])/std::log(2.),
      order2 = std::log(errors[1]/errors[2])/std::log(2.);
  Require(order1 > 3.95 && order2 > 3.95,"sine composed second derivative convergence");
  std::cout << "{\"check\":\"sine\",\"N\":[16,32,64],\"errors\":["
      << errors[0] << ',' << errors[1] << ',' << errors[2] << "],\"orders\":["
      << order1 << ',' << order2 << "],\"nine_point_error\":" << convolution_error
      << ",\"mixed_Dx_composition_error\":" << mixed_composition_error << "}\n";
}

int main(int argc, char **argv) {
  Kokkos::ScopeGuard guard(argc,argv);
  std::cout << std::setprecision(17);
  try {
    for (int degree : {2,3}) {
      h::LayerGaugeParameters gauge;
      gauge.physical_trace_lapse = true;
      gauge.preferred_source = false;
      h::CartesianConformalPatch patch(Grid(24),.5,degree,{true,.05,.95},gauge,true);
      patch.kappa1 = 10;
      AuditPlan(patch);
      AuditActualPatch(patch);
    }
    AuditSin();
    std::cout << "{\"check\":\"complete\",\"status\":\"PASS\"}\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
