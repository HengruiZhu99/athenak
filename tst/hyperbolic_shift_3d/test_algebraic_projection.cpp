// Pointwise repair only; no evolution or production-Z4c campaign.
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include "z4c/hyperboloidal/algebraic_projection.hpp"
#include "z4c/hyperboloidal/hyperbolic_composite.hpp"

namespace h = z4c::hyperboloidal;
using State = std::array<Real,31>;

static void Require(bool good, const char *message) {
  if (!good) throw std::runtime_error(message);
}
static void Near(Real a, Real b, Real tol, const char *message) {
  Require(std::isfinite(a) && std::abs(a-b) <= tol*(1+std::abs(b)),message);
}
static void Pack(State &v, const Real g[3][3], const Real a[3][3]) {
  for (int t = 0; t < 6; ++t) {
    const int i = h::HyperbolicComposite::ti[t], j = h::HyperbolicComposite::tj[t];
    v[1+t] = g[i][j]-(i == j); v[7+t] = a[i][j];
  }
}
static void Rotate(const Real r[3][3], const Real in[3][3], Real out[3][3]) {
  for (int i = 0; i < 3; ++i) for (int j = i; j < 3; ++j) {
    Real value = 0;
    for (int k = 0; k < 3; ++k) for (int l = 0; l < 3; ++l)
      value += r[i][k]*r[j][l]*in[k][l];
    out[i][j] = out[j][i] = value;
  }
}

static void TestTensors() {
  Real worst = 0;
  for (int trial = 0; trial < 40; ++trial) {
    const Real t = .13*(trial+1), c = std::cos(t), s = std::sin(t);
    const Real c2 = std::cos(.7*t), s2 = std::sin(.7*t);
    const Real rotation[3][3] = {{c*c2,-s,c*s2},{s*c2,c,s*s2},{-s2,0,c2}};
    Real g[3][3] = {{1.4+.01*trial,0,0},{0,.7,0},{0,0,1.2}};
    Real a[3][3] = {{.3,.07,-.04},{.07,-.13,.09},{-.04,.09,.2}};
    Real rg[3][3], ra[3][3]; Rotate(rotation,g,rg); Rotate(rotation,a,ra);
    const Real det = g[0][0]*g[1][1]*g[2][2];
    const Real trace = a[0][0]/g[0][0]+a[1][1]/g[1][1]+a[2][2]/g[2][2];
    const Real scale = 1/std::cbrt(det);
    Real expected_g[3][3], expected_a[3][3];
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
      expected_g[i][j] = scale*g[i][j]; expected_a[i][j] = a[i][j]-g[i][j]*trace/3;
    }
    const auto result = h::ProjectAlgebraicTensors(g,a);
    Require(result.valid && result.before.valid && result.after.valid,"SPD repair rejected");
    Near(result.before.determinant,det,3e-15,"initial determinant diagnostic");
    Near(result.before.trace,trace,3e-15,"initial trace diagnostic");
    Near(result.scale,scale,3e-15,"normalization scale");
    Near(result.after.determinant,1,3e-15,"unit determinant");
    Near(result.after.trace,0,3e-15,"traceless curvature");
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
      Near(g[i][j],expected_g[i][j],3e-15,"ordinary metric repair");
      Near(a[i][j],expected_a[i][j],3e-15,"ordinary A repair must not rescale A");
    }
    Real rot_g[3][3], rot_a[3][3]; Rotate(rotation,g,rot_g); Rotate(rotation,a,rot_a);
    Require(h::ProjectAlgebraicTensors(rg,ra).valid,"rotated SPD repair rejected");
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
      worst = std::max({worst,std::abs(rg[i][j]-rot_g[i][j]),std::abs(ra[i][j]-rot_a[i][j])});
      Near(rg[i][j],rot_g[i][j],4e-15,"rotation covariance metric");
      Near(ra[i][j],rot_a[i][j],4e-15,"rotation covariance A");
    }
    Require(h::ProjectAlgebraicTensors(g,a).valid,"idempotent repair rejected");
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
      Near(g[i][j],expected_g[i][j],4e-15,"idempotent metric");
      Near(a[i][j],expected_a[i][j],4e-15,"idempotent A");
    }
  }
  std::cout << "rotation_covariance_max=" << worst << '\n';
}

static void TestRejection() {
  for (int test = 0; test < 10; ++test) {
    Real g[3][3] = {{1,0,0},{0,1,0},{0,0,1}}, a[3][3]{};
    if (test == 0) g[0][0] = g[1][1] = -1;  // positive det is not sufficient
    if (test == 1) g[2][2] = 0;
    if (test == 2) g[2][2] = -1;
    if (test == 3) g[0][0] = std::numeric_limits<Real>::quiet_NaN();
    if (test == 4) a[1][1] = std::numeric_limits<Real>::infinity();
    if (test == 5) g[0][1] = .01;
    if (test == 6) a[0][1] = .01;
    if (test == 7) g[0][0] = g[1][1] = g[2][2] = 1e200;
    if (test == 8) a[0][0] = a[1][1] = a[2][2] = 1e308;
    if (test == 9) {
      g[0][0] = 1e150; g[1][1] = g[2][2] = 1e-75;
      a[1][1] = 1e125;  // finite input/trace, overflowing projected A_xx
    }
    Real old_g[3][3], old_a[3][3];
    std::memcpy(old_g,g,sizeof g); std::memcpy(old_a,a,sizeof a);
    const auto result = h::ProjectAlgebraicTensors(g,a);
    Require(!result.valid && result.status != h::AlgebraicProjectionStatus::success,
            "invalid tensors admitted");
    Require(std::memcmp(old_g,g,sizeof g) == 0 && std::memcmp(old_a,a,sizeof a) == 0,
            "failed repair partially changed tensors");
  }
  State bad{}, compensation{}; bad[1] = bad[4] = -2;
  for (int f = 0; f < 31; ++f) compensation[f] = f+.125;
  const auto before = bad, old_compensation = compensation;
  Require(!h::ProjectAlgebraicNativeState(bad.data(),compensation.data()).valid,
          "invalid native metric admitted");
  Require(bad == before && compensation == old_compensation,"failed native repair modified state");
  State unrepresentable{}; unrepresentable[1] = -.9999999999999999;
  unrepresentable[4] = unrepresentable[6] = 1e16;
  const auto old_unrepresentable = unrepresentable;
  const auto packed = h::ProjectAlgebraicNativeState(unrepresentable.data(),compensation.data());
  Require(!packed.valid && packed.status == h::AlgebraicProjectionStatus::invalid_projected_metric,
          "unrepresentable packed diagonal admitted");
  Require(unrepresentable == old_unrepresentable && compensation == old_compensation,
          "packing rejection partially changed native state");
}

static void TestNativeDevice() {
  Kokkos::View<Real*[31]> data("algebraic native test",3), compensation("algebraic compensation",3);
  Kokkos::View<h::AlgebraicProjectionResult<Real>*> result("algebraic diagnostics",3);
  auto host = Kokkos::create_mirror_view(data), hc = Kokkos::create_mirror_view(compensation);
  State input[3];
  for (int p = 0; p < 3; ++p) for (int f = 0; f < 31; ++f) input[p][f] = .01*(f+1);
  const Real g[3][3] = {{1.3,.04,-.02},{.04,.8,.03},{-.02,.03,1.1}};
  const Real a[3][3] = {{.4,.03,.01},{.03,-.2,.02},{.01,.02,.05}};
  Pack(input[0],g,a);
  input[1].fill(0); input[1][0] = 1;  // exact stored CMC Minkowski reference
  for (int f = 1; f <= 12; ++f) input[2][f] = 0;
  input[2][1] = input[2][4] = -2;  // determinant positive, metric indefinite
  for (int p = 0; p < 3; ++p) for (int f = 0; f < 31; ++f) {
    host(p,f) = input[p][f]; hc(p,f) = .125+f;
  }
  Kokkos::deep_copy(data,host); Kokkos::deep_copy(compensation,hc);
  Kokkos::parallel_for("algebraic point projection",3,KOKKOS_LAMBDA(const int p) {
    // Local arrays work for every View layout; the helper assumes point-contiguous storage.
    Real point[31], residual[31];
    for (int f = 0; f < 31; ++f) { point[f] = data(p,f); residual[f] = compensation(p,f); }
    result(p) = h::ProjectAlgebraicNativeState(point,residual);
    for (int f = 0; f < 31; ++f) { data(p,f) = point[f]; compensation(p,f) = residual[f]; }
  });
  Kokkos::deep_copy(host,data); Kokkos::deep_copy(hc,compensation);
  const auto report = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),result);
  for (int p = 0; p < 3; ++p) {
    if (p == 2) {
      Require(!report(p).valid,"device admitted indefinite metric");
      for (int f = 0; f < 31; ++f)
        Require(host(p,f) == input[p][f] && hc(p,f) == .125+f,"device failure partially updated state");
      continue;
    }
    Require(report(p).valid,"device projection rejected");
    Near(report(p).after.determinant,1,3e-15,"device determinant");
    Near(report(p).after.trace,0,3e-15,"device trace");
    for (int f = 0; f < 31; ++f) {
      const bool projected = f >= 1 && f <= 12;
      if (!projected || p == 1) Require(host(p,f) == input[p][f],"unrelated field or Minkowski changed");
      Require(hc(p,f) == (projected ? 0 : .125+f),"wrong compensated-RK entry cleared");
    }
  }
  auto point = input[0];
  Require(h::ProjectAlgebraicNativeState(point.data()).valid,"optional compensation required");
  for (int f = 0; f < 31; ++f) Near(point[f],host(0,f),3e-15,"host/device projection mismatch");
}

static void TestScriReclosure() {
  h::HyperbolicComposite model(cubed_sphere::CompositeMesh(.125,{.45,.75,1},1,6,3));
  auto seed = [](const Real *x, Real *v) {
    v[0] = 1+.002*x[0];
    for (int f = 1; f < 31; ++f)
      v[f] = .001*(1+f%3)*std::sin((.7+.03*f)*x[0]-.3*x[1]+.2*x[2]);
  };
  for (size_t p = 0; p < model.grid.active.size(); ++p)
    seed(model.grid.geometry[p].x,&model.state[31*p]);
  for (size_t p = 0; p < model.scri.rows.size(); ++p)
    seed(model.scri.rows[p].geometry.x,&model.scri.values[31*p]);
  model.BoundaryValues();
  const auto old_state = model.state, old_surface = model.scri.values;
  for (size_t p = 0; p < model.grid.active.size(); ++p)
    Require(h::ProjectAlgebraicNativeState(&model.state[31*p]).valid,"volume repair rejected");
  Real stale_null = 0;
  for (size_t p = 0; p < model.scri.rows.size(); ++p) {
    Require(h::ProjectAlgebraicNativeState(&model.scri.values[31*p]).valid,"scri repair rejected");
    stale_null = std::max(stale_null,model.NullResidual(model.scri.rows[p].geometry.x,&model.scri.values[31*p]));
  }
  Require(stale_null > 1e-5,"test does not exercise required null reclosure");
  model.null_max = model.compatibility_max = 0;
  model.BoundaryValues();
  const auto padded = model.Padded(model.state);
  const auto rate = model.BoundaryRates(padded,.0001);
  Real null_rate = 0, d_change = 0;
  for (size_t p = 0; p < model.scri.rows.size(); ++p) {
    const auto *x = model.scri.rows[p].geometry.x;
    const auto sample = model.scri.Sample(p,padded,model.scri.values,31);
    const auto u = model.Jet(x,sample.value,sample.d,sample.dd);
    const auto geo = h::Geometry(u.metric);
    const auto B = model.Vector(19,sample.value,sample.d);
    const auto gauge = h::HyperbolicGauge(x,u,B,&sample.value[25]);
    Real q = 0, qdot = 0, normal = 0, normal_rate = 0, invn[3]{};
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j)
      invn[i] += geo.inverse[i][j]*x[j];
    for (int i = 0; i < 3; ++i) {
      q += x[i]*invn[i]; normal += x[i]*u.beta.value[i];
      Real beta_rate = gauge.shift_regular[i];
      for (int j = 0; j < 3; ++j) beta_rate += 4*x[j]*sample.d[22+i][j];
      normal_rate += x[i]*beta_rate;
      d_change = std::max(d_change,std::abs(model.scri.values[31*p+25+i]-old_surface[31*p+25+i]));
    }
    for (int t = 0; t < 6; ++t) {
      const int i = model.ti[t], j = model.tj[t];
      qdot -= (i == j ? 1 : 2)*invn[i]*rate[31*p+1+t]*invn[j];
    }
    null_rate = std::max(null_rate,std::abs(2*normal*normal_rate-rate[31*p]*q-u.chi.value*qdot));
    for (int f : {0,13,14,15,16,17})
      Require(model.scri.values[31*p+f] == old_surface[31*p+f],"scri chi/P/Theta/Lambda changed");
  }
  for (size_t p = 0; p < model.grid.active.size(); ++p) for (int f : {0,13,14,15,16,17})
    Require(model.state[31*p+f] == old_state[31*p+f],"volume chi/P/Theta/Lambda changed");
  Require(d_change > 1e-4,"dependent D was not recomputed");
  Require(model.null_max < 3e-14 && model.interpolated_null_max < 3e-14,
          "scri or ghost null closure lost after algebraic repair");
  Require(model.compatibility_max < 3e-13 && null_rate < 3e-12,
          "dependent D fails differentiated null relation");
  std::cout << "scri_null_before_reclosure=" << stale_null << " after=" << model.null_max
            << " ghost=" << model.interpolated_null_max << " D_compatibility=" << model.compatibility_max
            << " null_time_derivative=" << null_rate << '\n';
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc,argv); int result = 0;
  try {
    std::cout << std::setprecision(17) << "backend=" << Kokkos::DefaultExecutionSpace::name() << '\n';
    TestTensors(); TestRejection(); TestNativeDevice(); TestScriReclosure();
    std::cout << "algebraic projection: tensors, rejection, device, unchanged fields, compensation, scri passed\n";
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; result = 1; }
  Kokkos::finalize(); return result;
}
