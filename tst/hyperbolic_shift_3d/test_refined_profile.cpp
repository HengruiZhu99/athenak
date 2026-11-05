#include <mpi.h>
#include <iostream>
#include <iomanip>
#include "globals.hpp"
#include "refined_configuration.hpp"
#include "z4c/hyperboloidal/cartesian_amr.hpp"
#include "z4c/hyperboloidal/athenak_bridge.hpp"
#include "z4c/hyperboloidal/native_boosted_wormhole.hpp"
namespace h = z4c::hyperboloidal;
using Z = z4c::Z4c;
void Check(bool good, const char* message) {
  if (!good)
    throw std::runtime_error(message);
}

// Compare with the actual reference Cartesian Z4c::CalcRHS, not a duplicate
// implementation or an equilibrium-only cancellation. The data are off shell,
// with nonzero Theta/Lambda, transport, lapse/chi Hessians and A/B gradients.
double ReferenceInner(const h::HyperbolicShiftParameters<Real>& parameters, bool variable_metric,
                      double radius, bool below_guard=false) {
  // 64 roots support the production 12 ranks per node in GPU qualification.
  h::CartesianAMR mesh(8, 4, 64, 1);
  auto& input = mesh.input;
  input.SetString("time", "evolution", "dynamic");
  input.SetRealExact("z4c", "damp_kappa1", .03);
  input.SetRealExact("z4c", "damp_kappa2", 0);
  input.SetRealExact("z4c", "shift_Gamma", parameters.inner_gamma);
  input.SetRealExact("z4c", "shift_eta", 6);
  input.SetRealExact("z4c", "diss", 0);
  if(parameters.cartesian_chi_guard)
    input.SetRealExact("z4c", "chi_div_floor", parameters.inner_chi_guard);
  input.SetBoolean("z4c", "telegraph_lapse", true);
  input.SetRealExact("z4c", "telegraph_tau", 1. / 30);
  input.SetRealExact("z4c", "telegraph_kappa", 1. / 15);
  auto* pack = mesh.mesh->pmb_pack;
  pack->pz4c = new Z(pack, &input);
  auto& reference = *pack->pz4c;
  reference.diss = 0;
  auto host = Kokkos::create_mirror_view(reference.u0);
  for (int m = 0; m < mesh.mesh->nmb_thisrank; ++m)
    for (int k = 0; k < 16; ++k)
      for (int j = 0; j < 16; ++j)
        for (int i = 0; i < 16; ++i) {
          const double x = (i - 4 + .5) * .01125 - .045, y = (j - 4 + .5) * .01125 - .045,
                       z = (k - 4 + .5) * .01125 - .045;
          for (int f = 0; f < Z::nz4c; ++f)
            host(m, f, k, j, i) = 0;
          host(m, Z::I_Z4C_CHI, k, j, i) = .8 + .05 * x + .03 * y * y + .02 * x * z;
          if(below_guard)host(m, Z::I_Z4C_CHI, k, j, i)*=1e-9;
          host(m, Z::I_Z4C_ALPHA, k, j, i) = .7 + .04 * y + .08 * x * x + .03 * y * z;
          host(m, Z::I_Z4C_KHAT, k, j, i) = .12 + .04 * z;
          host(m, Z::I_Z4C_THETA, k, j, i) = .015 + .008 * x;
          host(m, Z::I_Z4C_GXX, k, j, i) = 1.2;
          host(m, Z::I_Z4C_GYY, k, j, i) = 1;
          host(m, Z::I_Z4C_GZZ, k, j, i) = 1 / 1.2;
          const double ax = .02 + .03 * x, az = .015 + .02 * y;
          host(m, Z::I_Z4C_AXX, k, j, i) = ax;
          host(m, Z::I_Z4C_AYY, k, j, i) = -ax / 1.2 - 1.2 * az;
          host(m, Z::I_Z4C_AZZ, k, j, i) = az;
          host(m, Z::I_Z4C_AXY, k, j, i) = .011 + .01 * z;
          host(m, Z::I_Z4C_AXZ, k, j, i) = .007 * x;
          host(m, Z::I_Z4C_AYZ, k, j, i) = -.003 * y;
          if (variable_metric) {
            // Polynomial shear: det(g)=1 exactly, with nonzero first/second metric
            // derivatives. Polynomial fields remove FD truncation as a confounder.
            const double q = .2 * x + .1 * y;
            host(m, Z::I_Z4C_GXX, k, j, i) = 1 + q * q;
            host(m, Z::I_Z4C_GXY, k, j, i) = q;
            host(m, Z::I_Z4C_GZZ, k, j, i) = 1;
            host(m, Z::I_Z4C_AXX, k, j, i) = -(1 + q * q) * host(m, Z::I_Z4C_AYY, k, j, i) - az +
                                             2 * q * host(m, Z::I_Z4C_AXY, k, j, i);
          }
          for (int a = 0; a < 3; ++a) {
            host(m, Z::I_Z4C_GAMX + a, k, j, i) = .1 * (a - 1) + .04 * x - .03 * y + .02 * z;
            host(m, Z::I_Z4C_BETAX + a, k, j, i) = .03 * (a + 1) + .02 * x - .01 * y + .015 * z * z;
            host(m, Z::I_Z4C_BX + a, k, j, i) = .005 * (a + 1) + .01 * x + .015 * y * y;
          }
        }
  Kokkos::deep_copy(reference.u0, host);
  reference.CalcRHS<3>(nullptr, 0);
  Kokkos::View<Real*> expected("constant Omega reference comparison", Z::nz4c);
  const auto fields = reference.z4c;
  const auto p = parameters;
  Kokkos::parallel_for(
      "nontrivial constant conformal frame", 1, KOKKOS_LAMBDA(int) {
        constexpr int k = 8, j = 7, i = 8;
        const Real idx[3] = {1 / .01125, 1 / .01125, 1 / .01125};
        const Real x[3] = {.36 * radius, .48 * radius, .8 * radius};
        auto u = h::LoadMeshJet<3>(fields, idx, 0, k, j, i);
        const auto physical = u;
        auto o = p.profile.At(x);
        const Real omega = o.omega;
        u.chi.value /= omega * omega;
        u.alpha.value *= omega;
        for (int a = 0; a < 3; ++a) {
          u.chi.d[a] = (physical.chi.d[a] - 2 * physical.chi.value * o.gradient[a] / omega) /
                       (omega * omega);
          u.alpha.d[a] = omega * physical.alpha.d[a] + physical.alpha.value * o.gradient[a];
          for (int b = 0; b < 3; ++b) {
            u.chi.dd[a][b] =
                (physical.chi.dd[a][b] -
                 2 *
                     (physical.chi.d[a] * o.gradient[b] + physical.chi.d[b] * o.gradient[a] +
                      physical.chi.value * o.hessian[a][b]) /
                     omega +
                 6 * physical.chi.value * o.gradient[a] * o.gradient[b] / (omega * omega)) /
                (omega * omega);
            u.alpha.dd[a][b] =
                omega * physical.alpha.dd[a][b] + physical.alpha.d[a] * o.gradient[b] +
                physical.alpha.d[b] * o.gradient[a] + physical.alpha.value * o.hessian[a][b];
            u.a.k[a][b] /= omega;
            for (int c = 0; c < 3; ++c)
              u.a.dk[c][a][b] =
                  (physical.a.dk[c][a][b] - physical.a.k[a][b] * o.gradient[c] / omega) / omega;
          }
        }
        h::VectorJet<Real> B{};
        Real D[3]{};
        for (int a = 0; a < 3; ++a) {
          B.value[a] = fields.vB_d(0, a, k, j, i);
          for (int b = 0; b < 3; ++b)
            B.d[b][a] = Dx<3>(b, idx, fields.vB_d, 0, a, k, j, i);
        }
        h::SetStationaryOmegaNormal(u.alpha.value, u.beta.value, u.alpha.d, u.beta.d, o);
        auto parts = h::ConformalRHS(u, o, Real(4), Real(-.99));
        h::Z4cRHS<Real> rate{};
        if (!h::HyperbolicDamping(x, u, o, parts, p) ||
            !h::AssembleInterior(parts, o.omega, rate)) {
          expected(0) = 1e100;
          return;
        }
        const auto gauge = h::HyperbolicGauge(x, u, B, D, p);
        expected(Z::I_Z4C_CHI) = rate.chi * omega * omega;
        expected(Z::I_Z4C_KHAT) = rate.trace;
        expected(Z::I_Z4C_THETA) = rate.theta;
        expected(Z::I_Z4C_ALPHA) = (gauge.lapse_regular + gauge.lapse_pole / omega) / omega;
        constexpr int ai[6] = {0, 0, 0, 1, 1, 2}, aj[6] = {0, 1, 2, 1, 2, 2};
        for (int f = 0; f < 6; ++f) {
          expected(Z::I_Z4C_GXX + f) = rate.metric[ai[f]][aj[f]];
          expected(Z::I_Z4C_AXX + f) = rate.a[ai[f]][aj[f]] * omega;
        }
        for (int a = 0; a < 3; ++a) {
          expected(Z::I_Z4C_GAMX + a) = rate.lambda[a];
          expected(Z::I_Z4C_BETAX + a) = gauge.shift_regular[a] + gauge.shift_pole[a] / omega;
          expected(Z::I_Z4C_BX + a) = gauge.flux_rate[a];
        }
      });
  const auto exact = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), expected);
  const auto actual = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), reference.u_rhs);
  double error = 0;
  for (int f = 0; f < Z::nz4c; ++f) {
    const double difference = std::abs(exact(f) - actual(0, f, 8, 7, 8));
    error = std::max(error, difference);
    if (difference > 2e-9)
      std::cerr << std::setprecision(17) << Z::Z4c_names[f] << " conformal=" << exact(f)
                << " reference=" << actual(0, f, 8, 7, 8) << "\n";
  }
  Check(error < 2e-9, "nontrivial inner/transition reference Z4c disagreement");
  return error;
}

// The Hamiltonian and physical momentum norm must be independent of the
// arbitrary conformal frame, including in the transition where dOmega != 0.
double FrameConstraintOracle(const h::ConformalProfile& profile) {
  double error = 0;
  for (double r : {.05, .175, .183, .195, .207, .219, .23, .5, .9}) {
    const Real x[3] = {.36 * r, .48 * r, .8 * r};
    const auto old = h::ConformalProfile{}.At(x), now = profile.At(x);
    h::Z4cJet<Real> u{};
    u.chi.value = .9 + .1 * x[0];
    u.chi.d[0] = .1;
    u.alpha.value = .7;
    u.trace.value = .2;
    u.theta.value = .013;
    u.trace.d[1] = .02;
    for (int i = 0; i < 3; ++i) {
      u.metric.g[i][i] = 1;
      u.a.k[i][i] = .01 * (i - 1);
      u.lambda.value[i] = .03 * (i + 1);
      u.beta.value[i] = .02 * (i + 1);
    }
    u.a.k[0][1] = u.a.k[1][0] = .01 * x[2];
    u.a.dk[2][0][1] = u.a.dk[2][1][0] = .01;
    auto v = u;
    const Real f = now.omega / old.omega, scale = 1 / (f * f);
    Real l[3]{}, ll[3][3]{};
    for (int i = 0; i < 3; ++i) {
      l[i] = now.gradient[i] / now.omega - old.gradient[i] / old.omega;
      for (int j = 0; j < 3; ++j)
        ll[i][j] = now.hessian[i][j] / now.omega -
                   now.gradient[i] * now.gradient[j] / (now.omega * now.omega) -
                   old.hessian[i][j] / old.omega +
                   old.gradient[i] * old.gradient[j] / (old.omega * old.omega);
    }
    v.chi.value *= scale;
    v.alpha.value *= f;
    for (int i = 0; i < 3; ++i) {
      v.chi.d[i] = scale * (u.chi.d[i] - 2 * u.chi.value * l[i]);
      for (int j = 0; j < 3; ++j) {
        v.chi.dd[i][j] = scale * (u.chi.dd[i][j] - 2 * u.chi.d[i] * l[j] - 2 * u.chi.d[j] * l[i] +
                                  u.chi.value * (4 * l[i] * l[j] - 2 * ll[i][j]));
        v.a.k[i][j] = u.a.k[i][j] / f;
        for (int d = 0; d < 3; ++d)
          v.a.dk[d][i][j] = (u.a.dk[d][i][j] - u.a.k[i][j] * l[d]) / f;
      }
    }
    const auto a = h::EvolvedConstraints(u, old), b = h::EvolvedConstraints(v, now);
    Check(a.valid && b.valid, "conformal constraint audit invalid");
    error = std::max(error, std::abs(a.hamiltonian - b.hamiltonian));
    error = std::max(error, std::abs(a.momentum_physical_norm2 - b.momentum_physical_norm2));
    for (int i = 0; i < 3; ++i)
      error = std::max(error, std::abs(a.momentum[i] - b.momentum[i]));
  }
  Check(error < 2e-12, "physical constraints depend on conformal frame");
  return error;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int code = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &global_variable::nranks);
  try {
    Check(argc == 2 || argc == 3, "supply configuration file and optional unsmoothed profile CSV");
    auto mesh = hyperbolic_reproduction::ContinuousCoreMesh(16);
    hyperbolic_reproduction::RefinedConfiguration c;
    c.Load(argv[1], mesh);
    c.CheckCollective();
    Check(c.gauge.profile.flat, "profile oracle requires flat configuration");
    Check(c.gauge.profile.flat_radius == c.inscribed_radius,
          "default is not the domain's inscribed sphere");
    Check(std::sqrt(3.) * c.core_half_width > c.gauge.profile.flat_radius,
          "cube corners incorrectly included");
    auto narrow = mesh;
    for (auto& b : narrow.blocks)
      if (b.face < 0) {
        b.lo[0] *= 17. / 18;
        b.hi[0] *= 17. / 18;
      }
    hyperbolic_reproduction::RefinedConfiguration nc;
    nc.Load(argv[1], narrow);
    Check(std::abs(nc.gauge.profile.flat_radius - (c.core_bounds_overridden?c.inscribed_radius:.17)) < 1e-15, "flat radius was hardcoded");
    double derivative_error = 0, map_error = 0, cartesian_derivative_error = 0;
    const auto profile = c.gauge.profile;
    for (int i = 0; i <= 100; ++i) {
      const double r =
          profile.flat_radius + (profile.outer_radius - profile.flat_radius) * (i + .25) / 101;
      const double delta = 1e-6;
      const auto q = profile.Radial(r), a = profile.Radial(r - delta),
                 b = profile.Radial(r + delta);
      derivative_error =
          std::max(derivative_error,
                   std::abs((b.value - a.value) / (2 * delta) - q.first) / (1 + std::abs(q.first)));
      derivative_error =
          std::max(derivative_error, std::abs((b.first - a.first) / (2 * delta) - q.second) /
                                         (1 + std::abs(q.second)));
      Check(q.value > 0 && q.value - r * q.first > 0, "degenerate radial compactification");
    }
    Check(derivative_error < 2e-6, "analytic profile derivative mismatch");
    for (double r : {.1, .183, .195, .207, .219, .3}) {
      Real x[3] = {.36 * r, .48 * r, .8 * r};
      const auto jet = profile.At(x);
      for (int a = 0; a < 3; ++a) {
        const double saved = x[a], eps = 1e-6;
        x[a] = saved + eps;
        const auto plus = profile.At(x);
        x[a] = saved - eps;
        const auto minus = profile.At(x);
        x[a] = saved;
        cartesian_derivative_error =
            std::max(cartesian_derivative_error,
                     std::abs((plus.omega - minus.omega) / (2 * eps) - jet.gradient[a]));
        for (int b = 0; b < 3; ++b)
          cartesian_derivative_error = std::max(
              cartesian_derivative_error,
              std::abs((plus.gradient[b] - minus.gradient[b]) / (2 * eps) - jet.hessian[a][b]) /
                  (1 + std::abs(jet.hessian[a][b])));
      }
    }
    Check(cartesian_derivative_error < 2e-6, "Cartesian profile Hessian mismatch");
    for (int bad = 0; bad < 4; ++bad) {
      auto invalid = profile;
      if (bad == 0)
        invalid.flat_radius = 0;
      if (bad == 1)
        invalid.outer_radius = invalid.flat_radius;
      if (bad == 2)
        invalid.plateau = .1;
      if (bad == 3)
        invalid.outer_radius = std::numeric_limits<double>::quiet_NaN();
      bool rejected = false;
      try {
        invalid.Validate();
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      Check(rejected, "invalid profile admitted");
    }
    std::stringstream checkpoint;
    c.WriteIdentity(checkpoint);
    c.ReadIdentity(checkpoint);
    std::stringstream wrong;
    c.WriteIdentity(wrong);
    auto other = c;
    other.identity += 'x';
    bool rejected = false;
    try {
      other.ReadIdentity(wrong);
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    Check(rejected, "different formulation checkpoint admitted");
    for (double r : {0., profile.flat_radius / 2, profile.flat_radius}) {
      const auto q = profile.Radial(r);
      Check(q.value == profile.plateau && q.first == 0 && q.second == 0, "nonconstant plateau");
    }
    for (double r : {profile.outer_radius, (profile.outer_radius+1)/2, .9, 1.}) {
      const auto q = profile.Radial(r);
      Check(q.value == (1 - r * r) / 2 && q.first == -r && q.second == -1,
            "outer/scri profile changed");
    }
    mesh = c.Mesh(mesh.cells);
    for (size_t b = 0; b < mesh.blocks.size(); ++b)
      for (double u : {.13, .5, .87}) {
        const Real q[3] = {u, .37, .63};
        Real x[3], back[3];
        mesh.Forward(b, q, x);
        mesh.Inverse(b, x, back);
        const auto g = mesh.Geometry(b, q);
        const auto& J = g.J;
        const double det = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) -
                           J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) +
                           J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
        Check(std::isfinite(det) && std::abs(det) > 1e-10, "singular mesh map");
        for (int a = 0; a < 3; ++a)
          map_error = std::max(map_error, std::abs(back[a] - q[a]));
      }
    Check(map_error < 2e-13, "map inverse mismatch");
    h::NativeBoostedWormhole native(c.boost_velocity,1./3.,true,c.initial_start_M,c.initial_end_M);
    for (double r : {.05, .17, .19, .21, .24, .3, .55, .8}) {
      const Real x[3] = {.6 * r, .8 * r, 0};
      auto old = native.Fields(x), now = old;
      c.InitialFields(x, now);
      const double old_o = (1 - r * r) / 2, new_o = profile.Radial(r).value, A = (1 + r * r) / 2;
      Check(std::abs(now[0] * new_o * new_o - old[0] * old_o * old_o) < 1e-12,
            "physical initial chi changed");
      Check(std::abs((now[18] + A) / new_o - (old[18] + A) / old_o) < 2e-13,
            "physical initial lapse changed");
      for (int f = 7; f < 13; ++f)
        Check(std::abs(now[f] * new_o - old[f] * old_o) < 1e-12,
              "physical initial curvature changed");
      for (int f = 1; f < 7; ++f)
        Check(now[f] == old[f], "initial conformal metric changed");
      for (int f = 15; f < 18; ++f)
        Check(now[f] == old[f], "initial Lambda changed");
    }
    const double frame_error = FrameConstraintOracle(profile);
    const double reference_error =
        std::max({ReferenceInner(c.gauge, false, .05), ReferenceInner(c.gauge, true, .05),
                  ReferenceInner(c.gauge, true, profile.flat_radius+.25*(profile.outer_radius-profile.flat_radius)), ReferenceInner(c.gauge, true, profile.flat_radius+.75*(profile.outer_radius-profile.flat_radius))});
    const auto central=h::CentralZ4cConfiguration{}.Resolve();
    const double guarded_reference_error=ReferenceInner(central.gauge,true,.05,true);
    if (argc == 3 && global_variable::my_rank == 0) {
      std::ofstream out(argv[2]);
      out << std::setprecision(17);
      out << "compact_r,omega,domega,ddomega,inner_gauge_weight,restoration_collar,background_"
             "areal_R\n";
      h::NativeWormhole background(1. / 3., 2e-13, c.initial_start_M, c.initial_end_M,
                                   h::NativeWormhole::TransitionProfile::polynomial_c6);
      for (int i = 1; i <= 1000; ++i) {
        const double r = i / 1000.;
        const auto o = profile.Radial(r);
        const auto g = h::ShiftCoefficients(r, 1., 1., c.gauge);
        out << r << ',' << o.value << ',' << o.first << ',' << o.second << ',' << g.inner << ','
            << g.collar << ',' << background.Point(r).areal << '\n';
      }
      Check(bool(out), "failed writing unsmoothed profiles");
    }
    if (global_variable::my_rank == 0)
      std::cout << std::setprecision(17)
                << "{\"passed\":true,\"flat_radius\":" << profile.flat_radius
                << ",\"k\":" << c.inner_map_scale << ",\"gamma\":" << c.gauge.inner_gamma
                << ",\"derivative_error\":" << derivative_error
                << ",\"frame_constraint_error\":" << frame_error
                << ",\"map_inverse_error\":" << map_error
                << ",\"reference_z4c_error\":" << reference_error
                << ",\"scaled_guard_z4c_error\":" << guarded_reference_error << "}\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    code = 1;
  }
  Kokkos::finalize();
  MPI_Finalize();
  return code;
}
