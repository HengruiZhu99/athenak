// Axisymmetric finite-ordinate transport of collisionless massless particles.
#include <algorithm>
#include <climits>
#include <cmath>
#include <stdexcept>
#include <string>

#include "boltzmann/boltzmann.hpp"
#include "boltzmann/moments.hpp"
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"
#include "parameter_input.hpp"
#include "z4c/tmunu.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_vertex_topology.hpp"

namespace boltzmann {
Boltzmann::Boltzmann(MeshBlockPack* pp, ParameterInput* pin)
    : pack(pp),
      npsi(pin->GetOrAddInteger("boltzmann", "npsi", 16)),
      nmu_half(pin->GetOrAddInteger("boltzmann", "nmu_half", 4)),
      transport_order(pin->GetOrAddInteger("boltzmann", "transport_order", 2)),
      flux("boltzmann_flux", 1, 1, 1, 1, 1) {
  const auto sign = pin->GetOrAddString("boltzmann", "sign", "prograde");
  if (sign != "prograde" && sign != "retrograde" && sign != "both")
    throw std::invalid_argument("boltzmann/sign must be prograde, retrograde, or both");
  if (npsi < 4 || npsi % 2 || nmu_half < 1 || nmu_half > INT_MAX / npsi / 2)
    throw std::invalid_argument("invalid boltzmann angular grid (npsi must be even)");
  for (int f = 1; f <= 3; ++f) {
    auto bc = pp->pmesh->mesh_bcs[f];
    if (bc != BoundaryFlag::outflow && !(f >= 2 && bc == BoundaryFlag::periodic))
      throw std::invalid_argument(
          "Boltzmann transport requires open outer boundaries or periodic z");
  }
  const auto rk = pin->GetOrAddString("time", "integrator", "rk2");
  if (rk != "rk1" && rk != "rk2" && rk != "rk3" && rk != "ssprk104")
    throw std::invalid_argument(
        "positive Boltzmann transport requires SSP rk1, rk2, rk3 or ssprk104");
  backreaction = pin->GetOrAddBoolean("boltzmann", "backreaction", true);
  fixed_metric = pin->GetOrAddBoolean("boltzmann", "fixed_metric", false);
  if (fixed_metric && backreaction)
    throw std::invalid_argument("a prescribed metric requires boltzmann/backreaction=false");
  nmu = nmu_half * (sign == "both" ? 2 : 1);
  mu_min = sign == "prograde" ? 0 : -1;
  nang = nmu * npsi;
  dmu = 1. / nmu_half;
  dpsi = 2 * M_PI / npsi;
  auto& b = pp->pmesh->mb_indcs;
  int cap = std::max(pp->nmb_thispack, pp->pmesh->nmb_maxperrank);
  int nj = b.nx2 + 2 * b.ng, ni = b.nx1 + 2 * b.ng;
  if (transport_order != 2 && transport_order != 4)
    throw std::invalid_argument("boltzmann/transport_order must be 2 or 4");
  if (transport_order == 4) {
    if (pp->pmesh->multilevel)
      throw std::invalid_argument("FV4 donor budgets require a uniform-level mesh");
    if (b.ng < 4 || nmu_half < 5 || npsi < 6)
      throw std::invalid_argument("FV4 needs four ghosts, five cosine bins and six sectors");
    for (auto& a : speeds)
      a = DvceArray5D<Real>("boltzmann characteristic", cap, nang, 1, nj, ni);
    polar_rules = DvceArray1D<PolarQuadrature>("colatitude integration", nmu);
    auto rules = Kokkos::create_mirror_view(polar_rules);
    for (int k = 0; k < nmu; ++k) rules(k) = MakePolarQuadrature(k, nmu_half, mu_min, dmu);
    Kokkos::deep_copy(polar_rules, rules);
    for (auto& a : polar_flux_weights)
      a = DvceArray5D<Real>("polar flux coefficient", cap, nang, 1, nj, ni);
    inverse_budget = DvceArray5D<Real>("boltzmann donor budget", cap, nang, 1, nj, ni);
  }
  u = DvceArray5D<Real>("boltzmann_u", cap, nang, 1, nj, ni);
  accumulator = DvceArray5D<Real>("boltzmann_u1", cap, nang, 1, nj, ni);
  moments = DvceArray5D<Real>("boltzmann_moments", cap, 10, 1, nj, ni);
  angular_mu = DvceArray5D<Real>("boltzmann_mu_flux", cap, nang, 1, nj, ni);
  angular_psi = DvceArray5D<Real>("boltzmann_psi_flux", cap, nang, 1, nj, ni);
  source = DvceArray5D<Real>("boltzmann_redshift", cap, nang, 1, nj, ni);
  face_geometry = DvceArray4D<FaceGeometry>("boltzmann_face_geometry", cap, 2, nj, ni);
  geometry = DvceArray3D<Geometry>("boltzmann_geometry", cap, nj, ni);
  Kokkos::realloc(flux.x1f, cap, nang, 1, nj, ni);
  Kokkos::realloc(flux.x2f, cap, nang, 1, nj, ni);
  if (pp->pmesh->multilevel) {
    coarse = DvceArray5D<Real>("boltzmann_coarse", cap, nang, 1, b.cnx2 + 2 * b.ng,
                               b.cnx1 + 2 * b.ng);
  }
  boundary = new MeshBoundaryValuesCC(pp, pin, false);
  boundary->InitializeBuffers(nang);
  weights = DvceArray2D<Real>("boltzmann_weights", nang, 10);
  auto w = Kokkos::create_mirror_view(weights);
  // Integrate the directional stress of a constant intensity exactly over each bin.
  const auto circle = [](Real x) { return 0.5 * (x * sqrt(fmax(0., 1 - x * x)) + asin(x)); };
  for (int n = 0; n < nang; ++n) {
    Real lo = mu_min + (n / npsi) * dmu, hi = fmin(1., lo + dmu);
    Real a = (n % npsi) * dpsi, bphi = a + dpsi;
    Real mu = (lo + hi) / 2, mu2 = (lo * lo + lo * hi + hi * hi) / 3;
    Real s = (circle(hi) - circle(lo)) / dmu;
    Real smu = (pow(fmax(0., 1 - lo * lo), 1.5) - pow(fmax(0., 1 - hi * hi), 1.5)) / (3 * dmu);
    Real c = (sin(bphi) - sin(a)) / dpsi, sn = (cos(a) - cos(bphi)) / dpsi;
    Real c2 = 0.5 + (sin(2 * bphi) - sin(2 * a)) / (4 * dpsi);
    w(n, 0) = (1 - mu2) * c2;
    w(n, 1) = (1 - mu2) * (cos(2 * a) - cos(2 * bphi)) / (4 * dpsi);
    w(n, 2) = smu * c;
    w(n, 3) = (1 - mu2) * (1 - c2);
    w(n, 4) = smu * sn;
    w(n, 5) = mu2;
    w(n, 6) = 1;
    w(n, 7) = s * c;
    w(n, 8) = s * sn;
    w(n, 9) = mu;
  }
  Kokkos::deep_copy(weights, w);
}

Boltzmann::~Boltzmann() { delete boundary; }

TaskStatus Boltzmann::GeometryAndStress(Driver* driver, int stage) {
  if (fixed_metric && speed_cache_valid && stage > 0 && stage < driver->nexp_stages)
    return TaskStatus::complete;
  const auto adm = pack->padm->adm;
  const auto b = pack->pmesh->mb_indcs;
  const auto sizes = pack->pmb->mb_size.d_view;
  const auto geo = geometry;
  const auto stress = moments;
  const auto state = u;
  const auto w = weights;
  const int count = nang, last = pack->nmb_thispack - 1;
  const bool fourth = transport_order == 4;
  const Real domega = dmu * dpsi;
  par_for(
      "boltzmann cell moments", DevExeSpace(), 0, last, b.js - 1, b.je + 1, b.is - 1, b.ie + 1,
      KOKKOS_LAMBDA(int m, int j, int i) {
        const auto& size = sizes(m);
        Real rho = size.x1min + (i - b.is + 0.5) * size.dx1;
        Geometry g{};
        g.Set(adm, m, j, i, rho, size.dx1, size.dx2, fourth);
        geo(m, j, i) = g;
        Real avg[10]{}, energy_sum = 0, compensation = 0;
        for (int n = 0; n < count; ++n) {
          Real value = state(m, n, 0, j, i) * domega / g.frame.volume;
          for (int l = 0; l < 10; ++l) avg[l] += value * w(n, l);
          if (fourth) {
            // Broad isotropic populations can occupy thousands of similar bins.
            const Real y = value - compensation, next = energy_sum + y;
            compensation = (next - energy_sum) - y;
            energy_sum = next;
          }
        }
        stress(m, 6, 0, j, i) = fourth ? energy_sum : avg[6];
        const int sym[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
        for (int a = 0; a < 3; ++a) {
          Real s = 0;
          Real scale_a = a == 2 ? rho : 1;
          for (int p = a; p < 3; ++p) s += g.frame.co[a][p] * avg[7 + p];
          stress(m, 7 + a, 0, j, i) = s / scale_a;
          for (int k = a; k < 3; ++k) {
            Real t = 0;
            for (int p = a; p < 3; ++p)
              for (int q = k; q < 3; ++q)
                t += g.frame.co[a][p] * g.frame.co[k][q] * avg[sym[p][q]];
            stress(m, sym[a][k], 0, j, i) = t / (scale_a * (k == 2 ? rho : 1));
          }
        }
      });
  if (!backreaction) return TaskStatus::complete;
  const auto v = pack->pz4c->layout;
  auto target = pack->ptmunu->u_tmunu;
  par_for(
      "boltzmann vertex source", DevExeSpace(), 0, last, v.js, v.je, v.is, v.ie,
      KOKKOS_LAMBDA(int m, int j, int i) {
        TransferVertex(stress, target, m, j, i, sizes(m).x1min == 0 && i == v.is);
      });
  // Coincident coarse and fine vertices must see the same RHS source.
  pack->pz4c->vertex_topology_plan->SynchronizeSharedNodes(target);
  return TaskStatus::complete;
}

void Boltzmann::InitializeRing(ParameterInput* pin) {
  const Real r0 = pin->GetOrAddReal("problem", "ring_radius", 8);
  const Real width = pin->GetOrAddReal("problem", "ring_width", 1);
  const Real tilt = pin->GetOrAddReal("problem", "ring_mu", 0.35);
  const Real spread = pin->GetOrAddReal("problem", "angular_width", 0.2);
  const Real amplitude = pin->GetOrAddReal("problem", "intensity", 0.001);
  if (!(width > 0 && spread > 0 && fabs(tilt) < 1 && amplitude >= 0 && r0 > 0))
    throw std::invalid_argument("invalid ring parameters");
  const auto state = u;
  const auto sizes = pack->pmb->mb_size.d_view;
  const auto b = pack->pmesh->mb_indcs;
  const auto adm = pack->padm->adm;
  const int numpsi = npsi;
  const Real hm = dmu, hp = dpsi, start = mu_min;
  par_for(
      "initialize boltzmann ring", DevExeSpace(), 0, pack->nmb_thispack - 1, 0, nang - 1, b.js,
      b.je, b.is, b.ie, KOKKOS_LAMBDA(int m, int n, int j, int i) {
        const auto& sz = sizes(m);
        Real r = sz.x1min + (i - b.is + 0.5) * sz.dx1, z = sz.x2min + (j - b.js + 0.5) * sz.dx2;
        Real mu = start + (n / numpsi + 0.5) * hm, psi = (n % numpsi + 0.5) * hp;
        Real dot = tilt * mu - sqrt((1 - tilt * tilt) * (1 - mu * mu)) * cos(psi);
        Real g[3][3];
        for (int a = 0; a < 3; ++a)
          for (int c = 0; c < 3; ++c) g[a][c] = adm.g_dd(m, a, c, 0, j, i);
        Frame frame{};
        frame.Set(g);
        state(m, n, 0, j, i) = amplitude * r * frame.volume *
                               exp(-0.5 * ((r - r0) * (r - r0) + z * z) / (width * width) +
                                   (dot - 1) / (spread * spread));
      });
}

void RingHistory(HistoryData* data, Mesh* mesh) {
  auto pack = mesh->pmb_pack;
  const auto b = mesh->mb_indcs;
  const auto sz = pack->pmb->mb_size.d_view;
  const auto matter = pack->pboltzmann;
  const auto moments = matter->moments;
  const auto adm = pack->padm->adm;
  // Integrate outside the last accepted star-shaped m=0 surface(s). FastFlow
  // owns the centers and normalized Y_l0 coefficients, including after restart.
  const auto& h = pack->z4c_restart_state.fastflow;
  int surfaces = h.converged ? h.center_count : 0;
  int modes = surfaces ? static_cast<int>(h.coefficients.size()) / surfaces : 0;
  Real centers[2] = {h.center_z0, h.center_z1};
  DvceArray1D<Real> coeff("boltzmann horizon coefficients", surfaces * modes);
  if (surfaces) {
    auto host = Kokkos::create_mirror_view(coeff);
    for (int n = 0; n < surfaces * modes; ++n) host(n) = h.coefficients[n];
    Kokkos::deep_copy(coeff, host);
  }
  const auto state = matter->u;
  Real minimum;
  Kokkos::parallel_reduce(
      "boltzmann minimum ordinate",
      Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<4>>(
          {0, 0, b.js, b.is}, {pack->nmb_thispack, matter->nang, b.je + 1, b.ie + 1}),
      KOKKOS_LAMBDA(int m, int n, int j, int i, Real& local) {
        local = fmin(local, state(m, n, 0, j, i));
      }, Kokkos::Min<Real>(minimum));
  data->nhist = 7;
  data->label[6] = "min_photon";
  data->hdata[6] = minimum;
  data->reduction[6] = HistoryData::Reduction::min;
  data->label[5] = "photon_Jext";
  data->label[0] = "photon_E";
  data->label[1] = "photon_Jz";
  data->label[2] = "escaped_E";
  data->label[3] = "escaped_Jz";
  data->label[4] = "metric_work";
  for (int k = 0; k < 3; ++k) {
    Real total = 0;
    Kokkos::parallel_reduce(
        "boltzmann history",
        Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<3>>(
            {0, b.js, b.is}, {pack->nmb_thispack, b.je + 1, b.ie + 1}),
        KOKKOS_LAMBDA(int m, int j, int i, Real& sum) {
          Real r = sz(m).x1min + (i - b.is + 0.5) * sz(m).dx1;
          Real g[3][3];
          for (int a = 0; a < 3; ++a)
            for (int c = 0; c < 3; ++c) g[a][c] = adm.g_dd(m, a, c, 0, j, i);
          Frame f{};
          f.Set(g);
          if (k == 2) {
            Real z = sz(m).x2min + (j - b.js + 0.5) * sz(m).dx2;
            for (int surface = 0; surface < surfaces; ++surface) {
              Real axial = z - centers[surface], rad = sqrt(r * r + axial * axial);
              Real ct = axial / rad, prev = 0, legendre = 1, horizon = 0;
              for (int l = 0; l < modes; ++l) {
                horizon +=
                    coeff(surface * modes + l) * sqrt((2 * l + 1) / (4 * M_PI)) * legendre;
                Real next = ((2 * l + 1) * ct * legendre - l * prev) / (l + 1);
                prev = legendre;
                legendre = next;
              }
              if (rad <= horizon) return;
            }
          }
          Real density = k == 0 ? moments(m, 6, 0, j, i) : r * moments(m, 9, 0, j, i);
          sum += 2 * M_PI * r * f.volume * sz(m).dx1 * sz(m).dx2 * density;
        },
        total);
    data->hdata[k == 2 ? 5 : k] = total;
  }
  data->hdata[2] = matter->escaped;
  data->hdata[3] = matter->escaped_j;
  data->hdata[4] = matter->work;
}
}  // namespace boltzmann
