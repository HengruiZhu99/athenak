#include <limits>
#include "boltzmann/boltzmann.hpp"
#include "boltzmann/reconstruction.hpp"
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"

namespace boltzmann {

TaskStatus Boltzmann::Fluxes(Driver* driver, int stage) {
  if (transport_order == 4) return FluxesFourth(driver, stage);
  const auto sz = pack->pmb->mb_size.d_view;
  const auto b = pack->pmesh->mb_indcs;
  const auto native = pack->pz4c->adm;
  const auto g = geometry;
  const auto faces = face_geometry;
  const auto state = u, angular1 = angular_mu, angular2 = angular_psi, q = source;
  const int hemisphere = nmu_half, sectors = npsi, count = nang, blocks = pack->nmb_thispack;
  const Real dm = dmu, dp = dpsi, start = mu_min;
  for (int d = 0; d < 2; ++d) {
    const int di = d == 0, dj = d == 1;
    const auto face = d == 0 ? flux.x1f : flux.x2f;
    par_for(
        "boltzmann spatial flux", DevExeSpace(), 0, blocks - 1, b.js, b.je + dj, b.is,
        b.ie + di, KOKKOS_LAMBDA(int m, int j, int i) {
          const bool axis = di && i == b.is && sz(m).x1min == 0;
          Real metric[3][3];
          for (int a = 0; a < 3; ++a)
            for (int c = 0; c < 3; ++c) {
              metric[a][c] = 0.5 * (native.g_dd(m, a, c, 0, j, i) +
                                    native.g_dd(m, a, c, 0, j + di, i + dj));
            }
          Frame frame{};
          frame.Set(metric);
          const Real alpha =
              0.5 * (native.alpha(m, 0, j, i) + native.alpha(m, 0, j + di, i + dj));
          const Real beta =
              0.5 * (native.beta_u(m, d, 0, j, i) + native.beta_u(m, d, 0, j + di, i + dj));
          auto& f = faces(m, d, j, i);
          f.minus_shift = axis ? 0 : -beta;
          for (int a = 0; a < 3; ++a) f.triad[a] = axis ? 0 : alpha * frame.inverse[a][d];
          f.killing_norm = (sz(m).x1min + (i - b.is + 0.5 * dj) * sz(m).dx1) * frame.co[2][2];
        });
    par_for(
        "boltzmann spatial transport", DevExeSpace(), 0, blocks - 1, 0, count - 1, b.js,
        b.je + dj, b.is, b.ie + di, KOKKOS_LAMBDA(int m, int n, int j, int i) {
          Real mu = start + (n / sectors + 0.5) * dm, psi = (n % sectors + 0.5) * dp;
          Real mer = sqrt(1 - mu * mu), dir[3] = {mer * cos(psi), mer * sin(psi), mu};
          Real speed = faces(m, d, j, i).Speed(dir);
          int k = speed >= 0 ? i - di : i, l = speed >= 0 ? j - dj : j;
          face(m, n, 0, j, i) =
              speed * FaceState(state(m, n, 0, l - dj, k - di), state(m, n, 0, l, k),
                                state(m, n, 0, l + dj, k + di), speed >= 0 ? 1 : -1);
        });
  }
  // Each angular face is stored once. Periodic wrapping merely changes its index.
  Real max_rate = 0;
  Kokkos::parallel_reduce(
      "boltzmann angular flux and CFL",
      Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<4>>({0, 0, b.js, b.is},
                                                          {blocks, count, b.je + 1, b.ie + 1}),
      KOKKOS_LAMBDA(int m, int n, int j, int i, Real& maxval) {
        int band = n / sectors, sector = n % sectors;
        Real mu = start + (band + 0.5) * dm, psi = (sector + 0.5) * dp;
        const Geometry& cell = g(m, j, i);
        Rates center = Characteristics(cell, mu, psi);
        Real vmu = 0;
        if (band % hemisphere != 0) {
          vmu = Characteristics(cell, start + band * dm, psi).mu;
        }
        Real vpsi = Characteristics(cell, mu, sector * dp).psi;
        angular1(m, n, 0, j, i) = vmu == 0 ? 0
                                           : AngularFlux(state, m, j, i, band, sector,
                                                         hemisphere, sectors, true, vmu);
        angular2(m, n, 0, j, i) =
            AngularFlux(state, m, j, i, band, sector, hemisphere, sectors, false, vpsi);
        q(m, n, 0, j, i) = center.redshift * state(m, n, 0, j, i);
        // An outgoing PLM face state is at most twice its donor mean.
        // Sum the outgoing rates on this cell's four pairs of faces.
        Real mer = sqrt(1 - mu * mu), dir[3] = {mer * cos(psi), mer * sin(psi), mu};
        Real next_mu = (band + 1) % hemisphere == 0
                           ? 0
                           : Characteristics(cell, start + (band + 1) * dm, psi).mu;
        Real next_psi = Characteristics(cell, mu, ((sector + 1) % sectors) * dp).psi;
        Real out = (fmax(0., faces(m, 0, j, i + 1).Speed(dir)) +
                    fmax(0., -faces(m, 0, j, i).Speed(dir))) /
                       sz(m).dx1 +
                   (fmax(0., faces(m, 1, j + 1, i).Speed(dir)) +
                    fmax(0., -faces(m, 1, j, i).Speed(dir))) /
                       sz(m).dx2 +
                   (fmax(0., next_mu) + fmax(0., -vmu)) / dm +
                   (fmax(0., next_psi) + fmax(0., -vpsi)) / dp;
        Real rate = 2 * out + fmax(0., -center.redshift);
        maxval = fmax(maxval, rate);
      },
      Kokkos::Max<Real>(max_rate));
  dtnew = 1 / fmax(max_rate, std::numeric_limits<Real>::min());
  return TaskStatus::complete;
}

TaskStatus Boltzmann::Update(Driver* driver, int stage) {
  const auto sz = pack->pmb->mb_size.d_view;
  const auto b = pack->pmesh->mb_indcs;
  const auto state = u, old = accumulator, fx = flux.x1f, fz = flux.x2f;
  const auto fm = angular_mu, fp = angular_psi, q = source;
  const int bands = nmu, sectors = npsi, count = nang, blocks = pack->nmb_thispack;
  const Real dm = dmu, dp = dpsi;
  const Real dt = pack->pmesh->dt * driver->beta[stage - 1];
  const Real g0 = driver->gam0[stage - 1], g1 = driver->gam1[stage - 1];
  if (stage == 1) {
    Kokkos::deep_copy(DevExeSpace(), old, state);
    budget_old[0] = escaped;
    budget_old[1] = escaped_j;
    budget_old[2] = work;
  }
  const Real accumulate = driver->delta[stage - 1];
  if (driver->integrator == "ssprk104" && stage == 6) {
    budget_old[0] += accumulate * escaped;
    budget_old[1] += accumulate * escaped_j;
    budget_old[2] += accumulate * work;
    par_for(
        "boltzmann saved SSP stage", DevExeSpace(), 0, blocks - 1, 0, count - 1, b.js, b.je,
        b.is, b.ie, KOKKOS_LAMBDA(int m, int n, int j, int i) {
          old(m, n, 0, j, i) += accumulate * state(m, n, 0, j, i);
        });
  }
  auto flags = pack->pmb->mb_bcs.d_view;
  const auto faces = face_geometry;
  const Real start = mu_min;
  Kokkos::Array<Real, 3> rates{};
  for (int v = 0; v < 3; ++v) {
    Real total = 0;
    Kokkos::parallel_reduce(
        "boltzmann integrated sources and fluxes",
        Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<4>>(
            {0, 0, b.js, b.is}, {blocks, count, b.je + 1, b.ie + 1}),
        KOKKOS_LAMBDA(int m, int n, int j, int i, Real& sum) {
          auto& s = sz(m);
          Real val = q(m, n, 0, j, i) * s.dx1 * s.dx2;
          if (v < 2) {
            Real fr = 0, ft = 0, fb = 0;
            if (i == b.ie && flags(m, BoundaryFace::outer_x1) == BoundaryFlag::outflow)
              fr = fx(m, n, 0, j, i + 1);
            if (j == b.je && flags(m, BoundaryFace::outer_x2) == BoundaryFlag::outflow)
              ft = fz(m, n, 0, j + 1, i);
            if (j == b.js && flags(m, BoundaryFace::inner_x2) == BoundaryFlag::outflow)
              fb = fz(m, n, 0, j, i);
            if (v == 1) {
              const Real mu = start + (n / sectors + 0.5) * dm;
              if (fr != 0) fr *= mu * faces(m, 0, j, i + 1).killing_norm;
              if (ft != 0) ft *= mu * faces(m, 1, j + 1, i).killing_norm;
              if (fb != 0) fb *= mu * faces(m, 1, j, i).killing_norm;
            }
            val = fr * s.dx2 + (ft - fb) * s.dx1;
          }
          sum += val;
        },
        total);
    rates[v] = total * (2 * M_PI * dm * dp);
  }
  escaped = g0 * escaped + g1 * budget_old[0] + dt * rates[0];
  escaped_j = g0 * escaped_j + g1 * budget_old[1] + dt * rates[1];
  work = g0 * work + g1 * budget_old[2] + dt * rates[2];
  par_for(
      "boltzmann conservative update", DevExeSpace(), 0, blocks - 1, 0, count - 1, b.js, b.je,
      b.is, b.ie, KOKKOS_LAMBDA(int m, int n, int j, int i) {
        int band = n / sectors, sector = n % sectors;
        Real mu_right = band + 1 == bands ? 0 : fm(m, n + sectors, 0, j, i);
        Real psi_right = fp(m, band * sectors + (sector + 1) % sectors, 0, j, i);
        Real rhs = q(m, n, 0, j, i) - (fx(m, n, 0, j, i + 1) - fx(m, n, 0, j, i)) / sz(m).dx1 -
                   (fz(m, n, 0, j + 1, i) - fz(m, n, 0, j, i)) / sz(m).dx2 -
                   (mu_right - fm(m, n, 0, j, i)) / dm - (psi_right - fp(m, n, 0, j, i)) / dp;
        state(m, n, 0, j, i) = g0 * state(m, n, 0, j, i) + g1 * old(m, n, 0, j, i) + dt * rhs;
      });
  return TaskStatus::complete;
}
}  // namespace boltzmann
