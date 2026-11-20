#include <limits>

#include "boltzmann/boltzmann.hpp"
#include "boltzmann/fourth_order.hpp"
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"

namespace boltzmann {
TaskStatus Boltzmann::FluxesFourth(Driver* driver, int stage) {
  const auto sz = pack->pmb->mb_size.d_view;
  const auto b = pack->pmesh->mb_indcs;
  const auto native = pack->pz4c->u_adm_native;
  const auto cells = geometry;
  const auto faces = face_geometry;
  const auto a = speeds;
  const auto inv = inverse_budget;
  const auto rules = polar_rules;
  const auto polar = polar_flux_weights;
  const int hemisphere = nmu_half, sectors = npsi, count = nang,
            blocks = pack->nmb_thispack;
  const Real start = mu_min, dm = dmu, dp = dpsi;
  const bool ssp4 = driver->integrator == "ssprk104";
  if (!speed_cache_valid) {
    for (int d = 0; d < 2; ++d) {
      const int di = d == 0, dj = d == 1;
      par_for(
          "FV4 spatial face frames", DevExeSpace(), 0, blocks - 1, b.js - 1,
          b.je + 1 + dj, b.is - 1, b.ie + 1 + di,
          KOKKOS_LAMBDA(int m, int j, int i) {
            using A = adm::ADM;
            auto midpoint = [&](int v) {
              return (9 * (native(m, v, 0, j, i) +
                           native(m, v, 0, j + di, i + dj)) -
                      native(m, v, 0, j - di, i - dj) -
                      native(m, v, 0, j + 2 * di, i + 2 * dj)) /
                     16;
            };
            const int sym[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
            Real metric[3][3];
            for (int k = 0; k < 3; ++k)
              for (int l = 0; l < 3; ++l)
                metric[k][l] = midpoint(A::I_ADM_GXX + sym[k][l]);
            Frame frame{};
            frame.Set(metric);
            const bool axis = di && i == b.is && sz(m).x1min == 0;
            auto& f = faces(m, d, j, i);
            f.minus_shift = axis ? 0 : -midpoint(A::I_ADM_BETAX + d);
            for (int k = 0; k < 3; ++k)
              f.triad[k] =
                  axis ? 0 : midpoint(A::I_ADM_ALPHA) * frame.inverse[k][d];
            f.killing_norm = (sz(m).x1min + (i - b.is + 0.5 * dj) * sz(m).dx1) *
                             frame.co[2][2];
          });
      par_for(
          "FV4 spatial face characteristics", DevExeSpace(), 0, blocks - 1, 0,
          count - 1, b.js - 1, b.je + 1 + dj, b.is - 1, b.ie + 1 + di,
          KOKKOS_LAMBDA(int m, int n, int j, int i) {
            Real mu = start + (n / sectors + 0.5) * dm,
                 psi = (n % sectors + 0.5) * dp;
            Real s = sqrt(fmax(0., 1 - mu * mu)),
                 dir[3] = {s * cos(psi), s * sin(psi), mu};
            a[d](m, n, 0, j, i) = faces(m, d, j, i).Speed(dir);
          });
    }
    par_for(
        "FV4 angular characteristics", DevExeSpace(), 0, blocks - 1, 0,
        count - 1, b.js - 1, b.je + 1, b.is - 1, b.ie + 1,
        KOKKOS_LAMBDA(int m, int n, int j, int i) {
          Real mu = start + (n / sectors + 0.5) * dm,
               psi = (n % sectors + 0.5) * dp;
          const Geometry& g = cells(m, j, i);
          a[2](m, n, 0, j, i) =
              n / sectors % hemisphere == 0
                  ? 0
                  : Characteristics(g, start + n / sectors * dm, psi).mu;
          const auto& rule = rules(n / sectors);
          Real mean = 0, weights[4]{};
          for (int q = 0; q < 4; ++q) {
            Real v = rule.weight[q] *
                     Characteristics(g, rule.mu[q], n % sectors * dp).psi;
            mean += v;
            for (int k = 0; k < 4; ++k) weights[k] += v * rule.basis[q][k];
          }
          a[3](m, n, 0, j, i) = mean;
          int c = 0;
          for (int k = 0; k < 4; ++k)
            if (rule.offset + k != 0) polar[c++](m, n, 0, j, i) = weights[k];
          a[4](m, n, 0, j, i) = Characteristics(g, mu, psi).redshift;
        });
    Real maxrate = 0, maxsource = 0;
    Kokkos::parallel_reduce(
        "FV4 outgoing flux budgets",
        Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<4>>(
            {0, 0, b.js - 1, b.is - 1}, {blocks, count, b.je + 2, b.ie + 2}),
        KOKKOS_LAMBDA(int m, int n, int j, int i, Real& largest, Real& sink) {
          PhaseCell p{i, j, n / sectors, n % sectors};
          Real h[4] = {sz(m).dx1, sz(m).dx2, dm, dp};
          Real total = fabs(p.Read(a[4], m, sectors));
          sink = fmax(sink, total);
          for (int d = 0; d < 4; ++d) {
            Real right = d == 2 && (p.band + 1) % hemisphere == 0
                             ? 0
                             : p.Shift(d, 1, sectors).Read(a[d], m, sectors);
            total += (fabs(right) + fabs(p.Read(a[d], m, sectors))) / h[d];
          }
          inv(m, n, 0, j, i) = total > 0 ? 1 / total : 0;
          if (j >= b.js && j <= b.je && i >= b.is && i <= b.ie)
            largest = fmax(largest, total);
        },
        Kokkos::Max<Real>(maxrate), Kokkos::Max<Real>(maxsource));
    source_free = maxsource == 0;
    if (source_free) Kokkos::deep_copy(DevExeSpace(), source, 0.);
    dtnew = (ssp4 ? 6 : 1) / fmax(maxrate, std::numeric_limits<Real>::min());
    speed_cache_valid = true;
  }
  if (!stage) return TaskStatus::complete;
  speed_cache_valid = fixed_metric;
  const Real euler_dt = pack->pmesh->dt / (ssp4 ? 6 : 1);
  const auto state = u;
  for (int d = 0; d < (source_free ? 4 : 5); ++d) {
    auto out = d == 0   ? flux.x1f
               : d == 1 ? flux.x2f
               : d == 2 ? angular_mu
               : d == 3 ? angular_psi
                        : source;
    par_for(
        "FV4 phase space flux", DevExeSpace(), 0, blocks - 1, 0, count - 1,
        b.js, b.je + (d == 1), b.is, b.ie + (d == 0),
        KOKKOS_LAMBDA(int m, int n, int j, int i) {
          PhaseCell p{i, j, n / sectors, n % sectors};
          if ((d == 2 && p.band % hemisphere == 0) ||
              (d == 0 && i == b.is && sz(m).x1min == 0)) {
            out(m, n, 0, j, i) = 0;
            return;
          }
          if (d < 4 && p.Read(state, m, sectors) == 0 &&
              p.Shift(d, -1, sectors).Read(state, m, sectors) == 0) {
            out(m, n, 0, j, i) = 0;
            return;
          }
          PhaseFace face{d, m, hemisphere, sectors, state, a[d]};
          Real value = d == 3 ? face.PolarFlux(p, rules(p.band).offset, polar)
                              : face.Product(p);
          if (d < 4 || value < 0) {
            PhaseCell donor =
                (d < 4 && value > 0) ? p.Shift(d, -1, sectors) : p;
            Real width = d == 0   ? sz(m).dx1
                         : d == 1 ? sz(m).dx2
                         : d == 2 ? dm
                         : d == 3 ? dp
                                  : 1;
            // Allocate one positive Euler budget across eight faces and the
            // sink. The small equal share also permits a transverse flux at a
            // velocity zero.
            Real norm = donor.Read(inv, m, sectors);
            Real share =
                norm == 0
                    ? 1. / 9
                    : (fabs(p.Read(a[d], m, sectors)) * norm / width + 0.02) /
                          1.18;
            // A small unspent fraction keeps the Euler mass positive after
            // rounding.
            // Subnormal donors cannot represent a fractional outflow budget.
            // Retain that mass until incoming flux restores normal precision.
            const Real mass = donor.Read(state, m, sectors);
            const Real cap = mass < 16 * std::numeric_limits<Real>::min()
                                 ? 0
                                 : (0.99 * share * (width / euler_dt)) * mass;
            value = copysign(fmin(fabs(value), cap), value);
          }
          out(m, n, 0, j, i) = value;
        });
  }
  return TaskStatus::complete;
}
}  // namespace boltzmann
