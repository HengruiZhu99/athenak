// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
//! \file z4c_hyperboloidal_rhs.cpp
//! \brief Rank-batched mapped RHS on disjoint native block work lists.

#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "bvals/multirate_exchange.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "z4c_hyperboloidal/dissipation_plan.hpp"
#include "z4c_hyperboloidal/endpoint_state.hpp"
#include "z4c_hyperboloidal/point_rhs.hpp"
#include "z4c_hyperboloidal/surface_exchange.hpp"
#include "z4c_hyperboloidal/z4c_hyperboloidal.hpp"

namespace z4c_hyperboloidal {

TaskStatus Z4cHyperboloidal::CalcRHS(Driver *driver, int stage, RHSSubset subset,
                                     Real max_characteristic_cfl,
                                     bvals::multirate::RecordExchange *progress) {
  if (!plans_ready_) throw std::logic_error("Hyperboloidal RHS has no stage plans");
  if (!driver->UsesSchemeARK() || stage < 1 || stage > 4) {
    throw std::invalid_argument("Hyperboloidal RHS requires classical RK4 stage");
  }
  const auto &lists = active_blocks.extent(0) ? active_rhs_points : rhs_points_;
  const auto selected = lists[static_cast<int>(subset)];
  if (!selected.extent(0)) return TaskStatus::complete;
  const auto cells = points.d_view;
  const auto maps = geometry;
  const auto blocks = pmy_pack->pmb->mb_map.d_view;
  const auto refs = reference;
  const auto sources = stationary_source;
  const auto weights = residual_weights;
  const auto omega = spatial_omega;
  const auto params = parameters;
  const auto physical = u0, delta = residual, output = u_rhs;
  const auto statuses = rhs_status;
  const auto gaps = rhs_gap, cfls = rhs_cfl;
  const auto dissipation = dissipation_plan_->Bind();
  // The joined dependent batch has current bulk ghosts and scri samples. Its
  // ordinary points ignore the endpoint cache, as in the original work lists.
  const bool radial = (subset == RHSSubset::Surface || subset == RHSSubset::Dependent) &&
                      lists[static_cast<int>(RHSSubset::Surface)].extent(0);
  const auto surface =
      radial ? surface_exchange_->PrepareRadialJets()
             : SurfaceExchange::CachedKernel{surface_exchange_->BindKernel(),
                                             DvceArray2D<mapped::Jet>()};
  const Real epsilon = ko, dt = active_step_dt > 0 ? active_step_dt : pmy_pack->pmesh->dt;
  const int count = n[0];
  const int selected_count = selected.extent_int(0);
  // Keep each subset's halo dependencies. RHS storage holds KO only until the
  // point kernel loads all fields, including Lambda's contribution to D.
  using KOIndex = std::int64_t;
  Kokkos::parallel_for(
      "hyperboloidal field KO",
      Kokkos::RangePolicy<Kokkos::IndexType<KOIndex>>(0, KOIndex(selected_count) * nvar),
      KOKKOS_LAMBDA(KOIndex task) {
        const int index = static_cast<int>(task % selected_count);
        const int f = static_cast<int>(task / selected_count);
        const int p = selected(index);
        const auto c = cells(p);
        output(c.m, f, c.k, c.j, c.i) =
            dissipation.Evaluate(p, f, {physical, c.m, c.k, c.j, c.i}, epsilon);
      });
  auto bulk_policy = Kokkos::RangePolicy<>(0, selected_count);
#ifdef KOKKOS_ENABLE_SYCL
  // A chunk of one selects Kokkos automatic sizing. Larger chunks set the workgroup.
  bulk_policy.set_chunk_size(rhs_workgroup_size_);
#endif
  // Each field's logical finite differences are independent. Store their Cartesian
  // jets component-major so adjacent points write adjacent device addresses.
  const bool precompute_jets = split_bulk_jets_;
  if (precompute_jets && selected_count > residual_jet_scratch_.extent_int(1)) {
    residual_jet_scratch_ = DvceArray2D<Real>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "hyperboloidal residual jets"),
        14 * Fields::independent_count, selected_count);
  }
  const auto scratch = residual_jet_scratch_;
  if (precompute_jets) {
    Kokkos::parallel_for(
        "hyperboloidal residual field jets",
        Kokkos::RangePolicy<Kokkos::IndexType<KOIndex>>(
            0, KOIndex(selected_count) * Fields::independent_count),
        KOKKOS_LAMBDA(KOIndex task) {
          const int index = static_cast<int>(task % selected_count);
          const int slot = static_cast<int>(task / selected_count);
          int field = slot;
          if (field >= 6) ++field;
          if (field >= 12) ++field;
          const int p = selected(index);
          const auto c = cells(p);
          const auto map = maps(p);
          const BlockPointReader reader{delta, c.m, c.k, c.j, c.i};
          const Real inv_h[3] = {Real(count), Real(count), Real(count)};
          Real beta[3], transport = 0;
          for (int a = 0; a < 3; ++a)
            beta[a] = reader(22 + a) + refs(p)[22 + a].f - map.x[a];
          const auto jet = radial && surface.HasRadialRow(p)
                               ? SampleEndpointJet(p, field, reader, map, inv_h, beta,
                                                   surface, transport)
                               : SampleResidualJet(reader, field, map, inv_h, beta,
                                                   field <= 6 || field == 18 ||
                                                       (field >= 22 && field < 25),
                                                   transport, blocks(c.m).face < 0);
          const int first = 14 * slot;
          scratch(first, index) = jet.f;
          for (int a = 0; a < 3; ++a) {
            scratch(first + 1 + a, index) = jet.d[a];
            for (int b = 0; b < 3; ++b)
              scratch(first + 4 + 3 * a + b, index) = jet.h[a][b];
          }
          scratch(first + 13, index) = transport;
        });
  }
  // The interior and non-scri dependent lists contain no radial closure rows.
  // Compile their point kernel without the endpoint reconstruction temporaries.
  auto evaluate = [&](auto radial_rows, auto precomputed) {
    constexpr bool with_radial_rows = decltype(radial_rows)::value;
    Kokkos::parallel_for(
        "hyperboloidal block RHS", bulk_policy, KOKKOS_LAMBDA(int index) {
          const int p = selected(index);
          const auto c = cells(p);
          const auto map = maps(p);
          const Real inv_h[3] = {Real(count), Real(count), Real(count)};
          const BlockPointReader reader{delta, c.m, c.k, c.j, c.i};
          BulkPointResult result;
          if constexpr (decltype(precomputed)::value) {
            const Real radius = SphericalPolicy::Radius(map.x);
            if (!Kokkos::isfinite(radius) || !(radius >= 0) ||
                (radius == 0 && !params.residual_weight.Regular())) {
              statuses(p) = static_cast<int>(PointStatus::invalid_input);
              return;
            }
            PointForcing forcing;
            for (int f = 0; f < nvar; ++f)
              forcing.dissipation[f] = output(c.m, f, c.k, c.j, c.i);
            const bool endpoint = with_radial_rows && surface.HasRadialRow(p);
            const auto load = [&](int f, bool second) {
              const int first = 14 * Fields::IndependentSlot(f);
              mapped::Jet q(scratch(first, index));
              for (int k = 0; k < 3; ++k) {
                q.d[k] = scratch(first + 1 + k, index);
                if (second) {
                  for (int l = 0; l < 3; ++l)
                    q.h[k][l] = scratch(first + 4 + 3 * k + l, index);
                }
              }
              forcing.transport[f] = scratch(first + 13, index);
              if (f == Fields::chi) {
                forcing.transport[f] *= params.residual_weight.Regular()
                                            ? weights(p).f
                                            : params.residual_weight.Value(map.x);
                if (params.residual_weight.Regular()) return refs(p)[f] + weights(p) * q;
                const auto r2 = SphericalPolicy::Radius2(map.x);
                return (q + refs(p)[f] / r2) * r2;
              }
              if (f == Fields::lapse) {
                ApplyCompatibleLapse(q, reader, map, inv_h, radius, params);
              }
              auto restored = q + refs(p)[f];
              if (endpoint && f >= Fields::shift && f < Fields::shift + 3)
                restored = restored + surface.NormalCorrection(p, map, f - Fields::shift);
              return restored;
            };
            result =
                EvaluateBulkFieldJets(map, params, sources(p), forcing, load, omega(p));
          } else {
            PointState state;
            if constexpr (with_radial_rows) {
              state = surface.HasRadialRow(p)
                          ? EndpointState(p, reader, map, inv_h, refs(p), params, surface,
                                          &weights(p))
                          : BuildPointState(reader, map, inv_h, refs(p), params, nullptr,
                                            &weights(p), blocks(c.m).face < 0);
            } else {
              state = BuildPointState(reader, map, inv_h, refs(p), params, nullptr,
                                      &weights(p), blocks(c.m).face < 0);
            }
            statuses(p) = static_cast<int>(state.status);
            if (state.status != PointStatus::success) return;
            PointForcing forcing;
            for (int f = 0; f < nvar; ++f) {
              forcing.transport[f] = state.transport[f];
              forcing.dissipation[f] = output(c.m, f, c.k, c.j, c.i);
            }
            result = EvaluateBulkPoint(map, state.jets, params, sources(p), forcing,
                                       &omega(p));
          }
          statuses(p) = static_cast<int>(result.status);
          gaps(p) = result.gap;
          cfls(p) = dt * count * (result.speed[0] + result.speed[1] + result.speed[2]);
          if (dt / fmin(params.tau, Real(InnerTelegrapher::tau)) > .25)
            statuses(p) = static_cast<int>(PointStatus::cfl_exceeded);
          for (int f = 0; f < nvar; ++f) output(c.m, f, c.k, c.j, c.i) = result.rate[f];
        });
  };
  if (radial && precompute_jets) {
    evaluate(std::true_type{}, std::true_type{});
  } else if (radial) {
    evaluate(std::true_type{}, std::false_type{});
  } else if (precompute_jets) {
    evaluate(std::false_type{}, std::true_type{});
  } else {
    evaluate(std::false_type{}, std::false_type{});
  }
#if MPI_PARALLEL_ENABLED
  // Give posted records one poll on the calling MPI thread. The host-scalar
  // validity reduction below waits for the bulk kernels to finish.
  if (progress != nullptr) {
    progress->Poll();
  }
#else
  (void)progress;
#endif
  int invalid = 0;
  Kokkos::parallel_reduce(
      "hyperboloidal RHS validity", Kokkos::RangePolicy<>(0, selected_count),
      KOKKOS_LAMBDA(int i, int &bad) {
        const int point = selected(i);
        if (statuses(point) != static_cast<int>(PointStatus::success)) {
          bad = 2;
        } else if (bad < 1 && max_characteristic_cfl > 0 &&
                   (!Kokkos::isfinite(cfls(point)) ||
                    cfls(point) > max_characteristic_cfl)) {
          bad = 1;
        }
      },
      Kokkos::Max<int>(invalid));
  if (invalid == 1) {
    throw std::runtime_error(
        "Multirate monitored characteristic CFL bound exceeded; fixed timestep and "
        "hierarchy retained, run stopped");
  }
  if (invalid == 2) throw std::runtime_error("Invalid native hyperboloidal bulk RHS");
  return TaskStatus::complete;
}
}  // namespace z4c_hyperboloidal
