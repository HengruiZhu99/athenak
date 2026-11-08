// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Test-only entry point: compile with the pinned historical include paths,
// macros and link command. The included Evolution implementation is unchanged.
#define HYPERBOLIC_NO_DRIVER_MAIN 1
#include "tst/hyperbolic_shift_3d/coupled_mpi.cpp"
#include "tst/hyperbolic_shift_3d/endpoint_policy/static_mesh.hpp"
#include "tst/hyperbolic_shift_3d/endpoint_policy/run_controls.hpp"

namespace legacy_stage_trace {
constexpr int fields = 31;
constexpr int records = 8;

void Require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

struct Dump {
  std::ofstream file;
  Dump(const std::string &label, int region, int count, int values) {
    file.open(label + (region ? "-scri-rank" : "-bulk-rank") +
              std::to_string(global_variable::my_rank) + ".bin", std::ios::binary);
    Require(bool(file), "cannot open legacy trace file");
    const char magic[16] = "LEGACY_TRACE_V1";
    const std::uint64_t size = count;
    const std::uint32_t shape[4] = {6, std::uint32_t(values), std::uint32_t(region), 1};
    file.write(magic, 16);
    file.write(reinterpret_cast<const char *>(&size), 8);
    file.write(reinterpret_cast<const char *>(shape), 16);
  }
  void Row(const std::array<std::int32_t, 6> &key, const Real *values, int count) {
    for (int f = 0; f < count; ++f)
      Require(std::isfinite(values[f]), "nonfinite owned legacy trace value");
    file.write(reinterpret_cast<const char *>(key.data()), sizeof(key));
    file.write(reinterpret_cast<const char *>(values), count * sizeof(Real));
  }
  void Close() {
    file.close();
    Require(bool(file), "legacy trace write failed");
  }
};

std::array<std::int32_t, 6> Key(Evolution &e, int region, int ordinal) {
  const auto &model = e.Model();
  if (region) {
    const auto &row = model.scri.rows[ordinal];
    return {ordinal, row.block, row.k, row.j, 12, model.grid.mesh.blocks[row.block].face};
  }
  const auto &cell = model.grid.cells[ordinal];
  return {ordinal, cell.block, cell.k, cell.j, cell.i,
          model.grid.mesh.blocks[cell.block].face};
}

void Geometry(Evolution &e) {
  for (int region = 0; region < 2; ++region) {
    const int first = region ? e.surface_first : e.first;
    const int count = region ? e.surface_local : e.local;
    Dump dump("geometry", region, count, 3);
    for (int q = 0; q < count; ++q) {
      const int ordinal = first + q;
      const Real *x = region ? e.Model().scri.rows[ordinal].geometry.x
                             : e.Model().grid.geometry[ordinal].x;
      dump.Row(Key(e, region, ordinal), x, 3);
    }
    dump.Close();
  }
}

void Snapshot(Evolution &e, const std::string &label, bool rhs = false) {
  for (int region = 0; region < 2; ++region) {
    const int first = region ? e.surface_first : e.first;
    const int count = region ? e.surface_local : e.local;
    Buffer owned;
    if (rhs) {
      owned = region ? e.edge_rates[0] : e.rates[0];
    } else {
      const auto source = region ? e.surface : e.state;
      owned = Kokkos::subview(source, std::make_pair(
          std::size_t(fields * first), std::size_t(fields * (first + count))));
    }
    Require(owned.extent(0) == std::size_t(fields * count), "legacy owned extent changed");
    const auto values = e.Download(owned);
    const auto ledger = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), region ? e.spherical_surface_chi_records : e.spherical_chi_records);
    Require(ledger.extent(0) == std::size_t(count), "legacy chi ownership changed");
    const int width = rhs ? fields : fields + records;
    Dump dump(label, region, count, width);
    for (int q = 0; q < count; ++q) {
      Real row[fields + records];
      std::copy_n(values.data() + fields * q, fields, row);
      if (!rhs) {
        const auto &r = ledger(q);
        const Real history[records] = {r.minimum, r.accepted_minimum, r.floor_count,
            r.nonpositive_count, r.nonfinite_count, r.max_correction,
            r.sum_correction, r.first_floor};
        std::copy_n(history, records, row + fields);
      }
      dump.Row(Key(e, region, first + q), row, width);
    }
    dump.Close();
  }
}
}  // namespace legacy_stage_trace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &global_variable::nranks);
  Kokkos::initialize(argc, argv);
  try {
    using legacy_stage_trace::Require;
    static_assert(sizeof(Real) == 8 && h::spherical_policy, "requires pinned float64 N12 policy");
    Require(argc == 2, "legacy_stage_trace_gpu INPUT (writes fresh files in cwd)");
    const auto controls = EndpointRunControls::Load(argv[1]);
    Require((controls.nlim == 1 || controls.nlim == 8) && controls.ko == .1 &&
            controls.chi_floor == 1e-10 && controls.dt == 0.00046944203735617445 &&
            controls.tlim > controls.nlim * controls.dt, "changed N12 trace controls");
    hyperbolic_reproduction::refined_configuration.Load(
        controls.configuration.string(), hyperbolic_reproduction::ContinuousCoreMesh(12));
    auto mesh = EndpointStaticMesh(hyperbolic_reproduction::refined_configuration, 12);
    Evolution e(12, 0, 2, "", &mesh);
    e.ConfigureSphericalNumerics(controls.ko, controls.chi_floor);
    Require(e.total == 229392 && e.surface_total == 864, "changed legacy N12 workload");
    e.Upload(e.initial_state);
    // The original run.cpp performs this RHS in its initial diagnostic snapshot.
    // All other snapshot operations are read-only. No new RHS is inserted in RK.
    Buffer diagnostic_edge;
    e.RHS(controls.dt, diagnostic_edge);
    legacy_stage_trace::Geometry(e);
    legacy_stage_trace::Snapshot(e, "accepted0");
    Real expected_time = 0;
    for (int step = 1; step <= controls.nlim; ++step) {
      e.clock_max = 0;
      e.cfl = 0;
      e.Step(controls.dt);
      expected_time += controls.dt;
      Require(e.steps == std::uint64_t(step) && e.time == expected_time,
              "legacy trace changed full-step accounting");
      if (step == 1) {
        // These persistent copies are the actual first RK-stage rates. The N12
        // spherical policy copies raw rates, without logarithmic transformations.
        legacy_stage_trace::Snapshot(e, "first-rhs", true);
        legacy_stage_trace::Snapshot(e, "accepted1");
      }
      if (step == 8) legacy_stage_trace::Snapshot(e, "accepted8");
    }
    // Separately preserve the original final-snapshot phase for byte comparison
    // to the unchanged evolve-static binary. Do not substitute it for accepted1/8.
    e.RHS(controls.dt, diagnostic_edge);
    legacy_stage_trace::Snapshot(e, "final-snapshot");
    if (e.rank == 0) {
      std::ofstream report("results.json");
      report << std::setprecision(17)
             << "{\"schema\":\"legacy-stage-trace-v1\",\"backend\":\""
             << Kokkos::DefaultExecutionSpace::name() << "\",\"ranks\":" << e.size
             << ",\"steps\":" << e.steps << ",\"dt\":" << controls.dt
             << ",\"time\":" << e.time
             << ",\"bulk_points\":229392,\"scri_points\":864,\"passed\":true}\n";
      report.close();
      Require(bool(report), "legacy trace result write failed");
    }
  } catch (const std::exception &error) {
    std::cerr << "rank " << global_variable::my_rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
