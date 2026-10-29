// Standalone communication control: no mesh, RHS or evolution is constructed.
#include <mpi.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include "athena.hpp"

namespace {
using Buffer = Kokkos::View<Real *, Kokkos::DefaultExecutionSpace::memory_space>;
using HostBuffer = Kokkos::View<Real *, Kokkos::HostSpace>;
using Clock = std::chrono::steady_clock;
constexpr int fields = 31;
enum Route { device = 0, host = 1, staged = 2 };
constexpr const char *route_names[] = {"device", "host", "staged"};
enum Metric { prefence, barrier, operation, mpi, d2h, h2d, completion_fence,
              volume_mpi, surface_mpi, sequence, validation, checked, mismatches, metrics };
constexpr const char *metric_names[] = {
  "producer_fence_seconds", "barrier_seconds", "operation_seconds", "mpi_seconds",
  "d2h_seconds", "h2d_seconds", "completion_fence_seconds", "volume_mpi_seconds",
  "surface_mpi_seconds", "sequence_seconds", "validation_seconds", "checked_values",
  "mismatches"};
using Measurement = std::array<double, metrics>;

void Require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}
void MPICheck(int code) {
  if (code == MPI_SUCCESS) return;
  char message[MPI_MAX_ERROR_STRING]; int size = 0;
  MPI_Error_string(code, message, &size);
  throw std::runtime_error(std::string(message, size));
}
double Seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now()-start).count();
}
std::string Json(const std::string &value) {
  std::ostringstream out; out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                         << static_cast<int>(c) << std::dec;
    else out << c;
  }
  return out.str()+'"';
}
int Integer(const std::string &text, int minimum, int maximum) {
  Require(!text.empty() && text.find_first_not_of("0123456789") == std::string::npos,
          "integer arguments require decimal digits");
  const auto value = std::stoll(text);
  Require(value >= minimum && value <= maximum, "integer argument is outside its allowed range");
  return static_cast<int>(value);
}
struct Options {
  std::vector<int> grids;
  std::string output, mode = "both", expected_backend;
  int repeats = 20, warmup = 3, mix_repeats = 0, expected_ranks = 0;
  bool layout_only = false, help = false;
};
Options Parse(int argc, char **argv) {
  Options o; std::set<std::string> seen;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    Require(arg == "--n" || seen.insert(arg).second, "duplicate option: "+arg);
    if (arg == "--help") { o.help = true; continue; }
    if (arg == "--layout-only") { o.layout_only = true; continue; }
    Require(arg == "--n" || arg == "--output" || arg == "--mode" || arg == "--repeats"
            || arg == "--warmup" || arg == "--mix-repeats" || arg == "--expect-ranks"
            || arg == "--expect-backend", "unknown option: "+arg);
    Require(i+1 < argc, "missing value for "+arg); const std::string value = argv[++i];
    if (arg == "--n") {
      const int n = Integer(value, 16, 32);
      Require(n == 16 || n == 24 || n == 32, "--n supports only 16, 24 and 32");
      Require(std::find(o.grids.begin(), o.grids.end(), n) == o.grids.end(), "duplicate grid");
      o.grids.push_back(n);
    } else if (arg == "--output") o.output = value;
    else if (arg == "--mode") o.mode = value;
    else if (arg == "--repeats") o.repeats = Integer(value, 1, 200);
    else if (arg == "--warmup") o.warmup = Integer(value, 0, 20);
    else if (arg == "--mix-repeats") o.mix_repeats = Integer(value, 0, 20);
    else if (arg == "--expect-ranks") o.expected_ranks = Integer(value, 1, 1024);
    else o.expected_backend = value;
  }
  if (o.help) { Require(argc == 2, "--help must be used alone"); return o; }
  Require(!o.output.empty(), "--output FILE is required");
  Require(o.mode == "barrier" || o.mode == "no_barrier" || o.mode == "both", "invalid --mode");
  if (o.grids.empty()) o.grids = {16, 24, 32};
  return o;
}
void Help() {
  std::cout << "usage: hyperbolic_shift_3d_mpi_communication --output FILE [options]\n"
    "  --n 16|24|32          repeatable; default all three, in that order\n"
    "  --repeats 1..200      measured rounds per single-buffer group (default 20)\n"
    "  --warmup 0..20        warmup rounds per group (default 3)\n"
    "  --mix-repeats 0..20   optional 4-volume/14-surface rounds (default 0)\n"
    "  --mode barrier|no_barrier|both  default both\n"
    "  --expect-ranks N --expect-backend NAME  optional launch identity checks\n"
    "  --layout-only        report exact layouts without allocating payload buffers\n"
    "Runs on one shared-memory node. Existing output files are rejected.\n"
    "All three routes use persistent buffers and exact full-array validation.\n";
}

// Exactly the cell/row partition and point-major field layout in Evolution.
struct Layout {
  int n, points, rank, ranks, first, local, total_values;
  bool edge;
  std::vector<int> counts, displacements;
  Layout(int resolution, bool surface, int r, int size)
      : n(resolution), points(surface ? 6*n*n : 19*n*n*n), rank(r), ranks(size),
        first(points*r/size), local(points*(r+1)/size-first), total_values(fields*points),
        edge(surface), counts(size), displacements(size) {
    Require(local > 0, "rank has no owned cells/surface rows");
    for (int p = 0; p < size; ++p) {
      displacements[p] = fields*(points*p/size);
      counts[p] = fields*(points*(p+1)/size-points*p/size);
    }
  }
};
KOKKOS_INLINE_FUNCTION
Real Pattern(int index, int owner, int epoch) {
  // All terms and sums are exact integers for the supported double-precision runs.
  return Real(index+1) + Real(owner+1)*33554432.0 + Real(epoch+1)*4294967296.0;
}
KOKKOS_INLINE_FUNCTION
int Owner(int index, int points, int ranks) {
  return static_cast<int>((static_cast<std::int64_t>(index/fields+1)*ranks-1)/points);
}
struct Payload {
  Layout layout;
  Buffer send, receive;
  HostBuffer host_send, host_receive;
  explicit Payload(Layout l) : layout(std::move(l)),
    send("benchmark device send", fields*layout.local),
    receive("benchmark device receive", layout.total_values),
    host_send("benchmark HostSpace send", fields*layout.local),
    host_receive("benchmark HostSpace receive", layout.total_values) {}
  void Prepare(int epoch, Route route) {
    const int first = fields*layout.first, rank = layout.rank;
    const auto local_send = send;
    Kokkos::parallel_for("benchmark deterministic send", Kokkos::RangePolicy<>(0, send.extent(0)),
      KOKKOS_LAMBDA(int i) { local_send(i) = Pattern(first+i, rank, epoch); });
    // Poison the staged host input so a missing device-to-host copy cannot pass.
    for (int i = 0; i < fields*layout.local; ++i)
      host_send(i) = route == staged ? std::numeric_limits<Real>::quiet_NaN()
                                    : Pattern(first+i, rank, epoch);
    Kokkos::deep_copy(receive, std::numeric_limits<Real>::quiet_NaN());
    for (int i = 0; i < layout.total_values; ++i)
      host_receive(i) = std::numeric_limits<Real>::quiet_NaN();
  }
  void Call(Route route, bool synchronize, Measurement &m) {
    auto start = Clock::now(); Kokkos::fence(); m[prefence] += Seconds(start);
    if (synchronize) {
      start = Clock::now(); MPICheck(MPI_Barrier(MPI_COMM_WORLD)); m[barrier] += Seconds(start);
    }
    const auto whole = Clock::now();
    if (route == staged) {
      start = Clock::now(); Kokkos::deep_copy(host_send, send); Kokkos::fence();
      m[d2h] += Seconds(start);
    }
    start = Clock::now();
    MPICheck(MPI_Allgatherv(route == device ? send.data() : host_send.data(),
      layout.counts[layout.rank], MPI_ATHENA_REAL,
      route == device ? receive.data() : host_receive.data(),
      layout.counts.data(), layout.displacements.data(), MPI_ATHENA_REAL, MPI_COMM_WORLD));
    const double elapsed = Seconds(start);
    m[mpi] += elapsed; m[layout.edge ? surface_mpi : volume_mpi] += elapsed;
    if (route == staged) {
      start = Clock::now(); Kokkos::deep_copy(receive, host_receive); Kokkos::fence();
      m[h2d] += Seconds(start);
    } else if (route == device) {
      start = Clock::now(); Kokkos::fence(); m[completion_fence] += Seconds(start);
    }
    m[operation] += Seconds(whole);
  }
  void Validate(Route route, int epoch, Measurement &m) const {
    const auto start = Clock::now();
    const int points = layout.points, ranks = layout.ranks;
    if (route != host) {
      const auto data = receive; int bad = 0;
      Kokkos::parallel_reduce("benchmark validate complete device receive",
        Kokkos::RangePolicy<>(0, layout.total_values), KOKKOS_LAMBDA(int i, int &count) {
          if (data(i) != Pattern(i, Owner(i, points, ranks), epoch)) ++count;
        }, bad);
      Kokkos::fence(); m[mismatches] += bad; m[checked] += layout.total_values;
    }
    if (route != device) {
      for (int i = 0; i < layout.total_values; ++i)
        if (host_receive(i) != Pattern(i, Owner(i, points, ranks), epoch)) ++m[mismatches];
      m[checked] += layout.total_values;
    }
    m[validation] += Seconds(start);
  }
};
struct Sample {
  int group, round, position, route, epoch;
  Measurement values{};
};
struct Group {
  int n, repeats, first_sample, sample_count;
  std::string workload;
  bool synchronized;
  std::vector<int> calls;
};
constexpr int orders[6][3] = {{0,1,2},{2,1,0},{1,2,0},{0,2,1},{2,0,1},{1,0,2}};
Measurement Run(Payload &volume, Payload &surface, const Group &group, Route route, int epoch) {
  const bool uses_volume = std::find(group.calls.begin(), group.calls.end(), 0) != group.calls.end();
  const bool uses_surface = std::find(group.calls.begin(), group.calls.end(), 1) != group.calls.end();
  if (uses_volume) volume.Prepare(epoch, route);
  if (uses_surface) surface.Prepare(epoch, route);
  Measurement m{}; const auto start = Clock::now();
  for (int edge : group.calls) (edge ? surface : volume).Call(route, group.synchronized, m);
  m[sequence] = Seconds(start);
  if (uses_volume) volume.Validate(route, epoch, m);
  if (uses_surface) surface.Validate(route, epoch, m);
  return m;
}
void CheckMeasurements(const std::vector<Sample> &samples, int first, int end) {
  int bad = 0;
  for (int i = first; i < end; ++i) {
    for (double value : samples[i].values) if (!std::isfinite(value) || value < 0) ++bad;
    if (samples[i].values[mismatches] != 0) ++bad;
  }
  MPICheck(MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
  Require(bad == 0, "nonfinite/negative timing or complete-array validation failure");
}
std::vector<std::string> GatherStrings(const std::string &local, int rank, int ranks) {
  const int length = local.size(); std::vector<int> lengths(ranks), offsets(ranks);
  MPICheck(MPI_Gather(&length, 1, MPI_INT, lengths.data(), 1, MPI_INT, 0, MPI_COMM_WORLD));
  for (int r = 1; r < ranks; ++r) offsets[r] = offsets[r-1]+lengths[r-1];
  std::vector<char> all(offsets.back()+lengths.back());
  MPICheck(MPI_Gatherv(local.data(), length, MPI_CHAR, all.data(), lengths.data(), offsets.data(),
                      MPI_CHAR, 0, MPI_COMM_WORLD));
  std::vector<std::string> result;
  if (rank == 0) for (int r = 0; r < ranks; ++r)
    result.emplace_back(all.data()+offsets[r], lengths[r]);
  return result;
}
std::string RankMetadata(int rank, int shared_size) {
  char host[MPI_MAX_PROCESSOR_NAME]; int length = 0;
  MPICheck(MPI_Get_processor_name(host, &length));
  std::ostringstream configuration; Kokkos::print_configuration(configuration, true);
  std::ostringstream out;
  out << "{\"rank\":" << rank << ",\"host\":" << Json(std::string(host, length))
      << ",\"shared_node_ranks\":" << shared_size << ",\"environment\":{";
  const char *names[] = {"ZE_AFFINITY_MASK", "ZE_FLAT_DEVICE_HIERARCHY",
    "ZE_ENABLE_PCI_ID_DEVICE_ORDER", "ONEAPI_DEVICE_SELECTOR", "MPIR_CVAR_ENABLE_GPU",
    "OMP_NUM_THREADS", "MPI_LOCALRANKID", "PALS_LOCAL_RANKID"};
  for (int i = 0; i < 8; ++i) {
    if (i) out << ',';
    const char *value = std::getenv(names[i]);
    out << Json(names[i]) << ':' << (value ? Json(value) : "null");
  }
  out << "},\"kokkos_configuration\":" << Json(configuration.str()) << '}';
  return out.str();
}
void Array(std::ostream &out, const std::vector<int> &values) {
  out << '['; for (size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << values[i]; }
  out << ']';
}
double Quantile(std::vector<double> values, double fraction) {
  std::sort(values.begin(), values.end());
  const double position = fraction*(values.size()-1); const size_t lower = position;
  return values[lower]+(position-lower)*(values[static_cast<size_t>(std::ceil(position))]-values[lower]);
}
void Write(const Options &o, int ranks, const std::vector<std::string> &rank_metadata,
           const std::vector<Group> &groups, const std::vector<Sample> &samples,
           const std::vector<double> &all, const std::vector<std::string> &command) {
  char library[MPI_MAX_LIBRARY_VERSION_STRING]; int library_length = 0;
  MPICheck(MPI_Get_library_version(library, &library_length));
  const std::filesystem::path output(o.output), temporary(o.output+".tmp");
  std::ofstream out(temporary); Require(out.good(), "cannot open output file");
  out << std::setprecision(17) << "{\"schema\":1,\"status\":\"passed\",\"benchmark\":\"mpi_allgatherv\""
      << ",\"layout_only\":" << (o.layout_only ? "true" : "false")
      << ",\"backend\":" << Json(Kokkos::DefaultExecutionSpace::name())
      << ",\"device_memory_space\":" << Json(Kokkos::DefaultExecutionSpace::memory_space::name())
      << ",\"host_memory_space\":\"HostSpace\",\"host_pinned_allocation_claimed\":false"
      << ",\"same_device_host_memory_space\":"
      << (std::is_same<Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace>::value ? "true" : "false")
      << ",\"real_bytes\":" << sizeof(Real) << ",\"mpi_size\":" << ranks
      << ",\"kokkos_version\":" << KOKKOS_VERSION << ",\"compiler\":" << Json(__VERSION__)
      << ",\"mpi_library\":" << Json(std::string(library, library_length))
      << ",\"clock\":\"std::chrono::steady_clock local elapsed; rank clocks not assumed synchronized\""
      << ",\"repeats\":" << o.repeats << ",\"warmup\":" << o.warmup
      << ",\"mix_repeats\":" << o.mix_repeats << ",\"mode\":" << Json(o.mode)
      << ",\"route_order\":\"six balanced forward/reverse permutations, repeated per group\""
      << ",\"validation\":\"exact full final receive arrays after every workload; staged checks host and device; preparation/validation excluded from operation time\""
      << ",\"mix_validation\":\"final volume and surface arrays; intermediate gathers use identical send payloads\""
      << ",\"timing_contract\":\"operation is sum of individually timed calls including copies/completion fences; sequence also includes producer fences/barriers; no bandwidth inference\""
      << ",\"command\":[";
  for (size_t i = 0; i < command.size(); ++i) { if (i) out << ','; out << Json(command[i]); }
  out << "],\"ranks\":[";
  for (int r = 0; r < ranks; ++r) { if (r) out << ','; out << rank_metadata[r]; }
  out << "],\"layouts\":["; bool comma = false;
  for (int n : o.grids) for (bool edge : {false, true}) {
    if (comma) out << ','; comma = true; const Layout l(n, edge, 0, ranks);
    out << "{\"n\":" << n << ",\"kind\":" << Json(edge ? "surface" : "volume")
        << ",\"points\":" << l.points << ",\"fields\":31,\"global_values\":" << l.total_values
        << ",\"global_receive_bytes\":" << sizeof(Real)*l.total_values << ",\"counts\":";
    Array(out, l.counts); out << ",\"displacements\":"; Array(out, l.displacements); out << '}';
  }
  out << "],\"groups\":[";
  const size_t per_rank = samples.size()*metrics;
  for (size_t g = 0; g < groups.size(); ++g) {
    if (g) out << ','; const auto &group = groups[g];
    out << "{\"n\":" << group.n << ",\"workload\":" << Json(group.workload)
        << ",\"mode\":" << Json(group.synchronized ? "barrier" : "no_barrier")
        << ",\"repeats\":" << group.repeats << ",\"call_sequence\":";
    Array(out, group.calls); out << ",\"samples\":[";
    for (int i = group.first_sample; i < group.first_sample+group.sample_count; ++i) {
      if (i != group.first_sample) out << ','; const auto &sample = samples[i];
      out << "{\"round\":" << sample.round << ",\"order_position\":" << sample.position
          << ",\"route\":" << Json(route_names[sample.route]) << ",\"epoch\":" << sample.epoch
          << ",\"rank_timings\":[";
      std::array<double, metrics> maxima{};
      for (int r = 0; r < ranks; ++r) {
        if (r) out << ','; out << "{\"rank\":" << r;
        for (int m = 0; m < metrics; ++m) {
          const double value = all[r*per_rank+static_cast<size_t>(i)*metrics+m];
          out << ',' << Json(metric_names[m]) << ':' << value; maxima[m] = std::max(maxima[m], value);
        }
        out << '}';
      }
      out << "],\"maximum_over_ranks\":{";
      for (int m = 0; m < metrics; ++m) { if (m) out << ','; out << Json(metric_names[m]) << ':' << maxima[m]; }
      out << "}}";
    }
    out << "],\"max_over_ranks_then_quantile\":{";
    for (int route = 0; route < 3; ++route) {
      if (route) out << ','; out << Json(route_names[route]) << ":{";
      for (int metric : {operation, mpi, sequence}) {
        if (metric != operation) out << ','; std::vector<double> maxima;
        for (int i = group.first_sample; i < group.first_sample+group.sample_count; ++i)
          if (samples[i].route == route) {
            double maximum = 0;
            for (int r = 0; r < ranks; ++r)
              maximum = std::max(maximum, all[r*per_rank+static_cast<size_t>(i)*metrics+metric]);
            maxima.push_back(maximum);
          }
        out << Json(metric_names[metric]) << ":{\"minimum\":" << Quantile(maxima, 0)
            << ",\"median\":" << Quantile(maxima, .5) << ",\"p95\":" << Quantile(maxima, .95)
            << ",\"maximum\":" << Quantile(maxima, 1) << '}';
      }
      out << '}';
    }
    out << "}}";
  }
  out << "]}\n"; out.close(); Require(out.good(), "output write failed");
  std::filesystem::rename(temporary, output);
}
}  // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv); int rank = 0, ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  bool initialized = false;
  try {
    const auto options = Parse(argc, argv);
    if (options.help) { if (rank == 0) Help(); MPI_Finalize(); return 0; }
    Require(sizeof(Real) == 8, "this control requires the production double-precision datatype");
    Require(options.expected_ranks == 0 || options.expected_ranks == ranks, "unexpected MPI rank count");
    Require(options.expected_backend.empty() || options.expected_backend == Kokkos::DefaultExecutionSpace::name(),
            "unexpected Kokkos backend");
    for (int n : options.grids) { Layout(n, false, rank, ranks); Layout(n, true, rank, ranks); }
    MPI_Comm shared; MPICheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared));
    int shared_size = 0; MPICheck(MPI_Comm_size(shared, &shared_size)); MPICheck(MPI_Comm_free(&shared));
    Require(shared_size == ranks, "benchmark requires one shared-memory node");
    if (rank == 0) {
      Require(!std::filesystem::exists(options.output) && !std::filesystem::exists(options.output+".tmp"),
              "output or temporary file already exists");
      const auto parent = std::filesystem::path(options.output).parent_path();
      if (!parent.empty()) std::filesystem::create_directories(parent);
    }
    std::vector<std::string> command(argv, argv+argc);
    Kokkos::initialize(argc, argv); initialized = true;
    const auto rank_metadata = GatherStrings(RankMetadata(rank, shared_size), rank, ranks);
    std::vector<Group> groups; std::vector<Sample> samples; int epoch = 0;
    if (!options.layout_only) for (int n : options.grids) {
      Payload volume(Layout(n, false, rank, ranks)), surface(Layout(n, true, rank, ranks));
      for (const std::string workload : {"volume", "surface", "rk_mix"}) {
        if (workload == "rk_mix" && options.mix_repeats == 0) continue;
        for (bool synchronized : {true, false}) {
          if (options.mode == "barrier" && !synchronized) continue;
          if (options.mode == "no_barrier" && synchronized) continue;
          Group group{n, workload == "rk_mix" ? options.mix_repeats : options.repeats,
            static_cast<int>(samples.size()), 0, workload, synchronized, {}};
          if (workload == "volume") group.calls = {0};
          else if (workload == "surface") group.calls = {1};
          else group.calls = {1,1,0,1, 1,1,0,1, 1,1,0,1, 1,1, 0,1, 1,1};
          std::vector<Sample> warmup;
          for (int round = 0; round < options.warmup; ++round) for (int position = 0; position < 3; ++position) {
            const int route = orders[round%6][position];
            Sample sample{0, round, position, route, ++epoch, {}};
            sample.values = Run(volume, surface, group, static_cast<Route>(route), epoch);
            warmup.push_back(sample);
          }
          CheckMeasurements(warmup, 0, warmup.size());
          // One group-start barrier, outside all samples, also bounds startup skew.
          MPICheck(MPI_Barrier(MPI_COMM_WORLD));
          for (int round = 0; round < group.repeats; ++round) for (int position = 0; position < 3; ++position) {
            const int route = orders[round%6][position];
            Sample sample{static_cast<int>(groups.size()), round, position, route, ++epoch, {}};
            sample.values = Run(volume, surface, group, static_cast<Route>(route), epoch);
            samples.push_back(sample); ++group.sample_count;
          }
          CheckMeasurements(samples, group.first_sample, samples.size()); groups.push_back(group);
        }
      }
    }
    std::vector<double> local; local.reserve(samples.size()*metrics);
    for (const auto &sample : samples) local.insert(local.end(), sample.values.begin(), sample.values.end());
    Require(local.size() <= std::numeric_limits<int>::max(), "timing payload exceeds MPI count range");
    std::vector<double> all(rank == 0 ? local.size()*ranks : 0);
    MPICheck(MPI_Gather(local.data(), local.size(), MPI_DOUBLE, all.data(), local.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD));
    if (rank == 0) {
      Write(options, ranks, rank_metadata, groups, samples, all, command);
      std::cout << "backend=" << Kokkos::DefaultExecutionSpace::name() << " mpi_size=" << ranks
                << " groups=" << groups.size() << " samples_per_rank=" << samples.size()
                << " mismatches=0 report=" << options.output << "\nCommunication benchmark passed\n";
    }
    Kokkos::finalize(); initialized = false; MPI_Finalize(); return 0;
  } catch (const std::exception &error) {
    std::cerr << "rank=" << rank << " communication benchmark error: " << error.what() << '\n';
    // A rank-local allocation/output/MPI failure must not strand peers in a collective.
    MPI_Abort(MPI_COMM_WORLD, 2);
    if (initialized) Kokkos::finalize();
    return 2;
  }
}
