// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include "z4c_hyperboloidal/multirate_evolution.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include "bvals/multirate_exchange.hpp"
#include "driver/multirate.hpp"
#include "driver/multirate_ready.hpp"
#include "driver/driver.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/multirate_load_balance.hpp"
#include "z4c_hyperboloidal/z4c_hyperboloidal.hpp"
#include "z4c_hyperboloidal/mapped_halo.hpp"
#include "z4c_hyperboloidal/surface_exchange.hpp"
#include "z4c_hyperboloidal/point_rhs.hpp"

namespace z4c_hyperboloidal {
namespace mr = time_integration::multirate;
namespace transport = bvals::multirate;
namespace bal = ::multirate::balance;
namespace {
template <class T>
DvceArray1D<T> Device(const std::vector<T>& v, const char* name) {
  DvceArray1D<T> d(Kokkos::view_alloc(Kokkos::WithoutInitializing,std::string(name)),v.size());
  auto h = Kokkos::create_mirror_view(d);
  for (size_t i = 0; i < v.size(); ++i) h(i) = v[i];
  Kokkos::deep_copy(d, h);
  return d;
}
// Every reader of a stage's pointer table has completed its reduction or fence
// before the next upload. Reuse both allocations across all local stages.
template <class T>
struct StageTable {
  DvceArray1D<T> device;
  typename DvceArray1D<T>::HostMirror host;
  DvceArray1D<T> Upload(const std::vector<T>& rows, const char* name) {
    if (device.extent(0) != rows.size()) {
      device = DvceArray1D<T>(Kokkos::view_alloc(Kokkos::WithoutInitializing,std::string(name)),
                              rows.size());
      host = Kokkos::create_mirror_view(device);
    }
    std::copy(rows.begin(),rows.end(),host.data());
    Kokkos::deep_copy(device,host);
    return device;
  }
};
std::vector<int> ParseLevels(const std::string& s) {
  std::vector<int> out;
  std::istringstream in(s);
  std::string token;
  while (std::getline(in, token, ',')) {
    size_t used = 0;
    int n = std::stoi(token, &used);
    if (token.find_first_not_of(" \t", used) != std::string::npos || n < 0 || n > 20)
      throw std::invalid_argument("multirate spherical_levels requires integers in [0,20]");
    out.push_back(n);
  }
  if (s.empty() || s.back() == ',' || out.empty())
    throw std::invalid_argument("Empty multirate spherical_levels");
  return out;
}
std::string Csv(const std::vector<int>& v) {
  std::ostringstream s;
  for (size_t i = 0; i < v.size(); ++i) s << (i ? "," : "") << v[i];
  return s.str();
}
using Key = transport::RecordKey;
using Point = transport::Point;
struct Combination {
  const Real *base = nullptr, *slope[4]{};
  Real weight[4]{};
  int count = 0;
  KOKKOS_INLINE_FUNCTION Real Value(size_t index, int field) const {
    Real value = base[index * Fields::count + field];
    if (Fields::Independent(field))
      for (int a = 0; a < count; ++a) value += weight[a] * slope[a][index * Fields::independent_count + Fields::IndependentSlot(field)];
    return value;
  }
};
struct NativePoint {
  int m, face, k, j, i, kind, ledger;
  size_t index;
};
// A pointer inside a value type keeps a one-dimensional Kokkos view: a raw
// pointer as the view's element template argument would add another rank.
struct RecordDestination {
  Real* data = nullptr;
};
struct DonorMeta {
  size_t index;
  int gid, kind, support[5]{};
  Real x[3]{}, ref[31]{}, chi_weight = 1, radial_weight[5]{};
};
// One sorted entry per distinct donor point consumed by a fixed local batch.
// The two halo exchanges share ordinals; surface bulk and edge are disjoint.
struct DonorDestination {
  int index, halo = -1, surface = -1, agreement = -1, coupling_row = -1;
};
struct DirectDonorViews {
  DvceArray1D<DonorDestination> targets;
  DvceArray2D<Real> halo_physical, halo_residual, surface_bulk, surface_edge, agreement;
  KOKKOS_INLINE_FUNCTION DonorDestination Destination(int i) const { return targets(i); }
  KOKKOS_INLINE_FUNCTION void Store(const DonorDestination& q, int kind, int field, int slot,
                                    Real u, Real delta) const {
    if (q.halo >= 0) halo_physical(q.halo,field) = u;
    if (slot < 0) return;
    if (q.halo >= 0) halo_residual(q.halo,slot) = delta;
    if (q.surface >= 0) {
      if (kind) surface_edge(q.surface,slot) = delta;
      else surface_bulk(q.surface,slot) = delta;
    }
    if (q.agreement >= 0) agreement(q.agreement,slot) = delta;
  }
};
struct FullDonorViews {
  DvceArray2D<Real> physical, residual;
  KOKKOS_INLINE_FUNCTION DonorDestination Destination(int i) const { return {i}; }
  KOKKOS_INLINE_FUNCTION void Store(const DonorDestination& q, int, int field, int,
                                    Real u, Real delta) const {
    physical(q.index,field) = u;
    residual(q.index,field) = delta;
  }
};
using Clock = std::chrono::steady_clock;
double Seconds(Clock::time_point a) {
  return std::chrono::duration<double>(Clock::now() - a).count();
}
std::vector<int> AllIntegers(const std::vector<int>& local) {
#if MPI_PARALLEL_ENABLED
  int n = local.size();
  std::vector<int> counts(global_variable::nranks), offset(counts.size());
  MPI_Allgather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
  for (size_t i = 1; i < counts.size(); ++i) offset[i] = offset[i - 1] + counts[i - 1];
  std::vector<int> all(offset.back() + counts.back());
  MPI_Allgatherv(local.data(), n, MPI_INT, all.data(), counts.data(), offset.data(), MPI_INT,
                 MPI_COMM_WORLD);
  return all;
#else
  return local;
#endif
}
}  // namespace

void Z4cHyperboloidal::SelectActiveBlocks(const std::vector<int>& blocks) {
  if (blocks == active_slots_) return;
  active_slots_ = blocks;
  std::vector<int> mask(nblocks);
  for (int m : blocks) mask.at(m) = 1;
  active_blocks = Device(mask, "multirate active blocks");
  for (int a = 0; a < 5; ++a) {
    std::vector<int> selected;
    for (int m : blocks) {
      const auto& points = rhs_points_by_block_[a][m];
      selected.insert(selected.end(), points.begin(), points.end());
    }
    active_rhs_points[a] = Device(selected, "multirate active RHS points");
  }
}
void Z4cHyperboloidal::ClearActiveBlocks() {
  active_slots_.clear();
  active_blocks = {};
  for (auto& v : active_rhs_points) v = {};
  active_step_dt = 0;
}
void MultirateEvolution::ReadOptions(ParameterInput& input) {
  if (input.GetOrAddInteger("multirate","send_cache_mib",0)<0)
    throw std::invalid_argument("multirate send_cache_mib must be nonnegative");
  if (input.GetOrAddInteger("multirate","receive_cache_mib",0)<0)
    throw std::invalid_argument("multirate receive_cache_mib must be nonnegative");
  const auto levels = input.GetOrAddString("multirate", "spherical_levels", "auto");
  if (levels != "auto") ParseLevels(levels);
  const auto balance = input.GetOrAddString("multirate", "balance", "metis");
  if (balance != "none" && balance != "scalar" && balance != "metis" && balance != "spread")
    throw std::invalid_argument("multirate balance must be none, scalar, metis, or spread");
  if (input.GetOrAddInteger("multirate","ranks_per_gpu",1)<1)
    throw std::invalid_argument("ranks_per_gpu must be positive");
  const auto cfl = input.GetOrAddReal("multirate", "max_characteristic_cfl", .35);
  if (!(cfl > 0 && std::isfinite(cfl)))
    throw std::invalid_argument("Invalid monitored multirate CFL bound");
  if (input.GetOrAddInteger("multirate", "warmup_intervals", 32) < 1 ||
      input.GetOrAddInteger("multirate", "review_interval", 128) < 1)
    throw std::invalid_argument("Multirate warmup and review periods must be positive");
  if (input.GetOrAddInteger("multirate", "stage_budget", 1048576) < 4)
    throw std::invalid_argument("Multirate stage budget must cover at least four stages");
  input.GetOrAddBoolean("multirate", "uniform_reference", false);
  if (input.GetOrAddBoolean("multirate","coalesce_stage_packets",false) &&
      (!input.GetOrAddBoolean("multirate","complete_local_batches",false) ||
       !input.GetOrAddBoolean("multirate","prepost_exchanges",true)))
    throw std::invalid_argument("coalesce_stage_packets requires complete_local_batches and prepost_exchanges");
  input.GetOrAddBoolean("multirate","prepost_exchanges",true);
  input.GetOrAddBoolean("multirate","complete_local_batches",false);
  input.GetOrAddBoolean("multirate","profile_peer_bytes",false);
  input.GetOrAddString("multirate", "replay_file", "");
  input.GetOrAddString("multirate", "partition_log", "");
}
std::string Z4cHyperboloidal::MultirateIdentity() const {
  if (!multirate_enabled_) return "";
  return ";multirate=EX5-EX5-4-M2-v1;spherical-levels=" +
         pinput_->GetOrAddString("multirate", "spherical_levels", "auto") + ";uniform-reference=" +
         (pinput_->GetOrAddBoolean("multirate", "uniform_reference", false) ? "true" : "false") +
         ";agreement=shared-stage-v1";
}

struct MultirateEvolution::Impl {
  Z4cHyperboloidal& p;
  MeshBlockPack* pack;
  Mesh* mesh;
  ParameterInput* pin;
  std::vector<chart_atlas::BlockMap> maps;
  std::vector<int> levels, steps;
  std::vector<std::set<int>> neighbors;
  std::vector<mr::DependencyEdge> edges;
  std::map<std::pair<int, int>, std::uint64_t> bytes_by_direction;
  std::unique_ptr<mr::Schedule> schedule;
  std::unique_ptr<transport::RecordExchange> exchange;
  std::vector<NativePoint> native_host;
  DvceArray1D<NativePoint> native;
  StageTable<Combination> reconstruction_rows, donor_rows;
  StageTable<RecordDestination> destinations;
  struct GatherPlan {
    DvceArray1D<int> index, fields;
    DvceArray2D<Real> buffer;
  };
  DvceArray2D<Real> donor_physical, donor_residual;
  std::map<const bvals::MappedExchange*, GatherPlan> gather_plans;
  std::vector<Point> requested;
  std::map<Point, int> ordinal;
  std::vector<DonorMeta> donor_host;
  DvceArray1D<DonorMeta> donors;
  struct Task {
    int gid, m;
    size_t stage;
    std::vector<Key> dependencies;
    double interior_seconds = 0;
  };
  using BatchKey = std::tuple<int, size_t, bal::KernelKind>;
  mr::RecordReadiness<Key, BatchKey> readiness, interior_readiness;
  bool complete_local_batches = false, peer_profile_pending = false;
  struct ActiveLayout {
    DvceArray1D<int> mask, native_indices;
    std::array<DvceArray1D<int>, 5> rhs_points;
    DirectDonorViews donor_views;
    std::vector<int> donor_gids;
    StageTable<Combination> coupling;
    bool donor_views_ready = false;
  };
  std::map<std::vector<int>, ActiveLayout> active_layouts;
  DvceArray1D<int> active_native_indices;
  void SelectBatch(const std::vector<int>& slots);
  std::vector<Task> tasks;
  std::map<Key, size_t> consumers;
  bal::WorkGraph graph;
  std::vector<bal::ExecutionTask> scoring_tasks;
  bal::BatchCostModel model;
  bal::BalanceOptions options;
  bal::RebalancePolicy policy;
  std::string balance, replay_path, log_path;
  std::vector<bal::PartitionRecord> replay;
  double max_cfl = .35, migration_cost = 0;
  explicit Impl(Z4cHyperboloidal& module, Driver* driver, bool rebuilt)
      : p(module), pack(p.pmy_pack), mesh(pack->pmesh), pin(p.pinput_) {
    max_cfl = pin->GetReal("multirate", "max_characteristic_cfl");
    balance = pin->GetString("multirate", "balance");
    options.backend = balance == "scalar" ? bal::Backend::scalar_reference : bal::Backend::metis;
    if (mesh->adaptive) options.maximum_blocks_per_rank = mesh->nmb_maxperrank;
    options.consider_sparse_compaction = balance == "metis";
    options.spread_finest = balance == "spread";
    options.ranks_per_gpu = pin->GetInteger("multirate","ranks_per_gpu");
    policy.warmup_intervals = pin->GetInteger("multirate", "warmup_intervals");
    policy.review_interval = pin->GetInteger("multirate", "review_interval");
    replay_path = pin->GetString("multirate", "replay_file");
    log_path = pin->GetString("multirate", "partition_log");
    BuildGraphAndLevels();
    BuildCostGraph();
    if (!replay_path.empty()) {
      std::ifstream in(replay_path);
      if (!in) throw std::runtime_error("Cannot read multirate replay_file");
      size_t bytes;
      while (in >> std::ws && !in.eof()) {
        if (!(in >> bytes)) throw std::runtime_error("Invalid partition replay length");
        if (in.get() != '\n' || bytes > 100000000)
          throw std::runtime_error("Invalid partition replay framing");
        std::string value(bytes, '\0');
        in.read(value.data(), bytes);
        if (!in) throw std::runtime_error("Truncated partition replay");
        auto r = bal::PartitionRecord::Deserialize(value);
        if (mesh->adaptive && r.graph_fingerprint != graph.Fingerprint()) continue;
        r.ValidateFor(graph, global_variable::nranks);
        if (!replay.empty() && r.interval <= replay.back().interval)
          throw std::runtime_error("Partition replay intervals must increase");
        replay.push_back(std::move(r));
      }
      if (replay.empty() || replay.front().interval > static_cast<std::uint64_t>(mesh->ncycle))
        throw std::runtime_error(
            "Replay needs an initial partition at or before the starting endpoint");
    }
    auto owner = Owners();
    if (!replay.empty()) {
      for (const auto& r : replay)
        if (r.interval <= static_cast<std::uint64_t>(mesh->ncycle)) owner = r.owner_by_gid;
    } else if (balance != "none" && (mesh->ncycle == 0 || rebuilt)) {
      const auto initial =
          bal::Partition(graph, global_variable::nranks, options, model, scoring_tasks,
                         balance == "scalar" && !mesh->adaptive ? std::vector<int>{} : owner);
      owner = initial.owner_by_gid;
      if (global_variable::my_rank == 0)
        std::cout << "MULTIRATE_PARTITION cycle=" << mesh->ncycle << " candidate=" << initial.selected_candidate
                  << " predicted_seconds=" << initial.score.seconds << "\n";
    }
    if (owner != Owners()) {
      const auto start = Clock::now();
      mesh->RedistributeStaticBlocks(owner);
      migration_cost = Seconds(start);
      p.InitializeBoundaryValues(driver);
    }
    BuildLocal();
    pin->SetString("multirate", "block_owners", Csv(owner));
    if (rebuilt || mesh->ncycle == 0 || !std::ifstream(log_path).good()) LogPartition(mesh->ncycle);
    if (global_variable::my_rank == 0)
      std::cout << "Multirate graph fingerprint=" << graph.Fingerprint()
                << " levels=" << Csv(levels) << " tasks=" << scoring_tasks.size() << "\n";
  }
  std::vector<int> Owners() const {
    return {mesh->rank_eachmb, mesh->rank_eachmb + mesh->nmb_total};
  }
  void BuildGraphAndLevels();
  void BuildCostGraph();
  void SetPhaseCosts(const bal::BatchCostModel& profile);
  void BuildLocal();
  Combination Combine(size_t stage, int gid) const;
  void Reconstruct(const std::vector<int>& slots, const std::vector<Combination>& combinations,
                   double time, bool accepted);
  std::vector<DvceArray1D<Real>> Pack(const std::vector<int>& slots, bool rhs);
  void Publish(const std::vector<int>& slots, const std::vector<DvceArray1D<Real>>& records,
               std::uint32_t step, int stage);
  GatherPlan& ExchangeView(const bvals::MappedExchange& plan, int kind);
  DirectDonorViews MakeDonorViews(const std::vector<int>& slots, std::vector<int>& donor_gids);
  DvceArray2D<Real> Gather(const bvals::MappedExchange& plan, int kind,
                           const DvceArray2D<Real>& values);
  void StageDonors(size_t stage, const std::vector<int>& slots, DvceArray2D<Real>& physical,
                   DvceArray2D<Real>& residual);
  Clock::time_point interval_start;
  std::size_t completed = 0;
  bool interval_active = false;
  std::vector<unsigned long long> level_work;
  unsigned long long batch_count = 0, max_batch = 0, donor_table_bytes = 0;
  double communication_wait = 0;
  // Host elapsed time in disjoint parts of task-list execution. Kernel launches
  // may complete in a later phase. These counters never introduce a device fence.
  std::array<double,5> phase_seconds{};
  unsigned long long interior_batches_with_receives = 0, interior_receives = 0;
  void BeginInterval();
  void Evaluate(Driver*, bool interior, const std::vector<size_t>& batch);
  bool Poll(Driver*);
  void CompleteInterval(Driver*);
  void LogPartition(std::uint64_t interval) {
    if (global_variable::my_rank != 0) return;
    const auto owner=Owners();
    std::vector<int> finest(global_variable::nranks);
    for (const auto& b : graph.blocks)
      if (b.time_level==schedule->MaxLevel()) ++finest[owner[b.gid]];
    std::cout<<"MULTIRATE_PLACEMENT cycle="<<interval<<" finest_level="<<schedule->MaxLevel()
             <<" blocks_by_rank="<<Csv(finest)<<"\n";
    if (log_path.empty()) return;
    bal::PartitionRecord r{interval, graph.Fingerprint(), global_variable::nranks, Owners()};
    const auto s = r.Serialize();
    std::ofstream out(log_path, interval == 0 ? std::ios::trunc : std::ios::app);
    if (!out) throw std::runtime_error("Cannot write partition_log");
    out << s.size() << '\n' << s;
  }
};
void MultirateEvolution::Impl::BuildGraphAndLevels() {
  if (!mesh->atlas) throw std::invalid_argument("Scheme A multirate requires its mapped atlas");
  const int total = mesh->nmb_total;
  maps.resize(total);
  neighbors.resize(total);
  std::vector<mr::BlockSpec> blocks(total);
  int regions = 0;
  for (int g = 0; g < total; ++g) {
    maps[g] = mesh->atlas->GetBlockMap(mesh->lloc_eachmb[g]);
    const auto& m = maps[g];
    regions = std::max(regions, m.layer + 1);
    blocks[g] = {g,       m.face < 0, mesh->lloc_eachmb[g].level - mesh->root_level,
                 m.layer, -1,         m.outer_surface ? 0 : -1};
  }
  std::map<std::pair<int, int>, std::set<Point>> dependencies;
  std::set<int> scri_support;
  auto add = [&](int recipient, const bvals::DonorPoint& point, int kind) {
    if (recipient == point.gid) return;
    auto& set = dependencies[{recipient, point.gid}];
    set.insert({point.gid, point.k, point.j, point.i, kind});
    if (kind)
      for (int k = 0; k < 5; ++k)
        set.insert(
            {point.gid, p.origin + point.k, p.origin + point.j, p.origin + p.n[0] - 5 + k, 0});
  };
  for (const auto& r : p.mapped_halo_->Rows())
    for (const auto& d : r.donors) add(pack->GlobalId(r.target.m), d.point, 0);
  const auto rows =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.surface_exchange_->Rows());
  const auto terms =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.surface_exchange_->Terms());
  for (int q = 0; q < p.nsurface_points; ++q) {
    const int gid = pack->GlobalId(p.face_blocks.h_view(p.surface_points.h_view(q).face));
    for (int t = rows(q).begin; t < rows(q).end; ++t) {
      const auto term = terms(t);
      const auto& ex =
          term.boundary ? p.surface_exchange_->EdgeExchange() : p.surface_exchange_->BulkExchange();
      const auto donor = ex.UniqueDonors().at(term.ordinal);
      add(gid, donor, term.boundary);
      scri_support.insert(donor.gid);
    }
  }
  // Radial bulk closure rows have local bulk stencil values and exchanged edge values.
  const auto kernel = p.surface_exchange_->BindKernel();
  const auto radial_rows =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), kernel.radial_rows);
  const auto radial_index =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), kernel.radial_index);
  const auto radial_terms =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), kernel.radial_terms);
  for (int q = 0; q < p.npoints; ++q)
    if (radial_index(q) >= 0) {
      const int gid = pack->GlobalId(p.points.h_view(q).m);
      const auto row = radial_rows(radial_index(q));
      for (int t = row.begin; t < row.end; ++t)
        if (radial_terms(t).boundary) {
          const auto d =
              p.surface_exchange_->EdgeExchange().UniqueDonors().at(radial_terms(t).ordinal);
          add(gid, d, 1);
          scri_support.insert(d.gid);
        }
    }
  for (const auto& recipient : p.agreement_plan_->MultirateDonors())
    for (const auto& donor : recipient.second) add(recipient.first, donor, 0);
  std::vector<int> local;
  for (const auto& entry : dependencies)
    local.insert(local.end(),
                 {entry.first.first, entry.first.second, static_cast<int>(entry.second.size())});
  const auto all = AllIntegers(local);
  for (size_t i = 0; i < all.size(); i += 3) {
    const int recipient = all[i], donor = all[i + 1];
    bytes_by_direction[{recipient, donor}] =
        std::uint64_t(all[i + 2]) * Fields::count * sizeof(Real);
    edges.push_back({recipient, donor});
    neighbors[recipient].insert(donor);
  }
  const auto support = AllIntegers(std::vector<int>(scri_support.begin(), scri_support.end()));
  for (int g : support) blocks[g].scri_group = 0;
  for (int gid : p.agreement_plan_->AgreementBlocks()) blocks.at(gid).agreement_group = 0;
  const bool uniform = pin->GetOrAddBoolean("multirate", "uniform_reference", false);
  auto input = pin->GetOrAddString("multirate", "spherical_levels", "auto");
  std::vector<int> spherical(regions);
  std::vector<double> rates(total);
  // Value-only characteristic estimate, using precisely the RHS speed formula.
  const auto state = p.u0;
  const auto geometry = p.geometry;
  const auto cells = p.points.d_view;
  const auto parameters = p.parameters;
  const int resolution = p.n[0];
  DvceArray1D<double> local_rates("multirate block characteristic maxima", p.nblocks);
  int invalid = 0;
  Kokkos::parallel_reduce("multirate topology stability bound", Kokkos::RangePolicy<>(0,p.npoints),
    KOKKOS_LAMBDA(int point, int &bad) {
    const auto c = cells(point);
    const auto map = geometry(point);
    CartesianChartJets j{};
    for (int f = 0; f < Fields::count; ++f) j[f].f = state(c.m, f, c.k, c.j, c.i);
    const auto u = SphericalPolicy::Physical(map.x, j);
    const auto g = Geometry(u.metric);
    const auto coefficient =
        ShiftCoefficients(SphericalPolicy::Radius(map.x), u.alpha.value, u.chi.value, parameters);
    Real chi = u.chi.value, rate = 0;
    if (parameters.cartesian_chi_guard) {
      const auto omega = parameters.profile.At(map.x);
      chi += coefficient.inner *
             (fmax(chi, parameters.inner_chi_guard / (omega.omega * omega.omega)) - chi);
    }
    for (int a = 0; a < 3; ++a) {
      Real transport = 0, Q = 0, E = 0;
      for (int b = 0; b < 3; ++b) {
        transport += map.J[a][b] * u.beta.value[b];
        E += map.J[a][b] * map.J[a][b];
        for (int d = 0; d < 3; ++d) Q += g.inverse[b][d] * map.J[a][b] * map.J[a][d];
      }
      Real speed2 =
          fmax(GaugeShiftSpeedSquared(coefficient, Q),
               GaugeLapseSpeedSquared(u.alpha.value, chi, coefficient, Q, E, parameters));
      if (parameters.enforce_cfl) speed2 = fmax(speed2, u.alpha.value * u.alpha.value * chi * Q);
      rate += resolution * (fabs(transport) + sqrt(speed2));
    }
    if (Kokkos::isfinite(rate)) Kokkos::atomic_max(&local_rates(c.m), double(rate));
    else ++bad;
  }, invalid);
  const auto maxima = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), local_rates);
  for (int m=0; m<p.nblocks; ++m) rates[pack->GlobalId(m)] = maxima(m);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, rates.data(), total, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
  if (invalid) throw std::runtime_error("Invalid multirate characteristic bound on accepted mesh");
  auto required = [&](int gid) {
    const double ratio =
        std::max(p.requested_dt * rates[gid] / max_cfl,
                 p.requested_dt / (.25 * std::min(p.parameters.tau, Real(InnerTelegrapher::tau))));
    return ratio <= 1 ? 0 : int(std::ceil(std::log2(ratio)));
  };
  if (uniform) {
    for (auto& b : blocks) b.physical_level = 0;
    input = Csv(spherical);  // Explicit qualification path, never a production default.
  } else if (input == "auto") {
    for (const auto& b : blocks)
      if (!b.cartesian)
        spherical[b.radial_region] = std::max(spherical[b.radial_region], required(b.gid));
    // Raise only automatic spherical settings and declared Cartesian agreement members.
    for (int iteration = 0; iteration < 64; ++iteration) {
      bool changed = false;
      int group = 0;
      for (const auto& b : blocks)
        if (b.agreement_group >= 0)
          group = std::max(group, b.cartesian ? b.physical_level : spherical[b.radial_region]);
      while (std::ldexp(p.requested_dt, -group) * p.agreement_plan_->RateBound() > 2.5) ++group;
      for (const auto& b : blocks)
        if (b.agreement_group >= 0 && !b.cartesian && spherical[b.radial_region] < group) {
          spherical[b.radial_region] = group;
          changed = true;
        }
      int closure_level = 0;
      for (const auto& b : blocks)
        if (b.scri_group >= 0 && !b.cartesian)
          closure_level = std::max(closure_level, spherical[b.radial_region]);
      for (const auto& b : blocks)
        if (b.scri_group >= 0 && !b.cartesian && spherical[b.radial_region] < closure_level) {
          spherical[b.radial_region] = closure_level;
          changed = true;
        }
      auto level = [&](int g) {
        const auto& b = blocks[g];
        return b.cartesian ? (b.agreement_group >= 0 ? group : b.physical_level)
                           : spherical[b.radial_region];
      };
      for (const auto& e : edges) {
        int a = e.first, b = e.second;
        if (level(a) > level(b)) std::swap(a, b);
        if (level(b) > level(a) + 1 && !blocks[a].cartesian) {
          spherical[blocks[a].radial_region] = level(b) - 1;
          changed = true;
        }
      }
      if (!changed) break;
      if (iteration == 63)
        throw std::runtime_error("Automatic temporal-level balancing did not converge");
    }
    input = Csv(spherical);
  } else
    spherical = ParseLevels(input);
  if (spherical.size() != static_cast<size_t>(regions))
    throw std::invalid_argument(
        "spherical_levels must contain exactly one level per physical radial region");
  const auto assignment = mr::AssignLevels(blocks, spherical, edges);
  levels = assignment.levels;
  for (int g = 0; g < total; ++g) {
    if (levels[g] < required(g))
      throw std::invalid_argument(
          "Multirate fixed step violates initial stability bound on GID " + std::to_string(g) +
          " at temporal level " + std::to_string(levels[g]) +
          "; reduce time/dt or increase an explicit spherical level consistently with donor edges");
    if (blocks[g].agreement_group >= 0 &&
        std::ldexp(p.requested_dt, -levels[g]) * p.agreement_plan_->RateBound() > 2.5)
      throw std::invalid_argument("Multirate shared agreement timestep exceeds stability bound");
  }
  if (global_variable::my_rank == 0) {
    int limiting = 0;
    double cartesian = 0, spherical_max = 0;
    for (int gid = 0; gid < total; ++gid) {
      const double local = std::ldexp(p.requested_dt, -levels[gid]) * rates[gid];
      if (local > std::max(cartesian, spherical_max)) limiting = gid;
      auto& maximum = blocks[gid].cartesian ? cartesian : spherical_max;
      maximum = std::max(maximum, local);
    }
    std::cout << "MULTIRATE_CFL cycle=" << mesh->ncycle << " limiting_gid=" << limiting
              << " time_level=" << levels[limiting] << " spherical_region=" << maps[limiting].layer
              << " cartesian_max=" << cartesian << " spherical_max=" << spherical_max
              << " bound=" << max_cfl << "\n";
  }
  if (!mesh->adaptive) pin->SetString("multirate", "spherical_levels", Csv(spherical));
  const int budget = pin->GetOrAddInteger("multirate", "stage_budget", 1048576);
  if (budget < 4)
    throw std::invalid_argument("Multirate stage_budget must cover at least four stages");
  schedule = std::make_unique<mr::Schedule>(assignment.maximum_level, budget);
  steps.resize(total);
  for (int g = 0; g < total; ++g) steps[g] = schedule->Steps(levels[g]);
  if (global_variable::my_rank == 0)
    std::cout << "Multirate agreement participants=" << p.agreement_plan_->AgreementBlocks().size()
              << " Cartesian promotions=" << assignment.agreement_promotions.size()
              << " spherical levels=" << Csv(spherical) << "\n";
}

void MultirateEvolution::Impl::BuildCostGraph() {
  graph = {};
  scoring_tasks.clear();
  using Stage = std::pair<int, size_t>;
  std::map<Stage, int> input_task, rhs_task, slope_task, accepted_task;
  // Actual local point counts include evolved overlap and scri work.
  std::vector<int> counts(mesh->nmb_total, 0), edgecounts(mesh->nmb_total, 0);
  for (int q = 0; q < p.npoints; ++q) ++counts[pack->GlobalId(p.points.h_view(q).m)];
  for (int q = 0; q < p.nsurface_points; ++q)
    ++edgecounts[pack->GlobalId(p.face_blocks.h_view(p.surface_points.h_view(q).face))];
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, counts.data(), counts.size(), MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, edgecounts.data(), edgecounts.size(), MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
#endif
  for (int g = 0; g < mesh->nmb_total; ++g) {
    const auto kind = maps[g].outer_surface ? bal::KernelKind::scri
                                            : (maps[g].face < 0 ? bal::KernelKind::cartesian
                                                                : bal::KernelKind::spherical);
    const std::uint64_t units = counts[g] + edgecounts[g];
    bal::BlockCost b{g, levels[g], kind, units, units * Fields::count * sizeof(Real) * 8, {}};
    graph.blocks.push_back(std::move(b));
    const auto& group = p.agreement_plan_->AgreementBlocks();
    const bool agreement = std::binary_search(group.begin(), group.end(), g);
    for (std::uint32_t step = 0; step < schedule->Steps(levels[g]); ++step)
      for (int stage = 0; stage < 4; ++stage) {
        const auto id = schedule->Id(levels[g], step, stage);
        const Stage key{g, id};
        auto append = [&](unsigned phase, bal::KernelKind op, int predecessor) {
          const int index = scoring_tasks.size();
          scoring_tasks.push_back({g, 4 * id + phase, op, units, {}});
          if (predecessor >= 0) scoring_tasks.back().dependencies.push_back({predecessor, 0});
          return index;
        };
        input_task[key] = append(0, bal::KernelKind::projection, -1);
        rhs_task[key] = append(1, kind, input_task[key]);
        slope_task[key] =
            agreement ? append(2, bal::KernelKind::agreement, rhs_task[key]) : rhs_task[key];
        if (stage == 3)
          accepted_task[key] = append(3, bal::KernelKind::projection, slope_task[key]);
      }
  }
  struct RecordCounts {std::uint64_t states, slopes;};
  std::map<std::pair<int,int>,RecordCounts> frequencies;
  const bool planned=pin->GetBoolean("multirate","prepost_exchanges");
  for (const auto& e : bytes_by_direction) {
    const int recipient=e.first.first, donor=e.first.second;
    const auto pair=std::make_pair(levels[recipient],levels[donor]);
    auto [frequency,inserted]=frequencies.emplace(pair,RecordCounts{std::uint64_t(steps[donor])+1,4ULL*steps[donor]});
    if (planned && inserted) {
      std::set<std::pair<int,int>> used;
      for (int step=0;step<steps[recipient];++step) for(int stage=0;stage<4;++stage) {
        const auto& row=schedule->Row(schedule->Id(pair.first,step,stage),pair.second);
        used.emplace(row.completed_steps,-1);
        for(int t=0;t<row.count;++t) {
          const auto& key=schedule->Stage(row.terms[t].source).key;
          used.emplace(key.local_step,key.stage);
        }
      }
      const std::size_t accepted=std::count_if(used.begin(),used.end(),[](const auto &k){return k.second==-1;});
      frequency->second={accepted,used.size()-accepted};
    }
    const auto [states,slopes]=frequency->second;
    if (states) graph.edges.push_back({recipient,donor,e.second,states});
    if (slopes) graph.edges.push_back({recipient,donor,e.second/Fields::count*Fields::independent_count,slopes});
  }
  for (const auto& entry : input_task) {
    const auto [g, stage] = entry.first;
    auto coupled = neighbors[g];
    coupled.insert(g);
    for (int donor : coupled) {
      auto& task = scoring_tasks[donor == g ? entry.second : rhs_task.at(entry.first)];
      const auto bytes = donor == g ? 0 : bytes_by_direction.at({g, donor});
      const auto& row = schedule->Row(stage, levels[donor]);
      if (row.completed_steps) {
        auto id = schedule->Id(levels[donor], row.completed_steps - 1, 3);
        task.dependencies.push_back({accepted_task.at({donor, id}), bytes});
      }
      for (int i = 0; i < row.count; ++i)
        task.dependencies.push_back({slope_task.at({donor, row.terms[i].source}), bytes/Fields::count*Fields::independent_count});
    }
  }
  SetPhaseCosts(model);
  graph.Validate();
}
void MultirateEvolution::Impl::SetPhaseCosts(const bal::BatchCostModel& profile) {
  for (auto& block : graph.blocks) block.phase_costs.assign(3 * (schedule->MaxLevel() + 1), 0.);
  for (const auto& task : scoring_tasks) {
    auto& block = graph.blocks[task.block];
    const int phase = task.kernel == bal::KernelKind::projection
                          ? 1
                          : (task.kernel == bal::KernelKind::agreement ? 2 : 0);
    block.phase_costs[3 * block.time_level + phase] +=
        profile.Predict(task.kernel, task.work_units);
  }
}
void MultirateEvolution::Impl::BuildLocal() {
  peer_profile_pending = pin->GetBoolean("multirate", "profile_peer_bytes");
  active_layouts.clear();
  active_native_indices = {};
  complete_local_batches = pin->GetBoolean("multirate", "complete_local_batches");
  readiness = {};
  interior_readiness = {};
  native_host.clear();
  requested.clear();
  ordinal.clear();
  tasks.clear();
  consumers.clear();
  std::vector<Point> owned;
  for (int q = 0; q < p.npoints; ++q) {
    const auto c = p.points.h_view(q);
    owned.push_back({pack->GlobalId(c.m), c.k, c.j, c.i, 0});
  }
  for (int q = 0; q < p.nsurface_points; ++q) {
    const auto c = p.surface_points.h_view(q);
    owned.push_back({pack->GlobalId(p.face_blocks.h_view(c.face)), c.k, c.j, 0, 1});
  }
  auto request = [&](const bvals::MappedExchange& ex, int kind) {
    for (const auto& q : ex.UniqueDonors()) requested.push_back({q.gid, q.k, q.j, q.i, kind});
  };
  request(p.mapped_halo_->ResidualExchange(), 0);
  request(p.surface_exchange_->BulkExchange(), 0);
  request(p.surface_exchange_->EdgeExchange(), 1);
  request(p.agreement_plan_->Exchange(), 0);
  gather_plans.clear();
  const auto original = requested;
  for (const auto& q : original)
    if (q.kind)
      for (int i = 0; i < 5; ++i)
        requested.push_back({q.gid, p.origin + q.k, p.origin + q.j, p.origin + p.n[0] - 5 + i, 0});
  std::sort(requested.begin(), requested.end());
  requested.erase(std::unique(requested.begin(), requested.end()), requested.end());
  auto scratch=[&](const char* name) {
    return DvceArray2D<Real>(Kokkos::view_alloc(Kokkos::WithoutInitializing,std::string(name)),
                             requested.size(),Fields::count);
  };
  if (!complete_local_batches) {
    donor_physical=scratch("multirate donor physical");
    donor_residual=scratch("multirate donor residual");
  }
  exchange = std::make_unique<transport::RecordExchange>(pack, owned, requested, Fields::count);
  for (int q = 0; q < p.npoints; ++q) {
    const auto c = p.points.h_view(q);
    const Point point{pack->GlobalId(c.m), c.k, c.j, c.i, 0};
    native_host.push_back({c.m, -1, c.k, c.j, c.i, 0, q, exchange->PointIndex(point)});
  }
  for (int q = 0; q < p.nsurface_points; ++q) {
    const auto c = p.surface_points.h_view(q);
    const int m = p.face_blocks.h_view(c.face);
    const Point point{pack->GlobalId(m), c.k, c.j, 0, 1};
    native_host.push_back({m, c.face, c.k, c.j, 0, 1, q, exchange->PointIndex(point)});
  }
  native = Device(native_host, "multirate native record layout");
  for (size_t q = 0; q < requested.size(); ++q) ordinal.emplace(requested[q], q);
  donor_host.clear();
  SphericalReference3D reference(p.parameters);
  for (const auto& point : requested) {
    DonorMeta meta{};
    meta.index = exchange->PointIndex(point);
    meta.gid = point.gid;
    meta.kind = point.kind;
    const auto& map = maps[point.gid];
    const Real q[3] = {point.kind ? 1 : (point.i - p.origin + map.node_offset[0]) / p.n[0],
                       (point.j - (point.kind ? 0 : p.origin) + map.node_offset[1]) / p.n[0],
                       (point.k - (point.kind ? 0 : p.origin) + map.node_offset[2]) / p.n[0]};
    map.Forward(q, meta.x);
    const auto ref = reference.At(meta.x, point.kind);
    for (int f = 0; f < Fields::count; ++f) meta.ref[f] = ref[f].f;
    meta.chi_weight = point.kind ? 1 : p.parameters.residual_weight.Value(meta.x);
    if (point.kind)
      for (int i = 0; i < 5; ++i) {
        meta.support[i] = ordinal.at(
            {point.gid, p.origin + point.k, p.origin + point.j, p.origin + p.n[0] - 5 + i, 0});
        meta.radial_weight[i] = 1;
        for (int j = 0; j < 5; ++j)
          if (i != j) meta.radial_weight[i] *= (5 - map.node_offset[0] - j) / Real(i - j);
      }
    donor_host.push_back(meta);
  }
  donors = Device(donor_host, "multirate donor geometry");
  for (int m = 0; m < p.nblocks; ++m) {
    const int gid = pack->GlobalId(m);
    auto coupled = neighbors[gid];
    coupled.insert(gid);
    for (std::uint32_t step = 0; step < schedule->Steps(levels[gid]); ++step)
      for (int stage = 0; stage < 4; ++stage) {
        Task task{gid, m, schedule->Id(levels[gid], step, stage), {}, 0.};
        std::set<Key> keys;
        for (int donor : coupled) {
          const auto& r = schedule->Row(task.stage, levels[donor]);
          keys.insert({donor, r.completed_steps, -1});
          for (int i = 0; i < r.count; ++i) {
            const auto& k = schedule->Stage(r.terms[i].source).key;
            keys.insert({donor, k.local_step, k.stage});
          }
        }
        task.dependencies.assign(keys.begin(), keys.end());
        for (const auto& key : keys) ++consumers[key];
        const BatchKey batch{-static_cast<int>(schedule->Stage(task.stage).critical_depth),
                             task.stage, graph.blocks[gid].kernel};
        std::vector<Key> own_keys, halo_keys = task.dependencies;
        for (const auto& key : task.dependencies)
          if (key.gid == gid) own_keys.push_back(key);
        interior_readiness.Add(batch, own_keys);
        // Local completion event. Stages 4..7 never occur in a wire record.
        halo_keys.push_back({gid, step, stage + 4});
        readiness.Add(batch, halo_keys);
        tasks.push_back(std::move(task));
      }
    ++consumers[{gid, static_cast<std::uint32_t>(steps[gid]), -1}];
  }
  exchange->SetSlopeFields(Fields::independent_count);
  exchange->SetReceiveCacheBytes(static_cast<std::size_t>(pin->GetInteger("multirate","receive_cache_mib"))*1024*1024);
  exchange->SetSendCacheBytes(static_cast<std::size_t>(pin->GetInteger("multirate","send_cache_mib"))*1024*1024);
  if (pin->GetBoolean("multirate","prepost_exchanges")) {
    std::vector<int> groups;
    if (pin->GetBoolean("multirate","coalesce_stage_packets")) {
      // Group only blocks published by the very same level/stage/kernel batch.
      groups.resize(graph.blocks.size());
      for (const auto& block : graph.blocks)
        groups[block.gid]=block.time_level*static_cast<int>(bal::KernelKind::count)
                          + static_cast<int>(block.kernel);
    }
    exchange->CompileReceives(consumers,groups);
  }
}
Combination MultirateEvolution::Impl::Combine(size_t stage, int gid) const {
  const auto& r = schedule->Row(stage, levels[gid]);
  Combination c;
  const Key base{gid, r.completed_steps, -1};
  if (!exchange->Has(base)) return c;
  c.base = exchange->Data(base).data();
  c.count = r.count;
  for (int a = 0; a < r.count; ++a) {
    const auto& k = schedule->Stage(r.terms[a].source).key;
    const Key key{gid, k.local_step, k.stage};
    if (!exchange->Has(key)) return Combination{};
    c.slope[a] = exchange->Data(key).data();
    c.weight[a] = p.requested_dt * r.terms[a].coefficient;
  }
  return c;
}
void MultirateEvolution::Impl::SelectBatch(const std::vector<int>& slots) {
  if (p.active_slots_ == slots) return;
  if (!complete_local_batches) {
    p.SelectActiveBlocks(slots);
    return;
  }
  // Membership of each complete local stage/kernel batch is fixed for a
  // placement. Reuse its device point lists across stages and base intervals.
  auto [layout, inserted] = active_layouts.try_emplace(slots);
  if (inserted) {
    p.SelectActiveBlocks(slots);
    layout->second.mask = p.active_blocks;
    layout->second.rhs_points = p.active_rhs_points;
    std::vector<int> indices;
    if (slots.size() != static_cast<std::size_t>(p.nblocks)) {
      // Complete local batches have a fixed membership until redistribution or AMR.
      // The index table covers owned bulk points and owned scri points together.
      std::vector<bool> selected(p.nblocks);
      for (int m : slots) selected[m] = true;
      for (std::size_t q = 0; q < native_host.size(); ++q)
        if (selected[native_host[q].m]) indices.push_back(q);
    }
    layout->second.native_indices = Device(indices,"multirate active native ordinals");
  } else {
    p.active_slots_ = slots;
    p.active_blocks = layout->second.mask;
    p.active_rhs_points = layout->second.rhs_points;
  }
  active_native_indices = layout->second.native_indices;
}
void MultirateEvolution::Impl::Reconstruct(const std::vector<int>& slots,
                                           const std::vector<Combination>& combinations,
                                           double time, bool accepted) {
  SelectBatch(slots);
  const auto combinations_d = reconstruction_rows.Upload(combinations,"multirate reconstruction rows");
  const auto points = native;
  const auto state = p.u0;
  const auto edge = p.s0;
  const auto active = p.active_blocks;
  const auto records = p.chi_records, srecords = p.surface_chi_records;
  const Real floor = p.chi_floor;
  const auto geom = p.geometry;
  const auto reference = p.reference;
  const auto residual = p.residual;
  const auto weight = p.parameters.residual_weight;
  const auto indices = active_native_indices;
  const bool indexed = complete_local_batches && slots.size() != static_cast<std::size_t>(p.nblocks);
  const int count = indexed ? indices.extent_int(0) : points.extent_int(0);
  int invalid = 0;
  p.InvalidateDiagnostics();
  Kokkos::parallel_reduce(
      "multirate reconstruct and project owned stage", Kokkos::RangePolicy<>(0, count),
      KOKKOS_LAMBDA(int i, int& bad) {
        const auto point = points(indexed ? indices(i) : i);
        if (!indexed && !active(point.m)) return;
        const auto c = combinations_d(point.m);
        if (!c.base) {++bad; return;}
        CartesianChartJets jets{};
        for (int f = 0; f < Fields::count; ++f) {
          Real value = c.Value(point.index, f);
          if (f == Fields::chi) {
            if (point.kind)
              value = FloorChiCandidate(value, floor, time, accepted, srecords(point.ledger));
            else
              value = FloorChiCandidate(value, floor, time, accepted, records(point.ledger));
          }
          if (!Kokkos::isfinite(value)) ++bad;
          jets[f].f = value;
        }
        if (!TryExpandCartesianChart(jets)) {++bad; return;}
        if (point.kind) {
          for (int f = 0; f < Fields::count; ++f)
            edge(point.face,f,point.k,point.j) = jets[f].f;
          return;
        }
        const int q = point.ledger;
        const auto& x = geom(q).x;
        for (int f = 0; f < Fields::count; ++f)
          state(point.m,f,point.k,point.j,point.i) = jets[f].f;
        if (accepted) {
          const auto u = SphericalPolicy::Physical(x,jets);
          if (!(u.chi.value > 0 && u.alpha.value > 0 && Geometry(u.metric).valid)) ++bad;
        } else {
          const Real r2 = weight.Value(x);
          for (int f = 0; f < Fields::count; ++f)
            residual(point.m,f,point.k,point.j,point.i) = Fields::Independent(f)
                ? (jets[f].f-reference(q)[f].f)/(f==Fields::chi ? r2 : 1)
                : 0;
        }
      },invalid);
  if (invalid) throw std::runtime_error("Invalid multirate owned stage or accepted geometry");
  // Scri reads five radial support values written by the complete bulk projection.
  if (graph.blocks[pack->GlobalId(slots.front())].kernel == bal::KernelKind::scri)
    p.surface_exchange_->Project();
}
std::vector<DvceArray1D<Real>> MultirateEvolution::Impl::Pack(const std::vector<int>& slots,
                                                              bool rhs) {
  std::vector<DvceArray1D<Real>> values(p.nblocks);
  std::vector<RecordDestination> ptrs(p.nblocks);
  const int width=rhs ? Fields::independent_count : Fields::count;
  for (int m : slots) {
    values[m] = DvceArray1D<Real>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "multirate immutable record"),
        exchange->OwnedPoints(pack->GlobalId(m)).size() * width);
    ptrs[m].data = values[m].data();
  }
  const auto pointers = destinations.Upload(ptrs,"multirate record destinations");
  const auto points = native;
  const auto state = rhs ? p.u_rhs : p.u0;
  const auto edge = rhs ? p.s_rhs : p.s0;
  const auto indices = active_native_indices;
  const bool indexed = complete_local_batches && slots.size() != static_cast<std::size_t>(p.nblocks);
  const int count = indexed ? indices.extent_int(0) : points.extent_int(0);
  Kokkos::parallel_for(
      "multirate pack record batch", Kokkos::RangePolicy<>(0,count),
      KOKKOS_LAMBDA(int i) {
        const auto a = points(indexed ? indices(i) : i);
        Real* out = pointers(a.m).data;
        if (!out) return;
        for (int f=0; f<Fields::count; ++f) {
          if (rhs && !Fields::Independent(f)) continue;
          const int column=rhs ? Fields::IndependentSlot(f) : f;
          out[a.index*width+column]=a.kind ? edge(a.face,f,a.k,a.j) : state(a.m,f,a.k,a.j,a.i);
        }
      });
  Kokkos::fence("multirate records ready for transport");
  return values;
}
void MultirateEvolution::Impl::Publish(const std::vector<int>& slots,
                                       const std::vector<DvceArray1D<Real>>& records,
                                       std::uint32_t step, int stage) {
  std::vector<transport::RecordExchange::Publication> batch;
  batch.reserve(slots.size());
  for (int m : slots)
    batch.emplace_back(Key{pack->GlobalId(m),step,stage},records[m]);
  exchange->PublishBatch(batch);
}
DirectDonorViews MultirateEvolution::Impl::MakeDonorViews(
    const std::vector<int>& slots, std::vector<int>& donor_gids) {
  DirectDonorViews out;
  std::vector<DonorDestination> targets(requested.size(), {-1});
  auto dest = [&](const bvals::DonorPoint& point, int kind) -> DonorDestination& {
    const int q = ordinal.at({point.gid,point.k,point.j,point.i,kind});
    targets[q].index = q;
    return targets[q];
  };
  const auto& halo = p.mapped_halo_->ResidualExchange();
  out.halo_physical = ExchangeView(p.mapped_halo_->PhysicalExchange(),0).buffer;
  out.halo_residual = ExchangeView(halo,0).buffer;
  for (int i : p.mapped_halo_->DonorOrdinalsForBatch(slots))
    dest(halo.UniqueDonors()[i],0).halo = i;
  if (graph.blocks[pack->GlobalId(slots.front())].kernel == bal::KernelKind::scri) {
    // All scri and radial-closure blocks share one temporal level. A complete
    // scri batch therefore consumes the rank's entire bulk/edge exchange.
    auto surface = [&](const bvals::MappedExchange& plan, int kind) {
      const auto& points = plan.UniqueDonors();
      for (size_t i = 0; i < points.size(); ++i) dest(points[i],kind).surface = i;
      return ExchangeView(plan,kind).buffer;
    };
    out.surface_bulk = surface(p.surface_exchange_->BulkExchange(),0);
    out.surface_edge = surface(p.surface_exchange_->EdgeExchange(),1);
  }
  const auto& group = p.agreement_plan_->AgreementBlocks();
  if (std::any_of(slots.begin(),slots.end(),[&](int m) {
        return std::binary_search(group.begin(),group.end(),pack->GlobalId(m));
      })) {
    const auto& plan = p.agreement_plan_->Exchange();
    const auto& rows = p.agreement_plan_->RowsForBatch(slots,p.active_blocks);
    for (int i : p.agreement_plan_->DonorOrdinals(rows))
      dest(plan.UniqueDonors()[i],0).agreement = i;
    out.agreement = ExchangeView(plan,0).buffer;
  }
  targets.erase(std::remove_if(targets.begin(),targets.end(),
      [](const DonorDestination& t){ return t.index < 0; }),targets.end());
  // Requested points are ordered by GID. Store one coupling row for each
  // donor block actually read by this batch, including radial support points
  // which belong to the same block as their scri donor.
  for (auto& target : targets) {
    const int gid = requested[target.index].gid;
    if (donor_gids.empty() || donor_gids.back() != gid) donor_gids.push_back(gid);
    target.coupling_row = donor_gids.size() - 1;
  }
  out.targets = Device(targets,"multirate active donor destinations");
  return out;
}
void MultirateEvolution::Impl::StageDonors(size_t stage, const std::vector<int>& slots,
                                           DvceArray2D<Real>& physical,
                                           DvceArray2D<Real>& residual) {
  auto row = [&](int gid) {
    auto c = Combine(stage,gid);
    if (!c.base) throw std::logic_error("Ready task is missing a donor record");
    return c;
  };
  DvceArray1D<Combination> combinations_d;
  if (complete_local_batches) {
    auto& layout = active_layouts.at(slots);
    if (!layout.donor_views_ready) {
      layout.donor_views = MakeDonorViews(slots,layout.donor_gids);
      layout.donor_views_ready = true;
    }
    std::vector<Combination> rows;
    rows.reserve(layout.donor_gids.size());
    for (int gid : layout.donor_gids) rows.push_back(row(gid));
    combinations_d = layout.coupling.Upload(rows,"multirate batch donor coupling rows");
  } else {
    std::vector<Combination> rows(mesh->nmb_total);
    std::set<int> needed;
    for (int m : slots) {
      const int gid = pack->GlobalId(m);
      needed.insert(gid);
      needed.insert(neighbors[gid].begin(),neighbors[gid].end());
    }
    for (int gid : needed) rows[gid] = row(gid);
    combinations_d = donor_rows.Upload(rows,"multirate donor coupling rows");
  }
  donor_table_bytes += combinations_d.extent(0) * sizeof(Combination);
  const auto metadata = donors;
  physical = donor_physical;
  residual = donor_residual;
  const Real floor = p.chi_floor;
  auto reconstruct = [&](const auto& output, int count) {
    int invalid = 0;
    Kokkos::parallel_reduce(
      "multirate recipient donor values", Kokkos::RangePolicy<>(0, count),
      KOKKOS_LAMBDA(int i, int& bad) {
        const auto target = output.Destination(i);
        const auto meta = metadata(target.index);
        const auto c = combinations_d(target.coupling_row >= 0 ? target.coupling_row : meta.gid);
        CartesianChartJets jets{};
        if (!c.base) {
          int slot = 0;
          for (int f = 0; f < Fields::count; ++f)
            output.Store(target,meta.kind,f,Fields::Independent(f) ? slot++ : -1,0,0);
          return;
        }
        for (int f = 0; f < Fields::count; ++f) jets[f].f = c.Value(meta.index, f);
        if (Kokkos::isfinite(jets[0].f) && floor > 0 && jets[0].f < floor) jets[0].f = floor;
        if (!TryExpandCartesianChart(jets)) {
          ++bad;
          return;
        }
        if (meta.kind) {
          const auto physical = SphericalPolicy::Physical(meta.x, jets);
          const auto geometry = Geometry(physical.metric);
          Real Q = 0;
          for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) Q += geometry.inverse[a][b] * meta.x[a] * meta.x[b];
          if (!geometry.valid || !(jets[0].f > 0 && Q > 0)) {
            ++bad;
            return;
          }
          const Real speed = sqrt(jets[0].f * Q);
          jets[18].f = 0;
          for (int a = 0; a < 3; ++a) {
            jets[22 + a].f = (1 - speed) * meta.x[a];
            jets[28 + a].f = 0;
            Real C = 0;
            for (int k = 0; k < 5; ++k) {
              const auto support = metadata(meta.support[k]);
              C += meta.radial_weight[k] *
                   (c.Value(support.index, 25 + a) - c.Value(support.index, 15 + a));
            }
            jets[25 + a].f = jets[15 + a].f + C;
          }
        }
        int slot = 0;
        for (int f = 0; f < Fields::count; ++f) {
          const Real value = jets[f].f;
          const Real delta = Fields::Independent(f)
                                 ? (value - meta.ref[f]) / (f == 0 ? meta.chi_weight : 1)
                                 : 0;
          if (!Kokkos::isfinite(value) || !Kokkos::isfinite(delta)) ++bad;
          output.Store(target,meta.kind,f,Fields::Independent(f) ? slot++ : -1,value,delta);
        }
      },invalid);
    if (invalid) throw std::runtime_error("Invalid recipient-specific multirate donor stage");
  };
  if (complete_local_batches) {
    const auto& view = active_layouts.at(slots).donor_views;
    reconstruct(view,view.targets.extent_int(0));
  } else {
    reconstruct(FullDonorViews{physical,residual},metadata.extent_int(0));
  }
}
MultirateEvolution::Impl::GatherPlan& MultirateEvolution::Impl::ExchangeView(
    const bvals::MappedExchange& plan, int kind) {
  auto [it,inserted] = gather_plans.try_emplace(&plan);
  auto& layout = it->second;
  if (!inserted) return layout;
  const auto& points = plan.UniqueDonors();
  const auto& fields = plan.Fields();
  layout.buffer = DvceArray2D<Real>(Kokkos::view_alloc(Kokkos::WithoutInitializing,
      "multirate stage exchange view"),points.size(),fields.size());
  if (complete_local_batches) {
    // Direct stage writes use the same field slots as the native exchanges.
    const bool physical = &plan == &p.mapped_halo_->PhysicalExchange();
    size_t slot = 0;
    for (int f = 0; f < Fields::count; ++f) if (physical || Fields::Independent(f)) {
      if (slot >= fields.size() || fields[slot++] != f)
        throw std::logic_error("Unexpected projected exchange field order");
    }
    if (slot != fields.size()) throw std::logic_error("Unexpected projected exchange field count");
  } else {
    std::vector<int> indices;
    for (const auto& q : points) indices.push_back(ordinal.at({q.gid,q.k,q.j,q.i,kind}));
    layout.index = Device(indices,"multirate exchange ordinals");
    layout.fields = Device(fields,"multirate exchange fields");
  }
  return layout;
}
DvceArray2D<Real> MultirateEvolution::Impl::Gather(const bvals::MappedExchange& plan, int kind,
                                                   const DvceArray2D<Real>& values) {
  const auto& layout = ExchangeView(plan,kind);
  if (complete_local_batches) return layout.buffer;
  const auto index = layout.index, fields = layout.fields;
  const int nf = fields.extent_int(0);
  const auto out = layout.buffer;
  Kokkos::parallel_for(
      "multirate gather stage exchange", Kokkos::RangePolicy<>(0, index.extent(0) * nf),
      KOKKOS_LAMBDA(int q) {
        const int row = q / nf, col = q % nf;
        out(row, col) = values(index(row), fields(col));
      });
  return out;
}
void MultirateEvolution::Impl::BeginInterval() {
  interval_start = Clock::now();
  interval_active = true;
  completed = batch_count = max_batch = donor_table_bytes = 0;
  communication_wait = 0;
  phase_seconds.fill(0);
  interior_batches_with_receives = interior_receives = 0;
  level_work.assign(schedule->MaxLevel() + 1, 0);
  const auto epoch = static_cast<std::uint64_t>(mesh->ncycle);
  if (mesh->dt != p.requested_dt)
    throw std::runtime_error("Multirate base interval changed during run");
  exchange->Begin(epoch, steps, consumers, peer_profile_pending);
  readiness.Reset();
  interior_readiness.Reset();
  std::vector<int> all(p.nblocks);
  for (int m = 0; m < p.nblocks; ++m) all[m] = m;
  auto initial = Pack(all, false);
  Publish(all, initial, 0, -1);
  initial.clear();
}
void MultirateEvolution::Impl::Evaluate(Driver* driver, bool interior,
                                        const std::vector<size_t>& batch) {
  const auto epoch = static_cast<std::uint64_t>(mesh->ncycle);
  const auto stage = tasks[batch.front()].stage;
  const auto kind = graph.blocks[tasks[batch.front()].gid].kernel;
  std::vector<int> slots;
  for (auto task : batch) slots.push_back(tasks[task].m);
  if (complete_local_batches) std::sort(slots.begin(), slots.end());
  ++batch_count;
  max_batch = std::max(max_batch, static_cast<unsigned long long>(batch.size()));
  if (!interior) level_work[schedule->Stage(stage).key.level] += batch.size();
  const auto context = schedule->Context(stage, epoch, mesh->time, p.requested_dt);
  driver->stage_time = context.stage_time;
  driver->stage_state_time = context.stage_time;
  p.active_step_dt = context.local_dt;
  const auto batch_start = Clock::now();
  std::uint64_t units = 0;
  for (int m : slots) units += graph.blocks[pack->GlobalId(m)].work_units;
  std::vector<Combination> combinations(p.nblocks);
  if (interior) {
    for (int m : slots) combinations[m] = Combine(stage, pack->GlobalId(m));
    Reconstruct(slots, combinations, context.stage_time, false);

    model.Observe(bal::KernelKind::projection, units, Seconds(batch_start));
    const auto begin_interior = Clock::now();
    p.CalcRHS(driver, context.key.stage + 1, RHSSubset::Interior, max_cfl);
    const double elapsed_interior = Seconds(begin_interior);
    phase_seconds[0] += Seconds(batch_start);
    const double per_point = elapsed_interior / units;
    const auto before=exchange->CompletedReceives();
    exchange->Poll();
    const auto completed_receives=exchange->CompletedReceives()-before;
    interior_receives+=completed_receives;
    interior_batches_with_receives+=(completed_receives!=0);
    for (auto index : batch)
      tasks[index].interior_seconds = per_point * graph.blocks[tasks[index].gid].work_units;
    for (int m : slots)
      readiness.Arrive({pack->GlobalId(m), context.key.local_step, context.key.stage + 4});
    return;
  }
  SelectBatch(slots);
  DvceArray2D<Real> physical, delta;
  StageDonors(stage, slots, physical, delta);
  auto halo_delta = Gather(p.mapped_halo_->ResidualExchange(), 0, delta);
  auto halo_physical = Gather(p.mapped_halo_->PhysicalExchange(), 0, physical);
  p.mapped_halo_->FillExternal(p.residual, p.u0, halo_delta, halo_physical, p.active_blocks,
                               complete_local_batches ? &slots : nullptr);
  if (kind == bal::KernelKind::scri) {
    auto surface_bulk = Gather(p.surface_exchange_->BulkExchange(), 0, delta);
    auto surface_edge = Gather(p.surface_exchange_->EdgeExchange(), 1, delta);
    p.surface_exchange_->SetExternalValues(surface_edge);
    p.surface_exchange_->Sample(surface_bulk, surface_edge);
    p.surface_exchange_->Solve();
    p.surface_exchange_->Evaluate();
  }
  const auto begin_rhs = Clock::now();
  phase_seconds[1] += std::chrono::duration<double>(begin_rhs-batch_start).count();
  p.CalcRHS(driver, context.key.stage + 1, RHSSubset::Dependent, max_cfl);
  phase_seconds[2] += Seconds(begin_rhs);
  double rhs_seconds = Seconds(batch_start);
  for (auto index : batch) rhs_seconds += tasks[index].interior_seconds;
  const auto& participants = p.agreement_plan_->AgreementBlocks();
  if (std::any_of(slots.begin(), slots.end(), [&](int m) {
        return std::binary_search(participants.begin(), participants.end(), pack->GlobalId(m));
      })) {
    const auto begin_agreement = Clock::now();
    auto agreement = Gather(p.agreement_plan_->Exchange(), 0, delta);
    const auto* rows = complete_local_batches ?
        &p.agreement_plan_->RowsForBatch(slots,p.active_blocks) : nullptr;
    p.agreement_plan_->Evaluate(agreement,rows);
    p.agreement_plan_->AddRHS(p,rows);
    Kokkos::fence("multirate agreement timing");
    const double elapsed = Seconds(begin_agreement);
    model.Observe(bal::KernelKind::agreement, units, elapsed);
    phase_seconds[3] += elapsed;
  }
  const auto begin_publication = Clock::now();
  auto rhs = Pack(slots, true);
  Publish(slots, rhs, context.key.local_step, context.key.stage);
  if (context.key.stage == 3) {
    const auto begin_accept = Clock::now();
    for (int m : slots) {
      const int gid = pack->GlobalId(m);
      auto& c = combinations[m];
      c = {};
      c.base = exchange->Data({gid, context.key.local_step, -1}).data();
      c.count = 4;
      for (int a = 0; a < 4; ++a) {
        c.slope[a] = (a == 3 ? rhs[m] : exchange->Data({gid, context.key.local_step, a})).data();
        c.weight[a] = context.local_dt * mr::BaseB()[a];
      }
    }
    Reconstruct(slots, combinations, mesh->time + (context.key.local_step + 1) * context.local_dt,
                true);
    auto accepted = Pack(slots, false);
    model.Observe(bal::KernelKind::projection, units, Seconds(begin_accept));
    Publish(slots, accepted, context.key.local_step + 1, -1);
  }
  Kokkos::fence("multirate batch complete before release");
  phase_seconds[4] += Seconds(begin_publication);
  model.Observe(kind, units, rhs_seconds);
  for (size_t q : batch) {
    auto& t = tasks[q];
    for (const auto& key : t.dependencies) exchange->Release(key);
  }
}
bool MultirateEvolution::Impl::Poll(Driver* driver) {
  if (!interval_active) BeginInterval();
  const auto start = Clock::now();
  if (completed < tasks.size()) {
    exchange->Poll();
    for (const auto& key : exchange->TakeArrivals()) {
      readiness.Arrive(key);
      interior_readiness.Arrive(key);
    }
    auto batch = readiness.TakeReady(complete_local_batches);
    if (!batch.empty()) {
      Evaluate(driver, false, batch);
      completed += batch.size();
      return false;
    }
    batch = interior_readiness.TakeReady(complete_local_batches);
    if (!batch.empty()) {
      Evaluate(driver, true, batch);
      return false;
    }
  } else if (exchange->Finish()) {
    CompleteInterval(driver);
    interval_active = false;
    return true;
  }
  std::this_thread::yield();
  communication_wait += Seconds(start);
  return false;
}
void MultirateEvolution::Impl::CompleteInterval(Driver* driver) {
  const auto epoch = static_cast<std::uint64_t>(mesh->ncycle);
  std::vector<int> all(p.nblocks);
  std::iota(all.begin(), all.end(), 0);
  std::vector<Combination> final(p.nblocks);
  for (int m : all)
    final[m].base =
        exchange
            ->Data({pack->GlobalId(m), static_cast<std::uint32_t>(steps[pack->GlobalId(m)]), -1})
            .data();
  // Copy the already accepted records exactly, without re-applying a floor ledger update.
  const auto final_d = Device(final, "multirate final records");
  const auto points = native;
  const auto state = p.u0;
  const auto edge = p.s0;
  Kokkos::parallel_for(
      "multirate install synchronized accepted state",
      Kokkos::RangePolicy<>(0, native.extent_int(0)), KOKKOS_LAMBDA(int q) {
        const auto a = points(q);
        for (int f = 0; f < Fields::count; ++f) {
          const Real value = final_d(a.m).base[a.index * Fields::count + f];
          if (a.kind)
            edge(a.face, f, a.k, a.j) = value;
          else
            state(a.m, f, a.k, a.j, a.i) = value;
        }
      });
  Kokkos::fence("multirate endpoint installed");
  for (int m : all)
    exchange->Release(
        {pack->GlobalId(m), static_cast<std::uint32_t>(steps[pack->GlobalId(m)]), -1});
  p.ClearActiveBlocks();
  p.InitializeBoundaryValues(driver);
  p.InvalidateDiagnostics();
  driver->stage_time = driver->stage_state_time = mesh->time + p.requested_dt;
  p.TrackBinary(driver, 4);
  const std::uint64_t endpoint = epoch + 1;
  auto owner = Owners();
  bool change = false;
  for (const auto& r : replay)
    if (r.interval == endpoint) {
      owner = r.owner_by_gid;
      change = owner != Owners();
    }
  if (replay_path.empty() && balance != "none" && endpoint >= policy.warmup_intervals &&
      (endpoint == policy.warmup_intervals ||
       (policy.review_interval && endpoint % policy.review_interval == 0))) {
    // Profiles are synchronized only at a common endpoint, never at a stage.
    const std::string profile = model.Serialize();
    std::vector<int> framed{static_cast<int>(profile.size())};
    framed.insert(framed.end(), profile.begin(), profile.end());
    const auto gathered = AllIntegers(framed);
    bal::BatchCostModel shared_model;
    for (auto it = gathered.begin(); it != gathered.end();) {
      const int length = *it++;
      shared_model.Merge(bal::BatchCostModel::Deserialize(std::string(it, it + length)));
      it += length;
    }
    if (global_variable::my_rank == 0) {
      std::istringstream rows(shared_model.Serialize());
      std::string line;
      std::getline(rows, line);
      while (std::getline(rows, line))
        std::cout << "MULTIRATE_BATCH_PROFILE cycle=" << endpoint
                  << " kind_units_count_seconds=" << line << "\n";
    }
    SetPhaseCosts(shared_model);
    const auto candidate =
        bal::Partition(graph, global_variable::nranks, options, shared_model, scoring_tasks, owner);
    const auto current = bal::ScorePartition(graph, global_variable::nranks, owner, options,
                                             shared_model, scoring_tasks);
    double intervals_left =
        std::max(0., std::ceil((driver->tlim - mesh->time - p.requested_dt) / p.requested_dt));
    if (driver->nlim >= 0)
      intervals_left = std::min(intervals_left, std::max(0., double(driver->nlim) - endpoint));
    const auto remaining = static_cast<std::uint64_t>(std::min(intervals_left, 1e18));
    double cost = std::max(migration_cost, Seconds(interval_start));
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &cost, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
    const auto decision = bal::ConsiderRebalance(endpoint, remaining, true, current.seconds,
                                                 candidate.score.seconds, cost, policy);
    if (global_variable::my_rank == 0)
      std::cout << "MULTIRATE_REVIEW cycle=" << endpoint
                << " candidate=" << candidate.selected_candidate
                << " current_seconds=" << current.seconds
                << " predicted_seconds=" << candidate.score.seconds
                << " migration_allowance_seconds=" << cost << " remaining_intervals=" << remaining
                << " apply=" << decision.apply << "\n";
    if (decision.apply) {
      owner = candidate.owner_by_gid;
      change = owner != Owners();
    }
  }
  const auto stats = exchange->Stats();
#if MPI_PARALLEL_ENABLED
  if (peer_profile_pending) {
    const int ranks = global_variable::nranks;
    std::vector<std::uint64_t> bytes;
    if (global_variable::my_rank == 0) bytes.resize(static_cast<std::size_t>(ranks)*ranks);
    MPI_Gather(stats.sent_bytes_by_peer.data(),ranks,MPI_UINT64_T,bytes.data(),ranks,MPI_UINT64_T,
               0,MPI_COMM_WORLD);
    if (global_variable::my_rank == 0) {
      for (int rank=0; rank<ranks; ++rank) {
        std::cout << "MULTIRATE_PEER_BYTES cycle=" << endpoint << " rank=" << rank << " bytes=";
        for (int peer=0; peer<ranks; ++peer)
          std::cout << (peer ? "," : "") << bytes[static_cast<std::size_t>(rank)*ranks+peer];
        std::cout << "\n";
      }
    }
    peer_profile_pending = false;
  }
#endif
  if (pin->GetBoolean("multirate","profile_peer_bytes")) {
    // One line per rank at a common endpoint. Avoid summing maxima attained
    // on different ranks or reporting host phase times as GPU occupancy.
    std::ostringstream row;
    row << "MULTIRATE_RANK_PHASES cycle=" << endpoint << " rank=" << global_variable::my_rank
        << " interior=" << phase_seconds[0] << " donor_halo_surface=" << phase_seconds[1]
        << " dependent_rhs=" << phase_seconds[2] << " agreement=" << phase_seconds[3]
        << " publish_accept=" << phase_seconds[4] << " poll_no_ready=" << communication_wait << "\n";
    std::cout << row.str();
  }
  unsigned long long peaks[] = {stats.peak_record_bytes, stats.peak_packet_bytes, max_batch};
  unsigned long long sums[] = {stats.sent_bytes, batch_count, interior_receives, interior_batches_with_receives,
                               stats.send_packets, stats.pack_kernels, stats.reused_send_packets, donor_table_bytes, stats.reused_receive_packets};
#if MPI_PARALLEL_ENABLED
  MPI_Reduce(global_variable::my_rank == 0 ? MPI_IN_PLACE : peaks, peaks, 3, MPI_UNSIGNED_LONG_LONG,
             MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(global_variable::my_rank == 0 ? MPI_IN_PLACE : sums, sums, sizeof(sums)/sizeof(sums[0]), MPI_UNSIGNED_LONG_LONG,
             MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(global_variable::my_rank == 0 ? MPI_IN_PLACE : level_work.data(), level_work.data(),
             level_work.size(), MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(global_variable::my_rank == 0 ? MPI_IN_PLACE : &communication_wait,
             &communication_wait, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
#endif
  if (global_variable::my_rank == 0) {
    std::cout << "MULTIRATE_WORK cycle=" << endpoint << " peak_record_bytes_per_rank=" << peaks[0]
              << " peak_packet_bytes_per_rank=" << peaks[1] << " maximum_batch_blocks=" << peaks[2]
              << " sent_bytes=" << sums[0] << " batches=" << sums[1] << " records_completed_on_interior_poll=" << sums[2]
              << " interior_batches_with_receives=" << sums[3] << " send_packets=" << sums[4]
              << " send_pack_kernels=" << sums[5] << " reused_send_packets=" << sums[6]
              << " donor_coupling_table_bytes=" << sums[7] << " reused_receive_packets=" << sums[8]
              << " rhs_by_level=";
    for (size_t i = 0; i < level_work.size(); ++i) std::cout << (i ? "," : "") << level_work[i];
    std::cout << "\n";
  }
  if (change) {
    const auto start = Clock::now();
    mesh->RedistributeStaticBlocks(owner);
    p.InitializeBoundaryValues(driver);
    BuildLocal();
    migration_cost = Seconds(start);
    LogPartition(endpoint);
  }
  if (global_variable::my_rank == 0)
    std::cout << "MULTIRATE_INTERVAL cycle=" << endpoint << " seconds=" << Seconds(interval_start)
              << " stage_evaluations="
              << std::accumulate(level_work.begin(), level_work.end(), 0ULL)
              << " polling_wait_seconds=" << communication_wait
              << " migration_seconds=" << (change ? migration_cost : 0) << "\n";
}
MultirateEvolution::MultirateEvolution(Z4cHyperboloidal& p, Driver* driver, bool rebuilt)
    : impl_(std::make_unique<Impl>(p, driver, rebuilt)) {}
MultirateEvolution::~MultirateEvolution() = default;
bool MultirateEvolution::Poll(Driver* driver) { return impl_->Poll(driver); }
void Z4cHyperboloidal::InitializeMultirate(Driver* driver, bool rebuilt) {
  if (multirate_enabled_) {
    if (!rebuilt) {
      auto list = std::make_shared<TaskList>();
      TaskID none(0);
      list->AddTask(&Z4cHyperboloidal::AdvanceMultirate, this, none);
      pmy_pack->tl_map["multirate_interval"] = std::move(list);
    }
    multirate_.reset();
    multirate_ = std::make_unique<MultirateEvolution>(*this, driver, rebuilt);
  }
}
TaskStatus Z4cHyperboloidal::AdvanceMultirate(Driver* driver, int) {
  if (!multirate_) throw std::logic_error("Multirate evolution was not initialized");
  return multirate_->Poll(driver) ? TaskStatus::complete : TaskStatus::incomplete;
}
}  // namespace z4c_hyperboloidal
