// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef MESH_MULTIRATE_LOAD_BALANCE_HPP_
#define MESH_MULTIRATE_LOAD_BALANCE_HPP_

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace multirate::balance {

enum class KernelKind { cartesian, spherical, scri, projection, agreement, count };

// Units are seconds. Profiles must measure completed device work, not launch latency.
class BatchCostModel {
 public:
  struct Sample { std::uint64_t units = 0, count = 0; double seconds = 0; };
  void Observe(KernelKind kind, std::uint64_t units, double seconds);
  // Combine count-weighted timing bins gathered from every rank. Call on fresh
  // aggregates when merging cumulative rank profiles to avoid double counting.
  void Merge(const BatchCostModel &other);
  double Predict(KernelKind kind, std::uint64_t units) const;
  bool Calibrated(KernelKind kind) const;
  std::string Serialize() const;
  static BatchCostModel Deserialize(const std::string &text);
 private:
  std::map<KernelKind, std::map<std::uint64_t, Sample>> samples_;
};

struct BlockCost {
  int gid = -1;
  int time_level = 0;
  KernelKind kernel = KernelKind::cartesian;
  std::uint64_t work_units = 0;
  std::uint64_t memory_bytes = 0;
  // Expected compute seconds per base interval, separated by stage/level class.
  // Populate from the expanded integrator schedule, including non-RHS work.
  std::vector<double> phase_costs;
};

struct CouplingEdge {
  int a = -1, b = -1;
  std::uint64_t bytes = 0, exchanges = 0;
};

struct WorkGraph {
  // GIDs are a permutation of [0, blocks.size()); storage order is immaterial.
  std::vector<BlockCost> blocks;
  std::vector<CouplingEdge> edges;
  void Validate() const;
  std::uint64_t Fingerprint() const;
};

struct TaskDependency {
  int task = -1;
  // Payload needed specifically by this dependency, not a whole-block estimate.
  std::uint64_t bytes = 0;
};

struct ExecutionTask {
  int block = -1;
  // Equal phase means compatible numerical stage contexts, not equal timestamps.
  std::uint64_t phase = 0;
  KernelKind kernel = KernelKind::cartesian;
  std::uint64_t work_units = 0;
  std::vector<TaskDependency> dependencies;
};

enum class Backend { metis, scalar_reference };

struct BalanceOptions {
  Backend backend = Backend::metis;
  int seed = 17;
  double imbalance = 1.05;
  // A sparse phase is not made a hard METIS balance constraint.
  int minimum_blocks_per_rank_per_phase = 2;
  int compact_batch_blocks = 4;
  double network_latency_seconds = 2e-6;
  double network_bytes_per_second = 1e10;
  // Zero means no supplied capacity limit; memory is still a balance constraint.
  std::uint64_t memory_capacity_per_rank = 0;
  int maximum_blocks_per_rank = 0;  // AMR storage capacity, zero when unlimited.
  bool require_nonempty_ranks = true;
  bool consider_sparse_compaction = true;
  bool spread_finest = false;
  int ranks_per_gpu = 1;
};

struct PartitionScore {
  double seconds = 0;
  double compute_seconds = 0;
  double communication_seconds = 0;
  std::uint64_t cut_bytes_per_interval = 0;
  std::uint64_t maximum_rank_bytes = 0;
  std::vector<double> rank_busy_seconds;
  bool feasible = true;
};

struct CandidateScore { std::string name; PartitionScore score; };
struct PartitionResult {
  std::vector<int> owner_by_gid;
  std::string selected_candidate;
  PartitionScore score;
  std::vector<CandidateScore> candidates;
  bool used_metis = false;
};

bool MetisAvailable();
PartitionScore ScorePartition(const WorkGraph &, int ranks,
    const std::vector<int> &owners, const BalanceOptions &, const BatchCostModel &,
    const std::vector<ExecutionTask> &tasks = {});
PartitionResult Partition(const WorkGraph &, int ranks, const BalanceOptions &,
    const BatchCostModel &, const std::vector<ExecutionTask> &tasks = {},
    const std::vector<int> &current_owners = {});

struct RebalancePolicy {
  std::uint64_t warmup_intervals = 32;
  std::uint64_t review_interval = 128;
  double minimum_fractional_improvement = .05;
  double amortization_safety_factor = 2;
};
struct RebalanceDecision { bool apply = false; double predicted_net_saving = 0; std::string reason; };
RebalanceDecision ConsiderRebalance(std::uint64_t completed_intervals,
    std::uint64_t remaining_intervals, bool synchronized,
    double current_interval_seconds, double proposed_interval_seconds,
    double migration_and_rebuild_seconds, const RebalancePolicy & = {});

// Stable, versioned text: no raw struct padding, wall clocks or unordered iteration.
// Include in the synchronized checkpoint; replay overrides live profiling decisions.
struct PartitionRecord {
  std::uint64_t interval = 0, graph_fingerprint = 0;
  int ranks = 0;
  std::vector<int> owner_by_gid;
  std::string Serialize() const;
  static PartitionRecord Deserialize(const std::string &text);
  void ValidateFor(const WorkGraph &, int expected_ranks) const;
};

}  // namespace multirate::balance
#endif
