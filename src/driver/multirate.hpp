// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef DRIVER_MULTIRATE_HPP_
#define DRIVER_MULTIRATE_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace time_integration::multirate {

// Four active stages of EX5-EX5 4(3)[A], with the unused embedded/FSAL
// derivative omitted. These are not the classical RK4 stage coefficients.
constexpr int kStages = 4;
using StageId = std::size_t;
using Matrix = std::array<std::array<double, kStages>, kStages>;
const Matrix &BaseA();
const std::array<double, kStages> &BaseB();
const Matrix &SlowFromFast(int fast_substep);
const Matrix &FastFromSlow(int fast_substep);

struct StageKey {
  int level = 0;
  std::uint32_t local_step = 0;
  int stage = 0;  // zero-based
};

struct StepContext {
  std::uint64_t macro_index = 0;
  StageKey key;
  double start_time = 0;
  double base_dt = 0;
  double local_dt = 0;
  double stage_time = 0;
  std::uint64_t stage_tick = 0;
  std::uint64_t ticks_per_base = 0;
};

struct WeightedStage {
  StageId source = 0;
  double coefficient = 0;  // multiply by the shared base_dt
};

// Reconstruct a donor partition for a recipient stage as
//   accepted(donor_level, completed_steps) + base_dt * sum(weight * RHS).
// accepted(..., 0) is the state at the start of the base interval. A compact
// prefix state avoids retaining all derivatives from previously finished steps.
struct CouplingRow {
  int donor_level = 0;
  std::uint32_t completed_steps = 0;
  std::array<WeightedStage, kStages> terms{};
  int count = 0;
};

struct StageRecord {
  StageKey key;
  std::vector<CouplingRow> rows;
  std::vector<StageId> predecessors;
  std::vector<StageId> successors;
  std::size_t critical_depth = 1;
  bool CommitsStep() const { return key.stage == kStages - 1; }
};

// Host-side immutable numerical schedule. Production code can flatten the POD
// rows into device arrays and expand dependencies only across actual block
// couplings. The aggregate level graph is also usable as a reference executor.
class Schedule {
 public:
  explicit Schedule(int maximum_level, std::size_t stage_budget = 1048576);
  int MaxLevel() const { return maximum_level_; }
  std::uint32_t Steps(int level) const;
  StageId Id(int level, std::uint32_t local_step, int stage) const;
  const StageRecord &Stage(StageId id) const;
  const std::vector<StageRecord> &Stages() const { return stages_; }
  const std::vector<StageId> &TopologicalOrder() const { return order_; }
  const CouplingRow &Row(StageId target, int donor_level) const;
  // Diagnostic/reference expansion relative to the base-interval initial state.
  // Production reconstruction should use compact Row() accepted prefixes.
  std::vector<WeightedStage> DenseRow(StageId target, int donor_level) const;
  StepContext Context(StageId id, std::uint64_t macro_index,
                      double start_time, double base_dt) const;
  std::array<WeightedStage, kStages> StepWeights(int level,
                                               std::uint32_t step) const;

 private:
  CouplingRow MakeRow(StageId target, int donor_level) const;
  int maximum_level_;
  std::vector<StageId> offsets_;
  std::vector<StageRecord> stages_;
  std::vector<StageId> order_;
};

// Dependency readiness, deliberately independent of MPI, GPU streams, and
// priority policy. Complete() means all RHS/state writes and outgoing reads
// owned by this task have finished. For stage 4 it also means that its local
// accepted state has been committed. Physical stage times need not be monotone.
class ReadyScheduler {
 public:
  explicit ReadyScheduler(const Schedule &schedule);
  const std::set<StageId> &Ready() const { return ready_; }
  void Start(StageId id);
  void Complete(StageId id);
  bool Finished() const { return completed_ == states_.size(); }
  bool Running(StageId id) const;
  bool Completed(StageId id) const;
  std::size_t RemainingConsumers(StageId id) const;
  bool Releasable(StageId id) const;

 private:
  enum class State : unsigned char { waiting, ready, running, complete };
  const Schedule *schedule_;
  std::vector<State> states_;
  std::vector<std::size_t> missing_, consumers_;
  std::set<StageId> ready_;
  std::size_t completed_ = 0;
};

struct BlockSpec {
  int gid = -1;
  bool cartesian = false;
  int physical_level = 0;  // logical spatial level minus the root level
  int radial_region = -1;
  int agreement_group = -1;
  int scri_group = -1;
};
struct DependencyEdge { int first = -1, second = -1; };  // native block GIDs
struct LevelAdjustment { int gid = -1, from = 0, to = 0; };
struct LevelAssignment {
  std::vector<int> levels;  // follows input block order
  std::vector<LevelAdjustment> agreement_promotions;
  int maximum_level = 0;
};

// Spherical levels are exact, one per radial region. A declared agreement
// group may promote Cartesian members to its maximum level; it may not alter
// explicit spherical assignments. Scri groups must have one common level.
// Edges must include every direct RHS/exchange dependency, not only face peers.
LevelAssignment AssignLevels(const std::vector<BlockSpec> &blocks,
                             const std::vector<int> &spherical_levels,
                             const std::vector<DependencyEdge> &edges);

}  // namespace time_integration::multirate
#endif  // DRIVER_MULTIRATE_HPP_
