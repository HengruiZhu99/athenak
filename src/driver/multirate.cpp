// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include "multirate.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>

namespace time_integration::multirate {
namespace {
const Matrix kA{{{{0, 0, 0, 0}}, {{2./5, 0, 0, 0}},
                 {{-3./20, 3./4, 0, 0}}, {{19./44, -15./44, 10./11, 0}}}};
const std::array<double, kStages> kB{{11./72, 25./72, 25./72, 11./72}};
const Matrix kS1{{{{0, 0, 0, 0}}, {{4./5, 0, 0, 0}},
                  {{-9./5, 3, 0, 0}}, {{53./11, -95./11, 20./11, 4}}}};
const Matrix kS2{};
const Matrix kF1{{{{0, 0, 0, 0}}, {{1./5, 0, 0, 0}},
                  {{9./80, 3./16, 0, 0}}, {{0, 87./88, -43./88, 0}}}};
const Matrix kF2{{{{11./144, 25./144, 25./144, 11./144}},
                  {{-53./450, 184./225, 0, 0}},
                  {{391./600, 89./600, 0, 0}},
                  {{0, -93./88, 3379./1584, -11./144}}}};

void CheckLevel(int level) {
  if (level < 0 || level > 30)
    throw std::invalid_argument("temporal level must be in [0,30]");
}
std::string BlockName(int gid) { return "block " + std::to_string(gid); }
}  // namespace

const Matrix &BaseA() { return kA; }
const std::array<double, kStages> &BaseB() { return kB; }
const Matrix &SlowFromFast(int substep) {
  if (substep < 0 || substep > 1) throw std::out_of_range("fast substep must be 0 or 1");
  return substep == 0 ? kS1 : kS2;
}
const Matrix &FastFromSlow(int substep) {
  if (substep < 0 || substep > 1) throw std::out_of_range("fast substep must be 0 or 1");
  return substep == 0 ? kF1 : kF2;
}

std::uint32_t Schedule::Steps(int level) const {
  if (level < 0 || level > maximum_level_) throw std::out_of_range("temporal level outside schedule");
  return std::uint32_t{1} << level;
}
StageId Schedule::Id(int level, std::uint32_t step, int stage) const {
  if (step >= Steps(level) || stage < 0 || stage >= kStages)
    throw std::out_of_range("multirate stage outside schedule");
  return offsets_.at(level) + static_cast<StageId>(step)*kStages + stage;
}
const StageRecord &Schedule::Stage(StageId id) const { return stages_.at(id); }

Schedule::Schedule(int maximum_level, std::size_t stage_budget)
    : maximum_level_(maximum_level) {
  CheckLevel(maximum_level);
  const std::uint64_t count = kStages*((std::uint64_t{1} << (maximum_level+1))-1);
  if (count > stage_budget || count > std::numeric_limits<std::size_t>::max())
    throw std::length_error("multirate schedule exceeds configured stage budget");
  offsets_.resize(maximum_level+1);
  stages_.resize(static_cast<std::size_t>(count));
  StageId next = 0;
  for (int level = 0; level <= maximum_level; ++level) {
    offsets_[level] = next;
    for (std::uint32_t step = 0; step < Steps(level); ++step)
      for (int stage = 0; stage < kStages; ++stage)
        stages_[next++].key = {level, step, stage};
  }
  for (StageId id = 0; id < stages_.size(); ++id) {
    auto &record = stages_[id];
    const int level = record.key.level;
    for (int donor = std::max(0, level-1); donor <= std::min(maximum_level, level+1); ++donor) {
      record.rows.push_back(MakeRow(id, donor));
      const auto &row = record.rows.back();
      if (row.completed_steps != 0)
        record.predecessors.push_back(Id(donor, row.completed_steps-1, kStages-1));
      for (int j = 0; j < row.count; ++j) record.predecessors.push_back(row.terms[j].source);
    }
    std::sort(record.predecessors.begin(), record.predecessors.end());
    record.predecessors.erase(std::unique(record.predecessors.begin(), record.predecessors.end()),
                              record.predecessors.end());
    for (auto predecessor : record.predecessors)
      stages_[predecessor].successors.push_back(id);
  }
  std::vector<std::size_t> missing(stages_.size());
  for (StageId id = 0; id < stages_.size(); ++id) {
    missing[id] = stages_[id].predecessors.size();
    if (missing[id] == 0) order_.push_back(id);
  }
  for (std::size_t cursor = 0; cursor < order_.size(); ++cursor)
    for (auto dependent : stages_[order_[cursor]].successors)
      if (--missing[dependent] == 0) order_.push_back(dependent);
  if (order_.size() != stages_.size()) throw std::logic_error("cyclic multirate coefficient graph");
  for (auto it = order_.rbegin(); it != order_.rend(); ++it)
    for (auto child : stages_[*it].successors)
      stages_[*it].critical_depth = std::max(stages_[*it].critical_depth,
                                            stages_[child].critical_depth+1);
}

CouplingRow Schedule::MakeRow(StageId target, int donor) const {
  const auto key = stages_.at(target).key;
  if (std::abs(key.level-donor) > 1 || donor < 0 || donor > maximum_level_)
    throw std::invalid_argument("direct temporal coupling must be same-level or 2:1");
  CouplingRow row;
  row.donor_level = donor;
  const Matrix *matrix = &kA;
  std::uint32_t donor_step = key.local_step;
  if (donor == key.level+1) {
    donor_step = 2*key.local_step;
    matrix = &kS1;  // all four active rows of S2 are zero
  } else if (donor == key.level-1) {
    donor_step = key.local_step/2;
    matrix = &(key.local_step%2 == 0 ? kF1 : kF2);
  }
  row.completed_steps = donor_step;
  const double scale = 1.0/Steps(donor);
  for (int j = 0; j < kStages; ++j) {
    const double coefficient = (*matrix)[key.stage][j]*scale;
    if (coefficient != 0) row.terms[row.count++] = {Id(donor, donor_step, j), coefficient};
  }
  return row;
}

const CouplingRow &Schedule::Row(StageId target, int donor) const {
  for (const auto &row : stages_.at(target).rows)
    if (row.donor_level == donor) return row;
  throw std::invalid_argument("direct temporal coupling must be same-level or 2:1");
}

std::vector<WeightedStage> Schedule::DenseRow(StageId target, int donor) const {
  const auto &row = Row(target, donor);
  std::vector<WeightedStage> terms;
  terms.reserve(static_cast<std::size_t>(row.completed_steps)*kStages+row.count);
  for (std::uint32_t step = 0; step < row.completed_steps; ++step) {
    const auto weights = StepWeights(donor, step);
    terms.insert(terms.end(), weights.begin(), weights.end());
  }
  terms.insert(terms.end(), row.terms.begin(), row.terms.begin()+row.count);
  return terms;
}

StepContext Schedule::Context(StageId id, std::uint64_t macro, double start, double dt) const {
  if (!std::isfinite(start) || !std::isfinite(dt) || dt <= 0 || !std::isfinite(start+dt))
    throw std::invalid_argument("multirate interval must have finite time and positive duration");
  static constexpr std::array<std::uint64_t, kStages> numerators{{0, 2, 3, 5}};
  StepContext context;
  context.macro_index = macro;
  context.key = stages_.at(id).key;
  context.start_time = start;
  context.base_dt = dt;
  context.local_dt = dt/Steps(context.key.level);
  context.ticks_per_base = 5*(std::uint64_t{1} << maximum_level_);
  context.stage_tick = (5*std::uint64_t(context.key.local_step)+numerators[context.key.stage])
      *(std::uint64_t{1} << (maximum_level_-context.key.level));
  context.stage_time = start+dt*(static_cast<double>(context.stage_tick)/context.ticks_per_base);
  return context;
}

std::array<WeightedStage, kStages> Schedule::StepWeights(int level, std::uint32_t step) const {
  std::array<WeightedStage, kStages> weights;
  for (int i = 0; i < kStages; ++i) weights[i] = {Id(level, step, i), kB[i]/Steps(level)};
  return weights;
}

ReadyScheduler::ReadyScheduler(const Schedule &schedule) : schedule_(&schedule) {
  const auto count = schedule.Stages().size();
  states_.assign(count, State::waiting);
  missing_.resize(count); consumers_.resize(count);
  for (StageId id = 0; id < count; ++id) {
    missing_[id] = schedule.Stage(id).predecessors.size();
    consumers_[id] = schedule.Stage(id).successors.size();
    if (missing_[id] == 0) { states_[id] = State::ready; ready_.insert(id); }
  }
}
void ReadyScheduler::Start(StageId id) {
  if (states_.at(id) != State::ready) throw std::logic_error("multirate task is not ready");
  ready_.erase(id); states_[id] = State::running;
}
void ReadyScheduler::Complete(StageId id) {
  if (states_.at(id) != State::running) throw std::logic_error("multirate task is not running");
  states_[id] = State::complete; ++completed_;
  for (auto parent : schedule_->Stage(id).predecessors) --consumers_[parent];
  for (auto child : schedule_->Stage(id).successors) {
    if (--missing_[child] == 0) { states_[child] = State::ready; ready_.insert(child); }
  }
}
bool ReadyScheduler::Running(StageId id) const { return states_.at(id) == State::running; }
bool ReadyScheduler::Completed(StageId id) const { return states_.at(id) == State::complete; }
std::size_t ReadyScheduler::RemainingConsumers(StageId id) const { return consumers_.at(id); }
bool ReadyScheduler::Releasable(StageId id) const { return Completed(id) && consumers_.at(id) == 0; }

LevelAssignment AssignLevels(const std::vector<BlockSpec> &blocks,
                             const std::vector<int> &spherical_levels,
                             const std::vector<DependencyEdge> &edges) {
  if (blocks.empty()) throw std::invalid_argument("temporal assignment requires mesh blocks");
  for (int level : spherical_levels) CheckLevel(level);
  LevelAssignment result;
  result.levels.reserve(blocks.size());
  std::unordered_map<int, std::size_t> index;
  std::map<int, std::vector<std::size_t>> agreements, scri;
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    const auto &block = blocks[i];
    if (block.gid < 0 || !index.emplace(block.gid, i).second)
      throw std::invalid_argument("temporal assignment requires unique nonnegative GIDs");
    CheckLevel(block.physical_level);
    if (!block.cartesian && (block.radial_region < 0 ||
        static_cast<std::size_t>(block.radial_region) >= spherical_levels.size()))
      throw std::invalid_argument(BlockName(block.gid)+" has no exact spherical time level");
    result.levels.push_back(block.cartesian ? block.physical_level : spherical_levels[block.radial_region]);
    if (block.agreement_group >= 0) agreements[block.agreement_group].push_back(i);
    if (block.scri_group >= 0) scri[block.scri_group].push_back(i);
  }
  for (const auto &group : agreements) {
    int target = 0;
    for (auto i : group.second) target = std::max(target, result.levels[i]);
    for (auto i : group.second) {
      if (!blocks[i].cartesian && result.levels[i] != target)
        throw std::invalid_argument("agreement group "+std::to_string(group.first)+
            " requires spherical "+BlockName(blocks[i].gid)+" at exact time level "+std::to_string(target));
      if (result.levels[i] != target) {
        result.agreement_promotions.push_back({blocks[i].gid, result.levels[i], target});
        result.levels[i] = target;
      }
    }
  }
  for (const auto &group : scri) {
    const int level = result.levels[group.second.front()];
    for (auto i : group.second)
      if (result.levels[i] != level)
        throw std::invalid_argument("scri closure group "+std::to_string(group.first)+" spans time levels");
  }
  for (const auto &edge : edges) {
    const auto a = index.find(edge.first), b = index.find(edge.second);
    if (a == index.end() || b == index.end())
      throw std::invalid_argument("temporal dependency names an unknown block GID");
    const int la = result.levels[a->second], lb = result.levels[b->second];
    if (std::abs(la-lb) > 1)
      throw std::invalid_argument("temporal 2:1 violation: "+BlockName(edge.first)+" at level "+
          std::to_string(la)+" couples to "+BlockName(edge.second)+" at level "+std::to_string(lb));
  }
  result.maximum_level = *std::max_element(result.levels.begin(), result.levels.end());
  return result;
}
}  // namespace time_integration::multirate
