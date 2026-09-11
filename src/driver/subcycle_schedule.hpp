#ifndef DRIVER_SUBCYCLE_SCHEDULE_HPP_
#define DRIVER_SUBCYCLE_SCHEDULE_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace subcycling {
// Limits already include their respective safety factors. In particular, the
// caller applies CFL to the spatial ceiling, not to the explicit-source ceiling.
struct LevelStepLimit {
  int level;
  double spatial, source;
};
enum class IntervalLimiter { synchronization, spatial, source };
struct IntervalChoice {
  double dt;
  IntervalLimiter limiter;
  int level;  // -1 for the requested synchronization/end-time cap
};
// Integer ticks identify common physical times without accumulating floating
// point drift. The coarsest group contains [minimum_level, first_subcycled_level].
struct StepContext {
  int minimum_level, maximum_level;
  std::uint64_t begin_tick, end_tick, total_ticks;
  double interval_start, interval_dt;
  double Time(std::uint64_t tick) const {
    return interval_start + interval_dt*(static_cast<double>(tick)/total_ticks);
  }
  double StartTime() const { return Time(begin_tick); }
  double EndTime() const { return Time(end_tick); }
  double Dt() const {
    return interval_dt*(static_cast<double>(end_tick-begin_tick)/total_ticks);
  }
  double StageTime(double fraction) const {
    return interval_start + interval_dt*
        ((begin_tick+fraction*(end_tick-begin_tick))/total_ticks);
  }
};

// The callbacks own numerical state, parent predictors and rollback. Advance
// must leave a usable coarse predictor before recursion. Synchronize is called
// only after every child reaches this parent's end time. It performs restriction
// and coincident-node ownership resolution, not another evolution step.
// No regrid/output callback is made within this interval.
class Schedule {
 public:
  Schedule(int minimum_level, int maximum_level, unsigned maximum_ratio)
      : minimum_(minimum_level), maximum_(maximum_level) {
    if (minimum_ < 0 || maximum_ < minimum_ || maximum_ratio == 0 ||
        (maximum_ratio & (maximum_ratio-1)) != 0 || maximum_ratio > (1U<<20)) {
      throw std::invalid_argument("invalid subcycling levels or power-of-two ratio");
    }
    unsigned depth = 0;
    while ((1U<<depth) < maximum_ratio) ++depth;
    first_ = std::max(minimum_, maximum_-static_cast<int>(depth));
    ticks_ = std::uint64_t{1} << (maximum_-first_);
  }
  std::uint64_t Ticks() const { return ticks_; }
  std::uint64_t Substeps(int level) const {
    if (level < minimum_ || level > maximum_)
      throw std::invalid_argument("timestep level outside hierarchy");
    return std::uint64_t{1} << std::max(0, level-first_);
  }
  // Every evolved level, including covered predictors, must supply a contract.
  // Ascending order makes ties deterministic and catches missing/duplicate levels.
  IntervalChoice ChooseInterval(const std::vector<LevelStepLimit> &limits,
                                double cap) const {
    if (!std::isfinite(cap) || cap <= 0 ||
        limits.size() != static_cast<std::size_t>(maximum_-minimum_+1))
      throw std::invalid_argument("invalid subcycling timestep coverage or cap");
    IntervalChoice result{cap, IntervalLimiter::synchronization, -1};
    for (std::size_t n=0; n<limits.size(); ++n) {
      const auto &limit=limits[n];
      if (limit.level != minimum_+static_cast<int>(n) ||
          !std::isfinite(limit.spatial) || limit.spatial <= 0 ||
          !std::isfinite(limit.source) || limit.source <= 0)
        throw std::invalid_argument("invalid per-level subcycling timestep limit");
      const double q=static_cast<double>(Substeps(limit.level));
      // Compare before multiplication so an effectively unlimited source ceiling
      // (numeric_limits::max()) cannot overflow in a deep hierarchy.
      if (limit.spatial < result.dt/q)
        result={limit.spatial*q, IntervalLimiter::spatial, limit.level};
      if (limit.source < result.dt/q)
        result={limit.source*q, IntervalLimiter::source, limit.level};
    }
    return result;
  }
  template <typename Callbacks>
  void Run(double start, double dt, Callbacks &callbacks) const {
    if (!std::isfinite(start) || !std::isfinite(dt) || dt <= 0 ||
        !std::isfinite(start+dt) || start+dt == start ||
        start+dt/static_cast<double>(ticks_) == start) {
      throw std::invalid_argument("invalid subcycling interval time");
    }
    Recurse(first_, 0, ticks_, start, dt, callbacks);
  }
 private:
  int minimum_, maximum_, first_;
  std::uint64_t ticks_;
  template <typename Callbacks>
  void Recurse(int level, std::uint64_t begin, std::uint64_t end,
               double start, double dt, Callbacks &callbacks) const {
    StepContext step{level == first_ ? minimum_ : level, level,
                     begin, end, ticks_, start, dt};
    callbacks.Advance(step);
    if (level < maximum_) {
      const auto middle = begin+(end-begin)/2;
      Recurse(level+1, begin, middle, start, dt, callbacks);
      Recurse(level+1, middle, end, start, dt, callbacks);
    }
    callbacks.Synchronize(step);
  }
};
}  // namespace subcycling
#endif  // DRIVER_SUBCYCLE_SCHEDULE_HPP_
