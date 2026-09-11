#ifndef DRIVER_SUBCYCLE_SCHEDULE_HPP_
#define DRIVER_SUBCYCLE_SCHEDULE_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace subcycling {
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
