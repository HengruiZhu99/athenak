#include <cmath>
#include <iostream>
#include <map>
#include <stdexcept>
#include "driver/subcycle_schedule.hpp"
void Check(bool x) { if (!x) throw std::runtime_error("schedule regression"); }
struct TestState {
  std::map<int, std::uint64_t> times, predictors;
  std::map<int, int> calls;
  int finest;
  void Advance(const subcycling::StepContext &s) {
    for (int l=s.minimum_level; l<=s.maximum_level; ++l) {
      Check(times[l] == s.begin_tick);
      if (l>s.minimum_level || !predictors.count(l-1)) continue;
      Check(predictors[l-1] >= s.end_tick);
    }
    Check(s.StartTime() < s.EndTime());
    Check(s.StageTime(0) == s.StartTime());
    Check(s.StageTime(1) == s.EndTime());
    for (int l=s.minimum_level; l<=s.maximum_level; ++l) {
      predictors[l]=s.end_tick; ++calls[l];
    }
  }
  void Synchronize(const subcycling::StepContext &s) {
    if (s.maximum_level < finest) Check(times[s.maximum_level+1] == s.end_tick);
    for (int l=s.minimum_level; l<=s.maximum_level; ++l) times[l]=s.end_tick;
  }
};
int main() {
  for (unsigned ratio : {1,2,4,16,32}) {
    subcycling::Schedule scheduler(3,21,ratio);
    TestState state; state.finest=21;
    scheduler.Run(62.2,1.e-4,state);
    for (int l=3; l<=21; ++l) {
      Check(state.times[l] == scheduler.Ticks());
      const int expected = std::max(1,static_cast<int>(ratio) >> (21-l));
      Check(state.calls[l] == expected);
    }
  }
  subcycling::Schedule short_tree(3,4,32);
  Check(short_tree.Ticks() == 2);
  for (unsigned bad : {0,3,6}) {
    bool rejected=false;
    try { subcycling::Schedule invalid(0,4,bad); }
    catch (const std::invalid_argument &) { rejected=true; }
    Check(rejected);
  }
  // A failing evolution callback propagates immediately; no further steps or
  // synchronization occurs. The owning driver must restore its interval backup.
  struct Failure {
    int calls=0;
    void Advance(const subcycling::StepContext &) {
      ++calls; throw std::runtime_error("evolution failure");
    }
    void Synchronize(const subcycling::StepContext &) { Check(false); }
  } failure;
  bool failed=false;
  try { short_tree.Run(0,1,failure); } catch (const std::runtime_error &) { failed=true; }
  Check(failed && failure.calls == 1);
  std::cout << "PASS: parent prediction, child synchronization, capped ratios, failure propagation\n";
}
