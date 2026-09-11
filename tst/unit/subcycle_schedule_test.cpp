#include <cmath>
#include <iostream>
#include <map>
#include <limits>
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
      Check(scheduler.Substeps(l) == static_cast<unsigned>(state.calls[l]));
    }
  }
  subcycling::Schedule short_tree(3,4,32);
  Check(short_tree.Ticks() == 2);
  // Coarse levels share one step; finer levels get successively more steps.
  // A coarse source limit must constrain the entire synchronization interval.
  subcycling::Schedule limited(3,6,4);
  const double unlimited=std::numeric_limits<double>::max();
  std::vector<subcycling::LevelStepLimit> limits{
    {3,1.,unlimited},{4,.5,unlimited},{5,.25,unlimited},{6,.125,unlimited}};
  auto choice=limited.ChooseInterval(limits,2.);
  Check(choice.dt==.5 && choice.level==4 &&
        choice.limiter==subcycling::IntervalLimiter::spatial);
  limits[0].source=.125;
  choice=limited.ChooseInterval(limits,2.);
  Check(choice.dt==.125 && choice.level==3 &&
        choice.limiter==subcycling::IntervalLimiter::source);
  limits[0].source=unlimited;
  limits[3].source=.0625;
  choice=limited.ChooseInterval(limits,2.);
  Check(choice.dt==.25 && choice.level==6);
  for(const auto &limit:limits) {
    const double dt=choice.dt/limited.Substeps(limit.level);
    Check(dt<=limit.spatial && dt<=limit.source);
  }
  choice=limited.ChooseInterval(limits,.03125);
  Check(choice.dt==.03125 && choice.level==-1 &&
        choice.limiter==subcycling::IntervalLimiter::synchronization);
  // Missing, duplicate, out-of-order, nonfinite and nonpositive contracts fail
  // instead of allowing one unqualified predictor level to take a large step.
  for(int bad=0;bad<8;++bad) {
    auto invalid=limits;double cap=1.;
    if(bad==0) invalid.pop_back();
    if(bad==1) invalid[1].level=3;
    if(bad==2) std::swap(invalid[1],invalid[2]);
    if(bad==3) invalid[0].source=INFINITY;
    if(bad==4) invalid[0].spatial=NAN;
    if(bad==5) invalid[0].source=0.;
    if(bad==6) cap=-1.;
    if(bad==7) cap=INFINITY;
    bool rejected=false;
    try { limited.ChooseInterval(invalid,cap); }
    catch(const std::invalid_argument &) { rejected=true; }
    Check(rejected);
  }
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
  std::cout << "PASS: parent prediction, child synchronization, capped ratios, "
               "per-level timestep selection, failure propagation\n";
}
