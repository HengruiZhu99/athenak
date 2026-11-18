#include "driver/multirate.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace mr = time_integration::multirate;

namespace {
void Require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}
template<class F> void Throws(F f, const std::string &message) {
  bool thrown = false;
  try { f(); } catch (const std::exception &) { thrown = true; }
  Require(thrown, message);
}
bool Close(double a, double b, double tolerance = 3e-13) {
  return std::abs(a-b) <= tolerance*std::max({1., std::abs(a), std::abs(b)});
}

// A component-partitioned reference executor used to test the actual production
// coefficient graph. Each partition is a scalar here; dependencies, compact
// accepted prefixes, local commits and out-of-order execution are unchanged.
std::vector<double> Advance(const mr::Schedule &schedule, const std::vector<double> &initial,
    double start, double dt,
    const std::function<double(int, double, const std::vector<double>&)> &rhs,
    bool reverse_ready = false) {
  const int levels = schedule.MaxLevel()+1;
  std::vector<std::vector<double>> accepted(levels);
  for (int level = 0; level < levels; ++level) {
    accepted[level].assign(schedule.Steps(level)+1, std::numeric_limits<double>::quiet_NaN());
    accepted[level][0] = initial[level];
  }
  std::vector<double> slopes(schedule.Stages().size(), std::numeric_limits<double>::quiet_NaN());
  mr::ReadyScheduler ready(schedule);
  while (!ready.Finished()) {
    Require(!ready.Ready().empty(), "reference executor deadlock");
    const auto id = reverse_ready ? *ready.Ready().rbegin() : *ready.Ready().begin();
    ready.Start(id);
    const auto &stage = schedule.Stage(id);
    std::vector<double> state(levels, std::numeric_limits<double>::quiet_NaN());
    for (const auto &row : stage.rows) {
      double value = accepted[row.donor_level][row.completed_steps];
      for (int j = 0; j < row.count; ++j)
        value += dt*row.terms[j].coefficient*slopes[row.terms[j].source];
      Require(std::isfinite(value), "stage consumed unfinished donor data");
      state[row.donor_level] = value;
    }
    const auto context = schedule.Context(id, 0, start, dt);
    slopes[id] = rhs(stage.key.level, context.stage_time, state);
    Require(std::isfinite(slopes[id]), "nonfinite reference RHS");
    if (stage.CommitsStep()) {
      double value = accepted[stage.key.level][stage.key.local_step];
      for (const auto &weight : schedule.StepWeights(stage.key.level, stage.key.local_step))
        value += dt*weight.coefficient*slopes[weight.source];
      accepted[stage.key.level][stage.key.local_step+1] = value;
    }
    ready.Complete(id);
  }
  for (mr::StageId id = 0; id < schedule.Stages().size(); ++id)
    Require(ready.Releasable(id), "completed schedule retained an RHS consumer");
  std::vector<double> result(levels);
  for (int level = 0; level < levels; ++level) result[level] = accepted[level].back();
  return result;
}

void TestSchedule() {
  for (int maximum = 0; maximum <= 7; ++maximum) {
    mr::Schedule schedule(maximum);
    Require(schedule.Stages().size() == 4*((std::size_t{1} << (maximum+1))-1), "wrong stage count");
    Require(schedule.TopologicalOrder().size() == schedule.Stages().size(), "incomplete DAG");
    mr::ReadyScheduler scheduler(schedule);
    std::mt19937 generator(512+maximum);
    std::vector<mr::StageId> running;
    while (!scheduler.Finished()) {
      // Launch several tasks before completing them to exercise asynchronous
      // readiness; completion order must not be assumed to be launch order.
      if (!scheduler.Ready().empty()) {
        auto selected = scheduler.Ready().begin();
        std::advance(selected, generator()%scheduler.Ready().size());
        const auto id = *selected;
        scheduler.Start(id); running.push_back(id);
        Require(scheduler.Running(id), "started task not running");
        Throws([&] { scheduler.Start(id); }, "duplicate Start accepted");
      }
      if (!running.empty() && (generator()%2 || scheduler.Ready().empty())) {
        const auto position = generator()%running.size();
        const auto id = running[position];
        scheduler.Complete(id);
        Throws([&] { scheduler.Complete(id); }, "duplicate Complete accepted");
        running.erase(running.begin()+position);
      }
      Require(!running.empty() || !scheduler.Ready().empty() || scheduler.Finished(), "scheduler deadlock");
    }
    for (mr::StageId id = 0; id < schedule.Stages().size(); ++id) {
      Require(scheduler.Releasable(id), "lifetime did not retire");
      const auto &stage = schedule.Stage(id);
      const auto ctx = schedule.Context(id, 7, 2., .125);
      const double expected = (stage.key.local_step + std::array<double,4>{{0,.4,.6,1}}[stage.key.stage]) /
          schedule.Steps(stage.key.level);
      Require(Close(ctx.stage_time, 2.+.125*expected), "incorrect stage time");
      Require(ctx.stage_tick <= ctx.ticks_per_base, "tick exceeds interval");
      for (const auto &row : stage.rows) {
        double moment = static_cast<double>(row.completed_steps)/schedule.Steps(row.donor_level);
        for (int j = 0; j < row.count; ++j) moment += row.terms[j].coefficient;
        Require(Close(moment, expected), "inconsistent component stage time");
        double dense_sum = 0;
        for (auto term : schedule.DenseRow(id, row.donor_level)) dense_sum += term.coefficient;
        Require(Close(dense_sum, moment), "compact/dense prefix mismatch");
      }
    }
  }
  Throws([] { mr::Schedule bad(-1); }, "negative depth accepted");
  Throws([] { mr::Schedule bad(31); }, "overflow depth accepted");
  Throws([] { mr::Schedule bad(8, 64); }, "stage budget ignored");
  mr::Schedule schedule(2);
  Throws([&] { schedule.Row(schedule.Id(0,0,0),2); }, "4:1 direct coupling accepted");
  Throws([&] { schedule.Context(0,0,0,0); }, "zero base dt accepted");
  Throws([&] { schedule.Context(0,0,0,std::numeric_limits<double>::infinity()); }, "infinite dt accepted");
  mr::ReadyScheduler scheduler(schedule);
  Throws([&] { scheduler.Start(schedule.Id(0,0,3)); }, "unfinished dependency accepted");
}

void TestAssignment() {
  const std::vector<mr::BlockSpec> blocks{{4,true,0,-1,7,-1}, {10,true,2,-1,7,-1},
      {20,false,0,0,7,-1}, {30,false,0,1,-1,3}, {31,false,0,1,-1,3}};
  const std::vector<mr::DependencyEdge> edges{{4,10},{10,20},{20,30},{30,31}};
  const auto assignment = mr::AssignLevels(blocks,{2,3},edges);
  Require(assignment.levels == std::vector<int>({2,2,2,3,3}), "incorrect group assignment");
  Require(assignment.agreement_promotions.size() == 1 && assignment.agreement_promotions[0].gid == 4,
      "Cartesian agreement promotion not reported");
  Throws([&] { mr::AssignLevels(blocks,{1,3},edges); }, "exact spherical group level silently changed");
  Throws([&] { mr::AssignLevels(blocks,{2,4},edges); }, "nonlocal 4:1 donor edge accepted");
  Throws([&] { mr::AssignLevels(blocks,{2,3},{{4,999}}); }, "unknown donor GID accepted");
  auto inconsistent = blocks; inconsistent[2].scri_group = 3;
  Throws([&] { mr::AssignLevels(inconsistent,{2,3},edges); }, "mixed scri closure levels accepted");
  auto duplicate = blocks; duplicate[1].gid = duplicate[0].gid;
  Throws([&] { mr::AssignLevels(duplicate,{2,3},edges); }, "duplicate block accepted");
}

double Exact(int q, double t) { return .2+.1*q+.15*std::sin((1.+.2*q)*t); }
double Derivative(int q, double t) { return .15*(1.+.2*q)*std::cos((1.+.2*q)*t); }
double Coupled(int q, const std::vector<double> &y) {
  double value = -.4*y[q]+.07*y[q]*y[q];
  if (q>0) value += .3*std::sin(y[q-1])+.04*y[q-1]*y[q];
  if (q+1 < static_cast<int>(y.size())) value += .2*y[q+1]*y[q+1]-.03*y[q]*y[q+1];
  return value;
}
void TestConvergence() {
  for (int levels : {1,2,3,4,5}) {
    mr::Schedule schedule(levels-1);
    const auto rhs = [&](int q, double t, const std::vector<double> &y) {
      std::vector<double> exact(levels);
      for (int r = 0; r < levels; ++r) exact[r] = Exact(r,t);
      return Coupled(q,y)+Derivative(q,t)-Coupled(q,exact);
    };
    std::vector<double> errors;
    for (int steps : {5,10,20}) {
      std::vector<double> state(levels);
      for (int q = 0; q < levels; ++q) state[q] = Exact(q,0);
      const double dt = 1./steps;
      for (int step = 0; step < steps; ++step)
        state = Advance(schedule,state,step*dt,dt,rhs,step%2 != 0);
      double error = 0;
      for (int q = 0; q < levels; ++q) error = std::max(error,std::abs(state[q]-Exact(q,1)));
      errors.push_back(error);
    }
    const double order1 = std::log2(errors[0]/errors[1]), order2 = std::log2(errors[1]/errors[2]);
    std::cout << "levels=" << levels << " nonlinear errors=" << errors[0] << ',' << errors[1]
              << ',' << errors[2] << " orders=" << order1 << ',' << order2 << '\n';
    Require(order1>3.75 && order2>3.75, "coupled nonlinear temporal convergence below fourth order");
  }
  mr::Schedule schedule(3);
  const auto linear = [](int q,double,const std::vector<double>& y) {
    double v = -.2*y[q];
    if (q) v += .05*y[q-1];
    if (q+1 < static_cast<int>(y.size())) v += .03*y[q+1];
    return v;
  };
  const std::vector<double> y{1,2,3,4};
  const auto forward = Advance(schedule,y,0,.1,linear,false);
  const auto reverse = Advance(schedule,y,0,.1,linear,true);
  Require(forward == reverse, "different valid task orders changed the numerical update");
}

void DumpTableau(int levels) {
  mr::Schedule schedule(levels-1);
  std::cout << std::setprecision(17) << "{\"levels\":" << levels << ",\"rows\":[";
  bool first = true;
  for (mr::StageId id = 0; id < schedule.Stages().size(); ++id) {
    for (const auto &row : schedule.Stage(id).rows) {
      if (!first) std::cout << ',';
      first = false;
      std::cout << "{\"target\":" << id << ",\"donor\":" << row.donor_level << ",\"values\":[";
      std::vector<double> values(schedule.Steps(row.donor_level)*mr::kStages,0);
      const auto offset = schedule.Id(row.donor_level,0,0);
      for (auto term : schedule.DenseRow(id,row.donor_level)) values[term.source-offset] += term.coefficient;
      for (std::size_t j = 0; j < values.size(); ++j) {
        if (j) std::cout << ',';
        std::cout << values[j];
      }
      std::cout << "]}";
    }
  }
  std::cout << "]}\n";
}
}  // namespace

int main(int argc,char **argv) {
  try {
    if (argc == 3 && std::string(argv[1]) == "--tableau") { DumpTableau(std::stoi(argv[2])); return 0; }
    TestSchedule(); TestAssignment(); TestConvergence();
    std::cout << "multirate numerical and scheduling checks passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "multirate test failed: " << error.what() << '\n';
    return 1;
  }
}
