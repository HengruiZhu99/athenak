// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include "mesh/multirate_load_balance.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace mb = multirate::balance;
namespace {
void Check(bool test,const char *what) {if (!test) throw std::runtime_error(what);}
void Near(double a,double b,const char *what) {
  Check(std::abs(a-b)<=1e-12*std::max({1.,std::abs(a),std::abs(b)}),what);
}
template<class F> void Reject(F f,const char *what) {
  bool rejected=false;try {f();} catch (const std::exception &) {rejected=true;}
  Check(rejected,what);
}
mb::WorkGraph Graph(int count) {
  mb::WorkGraph g;
  for (int i=0;i<count;++i) {
    mb::BlockCost b;b.gid=i;b.time_level=i%2;b.work_units=1;b.memory_bytes=100;
    b.phase_costs={1,static_cast<double>(i%2)};g.blocks.push_back(b);
    if (i) g.edges.push_back({i-1,i,16,2});
  }
  return g;
}
void Profiles() {
  mb::BatchCostModel p;
  Check(!p.Calibrated(mb::KernelKind::scri),"unexpected calibrated profile");
  Near(p.Predict(mb::KernelKind::scri,0),0,"empty batch cost");
  p.Observe(mb::KernelKind::scri,10,2);p.Observe(mb::KernelKind::scri,10,4);
  p.Observe(mb::KernelKind::scri,20,5);
  Near(p.Predict(mb::KernelKind::scri,15),4,"profile interpolation");
  Near(p.Predict(mb::KernelKind::scri,5),3,"launch-cost floor");
  Near(p.Predict(mb::KernelKind::scri,40),10,"profile extrapolation");
  p.Observe(mb::KernelKind::scri,30,4);  // Noise must not make a bigger batch cheaper.
  Near(p.Predict(mb::KernelKind::scri,25),5,"monotone timing envelope");
  const auto text=p.Serialize();auto q=mb::BatchCostModel::Deserialize(text);
  Check(q.Serialize()==text,"profile exact serialization");
  q.Observe(mb::KernelKind::scri,10,6);
  Near(q.Predict(mb::KernelKind::scri,10),4,"profile sample count restored");
  p.Merge(q);
  Near(p.Predict(mb::KernelKind::scri,10),3.6,"cross-rank profile count weighting");
  mb::BatchCostModel other;other.Observe(mb::KernelKind::cartesian,8,2.5);
  p.Merge(other);
  Near(p.Predict(mb::KernelKind::cartesian,8),2.5,"rank-specific kernel sample lost");
  auto combined=mb::BatchCostModel::Deserialize(p.Serialize());
  combined.Observe(mb::KernelKind::scri,10,6);
  Near(combined.Predict(mb::KernelKind::scri,10),4,"merged profile count not retained");
  Reject([&]{p.Observe(mb::KernelKind::scri,0,1);},"zero-unit observation accepted");
  Reject([&]{p.Observe(mb::KernelKind::scri,1,std::numeric_limits<double>::quiet_NaN());},"NaN observation accepted");
  Reject([&]{mb::BatchCostModel::Deserialize(text+"corruption");},"profile trailing corruption accepted");
}
void IdentityAndReplay() {
  auto g=Graph(8);auto hash=g.Fingerprint();
  std::reverse(g.blocks.begin(),g.blocks.end());std::reverse(g.edges.begin(),g.edges.end());
  g.blocks[0].phase_costs[0]=9;
  Check(hash==g.Fingerprint(),"identity depends on ordering or measured timing");
  mb::PartitionRecord r;r.interval=128;r.graph_fingerprint=hash;r.ranks=2;
  r.owner_by_gid={0,1,0,1,0,1,0,1};
  auto text=r.Serialize();auto restored=mb::PartitionRecord::Deserialize(text);
  restored.ValidateFor(g,2);Check(text==restored.Serialize(),"partition exact replay roundtrip");
  Reject([&]{restored.ValidateFor(g,3);},"changed replay rank count accepted");
  g.blocks[0].time_level++;
  Reject([&]{restored.ValidateFor(g,2);},"changed replay time level accepted");
  Reject([&]{mb::PartitionRecord::Deserialize(text.substr(0,text.size()-4));},"truncated replay accepted");
  Reject([&]{mb::PartitionRecord::Deserialize(text+"x");},"trailing replay corruption accepted");
  g.blocks[0].gid=g.blocks[1].gid;Reject([&]{g.Validate();},"duplicate GID accepted");
}
void DagScoring() {
  auto g=Graph(2);mb::BatchCostModel p;
  p.Observe(mb::KernelKind::cartesian,1,4);p.Observe(mb::KernelKind::cartesian,2,4);
  mb::BalanceOptions o;o.network_latency_seconds=1;o.network_bytes_per_second=10;
  std::vector<mb::ExecutionTask> tasks{
    {0,0,mb::KernelKind::cartesian,1,{}},
    {1,0,mb::KernelKind::cartesian,1,{}},
    {0,1,mb::KernelKind::cartesian,1,{{1,10}}},
    {1,1,mb::KernelKind::cartesian,1,{{0,10}}}};
  auto distributed=mb::ScorePartition(g,2,{0,1},o,p,tasks);
  Near(distributed.seconds,10,"DAG transfer and parallel rank timing");
  Near(distributed.rank_busy_seconds[0],8,"rank busy excludes MPI wait");
  o.require_nonempty_ranks=false;
  auto batched=mb::ScorePartition(g,2,{0,0},o,p,tasks);
  Near(batched.seconds,8,"same-stage local GPU batching");
  Check(batched.cut_bytes_per_interval==0,"local edge counted as remote");
  tasks[0].dependencies={{2,0}};  // 0 -> 3 and 1 -> 2; add both to form a cycle.
  tasks[1].dependencies={{3,0}};
  Reject([&]{mb::ScorePartition(g,2,{0,1},o,p,tasks);},"cyclic DAG accepted");
  o.memory_capacity_per_rank=150;
  Check(!mb::ScorePartition(g,2,{0,0},o,p).feasible,"rank memory limit ignored");
}
void SparseCompaction() {
  mb::WorkGraph g;std::vector<mb::ExecutionTask> tasks;
  for (int i=0;i<8;++i) {
    mb::BlockCost b;b.gid=i;b.work_units=1;b.memory_bytes=100;
    b.time_level=i<4?0:3;b.phase_costs=i<4?std::vector<double>{.01,0}:std::vector<double>{0,4};
    g.blocks.push_back(b);
    tasks.push_back({i,0,mb::KernelKind::cartesian,1,{}});
  }
  for (int i=4;i<8;++i) {
    mb::ExecutionTask t{i,1,mb::KernelKind::cartesian,1,{}};
    for (int j=4;j<8;++j) t.dependencies.push_back({j,1});
    tasks.push_back(t);
    for (int j=4;j<i;++j) g.edges.push_back({j,i,1,1});
  }
  mb::BatchCostModel p;p.Observe(mb::KernelKind::cartesian,1,1);
  p.Observe(mb::KernelKind::cartesian,8,1);
  mb::BalanceOptions o;o.backend=mb::Backend::scalar_reference;
  o.network_latency_seconds=10;o.network_bytes_per_second=1e9;
  std::vector<int> current={0,1,2,3,0,1,2,3};
  auto before=mb::ScorePartition(g,4,current,o,p,tasks);
  auto result=mb::Partition(g,4,o,p,tasks,current);
  Check(result.score.seconds<before.seconds,"sparse fast region was unnecessarily spread across GPUs");
  Check(result.selected_candidate.find("compact")!=std::string::npos,"compaction candidate not selected");
  Check(result.candidates.size()>2,"sparse candidates not evaluated");
}
void BackendAndDeterminism() {
  auto g=Graph(24);mb::BalanceOptions o;mb::BatchCostModel p;
  if (mb::MetisAvailable()) {
    auto a=mb::Partition(g,4,o,p);auto b=mb::Partition(g,4,o,p);
    Check(a.used_metis && b.used_metis,"METIS backend was not executed");
    Check(a.owner_by_gid==b.owner_by_gid,"fixed-seed partition is nondeterministic");
    Check(a.owner_by_gid.size()==24 && a.score.feasible,"invalid METIS candidate");
    bool found=false;for (const auto &c:a.candidates) found|=c.name=="metis-stage-vector";
    Check(found,"METIS stage-vector candidate absent");
    // Equal total work can hide maximal phase imbalance: scalar LPT puts all
    // even GIDs on rank 0 and all odd GIDs on rank 1. The two phase constraints
    // must distribute both classes, not merely reproduce that scalar result.
    auto phases=Graph(16);
    for (auto &block:phases.blocks) block.phase_costs=block.gid%2
        ? std::vector<double>{0,1} : std::vector<double>{1,0};
    auto scalar_options=o;scalar_options.backend=mb::Backend::scalar_reference;
    scalar_options.consider_sparse_compaction=false;
    auto scalar=mb::Partition(phases,2,scalar_options,p);
    auto vector_options=o;vector_options.consider_sparse_compaction=false;
    auto vector=mb::Partition(phases,2,vector_options,p);
    Check(vector.score.seconds<.75*scalar.score.seconds,
          "METIS phase constraints failed to improve equal-total-work phase imbalance");
  } else {
    Reject([&]{mb::Partition(g,4,o,p);},"missing METIS silently fell back");
  }
  o.backend=mb::Backend::scalar_reference;
  auto a=mb::Partition(g,4,o,p);Check(!a.used_metis,"scalar backend claims METIS");
  o.memory_capacity_per_rank=50;
  Reject([&]{mb::Partition(g,4,o,p);},"impossible rank capacity accepted");
}
void FineGpuCoverage() {
  mb::WorkGraph graph;
  for(int g=0;g<48;++g) {
    mb::BlockCost b;
    b.gid=g;b.time_level=g<8?4:0;b.work_units=1;b.memory_bytes=1;b.phase_costs={1.};
    graph.blocks.push_back(b);
  }
  mb::BalanceOptions options;
  options.backend=mb::Backend::scalar_reference;options.consider_sparse_compaction=false;
  options.spread_finest=true;options.ranks_per_gpu=2;
  std::vector<int> current(48);
  for(int g=0;g<48;++g) current[g]=g/6;
  auto placement=mb::Partition(graph,8,options,{}, {},current);
  std::vector<int> per_tile(8);
  for(int g=0;g<8;++g) ++per_tile.at(placement.owner_by_gid[g]);
  Check(std::all_of(per_tile.begin(),per_tile.end(),[](int count){return count==1;}),
        "eight fast blocks did not cover four GPUs and their eight tiles");
}
void Rebalance() {
  Check(!mb::ConsiderRebalance(31,100,true,1,.5,1).apply,"rebalance before warmup");
  Check(!mb::ConsiderRebalance(32,100,false,1,.5,1).apply,"midstep rebalance allowed");
  Check(!mb::ConsiderRebalance(33,100,true,1,.5,1).apply,"off-cadence rebalance allowed");
  Check(!mb::ConsiderRebalance(32,1,true,1,.5,1).apply,"unamortized migration allowed");
  Check(!mb::ConsiderRebalance(32,100,true,1,.99,0).apply,"noise-driven migration allowed");
  auto d=mb::ConsiderRebalance(32,100,true,1,.5,1);
  Check(d.apply,"useful synchronized rebalance rejected");Near(d.predicted_net_saving,49,"net saving");
}
}  // namespace

int main() {
  try {
    Profiles();IdentityAndReplay();DagScoring();SparseCompaction();BackendAndDeterminism();FineGpuCoverage();Rebalance();
    std::cout<<"multirate balance tests passed; METIS="<<mb::MetisAvailable()<<'\n';
  } catch (const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
