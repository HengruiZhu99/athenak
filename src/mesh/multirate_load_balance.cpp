// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include "mesh/multirate_load_balance.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

#if defined(ATHENA_MULTIRATE_METIS) && ATHENA_MULTIRATE_METIS
#include <metis.h>
#endif

namespace multirate::balance {
namespace {
bool KindValid(KernelKind k) {
  return static_cast<int>(k)>=0 && k<KernelKind::count;
}
void Require(bool ok, const char *message) {
  if (!ok) throw std::invalid_argument(message);
}
std::uint64_t Add(std::uint64_t a,std::uint64_t b) {
  if (a>std::numeric_limits<std::uint64_t>::max()-b)
    throw std::overflow_error("Multirate cost counter overflow");
  return a+b;
}
std::uint64_t Multiply(std::uint64_t a,std::uint64_t b) {
  if (b && a>std::numeric_limits<std::uint64_t>::max()/b)
    throw std::overflow_error("Multirate communication counter overflow");
  return a*b;
}
void OptionsValid(const BalanceOptions &o) {
  Require((o.backend==Backend::metis || o.backend==Backend::scalar_reference) && o.seed>=0,
          "Invalid multirate partition backend or seed");
  Require(std::isfinite(o.imbalance) && o.imbalance>=1,"Invalid partition imbalance");
  Require(o.minimum_blocks_per_rank_per_phase>0 && o.compact_batch_blocks>0,
          "Invalid partition batch population");
  Require(std::isfinite(o.network_latency_seconds) && o.network_latency_seconds>=0 &&
          std::isfinite(o.network_bytes_per_second) && o.network_bytes_per_second>0,
          "Invalid multirate network model");
}
std::vector<const BlockCost *> ByGid(const WorkGraph &g) {
  std::vector<const BlockCost *> b(g.blocks.size());
  for (const auto &v:g.blocks) b[v.gid]=&v;
  return b;
}
void OwnersValid(std::size_t n,int ranks,const std::vector<int> &owners) {
  Require(ranks>0 && owners.size()==n,"Invalid multirate owner map size");
  for (int owner:owners) Require(owner>=0 && owner<ranks,"Invalid multirate owner rank");
}
double Cost(const BlockCost &b) {
  return std::accumulate(b.phase_costs.begin(),b.phase_costs.end(),0.0);
}
void NoTrailing(std::istream &in) {
  in>>std::ws;
  Require(in.eof(),"Unexpected trailing multirate checkpoint data");
}

std::vector<int> ScalarPartition(const WorkGraph &g,int ranks) {
  const auto blocks=ByGid(g);
  std::vector<int> order(g.blocks.size());std::iota(order.begin(),order.end(),0);
  std::stable_sort(order.begin(),order.end(),[&](int a,int b) {
    return Cost(*blocks[a])>Cost(*blocks[b]);
  });
  std::vector<int> owners(g.blocks.size());std::vector<double> loads(ranks,0);
  // Longest-processing-time scalar baseline; unlike the production legacy
  // contiguous partition, this comparison does not conflate GID ordering and cost.
  for (int gid:order) {
    int rank=std::min_element(loads.begin(),loads.end())-loads.begin();
    owners[gid]=rank;loads[rank]+=Cost(*blocks[gid]);
  }
  return owners;
}

void RepairEmpty(const WorkGraph &g,int ranks,std::vector<int> &owners) {
  const auto blocks=ByGid(g);std::vector<int> count(ranks,0);
  for (int owner:owners) ++count[owner];
  for (int rank=0;rank<ranks;++rank) if (!count[rank]) {
    int candidate=-1;
    for (std::size_t gid=0;gid<owners.size();++gid) if (count[owners[gid]]>1) {
      if (candidate<0 || Cost(*blocks[gid])<Cost(*blocks[candidate])) candidate=gid;
    }
    if (candidate<0) throw std::invalid_argument("More nonempty ranks than blocks");
    --count[owners[candidate]];owners[candidate]=rank;++count[rank];
  }
}

// Keep every rank's block count fixed. Exchange a fast leaf and a slower leaf
// chosen by the change in donor traffic. First balance GPUs, then their tiles.
void SpreadFinest(const WorkGraph &g, int ranks, const BalanceOptions &options,
                  std::vector<int> &owners) {
  const int tiles=options.ranks_per_gpu;
  Require(tiles>0 && ranks%tiles==0,"ranks_per_gpu must divide the MPI rank count");
  const auto blocks=ByGid(g);
  int finest=0;
  for (const auto *b:blocks) finest=std::max(finest,b->time_level);
  if (!finest) return;
  std::vector<std::vector<std::pair<int,double>>> neighbors(blocks.size());
  for (const auto &e:g.edges) if (e.a!=e.b) {
    const double cost=e.exchanges*(options.network_latency_seconds+e.bytes/options.network_bytes_per_second);
    neighbors[e.a].emplace_back(e.b,cost);
    neighbors[e.b].emplace_back(e.a,cost);
  }
  auto delta=[&](int gid,int rank,int swapped) {
    double result=0;
    for (const auto &[neighbor,cost]:neighbors[gid]) if (neighbor!=swapped)
      result+=cost*((owners[neighbor]!=rank)-(owners[neighbor]!=owners[gid]));
    return result;
  };
  auto distribute=[&](int first,int end,int group_size) {
    int groups=(end-first)/group_size;
    std::vector<int> count(groups,0);
    for (const auto *b:blocks) {
      int rank=owners[b->gid];
      if (b->time_level==finest && rank>=first && rank<end) ++count[(rank-first)/group_size];
    }
    while (groups>1) {
      int low=*std::min_element(count.begin(),count.end()),high=*std::max_element(count.begin(),count.end());
      if (high-low<=1) break;
      double best=std::numeric_limits<double>::infinity();
      int from=-1,to=-1;
      for (const auto *fast:blocks) {
        const int a=fast->gid,ra=owners[a];
        if (fast->time_level!=finest || ra<first || ra>=end || count[(ra-first)/group_size]!=high) continue;
        for (const auto *slow:blocks) {
          const int b=slow->gid,rb=owners[b];
          if (slow->time_level==finest || rb<first || rb>=end || count[(rb-first)/group_size]!=low) continue;
          const double cost=delta(a,rb,b)+delta(b,ra,a);
          if (cost<best) {best=cost;from=a;to=b;}
        }
      }
      // A device with only fine leaves has no slower block to exchange.
      if (from<0) break;
      --count[(owners[from]-first)/group_size];
      ++count[(owners[to]-first)/group_size];
      std::swap(owners[from],owners[to]);
    }
  };
  distribute(0,ranks,tiles);
  for (int first=0;first<ranks;first+=tiles) distribute(first,first+tiles,1);
}

std::vector<int> MetisPartition(const WorkGraph &g,int ranks,const BalanceOptions &o) {
#if defined(ATHENA_MULTIRATE_METIS) && ATHENA_MULTIRATE_METIS
  const auto blocks=ByGid(g);const std::size_t n=blocks.size();
  Require(n<=static_cast<std::size_t>(std::numeric_limits<idx_t>::max()),
          "Multirate graph exceeds METIS index width");
  if (ranks==1) return std::vector<int>(n,0);
  std::vector<std::vector<double>> constraints;
  constraints.emplace_back(n);
  for (std::size_t i=0;i<n;++i) constraints.back()[i]=Cost(*blocks[i]);
  for (std::size_t p=0;p<blocks[0]->phase_costs.size();++p) {
    std::size_t active=0;double sum=0;
    for (const auto *b:blocks) {active+=b->phase_costs[p]>0;sum+=b->phase_costs[p];}
    if (sum>0 && active>=static_cast<std::size_t>(ranks)*o.minimum_blocks_per_rank_per_phase) {
      constraints.emplace_back(n);
      for (std::size_t i=0;i<n;++i) constraints.back()[i]=blocks[i]->phase_costs[p];
    }
  }
  std::uint64_t total_memory=0;
  for (const auto *b:blocks) total_memory=Add(total_memory,b->memory_bytes);
  if (total_memory) {
    constraints.emplace_back(n);
    for (std::size_t i=0;i<n;++i) constraints.back()[i]=blocks[i]->memory_bytes;
  }
  idx_t nvtxs=n,ncon=constraints.size(),nparts=ranks;
  Require(constraints.size()<=static_cast<std::size_t>(std::numeric_limits<idx_t>::max()),
          "Too many METIS constraints");
  std::vector<idx_t> weights(n*constraints.size());
  // Normalize every constraint independently; preserve true zeros. A bounded
  // total prevents overflow inside METIS even with 32-bit idx_t.
  const double budget=std::min(1e8,static_cast<double>(std::numeric_limits<idx_t>::max())/8);
  Require(static_cast<double>(n)<budget,"Graph too large for safe METIS cost quantization");
  for (std::size_t p=0;p<constraints.size();++p) {
    const double sum=std::accumulate(constraints[p].begin(),constraints[p].end(),0.0);
    const double scale=(budget-n)/sum;
    for (std::size_t i=0;i<n;++i)
      weights[i*constraints.size()+p]=constraints[p][i]>0
          ? std::max<idx_t>(1,static_cast<idx_t>(std::llround(constraints[p][i]*scale))) : 0;
  }
  std::map<std::pair<int,int>,double> edge_cost;
  for (const auto &e:g.edges) if (e.a!=e.b && e.exchanges) {
    auto pair=std::minmax(e.a,e.b);
    edge_cost[{pair.first,pair.second}]+=e.exchanges*(o.network_latency_seconds+
                                      e.bytes/o.network_bytes_per_second);
  }
  double sum=0;for (const auto &e:edge_cost) sum+=2*e.second;
  Require(2*static_cast<double>(edge_cost.size())<budget,
          "Graph too large for safe METIS edge quantization");
  const double scale=sum>0 ? std::max(0.0,budget-2*edge_cost.size())/sum : 1;
  std::vector<std::vector<std::pair<int,idx_t>>> adjacency(n);
  for (const auto &e:edge_cost) {
    idx_t w=std::max<idx_t>(1,static_cast<idx_t>(std::llround(e.second*scale)));
    adjacency[e.first.first].push_back({e.first.second,w});
    adjacency[e.first.second].push_back({e.first.first,w});
  }
  std::vector<idx_t> xadj(n+1,0),adjncy,adjwgt;
  for (std::size_t i=0;i<n;++i) {
    for (const auto &e:adjacency[i]) {adjncy.push_back(e.first);adjwgt.push_back(e.second);}
    Require(adjncy.size()<=static_cast<std::size_t>(std::numeric_limits<idx_t>::max()),
            "METIS adjacency index overflow");
    xadj[i+1]=adjncy.size();
  }
  idx_t options[METIS_NOPTIONS];METIS_SetDefaultOptions(options);
  options[METIS_OPTION_SEED]=o.seed;options[METIS_OPTION_NUMBERING]=0;
  options[METIS_OPTION_NCUTS]=1;options[METIS_OPTION_OBJTYPE]=METIS_OBJTYPE_CUT;
  std::vector<real_t> ubvec(ncon,static_cast<real_t>(o.imbalance));
  std::vector<idx_t> part(n);idx_t objval=0;
  const int status=METIS_PartGraphKway(&nvtxs,&ncon,xadj.data(),adjncy.data(),
      weights.data(),nullptr,adjwgt.empty()?nullptr:adjwgt.data(),&nparts,
      nullptr,ubvec.data(),options,&objval,part.data());
  if (status!=METIS_OK) throw std::runtime_error("METIS multirate partition failed: "+std::to_string(status));
  return std::vector<int>(part.begin(),part.end());
#else
  (void)g;(void)ranks;(void)o;
  throw std::runtime_error("METIS multirate backend requested, but ATHENA_MULTIRATE_METIS is disabled");
#endif
}
}  // namespace

void BatchCostModel::Observe(KernelKind kind,std::uint64_t units,double seconds) {
  Require(KindValid(kind) && units>0 && std::isfinite(seconds) && seconds>0,
          "Invalid multirate batch timing");
  auto &sample=samples_[kind][units];sample.units=units;
  if (sample.count==std::numeric_limits<std::uint64_t>::max())
    throw std::overflow_error("Multirate profile sample count overflow");
  ++sample.count;sample.seconds+=(seconds-sample.seconds)/sample.count;
}

void BatchCostModel::Merge(const BatchCostModel &other) {
  for (const auto &kind:other.samples_) for (const auto &entry:kind.second) {
    auto &sample=samples_[kind.first][entry.first];
    const auto &incoming=entry.second;
    const auto combined=Add(sample.count,incoming.count);
    const double fraction=static_cast<long double>(incoming.count)/combined;
    sample.units=incoming.units;
    sample.seconds+=(incoming.seconds-sample.seconds)*fraction;
    sample.count=combined;
  }
}

double BatchCostModel::Predict(KernelKind kind,std::uint64_t units) const {
  Require(KindValid(kind),"Invalid multirate kernel class");
  if (!units) return 0;
  const auto found=samples_.find(kind);
  if (found==samples_.end() || found->second.empty())
    return 1e-6+1e-9*units;  // Explicit uncalibrated launch-plus-linear seed.
  std::uint64_t previous_units=0;double previous_seconds=0;
  for (const auto &entry:found->second) {
    const auto &sample=entry.second;
    const double seconds=std::max(previous_seconds,sample.seconds);
    if (units<=sample.units) {
      if (!previous_units) return seconds;
      const double f=static_cast<double>(units-previous_units)/(sample.units-previous_units);
      return previous_seconds+f*(seconds-previous_seconds);
    }
    previous_units=sample.units;previous_seconds=seconds;
  }
  return previous_seconds*static_cast<double>(units)/previous_units;
}

bool BatchCostModel::Calibrated(KernelKind kind) const {
  Require(KindValid(kind),"Invalid multirate kernel class");
  const auto found=samples_.find(kind);return found!=samples_.end() && !found->second.empty();
}

std::string BatchCostModel::Serialize() const {
  std::size_t count=0;for (const auto &kind:samples_) count+=kind.second.size();
  std::ostringstream out;out<<"multirate-batch-profile-v1 "<<count<<'\n'<<std::setprecision(17);
  for (const auto &kind:samples_) for (const auto &entry:kind.second) {
    const auto &s=entry.second;
    out<<static_cast<int>(kind.first)<<' '<<s.units<<' '<<s.count<<' '<<s.seconds<<'\n';
  }
  return out.str();
}

BatchCostModel BatchCostModel::Deserialize(const std::string &text) {
  std::istringstream in(text);std::string magic;std::size_t count;
  Require(static_cast<bool>(in>>magic>>count) && magic=="multirate-batch-profile-v1" &&
          count<=text.size(),"Invalid multirate batch profile header");
  BatchCostModel model;
  for (std::size_t i=0;i<count;++i) {
    int kind;Sample sample;
    Require(static_cast<bool>(in>>kind>>sample.units>>sample.count>>sample.seconds),
            "Truncated multirate batch profile");
    const auto key=static_cast<KernelKind>(kind);
    Require(KindValid(key) && sample.units>0 && sample.count>0 &&
            std::isfinite(sample.seconds) && sample.seconds>0,
            "Invalid multirate batch profile sample");
    Require(model.samples_[key].emplace(sample.units,sample).second,
            "Duplicate multirate batch profile sample");
  }
  NoTrailing(in);return model;
}

void WorkGraph::Validate() const {
  Require(!blocks.empty(),"Empty multirate balance graph");
  const std::size_t phases=blocks[0].phase_costs.size();
  Require(phases>0,"Multirate graph requires phase costs");
  std::vector<bool> seen(blocks.size(),false);
  for (const auto &b:blocks) {
    Require(b.gid>=0 && static_cast<std::size_t>(b.gid)<blocks.size() && !seen[b.gid],
            "Multirate graph GIDs must be a permutation of [0,N)");
    seen[b.gid]=true;
    Require(b.time_level>=0 && b.time_level<63 && KindValid(b.kernel) && b.work_units>0,
            "Invalid multirate block cost metadata");
    Require(b.phase_costs.size()==phases,"Inconsistent multirate phase cost dimensions");
    for (double c:b.phase_costs) Require(std::isfinite(c) && c>=0,"Invalid multirate phase cost");
    Require(std::isfinite(Cost(b)) && Cost(b)>0,"Multirate block requires positive finite work");
  }
  for (const auto &e:edges) {
    Require(e.a>=0 && e.b>=0 && static_cast<std::size_t>(e.a)<blocks.size() &&
            static_cast<std::size_t>(e.b)<blocks.size(),"Invalid multirate coupling endpoint");
    Multiply(e.bytes,e.exchanges);
  }
}

std::uint64_t WorkGraph::Fingerprint() const {
  Validate();std::uint64_t hash=1469598103934665603ULL;
  auto add=[&](std::uint64_t x) {
    for (int i=0;i<8;++i) {hash^=(x>>(8*i))&255;hash*=1099511628211ULL;}
  };
  add(blocks.size());const auto indexed=ByGid(*this);
  for (const auto *b:indexed) {
    add(b->gid);add(b->time_level);add(static_cast<int>(b->kernel));
    add(b->work_units);add(b->memory_bytes);
  }
  std::vector<std::tuple<int,int,std::uint64_t,std::uint64_t>> sorted;
  for (const auto &e:edges) sorted.emplace_back(std::min(e.a,e.b),std::max(e.a,e.b),e.bytes,e.exchanges);
  std::sort(sorted.begin(),sorted.end());add(sorted.size());
  for (const auto &e:sorted) {add(std::get<0>(e));add(std::get<1>(e));add(std::get<2>(e));add(std::get<3>(e));}
  return hash;  // Measured timing values deliberately do not identify the mesh.
}

bool MetisAvailable() {
#if defined(ATHENA_MULTIRATE_METIS) && ATHENA_MULTIRATE_METIS
  return true;
#else
  return false;
#endif
}

PartitionScore ScorePartition(const WorkGraph &g,int ranks,const std::vector<int> &owners,
    const BalanceOptions &o,const BatchCostModel &profile,const std::vector<ExecutionTask> &tasks) {
  g.Validate();OptionsValid(o);OwnersValid(g.blocks.size(),ranks,owners);
  PartitionScore score;score.rank_busy_seconds.assign(ranks,0);
  const auto indexed=ByGid(g);
  std::vector<std::uint64_t> memory(ranks,0);std::vector<int> population(ranks,0);
  for (const auto &b:g.blocks) {memory[owners[b.gid]]=Add(memory[owners[b.gid]],b.memory_bytes);++population[owners[b.gid]];}
  score.maximum_rank_bytes=*std::max_element(memory.begin(),memory.end());
  if ((o.require_nonempty_ranks && std::find(population.begin(),population.end(),0)!=population.end()) ||
      (o.maximum_blocks_per_rank && *std::max_element(population.begin(),population.end())>o.maximum_blocks_per_rank) ||
      (o.memory_capacity_per_rank && score.maximum_rank_bytes>o.memory_capacity_per_rank)) {
    score.feasible=false;score.seconds=std::numeric_limits<double>::infinity();return score;
  }
  std::uint64_t cut_exchanges=0;
  for (const auto &e:g.edges) if (owners[e.a]!=owners[e.b]) {
    score.cut_bytes_per_interval=Add(score.cut_bytes_per_interval,Multiply(e.bytes,e.exchanges));
    cut_exchanges=Add(cut_exchanges,e.exchanges);
  }
  score.communication_seconds=cut_exchanges*o.network_latency_seconds+
                              score.cut_bytes_per_interval/o.network_bytes_per_second;
  if (tasks.empty()) {
    // Conservative phase-synchronous estimate when the caller has not supplied
    // a task DAG. This is explicitly not a prediction of asynchronous overlap.
    for (std::size_t phase=0;phase<g.blocks[0].phase_costs.size();++phase) {
      std::vector<double> cost(ranks,0);
      for (const auto *b:indexed) cost[owners[b->gid]]+=b->phase_costs[phase];
      score.compute_seconds+=*std::max_element(cost.begin(),cost.end());
      for (int rank=0;rank<ranks;++rank) score.rank_busy_seconds[rank]+=cost[rank];
    }
    score.seconds=score.compute_seconds+score.communication_seconds;return score;
  }
  Require(tasks.size()<=static_cast<std::size_t>(std::numeric_limits<int>::max()),"Too many multirate scoring tasks");
  const int ntasks=tasks.size();std::vector<int> indegree(ntasks,0),order;
  std::vector<std::vector<int>> children(ntasks);
  std::vector<bool> represented(g.blocks.size(),false);
  std::vector<std::uint64_t> incoming(ranks,0),outgoing(ranks,0);
  for (int i=0;i<ntasks;++i) {
    const auto &t=tasks[i];
    Require(t.block>=0 && static_cast<std::size_t>(t.block)<owners.size() &&
            KindValid(t.kernel) && t.work_units>0,"Invalid multirate scoring task");
    represented[t.block]=true;
    std::set<int> seen;
    for (const auto &d:t.dependencies) {
      Require(d.task>=0 && d.task<ntasks && d.task!=i && seen.insert(d.task).second,
              "Invalid or duplicate multirate task dependency");
      ++indegree[i];children[d.task].push_back(i);
    }
  }
  Require(std::find(represented.begin(),represented.end(),false)==represented.end(),
          "Multirate scoring DAG omits a block");
  // Validate all task block indices before consulting a predecessor's owner.
  for (const auto &t:tasks) for (const auto &d:t.dependencies) {
    int source=owners[tasks[d.task].block],target=owners[t.block];
    if (source!=target) {
      outgoing[source]=Add(outgoing[source],d.bytes);
      incoming[target]=Add(incoming[target],d.bytes);
    }
  }
  auto pending=indegree;std::set<int> ready;
  for (int i=0;i<ntasks;++i) if (!pending[i]) ready.insert(i);
  while (!ready.empty()) {
    int i=*ready.begin();ready.erase(ready.begin());order.push_back(i);
    for (int j:children[i]) if (!--pending[j]) ready.insert(j);
  }
  Require(static_cast<int>(order.size())==ntasks,"Cycle in multirate scoring DAG");
  std::vector<double> tail(ntasks,0),finish(ntasks,-1),rank_available(ranks,0);
  for (auto it=order.rbegin();it!=order.rend();++it) {
    int i=*it;double following=0;
    for (int j:children[i]) following=std::max(following,tail[j]);
    tail[i]=profile.Predict(tasks[i].kernel,tasks[i].work_units)+following;
  }
  auto available=[&](int i) {
    double time=rank_available[owners[tasks[i].block]];
    for (const auto &d:tasks[i].dependencies) {
      double delay=owners[tasks[i].block]==owners[tasks[d.task].block] ? 0 :
          o.network_latency_seconds+d.bytes/o.network_bytes_per_second;
      time=std::max(time,finish[d.task]+delay);
    }
    return time;
  };
  ready.clear();for (int i=0;i<ntasks;++i) if (!indegree[i]) ready.insert(i);
  int completed=0;
  while (completed<ntasks) {
    int first=-1;double start=std::numeric_limits<double>::infinity();
    for (int i:ready) {
      double t=available(i);
      if (t<start || (t==start && (first<0 || tail[i]>tail[first]))) {first=i;start=t;}
    }
    if (first<0) throw std::logic_error("Unready multirate scoring graph");
    const int rank=owners[tasks[first].block];std::vector<int> batch;std::uint64_t work=0;
    for (int i:ready) if (owners[tasks[i].block]==rank && tasks[i].phase==tasks[first].phase &&
        tasks[i].kernel==tasks[first].kernel && available(i)<=start) {
      batch.push_back(i);work=Add(work,tasks[i].work_units);
    }
    double duration=profile.Predict(tasks[first].kernel,work);
    rank_available[rank]=start+duration;score.rank_busy_seconds[rank]+=duration;
    for (int i:batch) {ready.erase(i);finish[i]=start+duration;++completed;}
    for (int i:batch) for (int j:children[i]) if (!--indegree[j]) ready.insert(j);
  }
  score.seconds=*std::max_element(rank_available.begin(),rank_available.end());
  // The event model overlaps independent transfers, but finite NIC throughput
  // remains a lower bound even when the compute DAG exposes abundant overlap.
  for (int rank=0;rank<ranks;++rank)
    score.seconds=std::max(score.seconds,std::max(incoming[rank],outgoing[rank])/
                                      o.network_bytes_per_second);
  score.compute_seconds=*std::max_element(score.rank_busy_seconds.begin(),score.rank_busy_seconds.end());
  return score;
}

PartitionResult Partition(const WorkGraph &g,int ranks,const BalanceOptions &o,
    const BatchCostModel &profile,const std::vector<ExecutionTask> &tasks,
    const std::vector<int> &current) {
  g.Validate();OptionsValid(o);
  Require(ranks>0 && static_cast<std::size_t>(ranks)<=g.blocks.size(),
          "Multirate partition requires at least one block per rank");
  if (!current.empty()) OwnersValid(g.blocks.size(),ranks,current);
  PartitionResult result;result.score.seconds=std::numeric_limits<double>::infinity();
  auto candidate=[&](std::string name,std::vector<int> owners) {
    if (o.require_nonempty_ranks) RepairEmpty(g,ranks,owners);
    if (o.spread_finest) {SpreadFinest(g,ranks,o,owners);name="spread-"+name;}
    auto score=ScorePartition(g,ranks,owners,o,profile,tasks);
    result.candidates.push_back({name,score});
    // Prefer the current partition on equal predictions: migration is not free.
    if (score.feasible && score.seconds<result.score.seconds) {
      result.owner_by_gid=std::move(owners);result.selected_candidate=name;result.score=std::move(score);
    }
  };
  if (!current.empty()) candidate("current",current);
  const auto scalar=ScalarPartition(g,ranks);candidate("scalar-reference",scalar);
  std::vector<int> primary=scalar;
  if (o.backend==Backend::metis) {
    primary=MetisPartition(g,ranks,o);result.used_metis=true;candidate("metis-stage-vector",primary);
  }
  if (o.consider_sparse_compaction) {
    const auto blocks=ByGid(g);
    std::map<int,std::vector<int>> levels;
    for (const auto &b:g.blocks) levels[b.time_level].push_back(b.gid);
    for (auto &level:levels) std::sort(level.second.begin(),level.second.end());
    for (const auto &level:levels) {
      if (level.second.size()>=static_cast<std::size_t>(ranks)*o.compact_batch_blocks) continue;
      std::vector<double> resident(ranks,0);
      for (int gid:level.second) resident[primary[gid]]+=Cost(*blocks[gid]);
      std::vector<int> rank_order(ranks);std::iota(rank_order.begin(),rank_order.end(),0);
      std::stable_sort(rank_order.begin(),rank_order.end(),[&](int a,int b) {return resident[a]>resident[b];});
      for (int count=1;count<ranks;count*=2) {
        auto owners=primary;std::vector<double> load(count,0);
        // Greedy locality-aware placement of the sparse level. Remaining levels
        // retain their METIS placement; the measured DAG score decides whether
        // the larger batches justify reducing available parallelism.
        for (int gid:level.second) {
          int best=0;double best_cost=std::numeric_limits<double>::infinity();
          for (int k=0;k<count;++k) {
            double cut=0;
            for (const auto &e:g.edges) if (e.a==gid || e.b==gid) {
              int other=e.a==gid?e.b:e.a;
              if (owners[other]!=rank_order[k]) cut+=e.exchanges*(o.network_latency_seconds+e.bytes/o.network_bytes_per_second);
            }
            if (load[k]+cut<best_cost) {best=k;best_cost=load[k]+cut;}
          }
          owners[gid]=rank_order[best];load[best]+=Cost(*blocks[gid]);
        }
        candidate("compact-level-"+std::to_string(level.first)+"-ranks-"+std::to_string(count),std::move(owners));
        if (count>ranks/2) break;  // also prevents signed overflow
      }
    }
  }
  if (result.owner_by_gid.empty()) throw std::runtime_error("No feasible multirate partition fits rank memory/population limits");
  return result;
}

RebalanceDecision ConsiderRebalance(std::uint64_t completed,std::uint64_t remaining,
    bool synchronized,double current,double proposed,double migration,const RebalancePolicy &p) {
  Require(std::isfinite(current) && current>0 && std::isfinite(proposed) && proposed>0 &&
          std::isfinite(migration) && migration>=0 && p.review_interval>0 &&
          std::isfinite(p.minimum_fractional_improvement) && p.minimum_fractional_improvement>=0 &&
          p.minimum_fractional_improvement<1 && std::isfinite(p.amortization_safety_factor) &&
          p.amortization_safety_factor>=1,"Invalid multirate rebalance forecast");
  if (!synchronized) return {false,0,"not synchronized"};
  if (completed<p.warmup_intervals) return {false,0,"warming up"};
  if ((completed-p.warmup_intervals)%p.review_interval) return {false,0,"outside review interval"};
  if (!remaining) return {false,0,"no remaining work"};
  const double saving=(current-proposed)*remaining;
  if (proposed>=(1-p.minimum_fractional_improvement)*current)
    return {false,saving-migration,"insufficient predicted improvement"};
  if (saving<=p.amortization_safety_factor*migration)
    return {false,saving-migration,"migration does not amortize"};
  return {true,saving-migration,"amortized synchronized improvement"};
}

std::string PartitionRecord::Serialize() const {
  Require(!owner_by_gid.empty(),"Empty multirate partition record");
  OwnersValid(owner_by_gid.size(),ranks,owner_by_gid);
  std::ostringstream out;out<<"multirate-partition-v1 "<<interval<<' '<<graph_fingerprint<<' '
      <<ranks<<' '<<owner_by_gid.size()<<'\n';
  for (std::size_t i=0;i<owner_by_gid.size();++i) out<<i<<' '<<owner_by_gid[i]<<'\n';
  return out.str();
}

PartitionRecord PartitionRecord::Deserialize(const std::string &text) {
  std::istringstream in(text);std::string magic;PartitionRecord r;std::size_t count;
  Require(static_cast<bool>(in>>magic>>r.interval>>r.graph_fingerprint>>r.ranks>>count) &&
          magic=="multirate-partition-v1" && r.ranks>0 && count>0 && count<=text.size(),
          "Invalid multirate partition record header");
  r.owner_by_gid.resize(count);
  for (std::size_t i=0;i<count;++i) {
    std::size_t gid;Require(static_cast<bool>(in>>gid>>r.owner_by_gid[i]) && gid==i,
                           "Truncated or unordered multirate partition record");
  }
  OwnersValid(count,r.ranks,r.owner_by_gid);NoTrailing(in);return r;
}

void PartitionRecord::ValidateFor(const WorkGraph &g,int expected_ranks) const {
  Require(ranks==expected_ranks,"Replay partition rank count changed");
  OwnersValid(g.blocks.size(),ranks,owner_by_gid);
  Require(graph_fingerprint==g.Fingerprint(),"Replay partition graph/time-level identity changed");
}

}  // namespace multirate::balance
