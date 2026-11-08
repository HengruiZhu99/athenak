// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include "z4c_hyperboloidal/horizon_sampling.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "z4c_hyperboloidal/horizon_geometry.hpp"
#include "z4c_hyperboloidal/mapped_halo.hpp"
#include "z4c_hyperboloidal/spherical_policy.hpp"
#include "z4c_hyperboloidal/spherical_reference_3d.hpp"
#include "z4c_hyperboloidal/z4c_hyperboloidal.hpp"

namespace z4c_hyperboloidal {
namespace {
using Node=bvals::DonorPoint;
using Values=std::array<Real,15>;
void RequireAll(bool condition,const char *message) {
  int invalid=condition?0:1;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,&invalid,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
#endif
  if (invalid)throw std::runtime_error(message);
}
bool Any(bool condition) {
  int any=condition?1:0;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,&any,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
#endif
  return any!=0;
}
struct Query {
  bool valid=false;
  int point=0,gid=-1,first[3]{};
  Real x[3]{},w[3][6]{},dw[3][6]{};
  mapped::Point map;
  std::array<Node,216> nodes;
};
}  // namespace

struct HorizonSampler::Impl {
  MeshBlockPack *pack;
  Z4cHyperboloidal *module;
  bvals::MappedExchange exchange;
  std::unique_ptr<NativeDonorSelector> selector;
  std::unique_ptr<SphericalReference3D> reference;
  std::vector<int> canonical_order,fields;
  std::map<Node,std::vector<WeightedDonor>> rows;
  std::map<Node,Values> values;
  Statistics statistics;
  std::uint64_t version=0;
  int sequence=-1;
  bool current=false;

  Impl(MeshBlockPack *p,Z4cHyperboloidal *m):pack(p),module(m) {
    for (int f=0;f<15;++f)if (Fields::Independent(f))fields.push_back(f);
  }
  void Refresh() {
    const int next=pack->pmesh->GetAMRLoadBalanceUpdateSeq();
    // A local tensor node can depend on another rank's evolved donor. Even
    // when this rank's state version is unchanged, that cached node is stale
    // if any owner invalidated its accepted state.
    const bool dirty=!current || version!=module->StateVersion() || sequence!=next;
    if (!Any(dirty))return;
    values.clear();statistics={};
    reference=std::make_unique<SphericalReference3D>(module->parameters);
    if (!selector || sequence!=next) {
      selector=std::make_unique<NativeDonorSelector>(*pack->pmesh,module->origin);
      rows.clear();
      const auto &mesh=*pack->pmesh;
      std::vector<int> order(mesh.nmb_total);std::iota(order.begin(),order.end(),0);
      // Preserve the predecessor's coalesced padded-index accumulation order
      // on N12 without retaining a second mesh or changing native ownership.
      std::sort(order.begin(),order.end(),[&](int a,int b) {
        const auto &x=selector->Map(a),&y=selector->Map(b);
        if ((x.face<0)!=(y.face<0))return x.face<0;
        if (x.face<0)return std::tie(mesh.lloc_eachmb[a].level,x.lo[2],x.lo[1],x.lo[0])
            <std::tie(mesh.lloc_eachmb[b].level,y.lo[2],y.lo[1],y.lo[0]);
        return std::tie(x.layer,x.face,x.lo[2],x.lo[1],x.lo[0])
            <std::tie(y.layer,y.face,y.lo[2],y.lo[1],y.lo[0]);
      });
      canonical_order.resize(order.size());
      for (std::size_t i=0;i<order.size();++i)canonical_order[order[i]]=i;
    }
    version=module->StateVersion();sequence=next;current=true;
  }
  const std::vector<WeightedDonor> &Row(const Node &node) {
    const auto found=rows.find(node);if (found!=rows.end())return found->second;
    std::vector<WeightedDonor> row;
    if (selector->IsEvolved(node))row.push_back({node,1});
    else {
      const int n=selector->Cells(),o=selector->Origin();
      const auto &map=selector->Map(node.gid);
      const Real q[3]={(node.i-o+map.node_offset[0])/n,
                       (node.j-o+map.node_offset[1])/n,
                       (node.k-o+map.node_offset[2])/n};
      Real x[3];map.Forward(q,x);
      const Real radius=SphericalPolicy::Radius(x);
      if (radius>=pack->pmesh->atlas->RadialKnots().back()*(1-2e-14))
        throw std::runtime_error("Horizon tensor stencil touches singular scri");
      std::map<Node,Real> coalesced;
      for (const auto &d:selector->Interpolate(x,6))coalesced[d.point]+=d.weight;
      for (const auto &[point,weight]:coalesced)if (weight!=0) {
        if (!selector->IsEvolved(point))
          throw std::runtime_error("Horizon tensor stencil has an unevolved donor");
        row.push_back({point,weight});
      }
      std::sort(row.begin(),row.end(),[&](const WeightedDonor &a,const WeightedDonor &b) {
        return std::tie(canonical_order[a.point.gid],a.point.k,a.point.j,a.point.i)
            <std::tie(canonical_order[b.point.gid],b.point.k,b.point.j,b.point.i);
      });
    }
    if (row.empty())throw std::runtime_error("Horizon tensor stencil has no evolved donors");
    return rows.emplace(node,std::move(row)).first->second;
  }
  Query BuildQuery(int point,int gid,const Real *x) {
    Query q;q.point=point;q.gid=gid;
    for (int a=0;a<3;++a)q.x[a]=x[a];
    const auto &map=selector->Map(gid);
    Real local[3];map.Inverse(x,local);q.map=map.Geometry(local);
    const int n=selector->Cells(),ng=pack->pmesh->mb_indcs.ng,o=selector->Origin();
    for (int a=0;a<3;++a) {
      const Real t=n*local[a]-map.node_offset[a];
      q.first[a]=std::clamp(int(std::floor(t))-2,-ng,n+ng-6);
      for (int i=0;i<6;++i) {
        q.w[a][i]=1;
        for (int j=0;j<6;++j)if (j!=i)
          q.w[a][i]*=(t-q.first[a]-j)/Real(i-j);
        for (int k=0;k<6;++k)if (k!=i) {
          Real term=1./(i-k);
          for (int j=0;j<6;++j)if (j!=i && j!=k)
            term*=(t-q.first[a]-j)/Real(i-j);
          q.dw[a][i]+=term;
        }
      }
    }
    int index=0;
    for (int k=0;k<6;++k)for (int j=0;j<6;++j)for (int i=0;i<6;++i) {
      const Node node{gid,o+q.first[2]+k,o+q.first[1]+j,o+q.first[0]+i};
      Row(node);q.nodes[index++]=node;
    }
    q.valid=true;return q;
  }
  z4c::HorizonGeometry<Real> Evaluate(const Query &q) {
    CartesianChartJets jets{};int index=0;
    for (int k=0;k<6;++k)for (int j=0;j<6;++j)for (int i=0;i<6;++i) {
      const auto &value=values.at(q.nodes[index++]);
      const Real weight=q.w[0][i]*q.w[1][j]*q.w[2][k];
      const Real logical[3]={q.dw[0][i]*q.w[1][j]*q.w[2][k],
                            q.w[0][i]*q.dw[1][j]*q.w[2][k],
                            q.w[0][i]*q.w[1][j]*q.dw[2][k]};
      Real derivative[3]{};
      for (int a=0;a<3;++a)for (int A=0;A<3;++A)
        derivative[a]+=q.map.J[A][a]*selector->Cells()*logical[A];
      for (int f:fields) {
        jets[f].f+=weight*value[f];
        for (int a=0;a<3;++a)jets[f].d[a]+=derivative[a]*value[f];
      }
    }
    if (!SphericalPolicy::Restore(q.x,jets,reference->At(q.x)))return {};
    return PhysicalHorizonGeometry(SphericalPolicy::Physical(q.x,jets),
                                   module->parameters.profile.At(q.x));
  }
};

HorizonSampler::HorizonSampler(MeshBlockPack *pack,Z4cHyperboloidal *module)
    :impl_(std::make_unique<Impl>(pack,module)) {
  RequireAll(pack && module,"Horizon sampler requires a native module and pack");
}
HorizonSampler::~HorizonSampler()=default;
void HorizonSampler::Invalidate() {impl_->current=false;}
const HorizonSampler::Statistics &HorizonSampler::GetStatistics() const {
  return impl_->statistics;
}

void HorizonSampler::SampleBatch(int count,const Real *positions,
    z4c::HorizonGeometry<Real> *geometry,int *owners) {
  auto &s=*impl_;
  RequireAll(count>=0 && (count==0 || (positions && geometry && owners))
      && s.module->IsInitialized()
      && s.module->residual.extent_int(0)>=s.module->nblocks
      && s.module->residual.extent_int(1)>=15
      && s.module->residual.extent_int(2)>=s.module->storage[2]
      && s.module->residual.extent_int(3)>=s.module->storage[1]
      && s.module->residual.extent_int(4)>=s.module->storage[0],
      "Invalid native horizon sample batch or residual storage");
  bool refreshed=true;
  try {s.Refresh();}catch (const std::exception &) {refreshed=false;}
  RequireAll(refreshed,"Cannot refresh native horizon sampling metadata");
  ++s.statistics.batches;
  std::vector<Query> queries;queries.reserve(count);
  std::set<Node> missing;
  for (int p=0;p<count;++p) {
    geometry[p]={};owners[p]=0;
    const Real *x=positions+3*p;
    if (!std::isfinite(x[0]) || !std::isfinite(x[1]) || !std::isfinite(x[2]))continue;
    int gid=-1;
    try {gid=s.selector->Owner(x);}
    catch (const std::exception &) {
      // Missing topology must not strand peers at the later sparse collectives.
      // No owner is reported, so FastFlow rejects the incomplete trial surface.
      continue;
    }
    if (gid<0 || s.pack->pmesh->rank_eachmb[gid]!=global_variable::my_rank)continue;
    owners[p]=1;++s.statistics.owned_queries;
    try {
      auto q=s.BuildQuery(p,gid,x);
      for (const auto &node:q.nodes)if (s.values.find(node)==s.values.end())missing.insert(node);
      queries.push_back(std::move(q));
    } catch (const std::exception &) {
      // An owned invalid sample remains owner=1, valid=false. FastFlow makes
      // its complete/unique-surface decision collectively after this callback.
    }
  }
  if (Any(!missing.empty())) {
    std::vector<Node> requests;
    for (const auto &node:missing)for (const auto &d:s.rows.at(node))requests.push_back(d.point);
    s.exchange.Rebuild(s.pack,requests,s.fields);
    s.exchange.StartReceives();s.exchange.PackAndSend(s.module->residual);
    TaskList tasks;TaskID none(0);
    auto received=tasks.AddTask([&](Driver *,int) {return s.exchange.Receive();},none);
    tasks.AddTask([&](Driver *,int) {return s.exchange.ClearSend();},none);
    tasks.AddTask([&](Driver *,int) {return s.exchange.ClearRecv();},received);
    while (tasks.DoAvailable(nullptr,0)!=TaskListStatus::complete) {}
    const auto fetched=Kokkos::create_mirror_view_and_copy(HostMemSpace(),s.exchange.Values());
    const auto &ordinal=s.exchange.RequestOrdinals();std::size_t request=0;
    for (const auto &node:missing) {
      const auto &row=s.rows.at(node);Values value{};
      const std::size_t begin=request;
      if (s.selector->IsEvolved(node)) {
        for (std::size_t f=0;f<s.fields.size();++f)value[s.fields[f]]=fetched(ordinal[request],f);
        ++request;
      } else {
        for (std::size_t f=0;f<s.fields.size();++f)value[s.fields[f]]=fetched(ordinal[begin],f);
        for (const auto &d:row) {
          for (std::size_t f=0;f<s.fields.size();++f)
            value[s.fields[f]]+=d.weight*(fetched(ordinal[request],f)-fetched(ordinal[begin],f));
          ++request;
        }
        ++s.statistics.ghost_nodes;
      }
      s.values.emplace(node,value);++s.statistics.nodes;
    }
    ++s.statistics.exchanges;
    s.statistics.requested_donors+=s.exchange.UniqueDonors().size();
    s.statistics.remote_donors+=s.exchange.RemoteDonors();
  }
  for (const auto &q:queries) {
    try {geometry[q.point]=s.Evaluate(q);}
    catch (const std::exception &) {geometry[q.point]={};}
  }
}
}  // namespace z4c_hyperboloidal
