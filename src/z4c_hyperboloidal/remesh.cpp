// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include "z4c_hyperboloidal/remesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>

#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "z4c_hyperboloidal/cartesian_algebraic_chart.hpp"
#include "z4c_hyperboloidal/mapped_halo.hpp"
#include "z4c_hyperboloidal/surface_remesh.hpp"

namespace z4c_hyperboloidal {
namespace {
void RequireAll(bool valid,const std::string &message) {
  int failed=valid?0:1;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,&failed,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
#endif
  if(failed)throw std::runtime_error("Hyperboloidal remesh: "+message);
}
template<class T>
void CopyDevice(const std::vector<T> &source,DvceArray1D<T> &dest,const char *label) {
  dest=DvceArray1D<T>(std::string(label),source.size());
  auto host=Kokkos::create_mirror_view(dest);
  for(size_t i=0;i<source.size();++i)host(i)=source[i];
  Kokkos::deep_copy(dest,host);
}
int Lower(const chart_atlas::BlockMap &map,int axis) {
  return -map.overlap_sides[2*axis];
}
int Upper(const chart_atlas::BlockMap &map,int axis,int n) {
  return n+map.overlap_sides[2*axis+1];
}
int PointCount(const chart_atlas::BlockMap &map,int n) {
  int count=1;
  for(int a=0;a<3;++a)count*=Upper(map,a,n)-Lower(map,a);
  return count;
}
int PointOffset(const chart_atlas::BlockMap &map,int n,int i,int j,int k) {
  return ((k-Lower(map,2))*(Upper(map,1,n)-Lower(map,1))+j-Lower(map,1))
      *(Upper(map,0,n)-Lower(map,0))+i-Lower(map,0);
}
bool SameLocation(const LogicalLocation &a,const LogicalLocation &b) {
  return a.level==b.level&&a.chart_id==b.chart_id&&a.lx1==b.lx1
      &&a.lx2==b.lx2&&a.lx3==b.lx3;
}

// Preserve independently evolved chart/overlap lineage whenever the old
// replacement family covers the target. Only newly expanded coarse overlap
// beyond that coverage uses the ordinary physical-space donor selector.
std::vector<WeightedDonor> FamilyInterpolate(const RemeshLayout &layout,
    const NativeDonorSelector &selector,const std::vector<int> &family,
    int new_gid,int i,int j,int k,bool *used_global) {
  if(used_global)*used_global=false;
  const int n=layout.cells,index[3]={i,j,k};
  const auto &target=layout.new_maps[new_gid];
  Real chart[3],local[3],x[3];
  for(int a=0;a<3;++a) {
    local[a]=(index[a]+target.node_offset[a])/n;
    chart[a]=target.lo[a]+local[a]*(target.hi[a]-target.lo[a]);
  }
  target.Forward(local,x);
  int selected=-1;bool selected_nominal=false;Real best_margin=-1e300;
  Real selected_t[3]{};
  for(int gid:family) {
    const auto &map=layout.old_maps[gid];
    if(map.face!=target.face||map.layer!=target.layer)continue;
    Real t[3],margin=1e300;bool inside=true,nominal=true;
    for(int a=0;a<3;++a) {
      const Real q=n*(chart[a]-map.lo[a])/(map.hi[a]-map.lo[a]);
      t[a]=q-map.node_offset[a];
      inside=inside&&q>=Lower(map,a)-2e-12&&q<=Upper(map,a,n)+2e-12;
      nominal=nominal&&q>=0&&q<n;
      margin=std::min(margin,std::min(q-Lower(map,a),Upper(map,a,n)-q));
    }
    if(!inside)continue;
    if(selected<0||(nominal&&!selected_nominal)
        ||(nominal==selected_nominal&&(margin>best_margin
            ||(margin==best_margin&&gid<selected)))) {
      selected=gid;selected_nominal=nominal;best_margin=margin;
      for(int a=0;a<3;++a)selected_t[a]=t[a];
    }
  }
  if(selected<0) {
    if(used_global)*used_global=true;
    return selector.Interpolate(x,6);
  }
  const auto &map=layout.old_maps[selected];
  int first[3];Real weights[3][6];
  for(int a=0;a<3;++a) {
    first[a]=std::clamp(static_cast<int>(std::floor(selected_t[a]))-2,
        Lower(map,a),Upper(map,a,n)-6);
    for(int q=0;q<6;++q) {
      weights[a][q]=1;
      for(int other=0;other<6;++other)if(q!=other)
        weights[a][q]*=(selected_t[a]-(first[a]+other))/Real(q-other);
    }
  }
  std::vector<WeightedDonor> row;row.reserve(216);
  for(int kk=0;kk<6;++kk)for(int jj=0;jj<6;++jj)for(int ii=0;ii<6;++ii)
    row.push_back({{selected,layout.origin+first[2]+kk,layout.origin+first[1]+jj,
                  layout.origin+first[0]+ii},weights[0][ii]*weights[1][jj]*weights[2][kk]});
  return row;
}

// Each old ledger is one indivisible history packet. Pick a nearest logical
// sample in its replacement family, clamping overlap which no longer exists.
// Distances use the old leaf's chart-coordinate spans; ties use newGID/k/j/i.
bvals::DonorPoint HistoryTarget(const RemeshLayout &layout,int old_gid,
    const std::vector<int> &family,int i,int j,int k) {
  const auto &old=layout.old_maps[old_gid];const int source[3]={i,j,k};
  Real chart[3];
  for(int a=0;a<3;++a)chart[a]=old.lo[a]+(source[a]+old.node_offset[a])
      /layout.cells*(old.hi[a]-old.lo[a]);
  bvals::DonorPoint best{-1,0,0,0};Real best_distance=std::numeric_limits<Real>::max();
  for(int gid:family) {
    const auto &map=layout.new_maps[gid];int q[3];Real distance=0;
    for(int a=0;a<3;++a) {
      const Real t=layout.cells*(chart[a]-map.lo[a])/(map.hi[a]-map.lo[a])
          -map.node_offset[a];
      q[a]=std::clamp(static_cast<int>(std::ceil(t-.5)),Lower(map,a),
          Upper(map,a,layout.cells)-1);
      const Real at=map.lo[a]+(q[a]+map.node_offset[a])/layout.cells
          *(map.hi[a]-map.lo[a]);
      const Real d=(at-chart[a])/(old.hi[a]-old.lo[a]);distance+=d*d;
    }
    const bvals::DonorPoint candidate{gid,q[2],q[1],q[0]};
    if(best.gid<0||distance<best_distance||(distance==best_distance&&candidate<best)) {
      best=candidate;best_distance=distance;
    }
  }
  if(best.gid<0)throw std::runtime_error("old history has no replacement family");
  return best;
}
}  // namespace

std::vector<WeightedDonor> BuildBulkTransferRow(const RemeshLayout &layout,
    const NativeDonorSelector &selector,const std::vector<int> &old_family,
    int new_gid,int i,int j,int k,bool *used_global) {
  return FamilyInterpolate(layout,selector,old_family,new_gid,i,j,k,used_global);
}

std::vector<int> ReplacementFamily(const RemeshLayout &layout,int old_gid) {
  const int first=layout.old_to_new.at(old_gid);
  if(layout.new_level.at(first)<=layout.old_level.at(old_gid))return {first};
  std::vector<int> family;
  for(int gid=first;gid<int(layout.new_to_old.size())&&layout.new_to_old[gid]==old_gid;++gid)
    family.push_back(gid);
  if(family.empty())throw std::runtime_error("empty refined replacement family");
  return family;
}

struct RemeshState::Impl {
  struct Row {int target,m,k,j,i,begin,end;};  // target<0 means CC staging
  struct Entry {int ordinal;Real weight;};
  struct HistoryRow {int begin,end;};
  RemeshLayout layout;
  std::vector<BlockPoint> targets;
  std::vector<unsigned char> nominal_core;
  DvceArray5D<Real> nominal,coarse;
  DvceArray2D<Real> transferred;
  DvceArray1D<ChiRecord> records;
  SurfaceRemesh surface;
  bool ready=false;
};

RemeshState::RemeshState():impl_(std::make_unique<Impl>()) {}
RemeshState::~RemeshState()=default;
DvceArray5D<Real> &RemeshState::Nominal() {return impl_->nominal;}
DvceArray5D<Real> &RemeshState::Coarse() {return impl_->coarse;}

void RemeshState::Prepare(MeshBlockPack &pack,const Z4cHyperboloidal &p,
                          MeshRefinement &refinement) {
  auto &s=*impl_;auto &layout=s.layout;auto &mesh=*pack.pmesh;
  RequireAll(!s.ready,"attempted to reuse a completed remesh context");
  layout.cells=p.n[0];layout.origin=p.origin;layout.old_first=pack.gids;
  layout.old_count=p.nblocks;layout.new_first=refinement.new_gids_eachrank[global_variable::my_rank];
  layout.new_count=refinement.new_nmb_eachrank[global_variable::my_rank];
  layout.mesh_sequence=mesh.GetAMRLoadBalanceUpdateSeq();
  const int n=layout.cells,ng=mesh.mb_indcs.ng;
  const int new_total=std::accumulate(refinement.new_nmb_eachrank,
      refinement.new_nmb_eachrank+global_variable::nranks,0);
  layout.old_owner.assign(mesh.rank_eachmb,mesh.rank_eachmb+mesh.nmb_total);
  layout.new_owner.assign(refinement.new_rank_eachmb,refinement.new_rank_eachmb+new_total);
  layout.old_to_new.assign(refinement.oldtonew,refinement.oldtonew+mesh.nmb_total);
  layout.new_to_old.assign(refinement.newtoold,refinement.newtoold+new_total);
  layout.old_level.resize(mesh.nmb_total);layout.old_maps.reserve(mesh.nmb_total);
  for(int gid=0;gid<mesh.nmb_total;++gid) {
    layout.old_maps.push_back(mesh.atlas->GetBlockMap(mesh.lloc_eachmb[gid]));
    layout.old_level[gid]=mesh.lloc_eachmb[gid].level;
  }
  layout.new_level.resize(new_total);layout.changed.resize(new_total);
  for(int gid=0;gid<new_total;++gid) {
    layout.new_maps.push_back(mesh.atlas->GetBlockMap(refinement.new_lloc_eachmb[gid]));
    layout.new_level[gid]=refinement.new_lloc_eachmb[gid].level;
    layout.changed[gid]=!SameLocation(mesh.lloc_eachmb[layout.new_to_old[gid]],
                                     refinement.new_lloc_eachmb[gid]);
  }
  NativeDonorSelector selector(mesh,p.origin);
  std::vector<int> offsets(new_total+1);
  for(int gid=0;gid<new_total;++gid)offsets[gid+1]=offsets[gid]+PointCount(layout.new_maps[gid],n);
  const int local_offset=offsets[layout.new_first];
  const int local_points=offsets[layout.new_first+layout.new_count]-local_offset;
  std::vector<Impl::Row> rows;
  std::vector<Impl::Entry> entries;
  std::vector<bvals::DonorPoint> requests;
  std::map<bvals::DonorPoint,int> request_index;
  const auto add_row=[&](Impl::Row row,const std::vector<WeightedDonor> &donors) {
    if(donors.empty())throw std::runtime_error("empty physical transfer row");
    row.begin=entries.size();
    for(const auto &donor:donors) {
      if(!selector.IsEvolved(donor.point)||!std::isfinite(donor.weight))
        throw std::runtime_error("invalid physical transfer donor");
      auto insertion=request_index.emplace(donor.point,requests.size());
      if(insertion.second)requests.push_back(donor.point);
      entries.push_back({insertion.first->second,donor.weight});
    }
    row.end=entries.size();rows.push_back(row);
  };
  std::vector<std::vector<bvals::DonorPoint>> histories(local_points);
  std::string error;
  try {
    for(int gid=layout.new_first;gid<layout.new_first+layout.new_count;++gid) {
      const auto &map=layout.new_maps[gid];const int old=layout.new_to_old[gid];
      std::vector<int> family;
      if(layout.old_level[old]<=layout.new_level[gid])family.push_back(old);
      else for(int source=0;source<mesh.nmb_total;++source)
        if(layout.old_to_new[source]==gid)family.push_back(source);
      for(int k=Lower(map,2);k<Upper(map,2,n);++k)
        for(int j=Lower(map,1);j<Upper(map,1,n);++j)
          for(int i=Lower(map,0);i<Upper(map,0,n);++i) {
            const int point=s.targets.size();s.targets.push_back({gid-layout.new_first,
                k+p.origin,j+p.origin,i+p.origin});
            const bool core=layout.changed[gid]&&map.face<0
                &&i>=0&&i<n&&j>=0&&j<n&&k>=0&&k<n;
            s.nominal_core.push_back(core);
            if(core)continue;
            const auto donors=layout.changed[gid]?BuildBulkTransferRow(layout,selector,family,gid,i,j,k)
                :std::vector<WeightedDonor>{{{old,k+p.origin,j+p.origin,i+p.origin},1}};
            add_row({point,0,0,0,0,0,0},donors);
          }
    }
    // Native high-order prolongation consumes tensor corners as well as the
    // ordinary evolution's cross-shaped derivative halo. Fill its entire core
    // staging buffer independently, retaining actual raw evolved overlap.
    for(int m=0;m<p.nblocks;++m) {
      const int gid=pack.gids+m;const auto &map=layout.old_maps[gid];
      if(map.face>=0)continue;
      for(int k=0;k<n+2*ng;++k)for(int j=0;j<n+2*ng;++j)for(int i=0;i<n+2*ng;++i) {
        const bvals::DonorPoint direct{gid,k-ng+p.origin,j-ng+p.origin,i-ng+p.origin};
        std::vector<WeightedDonor> donors;
        if(selector.IsEvolved(direct))donors={{direct,1}};
        else {
          const Real q[3]={(i-ng+map.node_offset[0])/n,(j-ng+map.node_offset[1])/n,
                           (k-ng+map.node_offset[2])/n};
          Real x[3];map.Forward(q,x);donors=selector.Interpolate(x,6);
        }
        add_row({-1,m,k,j,i,0,0},donors);
      }
    }
    // Iterate old identities, not owner ranks, so additive history folds have
    // exactly the same order after any load balancing decision.
    for(int old=0;old<mesh.nmb_total;++old) {
      const auto family=ReplacementFamily(layout,old);const auto &map=layout.old_maps[old];
      for(int k=Lower(map,2);k<Upper(map,2,n);++k)
        for(int j=Lower(map,1);j<Upper(map,1,n);++j)
          for(int i=Lower(map,0);i<Upper(map,0,n);++i) {
            const auto target=layout.changed[family.front()]?HistoryTarget(layout,old,family,i,j,k)
                :bvals::DonorPoint{family.front(),k,j,i};
            if(layout.new_owner[target.gid]!=global_variable::my_rank)continue;
            const int point=offsets[target.gid]-local_offset
                +PointOffset(layout.new_maps[target.gid],n,target.i,target.j,target.k);
            histories.at(point).push_back({old,k+p.origin,j+p.origin,i+p.origin});
          }
    }
  } catch(const std::exception &e) {error=e.what();}
  RequireAll(error.empty(),error.empty()?"another rank rejected transfer geometry":error);
  RequireAll(int(s.targets.size())==local_points,"incomplete new local point list");
  const int capacity=std::max({mesh.nmb_maxperrank,p.nblocks,layout.new_count});
  s.nominal=DvceArray5D<Real>("hyperboloidal native CC AMR staging",capacity,Fields::count,
      n+2*ng,n+2*ng,n+2*ng);
  s.coarse=DvceArray5D<Real>("hyperboloidal native coarse AMR staging",capacity,Fields::count,
      n/2+2*ng,n/2+2*ng,n/2+2*ng);
  // Shell entries are deliberately unused by Install: native CC operators do
  // not represent the nodal radial chart. Keep discarded staging finite.
  Kokkos::deep_copy(s.nominal,0.0);Kokkos::deep_copy(s.coarse,0.0);
  s.transferred=DvceArray2D<Real>("hyperboloidal remeshed bulk",local_points,Fields::count);
  Kokkos::deep_copy(s.transferred,std::numeric_limits<Real>::quiet_NaN());
  s.records=DvceArray1D<ChiRecord>("hyperboloidal remeshed histories",local_points);

  bvals::MappedExchange physical;
  std::vector<int> fields(Fields::count);std::iota(fields.begin(),fields.end(),0);
  physical.Rebuild(layout.old_first,layout.old_count,layout.old_owner,layout.mesh_sequence,requests,fields);
  for(auto &entry:entries)entry.ordinal=physical.RequestOrdinals().at(entry.ordinal);
  DvceArray1D<Impl::Row> device_rows;DvceArray1D<Impl::Entry> device_entries;
  CopyDevice(rows,device_rows,"hyperboloidal remesh rows");
  CopyDevice(entries,device_entries,"hyperboloidal remesh weights");
  physical.StartReceives();physical.PackAndSend(p.u0);
  while(physical.Receive()!=TaskStatus::complete) {}
  const auto values=physical.Values();const auto nominal=s.nominal;
  const auto transferred=s.transferred;
  Kokkos::parallel_for("hyperboloidal remesh physical interpolation",Kokkos::RangePolicy<>(0,rows.size()),
    KOKKOS_LAMBDA(int row_id) {
      const auto row=device_rows(row_id);
      for(int f=0;f<Fields::count;++f) {
        Real value;
        if(row.end==row.begin+1&&device_entries(row.begin).weight==1)
          value=values(device_entries(row.begin).ordinal,f);
        else {value=0;for(int q=row.begin;q<row.end;++q)
          value+=device_entries(q).weight*values(device_entries(q).ordinal,f);}
        if(row.target<0)nominal(row.m,f,row.k,row.j,row.i)=value;
        else transferred(row.target,f)=value;
      }
    });
  DevExeSpace().fence("hyperboloidal remesh physical values consumed");
  while(physical.ClearSend()!=TaskStatus::complete) {}
  while(physical.ClearRecv()!=TaskStatus::complete) {}

  std::vector<Impl::HistoryRow> history_rows;
  std::vector<bvals::DonorPoint> history_requests;
  for(const auto &history:histories) {
    const int begin=history_requests.size();
    history_requests.insert(history_requests.end(),history.begin(),history.end());
    history_rows.push_back({begin,static_cast<int>(history_requests.size())});
  }
  DvceArray5D<Real> old_records("hyperboloidal old history packets",p.nblocks,8,
      p.storage[2],p.storage[1],p.storage[0]);
  Kokkos::deep_copy(old_records,std::numeric_limits<Real>::quiet_NaN());
  const auto cells=p.points.d_view;const auto records=p.chi_records;
  Kokkos::parallel_for("hyperboloidal pack history packets",Kokkos::RangePolicy<>(0,p.npoints),
    KOKKOS_LAMBDA(int point) {
      const auto c=cells(point);Real record[8];PackChiRecord(records(point),record);
      for(int f=0;f<8;++f)old_records(c.m,f,c.k,c.j,c.i)=record[f];
    });
  bvals::MappedExchange history;
  history.Rebuild(layout.old_first,layout.old_count,layout.old_owner,layout.mesh_sequence,
                  history_requests,{0,1,2,3,4,5,6,7});
  DvceArray1D<int> ordinals;DvceArray1D<Impl::HistoryRow> history_device_rows;
  CopyDevice(history.RequestOrdinals(),ordinals,"hyperboloidal history donor ordinals");
  CopyDevice(history_rows,history_device_rows,"hyperboloidal history fold rows");
  history.StartReceives();history.PackAndSend(old_records);
  while(history.Receive()!=TaskStatus::complete) {}
  const auto history_values=history.Values();const auto new_records=s.records;
  int invalid_history=0;
  Kokkos::parallel_reduce("hyperboloidal fold history packets",Kokkos::RangePolicy<>(0,local_points),
    KOKKOS_LAMBDA(int point,int &invalid) {
      const auto row=history_device_rows(point);ChiRecord record;
      for(int q=row.begin;q<row.end;++q) {
        Real values[8];for(int f=0;f<8;++f) {
          values[f]=history_values(ordinals(q),f);
          if(!Kokkos::isfinite(values[f]))++invalid;
        }
        const auto packet=UnpackChiRecord(values);
        if(q==row.begin)record=packet;else MergeChiRecord(record,packet);
      }
      new_records(point)=record;
    },invalid_history);
  DevExeSpace().fence("hyperboloidal remesh histories consumed");
  while(history.ClearSend()!=TaskStatus::complete) {}
  while(history.ClearRecv()!=TaskStatus::complete) {}
  RequireAll(invalid_history==0,"nonfinite inherited history packet");
  s.surface.Prepare(p,layout);
  s.ready=true;
}

void RemeshState::Install(Z4cHyperboloidal &p) const {
  const auto &s=*impl_;const auto &layout=s.layout;
  bool valid=s.ready&&p.nblocks==layout.new_count&&p.npoints==int(s.targets.size());
  if(valid)for(int point=0;point<p.npoints;++point) {
    const auto a=p.points.h_view(point),b=s.targets[point];
    valid=valid&&a.m==b.m&&a.k==b.k&&a.j==b.j&&a.i==b.i;
  }
  RequireAll(valid,"new pack does not match prepared transfer targets");
  DvceArray1D<unsigned char> changed,nominal_core;
  std::vector<unsigned char> changed_local(layout.changed.begin()+layout.new_first,
      layout.changed.begin()+layout.new_first+layout.new_count);
  CopyDevice(changed_local,changed,"hyperboloidal changed leaves");
  CopyDevice(s.nominal_core,nominal_core,"hyperboloidal native core transfer points");
  const auto points=p.points.d_view;const auto state=p.u0,nominal=s.nominal;
  const auto values=s.transferred;
  const auto geometry=p.geometry;
  const int shift=p.origin-p.nghost;int bad=0;
  Kokkos::parallel_reduce("hyperboloidal install remeshed bulk",Kokkos::RangePolicy<>(0,p.npoints),
    KOKKOS_LAMBDA(int point,int &invalid) {
      const auto c=points(point);CartesianChartJets jets{};
      for(int f=0;f<Fields::count;++f) {
        const Real value=nominal_core(point)?nominal(c.m,f,c.k-shift,c.j-shift,c.i-shift)
            :values(point,f);
        jets[f].f=changed(c.m)&&!Fields::Independent(f)?0:value;
      }
      if(changed(c.m)&&!TryExpandCartesianChart(jets)) {++invalid;return;}
      const auto &x=geometry(point).x;
      const Real r=Kokkos::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
      bool finite=jets[Fields::chi].f>0&&(1+r*r)/2+jets[Fields::lapse].f>0;
      for(int f=0;f<Fields::count;++f) {
        finite=finite&&Kokkos::isfinite(jets[f].f);state(c.m,f,c.k,c.j,c.i)=jets[f].f;
      }
      if(!finite)++invalid;
    },bad);
  RequireAll(bad==0,"invalid transferred bulk state");
  Kokkos::deep_copy(p.chi_records,s.records);
  s.surface.Install(p);
}

void Z4cHyperboloidal::PrepareRemesh(MeshRefinement &refinement) {
  RequireAll(initialized_&&!remesh_,"remesh requires one initialized accepted state");
  remesh_=std::make_unique<RemeshState>();
  remesh_->Prepare(*pmy_pack,*this,refinement);
}
void Z4cHyperboloidal::FinishRemesh(MeshRefinement &) {
  RequireAll(bool(remesh_),"remesh finish has no prepared state");
  ResetMeshStorage();remesh_->Install(*this);initialized_=true;remesh_.reset();
}
DvceArray5D<Real> &Z4cHyperboloidal::RemeshNominal() {
  if(!remesh_)throw std::logic_error("no native nominal transfer staging");
  return remesh_->Nominal();
}
DvceArray5D<Real> &Z4cHyperboloidal::RemeshCoarse() {
  if(!remesh_)throw std::logic_error("no native coarse transfer staging");
  return remesh_->Coarse();
}
}  // namespace z4c_hyperboloidal
