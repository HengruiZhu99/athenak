// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include "z4c_hyperboloidal/surface_remesh.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>

#include "z4c_hyperboloidal/cartesian_algebraic_chart.hpp"

namespace z4c_hyperboloidal {
namespace {
constexpr int payload_fields=Fields::count+8;
struct DeviceRow {int face,k,j,begin,end,hbegin,hend; bool copy;};
struct DeviceTerm {int ordinal; Real weight;};

void RequireAll(bool yes,const std::string &message) {
  int bad=yes ? 0 : 1;
#if MPI_PARALLEL_ENABLED
  int any=0;
  MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
  bad=any;
#endif
  if (bad)throw std::runtime_error("Scri remesh: "+message);
}

template<class T>
void Upload(DvceArray1D<T> &target,const std::vector<T> &source) {
  Kokkos::realloc(target,source.size());
  auto host=Kokkos::create_mirror_view(target);
  for (std::size_t i=0;i<source.size();++i)host(i)=source[i];
  Kokkos::deep_copy(target,host);
}

std::vector<int> OldFamily(const RemeshLayout &l,int gid) {
  const int old=l.new_to_old.at(gid);
  if (old<0 || old>=static_cast<int>(l.old_maps.size()))
    throw std::runtime_error("invalid old scri ancestor");
  if (l.old_level.at(old)<=l.new_level.at(gid))return {old};
  std::vector<int> family;
  for (std::size_t i=0;i<l.old_maps.size();++i)
    if (l.old_to_new.at(i)==gid)family.push_back(i);
  return family;
}

int ContainingOldPatch(const RemeshLayout &l,int gid,int j,int k) {
  const auto &map=l.new_maps.at(gid);
  const Real q[2]={map.lo[1]+(j+.5)*(map.hi[1]-map.lo[1])/l.cells,
                   map.lo[2]+(k+.5)*(map.hi[2]-map.lo[2])/l.cells};
  int owner=-1;
  // Native angular blocks are half-open internally. At the chart's upper edge
  // include the last block; sample centers themselves never lie on that edge.
  for (int old:OldFamily(l,gid)) {
    const auto &candidate=l.old_maps.at(old);
    if (!candidate.outer_surface || candidate.chart_id!=map.chart_id)continue;
    bool contains=true;
    for (int a=0;a<2;++a)contains=contains && q[a]>=candidate.lo[a+1] &&
        (q[a]<candidate.hi[a+1] ||
         (candidate.hi[a+1]==1 && q[a]==candidate.hi[a+1]));
    if (contains && (owner<0 || old<owner))owner=old;
  }
  if (owner<0)throw std::runtime_error("new scri point is outside its old outer family");
  return owner;
}

std::vector<SurfaceTransferTerm> Interpolate(const RemeshLayout &l,int gid,int j,int k) {
  const int old=ContainingOldPatch(l,gid,j,k),n=l.cells;
  const auto &target=l.new_maps.at(gid);
  Real x[3],local[3];
  const Real at[3]={1,(j+.5)/n,(k+.5)/n};
  target.Forward(at,x);
  l.old_maps.at(old).Inverse(x,local);
  int first[2];Real weight[2][6];
  for (int a=0;a<2;++a) {
    const Real t=n*local[a+1]-.5;
    first[a]=std::clamp(static_cast<int>(std::floor(t))-2,0,n-6);
    for (int p=0;p<6;++p) {
      weight[a][p]=1;
      for (int q=0;q<6;++q)if (q!=p)
        weight[a][p]*=(t-first[a]-q)/Real(p-q);
    }
  }
  std::vector<SurfaceTransferTerm> out;
  for (int v=0;v<6;++v)for (int u=0;u<6;++u)
    out.push_back({{old,first[1]+v,first[0]+u,0},weight[0][u]*weight[1][v]});
  return out;
}

int NearestIndex(Real t,int n) {
  const Real below=std::floor(t);
  return std::clamp(static_cast<int>(below+(t-below>Real(.5) ? 1 : 0)),0,n-1);
}

std::tuple<int,int,int> HistoryTarget(const RemeshLayout &l,int old,int j,int k,
                                     const std::vector<int> &family) {
  const auto &source=l.old_maps.at(old);
  const Real q[2]={source.lo[1]+(j+.5)*(source.hi[1]-source.lo[1])/l.cells,
                   source.lo[2]+(k+.5)*(source.hi[2]-source.lo[2])/l.cells};
  Real best=std::numeric_limits<Real>::max();
  std::tuple<int,int,int> target{-1,-1,-1};
  for (int gid:family) {
    const auto &map=l.new_maps.at(gid);
    if (!map.outer_surface || map.chart_id!=source.chart_id)continue;
    const int jj=NearestIndex(l.cells*(q[0]-map.lo[1])/(map.hi[1]-map.lo[1])-.5,l.cells);
    const int kk=NearestIndex(l.cells*(q[1]-map.lo[2])/(map.hi[2]-map.lo[2])-.5,l.cells);
    const Real u=map.lo[1]+(jj+.5)*(map.hi[1]-map.lo[1])/l.cells;
    const Real v=map.lo[2]+(kk+.5)*(map.hi[2]-map.lo[2])/l.cells;
    const Real du=(u-q[0])/(source.hi[1]-source.lo[1]);
    const Real dv=(v-q[1])/(source.hi[2]-source.lo[2]);
    const Real distance=du*du+dv*dv;
    const auto candidate=std::make_tuple(gid,kk,jj);
    if (distance<best || (distance==best && candidate<target)) {
      best=distance;target=candidate;
    }
  }
  if (std::get<0>(target)<0)throw std::runtime_error("old scri history has no outer replacement");
  return target;
}
}  // namespace

std::vector<SurfaceTransferRow> BuildSurfaceTransferRows(const RemeshLayout &l) {
  const int n=l.cells;
  if (n<6 || l.new_first<0 || l.new_count<0 ||
      l.new_first+l.new_count>static_cast<int>(l.new_maps.size()) ||
      l.old_maps.size()!=l.old_to_new.size() || l.old_maps.size()!=l.old_level.size() ||
      l.new_maps.size()!=l.new_to_old.size() || l.new_maps.size()!=l.new_level.size() ||
      l.new_maps.size()!=l.changed.size())throw std::runtime_error("invalid scri remesh layout");
  std::vector<SurfaceTransferRow> rows;
  std::vector<int> faces(l.new_count,-1);
  int nfaces=0;
  for (int m=0;m<l.new_count;++m) {
    const int gid=l.new_first+m;
    if (!l.new_maps[gid].outer_surface)continue;
    const int face=nfaces++;faces[m]=face;
    for (int k=0;k<n;++k)for (int j=0;j<n;++j) {
      SurfaceTransferRow row{gid,face,k,j,l.changed[gid]==0,{},{}};
      if (row.copy) {
        const int old=l.new_to_old[gid];
        if (old<0 || old>=static_cast<int>(l.old_maps.size()) ||
            !l.old_maps[old].outer_surface)
          throw std::runtime_error("unchanged scri patch lacks an old outer leaf");
        row.fields.push_back({{old,k,j,0},1});
        row.history.push_back({old,k,j,0});
      } else row.fields=Interpolate(l,gid,j,k);
      rows.push_back(std::move(row));
    }
  }
  // Walk old native identity order on every rank, but retain only packets needed
  // by this rank's new faces. The exchanged field data remain sparse and local.
  for (std::size_t old=0;old<l.old_maps.size();++old) {
    if (!l.old_maps[old].outer_surface)continue;
    const auto family=ReplacementFamily(l,old);
    if (family.empty())throw std::runtime_error("empty scri replacement family");
    if (family.size()==1 && !l.changed.at(family[0]))continue;
    for (int k=0;k<n;++k)for (int j=0;j<n;++j) {
      const auto [gid,kk,jj]=HistoryTarget(l,old,j,k,family);
      if (gid<l.new_first || gid>=l.new_first+l.new_count)continue;
      const int face=faces.at(gid-l.new_first);
      if (face<0)throw std::runtime_error("history target is not an owned outer patch");
      auto &row=rows.at((face*n+kk)*n+jj);
      if (row.copy)throw std::runtime_error("changed history reached an unchanged patch");
      row.history.push_back({static_cast<int>(old),k,j,0});
    }
  }
  return rows;
}

void SurfaceRemesh::Prepare(const Z4cHyperboloidal &p,const RemeshLayout &l) {
  ready_=false;
  std::vector<SurfaceTransferRow> rows;
  std::string reason;
  try {
    if (p.nblocks!=l.old_count || p.n[0]!=l.cells || p.n[1]!=l.cells ||
        p.n[2]!=l.cells || l.old_maps.size()!=l.old_owner.size())
      throw std::runtime_error("old module does not match native remesh metadata");
    rows=BuildSurfaceTransferRows(l);
  } catch (const std::exception &error) {reason=error.what();}
  RequireAll(reason.empty(),reason.empty() ? "plan failed on another rank" : reason);
  cells_=l.cells;blocks_=l.new_count;face_blocks_.clear();
  for (int m=0;m<l.new_count;++m)
    if (l.new_maps[l.new_first+m].outer_surface)face_blocks_.push_back(m);
  std::vector<bvals::DonorPoint> requests;
  std::vector<DeviceRow> device_rows;
  std::vector<DeviceTerm> terms;
  std::vector<int> history;
  for (const auto &row:rows) {
    DeviceRow r{row.face,row.k,row.j,static_cast<int>(terms.size()),0,
                static_cast<int>(history.size()),0,row.copy};
    for (const auto &term:row.fields) {
      terms.push_back({static_cast<int>(requests.size()),term.weight});
      requests.push_back(term.donor);
    }
    for (const auto &donor:row.history) {
      history.push_back(requests.size());requests.push_back(donor);
    }
    r.end=terms.size();r.hend=history.size();device_rows.push_back(r);
  }
  std::vector<int> fields(payload_fields);
  for (int f=0;f<payload_fields;++f)fields[f]=f;
  bvals::MappedExchange exchange;
  exchange.Rebuild(l.old_first,l.old_count,l.old_owner,l.mesh_sequence,requests,fields);
  remote_donors_=exchange.RemoteDonors();
  for (auto &term:terms)term.ordinal=exchange.RequestOrdinals().at(term.ordinal);
  for (auto &ordinal:history)ordinal=exchange.RequestOrdinals().at(ordinal);
  DvceArray1D<DeviceRow> dr;
  DvceArray1D<DeviceTerm> dt;
  DvceArray1D<int> dh;
  Upload(dr,device_rows);Upload(dt,terms);Upload(dh,history);
  DvceArray5D<Real> staging("old owned scri remesh",p.nblocks,payload_fields,cells_,cells_,1);
  Kokkos::deep_copy(staging,std::numeric_limits<Real>::quiet_NaN());
  const auto points=p.surface_points.d_view;
  const auto blocks=p.face_blocks.d_view;
  const auto old_state=p.s0;const auto old_records=p.surface_chi_records;
  Kokkos::parallel_for("pack scri remesh",Kokkos::RangePolicy<>(0,p.nsurface_points),
    KOKKOS_LAMBDA(int q) {
      const auto c=points(q);const int m=blocks(c.face);
      for (int f=0;f<Fields::count;++f)staging(m,f,c.k,c.j,0)=old_state(c.face,f,c.k,c.j);
      Real record[8];PackChiRecord(old_records(q),record);
      for (int a=0;a<8;++a)staging(m,Fields::count+a,c.k,c.j,0)=record[a];
    });
  exchange.StartReceives();exchange.PackAndSend(staging);
  while (exchange.Receive()!=TaskStatus::complete)std::this_thread::yield();
  while (exchange.ClearSend()!=TaskStatus::complete)std::this_thread::yield();
  exchange.ClearRecv();
  const auto values=exchange.Values();
  Kokkos::realloc(state_,face_blocks_.size(),Fields::count,cells_,cells_);
  Kokkos::realloc(records_,rows.size());
  const auto state=state_;const auto records=records_;
  int invalid=0;
  Kokkos::parallel_reduce("transfer owned scri",Kokkos::RangePolicy<>(0,rows.size()),
    KOKKOS_LAMBDA(int q,int &bad) {
      const auto row=dr(q);
      CartesianChartJets jets{};
      for (int f=0;f<Fields::count;++f) {
        Real value=0;
        if (row.copy)value=values(dt(row.begin).ordinal,f);
        else if (Fields::Independent(f))for (int t=row.begin;t<row.end;++t)
          value+=dt(t).weight*values(dt(t).ordinal,f);
        jets[f].f=value;
        if (!Kokkos::isfinite(value))++bad;
      }
      if (!row.copy && !TryExpandCartesianChart(jets))++bad;
      for (int f=0;f<Fields::count;++f) {
        state(row.face,f,row.k,row.j)=jets[f].f;
        if (!Kokkos::isfinite(jets[f].f))++bad;
      }
      ChiRecord record;
      for (int h=row.hbegin;h<row.hend;++h) {
        Real fields[8];
        for (int a=0;a<8;++a) {
          fields[a]=values(dh(h),Fields::count+a);
          if (!Kokkos::isfinite(fields[a]))++bad;
        }
        const auto packet=UnpackChiRecord(fields);
        if (h==row.hbegin)record=packet;
        else MergeChiRecord(record,packet);
      }
      records(q)=record;
    },invalid);
  RequireAll(invalid==0,"nonfinite transferred data or invalid algebraic chart");
  ready_=true;
}

void SurfaceRemesh::Install(Z4cHyperboloidal &p) const {
  bool valid=ready_ && p.nblocks==blocks_ && p.n[0]==cells_ &&
      p.n[1]==cells_ && p.n[2]==cells_ && p.nfaces==static_cast<int>(face_blocks_.size());
  if (valid)for (int face=0;face<p.nfaces;++face)
    valid=valid && p.face_blocks.h_view(face)==face_blocks_[face];
  RequireAll(valid,"new native face ownership does not match prepared transfer");
  Kokkos::deep_copy(p.s0,state_);
  Kokkos::deep_copy(p.surface_chi_records,records_);
}
}  // namespace z4c_hyperboloidal
