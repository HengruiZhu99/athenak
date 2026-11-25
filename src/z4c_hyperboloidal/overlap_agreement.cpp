// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include "z4c_hyperboloidal/overlap_agreement.hpp"
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <unordered_set>
#include <iostream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string>
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "utils/phase_timer.hpp"
#include "z4c_hyperboloidal/z4c_hyperboloidal.hpp"

namespace z4c_hyperboloidal {
namespace {
template<class T> void Copy(const std::vector<T> &v,DvceArray1D<T> &d,const char *name) {
  d=DvceArray1D<T>(std::string(name),v.size());auto h=Kokkos::create_mirror_view(d);
  for (std::size_t i=0;i<v.size();++i) h(i)=v[i];Kokkos::deep_copy(d,h);
}
Real Volume(const chart_atlas::BlockMap &map,int n,int i,int j,int k) {
  const Real q[3]={(i+map.node_offset[0])/n,(j+map.node_offset[1])/n,
                   (k+map.node_offset[2])/n};
  const auto p=map.Geometry(q);const auto &a=p.J;
  const Real det=a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])
      -a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])
      +a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]);
  const Real v=1/(std::abs(det)*n*n*n);
  if (!(v>0) || !std::isfinite(v)) throw std::runtime_error("invalid agreement volume");
  return v;
}
Real SmoothStep(Real t) {
  t=std::clamp(t,Real(0),Real(1));
  return t*t*t*(10+t*(-15+6*t));
}
}  // namespace

std::array<Real,6> VisitAgreementStencils(const Mesh &mesh,const NativeDonorSelector &selector,
    const AgreementParameters &parameters,const AgreementStencilVisitor &visit) {
  const int n=selector.Cells(),o=selector.Origin();
  // A conservative rectangular envelope includes the entire nominal cube and
  // the overlap extension available on every exterior leaf. Refinement can
  // make those extensions differ; taper to the smallest one on each face,
  // rather than introducing jumps in W when the donor leaf changes. Corrections
  // may still reach evolved donors outside this sampling envelope.
  std::array<Real,6> cube_extent;
  for (auto &e:cube_extent)e=std::numeric_limits<Real>::infinity();

  for(int gid=0;gid<mesh.nmb_total;++gid) {
    const auto &map=selector.Map(gid);if(map.face>=0)continue;
    for(int a=0;a<3;++a)for(int side=0;side<2;++side) {
      const int overlap=map.overlap_sides[2*a+side];if(!overlap)continue;
      const Real h=(map.hi[a]-map.lo[a])/n;
      const Real edge=(side?map.hi[a]:-map.lo[a])+(overlap-.5)*h;
      cube_extent[2*a+side]=std::min(cube_extent[2*a+side],edge);
    }
  }
  for(Real e:cube_extent)if(!(std::isfinite(e) && e>parameters.inner))
    throw std::runtime_error("full overlap lacks a bounded Cartesian support envelope");

  // Sample the union of evolved nodes in both representations. Each
  // quadrature contributes half its volume. Identity selections on both sets
  // prevent an under-resolved spherical sample grid from missing Cartesian
  // grid-scale disagreement (and conversely), without an auxiliary grid.
  for(int gid=0;gid<mesh.nmb_total;++gid) {
    const auto &map=selector.Map(gid);
    const int lo[3]={selector.Lower(gid,0),
                     selector.Lower(gid,1),
                     selector.Lower(gid,2)};
    const int hi[3]={selector.Upper(gid,0),
                     selector.Upper(gid,1),
                     selector.Upper(gid,2)};
    for(int k=lo[2];k<hi[2];++k)for(int j=lo[1];j<hi[1];++j)for(int i=lo[0];i<hi[0];++i) {
      const bvals::DonorPoint sample{gid,k+o,j+o,i+o};
      if(!selector.IsEvolved(sample))continue;
      const Real q[3]={(i+map.node_offset[0])/n,(j+map.node_offset[1])/n,
                       (k+map.node_offset[2])/n};Real x[3];map.Forward(q,x);
      const Real r=std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
      if(r<=parameters.inner)continue;
      Real taper=SmoothStep((r-parameters.inner)/parameters.taper_width);
      for(int a=0;a<3;++a) {
        taper*=SmoothStep((cube_extent[2*a]+x[a])/parameters.taper_width);
        taper*=SmoothStep((cube_extent[2*a+1]-x[a])/parameters.taper_width);
      }
      if(taper<=0)continue;

      std::vector<WeightedDonor> row;
      if(map.face<0) {
        row=selector.Spherical(x,6,true);
        for(const auto &d:row)
          if(selector.Map(d.point.gid).face<0)
            throw std::logic_error("full agreement spherical row contains Cartesian donors");
        for(auto &d:row)d.weight=-d.weight;
        row.push_back({sample,1});
      } else {
        row=selector.Cartesian(x);
        if(row.empty())throw std::runtime_error("agreement sample lacks six-point evolved Cartesian coverage at r="+std::to_string(r));
        row.push_back({sample,-1});
      }
      visit(row,sample,taper,r,map.face<0);
    }
  }
  return cube_extent;
}

void OverlapAgreementPlan::Rebuild(MeshBlockPack *pack,const Z4cHyperboloidal &p) {
  batches_.clear();
  Kokkos::Timer build_timer;
  PhaseTimer phases("agreement_plan",pack->pmesh->ncycle);
  const auto &mesh=*pack->pmesh;const int n=p.n[0],o=p.origin;
  kappa_=p.agreement.kappa;
  NativeDonorSelector selector(mesh,o,p.activity);
  dependency_edges_.clear();agreement_blocks_.clear();multirate_donors_.clear();
  // The containers below only find earlier occurrences. Every list they
  // feed is emitted in the same sorted order as before, and every sum is
  // accumulated in the order in which the samples are visited.
  std::unordered_set<std::uint64_t> graph;
  std::vector<char> group(mesh.nmb_total,0);
  if (p.UsesMultirate()) stencil_terms_.assign(mesh.nmb_total, 0);
  std::vector<Row> rows;std::vector<Entry> entries;
  std::vector<bvals::DonorPoint> requests;
  // Storage index of a point in its block, or -1 outside the block storage.
  const std::int64_t extent[3]={p.storage[0],p.storage[1],p.storage[2]};
  const std::size_t block_size=extent[0]*extent[1]*extent[2];
  const auto slot=[&](const bvals::DonorPoint &d)->std::int64_t {
    if(d.gid<0 || d.gid>=mesh.nmb_total || d.i<0 || d.i>=extent[0] || d.j<0
       || d.j>=extent[1] || d.k<0 || d.k>=extent[2])return -1;
    return (d.k*extent[1]+d.j)*extent[0]+d.i;
  };
  // Volume is a lookup-only cache; a valid volume is positive.
  std::vector<std::vector<Real>> volumes(mesh.nmb_total);
  auto volume=[&](const bvals::DonorPoint &d) {
    const auto s=slot(d);
    if(s<0)return Volume(selector.Map(d.gid),n,d.i-o,d.j-o,d.k-o);
    auto &block=volumes[d.gid];
    if(block.empty())block.assign(block_size,0);
    if(block[s]==0)block[s]=Volume(selector.Map(d.gid),n,d.i-o,d.j-o,d.k-o);
    return block[s];
  };
  // Local recipients of the transpose in order of first appearance.
  std::vector<std::vector<int>> recipient_slots(mesh.nmb_total);
  std::vector<bvals::DonorPoint> recipients;
  std::vector<std::vector<Transpose>> transpose;
  std::vector<Real> bounds;
  const auto recipient=[&](const bvals::DonorPoint &d) {
    const auto s=slot(d);
    if(s<0)throw std::logic_error("agreement recipient outside block storage");
    auto &block=recipient_slots[d.gid];
    if(block.empty())block.assign(block_size,-1);
    if(block[s]<0) {
      block[s]=static_cast<int>(recipients.size());
      recipients.push_back(d);transpose.emplace_back();bounds.push_back(0);
    }
    return block[s];
  };
  // A set kept as a list that is sorted and stripped of repeats as it grows.
  struct PointList {
    std::vector<bvals::DonorPoint> points;
    std::size_t limit=1<<16;
    void Compact() {
      std::sort(points.begin(),points.end());
      points.erase(std::unique(points.begin(),points.end(),
          [](const bvals::DonorPoint &a,const bvals::DonorPoint &b) {
            return !(a<b) && !(b<a);
          }),points.end());
      limit=std::max<std::size_t>(1<<16,2*points.size());
    }
    void Add(const bvals::DonorPoint &d) {
      points.push_back(d);
      if(points.size()>=limit)Compact();
    }
  };
  std::map<int,PointList> multirate_lists;
  int global_rows=0;std::size_t owner_rows=0,owner_entries=0;
  int cartesian_samples=0,spherical_samples=0;
  int balanced_samples=0;
  Real min_radius=std::numeric_limits<Real>::infinity(),max_radius=0;
  PointList owner_remote;
  std::vector<int> row_blocks;
  const auto add_row=[&](const std::vector<WeightedDonor> &row,
                         const bvals::DonorPoint &sample,Real w) {
    if (p.UsesMultirate()) {
      row_blocks.clear();
      // Attribute one forward stencil to its sample and the transpose to its
      // recipients. This counts mathematical work independently of MPI placement.
      stencil_terms_[sample.gid] += row.size();
      for (const auto &d : row) {
        ++stencil_terms_[d.point.gid];
        if (d.weight != 0 && (row_blocks.empty() || row_blocks.back() != d.point.gid))
          row_blocks.push_back(d.point.gid);
      }
      std::sort(row_blocks.begin(),row_blocks.end());
      row_blocks.erase(std::unique(row_blocks.begin(),row_blocks.end()),row_blocks.end());
      for(int a:row_blocks) {
        group[a]=1;
        for(int b:row_blocks)if(a<b)
          graph.insert((static_cast<std::uint64_t>(a)<<32)|static_cast<std::uint32_t>(b));
        if(mesh.rank_eachmb[a]!=global_variable::my_rank)continue;
        auto &list=multirate_lists[a];
        for(const auto &d:row)if(d.point.gid!=a && d.weight!=0)list.Add(d.point);
      }
    }
    Real norm=0;bool needed=false;
    for(const auto &d:row) {
      norm+=std::abs(d.weight);
      needed=needed || mesh.rank_eachmb[d.point.gid]==global_variable::my_rank;
    }
    ++global_rows;if(!needed)return;
    const bool owned=mesh.rank_eachmb[sample.gid]==global_variable::my_rank;
    if(owned) {
      ++owner_rows;owner_entries+=row.size();
      for(const auto &d:row)if(mesh.rank_eachmb[d.point.gid]!=global_variable::my_rank)
        owner_remote.Add(d.point);
    }
    const int id=rows.size();
    Row device_row{static_cast<int>(entries.size()),0,w,
                  mesh.rank_eachmb[sample.gid]==global_variable::my_rank};
    for(const auto &d:row) {
      if(!selector.IsEvolved(d.point))throw std::runtime_error("inactive agreement recipient");
      entries.push_back({static_cast<int>(requests.size()),d.weight});requests.push_back(d.point);
      if(mesh.rank_eachmb[d.point.gid]==global_variable::my_rank) {
        const Real coefficient=w*d.weight/volume(d.point);
        const int target=recipient(d.point);
        transpose[target].push_back({id,coefficient});
        bounds[target]+=std::abs(coefficient)*norm;
      }
    }
    device_row.end=entries.size();rows.push_back(device_row);
  };
  const auto cube_extent=VisitAgreementStencils(mesh,selector,p.agreement,
      [&](const auto &row,const bvals::DonorPoint &sample,Real taper,Real r,bool cartesian) {
      if(cartesian)++cartesian_samples;else ++spherical_samples;
      min_radius=std::min(min_radius,r);max_radius=std::max(max_radius,r);
      Real sample_weight=.5*volume(sample)*taper;

      // Bound each rank-one penalty's nonzero H-energy eigenvalue by kappa.
      // Coarse shell quadrature must not give a single sample an arbitrarily
      // large rate on a much smaller Cartesian donor volume. Use the same
      // positive W in discrepancy energy and both transpose contributions.
      Real inverse_mass=0;
      for(const auto &d:row)inverse_mass+=d.weight*d.weight/volume(d.point);
      if(!(inverse_mass>0 && std::isfinite(inverse_mass)))
        throw std::runtime_error("invalid full-overlap interpolation row mass");
      const Real balanced=taper/inverse_mass;
      if(balanced<sample_weight) {sample_weight=balanced;++balanced_samples;}

      add_row(row,sample,sample_weight);
  });
  phases.Mark("stencils");
  for(const auto key:graph)
    dependency_edges_.emplace_back(static_cast<int>(key>>32),
                                   static_cast<int>(key&0xffffffffu));
  std::sort(dependency_edges_.begin(),dependency_edges_.end());
  for(int gid=0;gid<mesh.nmb_total;++gid)if(group[gid])agreement_blocks_.push_back(gid);
  for(auto &list:multirate_lists) {
    if(list.second.points.empty())continue;
    list.second.Compact();
    auto &donors=multirate_donors_[list.first];
    for(const auto &d:list.second.points)donors.insert(donors.end(),d);
  }
  owner_remote.Compact();
  if(!global_rows)throw std::runtime_error("full overlap contains no samples");
  rate_bound_=0;
  for(const Real b:bounds)rate_bound_=std::max(rate_bound_,kappa_*b);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,&rate_bound_,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
#endif
  // Gershgorin/induced infinity bound for H^-1 A^T W A, similar to a symmetric
  // positive semidefinite matrix. 2.5 is inside RK4's negative-real interval.
  if(!p.UsesMultirate() && p.requested_dt*rate_bound_>2.5)
    throw std::invalid_argument("agreement timestep exceeds 2.5 / penalty rate bound: dt="
        +std::to_string(p.requested_dt)+", bound="+std::to_string(rate_bound_));
  std::vector<int> fields;for(int f=0;f<Fields::count;++f)
    if(Fields::Independent(f))fields.push_back(f);
  exchange_.Rebuild(pack,requests,fields);
  phases.Mark("exchange");
  for(auto &e:entries)e.ordinal=exchange_.RequestOrdinals()[e.ordinal];
  std::vector<Target> targets;std::vector<Transpose> adjacency;
  // Recipients in ascending donor order, as an ordered map would list them.
  std::vector<int> recipient_order(recipients.size());
  std::iota(recipient_order.begin(),recipient_order.end(),0);
  std::sort(recipient_order.begin(),recipient_order.end(),[&](int a,int b) {
    return recipients[a]<recipients[b];
  });
  for(const int index:recipient_order) {
    const auto &d=recipients[index];const auto &map=selector.Map(d.gid);
    const auto &list=transpose[index];
    const Real q[3]={(d.i-o+map.node_offset[0])/n,(d.j-o+map.node_offset[1])/n,
                     (d.k-o+map.node_offset[2])/n};Real x[3];map.Forward(q,x);
    const int begin=adjacency.size();adjacency.insert(adjacency.end(),list.begin(),list.end());
    targets.push_back({{pack->LocalId(d.gid),d.k,d.j,d.i},begin,static_cast<int>(adjacency.size()),
                       p.parameters.residual_weight.Value(x)});
  }
  // Schedule similarly sized gathers together; each target retains its entry order.
  std::stable_sort(targets.begin(),targets.end(),[](const Target &a,const Target &b) {
    return a.end-a.begin>b.end-b.begin;
  });
  // Split high-degree rows across work groups while keeping adjacent fields
  // together. Only completed partial sums cross the kernel boundary.
  std::vector<Segment> segments;
  std::vector<int> segment_offsets{0};
  constexpr int segment_size = 64;
  for (const auto &target : targets) {
    for (int e = target.begin; e < target.end; e += segment_size)
      segments.push_back({e, std::min(e + segment_size, target.end), target.point.m});
    segment_offsets.push_back(static_cast<int>(segments.size()));
  }
  Copy(segments, segments_, "agreement transpose segments");
  Copy(segment_offsets, segment_offsets_, "agreement segment offsets");
  partials_ = DvceArray2D<Real>(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "agreement partial corrections"),
      segments.size(), Fields::independent_count);
  Copy(rows,rows_,"agreement rows");Copy(entries,entries_,"agreement entries");
  Copy(targets,targets_,"agreement recipients");Copy(adjacency,transpose_,"agreement transpose");
  Copy(fields,fields_,"agreement fields");
  discrepancy_ = DvceArray2D<Real>("agreement discrepancies", rows.size(), 32);
  phases.Mark("transpose_device");
  phases.Report();
  // Initialization/rebuild receipt, never a stage collective or device fence.
  std::ostringstream receipt;receipt<<std::setprecision(17)
    <<"AGREEMENT_PLAN rank="<<global_variable::my_rank<<" active_points="<<p.npoints
    <<" global_samples="<<global_rows<<" local_rows="<<rows.size()
    <<" cartesian_samples="<<cartesian_samples<<" spherical_samples="<<spherical_samples
    <<" sample_rmin="<<min_radius<<" sample_rmax="<<max_radius
    <<" balanced_samples="<<balanced_samples
    <<" entries="<<entries.size()<<" recipients="<<targets.size()
    <<" receive_donors="<<exchange_.RemoteDonors()<<" send_donors="<<exchange_.SendDonors()
    <<" owner_once_rows="<<owner_rows<<" owner_once_entries="<<owner_entries
    <<" owner_once_remote_gather="<<owner_remote.points.size()
    <<" owner_once_remote_return="<<owner_remote.points.size()
    <<" kappa="<<kappa_<<" rate_bound="<<rate_bound_<<" dt="<<p.requested_dt
    <<" build_seconds="<<build_timer.seconds()<<'\n';

  receipt<<"FULL_OVERLAP_POLICY unmasked_activity=1 taper_width="<<p.agreement.taper_width
    <<" envelope=";
  for(Real e:cube_extent)receipt<<e<<',';
  receipt<<'\n';

  std::cout<<receipt.str()<<std::flush;
}

const OverlapAgreementPlan::BatchRows &OverlapAgreementPlan::RowsForBatch(
    const std::vector<int> &slots,const DvceArray1D<int> &active) {
  const auto found=batches_.find(slots);
  if(found!=batches_.end())return found->second;
  const auto targets=targets_;
  const auto transpose=transpose_;
  DvceArray1D<int> needed_rows("agreement active discrepancy mask",rows_.extent(0));
  DvceArray1D<int> needed_targets("agreement active transpose mask",targets.extent(0));
  Kokkos::parallel_for("mark active agreement rows",Kokkos::RangePolicy<>(0,targets.extent_int(0)),
    KOKKOS_LAMBDA(int t) {
      const auto target=targets(t);
      if(!active(target.point.m))return;
      needed_targets(t)=1;
      for(int e=target.begin;e<target.end;++e)Kokkos::atomic_exchange(&needed_rows(transpose(e).row),1);
    });
  auto indices=[](const DvceArray1D<int> &mask,const char *label) {
    const auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),mask);
    std::vector<int> selected;
    for(int i=0;i<mask.extent_int(0);++i)if(host(i))selected.push_back(i);
    DvceArray1D<int> list;
    Copy(selected,list,label);
    return list;
  };
  BatchRows batch{indices(needed_rows,"agreement batch rows"),
                  indices(needed_targets,"agreement batch targets")};
  return batches_.emplace(slots,std::move(batch)).first->second;
}

std::vector<int> OverlapAgreementPlan::DonorOrdinals(const BatchRows &batch) const {
  // Every discrepancy entry may be read, including those with a zero coefficient.
  DvceArray1D<int> mask("agreement batch donor mask",exchange_.UniqueDonors().size());
  const auto selected=batch.rows;
  const auto rows=rows_;
  const auto entries=entries_;
  Kokkos::parallel_for("mark agreement batch donors",Kokkos::RangePolicy<>(0,selected.extent_int(0)),
    KOKKOS_LAMBDA(int i) {
      const auto row=rows(selected(i));
      for(int j=row.begin;j<row.end;++j)Kokkos::atomic_exchange(&mask(entries(j).ordinal),1);
    });
  const auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),mask);
  std::vector<int> ordinals;
  for(int i=0;i<mask.extent_int(0);++i)if(host(i))ordinals.push_back(i);
  return ordinals;
}

void OverlapAgreementPlan::Evaluate(bvals::MappedExchange::ConstValues values,const BatchRows *batch) {
  const auto rows=rows_;const auto entries=entries_;const auto fields=fields_;
  if(!values.data())values=exchange_.Values();const auto d=discrepancy_;
  // Values/discrepancies store field as the contiguous index. One work item
  // per (row,field) coalesces donor loads across lanes while retaining the
  // exact entry order of every scalar dot product.
  const bool compact=batch!=nullptr;
  const auto selected=compact?batch->rows:DvceArray1D<int>{};
  const int count=compact?selected.extent_int(0):rows.extent_int(0);
  Kokkos::parallel_for("overlap agreement discrepancy",
    Kokkos::RangePolicy<>(0,count*Fields::independent_count),
    KOKKOS_LAMBDA(int q) {
      const int index=q/Fields::independent_count,f=q%Fields::independent_count;
      const int r=compact?selected(index):index;
      const auto row=rows(r);Real sum=0;
      for(int e=row.begin;e<row.end;++e) {
        const auto a=entries(e);Real value=values(a.ordinal,f);
        if(fields(f)>=Fields::D)value-=values(a.ordinal,f-10);
        sum+=a.coefficient*value;
      }
      d(r,f)=sum;
    });
}

void OverlapAgreementPlan::AddRHS(const Z4cHyperboloidal &p,const BatchRows *batch) const {
  const bool compact=batch!=nullptr;
  const auto selected=compact?batch->targets:DvceArray1D<int>{};
  const int count=compact?selected.extent_int(0):targets_.extent_int(0);
  const auto targets=targets_;const auto adjacency=transpose_;const auto d=discrepancy_;
  const auto fields=fields_;const auto rhs=p.u_rhs;const Real kappa=kappa_;
  const auto active=p.active_blocks;
  constexpr int lanes = 32;
  static_assert(Fields::independent_count <= lanes);
  if (!count) return;
  const auto segments = segments_;
  const auto offsets = segment_offsets_;
  const auto partials = partials_;
  Kokkos::parallel_for(
      "agreement transpose partial sums",
      Kokkos::RangePolicy<Kokkos::IndexType<std::int64_t>>(0, segments.extent(0) * lanes),
      KOKKOS_LAMBDA(std::int64_t q) {
        const int s = static_cast<int>(q / lanes), f = static_cast<int>(q % lanes);
        if (f >= Fields::independent_count) return;
        const auto segment = segments(s);
        if (active.extent(0) && !active(segment.block)) return;
        Real sum = 0;
        for (int e = segment.begin; e < segment.end; ++e) {
          const auto a = adjacency(e);
          sum -= kappa * a.coefficient * d(a.row, f);
        }
        partials(s, f) = sum;
      });
  Kokkos::parallel_for(
      "agreement transpose apply", Kokkos::RangePolicy<>(0, count * lanes),
      KOKKOS_LAMBDA(int q) {
        const int index = q / lanes, f = q % lanes;
        if (f >= Fields::independent_count) return;
        const int t = compact ? selected(index) : index;
        const auto target = targets(t);
        const auto c = target.point;
        if (active.extent(0) && !active(c.m)) return;
        const int field = fields(f);
        Real sum = 0, lambda = 0;
        for (int s = offsets(t); s < offsets(t + 1); ++s) {
          sum += partials(s, f);
          if (field >= Fields::D) lambda += partials(s, f - 10);
        }
        if (field == Fields::chi) sum *= target.chi_weight;
        rhs(c.m, field, c.k, c.j, c.i) += sum + lambda;
      });
}

std::array<Real,8> OverlapAgreementPlan::History() const {
  // Local sums; AthenaK history performs the standard rank reduction.
  std::array<Real,8> out{};const auto rows=rows_;const auto d=discrepancy_;const auto fields=fields_;
  for(int group=0;group<6;++group) {
    Real sum=0;
    Kokkos::parallel_reduce("agreement group norm",Kokkos::RangePolicy<>(0,rows.extent_int(0)),
      KOKKOS_LAMBDA(int r,Real &value) {
        if(!rows(r).owned)return;
        for(int f=0;f<Fields::independent_count;++f) {
          const int field=fields(f);
          const int g=field==0?0:field<7?1:field<15?2:field<18?3:field<25?4:5;
          if(g==group)value+=rows(r).weight*d(r,f)*d(r,f);
        }
      },sum);
    out[group]=sum;out[6]-=kappa_*sum;
  }
  Kokkos::parallel_reduce("agreement sample volume",Kokkos::RangePolicy<>(0,rows.extent_int(0)),
    KOKKOS_LAMBDA(int r,Real &value) {if(rows(r).owned)value+=rows(r).weight;},out[7]);
  return out;
}
}  // namespace z4c_hyperboloidal
