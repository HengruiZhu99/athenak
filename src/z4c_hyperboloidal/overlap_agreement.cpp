// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include "z4c_hyperboloidal/overlap_agreement.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <unordered_map>
#include <iostream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string>
#include "globals.hpp"
#include "mesh/mesh.hpp"
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

void OverlapAgreementPlan::Rebuild(MeshBlockPack *pack,const Z4cHyperboloidal &p) {
  Kokkos::Timer build_timer;
  const auto &mesh=*pack->pmesh;const int n=p.n[0],o=p.origin;
  kappa_=p.agreement.kappa;
  NativeDonorSelector selector(mesh,o,p.activity);
  // A conservative rectangular envelope includes the entire nominal cube and
  // the overlap extension available on every exterior leaf. Refinement can
  // make those extensions differ; taper to the smallest one on each face,
  // rather than introducing jumps in W when the donor leaf changes. Corrections
  // may still reach evolved donors outside this sampling envelope.
  Real cube_extent[6];
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
  for(Real e:cube_extent)if(!(std::isfinite(e) && e>p.agreement.inner))
    throw std::runtime_error("full overlap lacks a bounded Cartesian support envelope");

  std::vector<Row> rows;std::vector<Entry> entries;
  std::vector<bvals::DonorPoint> requests;
  std::map<bvals::DonorPoint,std::vector<Transpose>> transpose;
  // Volume is a lookup-only cache: hashing changes neither row order nor sums.
  const auto hash=[](const bvals::DonorPoint &d) {
    std::size_t h=(1469598103934665603ULL^static_cast<unsigned>(d.gid))*1099511628211ULL;
    for(int v:{d.k,d.j,d.i})h=(h^static_cast<unsigned>(v))*1099511628211ULL;
    return h;
  };
  const auto equal=[](const bvals::DonorPoint &a,const bvals::DonorPoint &b) {
    return a.gid==b.gid && a.k==b.k && a.j==b.j && a.i==b.i;
  };
  std::unordered_map<bvals::DonorPoint,Real,decltype(hash),decltype(equal)>
      volumes(0,hash,equal);
  std::map<bvals::DonorPoint,Real> bounds;
  auto volume=[&](const bvals::DonorPoint &d) {
    auto found=volumes.find(d);if(found!=volumes.end())return found->second;
    return volumes.emplace(d,Volume(selector.Map(d.gid),n,d.i-o,d.j-o,d.k-o)).first->second;
  };
  int global_rows=0;std::size_t owner_rows=0,owner_entries=0;
  int cartesian_samples=0,spherical_samples=0;
  int balanced_samples=0;
  Real min_radius=std::numeric_limits<Real>::infinity(),max_radius=0;
  std::set<bvals::DonorPoint> owner_remote;
  const auto add_row=[&](const std::vector<WeightedDonor> &row,
                         const bvals::DonorPoint &sample,Real w) {
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
        owner_remote.insert(d.point);
    }
    const int id=rows.size();
    Row device_row{static_cast<int>(entries.size()),0,w,
                  mesh.rank_eachmb[sample.gid]==global_variable::my_rank};
    for(const auto &d:row) {
      if(!selector.IsEvolved(d.point))throw std::runtime_error("inactive agreement recipient");
      entries.push_back({static_cast<int>(requests.size()),d.weight});requests.push_back(d.point);
      if(mesh.rank_eachmb[d.point.gid]==global_variable::my_rank) {
        const Real coefficient=w*d.weight/volume(d.point);
        transpose[d.point].push_back({id,coefficient});
        bounds[d.point]+=std::abs(coefficient)*norm;
      }
    }
    device_row.end=entries.size();rows.push_back(device_row);
  };
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
      if(r<=p.agreement.inner)continue;
      Real taper=SmoothStep((r-p.agreement.inner)/p.agreement.taper_width);
      for(int a=0;a<3;++a) {
        taper*=SmoothStep((cube_extent[2*a]+x[a])/p.agreement.taper_width);
        taper*=SmoothStep((cube_extent[2*a+1]-x[a])/p.agreement.taper_width);
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
      if(map.face<0)++cartesian_samples;else ++spherical_samples;
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
    }
  }
  if(!global_rows)throw std::runtime_error("full overlap contains no samples");
  rate_bound_=0;
  for(const auto &b:bounds)rate_bound_=std::max(rate_bound_,kappa_*b.second);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,&rate_bound_,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
#endif
  // Gershgorin/induced infinity bound for H^-1 A^T W A, similar to a symmetric
  // positive semidefinite matrix. 2.5 is inside RK4's negative-real interval.
  if(p.requested_dt*rate_bound_>2.5)
    throw std::invalid_argument("agreement timestep exceeds 2.5 / penalty rate bound: dt="
        +std::to_string(p.requested_dt)+", bound="+std::to_string(rate_bound_));
  std::vector<int> fields;for(int f=0;f<Fields::count;++f)
    if(Fields::Independent(f))fields.push_back(f);
  exchange_.Rebuild(pack,requests,fields);
  for(auto &e:entries)e.ordinal=exchange_.RequestOrdinals()[e.ordinal];
  std::vector<Target> targets;std::vector<Transpose> adjacency;
  for(const auto &item:transpose) {
    const auto &d=item.first;const auto &map=selector.Map(d.gid);
    const Real q[3]={(d.i-o+map.node_offset[0])/n,(d.j-o+map.node_offset[1])/n,
                     (d.k-o+map.node_offset[2])/n};Real x[3];map.Forward(q,x);
    const int begin=adjacency.size();adjacency.insert(adjacency.end(),item.second.begin(),item.second.end());
    targets.push_back({{d.gid-pack->gids,d.k,d.j,d.i},begin,static_cast<int>(adjacency.size()),
                       p.parameters.residual_weight.Value(x)});
  }
  // Schedule similarly sized gathers together; each target retains its entry order.
  std::stable_sort(targets.begin(),targets.end(),[](const Target &a,const Target &b) {
    return a.end-a.begin>b.end-b.begin;
  });
  Copy(rows,rows_,"agreement rows");Copy(entries,entries_,"agreement entries");
  Copy(targets,targets_,"agreement recipients");Copy(adjacency,transpose_,"agreement transpose");
  Copy(fields,fields_,"agreement fields");
  discrepancy_=DvceArray2D<Real>("agreement discrepancies",rows.size()+targets.size(),32);
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
    <<" owner_once_remote_gather="<<owner_remote.size()
    <<" owner_once_remote_return="<<owner_remote.size()
    <<" kappa="<<kappa_<<" rate_bound="<<rate_bound_<<" dt="<<p.requested_dt
    <<" build_seconds="<<build_timer.seconds()<<'\n';

  receipt<<"FULL_OVERLAP_POLICY unmasked_activity=1 taper_width="<<p.agreement.taper_width
    <<" envelope=";
  for(Real e:cube_extent)receipt<<e<<',';
  receipt<<'\n';

  std::cout<<receipt.str()<<std::flush;
}

void OverlapAgreementPlan::Evaluate() {
  const auto rows=rows_;const auto entries=entries_;const auto fields=fields_;
  const auto values=exchange_.Values();const auto d=discrepancy_;
  // Values/discrepancies store field as the contiguous index. One work item
  // per (row,field) coalesces donor loads across lanes while retaining the
  // exact entry order of every scalar dot product.
  Kokkos::parallel_for("overlap agreement discrepancy",
    Kokkos::RangePolicy<>(0,rows.extent_int(0)*Fields::independent_count),
    KOKKOS_LAMBDA(int q) {
      const int r=q/Fields::independent_count,f=q%Fields::independent_count;
      const auto row=rows(r);Real sum=0;
      for(int e=row.begin;e<row.end;++e) {
        const auto a=entries(e);Real value=values(a.ordinal,f);
        if(fields(f)>=Fields::D)value-=values(a.ordinal,f-10);
        sum+=a.coefficient*value;
      }
      d(r,f)=sum;
    });
}

void OverlapAgreementPlan::AddRHS(const Z4cHyperboloidal &p) const {
  const auto targets=targets_;const auto adjacency=transpose_;const auto d=discrepancy_;
  const auto fields=fields_;const auto rhs=p.u_rhs;const Real kappa=kappa_;
  const int scratch_begin=rows_.extent_int(0);
  constexpr int field_lanes=32;
  static_assert(Fields::independent_count<=field_lanes);
  // Preserve each scalar sum's adjacency order. Store completed sums after
  // the discrepancy rows so D can reuse the identical Lambda sum.
  Kokkos::parallel_for("overlap agreement transpose",
    Kokkos::RangePolicy<>(0,targets.extent_int(0)*field_lanes),
    KOKKOS_LAMBDA(int q) {
      const int t=q/field_lanes,f=q%field_lanes;
      if(f>=Fields::independent_count)return;
      const auto target=targets(t);Real correction=0;
      #pragma unroll 4
      for(int e=target.begin;e<target.end;++e) {
        const auto a=adjacency(e);
        correction-=kappa*a.coefficient*d(a.row,f);
      }
      d(scratch_begin+t,f)=correction;
    });
  Kokkos::parallel_for("overlap agreement apply",
    Kokkos::RangePolicy<>(0,targets.extent_int(0)*field_lanes),
    KOKKOS_LAMBDA(int q) {
      const int t=q/field_lanes,f=q%field_lanes;
      if(f>=Fields::independent_count)return;
      const auto target=targets(t);const auto c=target.point;const int field=fields(f);
      Real correction=d(scratch_begin+t,f);
      if(field==Fields::chi)correction*=target.chi_weight;
      if(field>=Fields::D)correction+=d(scratch_begin+t,f-10);
      rhs(c.m,field,c.k,c.j,c.i)+=correction;
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
