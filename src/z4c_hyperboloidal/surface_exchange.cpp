//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file surface_exchange.cpp
//! \brief Native block-owned scri derivatives, projection and normal compatibility.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "mesh/mesh.hpp"
#include "z4c_hyperboloidal/mapped_halo.hpp"
#include "z4c_hyperboloidal/point_rhs.hpp"
#include "z4c_hyperboloidal/surface_exchange.hpp"

namespace z4c_hyperboloidal {
namespace {
using J=mapped::Jet;
using Key=std::tuple<bool,int,int,int,int>;
struct HostTerm {bool boundary; bvals::DonorPoint point; J weight;};

template<class T>
void Upload(DvceArray1D<T> &view,const std::vector<T> &values) {
  Kokkos::realloc(view,values.size());
  auto host=Kokkos::create_mirror_view(view);
  for (size_t i=0;i<values.size();++i)host(i)=values[i];
  Kokkos::deep_copy(view,host);
}

std::vector<WeightedDonor> SurfaceDonors(const NativeDonorSelector &selector,
                                        const Real x[3]) {
  const int gid=selector.Owner(x),n=selector.Cells();
  if (gid<0 || !selector.Map(gid).outer_surface)
    throw std::runtime_error("Scri angular donor is not an outer native leaf");
  Real q[3]; selector.Map(gid).Inverse(x,q);
  int first[2];Real weights[2][6];
  for (int a=0;a<2;++a) {
    const Real t=n*q[a+1]-.5;
    first[a]=std::clamp(int(std::floor(t))-2,0,n-6);
    for (int i=0;i<6;++i) {
      weights[a][i]=1;
      for (int j=0;j<6;++j)if (i!=j)
        weights[a][i]*=(t-first[a]-j)/Real(i-j);
    }
  }
  std::vector<WeightedDonor> out;
  for (int k=0;k<6;++k)for (int j=0;j<6;++j)
    out.push_back({{gid,first[1]+k,first[0]+j,0},weights[0][j]*weights[1][k]});
  return out;
}

// Preserve insertion and summation order of the reference tensor rows. The map
// only finds a previously encountered donor; it does not sort the row itself.
void Add(std::vector<HostTerm> &terms,std::map<Key,size_t> &positions,
         bool boundary,const bvals::DonorPoint &point,const J &source,Real weight) {
  if (weight==0)return;
  const auto key=std::make_tuple(boundary,point.gid,point.k,point.j,point.i);
  const auto inserted=positions.emplace(key,terms.size());
  if (inserted.second)terms.push_back({boundary,point,{}});
  auto &term=terms[inserted.first->second].weight;
  term.f+=weight*source.f;
  for (int a=0;a<3;++a) {
    term.d[a]+=weight*source.d[a];
    for (int b=0;b<3;++b)term.h[a][b]+=weight*source.h[a][b];
  }
}

bool Nonzero(const J &w) {
  if (w.f!=0)return true;
  for (int a=0;a<3;++a) {
    if (w.d[a]!=0)return true;
    for (int b=0;b<3;++b)if (w.h[a][b]!=0)return true;
  }
  return false;
}

std::vector<HostTerm> BuildRow(const NativeDonorSelector &selector,int gid,
    int i,int j,int k,bool boundary_row) {
  const int n=selector.Cells(),origin=selector.Origin();
  const auto &map=selector.Map(gid);
  const Real q[3]={boundary_row ? 1 : (i+map.node_offset[0])/n,
                  (j+.5)/n,(k+.5)/n};
  const auto geometry=map.Geometry(q);
  const int first[3]={n-6,j-2,k-2},width[3]={7,5,5};
  Real nodes[3][7]{};
  for (int a=0;a<3;++a)for (int b=0;b<width[a];++b)
    nodes[a][b]=(a==0 && b==6) ? 1 : (first[a]+b+map.node_offset[a])/n;
  std::vector<HostTerm> terms;
  std::map<Key,size_t> positions;
  if (boundary_row) {
    const bvals::DonorPoint anchor{gid,k,j,0};
    terms.push_back({true,anchor,{}});
    positions.emplace(Key{true,gid,k,j,0},0);
  }
  for (int c=0;c<5;++c)for (int b=0;b<5;++b)for (int a=0;a<7;++a) {
    const auto w=EndpointWeight(geometry,q,nodes,width,a,b,c);
    if (!Nonzero(w))continue;
    const int aj=first[1]+b,ak=first[2]+c;const bool edge=a==6;
    if ((!boundary_row && !edge) || (aj>=0 && aj<n && ak>=0 && ak<n)) {
      const bvals::DonorPoint point=edge ? bvals::DonorPoint{gid,ak,aj,0}
          : bvals::DonorPoint{gid,origin+ak,origin+aj,origin+first[0]+a};
      Add(terms,positions,edge,point,w,1);
    } else {
      const Real local[3]={edge ? 1 : (first[0]+a+map.node_offset[0])/n,
                           (aj+.5)/n,(ak+.5)/n};
      Real x[3];map.Forward(local,x);
      const auto donors=edge ? SurfaceDonors(selector,x) : selector.Interpolate(x,6);
      for (const auto &donor:donors)Add(terms,positions,edge,donor.point,w,donor.weight);
    }
  }
  return terms;
}
}  // namespace

SurfaceExchange::SurfaceExchange(MeshBlockPack *pack,Z4cHyperboloidal *module)
    :pack_(pack),module_(module) {Rebuild();}

void SurfaceExchange::Rebuild() {
  auto &p=*module_;
  ready_=false;
  correction_jets_ready_=false;
  const int n=p.n[0];
  std::vector<int> fields;
  std::vector<bvals::DonorPoint> bulk_requests,edge_requests;
  std::vector<Row> rows,radial_rows;
  std::vector<Term> terms;
  std::vector<RadialTerm> radial_terms;
  std::vector<int> radial_index(p.npoints,-1);
  std::vector<int> radial_points;
  std::vector<Real> spacing(p.nfaces),gap(p.nfaces);
  int failed=0;std::string reason;
  try {
    NativeDonorSelector selector(*pack_->pmesh,p.origin);
  for (int f=0;f<Fields::count;++f) {
    fields_.index[f]=-1;
    if (Fields::Independent(f)) {fields_.index[f]=fields.size();fields.push_back(f);}
  }
  for (int face=0;face<p.nfaces;++face) {
    const int gid=pack_->gids+p.face_blocks.h_view(face);
    const auto &map=selector.Map(gid);
    spacing[face]=(map.outer_radius-map.inner_radius)*(map.hi[0]-map.lo[0])/n;
    gap[face]=1-map.node_offset[0];
    if (!(spacing[face]>0 && gap[face]>0))
      throw std::runtime_error("Invalid native scri radial spacing");
  }
  for (int point=0;point<p.nsurface_points;++point) {
    const auto c=p.surface_points.h_view(point);
    const int gid=pack_->gids+p.face_blocks.h_view(c.face);
    const auto &map=selector.Map(gid);
    const Real q[3]={1,(c.j+.5)/n,(c.k+.5)/n};
    const auto geometry=map.Geometry(q);const Real r=SphericalPolicy::Radius(geometry.x);
    Real inverse=0;
    for (int a=0;a<3;++a)inverse+=geometry.J[0][a]*geometry.x[a]/r;
    Row row{static_cast<int>(terms.size()),0,1/(inverse*n)};
    for (const auto &term:BuildRow(selector,gid,n,c.j,c.k,true)) {
      auto &requests=term.boundary ? edge_requests : bulk_requests;
      terms.push_back({static_cast<int>(requests.size()),term.boundary,term.weight});
      requests.push_back(term.point);
    }
    row.end=terms.size();rows.push_back(row);
  }
  for (int point=0;point<p.npoints;++point) {
    const auto c=p.points.h_view(point);const int face=p.block_faces.h_view(c.m);
    if (face<0 || c.i-p.origin<n-3)continue;
    const int gid=pack_->gids+c.m;
    Row row{static_cast<int>(radial_terms.size()),0,spacing[face]};
    for (const auto &term:BuildRow(selector,gid,c.i-p.origin,c.j-p.origin,
                                  c.k-p.origin,false)) {
      int ordinal=-1;
      if (term.boundary) {
        ordinal=edge_requests.size();edge_requests.push_back(term.point);
      }
      radial_terms.push_back({ordinal,term.boundary,
          {c.m,term.point.k,term.point.j,term.point.i},term.weight});
    }
    row.end=radial_terms.size();radial_index[point]=radial_rows.size();
    radial_rows.push_back(row);
    radial_points.push_back(point);
  }
  } catch (const std::exception &error) {
    failed=1;reason=error.what();
  }
  // A geometry failure can occur only on the rank owning a particular outer
  // patch. Report it collectively before any peer enters transport metadata.
  int any_failed=failed;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(&failed,&any_failed,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
#endif
  if (any_failed)throw std::runtime_error(failed ? "Scri plan: "+reason
      : "Scri plan construction failed on another rank");
  bulk_exchange_.Rebuild(pack_,bulk_requests,fields);
  surface_exchange_.Rebuild(pack_,edge_requests,fields);
  for (auto &term:terms)term.ordinal=(term.boundary ? surface_exchange_
      : bulk_exchange_).RequestOrdinals()[term.ordinal];
  for (auto &term:radial_terms)if (term.boundary)
    term.ordinal=surface_exchange_.RequestOrdinals()[term.ordinal];
  Upload(rows_,rows);Upload(terms_,terms);Upload(radial_rows_,radial_rows);
  Upload(radial_terms_,radial_terms);Upload(radial_index_,radial_index);
  Upload(radial_points_,radial_points);Upload(radial_fields_,fields);
  Kokkos::realloc(radial_jets_,radial_rows.size(),fields.size());
  Upload(face_spacing_,spacing);Upload(face_gap_,gap);
  Kokkos::realloc(sampled_,p.nsurface_points);
  Kokkos::realloc(normal_cases_,p.nsurface_points,4);
  Kokkos::realloc(metric_geometry_,p.nsurface_points);
  Kokkos::realloc(penrose_geometry_,p.nsurface_points);
  Kokkos::realloc(correction_,p.nfaces,3,n,n);
  Kokkos::realloc(correction_jets_,p.nfaces,3,n,n);
  Kokkos::realloc(surface_staging_,p.nblocks,Fields::count,n,n,1);
  Kokkos::deep_copy(surface_staging_,std::numeric_limits<Real>::quiet_NaN());
  Kokkos::deep_copy(correction_,std::numeric_limits<Real>::quiet_NaN());
  ready_=true;
}

void SurfaceExchange::CheckReady() const {
  if (!ready_)throw std::logic_error("Scri plan has not been built successfully");
}

TaskStatus SurfaceExchange::Project() {
  CheckReady();
  auto &p=*module_;const auto points=p.surface_points.d_view;
  const auto owners=p.face_blocks.d_view;const auto maps=p.surface_geometry;
  const auto state=p.u0;const auto edge=p.s0;const auto gap=face_gap_;
  const int n=p.n[0],origin=p.origin;
  int invalid=0;
  Kokkos::parallel_reduce("hyperboloidal local scri trace",
    Kokkos::RangePolicy<>(0,p.nsurface_points),KOKKOS_LAMBDA(int point,int &bad) {
      const auto c=points(point);const int m=owners(c.face);
      CartesianChartJets jets{};
      for (int f=0;f<Fields::count;++f)jets[f].f=edge(c.face,f,c.k,c.j);
      if (!TryExpandCartesianChart(jets)) {++bad;return;}
      const auto map=maps(point);const auto *x=map.x;
      const auto physical=SphericalPolicy::Physical(x,jets);
      const auto geometry=Geometry(physical.metric);
      if (!geometry.valid || !(jets[0].f>0)) {++bad;return;}
      Real Q=0;
      for (int a=0;a<3;++a)for (int b=0;b<3;++b)
        Q+=geometry.inverse[a][b]*x[a]*x[b];
      if (!(Q>0)) {++bad;return;}
      const Real speed=sqrt(jets[0].f*Q);jets[18].f=0;
      for (int a=0;a<3;++a) {
        jets[22+a].f=(1-speed)*x[a];jets[28+a].f=0;
        Real C=0;
        for (int k=0;k<5;++k) {
          Real weight=1;
          for (int l=0;l<5;++l)if (k!=l)weight*=(4+gap(c.face)-l)/Real(k-l);
          C+=weight*(state(m,25+a,origin+c.k,origin+c.j,origin+n-5+k)
                     -state(m,15+a,origin+c.k,origin+c.j,origin+n-5+k));
        }
        jets[25+a].f=jets[15+a].f+C;
      }
      for (int f=0;f<Fields::count;++f) {
        edge(c.face,f,c.k,c.j)=jets[f].f;
        if (!Kokkos::isfinite(jets[f].f))++bad;
      }
    },invalid);
  if (invalid)throw std::runtime_error("Invalid block-owned scri projection");
  return TaskStatus::complete;
}

TaskStatus SurfaceExchange::StartReceives() {
  CheckReady();
  bulk_exchange_.StartReceives();surface_exchange_.StartReceives();
  return TaskStatus::complete;
}
TaskStatus SurfaceExchange::PackAndSend() {
  CheckReady();
  auto &p=*module_;const auto points=p.surface_points.d_view;
  const auto owners=p.face_blocks.d_view;const auto edge=p.s0;
  const auto reference=p.surface_reference;const auto staging=surface_staging_;
  Kokkos::parallel_for("hyperboloidal local scri residual",
    Kokkos::RangePolicy<>(0,p.nsurface_points),KOKKOS_LAMBDA(int point) {
      const auto c=points(point);const int m=owners(c.face);
      for (int f=0;f<Fields::count;++f)
        staging(m,f,c.k,c.j,0)=Fields::Independent(f)
            ? edge(c.face,f,c.k,c.j)-reference(point)[f].f : 0;
    });
  bulk_exchange_.PackAndSend(p.residual);surface_exchange_.PackAndSend(staging);
  return TaskStatus::complete;
}
TaskStatus SurfaceExchange::Receive() {
  CheckReady();
  const auto bulk=bulk_exchange_.Receive(),edge=surface_exchange_.Receive();
  return bulk==TaskStatus::complete && edge==TaskStatus::complete
      ? TaskStatus::complete : TaskStatus::incomplete;
}
TaskStatus SurfaceExchange::Sample() {
  CheckReady();
  correction_jets_ready_=false;
  const auto rows=rows_;const auto terms=terms_;const auto output=sampled_;
  const auto bulk=bulk_exchange_.Values(),edge=surface_exchange_.Values();
  const auto slots=fields_;
  Kokkos::parallel_for("hyperboloidal scri residual jets",
    Kokkos::RangePolicy<>(0,module_->nsurface_points*Fields::count),KOKKOS_LAMBDA(int task) {
      const int point=task/Fields::count,f=task%Fields::count;
      J value;
      if (Fields::Independent(f)) {
        const auto row=rows(point);const auto first=terms(row.begin);
        const int slot=slots.index[f];
        const Real anchor=first.boundary ? edge(first.ordinal,slot) : bulk(first.ordinal,slot);
        value.f=anchor;
        for (int t=row.begin;t<row.end;++t) {
          const auto term=terms(t);
          const Real delta=(term.boundary ? edge(term.ordinal,slot)
              : bulk(term.ordinal,slot))-anchor;
          value.f+=term.weight.f*delta;
          for (int a=0;a<3;++a) {
            value.d[a]+=term.weight.d[a]*delta;
            for (int b=0;b<3;++b)value.h[a][b]+=term.weight.h[a][b]*delta;
          }
        }
      }
      output(point)[f]=value;
    });
  return TaskStatus::complete;
}

TaskStatus SurfaceExchange::Solve() {
  CheckReady();
  correction_jets_ready_=false;
  auto &p=*module_;const auto samples=sampled_;const auto maps=p.surface_geometry;
  const auto reference=p.surface_reference;const auto rows=rows_;
  const auto points=p.surface_points.d_view;const auto correction=correction_;
  const auto parameters=p.parameters;const auto source=p.boundary_source;
  // Separate scratch belongs to this plan, including diagnostic-only plans.
  // Every case overwrites its record before the elimination reads it.
  const auto cases=normal_cases_;
  const auto metric_geometry=metric_geometry_,penrose_geometry=penrose_geometry_;
  // Metric and chi jets are unchanged by all four normal cases and the final
  // shift correction. Refresh their geometry from this stage's samples once.
  Kokkos::parallel_for("hyperboloidal shared surface geometry",
    Kokkos::RangePolicy<>(0,p.nsurface_points),KOKKOS_LAMBDA(int point) {
      metric_geometry(point)=SpatialGeometry<Real>{};
      penrose_geometry(point)=SpatialGeometry<Real>{};
      auto jets=samples(point);const auto map=maps(point);
      if (!SphericalPolicy::Restore(map.x,jets,reference(point)))return;
      const auto u=SphericalPolicy::Physical(map.x,jets);
      if (!(u.chi.value>0 && u.alpha.value>0 && Kokkos::isfinite(u.chi.value)
            && Kokkos::isfinite(u.alpha.value)))return;
      metric_geometry(point)=Geometry(u.metric);
      penrose_geometry(point)=Geometry(PenroseMetric(u.metric,u.chi));
    });
  const Real penalty=p.penalty;int invalid=0;
  auto case_policy=Kokkos::RangePolicy<>(0,4*p.nsurface_points);
#ifdef KOKKOS_ENABLE_SYCL
  // Use the validated legacy normal-case group size; elimination is separate.
  case_policy.set_chunk_size(32);
#endif
  Kokkos::parallel_for("hyperboloidal independent normal cases",
    case_policy,KOKKOS_LAMBDA(int task) {
      const int point=task/4,which=task%4;
      cases(point,which)=NormalCase{};
      auto jets=samples(point);const auto map=maps(point);
      if (!SphericalPolicy::Restore(map.x,jets,reference(point)))return;
      if (which>0)
        for (int a=0;a<3;++a)jets[21+which].d[a]+=map.x[a];
      const auto result=EvaluateScriPoint(map,rows(point).spacing,jets,
                                        parameters,source,penalty,
                                        &metric_geometry(point),&penrose_geometry(point));
      if (!result.valid)return;
      for (int a=0;a<3;++a)cases(point,which).defect[a]=result.defect[a];
      cases(point,which).valid=true;
    });
  Kokkos::parallel_reduce("hyperboloidal local normal compatibility",
    Kokkos::RangePolicy<>(0,p.nsurface_points),KOKKOS_LAMBDA(int point,int &bad) {
      const auto c=points(point);
      for (int which=0;which<4;++which)
        if (!cases(point,which).valid) {++bad;return;}
      Real matrix[3][4]{};
      for (int a=0;a<3;++a)matrix[a][3]=-cases(point,0).defect[a];
      for (int b=0;b<3;++b)
        for (int a=0;a<3;++a)
          matrix[a][b]=cases(point,b+1).defect[a]-cases(point,0).defect[a];
      for (int col=0;col<3;++col) {
        int pivot=col;
        for (int row=col+1;row<3;++row)
          if (fabs(matrix[row][col])>fabs(matrix[pivot][col]))pivot=row;
        for (int k=col;k<4;++k) {
          const Real value=matrix[col][k];matrix[col][k]=matrix[pivot][k];
          matrix[pivot][k]=value;
        }
        if (!(fabs(matrix[col][col])>=1e-12)) {++bad;return;}
        const Real scale=matrix[col][col];
        for (int k=col;k<4;++k)matrix[col][k]/=scale;
        for (int row=0;row<3;++row)if (row!=col) {
          const Real multiple=matrix[row][col];
          for (int k=col;k<4;++k)matrix[row][k]-=multiple*matrix[col][k];
        }
      }
      for (int a=0;a<3;++a) {
        correction(c.face,a,c.k,c.j)=matrix[a][3];
        if (!Kokkos::isfinite(matrix[a][3]))++bad;
      }
    },invalid);
  if (invalid)throw std::runtime_error("Invalid block-owned normal compatibility solve");
  // The angular polynomial is shared by scri and the last two radial layers.
  // Its lift still uses each consumer's own map, spacing and endpoint gap.
  const auto kernel=BindKernel();const auto output=correction_jets_;
  auto correction_policy=Kokkos::RangePolicy<>(0,3*p.nsurface_points);
#ifdef KOKKOS_ENABLE_SYCL
  // Batch the independent angular jets across smaller SYCL workgroups.
  correction_policy.set_chunk_size(32);
#endif
  Kokkos::parallel_for("hyperboloidal shared normal correction jets",
    correction_policy,KOKKOS_LAMBDA(int task) {
      const int point=task/3,axis=task%3;const auto c=points(point);
      output(c.face,axis,c.k,c.j)=kernel.LogicalCorrection(c.face,c.j,c.k,axis);
    });
  correction_jets_ready_=true;
  return TaskStatus::complete;
}

TaskStatus SurfaceExchange::Evaluate() {
  CheckReady();
  auto &p=*module_;const auto samples=sampled_;const auto maps=p.surface_geometry;
  const auto reference=p.surface_reference;const auto rows=rows_;
  const auto points=p.surface_points.d_view;const auto rhs=p.s_rhs;
  const auto parameters=p.parameters;const auto source=p.boundary_source;
  const auto metric_geometry=metric_geometry_,penrose_geometry=penrose_geometry_;
  const Real penalty=p.penalty;auto kernel=BindKernel();
  if (correction_jets_ready_)kernel.correction_jets=correction_jets_;
  Real compatibility=0,clock=0;int invalid=0;
  Kokkos::parallel_reduce("hyperboloidal block-owned scri RHS",
    Kokkos::RangePolicy<>(0,p.nsurface_points),
    KOKKOS_LAMBDA(int point,int &bad,Real &defect,Real &clock_error) {
      auto jets=samples(point);const auto map=maps(point);const auto c=points(point);
      if (!SphericalPolicy::Restore(map.x,jets,reference(point))) {++bad;return;}
      for (int a=0;a<3;++a)
        jets[22+a]=jets[22+a]+kernel.Correction(c.face,c.j,c.k,map,a);
      const auto rate=EvaluateScriPoint(map,rows(point).spacing,jets,parameters,source,penalty,
                                        &metric_geometry(point),&penrose_geometry(point));
      if (!rate.valid) {++bad;return;}
      clock_error=fmax(clock_error,fabs(rate.clock));
      for (int a=0;a<3;++a)defect=fmax(defect,fabs(rate.defect[a]));
      for (int f=0;f<Fields::count;++f) {
        rhs(c.face,f,c.k,c.j)=rate.rate[f];
        if (!Kokkos::isfinite(rate.rate[f]))++bad;
      }
    },Kokkos::Sum<int>(invalid),Kokkos::Max<Real>(compatibility),Kokkos::Max<Real>(clock));
  if (invalid || compatibility>1e-7)
    throw std::runtime_error("Invalid block-owned scri RHS or normal compatibility");
  max_compatibility_=std::max(Real(0),compatibility);
  max_clock_=std::max(Real(0),clock);
  return TaskStatus::complete;
}

TaskStatus SurfaceExchange::ClearSend() {
  CheckReady();
  const auto bulk=bulk_exchange_.ClearSend(),edge=surface_exchange_.ClearSend();
  return bulk==TaskStatus::complete && edge==TaskStatus::complete
      ? TaskStatus::complete : TaskStatus::incomplete;
}
TaskStatus SurfaceExchange::ClearRecv() {
  CheckReady();
  const auto bulk=bulk_exchange_.ClearRecv(),edge=surface_exchange_.ClearRecv();
  return bulk==TaskStatus::complete && edge==TaskStatus::complete
      ? TaskStatus::complete : TaskStatus::incomplete;
}
SurfaceExchange::Kernel SurfaceExchange::BindKernel() const {
  CheckReady();
  auto &p=*module_;
  return {radial_index_,radial_rows_,radial_terms_,p.points.d_view,p.block_faces.d_view,
          correction_,face_spacing_,face_gap_,surface_exchange_.Values(),fields_,
          p.n[0],p.origin,{}};
}

SurfaceExchange::CachedKernel SurfaceExchange::PrepareRadialJets() {
  auto kernel=BindKernel();
  const auto points=radial_points_,fields=radial_fields_;
  const auto residual=module_->residual;
  const auto output=radial_jets_;
  const int count=fields.extent_int(0);
  using Index=std::int64_t;
  // Field-fast writes match LayoutRight(rows,field slots). Every field keeps
  // the direct row's anchor, term order and endpoint second-derivative mask.
  Kokkos::parallel_for("hyperboloidal endpoint radial fields",
    Kokkos::RangePolicy<Kokkos::IndexType<Index>>(0,Index(points.extent_int(0))*count),
    KOKKOS_LAMBDA(Index task) {
      const int row=static_cast<int>(task/count),slot=static_cast<int>(task%count);
      output(row,slot)=kernel.EndpointResidualJet(points(row),fields(slot),residual);
    });
  if (correction_jets_ready_)kernel.correction_jets=correction_jets_;
  return {kernel,radial_jets_};
}
}  // namespace z4c_hyperboloidal
