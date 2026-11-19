//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_hyperboloidal.cpp
//! \brief Native block storage and reference initialization for hyperboloidal Z4c.

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "mesh/mesh.hpp"
#include "z4c/compact_object_tracker.hpp"
#include "z4c_hyperboloidal/z4c_hyperboloidal.hpp"
#include "z4c_hyperboloidal/central_z4c_configuration.hpp"
#include "z4c_hyperboloidal/spherical_policy.hpp"
#include "z4c_hyperboloidal/mapped_halo.hpp"
#include "z4c_hyperboloidal/surface_exchange.hpp"
#include "z4c_hyperboloidal/dissipation_plan.hpp"
#include "z4c_hyperboloidal/remesh.hpp"
#include "z4c_hyperboloidal/diagnostics.hpp"
#include "z4c_hyperboloidal/multirate_evolution.hpp"

namespace z4c_hyperboloidal {

Z4cHyperboloidal::~Z4cHyperboloidal() = default;

Z4cHyperboloidal::Z4cHyperboloidal(MeshBlockPack *ppack, ParameterInput *pin)
    : pmy_pack(ppack), pinput_(pin) {
  auto *pm=ppack->pmesh;
  if (!pm->atlas || !pm->three_d || sizeof(Real)!=sizeof(double)) {
    throw std::invalid_argument("Hyperboloidal Z4c requires a 3D atlas and double precision");
  }
  const auto evolution=pin->GetString("time","evolution");
  if (evolution!="static") {
    if (evolution!="dynamic") {
      throw std::invalid_argument("Hyperboloidal evolution requires static or dynamic mode");
    }
    const auto method=pin->GetOrAddString("time","integrator","rk4_classical");
    multirate_enabled_=method=="mrgark4";
    if(multirate_enabled_)MultirateEvolution::ReadOptions(*pin);
    if (method!="rk4_classical" && !multirate_enabled_)
      throw std::invalid_argument("Hyperboloidal evolution requires rk4_classical or mrgark4");
  }
  split_bulk_jets_ = pin->GetOrAddBoolean("z4c_hyperboloidal", "split_bulk_jets",
                                          multirate_enabled_);
#ifdef KOKKOS_ENABLE_SYCL
  rhs_workgroup_size_ =
      pin->GetOrAddInteger("z4c_hyperboloidal", "rhs_workgroup_size",
                           multirate_enabled_ ? 16 : 128);
  const auto device = Kokkos::SYCL().sycl_queue().get_device();
  const auto limit = device.get_info<sycl::info::device::max_work_group_size>();
  if (rhs_workgroup_size_ < 1 || (rhs_workgroup_size_ & (rhs_workgroup_size_ - 1)) ||
      static_cast<std::size_t>(rhs_workgroup_size_) > limit)
    throw std::invalid_argument("rhs_workgroup_size must be a supported power of two");
#endif
  n[0]=pm->mb_indcs.nx1; n[1]=pm->mb_indcs.nx2; n[2]=pm->mb_indcs.nx3;
  nghost=pm->mb_indcs.ng;
  overlap=pin->GetOrAddInteger("mesh_atlas","evolved_overlap",0);
  if (nghost!=4 || n[0]!=n[1] || n[0]!=n[2] || n[0]<12 ||
      overlap<0 || overlap>n[0]/3 || n[0]%2!=0) {
    throw std::invalid_argument("Hyperboloidal stencil needs even cubic blocks N>=12, "
                                "four ghosts, and overlap<=N/3");
  }
  origin=nghost+overlap;
  for (int a=0;a<3;++a) storage[a]=n[a]+2*origin;
  nblocks=ppack->nmb_thispack;
  ko=pin->GetOrAddReal("z4c_hyperboloidal","ko",0.1);
  chi_floor=pin->GetOrAddReal("z4c_hyperboloidal","chi_floor",1e-10);
  penalty=pin->GetOrAddReal("z4c_hyperboloidal","scri_penalty",0.5);
  const auto &grid = pm->mesh_size;
  const Real coarse_dx = std::min({grid.dx1, grid.dx2, grid.dx3});
  int depth = 0;
  if (!multirate_enabled_ && !pin->DoesParameterExist("time", "dt")) {
    for (int gid = 0; gid < pm->nmb_total; ++gid) {
      const auto &leaf = pm->lloc_eachmb[gid];
      if (leaf.chart_id == 0) depth = std::max(depth, leaf.level - pm->root_level);
    }
  }
  requested_dt = pin->GetOrAddReal("time", "dt", std::ldexp(pm->cfl_no * coarse_dx,
                                                             -depth));
  if (!std::isfinite(ko) || ko<0 || !std::isfinite(chi_floor) || chi_floor<0 ||
      !std::isfinite(penalty) || penalty<0 || !std::isfinite(requested_dt) ||
      requested_dt<=0) throw std::invalid_argument("Invalid hyperboloidal numerical policy");
  dtnew=std::numeric_limits<Real>::max();
  CentralZ4cConfiguration config;
  config.inner_z4c_radius_M=pin->GetOrAddReal("z4c_hyperboloidal","inner_z4c_radius_M",20);
  config.compact_slicing_end=pin->GetOrAddReal("z4c_hyperboloidal","compact_slicing_end",.97);
  parameters=config.Resolve().gauge;
  const auto policy=pin->GetOrAddString("z4c_hyperboloidal","residual_weight","origin_r2");
  if(policy=="regular_c6") {
    auto &w=parameters.residual_weight;
    w.inner=pin->GetReal("z4c_hyperboloidal","weight_inner");
    w.outer=pin->GetReal("z4c_hyperboloidal","weight_outer");
    if(!std::isfinite(w.inner)||!std::isfinite(w.outer)||
       !(0<w.inner && w.inner<w.outer && w.outer<parameters.profile.flat_radius))
      throw std::invalid_argument("Residual weight requires 0<a<b<flat Omega radius");
  } else if(policy!="origin_r2") throw std::invalid_argument("Unknown residual weight policy");
  if(pin->GetString("problem","pgen_name")=="z4c_hyperboloidal_binary") {
    SetBinaryIdentity(pin->DoesParameterExist("problem", "binary_identity")
                      ? pin->GetString("problem", "binary_identity") : "prepare");
  }
  floor_inner_amr_chi=pin->GetOrAddBoolean("z4c_hyperboloidal","floor_inner_amr_chi",false);
  if(floor_inner_amr_chi && (!parameters.residual_weight.Regular() || !(chi_floor>0)))
    throw std::invalid_argument("Inner AMR chi floor requires regular weight and positive chi_floor");
  parameters.enforce_cfl=true;
  parameters.outer_lapse_damping=pin->GetOrAddReal("z4c_hyperboloidal",
                                                  "outer_lapse_damping",.25);
  if (pin->DoesParameterExist("z4c_hyperboloidal", "scri_treatment") &&
      pin->GetString("z4c_hyperboloidal", "scri_treatment") != "A")
    throw std::invalid_argument("Only scri treatment A is supported");
  const auto overlap_mode=pin->GetOrAddString("z4c_hyperboloidal","overlap_mode","agreement_full");
  if (overlap_mode!="agreement_full")
    throw std::invalid_argument("Only overlap_mode=agreement_full is supported");
  for (const char *key:{"cartesian_cutoff","agreement_inner","agreement_outer"})
    if (pin->DoesParameterExist("z4c_hyperboloidal",key))
      throw std::invalid_argument(std::string("Obsolete masked-overlap parameter: ")+key);
  agreement.inner=pm->atlas->SphericalInnerRadius();
  agreement.kappa=pin->GetOrAddReal("z4c_hyperboloidal","agreement_kappa",1);
  agreement.taper_width=pin->GetOrAddReal("z4c_hyperboloidal","agreement_taper_width",.015);
  if (!(agreement.inner>0 && agreement.kappa>0 && std::isfinite(agreement.kappa)
        && agreement.taper_width>0 && std::isfinite(agreement.taper_width)))
    throw std::invalid_argument("full overlap requires spherical shells, positive kappa and taper width");
  parameters.Validate();
  BuildLocalWorkLists();
  AllocateStorage();
}

void Z4cHyperboloidal::AllocateStorage() {
  if (!UsesMultirate()) {
    Kokkos::realloc(u1,nblocks,nvar,storage[2],storage[1],storage[0]);
    Kokkos::realloc(u_accum,nblocks,nvar,storage[2],storage[1],storage[0]);
    Kokkos::realloc(s1,nfaces,nvar,n[2],n[1]);
    Kokkos::realloc(s_accum,nfaces,nvar,n[2],n[1]);
  }
  Kokkos::realloc(u0,nblocks,nvar,storage[2],storage[1],storage[0]);
  Kokkos::realloc(u_rhs,nblocks,nvar,storage[2],storage[1],storage[0]);
  Kokkos::realloc(residual,nblocks,nvar,storage[2],storage[1],storage[0]);
  if (pmy_pack->pmesh->multilevel && !UsesMultirate()) {
    Kokkos::realloc(coarse_u0,nblocks,nvar,n[2]/2+2*origin,
                    n[1]/2+2*origin,n[0]/2+2*origin);
  }
  Kokkos::realloc(s0,nfaces,nvar,n[2],n[1]);
  Kokkos::realloc(s_rhs,nfaces,nvar,n[2],n[1]);
  Kokkos::realloc(geometry,npoints);
  Kokkos::realloc(reference,npoints);
  Kokkos::realloc(stationary_source,npoints);
  Kokkos::realloc(residual_weights, npoints);
  Kokkos::realloc(spatial_omega, npoints);
  Kokkos::realloc(surface_geometry,nsurface_points);
  Kokkos::realloc(surface_reference,nsurface_points);
  Kokkos::realloc(chi_records,npoints);
  Kokkos::realloc(surface_chi_records,nsurface_points);
  // Ghosts must be populated from residual interpolation before a stencil reads
  // them; analytic reference values are never used as a substitute for exchange.
  const Real poison=std::numeric_limits<Real>::quiet_NaN();
  Kokkos::deep_copy(u0,poison);
  Kokkos::deep_copy(residual,poison);
}

void Z4cHyperboloidal::ResetMeshStorage() {
  InvalidateDiagnostics();
  // Regridding is an accepted-step operation: no old stage or peer buffer may
  // still be in use. Preserve any shallow snapshots held by RemeshState by
  // dropping our views before allocation, even if the new extents are equal.
  Kokkos::fence();
  initialized_=false;
  plans_ready_=false;
  mapped_halo_.reset();
  agreement_plan_.reset();
  surface_exchange_.reset();
  dissipation_plan_.reset();
  for (auto &batch:rhs_points_) batch=DvceArray1D<int>();
  rhs_status=DvceArray1D<int>();
  rhs_gap=DvceArray1D<Real>(); rhs_cfl=DvceArray1D<Real>();
  for (auto *array:{&u0,&u1,&u_rhs,&u_accum,&coarse_u0,&residual})
    *array=DvceArray5D<Real>();
  for (auto *array:{&s0,&s1,&s_rhs,&s_accum}) *array=DvceArray4D<Real>();
  evolved_bounds=DualArray1D<EvolvedBounds>();
  points=DualArray1D<BlockPoint>();
  surface_points=DualArray1D<SurfacePoint>();
  face_blocks=DualArray1D<int>(); block_faces=DualArray1D<int>();
  geometry=DvceArray1D<mapped::Point>(); surface_geometry=DvceArray1D<mapped::Point>();
  reference=DvceArray1D<CartesianChartJets>();
  surface_reference=DvceArray1D<CartesianChartJets>();
  stationary_source=DvceArray1D<StationaryGaugeSource<Real>>();
  residual_weights = {};
  spatial_omega = {};
  chi_records=DvceArray1D<ChiRecord>(); surface_chi_records=DvceArray1D<ChiRecord>();
  nblocks=pmy_pack->nmb_thispack;
  BuildLocalWorkLists();
  AllocateStorage();
  BuildReferenceCache();
  const Real poison=std::numeric_limits<Real>::quiet_NaN();
  Kokkos::deep_copy(s0,poison);
  const ChiRecord empty{poison,poison,poison,poison,poison,poison,poison,poison};
  Kokkos::deep_copy(chi_records,empty);
  Kokkos::deep_copy(surface_chi_records,empty);
  // Install must explicitly populate every evolved field and every record,
  // including defaults for new points with no inherited history packet.
}

void Z4cHyperboloidal::BuildStagePlans() {
  if (!initialized_) throw std::logic_error("Stage plans require initialized geometry");
  plans_ready_=false;
  mapped_halo_=std::make_unique<MappedHalo>();
  mapped_halo_->Rebuild(pmy_pack,origin,activity);
  surface_exchange_=std::make_unique<SurfaceExchange>(pmy_pack,this);
  dissipation_plan_=std::make_unique<DissipationPlan>();
  dissipation_plan_->Rebuild(*pmy_pack,*this);
  std::vector<int> lookup(nblocks*storage[0]*storage[1]*storage[2],-1);
  const auto index=[&](int m,int k,int j,int i) {
    return ((m*storage[2]+k)*storage[1]+j)*storage[0]+i;
  };
  for (int p=0;p<npoints;++p) {
    const auto c=points.h_view(p);lookup[index(c.m,c.k,c.j,c.i)]=p;
  }
  const std::vector<HaloPoint> *batches[4]={&mapped_halo_->InteriorPoints(),
      &mapped_halo_->LocalHaloPoints(),&mapped_halo_->RemoteHaloPoints(),
      &mapped_halo_->SurfaceDependentPoints()};
  if (UsesMultirate())
    for (auto &by_block:rhs_points_by_block_) by_block.assign(nblocks,{});
  std::vector<bool> covered(npoints,false);
  std::vector<int> dependent;
  for (int batch=0;batch<4;++batch) {
    Kokkos::realloc(rhs_points_[batch],batches[batch]->size());
    auto host=Kokkos::create_mirror_view(rhs_points_[batch]);
    for (size_t i=0;i<batches[batch]->size();++i) {
      const auto c=(*batches[batch])[i];const int p=lookup[index(c.m,c.k,c.j,c.i)];
      if (p<0 || covered[p]) throw std::logic_error("RHS batches overlap or leave evolved state");
      covered[p]=true;host(i)=p;
      if (UsesMultirate()) {
        rhs_points_by_block_[batch][c.m].push_back(p);
        if (batch!=0)rhs_points_by_block_[static_cast<int>(RHSSubset::Dependent)][c.m].push_back(p);
      }
      if (batch!=0)dependent.push_back(p);
    }
    Kokkos::deep_copy(rhs_points_[batch],host);
  }
  for (bool yes:covered) if (!yes) throw std::logic_error("RHS batches omit evolved state");
  // Keep the dependency classifications available to diagnostics and oracles,
  // but evaluate their disjoint non-interior union in one native task.
  auto &late=rhs_points_[static_cast<int>(RHSSubset::Dependent)];
  Kokkos::realloc(late,dependent.size());
  auto late_host=Kokkos::create_mirror_view(late);
  for (std::size_t i=0;i<dependent.size();++i)late_host(i)=dependent[i];
  Kokkos::deep_copy(late,late_host);
  Kokkos::realloc(rhs_status,npoints);
  Kokkos::realloc(rhs_gap,npoints);
  Kokkos::realloc(rhs_cfl,npoints);

  agreement_plan_=std::make_unique<OverlapAgreementPlan>();
  agreement_plan_->Rebuild(pmy_pack,*this);

  plans_ready_=true;
}

void Z4cHyperboloidal::BuildLocalWorkLists() {
  auto &maps=pmy_pack->pmb->mb_map.h_view;
  std::vector<BlockPoint> cells;
  std::vector<int> faces;
  Kokkos::realloc(evolved_bounds,nblocks);
  Kokkos::realloc(block_faces,nblocks);
  for (int m=0;m<nblocks;++m) {
    const auto &map=maps(m);
    auto &bounds=evolved_bounds.h_view(m);
    for (int a=0;a<3;++a) {
      bounds.lo[a]=origin-map.overlap_sides[2*a];
      bounds.hi[a]=origin+n[a]+map.overlap_sides[2*a+1]-1;
    }
    block_faces.h_view(m)=-1;
    if (map.outer_surface) {
      block_faces.h_view(m)=faces.size();
      faces.push_back(m);
    }
    for (int k=bounds.lo[2];k<=bounds.hi[2];++k)
      for (int j=bounds.lo[1];j<=bounds.hi[1];++j)
        for (int i=bounds.lo[0];i<=bounds.hi[0];++i)
          if (activity.Evolved(map,n[0],i-origin,j-origin,k-origin))
            cells.push_back({m,k,j,i});
  }
  npoints=cells.size(); nfaces=faces.size(); nsurface_points=nfaces*n[1]*n[2];
  Kokkos::realloc(points,npoints);
  Kokkos::realloc(face_blocks,nfaces);
  Kokkos::realloc(surface_points,nsurface_points);
  for (int p=0;p<npoints;++p) points.h_view(p)=cells[p];
  for (int face=0;face<nfaces;++face) {
    face_blocks.h_view(face)=faces[face];
    for (int k=0;k<n[2];++k) for (int j=0;j<n[1];++j)
      surface_points.h_view((face*n[2]+k)*n[1]+j)={face,k,j};
  }
  evolved_bounds.template modify<HostMemSpace>();
  evolved_bounds.template sync<DevExeSpace>();
  points.template modify<HostMemSpace>(); points.template sync<DevExeSpace>();
  face_blocks.template modify<HostMemSpace>(); face_blocks.template sync<DevExeSpace>();
  block_faces.template modify<HostMemSpace>(); block_faces.template sync<DevExeSpace>();
  surface_points.template modify<HostMemSpace>(); surface_points.template sync<DevExeSpace>();
}

void Z4cHyperboloidal::BuildReferenceCache() {
  SphericalReference3D initial(parameters);
  auto g=Kokkos::create_mirror_view(geometry);
  auto ref=Kokkos::create_mirror_view(reference);
  auto sources=Kokkos::create_mirror_view(stationary_source);
  auto sg=Kokkos::create_mirror_view(surface_geometry);
  auto sr=Kokkos::create_mirror_view(surface_reference);
  auto maps=pmy_pack->pmb->mb_map.h_view;
  for (int p=0;p<npoints;++p) {
    const auto c=points.h_view(p);
    const auto map=maps(c.m);
    const Real q[3]={(c.i-origin+map.node_offset[0])/n[0],
                     (c.j-origin+map.node_offset[1])/n[1],
                     (c.k-origin+map.node_offset[2])/n[2]};
    g(p)=map.Geometry(q);
    const Real r=SphericalPolicy::Radius(g(p).x);
    if (!((r>0 || (r==0 && parameters.residual_weight.Regular())) && r<1)) throw std::runtime_error("Bulk evolved point outside punctured ball");
    ref(p)=initial.At(g(p).x);
    sources(p)=initial.source.At(r);
  }
  for (int p=0;p<nsurface_points;++p) {
    const auto c=surface_points.h_view(p);
    const auto map=maps(face_blocks.h_view(c.face));
    const Real q[3]={1,(c.j+.5)/n[1],(c.k+.5)/n[2]};
    sg(p)=map.Geometry(q);
    sr(p)=initial.At(sg(p).x,true);
  }
  boundary_source=initial.source.At(1);
  Kokkos::deep_copy(geometry,g); Kokkos::deep_copy(reference,ref);
  Kokkos::deep_copy(stationary_source,sources);
  const auto coords = geometry;
  const auto weights = residual_weights;
  const auto omega = spatial_omega;
  const auto weight = parameters.residual_weight;
  const auto profile = parameters.profile;
  Kokkos::parallel_for(
      "cache stationary bulk profiles", npoints, KOKKOS_LAMBDA(int p) {
        weights(p) = weight.At(coords(p).x);
        omega(p) = profile.At(coords(p).x);
      });
  Kokkos::deep_copy(surface_geometry,sg); Kokkos::deep_copy(surface_reference,sr);
}

void Z4cHyperboloidal::InitializeReference() {
  InvalidateDiagnostics();
  BuildReferenceCache();
  auto list=points.d_view; auto surf=surface_points.d_view;
  auto refs=reference; auto srefs=surface_reference;
  auto state=u0; auto edge=s0;
  Kokkos::parallel_for("hyperboloidal initial bulk",Kokkos::RangePolicy<>(0,npoints),
    KOKKOS_LAMBDA(int p) {
      const auto c=list(p);
      for (int f=0;f<nvar;++f) state(c.m,f,c.k,c.j,c.i)=refs(p)[f].f;
    });
  Kokkos::parallel_for("hyperboloidal initial scri",Kokkos::RangePolicy<>(0,nsurface_points),
    KOKKOS_LAMBDA(int p) {
      const auto c=surf(p);
      for (int f=0;f<nvar;++f) edge(c.face,f,c.k,c.j)=srefs(p)[f].f;
    });
  FormResidual();
  initialized_=true;
}

void Z4cHyperboloidal::FormResidual() {
  auto list=points.d_view; auto state=u0; auto ref=reference;
  const auto active=active_blocks;
  auto map=geometry; auto output=residual;
  const auto weight=parameters.residual_weight;
  Kokkos::parallel_for("hyperboloidal owned residual",Kokkos::RangePolicy<>(0,npoints),
    KOKKOS_LAMBDA(int p) {
      const auto c=list(p); if(active.extent(0) && !active(c.m))return;
      const auto &x=map(p).x;
      const Real r2=weight.Value(x);
      for (int f=0;f<nvar;++f) {
        output(c.m,f,c.k,c.j,c.i)=Fields::Independent(f)
            ? (state(c.m,f,c.k,c.j,c.i)-ref(p)[f].f)/(f==0 ? r2 : 1) : 0;
      }
    });
}

void Z4cHyperboloidal::EnforceAlgebraicConstraints() {
  InvalidateDiagnostics();
  auto list=points.d_view; auto state=u0;
  auto surf=surface_points.d_view; auto edge=s0;
  const auto active=active_blocks;const auto owners=face_blocks.d_view;
  int bad_bulk=0,bad_surface=0;
  Kokkos::parallel_reduce("hyperboloidal bulk algebraic closure",
    Kokkos::RangePolicy<>(0,npoints),KOKKOS_LAMBDA(int p,int &bad) {
      const auto c=list(p); if(active.extent(0) && !active(c.m))return;
      CartesianChartJets jets{};
      for (int f=0;f<nvar;++f) jets[f].f=state(c.m,f,c.k,c.j,c.i);
      if (!TryExpandCartesianChart(jets)) {++bad; return;}
      state(c.m,6,c.k,c.j,c.i)=jets[6].f;
      state(c.m,12,c.k,c.j,c.i)=jets[12].f;
    },bad_bulk);
  Kokkos::parallel_reduce("hyperboloidal scri algebraic closure",
    Kokkos::RangePolicy<>(0,nsurface_points),KOKKOS_LAMBDA(int p,int &bad) {
      const auto c=surf(p); if(active.extent(0) && !active(owners(c.face)))return;
      CartesianChartJets jets{};
      for (int f=0;f<nvar;++f) jets[f].f=edge(c.face,f,c.k,c.j);
      if (!TryExpandCartesianChart(jets)) {++bad; return;}
      edge(c.face,6,c.k,c.j)=jets[6].f;
      edge(c.face,12,c.k,c.j)=jets[12].f;
    },bad_surface);
  if (bad_bulk || bad_surface) throw std::runtime_error("Invalid hyperboloidal metric chart");
}
}  // namespace z4c_hyperboloidal
