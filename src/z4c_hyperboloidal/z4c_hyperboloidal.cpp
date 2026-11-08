//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_hyperboloidal.cpp
//! \brief Native block storage and reference initialization for hyperboloidal Z4c.

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "mesh/mesh.hpp"
#include "z4c_hyperboloidal/z4c_hyperboloidal.hpp"
#include "z4c_hyperboloidal/central_z4c_configuration.hpp"
#include "z4c_hyperboloidal/spherical_policy.hpp"
#include "z4c_hyperboloidal/mapped_halo.hpp"
#include "z4c_hyperboloidal/surface_exchange.hpp"
#include "z4c_hyperboloidal/dissipation_plan.hpp"
#include "z4c_hyperboloidal/remesh.hpp"
#include "z4c_hyperboloidal/diagnostics.hpp"

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
    if (pin->GetOrAddString("time","integrator","rk4_classical")!="rk4_classical") {
      throw std::invalid_argument("Hyperboloidal evolution requires rk4_classical");
    }
  }
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
  requested_dt=pin->GetReal("time","dt");
  if (!std::isfinite(ko) || ko<0 || !std::isfinite(chi_floor) || chi_floor<0 ||
      !std::isfinite(penalty) || penalty<0 || !std::isfinite(requested_dt) ||
      requested_dt<=0) throw std::invalid_argument("Invalid hyperboloidal numerical policy");
  dtnew=std::numeric_limits<Real>::max();
  CentralZ4cConfiguration config;
  config.inner_z4c_radius_M=pin->GetOrAddReal("z4c_hyperboloidal","inner_z4c_radius_M",20);
  config.compact_slicing_end=pin->GetOrAddReal("z4c_hyperboloidal","compact_slicing_end",.97);
  parameters=config.Resolve().gauge;
  parameters.enforce_cfl=true;
  parameters.outer_lapse_damping=pin->GetOrAddReal("z4c_hyperboloidal",
                                                  "outer_lapse_damping",.25);
  parameters.Validate();
  BuildLocalWorkLists();
  AllocateStorage();
}

void Z4cHyperboloidal::AllocateStorage() {
  Kokkos::realloc(u0,nblocks,nvar,storage[2],storage[1],storage[0]);
  Kokkos::realloc(u1,nblocks,nvar,storage[2],storage[1],storage[0]);
  Kokkos::realloc(u_rhs,nblocks,nvar,storage[2],storage[1],storage[0]);
  Kokkos::realloc(u_accum,nblocks,nvar,storage[2],storage[1],storage[0]);
  Kokkos::realloc(residual,nblocks,nvar,storage[2],storage[1],storage[0]);
  if (pmy_pack->pmesh->multilevel) {
    Kokkos::realloc(coarse_u0,nblocks,nvar,n[2]/2+2*origin,
                    n[1]/2+2*origin,n[0]/2+2*origin);
  }
  Kokkos::realloc(s0,nfaces,nvar,n[2],n[1]);
  Kokkos::realloc(s1,nfaces,nvar,n[2],n[1]);
  Kokkos::realloc(s_rhs,nfaces,nvar,n[2],n[1]);
  Kokkos::realloc(s_accum,nfaces,nvar,n[2],n[1]);
  Kokkos::realloc(geometry,npoints);
  Kokkos::realloc(reference,npoints);
  Kokkos::realloc(stationary_source,npoints);
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
  mapped_halo_->Rebuild(pmy_pack,origin);
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
  std::vector<bool> covered(npoints,false);
  for (int batch=0;batch<4;++batch) {
    Kokkos::realloc(rhs_points_[batch],batches[batch]->size());
    auto host=Kokkos::create_mirror_view(rhs_points_[batch]);
    for (size_t i=0;i<batches[batch]->size();++i) {
      const auto c=(*batches[batch])[i];const int p=lookup[index(c.m,c.k,c.j,c.i)];
      if (p<0 || covered[p]) throw std::logic_error("RHS batches overlap or leave evolved state");
      covered[p]=true;host(i)=p;
    }
    Kokkos::deep_copy(rhs_points_[batch],host);
  }
  for (bool yes:covered) if (!yes) throw std::logic_error("RHS batches omit evolved state");
  Kokkos::realloc(rhs_status,npoints);
  Kokkos::realloc(rhs_gap,npoints);
  Kokkos::realloc(rhs_cfl,npoints);
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
        for (int i=bounds.lo[0];i<=bounds.hi[0];++i) cells.push_back({m,k,j,i});
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
    if (!(r>0 && r<1)) throw std::runtime_error("Bulk evolved point outside punctured ball");
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
  auto map=geometry; auto output=residual;
  Kokkos::parallel_for("hyperboloidal owned residual",Kokkos::RangePolicy<>(0,npoints),
    KOKKOS_LAMBDA(int p) {
      const auto c=list(p); const auto &x=map(p).x;
      const Real r=SphericalPolicy::Radius(x),r2=r*r;
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
  int bad_bulk=0,bad_surface=0;
  Kokkos::parallel_reduce("hyperboloidal bulk algebraic closure",
    Kokkos::RangePolicy<>(0,npoints),KOKKOS_LAMBDA(int p,int &bad) {
      const auto c=list(p); CartesianChartJets jets{};
      for (int f=0;f<nvar;++f) jets[f].f=state(c.m,f,c.k,c.j,c.i);
      if (!TryExpandCartesianChart(jets)) {++bad; return;}
      state(c.m,6,c.k,c.j,c.i)=jets[6].f;
      state(c.m,12,c.k,c.j,c.i)=jets[12].f;
    },bad_bulk);
  Kokkos::parallel_reduce("hyperboloidal scri algebraic closure",
    Kokkos::RangePolicy<>(0,nsurface_points),KOKKOS_LAMBDA(int p,int &bad) {
      const auto c=surf(p); CartesianChartJets jets{};
      for (int f=0;f<nvar;++f) jets[f].f=edge(c.face,f,c.k,c.j);
      if (!TryExpandCartesianChart(jets)) {++bad; return;}
      edge(c.face,6,c.k,c.j)=jets[6].f;
      edge(c.face,12,c.k,c.j)=jets[12].f;
    },bad_surface);
  if (bad_bulk || bad_surface) throw std::runtime_error("Invalid hyperboloidal metric chart");
}
}  // namespace z4c_hyperboloidal
