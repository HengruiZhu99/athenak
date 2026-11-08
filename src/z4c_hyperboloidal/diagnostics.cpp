// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
//! \file diagnostics.cpp
//! \brief Native accepted-state constraints and sparse FastFlow coordination.

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

#include "mesh/mesh.hpp"
#include "z4c_hyperboloidal/diagnostics.hpp"
#include "z4c_hyperboloidal/dissipation_plan.hpp"
#include "z4c_hyperboloidal/endpoint_state.hpp"
#include "z4c_hyperboloidal/horizon_sampling.hpp"
#include "z4c_hyperboloidal/mapped_halo.hpp"
#include "z4c_hyperboloidal/z4c_hyperboloidal.hpp"

namespace z4c_hyperboloidal {
namespace {
void RequireAll(bool valid,const char *message) {
  int bad=valid?0:1;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,&bad,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
#endif
  if (bad) throw std::runtime_error(message);
}

TaskStatus Both(TaskStatus a,TaskStatus b) {
  if (a==TaskStatus::fail || b==TaskStatus::fail)
    throw std::runtime_error("Hyperboloidal diagnostic exchange failed");
  return a==TaskStatus::complete && b==TaskStatus::complete
      ? TaskStatus::complete : TaskStatus::incomplete;
}

struct SumConstraints {
  struct value_type {Real sum[10];};
  DvceArray1D<ConstraintPoint> points;
  DvceArray1D<Real> weights;
  DvceArray1D<int> exterior;
  KOKKOS_INLINE_FUNCTION void init(value_type &out) const {
    for (int a=0;a<10;++a) out.sum[a]=0;
  }
  KOKKOS_INLINE_FUNCTION void join(value_type &out,const value_type &other) const {
    for (int a=0;a<10;++a) out.sum[a]+=other.sum[a];
  }
  KOKKOS_INLINE_FUNCTION void operator()(int p,value_type &out) const {
    const auto &point=points(p);const auto &c=point.constraints;
    const Real w=weights(p);
    const Real values[5]={w,w*c.hamiltonian*c.hamiltonian,
        w*c.momentum_physical_norm2,w*point.theta*point.theta,w*c.z4.z_physical_norm2};
    for (int f=0;f<5;++f) {
      out.sum[f]+=values[f];
      if (exterior(p)) out.sum[5+f]+=values[f];
    }
  }
};
}  // namespace

Diagnostics &Z4cHyperboloidal::GetDiagnostics() {
  if (!initialized_) throw std::logic_error("Diagnostics require an initialized state");
  if (!plans_ready_) BuildStagePlans();
  if (!diagnostics_) diagnostics_=std::make_unique<Diagnostics>(pmy_pack,this,pinput_);
  return *diagnostics_;
}

std::array<Real,10> Z4cHyperboloidal::DiagnosticHistorySums() {
  return GetDiagnostics().HistorySums();
}

Diagnostics::Diagnostics(MeshBlockPack *pack,Z4cHyperboloidal *module,ParameterInput *pin)
    : pack_(pack),module_(module),pin_(pin) {}
Diagnostics::~Diagnostics() = default;

void Diagnostics::Rebuild() {
  Kokkos::Timer timer;
  auto &p=*module_;
  boundary_=std::make_unique<SurfaceExchange>(pack_,module_);
  Kokkos::realloc(points_,p.npoints);
  Kokkos::realloc(weights_,p.npoints);
  Kokkos::realloc(exterior_,p.npoints);
  DvceArray2D<Real> positions("hyperboloidal diagnostic positions",p.npoints,3);
  const auto cells=p.points.d_view;const auto maps=pack_->pmb->mb_map.d_view;
  const auto geometry=p.geometry;const auto weights=weights_;
  const auto volumes=p.dissipation_plan_->Bind().volumes;
  const int n=p.n[0],origin=p.origin;
  const Real inner=pack_->pmesh->atlas->SphericalInnerRadius();
  const Real half_width=pack_->pmesh->atlas->HalfWidth();
  Kokkos::parallel_for("hyperboloidal diagnostic quadrature",
    Kokkos::RangePolicy<>(0,p.npoints),KOKKOS_LAMBDA(int q) {
      const auto cell=cells(q);const auto map=maps(cell.m);
      const auto &x=geometry(q).x;
      for (int a=0;a<3;++a) positions(q,a)=x[a];
      const bool nominal=cell.i>=origin && cell.i<origin+n
          && cell.j>=origin && cell.j<origin+n && cell.k>=origin && cell.k<origin+n;
      Real weight=nominal?volumes(q):0;
      if (inner>0 && nominal) {
        const Real r=SphericalPolicy::Radius(x);
        const Real t=fmax(Real(0),fmin(Real(1),(r-inner)/(half_width-inner)));
        const Real shell=t*t*t*(10+t*(-15+6*t));
        weight*=map.face<0?1-shell:shell;
      }
      weights(q)=weight;
    });
  positions_=Kokkos::create_mirror_view_and_copy(HostMemSpace(),positions);
  mesh_sequence_=pack_->pmesh->GetAMRLoadBalanceUpdateSeq();
  Kokkos::fence();statistics_.startup_seconds+=timer.seconds();
}

void Diagnostics::Prepare() {
  Kokkos::Timer timer;
  auto &p=*module_;
  const auto cells=p.points.d_view;const auto surface=p.surface_points.d_view;
  const auto u=p.u0;const auto s=p.s0;
  int bulk_bad=0,surface_bad=0;
  Kokkos::parallel_reduce("hyperboloidal diagnostic bulk preflight",
    Kokkos::RangePolicy<>(0,p.npoints),KOKKOS_LAMBDA(int q,int &bad) {
      const auto c=cells(q);
      for (int f=0;f<Fields::count;++f)
        if (!Kokkos::isfinite(u(c.m,f,c.k,c.j,c.i))) ++bad;
    },bulk_bad);
  Kokkos::parallel_reduce("hyperboloidal diagnostic scri preflight",
    Kokkos::RangePolicy<>(0,p.nsurface_points),KOKKOS_LAMBDA(int q,int &bad) {
      const auto c=surface(q);
      for (int f=0;f<Fields::count;++f)
        if (!Kokkos::isfinite(s(c.face,f,c.k,c.j))) ++bad;
    },surface_bad);
  RequireAll(bulk_bad==0 && surface_bad==0,"Nonfinite owned hyperboloidal diagnostic state");

  p.FormResidual();
  auto &bulk=*p.mapped_halo_;auto &boundary=*boundary_;
  bulk.StartReceives();boundary.StartReceives();
  bulk.PackAndSend(p.residual,p.u0);boundary.PackAndSend();
  bulk.FillLocal(p.residual,p.u0);
  TaskList prepare;TaskID none(0);bool valid=true;
  auto received_bulk=prepare.AddTask([&](Driver *,int) {
    return bulk.Receive();
  },none);
  auto filled_bulk=prepare.AddTask([&](Driver *,int) {
    return bulk.FillRemote(p.residual,p.u0);
  },received_bulk);
  auto received_surface=prepare.AddTask([&](Driver *,int) {
    return boundary.Receive();
  },none);
  auto solved_surface=prepare.AddTask([&](Driver *,int) {
    // Retire every peer buffer before propagating a rank-local numerical error.
    try {
      const auto status=boundary.Sample();
      return status==TaskStatus::complete?boundary.Solve():status;
    } catch (const std::runtime_error &) {
      valid=false;return TaskStatus::complete;
    }
  },received_surface);
  prepare.AddTask([&](Driver *,int) {
    const auto a=bulk.ClearSend(),b=boundary.ClearSend();return Both(a,b);
  },none);
  auto used_receives=filled_bulk|solved_surface;
  prepare.AddTask([&](Driver *,int) {
    const auto a=bulk.ClearRecv(),b=boundary.ClearRecv();return Both(a,b);
  },used_receives);
  while (prepare.DoAvailable(nullptr,0)!=TaskListStatus::complete) {}
  RequireAll(valid,"Invalid hyperboloidal diagnostic normal reconstruction");
  Kokkos::fence();statistics_.preparation_seconds+=timer.seconds();
}

void Diagnostics::EvaluateConstraints() {
  Kokkos::Timer timer;
  const auto &p=*module_;
  const auto cells=p.points.d_view;const auto geometry=p.geometry;const auto refs=p.reference;
  const auto residual=p.residual;const auto output=points_;const auto params=p.parameters;
  const auto surface=boundary_->BindKernel();const int n=p.n[0];
  Kokkos::parallel_for("hyperboloidal accepted constraints",
    Kokkos::RangePolicy<>(0,p.npoints),KOKKOS_LAMBDA(int q) {
      const auto c=cells(q);const auto map=geometry(q);
      const BlockPointReader reader{residual,c.m,c.k,c.j,c.i};
      const Real inv_h[3]={Real(n),Real(n),Real(n)};
      const auto state=surface.HasRadialRow(q)
          ? EndpointState(q,reader,map,inv_h,refs(q),params,surface)
          : BuildPointState(reader,map,inv_h,refs(q),params);
      ConstraintPoint point;point.status=state.status;
      if (state.status==PointStatus::success) {
        const auto u=SphericalPolicy::Physical(map.x,state.jets);
        point.chi=u.chi.value;point.lapse=u.alpha.value;point.theta=u.theta.value;
        auto omega=params.profile.At(map.x);
        SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,omega);
        point.constraints=EvolvedConstraints(u,omega);
        if (!point.constraints.valid || point.constraints.momentum_physical_norm2<0
            || point.constraints.z4.z_physical_norm2<0)
          point.status=PointStatus::invalid_geometry;
      }
      output(q)=point;
    });
  int invalid=0;
  Kokkos::parallel_reduce("hyperboloidal accepted constraint validity",
    Kokkos::RangePolicy<>(0,p.npoints),KOKKOS_LAMBDA(int q,int &bad) {
      if (output(q).status!=PointStatus::success) ++bad;
    },invalid);
  RequireAll(invalid==0,"Invalid hyperboloidal accepted constraint geometry");
  Kokkos::fence();statistics_.constraint_seconds+=timer.seconds();
}

void Diagnostics::Refresh() {
  const auto &p=*module_;
  RequireAll(p.initialized_ && p.plans_ready_,"Hyperboloidal diagnostic plans are not ready");
  const int sequence=pack_->pmesh->GetAMRLoadBalanceUpdateSeq();
  int dirty=!ready_ || version_!=p.StateVersion() || mesh_sequence_!=sequence
      || cycle_!=pack_->pmesh->ncycle || time_!=pack_->pmesh->time;
#if MPI_PARALLEL_ENABLED
  // A changed owned field may feed a stencil or horizon sample on another rank.
  // Peers must refresh together even when only its owner invalidated the state.
  MPI_Allreduce(MPI_IN_PLACE,&dirty,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
#endif
  if (!dirty) return;
  ready_=horizon_ready_=sums_ready_=written_=false;
  surface_=FastFlowSurface{};
  if (!boundary_ || mesh_sequence_!=sequence) Rebuild();
  Prepare();EvaluateConstraints();
  version_=p.StateVersion();cycle_=pack_->pmesh->ncycle;time_=pack_->pmesh->time;
  ready_=true;++statistics_.refreshes;
}

const FastFlow::Result &Diagnostics::FindHorizon() {
  Refresh();
  if (horizon_ready_) return horizon_;
  Kokkos::Timer timer;
  if (!finder_) {
    // N12 uses the original fixed-origin finder. These defaults live in the
    // existing FastFlow input block; mapped evolution does not instantiate ADM.
    pin_->GetOrAddInteger("fastflow","lmax",12);
    pin_->GetOrAddInteger("fastflow","ntheta",24);
    pin_->GetOrAddInteger("fastflow","flow_iterations_0",300);
    pin_->GetOrAddReal("fastflow","flow_alpha_beta_const_0",.1);
    pin_->GetOrAddReal("fastflow","initial_radius_0",.02);
    pin_->GetOrAddReal("fastflow","mass_tol_0",1e-10);
    pin_->GetOrAddReal("fastflow","expansion_rms_tol_0",1e-3);
    pin_->GetOrAddReal("fastflow","start_time_0",0);
    pin_->GetOrAddReal("fastflow","stop_time_0",std::numeric_limits<Real>::max());
    sampler_=std::make_unique<HorizonSampler>(pack_,module_);
    finder_=std::make_unique<FastFlow>(pack_,pin_,0,
      [](const Real[3],z4c::HorizonGeometry<Real> &) -> bool {
        // A scalar callback must never enter a collective sparse exchange.
        // The batch callback below is the only native sampling entry point.
        throw std::logic_error("Native horizon sampling requires the batch callback");
      });
    finder_->SetBatchGeometrySampler([this](int n,const Real *x,
        z4c::HorizonGeometry<Real> *out,int *owners) {
      sampler_->SampleBatch(n,x,out,owners);
    });
  }
  finder_->Find(pack_->pmesh->ncycle,pack_->pmesh->time);
  horizon_=finder_->GetResult();surface_=finder_->GetSurface();
  RequireAll(horizon_.found && horizon_.complete_surface && surface_.Valid(),
             "Hyperboloidal history requires a complete converged horizon");
  auto mask=Kokkos::create_mirror_view(exterior_);
  for (int q=0;q<module_->npoints;++q) {
    const Real x[3]={positions_(q,0),positions_(q,1),positions_(q,2)};
    mask(q)=surface_.Outside(x)?1:0;
  }
  Kokkos::deep_copy(exterior_,mask);
  Kokkos::fence();statistics_.horizon_seconds+=timer.seconds();
  ++statistics_.searches;horizon_ready_=true;
  return horizon_;
}

std::array<Real,10> Diagnostics::HistorySums() {
  FindHorizon();
  if (!sums_ready_) {
    Kokkos::Timer timer;SumConstraints::value_type out;
    Kokkos::parallel_reduce("hyperboloidal history sums",
        Kokkos::RangePolicy<>(0,module_->npoints),SumConstraints{points_,weights_,exterior_},out);
    for (int a=0;a<10;++a) sums_[a]=out.sum[a];
    statistics_.reduction_seconds+=timer.seconds();sums_ready_=true;
  }
  if (!written_) {
    Kokkos::Timer timer;
    finder_->Write(pack_->pmesh->ncycle,pack_->pmesh->time);
    statistics_.io_seconds+=timer.seconds();written_=true;
  }
  // HistoryOutput owns MPI summation and only then forms the eight RMS values.
  return sums_;
}
}  // namespace z4c_hyperboloidal
