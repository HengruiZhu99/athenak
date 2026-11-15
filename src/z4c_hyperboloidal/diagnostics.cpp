// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
//! \file diagnostics.cpp
//! \brief Native accepted-state constraints and sparse FastFlow coordination.

#include <algorithm>
#include <limits>
#include <iostream>
#include "globals.hpp"
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
struct SumShell {
  using value_type=SumConstraints::value_type;
  DvceArray1D<ConstraintPoint> points;
  DvceArray1D<Real> weights;
  DvceArray1D<mapped::Point> maps;
  DvceArray1D<CartesianChartJets> reference;
  DvceArray1D<BlockPoint> cells;
  DvceArray5D<Real> state;
  Real inner,outer;
  KOKKOS_INLINE_FUNCTION void init(value_type &v) const {for(auto &x:v.sum)x=0;}
  KOKKOS_INLINE_FUNCTION void join(value_type &a,const value_type &b) const {
    for(int i=0;i<10;++i)a.sum[i]+=b.sum[i];
  }
  KOKKOS_INLINE_FUNCTION void operator()(int q,value_type &out) const {
    const auto &x=maps(q).x;const Real r=SphericalPolicy::Radius(x);
    if(r<inner || r>=outer)return;
    const auto point=points(q);const auto c=point.constraints;
    const auto cell=cells(q);const auto ref=reference(q);const Real w=weights(q);
    const Real dc=point.chi-ref[0].f;
    const Real da=point.lapse-ref[18].f-(1+r*r)/2;
    Real db=0;for(int a=0;a<3;++a)db+=(state(cell.m,22+a,cell.k,cell.j,cell.i)-ref[22+a].f)*x[a]/r;
    const Real values[10]={1,c.hamiltonian*c.hamiltonian,c.momentum_physical_norm2,
      point.theta*point.theta,c.z4.z_physical_norm2,dc*dc,da*da,db*db,dc,da};
    for(int i=0;i<10;++i)out.sum[i]+=w*values[i];
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

std::array<Real,10> Z4cHyperboloidal::DiagnosticShellSums(int band) {
  return GetDiagnostics().ShellSums(band);
}
std::array<Real,10> Diagnostics::ShellSums(int band) {
  if(band<0 || band>=4)throw std::invalid_argument("invalid scri diagnostic band");
  Refresh();const Real edges[5]={.80,.88,.94,.97,1.};auto &p=*module_;
  SumShell::value_type result;
  Kokkos::parallel_reduce("fixed physical compact shell diagnostics",
    Kokkos::RangePolicy<>(0,p.npoints),SumShell{points_,weights_,p.geometry,p.reference,
      p.points.d_view,p.u0,edges[band],edges[band+1]},result);
  std::array<Real,10> out;for(int i=0;i<10;++i)out[i]=result.sum[i];return out;
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
  const int components=module_->initial_data_identity.empty()?1:2;
  if (finders_.empty()) {
    pin_->GetOrAddInteger("fastflow","lmax",12);
    pin_->GetOrAddInteger("fastflow","ntheta",24);
    sampler_=std::make_unique<HorizonSampler>(pack_,module_);
    for(int h=0;h<components;++h) {
      const std::string suffix="_"+std::to_string(h);
      pin_->GetOrAddInteger("fastflow","flow_iterations"+suffix,300);
      pin_->GetOrAddReal("fastflow","flow_alpha_beta_const"+suffix,.1);
      pin_->GetOrAddReal("fastflow","initial_radius"+suffix,components==1?.02:.009);
      pin_->GetOrAddReal("fastflow","mass_tol"+suffix,1e-10);
      pin_->GetOrAddReal("fastflow","expansion_rms_tol"+suffix,1e-3);
      pin_->GetOrAddReal("fastflow","start_time"+suffix,0);
      pin_->GetOrAddReal("fastflow","stop_time"+suffix,std::numeric_limits<Real>::max());
      if(components==2) {
        // Component centers must be explicit. Never search both at the origin.
        pin_->GetReal("fastflow","center_x"+suffix);
        pin_->GetReal("fastflow","center_y"+suffix);
        pin_->GetReal("fastflow","center_z"+suffix);
      }
      auto finder=std::make_unique<FastFlow>(pack_,pin_,h,
        [](const Real[3],z4c::HorizonGeometry<Real> &) -> bool {
          throw std::logic_error("Native horizon sampling requires the batch callback");
        });
      finder->SetBatchGeometrySampler([this](int n,const Real *x,
          z4c::HorizonGeometry<Real> *out,int *owners) {
        sampler_->SampleBatch(n,x,out,owners);
      });
      finders_.push_back(std::move(finder));
    }
  }
  // Fixed collective order on every rank, including failed/skipped components.
  // GetSurface explicitly clears stale accepted surfaces on an unsuccessful call.
  horizons_.clear();surfaces_.clear();bool complete=true;
  int component=0;
  for(auto &finder:finders_) {
    if(components==2) {
      const auto *center=module_->ComponentCenter(component);
      for(int a=0;a<3;++a)finder->center[a]=center[a];
    }
    ++component;
    finder->Find(pack_->pmesh->ncycle,pack_->pmesh->time);
    horizons_.push_back(finder->GetResult());surfaces_.push_back(finder->GetSurface());
    const auto &result=horizons_.back();
    if(components==2 && global_variable::my_rank==0)
      std::cout<<"binary horizon component="<<component-1
        <<" status="<<static_cast<int>(result.status)<<" complete="<<result.complete_surface
        <<" center="<<finder->center[0]<<','<<finder->center[1]<<','<<finder->center[2]
        <<" area="<<result.area<<" mass_coordinate_spin="<<result.mass
        <<" iterations="<<result.iterations
        <<" expansion_rms="<<result.expansion_rms
        <<" last_trial_expansion_rms="<<result.last_expansion_rms
        <<" acceptance="<<(finder->expansion_rms_tol<0?"mass_only":"mass_and_expansion")
        <<" expansion_above_diagnostic_threshold="
        <<(result.last_expansion_rms>pin_->GetReal("fastflow",
              "expansion_rms_tol_"+std::to_string(component-1)))<<std::endl;
    complete=complete && result.found && result.complete_surface && surfaces_.back().Valid();
  }
  horizon_=horizons_.front();surface_=surfaces_.front();
  RequireAll(complete,"Hyperboloidal exterior requires every component horizon; failed, skipped or missing surface");
  auto mask=Kokkos::create_mirror_view(exterior_);
  for (int q=0;q<module_->npoints;++q) {
    const Real x[3]={positions_(q,0),positions_(q,1),positions_(q,2)};
    mask(q)=1;
    for(const auto &surface:surfaces_)if(!surface.Outside(x))mask(q)=0;
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
    for(auto &finder:finders_)finder->Write(pack_->pmesh->ncycle,pack_->pmesh->time);
    statistics_.io_seconds+=timer.seconds();written_=true;
  }
  // HistoryOutput owns MPI summation and only then forms the eight RMS values.
  return sums_;
}
}  // namespace z4c_hyperboloidal
