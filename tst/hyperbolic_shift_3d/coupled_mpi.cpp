#include <mpi.h>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include "globals.hpp"
#include "parameter_input.hpp"
#include "z4c/fastflow.hpp"
#include "z4c/hyperboloidal/horizon_geometry.hpp"
#include "z4c/hyperboloidal/hyperbolic_composite_exchange.hpp"
#include "z4c/hyperboloidal/hyperbolic_scri_device.hpp"
#include "z4c/hyperboloidal/hyperbolic_angular_dissipation_device.hpp"
#include "z4c/hyperboloidal/hyperbolic_sampled_normal_device.hpp"
#include "z4c/hyperboloidal/algebraic_projection.hpp"
#include "fixed_composite_mesh.hpp"
#include "z4c/hyperboloidal/sparse_stage_exchange.hpp"
#ifndef HYPERBOLIC_SPARSE_STAGE
#define HYPERBOLIC_SPARSE_STAGE 0
#endif
constexpr bool sparse_stage=HYPERBOLIC_SPARSE_STAGE;
#ifndef HYPERBOLIC_ALGEBRAIC_FINAL_PROJECTION
#define HYPERBOLIC_ALGEBRAIC_FINAL_PROJECTION 0
#endif
#ifndef HYPERBOLIC_SCRI_ANGULAR_EXCHANGE
#define HYPERBOLIC_SCRI_ANGULAR_EXCHANGE 0
#endif
#ifndef HYPERBOLIC_SCRI_ANGULAR_DISSIPATION
#define HYPERBOLIC_SCRI_ANGULAR_DISSIPATION 0
#endif
#ifndef HYPERBOLIC_SCRI_SAMPLED_NORMAL
#define HYPERBOLIC_SCRI_SAMPLED_NORMAL 0
#endif
#ifndef HYPERBOLIC_SCRI_DONOR_SEVEN
#define HYPERBOLIC_SCRI_DONOR_SEVEN 0
#endif
#ifndef HYPERBOLIC_INNER_TRANSITION_LOGICAL
#define HYPERBOLIC_INNER_TRANSITION_LOGICAL 0
#endif
#ifndef HYPERBOLIC_CORE_SUBDIVISIONS
#define HYPERBOLIC_CORE_SUBDIVISIONS 1
#endif
#ifndef HYPERBOLIC_EVOLVED_OVERLAP
#define HYPERBOLIC_EVOLVED_OVERLAP 0
#endif
#ifndef HYPERBOLIC_SPHERICAL_OVERLAP
#define HYPERBOLIC_SPHERICAL_OVERLAP 0
#endif
#ifndef HYPERBOLIC_CONTINUOUS_POSITIVE
#define HYPERBOLIC_CONTINUOUS_POSITIVE 0
#endif
#ifndef HYPERBOLIC_BOUNDARY_REUSE
#define HYPERBOLIC_BOUNDARY_REUSE 0
#endif
constexpr bool boundary_reuse=HYPERBOLIC_BOUNDARY_REUSE;
constexpr bool continuous_positive=HYPERBOLIC_CONTINUOUS_POSITIVE;
constexpr bool spherical_overlap=HYPERBOLIC_SPHERICAL_OVERLAP;
constexpr Real physical_lapse_floor=1e-10;
namespace h=z4c::hyperboloidal;
using Buffer=h::HyperbolicCompositeDevice::Buffer;
using HostBuffer=Kokkos::View<Real*,Kokkos::HostSpace>;
using V=std::vector<Real>;
constexpr bool algebraic_final_projection=HYPERBOLIC_ALGEBRAIC_FINAL_PROJECTION;
constexpr bool scri_angular_exchange=HYPERBOLIC_SCRI_ANGULAR_EXCHANGE;
constexpr bool scri_angular_dissipation=HYPERBOLIC_SCRI_ANGULAR_DISSIPATION;
constexpr bool scri_sampled_normal=HYPERBOLIC_SCRI_SAMPLED_NORMAL;
constexpr bool scri_donor_seven=HYPERBOLIC_SCRI_DONOR_SEVEN;
constexpr bool inner_transition_logical=HYPERBOLIC_INNER_TRANSITION_LOGICAL;
constexpr int evolved_overlap=HYPERBOLIC_EVOLVED_OVERLAP;
static_assert(evolved_overlap==0||(evolved_overlap==3&&inner_transition_logical),"overlap experiment requires V7 operators and three evolved layers");
constexpr int core_subdivisions=HYPERBOLIC_CORE_SUBDIVISIONS;
static_assert(!evolved_overlap||core_subdivisions==1||spherical_overlap,"V9 currently binds a single Cartesian core");
static_assert(core_subdivisions==1||(core_subdivisions==2&&inner_transition_logical),"core2 control requires V7 operators");
static_assert(!inner_transition_logical||scri_donor_seven,"inner transition candidate requires V6 scri closure");
static_assert(!scri_angular_exchange||algebraic_final_projection,"angular candidate requires algebraic repair");
static_assert(!scri_angular_dissipation||scri_angular_exchange,"dissipation candidate requires cross-face jets");
static_assert(!scri_sampled_normal||scri_angular_dissipation,"sampled-normal candidate requires V4 geometry/closure");
static_assert(!scri_donor_seven||scri_sampled_normal,"direct radial candidate requires sampled numerator");
static_assert(!spherical_overlap||(evolved_overlap==3&&core_subdivisions==2),"spherical overlap requires eight core blocks and three overlap cells");
static_assert(!continuous_positive||(spherical_overlap&&core_subdivisions==2),"positive scheme requires continuous spherical-overlap mesh");
constexpr const char*checkpoint_magic=continuous_positive?"HYP3D_MPI_V11":spherical_overlap?"HYP3D_MPI_V10":evolved_overlap?"HYP3D_MPI_V9":core_subdivisions==2?"HYP3D_MPI_V8":inner_transition_logical?"HYP3D_MPI_V7":scri_donor_seven?"HYP3D_MPI_V6":scri_sampled_normal?"HYP3D_MPI_V5":scri_angular_dissipation?"HYP3D_MPI_V4":scri_angular_exchange?"HYP3D_MPI_V3":algebraic_final_projection?"HYP3D_MPI_V2":"HYP3D_MPI_V1";
constexpr const char*algebraic_scheme=algebraic_final_projection?"final_stage_standard":"none";
constexpr const char*scri_angular_scheme=scri_angular_exchange?"cross_face_centered5":"face_local6";
constexpr const char*scri_dissipation_scheme=scri_angular_dissipation?"degree7_weighted_negative_square":"none";
struct Header {
  char magic[16]{};int n=0,nf=31,kind=0,admitted=0;
  uint64_t cells=0,surface=0,steps=0;Real time=0,dt=0;
};
// V1--V3 retain their original bytes. V4 binds every restart to the added
// operator strength and degree before reading any of the four state arrays.
struct DissipationConfig { Real epsilon=0;int32_t degree=7,reserved=0; };
// V5 appends an exact normal-factorization identity, never reinterpreting V4.
// V6 binds both the seven-node normal derivative and the last-three-row bulk
// radial closure. V7 uses scheme=3 to bind the logical inner-transition halos;
// V6 and V7 retain the same last-three-row scri closure.
struct NormalConfig { int32_t scheme=inner_transition_logical?3:scri_donor_seven?2:1,radial_nodes=scri_donor_seven?7:6,
  bulk_derivative_order=4,reserved=scri_donor_seven?3:0; };
// V8 appends this fixed-topology identity. Older schemas retain every
// byte. The descriptor excludes arbitrary leaf trees and dynamic regridding.
struct MeshConfig { int32_t scheme=continuous_positive?4:spherical_overlap?3:evolved_overlap?2:1,core_subdivisions=::core_subdivisions,
  shell_layers=3,reserved=evolved_overlap; };
struct PositiveConfig { Real physical_lapse_floor=::physical_lapse_floor;int32_t scheme=1,reserved=0; };
static_assert(sizeof(PositiveConfig)==16,"positive-variable configuration layout");
struct GeometryConfig { Real core_half_width=.18,inner_radius=.16; };
static_assert(sizeof(GeometryConfig)==16,"geometry layout requires float64");
static_assert(sizeof(MeshConfig)==16,"mesh configuration layout requires four int32 values");
static_assert(sizeof(Header)==72&&sizeof(DissipationConfig)==16&&sizeof(NormalConfig)==16,"checkpoint layout requires float64");
class Evolution {
 public:
  int rank,size,n,total,first,last,local,first_block,last_block;
  int surface_total,surface_first,surface_local;
  const std::chrono::steady_clock::time_point construction_started=std::chrono::steady_clock::now();
  double setup_seconds[8]={};
  h::HyperbolicComposite model;
  std::unique_ptr<h::HyperbolicCompositeDevice> kernel;
  std::unique_ptr<h::HyperbolicCompositeExchange> exchange;
  std::unique_ptr<h::HyperbolicCompositeExchange> normal_exchange;
  std::unique_ptr<h::HyperbolicScriDevice> boundary;
  std::unique_ptr<h::HyperbolicSampledNormalDevice> sampled_normal;
  std::unique_ptr<h::HyperbolicAngularDissipationDevice> angular_dissipation;
  Buffer angular_residual;
  std::unique_ptr<h::SparseStageExchange> sparse;
  h::MeshPlanEpoch mesh_epoch{};
  bool sparse_poison=false;
  std::unique_ptr<h::HyperbolicScriDevice> global_projector;
  bool boundary_cached=false;
  Real boundary_cached_dt=0;
  uint64_t boundary_cache_hits=0;
  double collective_seconds[3]={0,0,0};  // volume, surface, angular residual
  uint64_t collective_calls[3]={0,0,0};
  Buffer state,base,local_state,compensation,next_compensation,reference_lapse,failed_state,floor_statistics;
  Buffer surface,surface_base,surface_stage,surface_compensation,next_surface_compensation;
  std::array<Buffer,4> rates,edge_rates;
  V failed_edge,owned_padding;
  std::vector<int> counts,displacements,surface_counts,surface_displacements;
  std::vector<int> residual_counts,residual_displacements;
  Real time=0,gap=1e300,cfl=0,null_ghost=0,null_before=0,last_rhs=0;
  Real angular_epsilon=0,angular_upper_bound=0,angular_cfl=0,angular_cfl_max=0;
  uint64_t steps=0;
  double rhs_seconds=0,exchange_seconds=0,closure_seconds=0,diagnostic_seconds=0,halo_seconds=0;
  explicit Evolution(int resolution,Real epsilon=0,int kind=0):n(resolution),model(continuous_positive?hyperbolic_reproduction::ContinuousCoreMesh(n):spherical_overlap?hyperbolic_reproduction::SphericalOverlapMesh(n):hyperbolic_reproduction::FixedMesh(n,core_subdivisions,evolved_overlap),false,scri_angular_exchange,epsilon,
    scri_sampled_normal?h::HyperbolicComposite::NormalDerivative::sampled_numerator:h::HyperbolicComposite::NormalDerivative::analytic_jets,
    scri_donor_seven?h::HyperbolicComposite::RadialClosure::donor_seven:h::HyperbolicComposite::RadialClosure::ghost_lift),angular_epsilon(epsilon){
    auto setup_tick=construction_started;
    auto setup_phase=[&](int i){const auto now=std::chrono::steady_clock::now();setup_seconds[i]=std::chrono::duration<double>(now-setup_tick).count();setup_tick=now;};
    if constexpr(inner_transition_logical)model.grid.SetInnerTransitionLogical(true);
    model.positive_dissipation=continuous_positive;model.puncture_reference=kind==2;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
    total=model.grid.active.size();first=total*rank/size;last=total*(rank+1)/size;local=last-first;
    surface_total=model.scri.rows.size();surface_first=surface_total*rank/size;surface_local=surface_total*(rank+1)/size-surface_first;
    if(local==0||surface_local==0)throw std::runtime_error("rank has no active cells/surface rows");
    first_block=model.grid.mesh.blocks.size();last_block=0;
    for(int p=first;p<last;++p){const int block=model.grid.cells[p].block;
      first_block=std::min(first_block,block);last_block=std::max(last_block,block+1);}
    counts.resize(size);displacements.resize(size);surface_counts.resize(size);surface_displacements.resize(size);
    for(int r=0;r<size;++r){displacements[r]=total*r/size*31;counts[r]=(total*(r+1)/size-total*r/size)*31;
      surface_displacements[r]=surface_total*r/size*31;surface_counts[r]=(surface_total*(r+1)/size-surface_total*r/size)*31;}
    setup_phase(0);
    kernel=std::make_unique<h::HyperbolicCompositeDevice>(model.grid,first,local,model.radial_plan.get());
    std::vector<int> owned_blocks;
    if constexpr(continuous_positive){std::set<int> blocks;for(int p=first;p<last;++p)blocks.insert(model.grid.cells[p].block);owned_blocks.assign(blocks.begin(),blocks.end());}
    setup_phase(1);
    exchange=std::make_unique<h::HyperbolicCompositeExchange>(model,first_block,last_block,owned_blocks,continuous_positive);
    setup_phase(2);
    boundary=std::make_unique<h::HyperbolicScriDevice>(model,surface_first,surface_local);
    if constexpr(boundary_reuse)global_projector=std::make_unique<h::HyperbolicScriDevice>(model,0,surface_total);
    setup_phase(3);
    if constexpr(scri_sampled_normal){
      // Surface ownership is independent of bulk ownership. Its five radial
      // donors therefore require their own native halo plan, not this rank's
      // unrelated volume-block halos. Both plans read the gathered stage state.
      sampled_normal=std::make_unique<h::HyperbolicSampledNormalDevice>(model,surface_first,surface_local);
      normal_exchange=std::make_unique<h::HyperbolicCompositeExchange>(model,sampled_normal->FirstBlock(),sampled_normal->EndBlock(),std::vector<int>{},continuous_positive,false);
    }
    setup_phase(4);
    if(angular_epsilon>0){
      angular_dissipation=std::make_unique<h::HyperbolicAngularDissipationDevice>(*model.angular_plan,boundary_reuse?0:surface_first,boundary_reuse?surface_total:surface_local);
      angular_upper_bound=model.angular_plan->SpectralUpperBound();
      angular_residual=Buffer("global angular residual",21*surface_total);
      residual_counts.resize(size);residual_displacements.resize(size);
      for(int r=0;r<size;++r){residual_counts[r]=surface_counts[r]/31*21;residual_displacements[r]=surface_displacements[r]/31*21;}
    }
    setup_phase(5);
    if constexpr(sparse_stage){
      std::vector<int> needed=exchange->ActiveDependencies();
      const auto&normal=normal_exchange->ActiveDependencies();needed.insert(needed.end(),normal.begin(),normal.end());
      const auto&scri=boundary->ActiveDependencies();needed.insert(needed.end(),scri.begin(),scri.end());
      std::vector<int> cuts(size+1);for(int r=0;r<=size;++r)cuts[r]=total*r/size;
      sparse=std::make_unique<h::SparseStageExchange>();sparse->Rebuild(mesh_epoch,cuts,31,needed);
      const char*poison=std::getenv("HYPERBOLIC_SPARSE_POISON");sparse_poison=poison&&std::string(poison)=="1";
    }
    setup_phase(6);
    state=Buffer("global active stage",31*total);base=Buffer("accepted stage",31*total);
    local_state=Buffer("local stage",31*local);compensation=Buffer("local RK compensation",31*local);
    next_compensation=Buffer("next RK compensation",31*local);Kokkos::deep_copy(compensation,Real(0));
    surface=Buffer("global surface stage",31*surface_total);surface_base=Buffer("accepted surface stage",31*surface_total);
    surface_stage=Buffer("local surface stage",31*surface_local);surface_compensation=Buffer("surface RK compensation",31*surface_local);
    next_surface_compensation=Buffer("next surface RK compensation",31*surface_local);Kokkos::deep_copy(surface_compensation,Real(0));
    for(int s=0;s<4;++s){rates[s]=Buffer("persistent volume RK rate",31*local);edge_rates[s]=Buffer("persistent surface RK rate",31*surface_local);}
    CopyFromHost(model.scri.values,surface);exchange->surface=surface;
    if constexpr(scri_sampled_normal)normal_exchange->surface=surface;
    reference_lapse=Buffer("reference lapse",local);HostBuffer lapse("reference lapse host",local);
    for(int p=0;p<local;++p){const Real r=model.Radius(model.grid.geometry[first+p].x);lapse(p)=(1+r*r)/2;}
    Kokkos::deep_copy(reference_lapse,lapse);
    floor_statistics=Buffer("physical lapse floor statistics",2);Kokkos::deep_copy(floor_statistics,Real(0));
    setup_phase(7);
  }
  void SynchronizeError(int failed,std::string message){
    int all=0;MPI_Allreduce(&failed,&all,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
    if(all){int owner=failed?rank:size;MPI_Allreduce(MPI_IN_PLACE,&owner,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
      char text[512]{};if(rank==owner)std::strncpy(text,message.c_str(),sizeof(text)-1);
      MPI_Bcast(text,sizeof(text),MPI_CHAR,owner,MPI_COMM_WORLD);throw std::runtime_error(text);}
  }
  static void CopyFromHost(const V&values,Buffer target){
    if(values.size()!=target.extent(0))throw std::runtime_error("device upload size mismatch");
    HostBuffer host("host upload",target.extent(0));for(size_t i=0;i<values.size();++i)host(i)=values[i];Kokkos::deep_copy(target,host);
  }
  void Upload(const V&values){boundary_cached=false;CopyFromHost(values,state);
    if constexpr(sparse_stage){Kokkos::deep_copy(local_state,Kokkos::subview(state,std::make_pair(size_t(31*first),size_t(31*last))));ExchangeStage();}}
  void ExchangeStage(){
    boundary_cached=false;
    if constexpr(sparse_stage){const auto t=sparse->Exchange(local_state,state,mesh_epoch,sparse_poison);
      exchange_seconds+=t.pack+t.mpi+t.unpack;
    }else Gather(local_state,state);
  }
  Buffer CompleteState(Buffer partial){
    if constexpr(!sparse_stage)return partial;
    Buffer complete("complete snapshot state",partial.extent(0));
    Buffer owned=Kokkos::subview(partial,std::make_pair(size_t(31*first),size_t(31*last)));
    Gather(owned,complete);return complete;
  }
  V Download(Buffer data){auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),data);V values(data.extent(0));
    for(size_t i=0;i<values.size();++i)values[i]=host(i);return values;}
  void Gather(Buffer local_values,Buffer global,bool edge=false){
    boundary_cached=false;
    Kokkos::fence();const auto start=std::chrono::steady_clock::now();
    const auto&c=edge?surface_counts:counts;const auto&d=edge?surface_displacements:displacements;
    MPI_Allgatherv(local_values.data(),c[rank],MPI_ATHENA_REAL,global.data(),c.data(),d.data(),MPI_ATHENA_REAL,MPI_COMM_WORLD);
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    exchange_seconds+=seconds;collective_seconds[edge?1:0]+=seconds;++collective_calls[edge?1:0];
  }
  Buffer RefreshBoundary(Real dt){
    // Reuse only an identical gathered state and timestep. Gather, upload and
    // rollback invalidate this cache. The accepted step already computed these
    // same physical rates; no RK stage or projection is omitted after a change.
    if(boundary_reuse&&boundary_cached&&boundary_cached_dt==dt){++boundary_cache_hits;return boundary->LocalRHS();}
    // This is an isolated negative-real-axis RK4 bound with margin. Passing it
    // does not establish stability of the coupled evolution or interpolation.
    if(angular_dissipation){
      angular_cfl=dt*angular_epsilon*n*angular_upper_bound;
      angular_cfl_max=std::max(angular_cfl_max,angular_cfl);
      SynchronizeError(!(dt>0)||!std::isfinite(angular_cfl)||angular_cfl>2.7,"angular dissipation RK4 bound exceeded");
    }
    const auto start=std::chrono::steady_clock::now();const double exchange_before=exchange_seconds;
    const auto projected=boundary_reuse?global_projector->Project(surface):boundary->Project(surface);
    SynchronizeError(projected.bad_rows,"device scri projection failure at row "+std::to_string(projected.first_bad)+" reason "+h::HyperbolicScriDevice::Reason(projected.reason));
    if constexpr(boundary_reuse)Kokkos::deep_copy(surface,global_projector->LocalSurface());
    else Gather(boundary->LocalSurface(),surface,true);
    Buffer pole_samples;
    if constexpr(scri_sampled_normal){
      // Current geometric ghosts precede native interior numerator sampling.
      // The geometric numerator is independent of finite provisional D values.
      const auto halo=normal_exchange->Fill(state);
      null_ghost=std::max(null_ghost,halo.after);null_before=std::max(null_before,halo.before);
      SynchronizeError(halo.invalid,"nonpositive interpolated scri geometry");
      const auto normal=sampled_normal->Evaluate(normal_exchange->padded,surface);
      SynchronizeError(normal.bad_rows,"device sampled-normal numerator failure at row "+std::to_string(normal.first_bad)+" reason "+h::HyperbolicSampledNormalDevice::Reason(normal.reason));
      pole_samples=sampled_normal->LocalPoleSamples();
    }
    Buffer additive;
    if(angular_dissipation){
      const auto residual=angular_dissipation->Residual(surface);
      SynchronizeError(residual.bad_values,"nonfinite angular residual at row "+std::to_string(residual.first_bad));
      Buffer residual_input;
      if constexpr(boundary_reuse)residual_input=angular_dissipation->LocalResidual();
      else{
        Kokkos::fence();const auto gather_start=std::chrono::steady_clock::now();
        MPI_Allgatherv(angular_dissipation->LocalResidual().data(),residual_counts[rank],MPI_ATHENA_REAL,
          angular_residual.data(),residual_counts.data(),residual_displacements.data(),MPI_ATHENA_REAL,MPI_COMM_WORLD);
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-gather_start).count();
        exchange_seconds+=seconds;collective_seconds[2]+=seconds;++collective_calls[2];residual_input=angular_residual;
      }
      const auto applied=angular_dissipation->Apply(residual_input,angular_epsilon);
      SynchronizeError(applied.bad_values,"nonfinite angular rate at row "+std::to_string(applied.first_bad));
      if constexpr(boundary_reuse)additive=Kokkos::subview(angular_dissipation->LocalRHS(),
        std::make_pair(size_t(31*surface_first),size_t(31*(surface_first+surface_local))));
      else additive=angular_dissipation->LocalRHS();
    }
    // Dependent D is reconstructed from geometric rates including Q. Appending
    // Q after this call would violate the differentiated moving-normal closure.
    const auto q=boundary->Evaluate(state,surface,dt,additive,pole_samples);
    SynchronizeError(q.bad_rows,"device scri rate failure at row "+std::to_string(q.first_bad)+" reason "+h::HyperbolicScriDevice::Reason(q.reason));
    Gather(boundary->LocalSurface(),surface,true);
    Real stats[4]={projected.null_after,q.pole,q.clock,q.compatibility};MPI_Allreduce(MPI_IN_PLACE,stats,4,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
    model.null_max=std::max(model.null_max,stats[0]);model.pole_max=std::max(model.pole_max,stats[1]);
    model.clock_max=std::max(model.clock_max,stats[2]);model.compatibility_max=std::max(model.compatibility_max,stats[3]);
    boundary_cached=true;boundary_cached_dt=dt;
    closure_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()-(exchange_seconds-exchange_before);return boundary->LocalRHS();
  }
  Buffer RHS(Real dt,Buffer&edge_rate){
    edge_rate=RefreshBoundary(dt);const auto halo_start=std::chrono::steady_clock::now();auto halo=exchange->Fill(state);
    halo_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-halo_start).count();
    null_ghost=std::max(null_ghost,halo.after);null_before=std::max(null_before,halo.before);
    SynchronizeError(halo.invalid,"nonpositive interpolated scri geometry");
    const auto start=std::chrono::steady_clock::now();const auto stats=kernel->Evaluate(exchange->padded,dt,false,.5,surface,exchange->positive_ko);
    rhs_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    gap=std::min(gap,stats.min_gap);cfl=std::max(cfl,stats.max_cfl);
    SynchronizeError(stats.bad_cells,"device RHS failure at active cell "+std::to_string(stats.first_bad)+" reason "+std::to_string(stats.reason));
    const auto rate=kernel->LocalRHS();Real maximum=0;
    Kokkos::parallel_reduce("RHS maximum",Kokkos::RangePolicy<>(0,31*local),KOKKOS_LAMBDA(int i,Real&v){v=fmax(v,fabs(rate(i)));},Kokkos::Max<Real>(maximum));
    last_rhs=maximum;MPI_Allreduce(MPI_IN_PLACE,&last_rhs,1,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);return rate;
  }
  // Repair owned accepted point tensors before rebuilding derivatives. Surface
  // null values and dependent D are recomputed by the following RefreshBoundary.
  // Interpolated ghost tensors are not separately projected. Independent Lambda
  // is retained; this is the standard algebraic repair, not a geometry-preserving
  // redefinition of the conformal variables. Invalid candidates are never floored.
  int ProjectAccepted(Buffer values,Buffer residual,int points){
    int invalid=0;
    if constexpr(algebraic_final_projection){
      Kokkos::parallel_reduce("accepted algebraic tensor projection",Kokkos::RangePolicy<>(0,points),
        KOKKOS_LAMBDA(int p,int&bad){
          const auto result=h::ProjectAlgebraicNativeState(&values(31*p),&residual(31*p));
          if(!result.valid)++bad;
        },invalid);
    }
    return invalid;
  }
  void Step(Real dt){
    Kokkos::deep_copy(base,state);Kokkos::deep_copy(surface_base,surface);
    auto stage=[&](int s,Real h){const auto out=local_state,begin=base,k=rates[s],A=reference_lapse,fs=floor_statistics;const int offset=31*first;
      Kokkos::parallel_for("local RK stage",Kokkos::RangePolicy<>(0,31*local),KOKKOS_LAMBDA(int i){
        const int f=i%31,p=i/31;
        if(continuous_positive&&(f==0||f==18)){bool floored=false;
          out(i)=z4c::hyperboloidal::MaterializePositive(log(begin(offset+i)+(f==18?A(p):0))+h*k(i),A(p),f,physical_lapse_floor,floored);
          if(floored){Kokkos::atomic_add(&fs(0),Real(1));Kokkos::atomic_max(&fs(1),sqrt(2*A(p)-1));}
        }else out(i)=begin(offset+i)+h*k(i);
      });ExchangeStage();
      const auto es=surface_stage,eb=surface_base,ek=edge_rates[s];const int eo=31*surface_first;
      Kokkos::parallel_for("local surface RK stage",Kokkos::RangePolicy<>(0,31*surface_local),KOKKOS_LAMBDA(int i){es(i)=continuous_positive&&i%31==0?exp(log(eb(eo+i))+h*ek(i)):eb(eo+i)+h*ek(i);});Gather(es,surface,true);};
    try{
      for(int s=0;s<4;++s){Buffer edge;const auto rhs=RHS(dt,edge);
        if constexpr(continuous_positive){const auto k=rates[s],ek=edge_rates[s],v=state,sv=surface,A=reference_lapse;const int begin=first,eb=surface_first;
          Kokkos::parallel_for("positive volume RK rates",Kokkos::RangePolicy<>(0,31*local),KOKKOS_LAMBDA(int i){
            const int f=i%31,p=i/31;k(i)=rhs(i)/((f==0||f==18)?v(31*begin+i)+(f==18?A(p):0):1);});
          Kokkos::parallel_for("positive surface RK rates",Kokkos::RangePolicy<>(0,31*surface_local),KOKKOS_LAMBDA(int i){ek(i)=edge(i)/(i%31==0?sv(31*eb+i):1);});
        }else{Kokkos::deep_copy(rates[s],rhs);Kokkos::deep_copy(edge_rates[s],edge);}
        if(s<3)stage(s,s<2?dt/2:dt);}
      const auto out=local_state,begin=base,old=compensation,next=next_compensation,A=reference_lapse,fs=floor_statistics;const int offset=31*first;
      const auto k1=rates[0],k2=rates[1],k3=rates[2],k4=rates[3];
      Kokkos::parallel_for("compensated RK acceptance",Kokkos::RangePolicy<>(0,31*local),KOKKOS_LAMBDA(int i){
        const Real increment=dt*(k1(i)+2*k2(i)+2*k3(i)+k4(i))/6-old(i);
        const int f=i%31,p=i/31;
        if(continuous_positive&&(f==0||f==18)){const Real base_log=log(begin(offset+i)+(f==18?A(p):0)),candidate=base_log+increment;bool floored=false;
          out(i)=h::MaterializePositive(candidate,A(p),f,physical_lapse_floor,floored);next(i)=floored?0:(candidate-base_log)-increment;
          if(floored){Kokkos::atomic_add(&fs(0),Real(1));Kokkos::atomic_max(&fs(1),sqrt(2*A(p)-1));}
        }else{out(i)=begin(offset+i)+increment;next(i)=(out(i)-begin(offset+i))-increment;}
      });
      const int volume_projection_bad=ProjectAccepted(out,next,local);ExchangeStage();
      if constexpr(algebraic_final_projection)SynchronizeError(volume_projection_bad,"invalid accepted volume algebraic projection");
      const auto es=surface_stage,eb=surface_base,ec=surface_compensation,en=next_surface_compensation;const int eo=31*surface_first;
      const auto a=edge_rates[0],b=edge_rates[1],c=edge_rates[2],d=edge_rates[3];
      Kokkos::parallel_for("compensated surface RK acceptance",Kokkos::RangePolicy<>(0,31*surface_local),KOKKOS_LAMBDA(int i){
        const Real increment=dt*(a(i)+2*b(i)+2*c(i)+d(i))/6-ec(i);
        const int f=i%31;
        if(continuous_positive&&f==0){const Real base_log=log(eb(eo+i)),candidate=base_log+increment;es(i)=exp(candidate);en(i)=(candidate-base_log)-increment;}
        else{es(i)=eb(eo+i)+increment;en(i)=(f==18||f>=22)?0:(es(i)-eb(eo+i))-increment;}
      });
      const int surface_projection_bad=ProjectAccepted(es,en,surface_local);Gather(es,surface,true);
      if constexpr(algebraic_final_projection)SynchronizeError(surface_projection_bad,"invalid accepted surface algebraic projection");
      int invalid=0;const auto current=state,alpha0=reference_lapse;const int f=first;
      Kokkos::parallel_reduce("validate accepted geometry",Kokkos::RangePolicy<>(0,local),KOKKOS_LAMBDA(int p,int&bad){
        const int i=31*(f+p);const Real A=1+current(i+1),B=current(i+2),C=current(i+3),D=1+current(i+4),E=current(i+5),F=1+current(i+6);
        const Real det=A*(D*F-E*E)-B*(B*F-C*E)+C*(B*E-C*D);
        bool valid=current(i)>0&&alpha0(p)+current(i+18)>0&&A>0&&A*D-B*B>0&&det>0;
        for(int j=0;j<31;++j)valid=valid&&Kokkos::isfinite(current(i+j));if(!valid)++bad;
      },invalid);SynchronizeError(invalid,"invalid accepted volume geometry");
      RefreshBoundary(dt);Kokkos::deep_copy(compensation,next_compensation);Kokkos::deep_copy(surface_compensation,next_surface_compensation);time+=dt;++steps;
    }catch(...){
      failed_state=Buffer("failed stage",state.extent(0));Kokkos::deep_copy(failed_state,state);if(rank==0)failed_edge=Download(surface);
      boundary_cached=false;Kokkos::deep_copy(state,base);Kokkos::deep_copy(surface,surface_base);throw;
    }
  }
  void WriteCheckpoint(const std::string&path,int kind,Real dt,bool admitted){
    Buffer comp_global("global compensation",state.extent(0)),edge_comp_global("global surface compensation",surface.extent(0));
    const auto complete=CompleteState(state);
    Gather(compensation,comp_global);Gather(surface_compensation,edge_comp_global,true);int bad=0;std::string reason;
    if(rank==0)try{Header header;std::memcpy(header.magic,checkpoint_magic,std::strlen(checkpoint_magic)+1);header.n=n;header.kind=kind;header.admitted=admitted;
      header.cells=total;header.surface=surface.extent(0);header.steps=steps;header.time=time;header.dt=dt;
      const auto values=Download(complete),comps=Download(comp_global),edge=Download(surface),edge_comps=Download(edge_comp_global);std::ofstream out(path+".tmp",std::ios::binary);
      out.write(reinterpret_cast<const char*>(&header),sizeof(header));
      if constexpr(scri_angular_dissipation){const DissipationConfig config{angular_epsilon,7,0};
        out.write(reinterpret_cast<const char*>(&config),sizeof(config));}
      if constexpr(scri_sampled_normal){const NormalConfig config;
        out.write(reinterpret_cast<const char*>(&config),sizeof(config));}
      if constexpr(core_subdivisions==2||evolved_overlap){const MeshConfig config;
        out.write(reinterpret_cast<const char*>(&config),sizeof(config));}
      if constexpr(spherical_overlap){const GeometryConfig config;
        out.write(reinterpret_cast<const char*>(&config),sizeof(config));}
      if constexpr(continuous_positive){const PositiveConfig config;out.write(reinterpret_cast<const char*>(&config),sizeof(config));}
      for(const V*v:{&values,&comps,&edge,&edge_comps})out.write(reinterpret_cast<const char*>(v->data()),v->size()*sizeof(Real));
      out.close();if(!out)throw std::runtime_error("checkpoint write failed");std::filesystem::rename(path+".tmp",path);
    }catch(const std::exception&e){bad=1;reason=e.what();}SynchronizeError(bad,reason);
  }
  void LoadCheckpoint(const std::string&path,int kind,Real dt){
    Header header{};V values(31*total),comps(31*total),edge(31*surface_total),edge_comps(31*surface_total);int bad=0;std::string reason;
    if(rank==0)try{std::ifstream in(path,std::ios::binary);in.read(reinterpret_cast<char*>(&header),sizeof(header));
      if(!in||std::memcmp(header.magic,checkpoint_magic,std::strlen(checkpoint_magic)+1)||header.n!=n||header.nf!=31||header.kind!=kind||header.admitted!=1||header.cells!=size_t(total)||header.surface!=surface.extent(0)||header.dt!=dt)
        throw std::runtime_error("checkpoint schema/configuration/admission mismatch");
      if constexpr(scri_angular_dissipation){DissipationConfig config;
        in.read(reinterpret_cast<char*>(&config),sizeof(config));
        if(!in||!std::isfinite(config.epsilon)||config.epsilon!=angular_epsilon||config.degree!=7||config.reserved!=0)
          throw std::runtime_error("checkpoint angular dissipation configuration mismatch");}
      if constexpr(scri_sampled_normal){NormalConfig config;
        in.read(reinterpret_cast<char*>(&config),sizeof(config));
        const NormalConfig expected;
        if(!in||config.scheme!=expected.scheme||config.radial_nodes!=expected.radial_nodes||
           config.bulk_derivative_order!=expected.bulk_derivative_order||config.reserved!=expected.reserved)
          throw std::runtime_error("checkpoint sampled-normal configuration mismatch");}
      if constexpr(core_subdivisions==2||evolved_overlap){MeshConfig config;
        in.read(reinterpret_cast<char*>(&config),sizeof(config));
        const MeshConfig expected;
        if(!in||config.scheme!=expected.scheme||config.core_subdivisions!=expected.core_subdivisions||config.shell_layers!=3||config.reserved!=expected.reserved)
          throw std::runtime_error("checkpoint fixed-core topology mismatch");}
      if constexpr(spherical_overlap){GeometryConfig config;
        in.read(reinterpret_cast<char*>(&config),sizeof(config));const GeometryConfig expected;
        if(!in||config.core_half_width!=expected.core_half_width||config.inner_radius!=expected.inner_radius)
          throw std::runtime_error("checkpoint spherical geometry mismatch");}
      if constexpr(continuous_positive){PositiveConfig config;in.read(reinterpret_cast<char*>(&config),sizeof(config));const PositiveConfig expected;
        if(!in||config.physical_lapse_floor!=expected.physical_lapse_floor||config.scheme!=1||config.reserved!=0)
          throw std::runtime_error("checkpoint positive-variable configuration mismatch");}
      for(auto*v:{&values,&comps,&edge,&edge_comps})in.read(reinterpret_cast<char*>(v->data()),v->size()*sizeof(Real));
      char tail;if(!in||in.read(&tail,1))throw std::runtime_error("checkpoint truncated or trailing data");
      for(const auto*v:{&values,&comps,&edge,&edge_comps})for(Real x:*v)if(!std::isfinite(x))throw std::runtime_error("checkpoint nonfinite data");
    }catch(const std::exception&e){bad=1;reason=e.what();}
    SynchronizeError(bad,reason);MPI_Bcast(&header,sizeof(header),MPI_BYTE,0,MPI_COMM_WORLD);
    for(auto*v:{&values,&comps,&edge,&edge_comps})MPI_Bcast(v->data(),v->size(),MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
    Upload(values);CopyFromHost(edge,surface);CopyFromHost(V(comps.begin()+31*first,comps.begin()+31*last),compensation);
    CopyFromHost(V(edge_comps.begin()+31*surface_first,edge_comps.begin()+31*(surface_first+surface_local)),surface_compensation);time=header.time;steps=header.steps;
  }
  void DownloadOwnedPadded(){
    if(owned_padding.empty())owned_padding.resize(31*model.grid.count,0);
    const int width=model.grid.padded,blocksize=width*width*width;
    const int begin=first_block*blocksize,end=last_block*blocksize;HostBuffer host("owned padded snapshot",end-begin);
    for(int f=0;f<31;++f){auto piece=Kokkos::subview(exchange->padded,std::make_pair(size_t(f*model.grid.count+begin),size_t(f*model.grid.count+end)));
      Kokkos::deep_copy(host,piece);for(int i=begin;i<end;++i)owned_padding[f*model.grid.count+i]=host(i-begin);}
  }
};

using Diagnostic=h::HyperbolicCompositeDevice::DiagnosticSummary;

// Run local work between MPI phases. A rank-local I/O/device exception must be
// made collective before any rank enters the next communication operation.
template<class F>
void CollectiveLocal(F&& action) {
  int failed=0;std::string message;
  try { action(); } catch(const std::exception& ex) { failed=1;message=ex.what(); }
  int any=0;MPI_Allreduce(&failed,&any,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
  if(!any)return;
  int owner=failed?global_variable::my_rank:global_variable::nranks;
  MPI_Allreduce(MPI_IN_PLACE,&owner,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
  char text[512]{};
  if(global_variable::my_rank==owner)std::strncpy(text,message.c_str(),sizeof(text)-1);
  MPI_Bcast(text,sizeof(text),MPI_CHAR,owner,MPI_COMM_WORLD);
  throw std::runtime_error(text);
}

Diagnostic ReduceDiagnostic(Diagnostic q){
  static_assert(sizeof(q.total)==9*sizeof(Real),"diagnostic reduction layout");
  MPI_Allreduce(MPI_IN_PLACE,&q.total,9,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&q.exterior,9,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&q.min_chi,1,MPI_ATHENA_REAL,MPI_MIN,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&q.min_lapse,1,MPI_ATHENA_REAL,MPI_MIN,MPI_COMM_WORLD);
  for(Real*v:{&q.max_determinant_residual,&q.max_tracefree_residual,&q.max_state_deviation})
    MPI_Allreduce(MPI_IN_PLACE,v,1,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&q.invalid_cells,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  return q;
}
Real Norm(Real sum,Real volume){return volume>0?std::sqrt(sum/volume):std::numeric_limits<Real>::quiet_NaN();}
std::string JsonNumber(Real value){if(!std::isfinite(value))return "null";std::ostringstream s;s<<std::setprecision(17)<<value;return s.str();}
std::string JsonString(const std::string&value){std::string out="\"";for(char c:value){if(c=='"'||c=='\\')out+='\\';if(c=='\n')out+="\\n";else if(static_cast<unsigned char>(c)>=32)out+=c;}return out+'"';}

int main(int argc,char**argv){
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);
  Kokkos::initialize(argc,argv);int code=0;
  try{
    if(const char*expected=std::getenv("HYPERBOLIC_EXPECT_MPI_SIZE"))
      if(global_variable::nranks!=std::stoi(expected))throw std::runtime_error("MPI rank count differs from the bound launch plan");
    if(argc<9||argc>(scri_angular_dissipation?14:13))throw std::invalid_argument("usage: mpi n output kind target dt initial|- restart|- wall_seconds [diagnostic_interval=.1] [horizon_rms_tol=.001] [horizon_lmax=6] [horizon_ntheta=12] [angular_epsilon=2 (V4/V5 only)]");
    const int n=std::stoi(argv[1]);const std::string output=argv[2],kind_name=argv[3],initial=argv[6],restart=argv[7];
    const int kind=kind_name=="minkowski"?0:kind_name=="pulse"?1:kind_name=="puncture"?2:-1;
    const Real end=std::stod(argv[4]),dt=std::stod(argv[5]),wall=std::stod(argv[8]);
    const Real interval=argc>9?std::stod(argv[9]):.1,horizon_tolerance=argc>10?std::stod(argv[10]):.001;
    const auto angular_integer=[](const char*text){
      const std::string value(text);
      if(value.empty()||value.find_first_not_of("0123456789")!=std::string::npos)
        throw std::invalid_argument("horizon angular resolution requires decimal integers");
      return std::stoi(value);
    };
    const int horizon_lmax=argc>11?angular_integer(argv[11]):6;
    const int horizon_ntheta=argc>12?angular_integer(argv[12]):12;
    Real angular_epsilon=scri_angular_dissipation?2:0;
    if(argc>13){size_t used=0;angular_epsilon=std::stod(argv[13],&used);
      if(used!=std::strlen(argv[13]))throw std::invalid_argument("invalid angular dissipation strength");}
    if(!std::isfinite(angular_epsilon)||angular_epsilon<0)
      throw std::invalid_argument("angular dissipation strength must be finite and nonnegative");
    // GaussLegendreGrid stores 2*ntheta*ntheta in an int.
    if(horizon_lmax<1||horizon_ntheta<=horizon_lmax||horizon_ntheta>32767)
      throw std::invalid_argument("horizon angular resolution requires 1 <= lmax < ntheta <= 32767");
    if(n<6||n%2||kind<0||!(dt>0&&end>=0&&wall>0&&interval>0&&horizon_tolerance>0)
       ||!std::isfinite(dt)||!std::isfinite(end)||!std::isfinite(wall)
       ||!std::isfinite(interval)||!std::isfinite(horizon_tolerance))
      throw std::invalid_argument("invalid resolution/kind/time parameters");
    if(kind==2&&initial=="-"&&restart=="-")throw std::invalid_argument("puncture requires initial data or a bound checkpoint");
    if(initial!="-"&&restart!="-")throw std::invalid_argument("supply initial data or restart, not both");
    const auto started=std::chrono::steady_clock::now();
    std::unique_ptr<Evolution> evolution;
    CollectiveLocal([&]{evolution=std::make_unique<Evolution>(n,angular_epsilon,kind);});
    auto&e=*evolution;
    std::string error,status="completed";
    CollectiveLocal([&]{if(e.rank==0){
      const auto parent=std::filesystem::path(output).parent_path();
      if(!parent.empty())std::filesystem::create_directories(parent);
      for(const char*suffix:{".report.json",".report.json.tmp",".checkpoint",".checkpoint.tmp",
                            ".history.csv",".fastflow.horizon_summary_0.txt",
                            ".fastflow.horizon_shape_0.txt",".checkpoint.horizon-seed.txt"})
        if(std::filesystem::exists(output+suffix))throw std::runtime_error("refusing to overwrite result: "+output+suffix);
      if(initial!="-"){std::ifstream in(initial);size_t count=0;in>>count;
        if(count!=e.model.state.size())throw std::runtime_error("initial field count mismatch");
        for(auto&x:e.model.state)if(!(in>>x)||!std::isfinite(x))throw std::runtime_error("invalid initial state");
        std::string trailing;if(in>>trailing)throw std::runtime_error("trailing initial data");}
      else if(kind==1)for(int p=0;p<e.total;++p){const Real z=(e.model.Radius(e.model.grid.geometry[p].x)-.4)/.12;
        if(std::abs(z)<1)e.model.state[31*p+18]=.01*std::exp(1-1/(1-z*z));}
    }});
    MPI_Bcast(e.model.state.data(),e.model.state.size(),MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
    CollectiveLocal([&]{e.Upload(e.model.state);});
    if(restart!="-")e.LoadCheckpoint(restart,kind,dt);
    if(e.time>end+1e-14)throw std::runtime_error("checkpoint later than requested end");
    const Real start_time=e.time;const uint64_t start_steps=e.steps;
    const Real nan=std::numeric_limits<Real>::quiet_NaN();
    Real seed_radius=.07;int seed_loaded=0;
    // This optional sidecar is a search hint only. The volume/shell checkpoint
    // remains authoritative, and older checkpoints simply use the default hint.
    CollectiveLocal([&]{if(e.rank==0&&restart!="-"){
      std::ifstream seed(restart+".horizon-seed.txt");std::string tag,tail;
      Real checkpoint_time=0,measurement_time=0,radius=0;uint64_t checkpoint_steps=0;
      if(seed>>tag>>checkpoint_time>>checkpoint_steps>>measurement_time>>radius){
        if(tag=="HYP3D_FASTFLOW_SEED_V1"&&checkpoint_time==e.time&&checkpoint_steps==e.steps
           &&std::isfinite(measurement_time)&&measurement_time<=e.time&&measurement_time>=0
           &&std::isfinite(radius)&&radius>0&&radius<1&&!(seed>>tail)){
          seed_radius=radius;seed_loaded=1;
        }
      }
    }});
    MPI_Bcast(&seed_radius,1,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
    MPI_Bcast(&seed_loaded,1,MPI_INT,0,MPI_COMM_WORLD);
    ParameterInput pin;std::unique_ptr<FastFlow> finder;
    CollectiveLocal([&]{
      std::ostringstream input;
      input<<std::setprecision(17)<<"<job>\nbasename="<<output<<".fastflow\n<fastflow>\nntheta="<<horizon_ntheta<<"\nlmax="<<horizon_lmax<<"\nflow=standard\nflow_iterations_0=300\nflow_alpha_beta_const_0=.1\ninitial_radius_0="<<seed_radius<<"\nmass_tol_0=1e-10\nexpansion_rms_tol_0="<<horizon_tolerance<<"\nhmean_tol_0=1e8\nstart_time_0=0\nstop_time_0="<<end+1<<"\n";
      std::istringstream stream(input.str());pin.LoadFromStream(stream);
      if(kind==2)finder=std::make_unique<FastFlow>(nullptr,&pin,0,[&](const Real x[3],z4c::HorizonGeometry<Real>&q){
        const int block=e.model.grid.mesh.Owner(x);if(block<0)return false;
        // Split blocks have one sampler owner, selected by their central active
        // cell. Each participating rank stores that block's complete filled halo.
        const int center=block*n*n*n+n*n*n/2;
        if(center<e.first||center>=e.last)return false;
        try{V v;std::vector<std::array<Real,3>>d;std::vector<std::array<std::array<Real,3>,3>>dd;
          e.model.grid.SampleJets(e.owned_padding,31,x,v,d,dd,true);
          const auto u=e.model.Jet(x,v,d,dd);q=h::PhysicalHorizonGeometry(u,e.model.Omega(x,u));
        }catch(const std::exception&){q={};}return true;
      });
    });
    std::ofstream history;
    CollectiveLocal([&]{if(e.rank==0){
      history.open(output+".history.csv");history<<std::setprecision(17)
        <<"t,t_over_M,steps,min_chi,min_lapse,state_deviation,H_L2,M_L2,M_physical_L2,Theta_L2,Z_L2,H_exterior_L2,M_exterior_L2,exterior_cutoff,horizon_found,horizon_radius,horizon_areal_radius,horizon_mass,horizon_expansion_rms,horizon_trial_radius,horizon_trial_expansion_rms,horizon_complete_surface,horizon_iterations,null_surface,null_ghost,pole_max,clock_max,cfl_max,wall_seconds,determinant_residual,tracefree_residual,Mr_L2,Mr_exterior_L2,Theta_exterior_L2,Zr_L2,Zr_exterior_L2,horizon_irreducible_mass\n";
      history.flush();if(!history)throw std::runtime_error("history open/header write failed");
    }});
    Diagnostic diagnostic{};bool diagnostic_valid=false;Real diagnostic_time=nan;
    Real horizon_radius=nan,horizon_areal=nan,horizon_mass=nan,horizon_irreducible_mass=nan,horizon_expansion=nan;
    Real trial_radius=nan,trial_expansion=nan,cutoff=kind==2?nan:0;
    bool horizon_found=false,horizon_complete=false;int horizon_iterations=0;
    Real initial_rhs=nan;double last_diag_seconds=0;
    auto elapsed=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();};
    auto record=[&]{
      const auto begin=std::chrono::steady_clock::now();const double counted=e.closure_seconds+e.exchange_seconds;
      e.RefreshBoundary(dt);h::HyperbolicCompositeExchange::Summary halo{};
      CollectiveLocal([&]{halo=e.exchange->Fill(e.state);});
      e.null_ghost=std::max(e.null_ghost,halo.after);e.null_before=std::max(e.null_before,halo.before);
      e.SynchronizeError(halo.invalid,"invalid diagnostic halo");
      cutoff=kind==2?nan:0;horizon_found=false;
      horizon_radius=horizon_areal=horizon_mass=horizon_irreducible_mass=horizon_expansion=nan;
      if(finder){
        CollectiveLocal([&]{e.DownloadOwnedPadded();});
        finder->Find(e.steps,e.time);
        CollectiveLocal([&]{finder->Write(e.steps,e.time);});
        const auto result=finder->GetResult();
        horizon_found=result.found;horizon_complete=result.complete_surface;horizon_iterations=result.iterations;
        trial_expansion=result.last_expansion_rms;trial_radius=result.last_mean_radius;
        if(result.found){horizon_radius=result.mean_radius;horizon_areal=result.areal_radius;
          horizon_mass=result.mass;horizon_irreducible_mass=result.irreducible_mass;
          horizon_expansion=result.expansion_rms;cutoff=result.mean_radius;}
      }
      // Failure to locate a horizon is not a volume-evolution failure. Still
      // measure the full domain; with no reliable cutoff, exterior norms are NaN.
      Diagnostic next{};
      CollectiveLocal([&]{next=e.kernel->Diagnostics(e.exchange->padded,std::isfinite(cutoff)?cutoff:0,e.surface);});
      next=ReduceDiagnostic(next);
      e.SynchronizeError(next.invalid_cells,"invalid diagnostic volume geometry");
      if(!std::isfinite(cutoff))next.exterior=h::HyperbolicCompositeDevice::DiagnosticSums{};
      diagnostic=next;diagnostic_valid=true;diagnostic_time=e.time;
      MPI_Allreduce(MPI_IN_PLACE,&e.null_ghost,1,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE,&e.cfl,1,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
      CollectiveLocal([&]{if(e.rank==0){
        history<<e.time<<','<<(kind==2?e.time*3:e.time)<<','<<e.steps<<','<<diagnostic.min_chi<<','<<diagnostic.min_lapse<<','<<diagnostic.max_state_deviation
          <<','<<Norm(diagnostic.total.h2,diagnostic.total.volume)<<','<<Norm(diagnostic.total.m2,diagnostic.total.volume)<<','<<Norm(diagnostic.total.m2_physical,diagnostic.total.volume)
          <<','<<Norm(diagnostic.total.theta2,diagnostic.total.volume)<<','<<Norm(diagnostic.total.z2,diagnostic.total.volume)
          <<','<<Norm(diagnostic.exterior.h2,diagnostic.exterior.volume)<<','<<Norm(diagnostic.exterior.m2,diagnostic.exterior.volume)<<','<<cutoff
          <<','<<horizon_found<<','<<horizon_radius<<','<<horizon_areal<<','<<horizon_mass<<','<<horizon_expansion
          <<','<<trial_radius<<','<<trial_expansion<<','<<horizon_complete<<','<<horizon_iterations
          <<','<<e.model.null_max<<','<<e.null_ghost<<','<<e.model.pole_max<<','<<e.model.clock_max<<','<<e.cfl<<','<<elapsed()
          <<','<<diagnostic.max_determinant_residual<<','<<diagnostic.max_tracefree_residual
          <<','<<Norm(diagnostic.total.mr2,diagnostic.total.volume)<<','<<Norm(diagnostic.exterior.mr2,diagnostic.exterior.volume)
          <<','<<Norm(diagnostic.exterior.theta2,diagnostic.exterior.volume)
          <<','<<Norm(diagnostic.total.zr2,diagnostic.total.volume)<<','<<Norm(diagnostic.exterior.zr2,diagnostic.exterior.volume)
          <<','<<horizon_irreducible_mass<<'\n';
        history.flush();if(!history)throw std::runtime_error("history append failed");
        std::cout<<"t="<<e.time<<" steps="<<e.steps<<" min_alpha="<<diagnostic.min_lapse<<" H="<<Norm(diagnostic.total.h2,diagnostic.total.volume)<<" horizon="<<horizon_found<<" seconds="<<elapsed()<<std::endl;
      }});
      const double local_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
      e.diagnostic_seconds+=local_seconds-(e.closure_seconds+e.exchange_seconds-counted);last_diag_seconds=local_seconds;
      MPI_Allreduce(MPI_IN_PLACE,&last_diag_seconds,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    };
    Real last_recorded=-1;
    try{
      Buffer edge;e.RHS(dt,edge);initial_rhs=e.last_rhs;record();last_recorded=e.time;
      Real next_record=e.time+interval;double step_seconds=0;
      while(e.time<end-1e-14){
        double used=elapsed();MPI_Allreduce(MPI_IN_PLACE,&used,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
        // Every term in this branch must be identical on every rank.
        if(used+std::max(15.,last_diag_seconds*1.5)+2*step_seconds>=wall){status="checkpointed";break;}
        const auto tick=std::chrono::steady_clock::now();e.Step(std::min(dt,end-e.time));
        step_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-tick).count();
        MPI_Allreduce(MPI_IN_PLACE,&step_seconds,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
        if(e.time>=next_record-1e-14){record();last_recorded=e.time;next_record=e.time+interval;}
      }
      if(e.time!=last_recorded)record();
    }catch(const std::exception&x){status="failed";error=x.what();code=2;}
    bool checkpoint_written=false,checkpoint_admitted=status!="failed";
    try{
      e.WriteCheckpoint(output+".checkpoint",kind,dt,checkpoint_admitted);checkpoint_written=true;
      CollectiveLocal([&]{if(e.rank==0&&horizon_found&&std::isfinite(horizon_radius)){
        const std::string path=output+".checkpoint.horizon-seed.txt";
        std::ofstream seed(path+".tmp");seed<<std::setprecision(17)<<"HYP3D_FASTFLOW_SEED_V1 "
          <<e.time<<' '<<e.steps<<' '<<diagnostic_time<<' '<<horizon_radius<<'\n';
        seed.close();if(!seed)throw std::runtime_error("horizon seed write failed");
        std::filesystem::rename(path+".tmp",path);
      }});
    }catch(const std::exception&x){status="failed";error+=(error.empty()?"":"; ")+std::string(x.what());code=2;}
    MPI_Allreduce(MPI_IN_PLACE,&e.gap,1,MPI_ATHENA_REAL,MPI_MIN,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&e.null_before,1,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
    Real floor_totals[2]={0,0};
    if constexpr(continuous_positive){auto f=e.Download(e.floor_statistics);floor_totals[0]=f[0];floor_totals[1]=f[1];
      MPI_Allreduce(MPI_IN_PLACE,floor_totals,1,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE,floor_totals+1,1,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);}
    const Buffer complete_failed=e.failed_state.extent(0)?e.CompleteState(e.failed_state):Buffer{};
    double timing[5]={e.rhs_seconds,e.exchange_seconds,e.closure_seconds,e.diagnostic_seconds,e.halo_seconds};
    std::vector<double> rank_collectives(3*e.size);std::vector<uint64_t> rank_collective_calls(3*e.size);
    MPI_Gather(e.collective_seconds,3,MPI_DOUBLE,rank_collectives.data(),3,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Gather(e.collective_calls,3,MPI_UINT64_T,rank_collective_calls.data(),3,MPI_UINT64_T,0,MPI_COMM_WORLD);
    std::vector<double> rank_setup(8*e.size);MPI_Gather(e.setup_seconds,8,MPI_DOUBLE,rank_setup.data(),8,MPI_DOUBLE,0,MPI_COMM_WORLD);
    double sparse_data[7]={};
    if(e.sparse){const auto t=e.sparse->Seconds();sparse_data[0]=t.pack;sparse_data[1]=t.mpi;sparse_data[2]=t.unpack;
      sparse_data[3]=e.sparse->SendPoints();sparse_data[4]=e.sparse->ReceivePoints();sparse_data[5]=e.sparse->Calls();sparse_data[6]=e.sparse->Builds();}
    std::vector<double> rank_sparse(7*e.size);MPI_Gather(sparse_data,7,MPI_DOUBLE,rank_sparse.data(),7,MPI_DOUBLE,0,MPI_COMM_WORLD);
    std::vector<double> rank_timings(5*e.size);MPI_Gather(timing,5,MPI_DOUBLE,rank_timings.data(),5,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,timing,5,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    CollectiveLocal([&]{if(e.rank==0){
      if(e.failed_state.extent(0)){
        const auto failed=e.Download(complete_failed);std::ofstream capture(output+".failed-stage.bin",std::ios::binary);
        capture.write(reinterpret_cast<const char*>(failed.data()),failed.size()*sizeof(Real));
        capture.write(reinterpret_cast<const char*>(e.failed_edge.data()),e.failed_edge.size()*sizeof(Real));
        capture.close();if(!capture)throw std::runtime_error("failed-stage capture write failed");
      }
      const auto valid_number=[&](Real value){return JsonNumber(diagnostic_valid?value:nan);};
      std::ofstream report(output+".report.json.tmp");report<<std::setprecision(17)
        <<"{\"status\":"<<JsonString(status)<<",\"error\":"<<JsonString(error)<<",\"kind\":"<<JsonString(kind_name)
        <<",\"backend\":"<<JsonString(Kokkos::DefaultExecutionSpace::name())<<",\"mpi_size\":"<<e.size<<",\"n\":"<<n<<",\"cells\":"<<e.total
        <<",\"algebraic_projection\":"<<JsonString(algebraic_scheme)<<",\"checkpoint_schema\":"<<JsonString(checkpoint_magic)
        <<",\"scri_angular_stencil\":"<<JsonString(scri_angular_scheme)
        <<",\"scri_dissipation_scheme\":"<<JsonString(scri_dissipation_scheme)
        <<",\"scri_dissipation_epsilon\":"<<e.angular_epsilon
        <<",\"scri_dissipation_degree\":"<<(scri_angular_dissipation?7:0)
        <<",\"scri_dissipation_spectral_upper_bound\":"<<e.angular_upper_bound
        <<",\"scri_dissipation_cfl\":"<<JsonNumber(e.angular_cfl)
        <<",\"scri_dissipation_cfl_max\":"<<JsonNumber(e.angular_cfl_max)
        <<",\"scri_dissipation_max_neighbors\":"<<(e.model.angular_plan?e.model.angular_plan->Stats().max_neighbors:0)
        <<",\"scri_dissipation_max_incoming\":"<<(e.model.angular_plan?e.model.angular_plan->Stats().max_incoming:0)
        <<",\"scri_dissipation_max_moment_error\":"<<(e.model.angular_plan?e.model.angular_plan->Stats().max_moment_error:0)
        <<",\"scri_dissipation_max_coefficient_l1\":"<<(e.model.angular_plan?e.model.angular_plan->Stats().max_coefficient_l1:0)
        <<",\"checkpoint_header_bytes\":"<<(sizeof(Header)+(scri_angular_dissipation?sizeof(DissipationConfig):0)+(scri_sampled_normal?sizeof(NormalConfig):0)+((core_subdivisions==2||evolved_overlap)?sizeof(MeshConfig):0)+(spherical_overlap?sizeof(GeometryConfig):0)+(continuous_positive?sizeof(PositiveConfig):0));
      if constexpr(scri_sampled_normal)report<<",\"scri_normal_derivative\":\"sampled_native_numerator"<<(scri_donor_seven?7:6)<<"\""
        <<",\"scri_normal_radial_nodes\":"<<(scri_donor_seven?7:6)<<",\"scri_normal_bulk_derivative_order\":4";
      if constexpr(evolved_overlap)report<<",\"evolved_overlap\":"<<evolved_overlap<<",\"nominal_cells\":"<<e.model.grid.nominal_count;
      report<<",\"mesh_plan_setup_seconds_by_rank\":[";
      for(int rank=0;rank<e.size;++rank){if(rank)report<<',';report<<'[';for(int k=0;k<8;++k){if(k)report<<',';report<<rank_setup[8*rank+k];}report<<']';}report<<']';
      report<<",\"volume_exchange\":\""<<(sparse_stage?"cached_sparse_alltoallv":"full_allgatherv")<<"\",\"sparse_poison_unused\":"<<(e.sparse_poison?"true":"false");
      report<<",\"sparse_exchange_by_rank\":[";
      for(int rank=0;rank<e.size;++rank){if(rank)report<<',';report<<'[';for(int k=0;k<7;++k){if(k)report<<',';report<<rank_sparse[7*rank+k];}report<<']';}report<<']';
      report<<",\"boundary_execution\":\""<<(boundary_reuse?"cached_global_projection_angular":"partitioned_uncached")<<"\",\"boundary_cache_hits\":"<<e.boundary_cache_hits;
      if constexpr(continuous_positive)report<<",\"cartesian_seams\":\"uniform_global_stencil\",\"positive_variables\":\"log_chi_log_lapse_rk_and_ko\",\"physical_lapse_floor\":"<<physical_lapse_floor;
      if constexpr(spherical_overlap)report<<",\"shell_geometry\":\"spherical_equiangular\",\"core_half_width\":0.18,\"shell_inner_radius\":0.16,\"overlap_quadrature\":\"radial_quintic_partition\"";
      if constexpr(inner_transition_logical)report<<",\"interpatch_interpolation\":\"logical_inner_transition\"";
      if constexpr(core_subdivisions==2)report<<",\"mesh_topology\":\"cartesian_2x2x2_shells3\",\"cartesian_core_subdivisions\":2,\"mesh_blocks\":26";
      if constexpr(scri_donor_seven)report<<",\"scri_radial_closure\":\"donor_seven_last3\",\"scri_radial_bulk_rows\":3";
      report<<",\"minimum_owned_cells\":"<<e.total/e.size<<",\"maximum_owned_cells\":"<<(e.total+e.size-1)/e.size
        <<",\"t\":"<<e.time<<",\"start_time\":"<<start_time<<",\"target_time\":"<<end<<",\"t_over_M\":"<<(kind==2?e.time*3:e.time)
        <<",\"dt\":"<<dt<<",\"steps\":"<<e.steps<<",\"segment_steps\":"<<e.steps-start_steps<<",\"seconds\":"<<elapsed()
        <<",\"checkpoint_written\":"<<(checkpoint_written?"true":"false")<<",\"checkpoint_admitted\":"<<(checkpoint_written&&checkpoint_admitted?"true":"false")
        <<",\"diagnostic_time\":"<<JsonNumber(diagnostic_time)<<",\"diagnostics_current\":"<<(diagnostic_valid&&diagnostic_time==e.time?"true":"false")
        <<",\"constraint_measure\":\"normalized compactified Cartesian coordinate volume\""
        <<",\"exterior_method\":\"coordinate sphere at mean fast-flow radius for spherical-data comparison\""
        <<",\"initial_rhs\":"<<JsonNumber(initial_rhs)<<",\"last_rhs\":"<<JsonNumber(e.last_rhs)
        <<",\"min_chi\":"<<valid_number(diagnostic.min_chi)<<",\"min_lapse\":"<<valid_number(diagnostic.min_lapse)
        <<",\"max_determinant_residual\":"<<valid_number(diagnostic.max_determinant_residual)
        <<",\"max_tracefree_residual\":"<<valid_number(diagnostic.max_tracefree_residual)
        <<",\"max_state_deviation\":"<<valid_number(diagnostic.max_state_deviation)<<",\"max_null_surface\":"<<JsonNumber(e.model.null_max)
        <<",\"max_null_ghost\":"<<JsonNumber(e.null_ghost)<<",\"max_null_before_ghost_closure\":"<<JsonNumber(e.null_before)
        <<",\"max_pole\":"<<JsonNumber(e.model.pole_max)<<",\"max_clock\":"<<JsonNumber(e.model.clock_max)
        <<",\"max_cfl\":"<<JsonNumber(e.cfl)<<",\"min_speed_gap\":"<<JsonNumber(e.gap)
        <<",\"H_L2\":"<<valid_number(Norm(diagnostic.total.h2,diagnostic.total.volume))
        <<",\"M_L2\":"<<valid_number(Norm(diagnostic.total.m2,diagnostic.total.volume))
        <<",\"M_physical_L2\":"<<valid_number(Norm(diagnostic.total.m2_physical,diagnostic.total.volume))
        <<",\"H_exterior_L2\":"<<valid_number(Norm(diagnostic.exterior.h2,diagnostic.exterior.volume))
        <<",\"M_exterior_L2\":"<<valid_number(Norm(diagnostic.exterior.m2,diagnostic.exterior.volume))
        <<",\"Mr_L2\":"<<valid_number(Norm(diagnostic.total.mr2,diagnostic.total.volume))
        <<",\"Mr_exterior_L2\":"<<valid_number(Norm(diagnostic.exterior.mr2,diagnostic.exterior.volume))
        <<",\"Theta_exterior_L2\":"<<valid_number(Norm(diagnostic.exterior.theta2,diagnostic.exterior.volume))
        <<",\"Zr_L2\":"<<valid_number(Norm(diagnostic.total.zr2,diagnostic.total.volume))
        <<",\"Zr_exterior_L2\":"<<valid_number(Norm(diagnostic.exterior.zr2,diagnostic.exterior.volume))
        <<",\"exterior_coordinate_cutoff\":"<<JsonNumber(cutoff)<<",\"horizon_found\":"<<(horizon_found?"true":"false")
        <<",\"horizon_radius\":"<<JsonNumber(horizon_radius)<<",\"horizon_areal_radius\":"<<JsonNumber(horizon_areal)
        <<",\"horizon_mass\":"<<JsonNumber(horizon_mass)<<",\"horizon_expansion_rms\":"<<JsonNumber(horizon_expansion)
        <<",\"horizon_irreducible_mass\":"<<JsonNumber(horizon_irreducible_mass)
        <<",\"horizon_trial_radius\":"<<JsonNumber(trial_radius)<<",\"horizon_trial_expansion_rms\":"<<JsonNumber(trial_expansion)
        <<",\"horizon_complete_surface\":"<<(horizon_complete?"true":"false")<<",\"horizon_iterations\":"<<horizon_iterations
        <<",\"horizon_seed_radius\":"<<seed_radius<<",\"horizon_seed_loaded\":"<<(seed_loaded?"true":"false")
        <<",\"horizon_expansion_tolerance\":"<<horizon_tolerance
        <<",\"horizon_lmax\":"<<horizon_lmax<<",\"horizon_ntheta\":"<<horizon_ntheta<<",\"horizon_nphi\":"<<2*horizon_ntheta
        <<",\"lapse_floor_node_interventions\":"<<floor_totals[0]<<",\"lapse_floor_max_radius\":"<<floor_totals[1]
        <<",\"bulk_halo_seconds\":"<<timing[4]<<",\"timing_per_rank_exclusive\":true"
        <<",\"rhs_seconds\":"<<timing[0]<<",\"exchange_seconds\":"<<timing[1]
        <<",\"closure_seconds\":"<<timing[2]<<",\"diagnostic_seconds\":"<<timing[3]<<",\"phase_seconds_by_rank\":[";
      for(int rank=0;rank<e.size;++rank){if(rank)report<<',';report<<'[';for(int k=0;k<5;++k){if(k)report<<',';report<<rank_timings[5*rank+k];}report<<']';}report<<"],\"collective_seconds_by_rank\":[";
      for(int rank=0;rank<e.size;++rank){if(rank)report<<',';report<<'[';for(int k=0;k<3;++k){if(k)report<<',';report<<rank_collectives[3*rank+k];}report<<']';}
      report<<"],\"collective_calls_by_rank\":[";
      for(int rank=0;rank<e.size;++rank){if(rank)report<<',';report<<'[';for(int k=0;k<3;++k){if(k)report<<',';report<<rank_collective_calls[3*rank+k];}report<<']';}report<<"]}\n";
      report.close();if(!report)throw std::runtime_error("report write failed");
      std::filesystem::rename(output+".report.json.tmp",output+".report.json");
      std::cout<<"status="<<status<<" t="<<e.time<<" error="<<error<<std::endl;
    }});
  }catch(const std::exception&x){std::cerr<<"rank "<<global_variable::my_rank<<": "<<x.what()<<std::endl;code=1;}
  int collective_code=0;MPI_Allreduce(&code,&collective_code,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
  Kokkos::finalize();MPI_Finalize();return collective_code;
}
