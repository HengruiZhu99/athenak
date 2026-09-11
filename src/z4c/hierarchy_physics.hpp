#ifndef Z4C_HIERARCHY_PHYSICS_HPP_
#define Z4C_HIERARCHY_PHYSICS_HPP_
#include <functional>
#include <map>
#include "driver/hierarchy_rk4.hpp"
#include "driver/hierarchy_geometry.hpp"
#include "z4c/bulk_rhs.hpp"
#include "z4c/dissipation.hpp"
#include "z4c/axis_regularity.hpp"
#include "z4c/boundary_rhs.hpp"
#include "z4c/state_views.hpp"
#include "z4c/state_admissibility.hpp"
#include "z4c/spatial_timestep_point.hpp"
namespace z4c {
// Vacuum VC Cartoon consumer of the recursive engine. The caller supplies a
// COMMON-TIME gauge history; this class never reduces asynchronous level data.
// Source coefficients are explicit stage-time functions so production gauge
// iteration can wrap this operator without changing its numerical kernels.
template<int NGHOST>
struct HierarchyPhysics {
  subcycling::VertexParentStates &storage;
  const subcycling::HierarchyGeometry &geometry;
  Z4cGridLayout layout;
  Z4c::Options options;
  int root,root_x,root_y;
  Real diss;
  bool enforce_timestep_limits=false;
  Real timestep_cfl=.25;
  std::function<Real(double)> max_K,kappa1,shift_eta;
  std::vector<int> parities;
  std::array<bool,4> outer_faces;
  DvceArray5D<Real> mu,unused;
  // Fixed-topology cache: recreate this physics object after any regrid.
  DvceArray1D<int> timestep_nodes;
  std::map<int,std::pair<int,int>> timestep_ranges;
  Real classical_source_radius;
  HierarchyPhysics(subcycling::VertexParentStates &s,const subcycling::HierarchyGeometry &g,
      const Z4cGridLayout &l,const Z4c::Options &opt,int r,int rx,int ry,Real ko,
      const std::array<bool,4> &faces,std::function<Real(double)> maximum,
      std::function<Real(double)> damping,std::function<Real(double)> eta)
    :storage(s),geometry(g),layout(l),options(opt),root(r),root_x(rx),root_y(ry),diss(ko),
     max_K(maximum),kappa1(damping),shift_eta(eta),outer_faces(faces) {
    if(!max_K || !kappa1 || !shift_eta || s.Values().extent_int(1)!=Z4c::nz4c ||
       options.shift_mode!=Z4cShiftMode::prescribed_zero || options.extrap_order<2 ||
       options.extrap_order>4 || geometry.sizes.extent_int(0)!=s.Values().extent_int(0))
      throw std::invalid_argument("unsupported hierarchy Z4c configuration");
    for(int v=0;v<Z4c::nz4c;++v) parities.push_back(Z4cStateAxisParitySignFromPackedIndex(v));
    mu=DvceArray5D<Real>("hierarchy telegraph coefficient",s.Values().extent(0),1,l.n3,l.n2,l.n1);
    std::map<int,std::vector<int>> groups;
    const auto &levels=s.AllLevels().h_view;
    for(int m=0;m<levels.extent_int(0);++m) groups[levels(m)].push_back(m);
    if(groups.empty() || groups.begin()->first!=root)
      throw std::invalid_argument("missing hierarchy timestep root");
    timestep_nodes=DvceArray1D<int>("hierarchy timestep block lists",levels.extent(0));
    const auto ids=Kokkos::create_mirror(timestep_nodes);
    int offset=0,expected=root;
    for(const auto &group:groups) {
      if(group.first!=expected++) throw std::invalid_argument("missing hierarchy timestep level");
      timestep_ranges[group.first]={offset,static_cast<int>(group.second.size())};
      for(int m:group.second) ids(offset++)=m;
    }
    Kokkos::deep_copy(timestep_nodes,ids);
    ExplicitRKMethod method;method.stages=4;method.classical_rk4=true;
    classical_source_radius=ExplicitRKNegativeRealStabilityRadius(method);
  }
  // Synchronized-state limits for the campaign's zero-shift, undamped Z4c
  // system with max-domain telegraph damping. Include covered predictor nodes:
  // their stability matters even though they are excluded from physical extrema.
  std::vector<subcycling::LevelStepLimit> TimestepLimits(double time,Real cfl,
      int minimum_level=-1,int maximum_level=-1) const {
    if(!std::isfinite(time) || !std::isfinite(cfl) || cfl<=0 || cfl>1 ||
       options.slow_start_lapse || options.damp_kappa1!=0 ||
       options.target_kappa1!=0 || options.shift_eta!=0 ||
       (options.telegraph_lapse && options.telegraph_damping_prescription!=
          TelegraphDampingPrescription::max_domain_abs_K) ||
       kappa1(time)!=0 || shift_eta(time)!=0)
      throw std::invalid_argument("unsupported hierarchy timestep source configuration");
    const Real maximum=max_K(time);
    if(!std::isfinite(maximum) || maximum<0)
      throw std::runtime_error("invalid synchronized timestep gauge maximum");
    const auto opt=options;
    const auto coefficients=ScaleInvariantTelegraphCoefficients(
        maximum,maximum,opt.telegraph_tau,opt.telegraph_kappa);
    const Real rate=opt.telegraph_lapse ? coefficients.damping : 0;
    const Real source=SourceTimestepCeiling(opt.timestep_source_safety,
        classical_source_radius,rate);
    if(!std::isfinite(rate) || rate<0 || !std::isfinite(source) || source<=0)
      throw std::runtime_error("invalid hierarchy timestep source ceiling");
    const auto state=BindStateViews(storage.Values());
    const auto sizes=geometry.sizes.d_view;
    const auto ids=timestep_nodes;
    const int highest=timestep_ranges.rbegin()->first;
    const int first=minimum_level<0 ? root : minimum_level;
    const int last=maximum_level<0 ? highest : maximum_level;
    if(first<root || last>highest || last<first)
      throw std::invalid_argument("invalid timestep level range");
    const int ni=layout.ie-layout.is+1,nj=layout.je-layout.js+1;
    const int is=layout.is,js=layout.js,ks=layout.ks;
    std::vector<subcycling::LevelStepLimit> limits;
    for(int level=first;level<=last;++level) {
      const auto range=timestep_ranges.at(level);
      const int offset=range.first;
      const std::size_t count=static_cast<std::size_t>(range.second)*ni*nj;
      Real spatial=std::numeric_limits<Real>::max();
      Kokkos::parallel_reduce("hierarchy spatial timestep",
        Kokkos::RangePolicy<DevExeSpace,Kokkos::IndexType<std::size_t>>(0,count),
        KOKKOS_LAMBDA(std::size_t q,Real &value) {
          const int i=q%ni+is;q/=ni;const int j=q%nj+js;const int m=ids(offset+q/nj);
          const Real point=SpatialTimestepPoint(state,sizes(m),opt,m,ks,j,i,
                                                maximum,0,true,false,true);
          value=fmin(value,point);
        },Kokkos::Min<Real>(spatial));
      if(!std::isfinite(spatial) || spatial<=0 ||
         spatial==std::numeric_limits<Real>::max())
        throw std::runtime_error("invalid or missing hierarchy spatial timestep level");
      limits.push_back({level,cfl*spatial,source});
    }
    return limits;
  }
  // Only this stage's level group is at the queried stage time. Other levels
  // must not participate in the spatial reduction while asynchronously evolved.
  void CheckStageTimestep(const subcycling::StepContext &s,int stage) const {
    if(!enforce_timestep_limits) return;
    const double time=s.StageTime(classical_rk4::StageTime(stage));
    for(const auto &limit:TimestepLimits(time,timestep_cfl,s.minimum_level,s.maximum_level)) {
      const double ceiling=std::min(limit.spatial,limit.source);
      if(s.Dt()>ceiling*(1+32*std::numeric_limits<Real>::epsilon()))
        throw subcycling::IntervalStabilityFailure();
    }
  }
  void Prepare(const subcycling::StepContext &s,int,const subcycling::BlockBatches &b,
               const DvceArray5D<Real> &u) {
    EnforceLocalVertexAxis(layout,geometry.boundaries,b,u,options.vertex_axis_correction_tolerance);
    storage.FillAxisAtLogicalZero(layout,parities,s.minimum_level,s.maximum_level);
    if(options.extrap_order==2) storage.FillPhysicalGhosts<2>(layout,root,root_x,root_y,outer_faces,s.minimum_level,s.maximum_level);
    if(options.extrap_order==3) storage.FillPhysicalGhosts<3>(layout,root,root_x,root_y,outer_faces,s.minimum_level,s.maximum_level);
    if(options.extrap_order==4) storage.FillPhysicalGhosts<4>(layout,root,root_x,root_y,outer_faces,s.minimum_level,s.maximum_level);
  }
  void RHS(const subcycling::StepContext &s,int stage,const subcycling::BlockBatches &b,
           const DvceArray5D<Real> &u,const DvceArray5D<Real> &rhs) {
    CheckStageTimestep(s,stage);
    const double time=s.StageTime(classical_rk4::StageTime(stage));
    const Real maximum=max_K(time),damping=kappa1(time),eta=shift_eta(time);
    if(!std::isfinite(maximum) || maximum<0 || !std::isfinite(damping) || !std::isfinite(eta))
      throw std::runtime_error("invalid common-time gauge coefficients");
    auto state=BindStateViews(u),out=BindStateViews(rhs);Tmunu::Tmunu_vars vacuum;
    EvaluateZ4cBulkRHS<VertexCenteredZ4c,CartoonSO2,NGHOST>(layout,geometry.sizes,b,state,out,
        mu,options,true,vacuum,time,damping,eta,maximum,false,unused,false,unused);
    EnforceLocalVertexAxis(layout,geometry.boundaries,b,rhs,options.vertex_axis_correction_tolerance);
    AddZ4cDissipation<VertexCenteredZ4c,CartoonSO2,NGHOST>(layout,geometry.sizes,
        geometry.boundaries,b,u,rhs,diss,false,unused);
    ApplyLocalCartoonBoundaryRHS(layout,geometry.sizes,geometry.boundaries,b,state,out,
                                options.boundary_rhs_mode,false,NGHOST);
    EnforceLocalVertexAxis(layout,geometry.boundaries,b,rhs,options.vertex_axis_correction_tolerance);
  }
  void Project(const subcycling::StepContext &,int stage,const subcycling::BlockBatches &b,
               const DvceArray5D<Real> &u) {
    const bool telegraph=options.telegraph_lapse;
    b.For4("hierarchy zero shift and admissibility",layout.ks,layout.ke,layout.js,layout.je,
        layout.is,layout.ie,KOKKOS_LAMBDA(int m,int k,int j,int i) {
      for(int v=Z4c::I_Z4C_BETAX;v<=(telegraph ? Z4c::I_Z4C_BETAZ : Z4c::I_Z4C_BZ);++v)
        u(m,v,k,j,i)=0;
      for(int v=0;v<Z4c::nz4c;++v) if(!Kokkos::isfinite(u(m,v,k,j,i)))
        Kokkos::abort("nonfinite hierarchy Z4c state");
      if(!(u(m,Z4c::I_Z4C_ALPHA,k,j,i)>0) || !(u(m,Z4c::I_Z4C_CHI,k,j,i)>0))
        Kokkos::abort("nonpositive hierarchy lapse or chi");
      Real g[6],a[6];
      for(int v=0;v<6;++v) {g[v]=u(m,Z4c::I_Z4C_GXX+v,k,j,i);a[v]=u(m,Z4c::I_Z4C_AXX+v,k,j,i);}
      if(EvaluateConformalMetric(g[0],g[1],g[2],g[3],g[4],g[5]).reason!=Z4cStateFailureReason::valid)
        Kokkos::abort("invalid hierarchy conformal metric");
      if(stage==4) {
        if(!ProjectAdmissibleConformalState(g,a)) Kokkos::abort("hierarchy algebraic projection failed");
        for(int v=0;v<6;++v) {u(m,Z4c::I_Z4C_GXX+v,k,j,i)=g[v];u(m,Z4c::I_Z4C_AXX+v,k,j,i)=a[v];}
      }
    });
    EnforceLocalVertexAxis(layout,geometry.boundaries,b,u,options.vertex_axis_correction_tolerance);
  }
};
} // namespace z4c
#endif
