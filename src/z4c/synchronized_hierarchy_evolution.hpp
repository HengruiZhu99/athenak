#ifndef Z4C_SYNCHRONIZED_HIERARCHY_EVOLUTION_HPP_
#define Z4C_SYNCHRONIZED_HIERARCHY_EVOLUTION_HPP_
#include <memory>
#include "z4c/hierarchy_physics.hpp"
#include "driver/hierarchy_physical_maximum.hpp"
namespace z4c {
// One rank, vacuum VC Cartoon, one fixed topology. Recreate after AMR. Owns
// predictor fields and common-time gauge history; never advances live mesh time.
template<int NG,class Physics=HierarchyPhysics<NG>>
class SynchronizedHierarchyEvolution {
 public:
  std::unique_ptr<subcycling::Hierarchy> tree;
  Z4cGridLayout layout;
  subcycling::VertexParentStates storage;
  subcycling::HierarchyGeometry geometry;
  subcycling::HierarchyRK4 engine;
  subcycling::HierarchyPhysicalMaximum maximum;
  std::unique_ptr<Physics> physics;
  subcycling::CorrectorControl control;
  Real time=0,initial=0;
  SynchronizedHierarchyEvolution(Mesh *mesh,Z4c *z) {
    if(!mesh || !z || z->opt.damp_kappa1!=0 || z->opt.target_kappa1!=0 || z->opt.shift_eta!=0 ||
       !z->opt.telegraph_lapse || z->opt.telegraph_damping_prescription!=
          TelegraphDampingPrescription::max_domain_abs_K || mesh->mesh_bcs[0]!=BoundaryFlag::axis)
      throw std::invalid_argument("unsupported synchronized hierarchy configuration");
    z->RebuildSubcycleParents();
    tree=std::make_unique<subcycling::Hierarchy>(*z->subcycle_hierarchy);
    layout=z->layout;const auto l=layout;
    storage.InitializeAll(*tree,z->u0,l);
    const int rx=mesh->mesh_indcs.nx1/mesh->mb_indcs.nx1;
    const int ry=mesh->mesh_indcs.nx2/mesh->mb_indcs.nx2;
    std::array<BoundaryFlag,6> faces;for(int n=0;n<6;++n) faces[n]=mesh->mesh_bcs[n];
    std::array<bool,4> outer;
    for(int n=0;n<4;++n) outer[n]=faces[n]==BoundaryFlag::outflow ||
      faces[n]==BoundaryFlag::diode || faces[n]==BoundaryFlag::vacuum;
    geometry.Initialize(*tree,mesh->mesh_size,mesh->root_level,rx,ry,l,faces);
    std::vector<int> parity;for(int v=0;v<Z4c::nz4c;++v)
      parity.push_back(Z4cStateAxisParitySignFromPackedIndex(v));
    engine.Initialize<2*NG>(*tree,l,mesh->root_level,rx,ry,parity,z->opt.extrap_order,outer);
    engine.ReconcileInitialState(storage);
    maximum.Build(*tree,Z4c::I_Z4C_KHAT,Z4c::I_Z4C_THETA);
    leaf_ids=DvceArray1D<int>("hierarchy physical leaves",mesh->nmb_total);
    const auto ids_host=Kokkos::create_mirror(leaf_ids);int leaf_count=0;
    for(int n=0;n<static_cast<int>(tree->Nodes().size());++n)
      if(!tree->Nodes()[n].Covered()) ids_host(leaf_count++)=n;
    if(leaf_count!=mesh->nmb_total) throw std::runtime_error("hierarchy leaf ownership mismatch");
    Kokkos::deep_copy(leaf_ids,ids_host);
    initial=CurrentMaximum();
    physics=std::make_unique<Physics>(storage,geometry,l,z->opt,mesh->root_level,rx,ry,z->diss,outer,
      [this](double t) {
        if(!history) return initial;
        const auto found=cache.find(t);if(found!=cache.end()) return found->second;
        return cache.emplace(t,maximum.Evaluate(*history,t)).first->second;
      },[](double){return 0.;},[](double){return 0.;});
    // When all levels share an RK stage, use that actual stage vector's
    // physical-leaf maximum, as the native synchronous integrator does.
    // RK stages 2 and 3 have the same nominal time but different states;
    // a physical dense-output history cannot substitute for both of them.
    physics->stage_max_K=[this](const subcycling::StepContext &s,int stage,
                               const DvceArray5D<Real> &u) {
      if(s.minimum_level==physics->root &&
         s.maximum_level==physics->timestep_ranges.rbegin()->first)
        return PhysicalMaximum(u);
      return physics->max_K(s.StageTime(classical_rk4::StageTime(stage)));
    };
    physics->enforce_timestep_limits=true;physics->timestep_cfl=mesh->cfl_no;
    time=mesh->time;
  }
  SynchronizedHierarchyEvolution(const SynchronizedHierarchyEvolution &)=delete;
  SynchronizedHierarchyEvolution &operator=(const SynchronizedHierarchyEvolution &)=delete;
  Real CurrentMaximum() const {return PhysicalMaximum(storage.Values());}
  Real PhysicalMaximum(const DvceArray5D<Real> &u) const {
    const auto ids=leaf_ids;
    const int ni=layout.ie-layout.is+1,nj=layout.je-layout.js+1;
    const int is=layout.is,js=layout.js,ks=layout.ks;
    Real value=0;
    Kokkos::parallel_reduce("synchronized hierarchy max K",
      Kokkos::RangePolicy<DevExeSpace>(0,ids.extent_int(0)*ni*nj),
      KOKKOS_LAMBDA(int q,Real &v) {
        const int i=q%ni+is;q/=ni;const int j=q%nj+js;const int m=ids(q/nj);
        const Real k=u(m,Z4c::I_Z4C_KHAT,ks,j,i)+2*u(m,Z4c::I_Z4C_THETA,ks,j,i);
        const Real a=Kokkos::isfinite(k)?fabs(k):INFINITY;if(a>v) v=a;
      },Kokkos::Max<Real>(value));
    if(!std::isfinite(value)) throw std::runtime_error("nonfinite synchronized hierarchy K");
    return value;
  }
  struct AdvanceReport {
    subcycling::AcceptedInterval interval;
    std::vector<subcycling::LevelStepLimit> limits;
  };
  AdvanceReport Advance(Real cap,unsigned ratio) {
    history=nullptr;cache.clear();initial=CurrentMaximum();
    AdvanceReport result;
    result.limits=physics->TimestepLimits(time,physics->timestep_cfl);
    const subcycling::Schedule schedule(result.limits.front().level,result.limits.back().level,ratio);
    const auto choice=schedule.ChooseInterval(result.limits,cap);
    try {
      result.interval=engine.RunWithRetry(time,choice.dt,storage,*physics,ratio,control,{},
        [this](const auto &h){history=h.empty()?nullptr:&h;cache.clear();},
        [this](const auto &a,const auto &b){return maximum.Difference(a,b,control);});
    } catch(...) {history=nullptr;cache.clear();throw;}
    history=nullptr;cache.clear();initial=CurrentMaximum();
    time+=result.interval.dt;
    return result;
  }
  // Caller must rebuild native ghosts and derived geometry before diagnostics.
  void CopyAcceptedLeavesTo(const DvceArray5D<Real> &leaves) const {
    storage.CopyLeavesTo(layout,leaves);
  }
  void ImportSynchronizedLeaves(const DvceArray5D<Real> &leaves) {
    storage.CopyLeavesFrom(layout,leaves);
    engine.ReconcileInitialState(storage);
    history=nullptr;cache.clear();initial=CurrentMaximum();
  }
 private:
  DvceArray1D<int> leaf_ids;
  const subcycling::HierarchyRK4::Histories *history=nullptr;
  std::map<double,Real> cache;
};
}
#endif
