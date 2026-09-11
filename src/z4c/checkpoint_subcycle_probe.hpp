#ifndef Z4C_CHECKPOINT_SUBCYCLE_PROBE_HPP_
#define Z4C_CHECKPOINT_SUBCYCLE_PROBE_HPP_
#include <filesystem>
#include <fstream>
#include <iomanip>
#include "z4c/hierarchy_physics.hpp"
#include "driver/hierarchy_physical_maximum.hpp"
namespace z4c {
template<int NG> struct CheckpointProbePhysics : HierarchyPhysics<NG> {
  using HierarchyPhysics<NG>::HierarchyPhysics;
  std::string first_rhs_snapshot;
  int snapshot_stage=1;
  void RHS(const subcycling::StepContext &step,int stage,const subcycling::BlockBatches &blocks,
           const DvceArray5D<Real> &u,const DvceArray5D<Real> &rhs) {
    const auto snapshot=stage==snapshot_stage ? first_rhs_snapshot : std::string();
    if(!snapshot.empty()) {
      const auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),u);
      std::ofstream out(first_rhs_snapshot,std::ios::binary);
      out.write(reinterpret_cast<const char *>(host.data()),host.size()*sizeof(Real));
      out.close();if(!out) throw std::runtime_error("probe RHS snapshot failed");
      first_rhs_snapshot.clear();
    }
    HierarchyPhysics<NG>::RHS(step,stage,blocks,u,rhs);
    if(!snapshot.empty()) {
      const auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),rhs);
      std::ofstream out(snapshot+".rhs",std::ios::binary);
      out.write(reinterpret_cast<const char *>(host.data()),host.size()*sizeof(Real));
      out.close();if(!out) throw std::runtime_error("probe RHS output failed");
    }
  }
};
// Qualification hook only: evolve a copied, fixed hierarchy over bounded intervals.
// Never copy the result back to the live MeshBlockPack or change checkpoint time.
template<int NG> void CheckpointSubcycleProbe(Mesh *mesh,Z4c *z,
    const std::string &directory,Real dt,unsigned ratio,Real duration=0) {
  if(!z || !std::isfinite(dt) || dt<=0 || !std::isfinite(duration) || duration<0 || ratio>32 ||
     z->opt.damp_kappa1!=0 || z->opt.target_kappa1!=0 || z->opt.shift_eta!=0 ||
     !z->opt.telegraph_lapse || z->opt.telegraph_damping_prescription!=
        TelegraphDampingPrescription::max_domain_abs_K ||
     mesh->mesh_bcs[0]!=BoundaryFlag::axis)
    throw std::invalid_argument("unsupported checkpoint probe configuration");
  if(!std::filesystem::create_directory(directory))
    throw std::runtime_error("checkpoint probe requires a new output directory");
  z->RebuildSubcycleParents();
  const auto &tree=*z->subcycle_hierarchy;const auto l=z->layout;
  subcycling::VertexParentStates storage;storage.InitializeAll(tree,z->u0,l);
  const int rx=mesh->mesh_indcs.nx1/mesh->mb_indcs.nx1;
  const int ry=mesh->mesh_indcs.nx2/mesh->mb_indcs.nx2;
  std::array<BoundaryFlag,6> faces;for(int n=0;n<6;++n) faces[n]=mesh->mesh_bcs[n];
  std::array<bool,4> outer;
  for(int n=0;n<4;++n) outer[n]=faces[n]==BoundaryFlag::outflow ||
    faces[n]==BoundaryFlag::diode || faces[n]==BoundaryFlag::vacuum;
  subcycling::HierarchyGeometry geometry;
  geometry.Initialize(tree,mesh->mesh_size,mesh->root_level,rx,ry,l,faces);
  std::vector<int> parity;for(int v=0;v<Z4c::nz4c;++v)
    parity.push_back(Z4cStateAxisParitySignFromPackedIndex(v));
  subcycling::HierarchyRK4 engine;
  engine.Initialize<2*NG>(tree,l,mesh->root_level,rx,ry,parity,z->opt.extrap_order,outer);
  engine.ReconcileInitialState(storage);
  subcycling::HierarchyPhysicalMaximum maximum;
  maximum.Build(tree,Z4c::I_Z4C_KHAT,Z4c::I_Z4C_THETA);
  // Initial coefficient from the synchronized physical leaves, never parents.
  const auto u=storage.Values();const int ni=l.ie-l.is+1,nj=l.je-l.js+1;
  const int is=l.is,js=l.js,ks=l.ks;
  DvceArray1D<int> leaf_ids("probe physical leaves",mesh->nmb_total);
  const auto ids_host=Kokkos::create_mirror(leaf_ids);int leaf_count=0;
  for(int n=0;n<static_cast<int>(tree.Nodes().size());++n)
    if(!tree.Nodes()[n].Covered()) ids_host(leaf_count++)=n;
  if(leaf_count!=mesh->nmb_total) throw std::runtime_error("probe leaf ownership mismatch");
  Kokkos::deep_copy(leaf_ids,ids_host);
  const auto current_maximum=[&]() {
    Real value=0;
    Kokkos::parallel_reduce("probe synchronized max K",
      Kokkos::RangePolicy<DevExeSpace>(0,leaf_count*ni*nj),
      KOKKOS_LAMBDA(int q,Real &v) {
        const int i=q%ni+is;q/=ni;const int j=q%nj+js;const int m=leaf_ids(q/nj);
        const Real k=u(m,Z4c::I_Z4C_KHAT,ks,j,i)+2*u(m,Z4c::I_Z4C_THETA,ks,j,i);
        const Real a=Kokkos::isfinite(k)?fabs(k):INFINITY;if(a>v) v=a;
      },Kokkos::Max<Real>(value));
    if(!std::isfinite(value)) throw std::runtime_error("nonfinite synchronized probe K");
    return value;
  };
  Real initial=current_maximum();
  const subcycling::HierarchyRK4::Histories *history=nullptr;
  std::map<double,Real> cache;
  CheckpointProbePhysics<NG> physics(storage,geometry,l,z->opt,mesh->root_level,rx,ry,z->diss,outer,
    [&](double t) {
      if(!history) return initial;
      const auto found=cache.find(t);if(found!=cache.end()) return found->second;
      return cache.emplace(t,maximum.Evaluate(*history,t)).first->second;
    },[](double){return 0.;},[](double){return 0.;});
  if(std::getenv("ATHENA_TEST_PROBE_FIRST_RHS"))
    physics.first_rhs_snapshot=directory+"/first_rhs_fields.bin";
  if(const char *stage=std::getenv("ATHENA_TEST_PROBE_RHS_STAGE")) {
    physics.snapshot_stage=std::stoi(stage);
    if(physics.snapshot_stage<1 || physics.snapshot_stage>4)
      throw std::invalid_argument("invalid probe snapshot stage");
  }
  subcycling::CorrectorControl control;
  const Real requested_dt=dt;
  const Real target=mesh->time+(duration>0 ? duration : dt);
  if(!std::isfinite(target) || target<=mesh->time)
    throw std::invalid_argument("unrepresentable probe end time");
  Real time=mesh->time;
  int accepted_intervals=0,total_attempts=0,total_passes=0;
  subcycling::AcceptedInterval interval;
  subcycling::CorrectorReport report;
  std::ofstream ceilings(directory+"/timestep_limits.csv");
  ceilings<<std::setprecision(17)<<"interval,time,level,substeps,spatial_with_cfl,source\n";
  std::ofstream steps(directory+"/intervals.csv");
  steps<<std::setprecision(17)<<"interval,start,end,dt,attempts,passes,max_abs_K\n";
  Kokkos::Timer timer;
  do {
    if(accepted_intervals>=1000000) throw std::runtime_error("probe interval budget exceeded");
    history=nullptr;cache.clear();initial=current_maximum();
    const auto limits=physics.TimestepLimits(time,mesh->cfl_no);
    const subcycling::Schedule schedule(limits.front().level,limits.back().level,ratio);
    const Real cap=duration>0 ? std::min(requested_dt,target-time) : requested_dt;
    const auto choice=schedule.ChooseInterval(limits,cap);
    for(const auto &limit:limits)
      ceilings<<accepted_intervals<<','<<time<<','<<limit.level<<','
              <<schedule.Substeps(limit.level)<<','<<limit.spatial<<','<<limit.source<<'\n';
    interval=engine.RunWithRetry(time,choice.dt,storage,physics,ratio,control,{},
      [&](const auto &h){history=h.empty()?nullptr:&h;cache.clear();},
      [&](const auto &a,const auto &b){return maximum.Difference(a,b,control);});
    report=interval.corrector;
    const Real next=time+interval.dt;
    if(next<=time || (duration>0 && next>target))
      throw std::runtime_error("invalid accepted probe interval endpoint");
    history=nullptr;cache.clear();initial=current_maximum();
    steps<<accepted_intervals<<','<<time<<','<<next<<','<<interval.dt<<','
         <<interval.attempts<<','<<interval.total_passes<<','<<initial<<'\n';
    time=next;++accepted_intervals;total_attempts+=interval.attempts;
    total_passes+=interval.total_passes;
  } while(duration>0 && time<target);
  Kokkos::fence("checkpoint probe evolution complete");const double seconds=timer.seconds();
  dt=duration>0 ? time-mesh->time : interval.dt;
  ceilings.close();steps.close();
  if(!ceilings || !steps) throw std::runtime_error("checkpoint interval output failed");
  // Binary layout: hierarchy leaf order, then variable,j,i (Real scalars).
  const auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),storage.Values());
  std::ofstream fields(directory+"/fields.bin",std::ios::binary);
  std::ofstream topology(directory+"/leaves.txt");
  for(int n=0;n<static_cast<int>(tree.Nodes().size());++n) if(!tree.Nodes()[n].Covered()) {
    const auto &node=tree.Nodes()[n];
    topology<<node.source_leaf<<' '<<node.key[0]<<' '<<node.key[1]<<' '<<node.key[2]<<'\n';
    for(int v=0;v<Z4c::nz4c;++v) for(int j=l.js;j<=l.je;++j) for(int i=l.is;i<=l.ie;++i) {
      const Real x=host(n,v,l.ks,j,i);fields.write(reinterpret_cast<const char *>(&x),sizeof(x));
    }
  }
  fields.close();topology.close();
  std::ofstream meta(directory+"/probe.txt");meta<<std::setprecision(17)
    <<"checkpoint_time="<<mesh->time<<"\nend_time="<<mesh->time+dt<<"\ndt="<<dt
    <<"\nrequested_dt="<<requested_dt<<"\nrequested_duration="<<duration
    <<"\naccepted_intervals="<<accepted_intervals<<"\ninterval_attempts="<<total_attempts
    <<"\ntotal_corrector_passes="<<total_passes
    <<"\nratio="<<ratio<<"\nspatial_order="<<z->opt.spatial_order
    <<"\ncorrector_atol="<<control.absolute_tolerance<<"\ncorrector_rtol="<<control.relative_tolerance
    <<"\ncorrector_max_passes="<<control.maximum_passes<<"\npasses="<<report.passes<<"\nseconds="<<seconds
    <<"\nendpoint_change="<<report.endpoint_change<<"\nhistory_change="<<report.history_change
    <<"\nfeedback_change="<<report.feedback_change<<"\nreal_bytes="<<sizeof(Real)
    <<"\nvariables="<<Z4c::nz4c<<"\nni="<<ni<<"\nnj="<<nj<<"\nleaves="<<mesh->nmb_total
    <<"\nmode=isolated_fixed_hierarchy_probe\n";
  meta.close();if(!fields || !topology || !meta) throw std::runtime_error("checkpoint probe output failed");
}
}
#endif
