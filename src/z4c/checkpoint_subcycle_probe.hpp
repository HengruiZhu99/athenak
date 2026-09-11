#ifndef Z4C_CHECKPOINT_SUBCYCLE_PROBE_HPP_
#define Z4C_CHECKPOINT_SUBCYCLE_PROBE_HPP_
#include <filesystem>
#include <fstream>
#include <iomanip>
#include "z4c/hierarchy_physics.hpp"
#include "z4c/synchronized_hierarchy_evolution.hpp"
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
  SynchronizedHierarchyEvolution<NG,CheckpointProbePhysics<NG>> evolution(mesh,z);
  const auto &tree=*evolution.tree;const auto l=evolution.layout;
  auto &storage=evolution.storage;auto &physics=*evolution.physics;
  const int ni=l.ie-l.is+1,nj=l.je-l.js+1;
  if(std::getenv("ATHENA_TEST_PROBE_FIRST_RHS"))
    physics.first_rhs_snapshot=directory+"/first_rhs_fields.bin";
  if(const char *stage=std::getenv("ATHENA_TEST_PROBE_RHS_STAGE")) {
    physics.snapshot_stage=std::stoi(stage);
    if(physics.snapshot_stage<1 || physics.snapshot_stage>4)
      throw std::invalid_argument("invalid probe snapshot stage");
  }
  const auto &control=evolution.control;
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
    Real cap=requested_dt;
    if(duration>0) {
      const Real remaining=target-time;
      // Accumulated additions can leave the final nominal interval a few ulps
      // longer than requested_dt. Integrate that endpoint rather than create
      // a spurious near-zero extra step. The actual spatial/source ceilings
      // still apply independently inside Advance.
      const Real roundoff=8*std::numeric_limits<Real>::epsilon()*
                          std::max(std::abs(time),std::abs(target));
      cap=remaining<=requested_dt+roundoff ? remaining : requested_dt;
    }
    const auto advanced=evolution.Advance(cap,ratio);
    const auto &limits=advanced.limits;
    const subcycling::Schedule schedule(limits.front().level,limits.back().level,ratio);
    for(const auto &limit:limits)
      ceilings<<accepted_intervals<<','<<time<<','<<limit.level<<','
              <<schedule.Substeps(limit.level)<<','<<limit.spatial<<','<<limit.source<<'\n';
    interval=advanced.interval;
    report=interval.corrector;
    const Real next=time+interval.dt;
    if(next<=time || (duration>0 && next>target))
      throw std::runtime_error("invalid accepted probe interval endpoint");
    const Real initial=evolution.initial;
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
