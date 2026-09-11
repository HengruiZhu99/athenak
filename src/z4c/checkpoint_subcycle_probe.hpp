#ifndef Z4C_CHECKPOINT_SUBCYCLE_PROBE_HPP_
#define Z4C_CHECKPOINT_SUBCYCLE_PROBE_HPP_
#include <filesystem>
#include <fstream>
#include <iomanip>
#include "z4c/hierarchy_physics.hpp"
#include "driver/hierarchy_physical_maximum.hpp"
namespace z4c {
// Qualification hook only: evolve a copied, fixed hierarchy for ONE interval.
// Never copy the result back to the live MeshBlockPack or change checkpoint time.
template<int NG> void CheckpointSubcycleProbe(Mesh *mesh,Z4c *z,
    const std::string &directory,Real dt,unsigned ratio) {
  if(!z || !std::isfinite(dt) || dt<=0 || dt>mesh->dt || ratio>32 ||
     z->opt.damp_kappa1!=0 || z->opt.target_kappa1!=0 || z->opt.shift_eta!=0 ||
     !z->opt.telegraph_lapse || z->opt.telegraph_damping_prescription!=
        TelegraphDampingPrescription::max_domain_abs_K ||
     mesh->mesh_bcs[0]!=BoundaryFlag::axis)
    throw std::invalid_argument("unsupported checkpoint probe or dt exceeds saved finest dt");
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
  subcycling::HierarchyPhysicalMaximum maximum;
  maximum.Build(tree,Z4c::I_Z4C_KHAT,Z4c::I_Z4C_THETA);
  // Initial coefficient from the synchronized physical leaves, never parents.
  const auto u=z->u0;const int ni=l.ie-l.is+1,nj=l.je-l.js+1;
  const int is=l.is,js=l.js,ks=l.ks;
  Real initial=0;
  Kokkos::parallel_reduce("probe initial max K",
    Kokkos::RangePolicy<DevExeSpace>(0,mesh->nmb_total*ni*nj),
    KOKKOS_LAMBDA(int q,Real &v) {
      const int i=q%ni+is;q/=ni;const int j=q%nj+js;const int m=q/nj;
      const Real k=u(m,Z4c::I_Z4C_KHAT,ks,j,i)+2*u(m,Z4c::I_Z4C_THETA,ks,j,i);
      const Real a=Kokkos::isfinite(k)?fabs(k):INFINITY;if(a>v) v=a;
    },Kokkos::Max<Real>(initial));
  if(!std::isfinite(initial)) throw std::runtime_error("nonfinite checkpoint K");
  const subcycling::HierarchyRK4::Histories *history=nullptr;
  std::map<double,Real> cache;
  HierarchyPhysics<NG> physics(storage,geometry,l,z->opt,mesh->root_level,rx,ry,z->diss,outer,
    [&](double t) {
      if(!history) return initial;
      const auto found=cache.find(t);if(found!=cache.end()) return found->second;
      return cache.emplace(t,maximum.Evaluate(*history,t)).first->second;
    },[](double){return 0.;},[](double){return 0.;});
  subcycling::CorrectorControl control;
  Kokkos::Timer timer;
  const auto report=engine.RunCorrected(mesh->time,dt,storage,physics,ratio,control,
    [&](const auto &h){history=h.empty()?nullptr:&h;cache.clear();},
    [&](const auto &a,const auto &b){return maximum.Difference(a,b,control);});
  Kokkos::fence("checkpoint probe evolution complete");const double seconds=timer.seconds();
  history=nullptr;cache.clear();
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
