// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Manufactured accepted states plus actual refine/coarsen lifecycle; no RK work.
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <tuple>
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "pgen/pgen.hpp"
#include "driver/driver.hpp"
#include "outputs/outputs.hpp"
#include "z4c_hyperboloidal/surface_exchange.hpp"
namespace h=z4c_hyperboloidal;
int operation=1;
void Criterion(MeshBlockPack *pack) {
  auto &mesh=*pack->pmesh;
  for(int m=0;m<pack->nmb_thispack;++m) {
    int gid=pack->gids+m;const auto map=pack->pmb->mb_map.h_view(m);
    int level=mesh.lloc_eachmb[gid].level;
    mesh.pmr->refine_flag.h_view(gid)=map.chart_id==1 && map.layer==11
      ? (operation==1 && level==mesh.root_level ? 1 : operation==2 && level==mesh.root_level+1 ? -1 : 0):0;
  }
}
template<class View> std::uint64_t Hash(const View &view) {
  auto host=Kokkos::create_mirror_view_and_copy(HostMemSpace(),view);
  const auto *bytes=reinterpret_cast<const unsigned char *>(host.data());
  std::uint64_t hash=1469598103934665603ULL;
  for(size_t i=0;i<host.span()*sizeof(typename View::value_type);++i) {hash^=bytes[i];hash*=1099511628211ULL;}
  return hash;
}
std::vector<std::uint64_t> Hashes(h::Z4cHyperboloidal &p) {
  return {Hash(p.u0),Hash(p.s0),Hash(p.u_rhs),Hash(p.s_rhs),Hash(p.residual),
    Hash(p.u_accum),Hash(p.s_accum),Hash(p.chi_records),Hash(p.surface_chi_records),p.StateVersion()};
}
void Seed(h::Z4cHyperboloidal &p) {
  auto cells=p.points.d_view;auto sg=p.surface_geometry;auto maps=p.geometry;
  auto u=p.u0;auto edge=p.s0;auto points=p.surface_points.d_view;
  Kokkos::parallel_for("recorder manufactured bulk",Kokkos::RangePolicy<>(0,p.npoints),KOKKOS_LAMBDA(int i) {
    auto c=cells(i);const auto &x=maps(i).x;
    for(int f=0;f<28;++f)if(h::Fields::Independent(f))u(c.m,f,c.k,c.j,c.i)+=1e-8*(f+1)*(1+x[0]+.3*x[1]*x[2]);
  });
  Kokkos::parallel_for("recorder manufactured scri",Kokkos::RangePolicy<>(0,p.nsurface_points),KOKKOS_LAMBDA(int i) {
    auto c=points(i);const auto &x=sg(i).x;
    for(int f=0;f<28;++f)if(h::Fields::Independent(f))edge(c.face,f,c.k,c.j)+=1e-8*(f+1)*(1+x[0]+.3*x[1]*x[2]);
  });
  p.EnforceAlgebraicConstraints();p.InvalidateDiagnostics();
}
// Independent local truth table: actual module geometry, references and raw state.
// Remote donors are checked in the serial test and by serial/MPI agreement.
void Truth(Mesh &mesh,int phase) {
  auto &p=*mesh.pmb_pack->pz4c_hyperboloidal;
  h::SurfaceExchange plan(mesh.pmb_pack,&p,true);
  std::map<std::tuple<int,int,int,int>,int> index;
  for(int q=0;q<p.npoints;++q) {auto c=p.points.h_view(q);index[{c.m,c.k,c.j,c.i}]=q;}
  auto u=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.u0);
  auto geom=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.geometry);
  auto refs=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.reference);
  std::ofstream out("truth."+std::to_string(phase)+".r"+std::to_string(global_variable::my_rank)+".bin",std::ios::binary);
  for(auto d:plan.BulkExchange().UniqueDonors()) {
    auto it=index.find({d.gid-mesh.pmb_pack->gids,d.k,d.j,d.i});if(it==index.end())continue;
    int q=it->second;std::vector<double> record={double(d.gid),double(d.k),double(d.j),double(d.i)};
    for(int f=0;f<31;++f)record.push_back(u(d.gid-mesh.pmb_pack->gids,f,d.k,d.j,d.i));
    const auto g=geom(q);for(double x:g.x)record.push_back(x);
    for(auto &r:g.J)for(double x:r)record.push_back(x);
    for(auto &s:g.H)for(auto &r:s)for(double x:r)record.push_back(x);
    for(const auto &jet:refs(q)) {record.push_back(jet.f);for(auto x:jet.d)record.push_back(x);for(auto &r:jet.h)for(auto x:r)record.push_back(x);}
    out.write(reinterpret_cast<const char *>(record.data()),record.size()*sizeof(double));
  }
}
int main(int argc,char **argv) {
#if MPI_PARALLEL_ENABLED
  MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);
#endif
  Kokkos::initialize(argc,argv);
  try {{
    ParameterInput pin;IOWrapper file;file.Open(argv[1],IOWrapper::FileMode::read);pin.LoadFromFile(file);file.Close();
    pin.SetString("mesh_refinement","refinement","adaptive");pin.SetInteger("mesh_refinement","num_levels",2);
    pin.SetInteger("mesh_refinement","max_nmb_per_rank",180/global_variable::nranks+8);
    pin.SetInteger("mesh_refinement","refinement_interval",0);pin.SetString("amr_criterion0","method","user");
    Mesh mesh(&pin);mesh.BuildTreeFromScratch(&pin);mesh.AddCoordinatesAndPhysics(&pin);
    mesh.pmr=new MeshRefinement(&mesh,&pin);mesh.pgen=std::make_unique<ProblemGenerator>(&pin,&mesh);mesh.pgen->user_ref_func=Criterion;
    auto &p=*mesh.pmb_pack->pz4c_hyperboloidal;Seed(p);p.BuildStagePlans();
    pin.SetString("time","evolution","dynamic");pin.SetString("time","integrator","rk4_classical");
    pin.SetInteger("time","nlim",0);pin.SetRealExact("time","tlim",1);
    Kokkos::Timer timer;Driver driver(&pin,&mesh,3600,&timer);Outputs outputs(&pin,&mesh);
    for(int phase=0;phase<3;++phase) {
      if(phase) {operation=phase;mesh.pmr->AdaptiveMeshRefinement(&driver,&pin);}
      mesh.ncycle=phase;mesh.time=phase*p.requested_dt;
      auto before=Hashes(p);
      for(auto *out:outputs.pout_list) {out->LoadOutputData(&mesh);out->WriteOutputFile(&mesh,&pin);}
      if(before!=Hashes(p))throw std::runtime_error("recorder mutated state/workspace/history");
      Truth(mesh,phase);
      if(global_variable::my_rank==0)std::cout<<"PASS scri readonly phase="<<phase<<" sequence="<<mesh.GetAMRLoadBalanceUpdateSeq()<<" blocks="<<mesh.nmb_total<<'\n';
    }
    // Topology invalidation at the same accepted time must still emit an epoch.
    mesh.MarkMeshUpdated();
    auto before=Hashes(p);
    for(auto *out:outputs.pout_list) {out->LoadOutputData(&mesh);out->WriteOutputFile(&mesh,&pin);}
    if(before!=Hashes(p))throw std::runtime_error("same-time epoch mutated state");
    // Synchronized output-only benchmark: exclude evolution load imbalance.
    for(int repeat=0;repeat<3;++repeat) {
      ++mesh.ncycle;mesh.time+=p.requested_dt;
#if MPI_PARALLEL_ENABLED
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      Kokkos::Timer output_timer;
      for(auto *out:outputs.pout_list) {out->LoadOutputData(&mesh);out->WriteOutputFile(&mesh,&pin);}
      double seconds=output_timer.seconds();
#if MPI_PARALLEL_ENABLED
      MPI_Allreduce(MPI_IN_PLACE,&seconds,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
#endif
      if(global_variable::my_rank==0)std::cout<<"SCRI_BENCH maximum_rank_seconds="<<seconds<<'\n';
    }
  }} catch(const std::exception &e) {std::cerr<<e.what()<<'\n';
#if MPI_PARALLEL_ENABLED
    MPI_Abort(MPI_COMM_WORLD,1);
#endif
    return 1;
  }
  Kokkos::finalize();
#if MPI_PARALLEL_ENABLED
  MPI_Finalize();
#endif
}
