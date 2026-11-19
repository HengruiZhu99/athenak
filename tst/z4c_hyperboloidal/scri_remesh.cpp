// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Exercise the shift corrections through a real outer refine/coarsen transaction.
#include <cmath>
#include <iostream>
#include <memory>
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "outputs/outputs.hpp"
#include "pgen/pgen.hpp"
#include "z4c_hyperboloidal/diagnostics.hpp"
#include "z4c_hyperboloidal/surface_exchange.hpp"
namespace h=z4c_hyperboloidal;
int operation=0,target_chart=-1,target_layer=-1;
void Criterion(MeshBlockPack *pack) {
  auto &mesh=*pack->pmesh;
  for(int m=0;m<pack->nmb_thispack;++m) {
    const auto map=pack->pmb->mb_map.h_view(m);
    mesh.pmr->refine_flag.h_view(pack->gids+m)=
      map.chart_id==target_chart && map.layer==target_layer ? (operation==1?1:-1):0;
  }
}
void Check(Mesh &mesh) {
  auto &p=*mesh.pmb_pack->pz4c_hyperboloidal;
  h::SurfaceExchange surface(mesh.pmb_pack,&p);surface.Project();p.InvalidateDiagnostics();
  p.GetDiagnostics().Refresh();
  surface.StartReceives();surface.PackAndSend();
  while(surface.Receive()!=TaskStatus::complete) {}
  surface.Sample();surface.Solve();
  if(surface.Correction().extent_int(1)!=3)throw std::runtime_error("missing shift corrections after rebuild");
  auto direct=surface.BindKernel();auto cached=direct;cached.correction_jets=surface.CorrectionJets();
  const auto maps=p.surface_geometry;const auto points=p.surface_points.d_view;
  Real error=0,amplitude=0;
  Kokkos::parallel_reduce("all three correction cache jets after remesh",
    Kokkos::RangePolicy<>(0,p.nsurface_points),KOKKOS_LAMBDA(int q,Real &err,Real &amp) {
      const auto c=points(q);
      amp=fmax(amp,fabs(cached.correction(c.face,0,c.k,c.j)));
      for(int f=0;f<3;++f) {
        auto a=direct.Correction(c.face,c.j,c.k,maps(q),f),b=cached.Correction(c.face,c.j,c.k,maps(q),f);
        err=fmax(err,fabs(a.f-b.f));
        for(int i=0;i<3;++i) {err=fmax(err,fabs(a.d[i]-b.d[i]));
          for(int j=0;j<3;++j)err=fmax(err,fabs(a.h[i][j]-b.h[i][j]));}
      }
    },Kokkos::Max<Real>(error),Kokkos::Max<Real>(amplitude));
  MPI_Allreduce(MPI_IN_PLACE,&error,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&amplitude,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
  if(!(error<1e-9 && amplitude>1e-8))throw std::runtime_error("shift cache not exercised or inconsistent");
  surface.Evaluate();
  while(surface.ClearSend()!=TaskStatus::complete) {}
  while(surface.ClearRecv()!=TaskStatus::complete) {}
  if(!global_variable::my_rank)std::cout<<"PASS phase="<<operation<<" blocks="<<mesh.nmb_total
    <<" shift_amplitude="<<amplitude<<" cache_error="<<error<<" compatibility="<<surface.MaxCompatibility()<<'\n'<<std::flush;
}
int main(int argc,char **argv) {
  MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);Kokkos::initialize(argc,argv);int failed=0;
  try {{
    ParameterInput pin;IOWrapper file;file.Open(argv[1],IOWrapper::FileMode::read);pin.LoadFromFile(file);file.Close();
    pin.SetString("z4c_hyperboloidal","residual_weight","regular_c6");
    pin.SetRealExact("z4c_hyperboloidal","weight_inner",.3);pin.SetRealExact("z4c_hyperboloidal","weight_outer",.55);
    pin.SetString("mesh_refinement","refinement","adaptive");pin.SetInteger("mesh_refinement","num_levels",3);
    pin.SetInteger("mesh_refinement","max_nmb_per_rank",200);pin.SetInteger("mesh_refinement","refinement_interval",0);
    pin.SetString("amr_criterion0","method","user");
    Mesh mesh(&pin);mesh.BuildTreeFromScratch(&pin);mesh.AddCoordinatesAndPhysics(&pin);
    mesh.pmr=new MeshRefinement(&mesh,&pin);mesh.pgen=std::make_unique<ProblemGenerator>(&pin,&mesh);
    mesh.pgen->user_ref_func=Criterion;auto &p=*mesh.pmb_pack->pz4c_hyperboloidal;
    pin.SetString("time","evolution","dynamic");pin.SetString("time","integrator","rk4_classical");
    pin.SetInteger("time","nlim",0);pin.SetRealExact("time","tlim",1);
    Kokkos::Timer timer;Driver driver(&pin,&mesh,1800,&timer);Outputs outputs(&pin,&mesh);
    driver.Initialize(&mesh,&pin,&outputs,false);
    for(int m=0;m<mesh.pmb_pack->nmb_thispack;++m) {
      const auto map=mesh.pmb_pack->pmb->mb_map.h_view(m);
      if(map.outer_surface && map.face==0) {target_chart=map.chart_id;target_layer=map.layer;}
    }
    MPI_Allreduce(MPI_IN_PLACE,&target_chart,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&target_layer,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
    if(target_chart<0)throw std::runtime_error("missing outer fixture chart");
    const auto state=p.u0;const auto edge=p.s0;const auto cells=p.points.d_view;
    const auto sp=p.surface_points.d_view;const auto maps=p.geometry,smaps=p.surface_geometry;
    Kokkos::parallel_for("seed angular regularity error bulk",p.npoints,KOKKOS_LAMBDA(int q) {
      const auto c=cells(q);const auto &x=maps(q).x;
      state(c.m,0,c.k,c.j,c.i)*=1+1e-6*(x[0]*x[0]-x[1]*x[1]);
    });
    Kokkos::parallel_for("seed angular regularity error scri",p.nsurface_points,KOKKOS_LAMBDA(int q) {
      const auto c=sp(q);const auto &x=smaps(q).x;
      edge(c.face,0,c.k,c.j)*=1+1e-6*(x[0]*x[0]-x[1]*x[1]);
    });
    const int original=mesh.nmb_total;Check(mesh);
    for(operation=1;operation<=2;++operation) {
      const int sequence=mesh.GetAMRLoadBalanceUpdateSeq();
      mesh.pmr->AdaptiveMeshRefinement(&driver,&pin);
      if(mesh.GetAMRLoadBalanceUpdateSeq()!=sequence+1 || !p.StagePlansReady())throw std::runtime_error("missing native AMR rebuild");
      if(operation==1 && mesh.nmb_total!=original+7)throw std::runtime_error("outer leaf not refined into eight children");
      if(operation==2 && mesh.nmb_total!=original)throw std::runtime_error("coarsen failed to restore topology");
      Check(mesh);
    }
  }}catch(const std::exception &e){std::cerr<<"rank "<<global_variable::my_rank<<" FAIL "<<e.what()<<'\n';failed=1;}
  MPI_Allreduce(MPI_IN_PLACE,&failed,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
  Kokkos::finalize();MPI_Finalize();return failed;
}
