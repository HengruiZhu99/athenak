// Bounded native refine/coarsen test using the imported binary, no RK updates.
#include <cmath>
#include <iostream>
#include <memory>
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "outputs/outputs.hpp"
#include "pgen/pgen.hpp"
#include "z4c_hyperboloidal/z4c_hyperboloidal.hpp"
#include "z4c_hyperboloidal/diagnostics.hpp"
int operation=0;
void Criterion(MeshBlockPack *pack) {
  auto &mesh=*pack->pmesh;
  for(int m=0;m<pack->nmb_thispack;++m) {
    const int gid=pack->gids+m;const auto map=pack->pmb->mb_map.h_view(m);
    bool select=false;
    // Refine the complete siblings around each puncture-containing leaf;
    // coarsening selects every child of those same geometric parent boxes.
    if(map.chart_id==0)for(double x:{-.20233727083953934,.20233727083953934}) {
      const double width=.2/16;
      const double lo[3]={std::floor((x+.3)/width)*width-.3,0,0};
      bool inside=true;for(int a=0;a<3;++a)
        inside=inside && map.lo[a]>=lo[a]-1e-13 && map.hi[a]<=lo[a]+width+1e-13;
      select=select||inside;
    }
    const int level=mesh.lloc_eachmb[gid].level-mesh.root_level;
    mesh.pmr->refine_flag.h_view(gid)=select?(operation==1&&level==4?1:operation==2&&level==5?-1:0):0;
  }
}
int main(int argc,char **argv) {
  MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);Kokkos::initialize(argc,argv);
  int failed=0;
  try {{
    ParameterInput pin;IOWrapper file;file.Open(argv[1],IOWrapper::FileMode::read);pin.LoadFromFile(file);file.Close();
    pin.SetString("mesh_refinement","refinement","adaptive");pin.SetInteger("mesh_refinement","num_levels",6);
    pin.SetInteger("mesh_refinement","max_nmb_per_rank",1400/global_variable::nranks+100);
    pin.SetInteger("mesh_refinement","refinement_interval",0);pin.SetString("amr_criterion0","method","user");
    Mesh mesh(&pin);mesh.BuildTreeFromScratch(&pin);mesh.AddCoordinatesAndPhysics(&pin);
    mesh.pmr=new MeshRefinement(&mesh,&pin);mesh.pgen=std::make_unique<ProblemGenerator>(&pin,&mesh);
    mesh.pgen->user_ref_func=Criterion;auto &p=*mesh.pmb_pack->pz4c_hyperboloidal;
    pin.SetString("time","evolution","dynamic");pin.SetString("time","integrator","rk4_classical");
    pin.SetInteger("time","nlim",0);pin.SetRealExact("time","tlim",1);
    Kokkos::Timer timer;Driver driver(&pin,&mesh,1800,&timer);Outputs outputs(&pin,&mesh);
    driver.Initialize(&mesh,&pin,&outputs,false);
    const int original=mesh.nmb_total;
    for(operation=1;operation<=2;++operation) {
      const int sequence=mesh.GetAMRLoadBalanceUpdateSeq();
      try {mesh.pmr->AdaptiveMeshRefinement(&driver,&pin);}
      catch(const std::exception &) {
        const auto u=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.u0);
        const auto geometry=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.geometry);
        int printed=0;
        for(int q=0;q<p.npoints;++q) {
          const auto c=p.points.h_view(q);const auto &x=geometry(q).x;
          const double chi=u(c.m,0,c.k,c.j,c.i),alpha=u(c.m,18,c.k,c.j,c.i)+(1+x[0]*x[0]+x[1]*x[1]+x[2]*x[2])/2;
          if((!(chi>0) || !(alpha>0)) && printed++<8)
            std::cerr<<"TRANSFER_FAILURE rank="<<global_variable::my_rank<<" xyz="<<x[0]<<','<<x[1]<<','<<x[2]
              <<" chi="<<chi<<" lapse="<<alpha<<'\n';
        }
        throw;
      }
      if(mesh.GetAMRLoadBalanceUpdateSeq()!=sequence+1 || !p.StagePlansReady())
        throw std::runtime_error("remesh or stage-plan rebuild did not occur");
      if(operation==1 && mesh.nmb_total<=original)throw std::runtime_error("no selected leaf refined");
      if(operation==2 && mesh.nmb_total!=original)throw std::runtime_error("coarsening did not restore topology");
      p.GetDiagnostics().Refresh();
      if(!global_variable::my_rank)std::cout<<"PASS remesh phase="<<operation<<" blocks="<<mesh.nmb_total
        <<" timestep="<<p.dtnew<<" cycle="<<mesh.ncycle<<'\n'<<std::flush;
    }
  }} catch(const std::exception &e){std::cerr<<"rank "<<global_variable::my_rank<<" FAIL "<<e.what()<<'\n';failed=1;}
  MPI_Allreduce(MPI_IN_PLACE,&failed,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
  Kokkos::finalize();MPI_Finalize();return failed;
}
