// Zero-step evaluation through the actual native lifecycle and RHS tasks.
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "pgen/pgen.hpp"
#include "z4c_hyperboloidal/diagnostics.hpp"
#include "z4c_hyperboloidal/z4c_hyperboloidal.hpp"
#include "z4c_hyperboloidal/restart.hpp"
namespace h=z4c_hyperboloidal;
int main(int argc,char **argv) {
  MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);Kokkos::initialize(argc,argv);
  int failed=0;
  try {{
    ParameterInput pin;IOWrapper file;file.Open(argv[1],IOWrapper::FileMode::read);
    pin.LoadFromFile(file);file.Close();
    Mesh mesh(&pin);mesh.BuildTreeFromScratch(&pin);mesh.AddCoordinatesAndPhysics(&pin);
    mesh.pgen=std::make_unique<ProblemGenerator>(&pin,&mesh);
    auto &p=*mesh.pmb_pack->pz4c_hyperboloidal;
    Kokkos::Timer timer;Driver driver(&pin,&mesh,1000,&timer);
    driver.integrator="rk4_classical";driver.nexp_stages=4;mesh.dt=p.requested_dt;
    driver.stage_time=mesh.time;driver.stage_state_time=mesh.time;
    // Same record packing/restoration used by native restart files. This also
    // checks that initialization-independent reference/residual caches rebuild.
    HostArray2D<Real> before,after;h::restart::Pack(p,before);
    h::restart::Unpack(p,before);h::restart::Pack(p,after);
    if(before.size()!=after.size())throw std::runtime_error("restart shape mismatch");
    for(std::size_t i=0;i<before.size();++i)if(before.data()[i]!=after.data()[i])
      throw std::runtime_error("restart record roundtrip changed a field");
    p.InitializeBoundaryValues(&driver);p.NewTimeStep(&driver,0);
    if(!global_variable::my_rank)std::cout<<"PASS import and native record roundtrip\n"<<std::flush;
    p.InitRecv(&driver,1);p.CopyU(&driver,1);p.PrepareStage(&driver,1);
    p.SendBulk(&driver,1);p.SendSurface(&driver,1);
    p.CalcRHS(&driver,1,h::RHSSubset::Interior);p.FillLocalHalos(&driver,1);
    while(p.RecvSurface(&driver,1)!=TaskStatus::complete){}
    p.SolveNormalCompatibility(&driver,1);p.CalcSurfaceRHS(&driver,1);
    while(p.RecvBulk(&driver,1)!=TaskStatus::complete){}
    p.FillRemoteHalos(&driver,1);p.CalcRHS(&driver,1,h::RHSSubset::Dependent);
    while(p.ClearSend(&driver,1)!=TaskStatus::complete){}
    while(p.ClearRecv(&driver,1)!=TaskStatus::complete){}
    auto cfl=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.rhs_cfl);
    Real maximum=0;for(int i=0;i<p.npoints;++i)maximum=std::max(maximum,cfl(i));
    MPI_Allreduce(MPI_IN_PLACE,&maximum,1,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
    Real defects[2]={p.StageMaxClock(),p.StageMaxCompatibility()};
    MPI_Allreduce(MPI_IN_PLACE,defects,2,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
    if(!global_variable::my_rank)std::cout<<std::setprecision(17)<<"PASS RHS dt="<<p.requested_dt
      <<" max_CFL="<<maximum<<" clock="<<defects[0]
      <<" normal="<<defects[1]<<'\n'<<std::flush;
    auto &diagnostics=p.GetDiagnostics();diagnostics.Refresh();
    const auto constraints=Kokkos::create_mirror_view_and_copy(HostMemSpace(),diagnostics.Points());
    const auto weights=Kokkos::create_mirror_view_and_copy(HostMemSpace(),diagnostics.Weights());
    double sum[5]{};
    for(int i=0;i<p.npoints;++i){const auto &c=constraints(i).constraints;const double w=weights(i);
      sum[0]+=w;sum[1]+=w*c.hamiltonian*c.hamiltonian;sum[2]+=w*c.momentum_physical_norm2;
      sum[3]+=w*constraints(i).theta*constraints(i).theta;sum[4]+=w*c.z4.z_physical_norm2;}
    MPI_Allreduce(MPI_IN_PLACE,sum,5,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    if(!global_variable::my_rank){std::cout<<"full_volume="<<sum[0];
      for(int i=1;i<5;++i)std::cout<<" constraint_rms_"<<i<<'='<<std::sqrt(sum[i]/sum[0]);std::cout<<'\n'<<std::flush;}
    const auto &horizons=diagnostics.FindHorizons();
    if(!global_variable::my_rank)for(std::size_t i=0;i<horizons.size();++i){const auto &r=horizons[i];
      std::cout<<"component="<<i<<" found="<<r.found<<" area="<<r.area<<" mass_coordinate_spin="<<r.mass<<'\n';}
    if(argc>2 && std::string(argv[2])=="--dump-constraints") {
      const auto geometry=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.geometry);
      const auto exterior=Kokkos::create_mirror_view_and_copy(HostMemSpace(),diagnostics.ExteriorMask());
      std::ofstream dump("constraint-points-"+std::to_string(global_variable::my_rank)+".bin",std::ios::binary);
      for(int i=0;i<p.npoints;++i)if(weights(i)>0) {
        const auto &c=constraints(i).constraints;
        double row[12]={geometry(i).x[0],geometry(i).x[1],geometry(i).x[2],weights(i),
          double(exterior(i)),c.hamiltonian,c.momentum[0],c.momentum[1],c.momentum[2],
          c.z4.z_physical_norm2,c.z4.determinant_residual,c.z4.tracefree_residual};
        dump.write(reinterpret_cast<const char*>(row),sizeof(row));
      }
      dump.close();if(!dump)throw std::runtime_error("constraint dump failed");
    }
    const auto local=diagnostics.HistorySums();double global[10];
    MPI_Allreduce(local.data(),global,10,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    if(!global_variable::my_rank){std::cout<<"exterior_volume="<<global[5];
      for(int i=6;i<10;++i)std::cout<<" constraint_rms_"<<i-5<<'='<<std::sqrt(global[i]/global[5]);
      std::cout<<"\nPASS zero evolution steps; stability untested\n";}
  }} catch(const std::exception &e){std::cerr<<"rank "<<global_variable::my_rank<<" FAIL "<<e.what()<<'\n';failed=1;}
  MPI_Allreduce(MPI_IN_PLACE,&failed,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
  Kokkos::finalize();MPI_Finalize();return failed;
}
