#include <mpi.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "z4c/hyperboloidal/amr_fields.hpp"

// Exercises actual UpdateMeshBlockTree/RedistAndRefineMeshBlocks, including MPI
// migration of 31 native fields plus a separately registered compensation array.
int main(int argc,char**argv){
  MPI_Init(&argc,&argv);Kokkos::initialize(argc,argv);int result=0;
  MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);
  try {
    const int n=argc>1?std::stoi(argv[1]):12, ng=4, nf=31;
    ParameterInput pin;pin.SetReal("time","cfl_number",.1);
    for(int a=1;a<=3;++a){const auto x=std::to_string(a);
      pin.SetReal("mesh","x"+x+"min",-.18);pin.SetReal("mesh","x"+x+"max",.18);
      pin.SetInteger("mesh","nx"+x,2*n);pin.SetInteger("meshblock","nx"+x,n);
      pin.SetString("mesh","ix"+x+"_bc","outflow");pin.SetString("mesh","ox"+x+"_bc","outflow");}
    pin.SetInteger("mesh","nghost",ng);pin.SetString("mesh_refinement","refinement","adaptive");
    pin.SetString("amr_criterion0","method","location");
    pin.SetInteger("mesh_refinement","num_levels",3);pin.SetInteger("mesh_refinement","max_nmb_per_rank",80);
    Mesh mesh(&pin);mesh.BuildTreeFromScratch(&pin);mesh.pmb_pack->AddCoordinates(&pin);
    mesh.pmr=new MeshRefinement(&mesh,&pin);auto&amr=*mesh.pmr;
    DvceArray5D<Real> u("native31",80,nf,n+2*ng,n+2*ng,n+2*ng);
    DvceArray5D<Real> c("coarse31",80,nf,n/2+2*ng,n/2+2*ng,n/2+2*ng);
    DvceArray5D<Real> rk("compensation31",80,nf,n+2*ng,n+2*ng,n+2*ng);
    DvceArray5D<Real> crk("coarse compensation31",80,nf,n/2+2*ng,n/2+2*ng,n/2+2*ng);
    amr.RegisterCC("hyperboloidal.native31.v1",u,c);
    amr.RegisterCC("hyperboloidal.rk-compensation31.v1",rk,crk);
    const bool smooth=argc>2&&std::string(argv[2])=="smooth";
    auto value=[&](int f,Real x,Real y,Real z){
      if(f==0)return Real(1);
      if(smooth)return 1+f*.01+.01*(f+1)*std::sin(17*x+11*y-13*z);
      return 1+f*.01+(f+1)*(x+2*y+3*z+x*y+2*y*z+3*x*z+x*x*y*y+z*z*z*z);};
    auto fill=[&](bool halos_only){
      auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),u);
      auto r=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),rk);
      for(int m=0;m<mesh.nmb_thisrank;++m){const auto&s=mesh.pmb_pack->pmb->mb_size.h_view(m);
        for(int k=0;k<n+2*ng;++k)for(int j=0;j<n+2*ng;++j)for(int i=0;i<n+2*ng;++i){
          if(halos_only&&i>=ng&&i<ng+n&&j>=ng&&j<ng+n&&k>=ng&&k<ng+n)continue;
          const Real x=s.x1min+(i-ng+.5)*s.dx1,y=s.x2min+(j-ng+.5)*s.dx2,z=s.x3min+(k-ng+.5)*s.dx3;
          for(int f=0;f<nf;++f){h(m,f,k,j,i)=value(f,x,y,z);r(m,f,k,j,i)=1e-12*value(f,x,y,z);}
        }}
      Kokkos::deep_copy(u,h);Kokkos::deep_copy(rk,r);
    };
    fill(false);Real max_error=0,rk_error=0;int moved=0;
    auto check=[&]{auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),u);
      auto r=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),rk);
      for(int m=0;m<mesh.nmb_thisrank;++m){const auto&s=mesh.pmb_pack->pmb->mb_size.h_view(m);
        for(int k=ng;k<ng+n;++k)for(int j=ng;j<ng+n;++j)for(int i=ng;i<ng+n;++i)
          for(int f=0;f<nf;++f){const Real exact=value(f,s.x1min+(i-ng+.5)*s.dx1,s.x2min+(j-ng+.5)*s.dx2,s.x3min+(k-ng+.5)*s.dx3);
            if(!std::isfinite(h(m,f,k,j,i))||!std::isfinite(r(m,f,k,j,i)))throw std::runtime_error("nonfinite migrated state");
            max_error=std::max(max_error,std::abs(h(m,f,k,j,i)-exact));
            rk_error=std::max(rk_error,std::abs(r(m,f,k,j,i)*1e12-exact));}}
    };
    // Refine one root octant, refine a neighboring corner again (2:1 balancing),
    // then coarsen all siblings twice. The 15/71/... block counts are uneven.
    for(int event=0;event<4;++event){
      Kokkos::realloc(amr.refine_flag,mesh.nmb_total);
      for(int m=0;m<mesh.nmb_total;++m){const auto&l=mesh.lloc_eachmb[m];
        amr.refine_flag.h_view(m)= event==0 ? (m==0) : event==1 ? (m==7) : l.level>mesh.root_level?-1:0;}
      amr.refine_flag.modify<HostMemSpace>();amr.refine_flag.sync<DevExeSpace>();
      int added=0,removed=0;amr.UpdateMeshBlockTree(added,removed);
      if(added||removed){amr.RedistAndRefineMeshBlocks(&pin,added,removed);mesh.MarkMeshUpdated();}
      moved+=amr.nmb_send;check();fill(true);
      if(global_variable::my_rank==0)std::cout<<"event "<<event<<" leaves "<<mesh.nmb_total<<" added "<<added<<" removed "<<removed<<'\n';
    }
    MPI_Allreduce(MPI_IN_PLACE,&max_error,1,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&rk_error,1,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&moved,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    if(global_variable::my_rank==0)std::cout<<std::setprecision(17)<<"{\"ranks\":"<<global_variable::nranks<<",\"n\":"<<n<<",\"fields\":31,\"state_error\":"<<max_error<<",\"compensation_scaled_error\":"<<rk_error<<",\"sent_blocks\":"<<moved<<"}\n";
    if(max_error>(smooth?.1:1e-11)||rk_error>(smooth?.1:1e-11)||mesh.nmb_total!=8||(global_variable::nranks>1&&moved==0))throw std::runtime_error("AMR transfer qualification failed");
  }catch(const std::exception&e){std::cerr<<e.what()<<'\n';result=1;}
  MPI_Allreduce(MPI_IN_PLACE,&result,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
  Kokkos::finalize();MPI_Finalize();return result;
}
