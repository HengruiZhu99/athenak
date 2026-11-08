// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Import an already-validated canonical trajectory snapshot, then use the
// actual native RestartOutput writer. This is a format fixture, not evolution.
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"
#include "pgen/pgen.hpp"
#include "z4c_hyperboloidal/mapped_halo.hpp"
#include "z4c_hyperboloidal/remesh.hpp"

namespace h=z4c_hyperboloidal;
void Require(bool yes,const char *message) {if (!yes)throw std::runtime_error(message);}
std::vector<Real> Read(const std::string &path,std::size_t start,std::size_t count,
                       std::size_t total) {
  std::ifstream file(path,std::ios::binary|std::ios::ate);
  Require(file.good() && std::size_t(file.tellg())==total*sizeof(Real),"trajectory fixture length differs");
  file.seekg(start*sizeof(Real));std::vector<Real> values(count);
  file.read(reinterpret_cast<char *>(values.data()),count*sizeof(Real));
  Require(file.good(),"cannot read trajectory fixture");return values;
}
int main(int argc,char **argv) {
  MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);Kokkos::initialize(argc,argv);
  try {{
    Require(argc==3,"usage: n12_checkpoint_fixture n12-input trajectory-directory");
    ParameterInput pin;IOWrapper file;file.Open(argv[1],IOWrapper::FileMode::read);
    pin.LoadFromFile(file);file.Close();
    pin.SetString("time","evolution","dynamic");pin.SetString("time","integrator","rk4_classical");
    pin.SetReal("time","tlim",1);pin.SetInteger("time","nlim",8);
    pin.SetString("output1","file_type","rst");pin.SetInteger("output1","dcycle",8);
    pin.SetBoolean("output1","single_file_per_rank",false);
    Mesh mesh(&pin);mesh.BuildTreeFromScratch(&pin);mesh.AddCoordinatesAndPhysics(&pin);
    mesh.pgen=std::make_unique<ProblemGenerator>(&pin,&mesh);
    auto &p=*mesh.pmb_pack->pz4c_hyperboloidal;h::NativeDonorSelector selector(mesh,p.origin);
    int bulk_offset=0,edge_offset=0,total_bulk=0,total_edge=0;
    for(int gid=0;gid<mesh.nmb_total;++gid) {
      int count=1;for(int a=0;a<3;++a)count*=selector.Upper(gid,a)-selector.Lower(gid,a);
      if(gid<mesh.pmb_pack->gids) {bulk_offset+=count;if(selector.Map(gid).outer_surface)edge_offset+=144;}
      total_bulk+=count;if(selector.Map(gid).outer_surface)total_edge+=144;
    }
    Require(total_bulk==229392 && total_edge==864,"trajectory fixture is not fixed N12");
    const std::string directory=argv[2];
    const auto bulk=Read(directory+"/native-step8-bulk.bin",31*bulk_offset,31*p.npoints,31*total_bulk);
    const auto edge=Read(directory+"/native-step8-surface.bin",31*edge_offset,31*p.nsurface_points,31*total_edge);
    const auto chi=Read(directory+"/native-step8-chi.bin",8*bulk_offset,8*p.npoints,8*total_bulk);
    const auto schi=Read(directory+"/native-step8-surface-chi.bin",8*edge_offset,8*p.nsurface_points,8*total_edge);
    auto u=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.u0);
    auto s=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.s0);
    auto records=Kokkos::create_mirror_view(p.chi_records);
    auto surface_records=Kokkos::create_mirror_view(p.surface_chi_records);
    auto record=[](const Real *r) {return h::ChiRecord{r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7]};};
    for(int q=0;q<p.npoints;++q) {
      const auto c=p.points.h_view(q);
      for(int f=0;f<31;++f)u(c.m,f,c.k,c.j,c.i)=bulk[31*q+f];
      records(q)=record(chi.data()+8*q);
    }
    for(int q=0;q<p.nsurface_points;++q) {
      const auto c=p.surface_points.h_view(q);
      for(int f=0;f<31;++f)s(c.face,f,c.k,c.j)=edge[31*q+f];
      surface_records(q)=record(schi.data()+8*q);
    }
    Kokkos::deep_copy(p.u0,u);Kokkos::deep_copy(p.s0,s);
    Kokkos::deep_copy(p.chi_records,records);Kokkos::deep_copy(p.surface_chi_records,surface_records);
    mesh.time=0;mesh.dt=p.requested_dt;mesh.ncycle=8;
    for(int step=0;step<8;++step)mesh.time+=mesh.dt;
    OutputParameters op{};op.block_name="output1";op.file_type="rst";
    op.file_basename="trajectory_fixture";op.last_time=-1;op.dcycle=8;
    RestartOutput output(&pin,&mesh,op);output.LoadOutputData(&mesh);output.WriteOutputFile(&mesh,&pin);
    if(!global_variable::my_rank)std::cout<<"PASS native writer trajectory format fixture\n";
  }}catch(const std::exception &error) {
    std::cerr<<error.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);return 1;
  }
  Kokkos::finalize();MPI_Finalize();return 0;
}
