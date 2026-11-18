#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "z4c_hyperboloidal/restart.hpp"
#include "z4c_hyperboloidal/z4c_hyperboloidal.hpp"

namespace {
using z4c_hyperboloidal::ChiRecord;
using z4c_hyperboloidal::Z4cHyperboloidal;

void Seed(Mesh &mesh,Z4cHyperboloidal &state) {
  auto bulk=Kokkos::create_mirror_view_and_copy(HostMemSpace(),state.u0);
  auto edge=Kokkos::create_mirror_view_and_copy(HostMemSpace(),state.s0);
  auto chi=Kokkos::create_mirror_view(state.chi_records);
  auto schi=Kokkos::create_mirror_view(state.surface_chi_records);
  const auto record=[](Real x) {return ChiRecord{x+1,x+2,x+3,x+4,x+5,x+6,x+7,x+8};};
  for(int q=0;q<state.npoints;++q) {
    const auto c=state.points.h_view(q);
    const auto gid=mesh.pmb_pack->GlobalId(c.m);
    const Real value=10000000.0*gid+10000*c.k+100*c.j+c.i;
    for(int f=0;f<state.nvar;++f)bulk(c.m,f,c.k,c.j,c.i)=value+(f+1)*.015625;
    chi(q)=record(value);
  }
  for(int q=0;q<state.nsurface_points;++q) {
    const auto c=state.surface_points.h_view(q);
    const auto gid=mesh.pmb_pack->GlobalId(state.face_blocks.h_view(c.face));
    const Real value=10000000.0*gid+100*c.k+c.j;
    for(int f=0;f<state.nvar;++f)edge(c.face,f,c.k,c.j)=value+(f+1)*.015625;
    schi(q)=record(value);
  }
  Kokkos::deep_copy(state.u0,bulk);Kokkos::deep_copy(state.s0,edge);
  Kokkos::deep_copy(state.chi_records,chi);Kokkos::deep_copy(state.surface_chi_records,schi);
  state.InvalidateDiagnostics();
}

void CheckRecords(Mesh &mesh,const HostArray2D<Real> &expected) {
  HostArray2D<Real> current;
  z4c_hyperboloidal::restart::Pack(*mesh.pmb_pack->pz4c_hyperboloidal,current);
  int bad=current.extent(1)!=expected.extent(1);
  for(int m=0;m<mesh.nmb_thisrank && !bad;++m) {
    const int gid=mesh.pmb_pack->GlobalId(m);
    bad=std::memcmp(&current(m,0),&expected(gid,0),current.extent(1)*sizeof(Real))!=0;
    bad=bad || mesh.pmb_pack->LocalId(gid)!=m ||
        mesh.pmb_pack->pmb->mb_gid.h_view(m)!=gid;
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,&bad,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
#endif
  if(bad)throw std::runtime_error("Accepted bulk/scri/ChiRecord changed during migration");
}
}

int main(int argc,char **argv) {
  global_variable::my_rank=0;global_variable::nranks=1;
#if MPI_PARALLEL_ENABLED
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);
#endif
  Kokkos::initialize(argc,argv);
  int result=0;
  try {
    if(argc!=2)throw std::invalid_argument("Expected a native hyperboloidal input fixture");
    std::ifstream input(argv[1]);
    if(!input)throw std::invalid_argument("Cannot read native ownership input fixture");
    ParameterInput pin;pin.LoadFromStream(input);
    // Geometry and static RK4 fixture avoid making this lifecycle test depend on
    // a particular multirate schedule or any physical time integration.
    pin.SetString("time","integrator","rk4_classical");
    pin.SetString("multirate","block_owners","auto");
    Mesh mesh(&pin);mesh.BuildTreeFromScratch(&pin);mesh.AddCoordinatesAndPhysics(&pin);
    auto &state=*mesh.pmb_pack->pz4c_hyperboloidal;
    state.InitializeReference();Seed(mesh,state);
    HostArray2D<Real> original;
    z4c_hyperboloidal::restart::Pack(state,original);
    const auto scalars=original.extent(1);
    HostArray2D<Real> expected("global accepted test records",mesh.nmb_total,scalars);
    for(int gid=0;gid<mesh.nmb_total;++gid) {
      const int local=mesh.pmb_pack->LocalId(gid);
      if(local>=0)std::memcpy(&expected(gid,0),&original(local,0),scalars*sizeof(Real));
#if MPI_PARALLEL_ENABLED
      MPI_Bcast(&expected(gid,0),static_cast<int>(scalars),MPI_ATHENA_REAL,
                mesh.rank_eachmb[gid],MPI_COMM_WORLD);
#endif
    }
    for(int pass=0;pass<3;++pass) {
      std::vector<int> owners(mesh.nmb_total);
      for(int gid=0;gid<mesh.nmb_total;++gid)
        owners[gid]=(gid+pass)%global_variable::nranks;
      const int before=mesh.GetAMRLoadBalanceUpdateSeq();
      const bool changes=!std::equal(owners.begin(),owners.end(),mesh.rank_eachmb);
      mesh.RedistributeStaticBlocks(owners);CheckRecords(mesh,expected);
      if(mesh.GetAMRLoadBalanceUpdateSeq()!=before+static_cast<int>(changes))
        throw std::runtime_error("Ownership migration did not invalidate geometry plans once");
      if(changes && !pin.DoesParameterExist("multirate","block_owners"))
        throw std::runtime_error("Ownership migration did not persist the owner map");
      // Replaying the current map must be a true no-op.
      const int unchanged=mesh.GetAMRLoadBalanceUpdateSeq();
      mesh.RedistributeStaticBlocks(owners);CheckRecords(mesh,expected);
      if(mesh.GetAMRLoadBalanceUpdateSeq()!=unchanged)
        throw std::runtime_error("Identical ownership unnecessarily rebuilt the mesh");
    }
    if(global_variable::my_rank==0)std::cout<<"Accepted-state migration is byte-exact\n";
  } catch(const std::exception &error) {
    std::cerr<<error.what()<<'\n';result=1;
  }
  Kokkos::finalize();
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,&result,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
  MPI_Finalize();
#endif
  return result;
}
