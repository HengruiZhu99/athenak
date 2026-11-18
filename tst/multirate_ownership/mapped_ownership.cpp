#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "athena.hpp"
#include "bvals/mapped_exchange.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

int main(int argc,char **argv) {
  int rank=0,ranks=1,result=0;
#if MPI_PARALLEL_ENABLED
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&ranks);
#endif
  Kokkos::initialize(argc,argv);
  try {
    const int blocks=3*ranks;
    bvals::MappedExchange exchange;
    for(int distribution=0;distribution<2;++distribution) {
      std::vector<int> owners(blocks),local;
      for(int gid=0;gid<blocks;++gid) {
        owners[gid]=(gid+distribution)%ranks;
        if(owners[gid]==rank)local.push_back(gid);
      }
      std::reverse(local.begin(),local.end());
      std::vector<bvals::DonorPoint> requests;
      for(int gid=blocks-1;gid>=0;--gid)requests.push_back({gid,0,0,0});
      requests.push_back({0,0,0,0}); // repeated donor retains ordinal reuse
      exchange.Rebuild(local,owners,distribution,requests,{1,3});
      DvceArray5D<Real> state("noncontiguous donor state",local.size(),4,1,1,1);
      auto host=Kokkos::create_mirror_view(state);
      for(int pass=0;pass<2;++pass) {
        for(std::size_t m=0;m<local.size();++m)for(int f=0;f<4;++f)
          host(m,f,0,0,0)=1000*pass+100*local[m]+f;
        Kokkos::deep_copy(state,host);
        exchange.StartReceives(17+pass);
        exchange.PackAndSend(state);
        while(exchange.Receive()!=TaskStatus::complete) {}
        while(exchange.ClearSend()!=TaskStatus::complete) {}
        while(exchange.ClearRecv()!=TaskStatus::complete) {}
        auto values=Kokkos::create_mirror_view_and_copy(HostMemSpace(),exchange.Values());
        for(std::size_t q=0;q<requests.size();++q)for(int f=0;f<2;++f) {
          const Real expected=1000*pass+100*requests[q].gid+(f?3:1);
          if(values(exchange.RequestOrdinals()[q],f)!=expected)
            throw std::runtime_error("noncontiguous mapped donor value mismatch");
        }
      }
    }
  } catch(const std::exception &error) {std::cerr<<error.what()<<'\n';result=1;}
  Kokkos::finalize();
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,&result,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
  MPI_Finalize();
#endif
  return result;
}
