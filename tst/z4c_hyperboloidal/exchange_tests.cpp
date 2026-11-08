// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Native block donor transport: sparse/repeated stages, poison, duplicate points,
// arbitrary field order, zero-peer ranks, and ownership changes on rebuild.
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "bvals/mapped_exchange.hpp"

namespace {
void Require(bool valid,const char *message) {
  if (!valid) throw std::runtime_error(message);
}
Real Expected(int epoch,int stage,int gid,int f,int k,int j,int i) {
  return Real(epoch*100000000+stage*1000000+gid*10000+f*100+k*16+j*4+i);
}
void Complete(bvals::MappedExchange &exchange) {
  int polls=0;
  while (exchange.Receive() != TaskStatus::complete) {
    Require(++polls < 10000000,"mapped receives did not complete");
  }
  Require(exchange.ClearRecv() == TaskStatus::complete,"receive clear failed");
  polls=0;
  while (exchange.ClearSend() != TaskStatus::complete) {
    Require(++polls < 10000000,"mapped sends did not complete");
  }
}
}  // namespace

int main(int argc,char **argv) {
  int rank=0,nranks=1;
#if MPI_PARALLEL_ENABLED
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&nranks);
#endif
  Kokkos::initialize(argc,argv);
  {
    bvals::MappedExchange exchange;
    DvceArray5D<Real> state("owned blocks only",2,31,6,6,6);
    std::size_t comparisons=0;
    for (int epoch=0; epoch<3; ++epoch) {
      const bool reversed = epoch == 1;
      const int first_gid = 2*(reversed ? nranks-1-rank : rank);
      std::vector<int> owners(2*nranks);
      for (int g=0; g<2*nranks; ++g) owners[g] = reversed ? nranks-1-g/2 : g/2;
      const int peer = rank < 2 && nranks > 1 ? 1-rank : rank;
      const int peer_gid = 2*(reversed ? nranks-1-peer : peer);
      std::vector<bvals::DonorPoint> requested = {
          {peer_gid,1,2,3},{first_gid,3,2,1},{peer_gid,1,2,3},
          {peer_gid+1,2,1,3},{first_gid+1,1,3,2}};
      // The last rebuild has no donors at all on the last rank. That rank must
      // still participate in metadata exchange and service imports if requested.
      if (epoch == 2 && rank == nranks-1) requested.clear();
      std::vector<int> fields = {30,0,6,12,18,1,27};
      if (epoch == 1) {
        fields.clear();
        for (int f=0; f<28; ++f) if (f != 6 && f != 12) fields.push_back(f);
      }
      if (epoch == 2) {
        fields.clear();
        for (int f=0; f<31; ++f) fields.push_back(f);
      }
      exchange.Rebuild(first_gid,2,owners,epoch,requested,fields);
      Require(exchange.MeshSequence() == epoch,"mesh sequence not retained");
      Require(exchange.Fields() == fields,"field permutation not retained");
      Require(exchange.RequestOrdinals().size() == requested.size(),"request map size");
      if (!requested.empty()) {
        Require(exchange.RequestOrdinals()[0] == exchange.RequestOrdinals()[2],
                 "duplicate donor was not deduplicated");
      }
      if (rank >= 2 && epoch < 2) {
        Require(exchange.ReceivePeers() == 0 && exchange.SendPeers() == 0,
                 "self-only rank unexpectedly has a peer");
      }
      for (int stage=1; stage<=4; ++stage) {
        // Unused fields, cells, and all padding remain NaN. Only selected fields
        // on the expected donor region are initialized.
        auto host = Kokkos::create_mirror_view(state);
        Kokkos::deep_copy(host,std::numeric_limits<Real>::quiet_NaN());
        for (int m=0; m<2; ++m) for (int f : fields) {
          for (int k=1; k<=3; ++k) for (int j=1; j<=3; ++j) for (int i=1; i<=3; ++i) {
            host(m,f,k,j,i) = Expected(epoch,stage,first_gid+m,f,k,j,i);
          }
        }
        Kokkos::deep_copy(state,host);
        Require(exchange.StartReceives() == TaskStatus::complete,"receive start failed");
        Require(exchange.Receive() == TaskStatus::incomplete,"read before local pack");
        Require(exchange.ClearSend() == TaskStatus::incomplete,"send cleared before pack");
        Require(exchange.ClearRecv() == TaskStatus::incomplete,"recv cleared before unpack");
        Require(exchange.PackAndSend(state) == TaskStatus::complete,"donor pack failed");
        // Task re-entry after completion is harmless and must not send twice.
        Require(exchange.PackAndSend(state) == TaskStatus::complete,"repeat pack failed");
        Complete(exchange);
        auto values = Kokkos::create_mirror_view_and_copy(HostMemSpace(),exchange.Values());
        for (std::size_t n=0; n<requested.size(); ++n) {
          const auto point = requested[n];
          const int ordinal = exchange.RequestOrdinals()[n];
          for (std::size_t v=0; v<fields.size(); ++v) {
            const Real expected = Expected(epoch,stage,point.gid,fields[v],
                                            point.k,point.j,point.i);
            Require(values(ordinal,v) == expected,"mapped donor value changed or poisoned");
            ++comparisons;
          }
        }
      }
    }
    // Empty plan, zero buffers, and completed requests can be safely rebuilt.
    std::vector<int> owners(nranks);
    for (int r=0; r<nranks; ++r) owners[r]=r;
    exchange.Rebuild(rank,1,owners,99,{}, {0});
    Require(exchange.UniqueDonors().empty(),"empty plan retained donor points");
    exchange.StartReceives();
    exchange.PackAndSend(state);
    Complete(exchange);
#if MPI_PARALLEL_ENABLED
    if (nranks > 1) {
      // Rank one deliberately withholds its send until rank zero proves that
      // Receive/ClearRecv return incomplete. A blocking receive implementation
      // would deadlock this independent control-channel handshake.
      std::vector<bvals::DonorPoint> delayed;
      if (rank < 2) delayed.push_back({1-rank,1,1,1});
      exchange.Rebuild(rank,1,owners,100,delayed,{0});
      Kokkos::deep_copy(state,Real(rank+10));
      exchange.StartReceives();
      int token=1;
      if (rank == 1) MPI_Recv(&token,1,MPI_INT,0,79,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      exchange.PackAndSend(state);
      if (rank == 0) {
        Require(exchange.Receive() == TaskStatus::incomplete,
                 "MPI polling did not report the withheld donor");
        Require(exchange.ClearRecv() == TaskStatus::incomplete,
                 "receive clear passed while donor was withheld");
        MPI_Send(&token,1,MPI_INT,1,79,MPI_COMM_WORLD);
      }
      Complete(exchange);
      if (rank < 2) {
        auto values=Kokkos::create_mirror_view_and_copy(HostMemSpace(),exchange.Values());
        Require(values(0,0) == Real(11-rank),"delayed peer value failed");
        ++comparisons;
      }
    }
    unsigned long long local=comparisons,total=0;
    MPI_Reduce(&local,&total,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,0,MPI_COMM_WORLD);
#else
    auto total=comparisons;
#endif
    if (rank == 0) std::cout << "PASS native mapped exchange: ranks=" << nranks
                             << " comparisons=" << total << "\n";
  }
  Kokkos::finalize();
#if MPI_PARALLEL_ENABLED
  MPI_Finalize();
#endif
}
