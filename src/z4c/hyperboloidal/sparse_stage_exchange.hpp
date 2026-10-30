// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_SPARSE_STAGE_EXCHANGE_HPP_
#define Z4C_HYPERBOLOIDAL_SPARSE_STAGE_EXCHANGE_HPP_
#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "athena.hpp"

namespace z4c::hyperboloidal {
// The mesh owner advances these epochs before changing any cached topology,
// coordinates/maps, or ownership. All ranks must rebuild after such an event.
// A time/RK-stage change does not change the key. This is an exchange-plan
// lifecycle contract, not a dynamic AMR field-transfer implementation.
struct MeshPlanEpoch {
  uint64_t topology=0,geometry=0,partition=0;
  bool operator==(const MeshPlanEpoch&b)const {
    return topology==b.topology&&geometry==b.geometry&&partition==b.partition;
  }
};

// Persistent device pack/unpack plan. Storage retains global indices so all
// existing physical kernels keep their exact arithmetic and stencil order.
// Only owned and explicitly requested entries are current after Exchange.
// Full snapshots must gather owned entries separately, never dump this buffer.
class SparseStageExchange {
 public:
  using ExecutionSpace=Kokkos::DefaultExecutionSpace;
  using MemorySpace=ExecutionSpace::memory_space;
  using Buffer=Kokkos::View<Real*,MemorySpace>;
  using ConstBuffer=Kokkos::View<const Real*,MemorySpace>;
  using Indices=Kokkos::View<int*,MemorySpace>;
  struct Timing {double pack=0,mpi=0,unpack=0;};
 private:
  MPI_Comm comm_;
  int rank_=0,size_=0,nf_=0,total_=0,first_=0,local_=0;
  MeshPlanEpoch epoch_{};
  bool ready_=false;
  std::vector<int> send_counts_,send_offsets_,recv_counts_,recv_offsets_;
  Indices send_ids_,recv_ids_;
  Buffer send_,recv_;
  uint64_t builds_=0,calls_=0;
  Timing seconds_{};
  static double Now(){return MPI_Wtime();}
  void RequireCollective(bool valid,const char*reason)const {
    int bad=!valid,any=0;MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,comm_);
    if(any)throw std::invalid_argument(reason);
  }
  template<class Action>void ConstructCollective(Action action,const char*reason)const {
    bool okay=true;try{action();}catch(...){okay=false;}
    RequireCollective(okay,reason);
  }
  static Indices Upload(const char*name,const std::vector<int>&ids){
    Indices out(std::string(name),ids.size());auto host=Kokkos::create_mirror_view(out);
    for(size_t i=0;i<ids.size();++i)host(i)=ids[i];Kokkos::deep_copy(out,host);return out;
  }
 public:
  explicit SparseStageExchange(MPI_Comm comm=MPI_COMM_WORLD):comm_(comm){
    MPI_Comm_rank(comm_,&rank_);MPI_Comm_size(comm_,&size_);
  }
  void Rebuild(MeshPlanEpoch epoch,const std::vector<int>&cuts,int fields,
               const std::vector<int>&requested_points){
    // Mark the old plan invalid even if the new descriptor is rejected.
    ready_=false;
    bool valid=cuts.size()==size_t(size_+1)&&fields>0;
    if(valid){valid=cuts.front()==0&&cuts.back()>0&&cuts.back()<=INT_MAX/fields;
      for(int r=0;r<size_;++r)valid=valid&&cuts[r+1]>cuts[r];}
    RequireCollective(valid,"invalid sparse ownership descriptor");
    std::vector<int> lo,hi;
    ConstructCollective([&]{lo=cuts;hi=cuts;},"sparse ownership allocation failed on a rank");
    MPI_Allreduce(MPI_IN_PLACE,lo.data(),lo.size(),MPI_INT,MPI_MIN,comm_);
    MPI_Allreduce(MPI_IN_PLACE,hi.data(),hi.size(),MPI_INT,MPI_MAX,comm_);
    int fmin=fields,fmax=fields;MPI_Allreduce(MPI_IN_PLACE,&fmin,1,MPI_INT,MPI_MIN,comm_);
    MPI_Allreduce(MPI_IN_PLACE,&fmax,1,MPI_INT,MPI_MAX,comm_);
    uint64_t emin[3]={epoch.topology,epoch.geometry,epoch.partition},emax[3]={epoch.topology,epoch.geometry,epoch.partition};
    MPI_Allreduce(MPI_IN_PLACE,emin,3,MPI_UINT64_T,MPI_MIN,comm_);
    MPI_Allreduce(MPI_IN_PLACE,emax,3,MPI_UINT64_T,MPI_MAX,comm_);
    RequireCollective(lo==hi&&fmin==fmax&&std::equal(emin,emin+3,emax),"inconsistent sparse mesh/partition epoch");
    nf_=fields;total_=cuts.back();first_=cuts[rank_];local_=cuts[rank_+1]-first_;
    std::vector<int> needed;
    ConstructCollective([&]{needed=requested_points;std::sort(needed.begin(),needed.end());
      needed.erase(std::unique(needed.begin(),needed.end()),needed.end());},"sparse donor allocation failed on a rank");
    RequireCollective(needed.empty()||(needed.front()>=0&&needed.back()<total_),"sparse donor outside mesh");
    std::vector<int> requested,requests,incoming,request_offsets,incoming_offsets;
    ConstructCollective([&]{requests.assign(size_,0);incoming.assign(size_,0);
      request_offsets.assign(size_,0);incoming_offsets.assign(size_,0);
      for(int point:needed){const int owner=std::upper_bound(cuts.begin(),cuts.end(),point)-cuts.begin()-1;
        if(owner!=rank_){requested.push_back(point);++requests[owner];}}
    },"sparse request allocation failed on a rank");
    MPI_Alltoall(requests.data(),1,MPI_INT,incoming.data(),1,MPI_INT,comm_);
    int64_t sent64=0,received64=0;
    for(int r=0;r<size_;++r){sent64+=incoming[r];received64+=requests[r];}
    RequireCollective(sent64<=INT_MAX/nf_&&received64<=INT_MAX/nf_,"sparse message exceeds MPI int count");
    int sent=0,received=0;
    for(int r=0;r<size_;++r){request_offsets[r]=received;incoming_offsets[r]=sent;received+=requests[r];sent+=incoming[r];}
    std::vector<int> offered;
    ConstructCollective([&]{offered.resize(sent);},"sparse receive-index allocation failed on a rank");
    MPI_Alltoallv(requested.data(),requests.data(),request_offsets.data(),MPI_INT,
                  offered.data(),incoming.data(),incoming_offsets.data(),MPI_INT,comm_);
    valid=true;for(int&point:offered){valid=valid&&point>=first_&&point<first_+local_;point-=first_;}
    RequireCollective(valid,"sparse request sent to wrong owner");
    ConstructCollective([&]{
      send_ids_=Upload("persistent sparse send indices",offered);recv_ids_=Upload("persistent sparse receive indices",requested);
      send_=Buffer("persistent sparse send values",std::max(1,sent*nf_));
      recv_=Buffer("persistent sparse receive values",std::max(1,received*nf_));
      send_counts_=incoming;send_offsets_=incoming_offsets;recv_counts_=requests;recv_offsets_=request_offsets;
      for(auto*v:{&send_counts_,&send_offsets_,&recv_counts_,&recv_offsets_})for(int&n:*v)n*=nf_;
    },"sparse device-plan allocation failed on a rank");
    epoch_=epoch;ready_=true;++builds_;
  }
  Timing Exchange(ConstBuffer owned,Buffer global,MeshPlanEpoch epoch,bool poison_unused=false){
    const double started=Now();
    const auto ob=reinterpret_cast<uintptr_t>(owned.data()),gb=reinterpret_cast<uintptr_t>(global.data());
    const bool separate=ob+owned.extent(0)*sizeof(Real)<=gb||gb+global.extent(0)*sizeof(Real)<=ob;
    RequireCollective(ready_&&epoch==epoch_&&owned.extent(0)==size_t(nf_)*local_
                      &&global.extent(0)==size_t(nf_)*total_
                      &&separate,"stale sparse plan or incompatible state storage");
    const int nf=nf_,first=first_;const auto ids=send_ids_;const auto send=send_;
    Kokkos::parallel_for("sparse stage pack",Kokkos::RangePolicy<ExecutionSpace>(0,send_ids_.extent(0)*nf),
      KOKKOS_LAMBDA(int i){send(i)=owned(nf*ids(i/nf)+i%nf);});
    Kokkos::fence();const double packed=Now();
    MPI_Alltoallv(send_.data(),send_counts_.data(),send_offsets_.data(),MPI_ATHENA_REAL,
                  recv_.data(),recv_counts_.data(),recv_offsets_.data(),MPI_ATHENA_REAL,comm_);
    const double communicated=Now();
    if(poison_unused)Kokkos::deep_copy(global,std::numeric_limits<Real>::quiet_NaN());
    Kokkos::parallel_for("sparse owned stage copy",Kokkos::RangePolicy<ExecutionSpace>(0,owned.extent(0)),
      KOKKOS_LAMBDA(int i){global(nf*first+i)=owned(i);});
    const auto targets=recv_ids_;const auto recv=recv_;
    Kokkos::parallel_for("sparse stage unpack",Kokkos::RangePolicy<ExecutionSpace>(0,recv_ids_.extent(0)*nf),
      KOKKOS_LAMBDA(int i){global(nf*targets(i/nf)+i%nf)=recv(i);});
    Kokkos::fence();const Timing sample{packed-started,communicated-packed,Now()-communicated};
    seconds_.pack+=sample.pack;seconds_.mpi+=sample.mpi;seconds_.unpack+=sample.unpack;++calls_;return sample;
  }
  uint64_t Builds()const{return builds_;}
  uint64_t Calls()const{return calls_;}
  size_t SendPoints()const{return send_ids_.extent(0);}
  size_t ReceivePoints()const{return recv_ids_.extent(0);}
  Timing Seconds()const{return seconds_;}
};
}
#endif
