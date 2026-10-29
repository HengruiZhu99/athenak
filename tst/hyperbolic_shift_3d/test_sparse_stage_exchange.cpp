#include <mpi.h>
#include <iostream>
#include <fstream>
#include <functional>
#include "z4c/hyperboloidal/sparse_stage_exchange.hpp"
using Plan=z4c::hyperboloidal::SparseStageExchange;
using Epoch=z4c::hyperboloidal::MeshPlanEpoch;
int main(int argc,char**argv){
  MPI_Init(&argc,&argv);Kokkos::initialize(argc,argv);int rank,size,code=0;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
  try{
    Plan plan;Epoch epoch{};
    auto require=[&](bool okay){int bad=!okay;MPI_Allreduce(MPI_IN_PLACE,&bad,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);if(bad)throw std::runtime_error("sparse exchange test failed");};
    auto rejected=[&](const std::function<void()>&f){bool caught=false;try{f();}catch(const std::invalid_argument&){caught=true;}require(caught);};
    auto test=[&](int round,int points){
      constexpr int nf=3;
      std::vector<int> cuts(size+1);for(int r=0;r<=size;++r)cuts[r]=points*r/size;
      // A changed contiguous partition, including unequal local lengths.
      if(round==3&&size>1)++cuts[1];
      const int first=cuts[rank],count=cuts[rank+1]-first;
      std::vector<int> needed;
      if(round!=1)for(int i=0;i<points;++i)if((i+rank)%7==0||(rank==0&&i%2==0))needed.push_back(i);
      if(!needed.empty()){needed.push_back(needed.front());std::reverse(needed.begin(),needed.end());}
      plan.Rebuild(epoch,cuts,nf,needed);
      Plan::Buffer owned("test owned",nf*count),global("test partial global",nf*points);
      auto host=Kokkos::create_mirror_view(owned);
      for(int step=0;step<3;++step){
        for(int i=0;i<nf*count;++i)host(i)=Real(nf*first+i)+Real(step)/8;
        Kokkos::deep_copy(owned,host);plan.Exchange(owned,global,epoch,true);
        auto result=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),global);
        bool okay=true;
        for(int p=0;p<points;++p){const bool needed_here=(p>=first&&p<first+count)||std::find(needed.begin(),needed.end(),p)!=needed.end();
          for(int f=0;f<nf;++f)okay=okay&&(needed_here?result(nf*p+f)==Real(nf*p+f)+Real(step)/8:std::isnan(result(nf*p+f)));}
        require(okay);
      }
      // A stale key on just one rank must fail collectively, without a hang.
      auto stale=epoch;if(rank==0)++stale.geometry;
      rejected([&]{plan.Exchange(owned,global,stale);});
      plan.Exchange(owned,global,epoch,true);
      rejected([&]{Plan::Buffer alias=Kokkos::subview(global,std::make_pair(size_t(nf*first),size_t(nf*(first+count))));plan.Exchange(alias,global,epoch,true);});
      auto invalid=needed;if(rank==0)invalid.push_back(points);
      rejected([&]{plan.Rebuild(epoch,cuts,nf,invalid);});
      rejected([&]{plan.Exchange(owned,global,epoch);});
      plan.Rebuild(epoch,cuts,nf,needed);plan.Exchange(owned,global,epoch,true);
      auto wrongcuts=cuts;if(rank==0&&size>1)++wrongcuts.back();
      if(size>1)rejected([&]{plan.Rebuild(epoch,wrongcuts,nf,needed);});
    };
    test(0,97);++epoch.topology;test(1,113);++epoch.geometry;test(2,113);++epoch.partition;test(3,113);
    require(plan.Builds()==8&&plan.Calls()==20);
    if(rank==0)std::cout<<"PASS: sparse donor values, duplicate/empty requests, poisoned unused storage, refinement/geometry/repartition rebuild, one-rank stale key, invalid descriptor and alias rejection; ranks="<<size<<std::endl;
  }catch(const std::exception&e){std::cerr<<rank<<": "<<e.what()<<std::endl;code=1;}
  MPI_Allreduce(MPI_IN_PLACE,&code,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
  if(argc==3&&std::string(argv[1])=="--report"&&rank==0){
    std::ofstream report(argv[2]);
    report<<"{\"status\":\""<<(code?"failed":"completed")<<"\",\"backend\":\""
      <<Kokkos::DefaultExecutionSpace::name()<<"\",\"mpi_size\":"<<size
      <<",\"lifecycle_passed\":"<<(code?"false":"true")<<"}\n";
    report.close();if(!report)code=1;
  }
  MPI_Allreduce(MPI_IN_PLACE,&code,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
  Kokkos::finalize();MPI_Finalize();return code;
}
