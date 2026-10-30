// Exercise the real Evolution transaction, including checkpoint writing after
// recovery. The renamed CLI entry point is not invoked by this fixture.
#define main HyperbolicEvolutionProgramMain
#include "coupled_mpi.cpp"
#undef main

static void RequireTransaction(bool okay,const char*message){
  int bad=!okay;MPI_Allreduce(MPI_IN_PLACE,&bad,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
  if(bad)throw std::runtime_error(message);
}
static V AcceptedSnapshot(Evolution&e){
  auto volume=e.CompleteState(e.state);
  Buffer comp("transaction complete compensation",31*e.total),edge("transaction surface compensation",31*e.surface_total);
  e.Gather(e.compensation,comp);e.Gather(e.surface_compensation,edge,true);
  V result;for(auto b:{volume,comp,e.surface,edge}){const auto values=e.Download(b);result.insert(result.end(),values.begin(),values.end());}
  return result;
}
static bool Exact(const V&a,const V&b){return a.size()==b.size()&&std::memcmp(a.data(),b.data(),a.size()*sizeof(Real))==0;}
int main(int argc,char**argv){
  MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);
  Kokkos::initialize(argc,argv);int code=0;
  try{
    RequireTransaction(argc==2&&global_variable::nranks==2,"usage: two ranks and a fresh output prefix");
    Evolution e(12,2,1);V seed=e.initial_state;
    for(int p=0;p<e.total;++p){const Real z=(e.Model().Radius(e.Model().grid.geometry[p].x)-.4)/.12;
      if(std::abs(z)<1)seed[31*p+18]=.01*std::exp(1-1/(1-z*z));}
    e.Upload(seed);e.Step(.00025);
    auto compensation=e.Download(e.compensation);int nonzero=0;for(Real x:compensation)nonzero+=x!=0;
    MPI_Allreduce(MPI_IN_PLACE,&nonzero,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    RequireTransaction(nonzero>0,"transaction fixture lacks nonzero RK compensation");
    const auto before=AcceptedSnapshot(e);const auto cuts=e.ownership_cuts;
    auto invalid=cuts;if(e.rank==0)invalid[1]=0;bool rejected=false;
    try{e.Repartition(invalid);}catch(const std::invalid_argument&){rejected=true;}
    RequireTransaction(rejected&&e.rejected_repartitions==1&&e.ownership_changes==0&&e.ownership_cuts==cuts,
                       "one-rank invalid ownership did not recover collectively");
    RequireTransaction(Exact(before,AcceptedSnapshot(e)),"rejected repartition changed accepted state or compensation");
    e.WriteCheckpoint(std::string(argv[1])+".recovered.checkpoint",1,.00025,true);
    auto changed=cuts;changed[1]=e.total/10;e.Repartition(changed);
    RequireTransaction(Exact(before,AcceptedSnapshot(e)),"repartition changed accepted state or compensation");
    e.RebuildCurrentPlans();RequireTransaction(e.ownership_cuts==changed,"fixed-layout rebuild reset changed ownership");
    RequireTransaction(Exact(before,AcceptedSnapshot(e)),"post-repartition rebuild changed state");
    e.Step(.00025);const auto continued=AcceptedSnapshot(e);
    Evolution control(12,2,1);control.Upload(seed);control.Step(.00025);control.Step(.00025);
    RequireTransaction(Exact(continued,AcceptedSnapshot(control)),"continued trajectory differs after transaction/recovery");
    auto stress=e.Download(e.CompleteState(e.state));stress[31*5000+7]=1e5;stress[31*5000+10]=-1e5;
    e.Upload(stress);control.Upload(stress);e.Repartition(cuts);
    std::string errors[2];int index=0;
    for(auto*run:{&e,&control}){try{run->Step(.00025);}catch(const std::exception&ex){errors[index]=ex.what();}++index;}
    RequireTransaction(!errors[0].empty()&&errors[0]==errors[1],"repartitioned stress stage was not rejected identically");
    RequireTransaction(Exact(AcceptedSnapshot(e),AcceptedSnapshot(control)),"rollback differs after repartition");
    RequireTransaction(Exact(e.Download(e.CompleteState(e.failed_state)),control.Download(control.CompleteState(control.failed_state))),
                       "complete failed-stage capture differs after repartition");
    e.WriteCheckpoint(std::string(argv[1])+".failed.checkpoint",1,.00025,false);
    if(e.rank==0)std::cout<<"PASS: one-rank invalid repartition recovered; complete accepted fields and nonzero RK compensation preserved exactly; checkpoint after recovery; changed-ownership rebuild and continued trajectory exact; repartitioned rollback and complete failed-stage capture exact; ranks=2 backend="<<Kokkos::DefaultExecutionSpace::name()<<std::endl;
  }catch(const std::exception&ex){std::cerr<<global_variable::my_rank<<": "<<ex.what()<<std::endl;code=1;}
  MPI_Allreduce(MPI_IN_PLACE,&code,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);Kokkos::finalize();MPI_Finalize();return code;
}
