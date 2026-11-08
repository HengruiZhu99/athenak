// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Test-only observer of the original native Driver TaskLists; no legacy evolution.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "globals.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "outputs/outputs.hpp"
#include "z4c_hyperboloidal/mapped_halo.hpp"
#include "z4c_hyperboloidal/remesh.hpp"
#include "z4c_hyperboloidal/z4c_hyperboloidal.hpp"
namespace native=z4c_hyperboloidal;
constexpr int nf=31,nr=8,n=12;
void Require(bool yes,const char *message) {if(!yes)throw std::runtime_error(message);}
TaskID AllTasks(const std::shared_ptr<TaskList>&list) {
  TaskID result(0);for(int i=1;i<=list->Size();++i)result=result|TaskID(i);return result;
}
struct Dump {
  std::ofstream file;
  Dump(const std::string &label,int region,int count,int values) {
    file.open(label+(region?"-scri-rank":"-bulk-rank")+
              std::to_string(global_variable::my_rank)+".bin",std::ios::binary);
    Require(bool(file),"cannot open trace dump");
    const char magic[16]="NATIVE_TRACE_V1";
    const std::uint64_t size=count;const std::uint32_t shape[4]={6,std::uint32_t(values),std::uint32_t(region),1};
    file.write(magic,16);file.write(reinterpret_cast<const char*>(&size),8);
    file.write(reinterpret_cast<const char*>(shape),16);
  }
  void Row(const std::array<std::int32_t,6>&key,const Real *values,int count) {
    for(int f=0;f<count;++f)Require(std::isfinite(values[f]),"nonfinite owned trace value");
    file.write(reinterpret_cast<const char*>(key.data()),sizeof(key));
    file.write(reinterpret_cast<const char*>(values),count*sizeof(Real));
  }
  void Close() {file.close();Require(bool(file),"trace write failed");}
};
struct Stage {int cycle,stage;Real rhs_time,state_time,compatibility,clock,chi,rhs_max;};
struct Observer {
  Mesh &mesh;native::Z4cHyperboloidal &p;bool poison;
  int before=0,after=0,accepted=0;
  std::uint64_t unowned_cells=0;
  std::vector<std::array<std::int32_t,6>> bulk_keys,edge_keys;
  std::vector<Stage> stages;
  Observer(Mesh &mesh_,native::Z4cHyperboloidal &p_,bool poison_)
      :mesh(mesh_),p(p_),poison(poison_) {
    native::NativeDonorSelector selector(mesh,p.origin);std::vector<int> offset(mesh.nmb_total+1);
    for(int gid=0;gid<mesh.nmb_total;++gid){int count=1;for(int a=0;a<3;++a)
      count*=selector.Upper(gid,a)-selector.Lower(gid,a);offset[gid+1]=offset[gid]+count;}
    Require(offset.back()==229392&&mesh.nmb_total==106,"trace requires fixed N12 topology");
    std::uint64_t bounded=0;
    for(int m=0;m<p.nblocks;++m) {const auto b=p.evolved_bounds.h_view(m);std::uint64_t count=1;
      for(int a=0;a<3;++a)count*=b.hi[a]-b.lo[a]+1;bounded+=count;}
    Require(bounded==std::uint64_t(p.npoints),"evolved bounds differ from owned work list");
    unowned_cells=std::uint64_t(p.nblocks)*p.storage[0]*p.storage[1]*p.storage[2]-bounded;
    for(int q=0;q<p.npoints;++q) {
      const auto c=p.points.h_view(q);const auto b=p.evolved_bounds.h_view(c.m);
      Require(c.i>=b.lo[0]&&c.i<=b.hi[0]&&c.j>=b.lo[1]&&c.j<=b.hi[1]
          &&c.k>=b.lo[2]&&c.k<=b.hi[2],"owned evolved overlap outside preserved bounds");
      const int gid=mesh.pmb_pack->gids+c.m,i=c.i-p.origin,j=c.j-p.origin,k=c.k-p.origin;
      const int wi=selector.Upper(gid,0)-selector.Lower(gid,0),wj=selector.Upper(gid,1)-selector.Lower(gid,1);
      const int ordinal=offset[gid]+((k-selector.Lower(gid,2))*wj+j-selector.Lower(gid,1))*wi+i-selector.Lower(gid,0);
      bulk_keys.push_back({ordinal,gid,k,j,i,selector.Map(gid).face});
    }
    for(int q=0;q<p.nsurface_points;++q) {
      const auto c=p.surface_points.h_view(q);const int gid=mesh.pmb_pack->gids+p.face_blocks.h_view(c.face);
      const auto &map=selector.Map(gid);Require(map.outer_surface,"nonouter surface owner");
      edge_keys.push_back({(map.face*n+c.k)*n+c.j,gid,c.k,c.j,n,map.face});
    }
    const auto g=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.geometry);
    const auto s=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.surface_geometry);
    Dump bulk("geometry",0,p.npoints,3),edge("geometry",1,p.nsurface_points,3);
    for(int q=0;q<p.npoints;++q)bulk.Row(bulk_keys[q],g(q).x,3);
    for(int q=0;q<p.nsurface_points;++q)edge.Row(edge_keys[q],s(q).x,3);
    bulk.Close();edge.Close();
  }
  void Snapshot(const std::string &label,bool rhs=false) const {
    const auto u=Kokkos::create_mirror_view_and_copy(HostMemSpace(),rhs?p.u_rhs:p.u0);
    const auto s=Kokkos::create_mirror_view_and_copy(HostMemSpace(),rhs?p.s_rhs:p.s0);
    const auto records=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.chi_records);
    const auto sr=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.surface_chi_records);
    const int count=rhs?nf:nf+nr;
    Dump bulk(label,0,p.npoints,count),edge(label,1,p.nsurface_points,count);
    for(int q=0;q<p.npoints;++q){const auto c=p.points.h_view(q);Real v[nf+nr];
      for(int f=0;f<nf;++f)v[f]=u(c.m,f,c.k,c.j,c.i);
      if(!rhs)native::PackChiRecord(records(q),v+nf);bulk.Row(bulk_keys[q],v,count);}
    for(int q=0;q<p.nsurface_points;++q){const auto c=p.surface_points.h_view(q);Real v[nf+nr];
      for(int f=0;f<nf;++f)v[f]=s(c.face,f,c.k,c.j);
      if(!rhs)native::PackChiRecord(sr(q),v+nf);edge.Row(edge_keys[q],v,count);}
    bulk.Close();edge.Close();
  }
  TaskStatus Before(Driver *driver,int stage) {
    Require(stage==before%4+1&&mesh.ncycle==before/4,"unexpected before-stage order");++before;
    Require(driver->UsesClassicalRK4()&&mesh.dt==p.requested_dt,"changed integrator or dt");
    if(mesh.ncycle==0)Snapshot("stage"+std::to_string(stage)+"-input");
    if(poison){const Real nan=std::numeric_limits<Real>::quiet_NaN();
      Kokkos::deep_copy(p.u_rhs,nan);Kokkos::deep_copy(p.s_rhs,nan);
      Kokkos::deep_copy(p.residual,nan);Kokkos::deep_copy(p.rhs_status,-777);
      const auto state=p.u0;const auto bounds=p.evolved_bounds.d_view;
      const int nk=p.storage[2],nj=p.storage[1],ni=p.storage[0];
      Kokkos::parallel_for("trace poison only unowned ghosts",Kokkos::RangePolicy<>(0,p.nblocks*nk*nj*ni),
        KOKKOS_LAMBDA(int q){const int i=q%ni,j=q/ni%nj,k=q/(ni*nj)%nk,m=q/(ni*nj*nk);const auto b=bounds(m);
          if(i<b.lo[0]||i>b.hi[0]||j<b.lo[1]||j>b.hi[1]||k<b.lo[2]||k>b.hi[2])
            for(int f=0;f<nf;++f)state(m,f,k,j,i)=nan;});
    }
    return TaskStatus::complete;
  }
  TaskStatus After(Driver *driver,int stage) {
    Require(stage==after%4+1&&mesh.ncycle==after/4&&before==after+1,"unexpected after-stage order");++after;
    const Real rhs_fraction[4]={0,.5,.5,1},state_fraction[4]={.5,.5,1,1};
    Require(driver->stage_time==mesh.time+mesh.dt*rhs_fraction[stage-1]
        &&driver->stage_state_time==mesh.time+mesh.dt*state_fraction[stage-1],"changed stage clock");
    // The original ExpRKUpdate reads u_rhs/s_rhs and only writes state/accumulator;
    // ProjectSurface writes s0 only. Thus these are the just-consumed stage rates.
    if(mesh.ncycle==0){Snapshot("stage"+std::to_string(stage)+"-output");
      if(stage==1)Snapshot("first-rhs",true);}
    const auto points=p.points.d_view;const auto faces=p.surface_points.d_view;
    const auto u=p.u0,r=p.u_rhs;const auto s=p.s0,sr=p.s_rhs;const auto status=p.rhs_status;
    int bad_bulk=0,bad_edge=0;Real min_bulk=1e300,min_edge=1e300,max_bulk=0,max_edge=0;
    Kokkos::parallel_reduce("trace owned bulk checks",Kokkos::RangePolicy<>(0,p.npoints),
      KOKKOS_LAMBDA(int q,int&bad,Real&minimum,Real&maximum){const auto c=points(q);
        minimum=fmin(minimum,u(c.m,0,c.k,c.j,c.i));if(status(q)!=0)++bad;
        for(int f=0;f<nf;++f){if(!Kokkos::isfinite(u(c.m,f,c.k,c.j,c.i))||!Kokkos::isfinite(r(c.m,f,c.k,c.j,c.i)))++bad;
          maximum=fmax(maximum,fabs(r(c.m,f,c.k,c.j,c.i)));}},
      Kokkos::Sum<int>(bad_bulk),Kokkos::Min<Real>(min_bulk),Kokkos::Max<Real>(max_bulk));
    Kokkos::parallel_reduce("trace owned scri checks",Kokkos::RangePolicy<>(0,p.nsurface_points),
      KOKKOS_LAMBDA(int q,int&bad,Real&minimum,Real&maximum){const auto c=faces(q);minimum=fmin(minimum,s(c.face,0,c.k,c.j));
        for(int f=0;f<nf;++f){if(!Kokkos::isfinite(s(c.face,f,c.k,c.j))||!Kokkos::isfinite(sr(c.face,f,c.k,c.j)))++bad;
          maximum=fmax(maximum,fabs(sr(c.face,f,c.k,c.j)));}},
      Kokkos::Sum<int>(bad_edge),Kokkos::Min<Real>(min_edge),Kokkos::Max<Real>(max_edge));
    int bad=bad_bulk+bad_edge;MPI_Allreduce(MPI_IN_PLACE,&bad,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    Require(bad==0,"nonfinite/unwritten owned state or RHS after native stage");
    Real maxima[3]={p.StageMaxCompatibility(),p.StageMaxClock(),std::max(max_bulk,max_edge)};
    Real minimum=std::min(min_bulk,min_edge);
    MPI_Allreduce(MPI_IN_PLACE,maxima,3,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&minimum,1,MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);
    Require(std::isfinite(maxima[0])&&std::isfinite(maxima[1])&&std::isfinite(maxima[2])
        &&std::isfinite(minimum)&&maxima[0]<=1e-7&&minimum>=p.chi_floor,"native stage admissibility failed");
    stages.push_back({mesh.ncycle,stage,driver->stage_time,driver->stage_state_time,maxima[0],maxima[1],minimum,maxima[2]});
    return TaskStatus::complete;
  }
  TaskStatus Accepted(Driver *driver,int stage) {
    Require(stage==1&&mesh.ncycle==accepted&&after==4*(accepted+1),"unexpected accepted-step order");++accepted;
    if(accepted==1||accepted==8)Snapshot("accepted"+std::to_string(accepted));
    return TaskStatus::complete;
  }
};
int main(int argc,char **argv) {
  MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);Kokkos::initialize(argc,argv);
  try{{
    Require(argc==3,"native_stage_trace INPUT clean|poison (output in cwd)");
    const std::string mode=argv[2];Require(mode=="clean"||mode=="poison","invalid trace mode");
    ParameterInput pin;IOWrapper input;input.Open(argv[1],IOWrapper::FileMode::read);pin.LoadFromFile(input);input.Close();
    Mesh mesh(&pin);mesh.BuildTreeFromScratch(&pin);mesh.AddCoordinatesAndPhysics(&pin);
    mesh.pgen=std::make_unique<ProblemGenerator>(&pin,&mesh);auto &p=*mesh.pmb_pack->pz4c_hyperboloidal;
    Outputs output(&pin,&mesh);Kokkos::Timer timer;Driver driver(&pin,&mesh,1000,&timer);
    Require(driver.UsesClassicalRK4()&&driver.nlim==8,"trace requires eight native classical RK4 steps");
    driver.Initialize(&mesh,&pin,&output,false);
    Require(p.StagePlansReady()&&mesh.dt==p.requested_dt,"native initialization missing plans/exact dt");
    Observer observe(mesh,p,mode=="poison");observe.Snapshot("accepted0");
    auto before=mesh.pmb_pack->tl_map.at("before_stagen"),after=mesh.pmb_pack->tl_map.at("after_stagen");
    auto accepted=mesh.pmb_pack->tl_map.at("after_timeintegrator");
    auto dep_before=AllTasks(before),dep_after=AllTasks(after),dep_accepted=AllTasks(accepted);
    before->AddTask([&](Driver*d,int s){return observe.Before(d,s);},dep_before);
    after->AddTask([&](Driver*d,int s){return observe.After(d,s);},dep_after);
    accepted->AddTask([&](Driver*d,int s){return observe.Accepted(d,s);},dep_accepted);
    driver.Execute(&mesh,&pin,&output,false);
    Require(mesh.ncycle==8&&observe.before==32&&observe.after==32&&observe.accepted==8,"incomplete native trace");
    Require(mesh.dt==p.requested_dt&&mesh.dt_last_completed==p.requested_dt,"changed final dt");
    std::uint64_t counts[4]={std::uint64_t(p.npoints),std::uint64_t(p.nsurface_points),observe.unowned_cells,std::uint64_t(p.nsurface_points==0)};
    MPI_Allreduce(MPI_IN_PLACE,counts,4,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
    Require(counts[0]==229392&&counts[1]==864,"global trace workload changed");
    if(!global_variable::my_rank){std::ofstream report("results.json");report<<std::setprecision(17)
      <<"{\"schema\":\"native-stage-trace-v1\",\"backend\":\""<<DevExeSpace::name()<<"\",\"ranks\":"<<global_variable::nranks
      <<",\"mode\":\""<<mode<<"\",\"passed\":true,\"steps\":8,\"dt\":"<<mesh.dt<<",\"time\":"<<mesh.time
      <<",\"bulk_points\":"<<counts[0]<<",\"scri_points\":"<<counts[1]<<",\"unowned_cells_per_stage\":"<<counts[2]
      <<",\"zero_surface_ranks\":"<<counts[3]<<",\"stages\":[";
      for(std::size_t i=0;i<observe.stages.size();++i){const auto&s=observe.stages[i];report<<(i?",":"")
        <<"{\"cycle\":"<<s.cycle<<",\"stage\":"<<s.stage<<",\"rhs_time\":"<<s.rhs_time<<",\"state_time\":"<<s.state_time
        <<",\"compatibility\":"<<s.compatibility<<",\"clock\":"<<s.clock<<",\"minimum_chi\":"<<s.chi<<",\"maximum_rhs\":"<<s.rhs_max<<'}';}
      report<<"]}\n";report.close();Require(bool(report),"trace result write failed");
      std::cout<<"PASS native stage trace "<<mode<<" eight steps32 stages\n";
    }
  }}catch(const std::exception&e){std::cerr<<"rank"<<global_variable::my_rank<<": "<<e.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);}
  Kokkos::finalize();MPI_Finalize();return 0;
}
