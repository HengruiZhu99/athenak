// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Four original Driver cycles with actual accepted-step refinement/coarsening.
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
#include "mesh/mesh.hpp"
#include "mesh/refinement_criteria.hpp"
#include "driver/driver.hpp"
#include "outputs/outputs.hpp"
#include "pgen/pgen.hpp"
#include "z4c_hyperboloidal/remesh.hpp"
#include "z4c_hyperboloidal/spherical_policy.hpp"

namespace h=z4c_hyperboloidal;
namespace {
void Require(bool yes,const char *what) {if (!yes)throw std::runtime_error(what);}
TaskID AllTasks(const std::shared_ptr<TaskList> &list) {
  TaskID result(0);for (int i=1;i<=list->Size();++i)result=result|TaskID(i);return result;
}
bool Selected(const chart_atlas::BlockMap &map) {
  if (map.chart_id==0)
    return map.hi[0]<=-.1/3+1e-15 && map.hi[1]<=-.1/3+1e-15 &&
           map.hi[2]<=-.1/3+1e-15;
  return map.chart_id==1 && (map.layer==0 || map.layer==11);
}
std::uint64_t refreshed_neighbors=0;
void Criterion(MeshBlockPack *pack) {
  auto &mesh=*pack->pmesh;auto &p=*pack->pz4c_hyperboloidal;
  const auto u=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.u0);
  const auto s=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.s0);
  const int o=p.origin,n=p.n[0];
  // The preceding accepted observer poisons every ghost. This callback runs
  // after the real derivative criterion's accepted-state boundary refresh.
  for (int m=0;m<p.nblocks;++m)for (int k=o;k<o+n;++k)
  for (int j=o;j<o+n;++j)for (int i=o;i<o+n;++i) {
    const int face=p.block_faces.h_view(m);
    const Real qp=face>=0 && i==o+n-1 ? s(face,0,k-o,j-o) : u(m,0,k,j,i+1);
    for (Real q:{qp,u(m,0,k,j,i-1),u(m,0,k,j+1,i),u(m,0,k,j-1,i),
                 u(m,0,k+1,j,i),u(m,0,k-1,j,i)}) {
      Require(std::isfinite(q),"accepted derivative criterion saw an unfilled ghost");
      ++refreshed_neighbors;
    }
  }
  for (int m=0;m<pack->nmb_thispack;++m) {
    const int gid=pack->gids+m,level=mesh.lloc_eachmb[gid].level;
    const auto map=pack->pmb->mb_map.h_view(m);
    mesh.pmr->refine_flag.h_view(gid)=Selected(map)
        ? (mesh.ncycle==1 && level==mesh.root_level ? 1
           : mesh.ncycle==3 && level==mesh.root_level+1 ? -1 : 0) : 0;
  }
}
std::array<Real,8> Totals(const h::Z4cHyperboloidal &p,bool surface) {
  const auto data=Kokkos::create_mirror_view_and_copy(HostMemSpace(),
      surface ? p.surface_chi_records : p.chi_records);
  std::array<Real,8> out={1e300,1e300,0,0,0,0,0,1e300};
  for (std::size_t n=0;n<data.extent(0);++n) {
    Real q[8];h::PackChiRecord(data(n),q);
    for (int a=0;a<8;++a) {
      Require(std::isfinite(q[a]),"nonfinite chi record");
      if (a==0 || a==1 || a==7)out[a]=std::min(out[a],q[a]);
      else if (a==5)out[a]=std::max(out[a],q[a]);else out[a]+=q[a];
    }
  }
  for (int a=0;a<8;++a)MPI_Allreduce(MPI_IN_PLACE,&out[a],1,MPI_DOUBLE,
      a==0 || a==1 || a==7 ? MPI_MIN : a==5 ? MPI_MAX : MPI_SUM,MPI_COMM_WORLD);
  return out;
}
void PoisonGhosts(h::Z4cHyperboloidal &p) {
  const auto u=p.u0;const auto bounds=p.evolved_bounds.d_view;
  const int ni=p.storage[0],nj=p.storage[1],nk=p.storage[2];
  const Real nan=std::numeric_limits<Real>::quiet_NaN();
  Kokkos::parallel_for("AMR evolution poison ghosts",
      Kokkos::RangePolicy<>(0,p.nblocks*ni*nj*nk),KOKKOS_LAMBDA(int q) {
    const int i=q%ni,j=q/ni%nj,k=q/(ni*nj)%nk,m=q/(ni*nj*nk);const auto b=bounds(m);
    if (i<b.lo[0] || i>b.hi[0] || j<b.lo[1] || j>b.hi[1] || k<b.lo[2] || k>b.hi[2])
      for (int f=0;f<31;++f)u(m,f,k,j,i)=nan;
  });
}
void Dump(Mesh &mesh) {
  const auto &p=*mesh.pmb_pack->pz4c_hyperboloidal;
  const auto u=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.u0);
  const auto s=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.s0);
  const auto chi=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.chi_records);
  const auto schi=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.surface_chi_records);
  const std::string stem="cycle"+std::to_string(mesh.ncycle)+"-rank"+
      std::to_string(global_variable::my_rank);
  std::ofstream file(stem+".bin",std::ios::binary),meta(stem+".txt");
  Require(bool(file)&&bool(meta),"cannot open complete AMR evolution snapshot");
  const auto write=[&](int m,int kind,int i,int j,int k,const Real *v) {
    const auto &loc=mesh.lloc_eachmb[mesh.pmb_pack->gids+m];
    const std::int32_t key[9]={loc.chart_id,loc.level,loc.lx1,loc.lx2,loc.lx3,kind,k,j,i};
    for (int f=0;f<39;++f)Require(std::isfinite(v[f]),"snapshot contains nonfinite value");
    file.write(reinterpret_cast<const char *>(key),sizeof(key));
    file.write(reinterpret_cast<const char *>(v),39*sizeof(Real));
  };
  for (int m=0;m<p.nblocks;++m) {
    const auto &loc=mesh.lloc_eachmb[mesh.pmb_pack->gids+m];
    const auto b=p.evolved_bounds.h_view(m);
    meta<<loc.chart_id<<' '<<loc.level<<' '<<loc.lx1<<' '<<loc.lx2<<' '<<loc.lx3;
    for (int a=0;a<3;++a)meta<<' '<<b.lo[a]<<' '<<b.hi[a];
    meta<<' '<<(p.block_faces.h_view(m)>=0)<<'\n';
  }
  for (int point=0;point<p.npoints;++point) {
    const auto c=p.points.h_view(point);Real v[39];
    for (int f=0;f<31;++f)v[f]=u(c.m,f,c.k,c.j,c.i);
    h::PackChiRecord(chi(point),v+31);write(c.m,0,c.i,c.j,c.k,v);
  }
  for (int point=0;point<p.nsurface_points;++point) {
    const auto c=p.surface_points.h_view(point);Real v[39];
    for (int f=0;f<31;++f)v[f]=s(c.face,f,c.k,c.j);
    h::PackChiRecord(schi(point),v+31);
    write(p.face_blocks.h_view(c.face),1,p.n[0],c.j,c.k,v);
  }
  Require(bool(file)&&bool(meta),"snapshot write failed");
  std::uint64_t counts[3]={std::uint64_t(p.npoints),std::uint64_t(p.nsurface_points),
                           std::uint64_t(p.nfaces==0)};
  MPI_Allreduce(MPI_IN_PLACE,counts,3,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
  if (global_variable::my_rank==0)std::cout<<std::setprecision(17)
    <<"AMR_CYCLE {\"cycle\":"<<mesh.ncycle<<",\"time\":"<<mesh.time
    <<",\"dt\":"<<mesh.dt<<",\"blocks\":"<<mesh.nmb_total
    <<",\"sequence\":"<<mesh.GetAMRLoadBalanceUpdateSeq()
    <<",\"bulk_points\":"<<counts[0]<<",\"scri_points\":"<<counts[1]
    <<",\"zero_scri_ranks\":"<<counts[2]<<"}\n";
}
struct Observe {
  Mesh &mesh;h::Z4cHyperboloidal &p;
  int sequence,before=0,after=0,accepted=0;
  std::array<Real,8> bulk_history{},surface_history{};
  explicit Observe(Mesh &m):mesh(m),p(*m.pmb_pack->pz4c_hyperboloidal),
      sequence(m.GetAMRLoadBalanceUpdateSeq()) {}
  void BeginCycle() {
    const int cycle=mesh.ncycle;
    Require(cycle>=0&&cycle<=4,"unexpected AMR cycle");
    Require(mesh.nmb_total==(cycle==1 || cycle==2 ? 127:106),"AMR topology schedule changed");
    Require(mesh.GetAMRLoadBalanceUpdateSeq()==sequence+(cycle>=1)+(cycle>=3),
            "native AMR topology sequence changed");
    for (int gid=0;gid<mesh.nmb_total;++gid)
      Require(mesh.cost_eachmb[gid]==1,"native equal-weight load balancing changed");
    Require(p.IsInitialized()&&p.StagePlansReady(),"AMR state/plans not ready");
    Require(mesh.dt==p.requested_dt && p.dtnew==p.requested_dt,
            "native fixed timestep changed after AMR");
    if (cycle) {
      Require(Totals(p,false)==bulk_history && Totals(p,true)==surface_history,
              "accepted-step remesh changed chi history totals/extrema");
    }
    std::vector<int> covered(p.npoints,0);
    for (int subset=0;subset<4;++subset) {
      const auto points=Kokkos::create_mirror_view_and_copy(HostMemSpace(),
          p.RHSPoints(static_cast<h::RHSSubset>(subset)));
      for (std::size_t q=0;q<points.size();++q) {
        Require(points(q)>=0&&points(q)<p.npoints,"RHS partition retained an old point");
        ++covered[points(q)];
      }
    }
    for (int count:covered)Require(count==1,"RHS partitions skip/duplicate an AMR point");
    for (const auto &criterion:mesh.pmr->pmrc->rcrit)if (criterion.hyper_field>=0)
      Require(criterion.rdata.data()==p.u0.data(),"chi criterion retained old bulk storage");
    Dump(mesh);
  }
  TaskStatus Poison(Driver *driver,int stage) {
    Require(stage==before%4+1 && mesh.ncycle==before/4,"before-stage order changed");++before;
    Require(driver->UsesClassicalRK4()&&mesh.dt==p.requested_dt,"integrator/timestep changed");
    const Real nan=std::numeric_limits<Real>::quiet_NaN();
    Kokkos::deep_copy(p.u_rhs,nan);Kokkos::deep_copy(p.s_rhs,nan);
    Kokkos::deep_copy(p.residual,nan);Kokkos::deep_copy(p.rhs_status,-777);
    Kokkos::deep_copy(p.rhs_gap,nan);Kokkos::deep_copy(p.rhs_cfl,nan);
    PoisonGhosts(p);return TaskStatus::complete;
  }
  TaskStatus CheckStage(Driver *driver,int stage) {
    Require(stage==after%4+1 && mesh.ncycle==after/4 && before==after+1,
            "after-stage order changed");++after;
    Require(driver->stage_time==time_integration::ClassicalRK4::RHSTime(mesh.time,mesh.dt,stage)
        && driver->stage_state_time==time_integration::ClassicalRK4::StateTime(mesh.time,mesh.dt,stage),
        "RK4 stage clocks changed");
    const auto u=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.u0);
    const auto rhs=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.u_rhs);
    const auto status=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.rhs_status);
    const auto gap=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.rhs_gap);
    const auto cfl=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.rhs_cfl);
    const auto geom=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.geometry);
    const auto s=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.s0);
    const auto srhs=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.s_rhs);
    Real maxima[3]={p.StageMaxCompatibility(),p.StageMaxClock(),0};
    std::uint64_t checks[2]={0,0};
    for (int point=0;point<p.npoints;++point) {
      const auto c=p.points.h_view(point);h::CartesianChartJets jets{};
      Require(status(point)==0&&std::isfinite(gap(point))&&std::isfinite(cfl(point)),
              "AMR bulk point lacks current-stage RHS/status");
      maxima[2]=std::max(maxima[2],cfl(point));
      for (int f=0;f<31;++f) {
        jets[f].f=u(c.m,f,c.k,c.j,c.i);
        Require(std::isfinite(jets[f].f)&&std::isfinite(rhs(c.m,f,c.k,c.j,c.i)),
                "nonfinite or unwritten bulk stage state/RHS");++checks[0];
      }
      const Real r=h::SphericalPolicy::Radius(geom(point).x);
      Require(h::TryExpandCartesianChart(jets)&&jets[0].f>0&&jets[18].f+(1+r*r)/2>0,
              "invalid AMR stage bulk geometry");
    }
    for (int point=0;point<p.nsurface_points;++point) {
      const auto c=p.surface_points.h_view(point);h::CartesianChartJets jets{};
      for (int f=0;f<31;++f) {
        jets[f].f=s(c.face,f,c.k,c.j);
        Require(std::isfinite(jets[f].f)&&std::isfinite(srhs(c.face,f,c.k,c.j)),
                "nonfinite or unwritten surface stage state/RHS");++checks[1];
      }
      Require(h::TryExpandCartesianChart(jets)&&jets[0].f>0&&jets[18].f==0,
              "invalid AMR stage scri geometry/clock");
    }
    MPI_Allreduce(MPI_IN_PLACE,maxima,3,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,checks,2,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
    Require(std::isfinite(maxima[0])&&maxima[0]<=1e-7&&std::isfinite(maxima[1]),
            "AMR stage compatibility/clock gate failed");
    if (global_variable::my_rank==0)std::cout<<std::setprecision(17)
      <<"AMR_STAGE {\"cycle\":"<<mesh.ncycle+1<<",\"stage\":"<<stage
      <<",\"rhs_time\":"<<driver->stage_time<<",\"state_time\":"<<driver->stage_state_time
      <<",\"bulk_values\":"<<checks[0]<<",\"scri_values\":"<<checks[1]
      <<",\"compatibility\":"<<maxima[0]<<",\"clock\":"<<maxima[1]
      <<",\"cfl\":"<<maxima[2]<<"}\n";
    return TaskStatus::complete;
  }
  TaskStatus Accepted(Driver *,int stage) {
    Require(stage==1&&mesh.ncycle==accepted&&after==4*(accepted+1),
            "accepted-step observer order changed");++accepted;
    const auto bulk=Totals(p,false),surface=Totals(p,true);
    for (auto records:{bulk,surface}) {
      Require(records[0]>0&&records[1]>0,"chi history contains nonpositive candidate");
      for (int a:{2,3,4,5,6})Require(records[a]==0,"unexpected chi floor/correction event");
      Require(records[7]==1e300,"unexpected first chi-floor time");
    }
    bulk_history=bulk;surface_history=surface;
    PoisonGhosts(p);return TaskStatus::complete;
  }
};
}  // namespace
int main(int argc,char **argv) {
  MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);Kokkos::initialize(argc,argv);
  try {{
    Require(argc==2,"usage: amr_evolution N12-initial-input");
    ParameterInput pin;IOWrapper file;file.Open(argv[1],IOWrapper::FileMode::read);
    pin.LoadFromFile(file);file.Close();
    pin.SetString("mesh_refinement","refinement","adaptive");
    pin.SetInteger("mesh_refinement","num_levels",2);
    pin.SetInteger("mesh_refinement","max_nmb_per_rank",160/global_variable::nranks+8);
    pin.SetInteger("mesh_refinement","refinement_interval",0);
    pin.SetString("amr_criterion0","method","slope");
    pin.SetString("amr_criterion0","variable","hyperboloidal_chi");
    pin.SetReal("amr_criterion0","value_max",1e6);
    pin.SetString("amr_criterion1","method","user");
    Mesh mesh(&pin);mesh.BuildTreeFromScratch(&pin);mesh.AddCoordinatesAndPhysics(&pin);
    mesh.pmr=new MeshRefinement(&mesh,&pin);
    mesh.pgen=std::make_unique<ProblemGenerator>(&pin,&mesh);
    mesh.pgen->user_ref_func=Criterion;
    auto &p=*mesh.pmb_pack->pz4c_hyperboloidal;
    // Test-only admission while the ordinary adaptive constructor gate stays closed.
    // All initial fields, RK weights, clocks and subsequent dt decisions are native.
    pin.SetString("time","evolution","dynamic");
    pin.SetString("time","integrator","rk4_classical");
    pin.SetInteger("time","nlim",4);pin.SetRealExact("time","tlim",10*p.requested_dt);
    Kokkos::Timer timer;Driver driver(&pin,&mesh,3600,&timer);Outputs output(&pin,&mesh);
    Require(output.pout_list.empty(),"test must not write ordinary outputs");
    driver.Initialize(&mesh,&pin,&output,false);
    Observe observer(mesh);
    auto begin=mesh.pmb_pack->tl_map.at("before_timeintegrator");
    auto before=mesh.pmb_pack->tl_map.at("before_stagen");
    auto after=mesh.pmb_pack->tl_map.at("after_stagen");
    auto end=mesh.pmb_pack->tl_map.at("after_timeintegrator");
    auto begin_dep=AllTasks(begin),before_dep=AllTasks(before);
    auto after_dep=AllTasks(after),end_dep=AllTasks(end);
    begin->AddTask([&](Driver *,int) {observer.BeginCycle();return TaskStatus::complete;},begin_dep);
    before->AddTask([&](Driver *d,int stage) {return observer.Poison(d,stage);},before_dep);
    after->AddTask([&](Driver *d,int stage) {return observer.CheckStage(d,stage);},after_dep);
    end->AddTask([&](Driver *d,int stage) {return observer.Accepted(d,stage);},end_dep);
    driver.Execute(&mesh,&pin,&output,false);observer.BeginCycle();
    Real expected=0;for (int cycle=0;cycle<4;++cycle)expected+=p.requested_dt;
    Require(mesh.ncycle==4&&mesh.time==expected&&mesh.dt_last_completed==p.requested_dt,
            "Driver did not complete four exact fixed steps");
    Require(observer.before==16&&observer.after==16&&observer.accepted==4,
            "coupled AMR stage coverage incomplete");
    MPI_Allreduce(MPI_IN_PLACE,&refreshed_neighbors,1,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
    Require(refreshed_neighbors==std::uint64_t(106+127+127+106)*12*12*12*6,
            "accepted halo verification omitted a topology");
    if (!global_variable::my_rank)std::cout<<"PASS four native Driver cycles with two AMR events"
        <<" refreshed_neighbors="<<refreshed_neighbors<<"\n";
  }} catch(const std::exception &error) {
    std::cerr<<"AMR evolution rank "<<global_variable::my_rank<<": "<<error.what()<<'\n';
    MPI_Abort(MPI_COMM_WORLD,1);return 1;
  }
  Kokkos::finalize();MPI_Finalize();return 0;
}
