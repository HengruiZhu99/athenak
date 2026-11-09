// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// One full native Driver cycle with poisoned workspaces and stage coverage checks.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"
#include "pgen/pgen.hpp"
#include "driver/driver.hpp"
#include "z4c_hyperboloidal/z4c_hyperboloidal.hpp"
#include "z4c_hyperboloidal/point_rhs.hpp"

namespace hyp = z4c_hyperboloidal;

namespace {
void Require(bool condition,const char *message) {
  if (!condition) throw std::runtime_error(message);
}

TaskID AllTasks(const std::shared_ptr<TaskList> &list) {
  TaskID dependency(0);
  for (int i=1;i<=list->Size();++i) dependency=dependency | TaskID(i);
  return dependency;
}

struct Checks {
  int before=0,after=0;
  std::uint64_t bulk_values=0,surface_values=0,ghost_values=0;
  Real max_cfl=0,max_rhs=0,max_update_error=0;

  TaskStatus Poison(hyp::Z4cHyperboloidal &p,Driver *driver,int stage) {
    Require(stage==++before,"native Driver repeated or skipped a before-stage hook");
    Require(driver->UsesClassicalRK4(),"native Driver selected wrong integrator");
    const Real poison=std::numeric_limits<Real>::quiet_NaN();
    Kokkos::deep_copy(p.u_rhs,poison);Kokkos::deep_copy(p.s_rhs,poison);
    Kokkos::deep_copy(p.residual,poison);
    Kokkos::deep_copy(p.rhs_status,-777);
    Kokkos::deep_copy(p.rhs_gap,poison);Kokkos::deep_copy(p.rhs_cfl,poison);
    const auto state=p.u0;const auto bounds=p.evolved_bounds.d_view;
    const int nk=p.storage[2],nj=p.storage[1],ni=p.storage[0];
    Kokkos::parallel_for("stage smoke poison physical ghosts",
      Kokkos::RangePolicy<>(0,p.nblocks*nk*nj*ni),KOKKOS_LAMBDA(int q) {
        const int i=q%ni,j=(q/ni)%nj,k=(q/(ni*nj))%nk,m=q/(ni*nj*nk);
        const auto b=bounds(m);
        if (i<b.lo[0] || i>b.hi[0] || j<b.lo[1] || j>b.hi[1] ||
            k<b.lo[2] || k>b.hi[2])
          for (int f=0;f<hyp::Fields::count;++f)state(m,f,k,j,i)=poison;
      });
    return TaskStatus::complete;
  }

  TaskStatus Verify(hyp::Z4cHyperboloidal &p,Driver *driver,int stage,Real dt) {
    Require(stage==++after && before==after,
            "native Driver repeated or skipped an after-stage hook");
    const Real rhs_time=time_integration::ClassicalRK4::RHSTime(0,dt,stage);
    const Real state_time=time_integration::ClassicalRK4::StateTime(0,dt,stage);
    Require(driver->stage_time==rhs_time && driver->stage_state_time==state_time,
            "native Driver stage clocks disagree with classical RK4");
    const auto state=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.u0);
    const auto base=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.u1);
    const auto rhs=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.u_rhs);
    const auto sum=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.u_accum);
    const auto status=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.rhs_status);
    const auto cfl=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.rhs_cfl);
    const auto gap=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.rhs_gap);
    const auto bulk_geometry=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.geometry);
    const auto edge=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.s0);
    const auto edge_base=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.s1);
    const auto edge_rhs=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.s_rhs);
    const auto edge_sum=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.s_accum);
    const auto geometry=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.surface_geometry);
    const auto updated=[&](Real initial,Real rate,Real accumulated,int field) {
      Real value=stage==4 ? initial+dt*accumulated/6
                          : initial+dt*(stage<3 ? Real(.5) : Real(1))*rate;
      if (field==hyp::Fields::chi && p.chi_floor>0 && value<p.chi_floor)
        value=p.chi_floor;
      return value;
    };
    const auto compare=[&](Real actual,Real expected) {
      const Real error=std::abs(actual-expected)/std::max({Real(1),std::abs(actual),
                                                        std::abs(expected)});
      max_update_error=std::max(max_update_error,error);
      Require(error<3e-15,"native RK update differs from accumulated current-stage rates");
    };
    for (int point=0;point<p.npoints;++point) {
      const auto c=p.points.h_view(point);
      Require(status(point)==static_cast<int>(hyp::PointStatus::success),
              "owned bulk point was skipped or reported an invalid RHS");
      Require(std::isfinite(cfl(point)) && std::isfinite(gap(point)),
              "owned bulk point lacks current-stage characteristic data");
      max_cfl=std::max(max_cfl,cfl(point));
      hyp::CartesianChartJets jets{};
      for (int f=0;f<hyp::Fields::count;++f) {
        const Real value=state(c.m,f,c.k,c.j,c.i),rate=rhs(c.m,f,c.k,c.j,c.i);
        Require(std::isfinite(value) && std::isfinite(rate),
                "nonfinite or unwritten owned bulk state/RHS");
        max_rhs=std::max(max_rhs,std::abs(rate));++bulk_values;
        jets[f].f=value;
        if (hyp::Fields::Independent(f))
          compare(value,updated(base(c.m,f,c.k,c.j,c.i),rate,
                                sum(c.m,f,c.k,c.j,c.i),f));
      }
      Require(hyp::TryExpandCartesianChart(jets),"invalid accepted bulk metric chart");
      const Real r=hyp::SphericalPolicy::Radius(bulk_geometry(point).x);
      Require(jets[0].f>0 && jets[18].f+(1+r*r)/2>0,
              "nonpositive updated bulk chi or physical lapse");
      compare(state(c.m,6,c.k,c.j,c.i),jets[6].f);
      compare(state(c.m,12,c.k,c.j,c.i),jets[12].f);
    }
    for (int face=0;face<p.nfaces;++face)for (int k=0;k<p.n[2];++k)
      for (int j=0;j<p.n[1];++j) {
        hyp::CartesianChartJets jets{};
        for (int f=0;f<hyp::Fields::count;++f) {
          const Real value=edge(face,f,k,j),rate=edge_rhs(face,f,k,j);
          Require(std::isfinite(value) && std::isfinite(rate),
                  "nonfinite or unwritten owned scri state/RHS");
          ++surface_values;jets[f].f=value;
          if (hyp::Fields::Independent(f) && f!=18 && !(f>=22 && f<=27))
            compare(value,updated(edge_base(face,f,k,j),rate,edge_sum(face,f,k,j),f));
        }
        Require(hyp::TryExpandCartesianChart(jets),"invalid accepted scri metric chart");
        compare(edge(face,6,k,j),jets[6].f);compare(edge(face,12,k,j),jets[12].f);
        Require(edge(face,18,k,j)==0,"updated scri clock was not projected");
        const auto map=geometry((face*p.n[2]+k)*p.n[1]+j);
        const auto physical=hyp::SphericalPolicy::Physical(map.x,jets);
        const auto metric=hyp::Geometry(physical.metric);
        Real q=0;
        for (int a=0;a<3;++a)for (int b=0;b<3;++b)
          q+=metric.inverse[a][b]*map.x[a]*map.x[b];
        for (int a=0;a<3;++a) {
          compare(edge(face,22+a,k,j),(1-std::sqrt(jets[0].f*q))*map.x[a]);
          Require(edge(face,28+a,k,j)==0,"scri dormant value changed");
        }
      }
    for (int m=0;m<p.nblocks;++m) {
      const auto b=p.evolved_bounds.h_view(m);
      for (int k=0;k<p.storage[2];++k)for (int j=0;j<p.storage[1];++j)
        for (int i=0;i<p.storage[0];++i) {
          if (i>=b.lo[0] && i<=b.hi[0] && j>=b.lo[1] && j<=b.hi[1] &&
              k>=b.lo[2] && k<=b.hi[2]) continue;
          for (int f=0;f<hyp::Fields::count;++f) {
            Require(std::isnan(rhs(m,f,k,j,i)),"RHS kernel wrote an unevolved ghost");
            ++ghost_values;
          }
        }
    }
    return TaskStatus::complete;
  }
};

void DumpAcceptedState(const hyp::Z4cHyperboloidal &p,const MeshBlockPack &pack) {
  static_assert(sizeof(Real)==8,"smoke dump requires float64");
  const auto bulk=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.u0);
  const auto edge=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.s0);
  const auto chi=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.chi_records);
  const auto edge_chi=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.surface_chi_records);
  const std::string rank=std::to_string(global_variable::my_rank);
  std::ofstream bout("accepted-bulk-rank"+rank+".bin",std::ios::binary);
  std::ofstream sout("accepted-scri-rank"+rank+".bin",std::ios::binary);
  Require(bool(bout) && bool(sout),"could not create smoke accepted-state dump");
  const std::uint64_t bulk_count=p.npoints,surface_count=p.nsurface_points;
  bout.write(reinterpret_cast<const char *>(&bulk_count),sizeof(bulk_count));
  sout.write(reinterpret_cast<const char *>(&surface_count),sizeof(surface_count));
  const auto records=[](const hyp::ChiRecord &c,Real *values) {
    values[31]=c.minimum;values[32]=c.accepted_minimum;
    values[33]=c.floor_count;values[34]=c.nonpositive_count;values[35]=c.nonfinite_count;
    values[36]=c.max_correction;values[37]=c.sum_correction;values[38]=c.first_floor;
  };
  for (int point=0;point<p.npoints;++point) {
    const auto c=p.points.h_view(point);
    const std::int32_t key[4]={pack.gids+c.m,c.k,c.j,c.i};Real values[39];
    for (int f=0;f<31;++f)values[f]=bulk(c.m,f,c.k,c.j,c.i);
    records(chi(point),values);
    bout.write(reinterpret_cast<const char *>(key),sizeof(key));
    bout.write(reinterpret_cast<const char *>(values),sizeof(values));
  }
  for (int point=0;point<p.nsurface_points;++point) {
    const auto c=p.surface_points.h_view(point);
    const std::int32_t key[3]={pack.gids+p.face_blocks.h_view(c.face),c.k,c.j};
    Real values[39];
    for (int f=0;f<31;++f)values[f]=edge(c.face,f,c.k,c.j);
    records(edge_chi(point),values);
    sout.write(reinterpret_cast<const char *>(key),sizeof(key));
    sout.write(reinterpret_cast<const char *>(values),sizeof(values));
  }
  Require(bool(bout) && bool(sout),"accepted-state dump write failed");
}
}  // namespace

int main(int argc,char **argv) {
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);
  Kokkos::initialize(argc,argv);
  try {
    {
      Require(argc==2,"usage: stage_smoke N12-static-input");
      ParameterInput pin;IOWrapper input;
      input.Open(argv[1],IOWrapper::FileMode::read);pin.LoadFromFile(input);input.Close();
      Mesh mesh(&pin);mesh.BuildTreeFromScratch(&pin);mesh.AddCoordinatesAndPhysics(&pin);
      mesh.pgen=std::make_unique<ProblemGenerator>(&pin,&mesh);
      auto &p=*mesh.pmb_pack->pz4c_hyperboloidal;
      Require(p.IsInitialized(),"native problem generator did not initialize module");
      p.BuildStagePlans();
      Require(p.StagePlansReady(),"native stage plan construction failed");
      Require(mesh.pmb_pack->pnr!=nullptr,"module did not register native NR tasks");
      Require(mesh.pmb_pack->tl_map.at("before_stagen")->Size()==1 &&
            mesh.pmb_pack->tl_map.at("stagen")->Size()==14 &&
              mesh.pmb_pack->tl_map.at("after_stagen")->Size()==2,
              "native NR did not assemble the complete hyperboloidal stage graph");

      // Test-only admission past the module's deliberate dynamic constructor
      // gate. Production input still rejects dynamic evolution. This does not
      // validate restart, AMR, diagnostics, or the not-yet-wired timestep bound.
      const Real dt=p.requested_dt;
      pin.SetString("time","evolution","dynamic");
      pin.SetString("time","integrator","rk4_classical");
      pin.SetInteger("time","nlim",1);pin.SetRealExact("time","tlim",2*dt);
      Kokkos::Timer timer;Driver driver(&pin,&mesh,3600,&timer);
      Outputs output(&pin,&mesh);
      Require(output.pout_list.empty(),"smoke fixture must not write production outputs");
      driver.Initialize(&mesh,&pin,&output,false);
      mesh.dt=dt;  // Exact N12 policy while mapped NewTimeStep integration is pending.
      Checks checks;
      auto before=mesh.pmb_pack->tl_map.at("before_stagen");
      auto after=mesh.pmb_pack->tl_map.at("after_stagen");
      auto begin_dep=AllTasks(before),end_dep=AllTasks(after);
      before->AddTask([&](Driver *d,int stage){return checks.Poison(p,d,stage);},begin_dep);
      after->AddTask([&](Driver *d,int stage){return checks.Verify(p,d,stage,dt);},end_dep);
      driver.Execute(&mesh,&pin,&output,false);
      Require(mesh.ncycle==1 && mesh.time==dt && mesh.dt_last_completed==dt,
              "native Driver did not complete exactly one pinned N12 step");
      Require(checks.before==4 && checks.after==4,"native Driver stage coverage incomplete");
      Require(checks.max_rhs>0,"smoke run executed an empty bulk RHS");
      DumpAcceptedState(p,*mesh.pmb_pack);
      std::uint64_t totals[7]={std::uint64_t(p.npoints),std::uint64_t(p.nsurface_points),
          checks.bulk_values,checks.surface_values,checks.ghost_values,
          std::uint64_t(p.nfaces==0),std::uint64_t(p.nblocks)};
      Real maxima[3]={checks.max_cfl,checks.max_rhs,checks.max_update_error};
      MPI_Allreduce(MPI_IN_PLACE,totals,7,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE,maxima,3,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
      Require(totals[0]==229392 && totals[1]==864 && totals[6]==106,
              "full N12 owned workload changed");
      if (global_variable::my_rank==0)std::cout<<std::setprecision(17)
          <<"PASS native Driver stage smoke: ranks="<<global_variable::nranks
          <<" cycles="<<mesh.ncycle<<" stages="<<checks.after<<" bulk="<<totals[0]
          <<" scri="<<totals[1]<<" bulk_value_checks="<<totals[2]
          <<" surface_value_checks="<<totals[3]<<" poisoned_ghost_checks="<<totals[4]
          <<" no_surface_ranks="<<totals[5]<<" max_cfl="<<maxima[0]
          <<" max_abs_rhs="<<maxima[1]<<" max_update_error="<<maxima[2]
          <<" time="<<mesh.time<<" legacy_comparison=not_performed\n";
    }
    Kokkos::finalize();MPI_Finalize();return 0;
  } catch (const std::exception &error) {
    std::cerr<<"stage smoke rank "<<global_variable::my_rank<<": "<<error.what()<<"\n";
    MPI_Abort(MPI_COMM_WORLD,1);return 1;
  }
}
