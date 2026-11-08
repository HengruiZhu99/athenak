// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Native history reduction/dispatch checks; constraint formulas have separate oracles.
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "outputs/outputs.hpp"
#include "pgen/pgen.hpp"
#include "z4c_hyperboloidal/diagnostics.hpp"
#include "z4c_hyperboloidal/remesh.hpp"

namespace h=z4c_hyperboloidal;
namespace {
void Require(bool yes,const char *message) {if (!yes)throw std::runtime_error(message);}
constexpr const char *labels[10]={"Volume","H-rms","M-rms","Theta-rms","Z-rms",
    "eVolume","eH-rms","eM-rms","eTheta-rms","eZ-rms"};
OutputParameters Parameters(const char *name) {
  OutputParameters op{};op.block_name="output1";op.file_type="hst";
  op.file_basename=name;op.data_format=" %.17e";op.last_time=-1;op.dt=.125;
  return op;
}
std::array<Real,10> Synthetic(bool local,int amplitude) {
  std::array<Real,10> sums{};
  // Same 24 weighted samples at every decomposition. Unequal local volumes,
  // unequal local RMS values and ranks with zero exterior volume are deliberate.
  for (int point=0;point<24;++point) {
    if (local && point%global_variable::nranks!=global_variable::my_rank)continue;
    const Real volume=point+1;
    for (int region=0;region<2;++region) {
      if (region && point%2)continue;
      sums[5*region]+=volume;
      for (int f=1;f<5;++f) {
        const Real value=amplitude*(point%5+f);
        sums[5*region+f]+=volume*value*value;
      }
    }
  }
  return sums;
}
void CheckNormalized(const HistoryData &data,std::array<Real,10> expected) {
  if (global_variable::my_rank)return;
  for (int region=0;region<2;++region)for (int f=1;f<5;++f)
    expected[5*region+f]=std::sqrt(expected[5*region+f]/expected[5*region]);
  for (int f=0;f<10;++f)
    Require(std::abs(data.hdata[f]-expected[f])<=
        32*std::numeric_limits<Real>::epsilon()*(1+std::abs(expected[f])),
        "history normalized before global reduction or changed raw volumes");
}
void SyntheticChecks(ParameterInput &pin,Mesh &mesh,HistoryOutput &history) {
  auto &data=history.hist_data.front();data.nhist=10;
  for (int f=0;f<10;++f)data.label[f]=labels[f];
  for (int sample=1;sample<=2;++sample) {
    const auto local=Synthetic(true,sample);
    for (int f=0;f<10;++f)data.hdata[f]=local[f];
    mesh.time=.25*sample;mesh.dt=.03125;
    history.WriteOutputFile(&mesh,&pin);CheckNormalized(data,Synthetic(false,sample));
  }
  for (int fault=0;fault<3;++fault) {
    const auto local=Synthetic(true,1);
    for (int f=0;f<10;++f)data.hdata[f]=local[f];
    if (fault==0)data.hdata[5]=0;
    if (fault==1)data.hdata[2]=-1;
    if (fault==2 && global_variable::my_rank==global_variable::nranks-1)
      data.hdata[4]=std::numeric_limits<Real>::quiet_NaN();
    int caught=0;
    try {history.WriteOutputFile(&mesh,&pin);} catch (const std::runtime_error &) {caught=1;}
    MPI_Allreduce(MPI_IN_PLACE,&caught,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    Require(caught==global_variable::nranks,"invalid history did not fail collectively");
  }
  // Existing non-hyperboloidal scalar sums and filenames retain their behavior.
  HistoryOutput ordinary(&pin,&mesh,Parameters("ordinary"));ordinary.hist_data.clear();
  for (auto physics:{PhysicsModule::HydroDynamics,PhysicsModule::MagnetoHydroDynamics,
                    PhysicsModule::SpaceTimeDynamics,PhysicsModule::UserDefined}) {
    ordinary.hist_data.emplace_back(physics);auto &out=ordinary.hist_data.back();
    out.nhist=2;out.label[0]="sum-a";out.label[1]="sum-b";
    out.hdata[0]=Synthetic(true,1)[0];out.hdata[1]=Synthetic(true,1)[1];
  }
  ordinary.WriteOutputFile(&mesh,&pin);
  if (!global_variable::my_rank)for (const auto &out:ordinary.hist_data)
    Require(out.hdata[0]==Synthetic(false,1)[0]&&out.hdata[1]==Synthetic(false,1)[1],
            "ordinary history sum changed");
}
std::vector<Real> Accepted(const h::Z4cHyperboloidal &p) {
  const auto u=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.u0);
  const auto s=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.s0);
  const auto chi=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.chi_records);
  const auto schi=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.surface_chi_records);
  std::vector<Real> values(39*(p.npoints+p.nsurface_points));
  for (int q=0;q<p.npoints;++q) {
    const auto c=p.points.h_view(q);auto *v=values.data()+39*q;
    for (int f=0;f<31;++f)v[f]=u(c.m,f,c.k,c.j,c.i);h::PackChiRecord(chi(q),v+31);
  }
  for (int q=0;q<p.nsurface_points;++q) {
    const auto c=p.surface_points.h_view(q);auto *v=values.data()+39*(p.npoints+q);
    for (int f=0;f<31;++f)v[f]=s(c.face,f,c.k,c.j);h::PackChiRecord(schi(q),v+31);
  }
  return values;
}
template<class View>void CheckSentinel(const View &view,Real expected) {
  const auto host=Kokkos::create_mirror_view_and_copy(HostMemSpace(),view);
  for (std::size_t i=0;i<host.size();++i)
    Require(host.data()[i]==expected,"history diagnostic changed RK/RHS workspace");
}
void RealChecks(ParameterInput &pin,Mesh &mesh,HistoryOutput &history) {
  auto &p=*mesh.pmb_pack->pz4c_hyperboloidal;
  Kokkos::Timer timer;Driver driver(&pin,&mesh,3600,&timer);Outputs output(&pin,&mesh);
  Require(output.pout_list.empty(),"history fixture unexpectedly enrolled ordinary outputs");
  driver.Initialize(&mesh,&pin,&output,false);
  Kokkos::deep_copy(p.u_rhs,13);Kokkos::deep_copy(p.s_rhs,14);
  Kokkos::deep_copy(p.u1,15);Kokkos::deep_copy(p.u_accum,16);
  Kokkos::deep_copy(p.s1,17);Kokkos::deep_copy(p.s_accum,18);
  const auto before=Accepted(p);
  const auto version=p.StateVersion();
  const auto statistics_before=p.GetDiagnostics().GetStatistics();
  const Real stage_compatibility=p.StageMaxCompatibility(),stage_clock=p.StageMaxClock();
  std::array<Real,10> previous{};
  for (int call=0;call<2;++call) {
    history.LoadOutputData(&mesh);
    const auto local=p.DiagnosticHistorySums();
    const auto statistics_after=p.GetDiagnostics().GetStatistics();
    Require(statistics_after.refreshes==statistics_before.refreshes+1 &&
            statistics_after.searches==statistics_before.searches+1,
            "unchanged accepted-state history repeated diagnostic work");
    auto &data=history.hist_data.front();
    Require(data.nhist==10,"history loader omitted a region/component");
    for (int f=0;f<10;++f) {
      Require(data.hdata[f]==local[f],"history loader changed rank-local raw sums");
      Require(data.label[f]==labels[f],"history column labels changed");
      if (call)Require(local[f]==previous[f],"repeated accepted-state diagnostic differs");
    }
    previous=local;
    auto global=local;
    MPI_Allreduce(MPI_IN_PLACE,global.data(),10,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    history.WriteOutputFile(&mesh,&pin);CheckNormalized(data,global);
    const auto after=Accepted(p);
    Require(before.size()==after.size() &&
            std::memcmp(before.data(),after.data(),before.size()*sizeof(Real))==0,
            "history diagnostic changed accepted fields/history");
    Require(p.StateVersion()==version && p.StageMaxCompatibility()==stage_compatibility &&
            p.StageMaxClock()==stage_clock,"history diagnostic altered stage/state bookkeeping");
    CheckSentinel(p.u_rhs,13);CheckSentinel(p.s_rhs,14);CheckSentinel(p.u1,15);
    CheckSentinel(p.u_accum,16);CheckSentinel(p.s1,17);CheckSentinel(p.s_accum,18);
  }
  const auto check_refresh=[&]() {
    const auto state_before=Accepted(p);
    const auto version_before=p.StateVersion();
    const auto count_before=p.GetDiagnostics().GetStatistics();
    (void)p.DiagnosticHistorySums();
    (void)p.DiagnosticHistorySums();
    const auto count_after=p.GetDiagnostics().GetStatistics();
    Require(count_after.refreshes==count_before.refreshes+1 &&
            count_after.searches==count_before.searches+1,
            "changed accepted-state history did not refresh every rank exactly once");
    const auto state_after=Accepted(p);
    Require(state_before.size()==state_after.size() &&
            std::memcmp(state_before.data(),state_after.data(),
                        state_before.size()*sizeof(Real))==0,
            "refreshed history changed accepted fields/history");
    Require(p.StateVersion()==version_before &&
            p.StageMaxCompatibility()==stage_compatibility && p.StageMaxClock()==stage_clock,
            "refreshed history altered stage/state bookkeeping");
    CheckSentinel(p.u_rhs,13);CheckSentinel(p.s_rhs,14);CheckSentinel(p.u1,15);
    CheckSentinel(p.u_accum,16);CheckSentinel(p.s1,17);CheckSentinel(p.s_accum,18);
  };
  // A single rank changes a real independent field. Other ranks must join the
  // refresh despite their unchanged local state-version token.
  const auto first=p.points.h_view(0);const auto state=p.u0;
  const Real initial_theta=before[h::Fields::theta];
  for (int restore=0;restore<2;++restore) {
    if (global_variable::my_rank==0) {
      const Real value=initial_theta+(restore?0:1e-8);
      Kokkos::parallel_for("history test one-rank theta mutation",
        Kokkos::RangePolicy<>(0,1),KOKKOS_LAMBDA(int) {
          state(first.m,h::Fields::theta,first.k,first.j,first.i)=value;
        });
      Kokkos::fence();p.InvalidateDiagnostics();
    }
    check_refresh();
  }
  // Pure clock/cycle changes are also cache keys, even with identical fields.
  mesh.time+=mesh.dt;++mesh.ncycle;check_refresh();
}
}  // namespace
int main(int argc,char **argv) {
  MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);Kokkos::initialize(argc,argv);
  try {{
    Require(argc==3,"usage: history_output fixture synthetic|real");
    const std::string mode=argv[2];Require(mode=="synthetic"||mode=="real","unknown test mode");
    static_assert(PhysicsModule::HydroDynamics==0&&PhysicsModule::MagnetoHydroDynamics==1&&
        PhysicsModule::SpaceTimeDynamics==2&&PhysicsModule::UserDefined==3,
        "existing history physics identifiers changed");
    ParameterInput pin;IOWrapper file;file.Open(argv[1],IOWrapper::FileMode::read);
    pin.LoadFromFile(file);file.Close();
    if (mode=="real") {
      pin.SetString("time","evolution","dynamic");
      pin.SetString("time","integrator","rk4_classical");
      pin.SetInteger("time","nlim",0);pin.SetReal("time","tlim",1);
    }
    Mesh mesh(&pin);mesh.BuildTreeFromScratch(&pin);mesh.AddCoordinatesAndPhysics(&pin);
    mesh.pgen=std::make_unique<ProblemGenerator>(&pin,&mesh);
    HistoryOutput history(&pin,&mesh,Parameters(mode.c_str()));
    Require(history.hist_data.size()==1 && history.hist_data[0].physics==
        PhysicsModule::HyperboloidalSpaceTimeDynamics,"missing native hyperboloidal dispatch");
    mesh.pgen->user_hist=true;auto only=Parameters("user-only");only.user_hist_only=true;
    HistoryOutput user_only(&pin,&mesh,only);
    Require(user_only.hist_data.size()==1 && user_only.hist_data[0].physics==
        PhysicsModule::UserDefined,"user_hist_only unexpectedly enrolls horizon diagnostics");
    mesh.pgen->user_hist=false;
    if (mode=="synthetic")SyntheticChecks(pin,mesh,history);
    else RealChecks(pin,mesh,history);
    if (!global_variable::my_rank)std::cout<<"PASS native history "<<mode
        <<" ranks="<<global_variable::nranks<<'\n';
  }} catch(const std::exception &error) {
    std::cerr<<"history output rank "<<global_variable::my_rank<<": "<<error.what()<<'\n';
    MPI_Abort(MPI_COMM_WORLD,1);return 1;
  }
  Kokkos::finalize();MPI_Finalize();return 0;
}
