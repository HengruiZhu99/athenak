// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Benchmark entry point only. Include an archived, unchanged predecessor tree.
#define HYPERBOLIC_NO_DRIVER_MAIN 1
#include "tst/hyperbolic_shift_3d/coupled_mpi.cpp"
#include "tst/hyperbolic_shift_3d/endpoint_policy/static_mesh.hpp"
#include "tst/hyperbolic_shift_3d/endpoint_policy/run_controls.hpp"
#include "tst/hyperbolic_shift_3d/endpoint_policy/fastflow_history.hpp"

namespace legacy_n12_benchmark {
constexpr int start_cycle=2486, warmup=10, window_steps=20, windows=3;
constexpr int end_cycle=start_cycle+warmup+window_steps*windows;
constexpr Real dt=0.00046944203735617445;

void Require(bool yes,const char *message) {
  if (!yes) throw std::runtime_error(message);
}

void All(bool yes,const char *message) {
  int bad=!yes;MPI_Allreduce(MPI_IN_PLACE,&bad,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
  Require(!bad,message);
}

std::vector<Real> Read(const std::filesystem::path &path,std::uint64_t first,
                       std::uint64_t count,std::uint64_t total,std::uint64_t prefix=0) {
  Require(std::filesystem::file_size(path)==prefix+total*sizeof(Real),"seed file extent changed");
  std::ifstream file(path,std::ios::binary);file.seekg(prefix+first*sizeof(Real));
  std::vector<Real> values(count);file.read(reinterpret_cast<char*>(values.data()),count*sizeof(Real));
  Require(bool(file),"seed read failed");
  for (Real value:values)Require(std::isfinite(value),"nonfinite seed value");
  return values;
}

void SeedRecords(Evolution &e,const std::filesystem::path &directory) {
  CollectiveLocal([&] {
    for(int region=0;region<2;++region) {
      const int first=region?e.surface_first:e.first,count=region?e.surface_local:e.local;
      const int total=region?e.surface_total:e.total;
      const auto data=Read(directory/(region?"surface-chi-records.bin":"chi-records.bin"),
                           8*first,8*count,8*total);
      auto target=region?e.spherical_surface_chi_records:e.spherical_chi_records;
      auto host=Kokkos::create_mirror_view(target);
      for(int q=0;q<count;++q) {
        auto &r=host(q);const Real *v=data.data()+8*q;
        r.minimum=v[0];r.accepted_minimum=v[1];r.floor_count=v[2];
        r.nonpositive_count=v[3];r.nonfinite_count=v[4];r.max_correction=v[5];
        r.sum_correction=v[6];r.first_floor=v[7];
      }
      Kokkos::deep_copy(target,host);
    }
  });
}

std::vector<Real> Records(Evolution &e,int region) {
  const auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
      region?e.spherical_surface_chi_records:e.spherical_chi_records);
  std::vector<Real> values(8*host.extent(0));
  for(std::size_t q=0;q<host.extent(0);++q) {
    const auto &r=host(q);const Real row[8]={r.minimum,r.accepted_minimum,r.floor_count,
      r.nonpositive_count,r.nonfinite_count,r.max_correction,r.sum_correction,r.first_floor};
    std::copy_n(row,8,values.data()+8*q);
  }
  return values;
}

std::vector<Real> Owned(Evolution &e,int region) {
  const int first=region?e.surface_first:e.first,count=region?e.surface_local:e.local;
  return e.Download(Kokkos::subview(region?e.surface:e.state,
      std::make_pair(std::size_t(31*first),std::size_t(31*(first+count)))));
}

void VerifyInitial(Evolution &e,const std::filesystem::path &directory) {
  CollectiveLocal([&] {
    const auto checkpoint=directory/"restart.bin";
    const std::uint64_t total=2*31*(e.total+e.surface_total);
    const auto prefix=std::filesystem::file_size(checkpoint)-total*sizeof(Real);
    Require(prefix>72&&prefix<65536,"bad SP1 prefix");
    for(int region=0;region<2;++region) {
      const int first=region?e.surface_first:e.first,count=region?e.surface_local:e.local;
      const auto fields=Owned(e,region), records=Records(e,region);
      const auto expected=Read(checkpoint,(region?2*31*e.total:0)+31*first,31*count,total,prefix);
      const auto history=Read(directory/(region?"surface-chi-records.bin":"chi-records.bin"),
                              8*first,8*count,8*(region?e.surface_total:e.total));
      Require(fields.size()==expected.size()&&records.size()==history.size(),"seed coverage differs");
      Require(std::memcmp(fields.data(),expected.data(),fields.size()*sizeof(Real))==0,
              "initial fields differ from converted native checkpoint");
      Require(std::memcmp(records.data(),history.data(),records.size()*sizeof(Real))==0,
              "initial records differ from native checkpoint");
    }
  });
}

void Snapshot(Evolution &e,const std::filesystem::path &out) {
  CollectiveLocal([&] {
    for(int region=0;region<2;++region) {
      const int first=region?e.surface_first:e.first,count=region?e.surface_local:e.local;
      const auto values=Owned(e,region),records=Records(e,region);
      std::ofstream file(out/(std::string("accepted-final")+(region?"-scri-rank":"-bulk-rank")+
                             std::to_string(e.rank)+".bin"),std::ios::binary);
      const char magic[16]="LEGACY_TRACE_V1";const std::uint64_t size=count;
      const std::uint32_t shape[4]={6,39,std::uint32_t(region),1};
      file.write(magic,16);file.write(reinterpret_cast<const char*>(&size),8);
      file.write(reinterpret_cast<const char*>(shape),16);
      for(int q=0;q<count;++q) {
        const int ordinal=first+q;std::int32_t key[6];
        if(region) {const auto &r=e.Model().scri.rows[ordinal];
          const std::int32_t k[6]={ordinal,r.block,r.k,r.j,12,e.Model().grid.mesh.blocks[r.block].face};
          std::copy_n(k,6,key);
        } else {const auto &c=e.Model().grid.cells[ordinal];
          const std::int32_t k[6]={ordinal,c.block,c.k,c.j,c.i,e.Model().grid.mesh.blocks[c.block].face};
          std::copy_n(k,6,key);
        }
        Real row[39];std::copy_n(values.data()+31*q,31,row);std::copy_n(records.data()+8*q,8,row+31);
        for(Real value:row)Require(std::isfinite(value),"nonfinite accepted field/history");
        file.write(reinterpret_cast<const char*>(key),sizeof(key));
        file.write(reinterpret_cast<const char*>(row),sizeof(row));
      }
      file.close();Require(bool(file),"accepted snapshot write failed");
    }
  });
}

struct Diagnostics {Real sums[10]{},fixed[5]{};Real compatibility=0,clock=0,cfl=0;};

Diagnostics Diagnose(Evolution &e,const std::filesystem::path &input,
                     const std::filesystem::path &out) {
  // Accepted state was saved before this original diagnostic RHS phase.
  const auto before_bulk=Records(e,0),before_surface=Records(e,1);
  Buffer edge;e.RHS(dt,edge);const auto complete=e.CompleteState(e.state);
  FastFlow::Result horizon{};FastFlowSurface shape;
  CollectiveLocal([&] {
    if(e.rank==0) {
      EndpointFastFlow finder(*e.spherical_model,input,out);
      horizon=finder.Find(e.Download(complete),e.steps,e.time,e.puncture);shape=finder.Surface();
      Require(horizon.found&&horizon.complete_surface,"incomplete benchmark horizon");
      WriteHorizonSurface(out/"horizon.json",horizon,shape);
    }
  });
  BroadcastHorizonSurface(shape,e.rank);
  Buffer mask("benchmark horizon exterior",e.local);auto host=Kokkos::create_mirror_view(mask);
  for(int p=0;p<e.local;++p)host(p)=shape.Outside(e.Model().grid.geometry[e.first+p].x)?1:0;
  Kokkos::deep_copy(mask,host);
  const auto diag=e.kernel()->Diagnostics(e.exchange()->padded,0,e.spherical_surface_residual,
                                        0,false,{.94,.16,.25},mask);
  e.SynchronizeError(diag.invalid_cells,"invalid benchmark diagnostic constraints");
  Diagnostics result;
  const Real values[10]={diag.total.volume,diag.total.h2,diag.total.m2_physical,
    diag.total.theta2,diag.total.z2_physical,diag.exterior.volume,diag.exterior.h2,
    diag.exterior.m2_physical,diag.exterior.theta2,diag.exterior.z2_physical};
  const Real fixed[5]={diag.fixed[0].volume,diag.fixed[0].h2,diag.fixed[0].m2_physical,
                       diag.fixed[0].theta2,diag.fixed[0].z2_physical};
  std::copy_n(values,10,result.sums);std::copy_n(fixed,5,result.fixed);
  MPI_Allreduce(MPI_IN_PLACE,result.sums,10,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,result.fixed,5,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
  Real maxima[3]={e.compatibility_max,e.clock_max,e.cfl};
  MPI_Allreduce(MPI_IN_PLACE,maxima,3,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
  result.compatibility=maxima[0];result.clock=maxima[1];result.cfl=maxima[2];
  for(Real value:result.sums)Require(std::isfinite(value)&&value>=0,"invalid full/exterior norm sum");
  for(Real value:result.fixed)Require(std::isfinite(value)&&value>=0,"invalid fixed-cut norm sum");
  Require(std::isfinite(maxima[0])&&maxima[0]<=1e-7&&std::isfinite(maxima[1])&&
          std::isfinite(maxima[2]),"invalid compatibility/clock/CFL");
  const auto after_bulk=Records(e,0),after_surface=Records(e,1);
  All(before_bulk==after_bulk&&before_surface==after_surface,"diagnostics changed chi history");
  return result;
}
}  // namespace legacy_n12_benchmark

int main(int argc,char **argv) {
  MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);Kokkos::initialize(argc,argv);
  try {
    using namespace legacy_n12_benchmark;
    static_assert(sizeof(Real)==8&&h::spherical_policy,"float64 SP1 policy required");
    All(argc==3&&global_variable::nranks==24,"legacy_n12_benchmark SEED_DIRECTORY NEW_OUTPUT; requires24ranks");
    const std::filesystem::path seed=argv[1],out=argv[2],input=seed/"run.in";
    All(!std::filesystem::exists(out),"refuse existing output");
    CollectiveLocal([&] {if(global_variable::my_rank==0)std::filesystem::create_directories(out);});
    MPI_Barrier(MPI_COMM_WORLD);
    bool clean=!Kokkos::Tools::profileLibraryLoaded();
    for(const char *name:{"KOKKOS_TOOLS_LIBS","KOKKOS_PROFILE_LIBRARY","KOKKOS_TOOLS_ARGS",
        "SYCL_PI_TRACE","SYCL_UR_TRACE","ZE_ENABLE_TRACING_LAYER","UNITRACE_ENABLE",
        "PTI_ENABLE","VTUNE_PROFILER_ENABLE"})clean&=std::getenv(name)==nullptr;
    All(clean,"unprofiled benchmark rejects detailed profiling");
    const auto controls=EndpointRunControls::Load(input);
    All(controls.cells_per_axis==12&&controls.nlim==end_cycle&&controls.dt==dt&&
        controls.tlim==2&&controls.ko==.1&&controls.chi_floor==1e-10&&!controls.track_puncture,
        "changed benchmark physics/workload");
#ifdef LEGACY_BENCH_HAS_PERF
    perf::enabled=false;perf::small_groups=std::getenv("HYPERBOLIC_PROFILE_SMALL_GROUPS")!=nullptr;
#endif
    const double setup_start=MPI_Wtime();
    hyperbolic_reproduction::refined_configuration.Load(controls.configuration.string(),
        hyperbolic_reproduction::ContinuousCoreMesh(12));
    auto mesh=EndpointStaticMesh(hyperbolic_reproduction::refined_configuration,12);
    Evolution e(12,0,2,"",&mesh);e.ConfigureSphericalNumerics(.1,1e-10);
    e.LoadCheckpoint((seed/"restart.bin").string(),2,dt,mesh.amr_cartesian_core);
    e.track_puncture=false;e.puncture={0,0,0};SeedRecords(e,seed);VerifyInitial(e,seed);
    Real expected_time=0;for(int step=0;step<start_cycle;++step)expected_time+=dt;
    All(e.total==229392&&e.surface_total==864&&e.steps==start_cycle&&e.time==expected_time,
        "changed initial N12 coverage/time");
    Kokkos::fence();double setup=MPI_Wtime()-setup_start;
    MPI_Allreduce(MPI_IN_PLACE,&setup,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    const double warm_start=MPI_Wtime();
    for(int i=0;i<warmup;++i)e.Step(dt);
    Kokkos::fence();double warm_seconds=MPI_Wtime()-warm_start;
    MPI_Allreduce(MPI_IN_PLACE,&warm_seconds,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    double elapsed[windows]{};std::vector<double> rank_seconds(windows*e.size);
    for(int window=0;window<windows;++window) {
      Kokkos::fence();MPI_Barrier(MPI_COMM_WORLD);const double tick=MPI_Wtime();
      for(int step=0;step<window_steps;++step)e.Step(dt);
      Kokkos::fence();const double local=MPI_Wtime()-tick;
      // All timing reductions and rank attribution happen after the timestamp.
      elapsed[window]=local;
      MPI_Allreduce(MPI_IN_PLACE,&elapsed[window],1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
      MPI_Gather(&local,1,MPI_DOUBLE,rank_seconds.data()+window*e.size,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
    }
    for(int step=start_cycle;step<end_cycle;++step)expected_time+=dt;
    All(e.steps==end_cycle&&e.time==expected_time,"incorrect complete-step accounting");
    for(double value:elapsed)All(std::isfinite(value)&&value>0,"invalid timing interval");
    const double write_start=MPI_Wtime();Snapshot(e,out);Kokkos::fence();
    double output_seconds=MPI_Wtime()-write_start;
    MPI_Allreduce(MPI_IN_PLACE,&output_seconds,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    const double diagnostic_start=MPI_Wtime();const auto diagnostic=Diagnose(e,input,out);
    Kokkos::fence();double diagnostic_seconds=MPI_Wtime()-diagnostic_start;
    MPI_Allreduce(MPI_IN_PLACE,&diagnostic_seconds,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    CollectiveLocal([&] {if(e.rank==0) {
      std::ofstream file(out/"results.json");file<<std::setprecision(17)
        <<"{\"schema\":\"legacy-n12-matched-benchmark-v1\",\"passed\":true,\"ranks\":24,"
        <<"\"backend\":\""<<Kokkos::DefaultExecutionSpace::name()<<"\",\"start_cycle\":"<<start_cycle
        <<",\"end_cycle\":"<<e.steps<<",\"time\":"<<e.time<<",\"dt\":"<<dt
        <<",\"warmup\":10,\"window_steps\":20,\"windows\":3,\"bulk_points\":229392,\"scri_points\":864,"
        <<"\"initial_fields_bitwise_seed\":true,\"initial_records_bitwise_seed\":true,"
        <<"\"profiling_library_loaded\":false,\"detailed_timers_enabled\":false,"
        <<"\"setup_including_seed_verification_seconds\":"<<setup<<",\"warmup_seconds\":"<<warm_seconds
        <<",\"accepted_output_seconds\":"<<output_seconds<<",\"diagnostic_and_horizon_output_seconds\":"<<diagnostic_seconds
        <<",\"maximum_rank_window_seconds\":[";
      for(int i=0;i<windows;++i){if(i)file<<',';file<<elapsed[i];}
      file<<"],\"rank_window_seconds\":[";
      for(int i=0;i<windows*e.size;++i){if(i)file<<',';file<<rank_seconds[i];}
      file<<"],\"full_and_horizon_exterior_sums\":[";
      for(int i=0;i<10;++i){if(i)file<<',';file<<diagnostic.sums[i];}
      file<<"],\"fixed_outer_sums\":[";
      for(int i=0;i<5;++i){if(i)file<<',';file<<diagnostic.fixed[i];}
      file<<"],\"compatibility_max\":"<<diagnostic.compatibility<<",\"clock_max\":"<<diagnostic.clock
          <<",\"CFL_max\":"<<diagnostic.cfl<<",\"complete_RK4_steps_timed\":60,"
          <<"\"performance_admission_pending\":true}\n";
      file.close();Require(bool(file),"result write failed");
    }});
  }catch(const std::exception &error) {
    std::cerr<<"rank "<<global_variable::my_rank<<": "<<error.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);
  }
  Kokkos::finalize();MPI_Finalize();return 0;
}
