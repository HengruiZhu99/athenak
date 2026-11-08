// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Diagnostic-only entry. The included qualified helpers/numerical tree are unchanged.
#define main legacy_n12_unused_benchmark_main
#include "legacy_n12_benchmark.cpp"
#undef main

namespace legacy_n12_endpoint_diagnostics {
using namespace legacy_n12_benchmark;  // NOLINT(build/namespaces)

void RestoreAndVerify(Evolution &e,const std::filesystem::path &accepted) {
  CollectiveLocal([&] {
    for(int region=0;region<2;++region) {
      const int first=region?e.surface_first:e.first,count=region?e.surface_local:e.local;
      const auto path=accepted/(std::string("accepted-final")+(region?"-scri-rank":"-bulk-rank")+
                                std::to_string(e.rank)+".bin");
      Require(std::filesystem::file_size(path)==40+std::uint64_t(count)*336,
              "accepted trace extent differs");
      std::ifstream file(path,std::ios::binary);char magic[16]{};std::uint64_t size=0;
      std::uint32_t shape[4]{};file.read(magic,16);file.read(reinterpret_cast<char*>(&size),8);
      file.read(reinterpret_cast<char*>(shape),16);
      Require(bool(file),"accepted trace header read failed");
      const char expected_magic[16]="LEGACY_TRACE_V1";
      Require(std::memcmp(magic,expected_magic,16)==0&&size==std::uint64_t(count)&&
          shape[0]==6&&shape[1]==39&&shape[2]==std::uint32_t(region)&&shape[3]==1,
          "accepted trace header differs");
      const auto fields=Owned(e,region);
      auto target=region?e.spherical_surface_chi_records:e.spherical_chi_records;
      auto records=Kokkos::create_mirror_view(target);
      std::vector<Real> expected_records(8*count);
      for(int q=0;q<count;++q) {
        std::int32_t key[6]{},expected[6]{};Real row[39]{};
        file.read(reinterpret_cast<char*>(key),sizeof(key));file.read(reinterpret_cast<char*>(row),sizeof(row));
        Require(bool(file),"accepted trace row read failed");
        const int ordinal=first+q;
        if(region) {
          const auto &c=e.Model().scri.rows[ordinal];
          const std::int32_t value[6]={ordinal,c.block,c.k,c.j,12,e.Model().grid.mesh.blocks[c.block].face};
          std::copy_n(value,6,expected);
        } else {
          const auto &c=e.Model().grid.cells[ordinal];
          const std::int32_t value[6]={ordinal,c.block,c.k,c.j,c.i,e.Model().grid.mesh.blocks[c.block].face};
          std::copy_n(value,6,expected);
        }
        Require(std::equal(key,key+6,expected),"accepted trace ordering/geometry keys differ");
        for(Real value:row)Require(std::isfinite(value),"nonfinite accepted field/history");
        Require(std::memcmp(fields.data()+31*q,row,31*sizeof(Real))==0,
                "post-load fields differ from control accepted endpoint");
        const Real *v=row+31;auto &r=records(q);
        Require(v[2]>=0&&v[3]>=0&&v[4]>=0&&v[5]>=0&&v[6]>=0&&
                std::floor(v[2])==v[2]&&std::floor(v[3])==v[3]&&std::floor(v[4])==v[4]&&
                v[2]<=9007199254740992.&&v[3]<=9007199254740992.&&v[4]<=9007199254740992.,
                "invalid accepted chi record");
        r.minimum=v[0];r.accepted_minimum=v[1];r.floor_count=v[2];
        r.nonpositive_count=v[3];r.nonfinite_count=v[4];r.max_correction=v[5];
        r.sum_correction=v[6];r.first_floor=v[7];std::copy_n(v,8,expected_records.data()+8*q);
      }
      Require(bool(file),"accepted trace read failed");Kokkos::deep_copy(target,records);
      const auto actual_records=Records(e,region);
      Require(actual_records.size()==expected_records.size()&&
              (expected_records.empty()||std::memcmp(actual_records.data(),expected_records.data(),expected_records.size()*sizeof(Real))==0),
              "accepted chi history restoration differs");
    }
  });
}
}  // namespace legacy_n12_endpoint_diagnostics

int main(int argc,char **argv) {
  MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);Kokkos::initialize(argc,argv);
  try {
    using namespace legacy_n12_endpoint_diagnostics;  // NOLINT(build/namespaces)
    static_assert(sizeof(Real)==8&&h::spherical_policy,"float64 SP1 policy required");
    All(argc==4&&global_variable::nranks==24,
        "legacy_n12_endpoint_diagnostics SEED_DIRECTORY ACCEPTED_DIRECTORY NEW_OUTPUT;24ranks");
    const std::filesystem::path seed=argv[1],accepted=argv[2],out=argv[3],input=seed/"run.in";
    All(!std::filesystem::exists(out),"refuse existing diagnostic output");
    CollectiveLocal([&]{if(global_variable::my_rank==0)std::filesystem::create_directories(out);});
    MPI_Barrier(MPI_COMM_WORLD);
    bool clean=!Kokkos::Tools::profileLibraryLoaded();
    for(const char *name:{"KOKKOS_TOOLS_LIBS","KOKKOS_PROFILE_LIBRARY","KOKKOS_TOOLS_ARGS",
        "SYCL_PI_TRACE","SYCL_UR_TRACE","ZE_ENABLE_TRACING_LAYER","UNITRACE_ENABLE",
        "PTI_ENABLE","VTUNE_PROFILER_ENABLE"})clean&=std::getenv(name)==nullptr;
    All(clean,"diagnostic qualification rejects external profiling");
    const auto controls=EndpointRunControls::Load(input);
    All(controls.cells_per_axis==12&&controls.nlim==end_cycle&&controls.dt==dt&&controls.tlim==2&&
        controls.ko==.1&&controls.chi_floor==1e-10&&!controls.track_puncture,
        "changed endpoint diagnostic physics/input");
    perf::enabled=false;perf::small_groups=std::getenv("HYPERBOLIC_PROFILE_SMALL_GROUPS")!=nullptr;
    hyperbolic_reproduction::refined_configuration.Load(controls.configuration.string(),
        hyperbolic_reproduction::ContinuousCoreMesh(12));
    auto mesh=EndpointStaticMesh(hyperbolic_reproduction::refined_configuration,12);
    Evolution e(12,0,2,"",&mesh);e.ConfigureSphericalNumerics(.1,1e-10);
    e.LoadCheckpoint((accepted/"endpoint-restart.bin").string(),2,dt,mesh.amr_cartesian_core);
    e.track_puncture=false;e.puncture={0,0,0};
    Real expected_time=0;for(int step=0;step<end_cycle;++step)expected_time+=dt;
    All(e.total==229392&&e.surface_total==864&&e.steps==end_cycle&&e.time==expected_time,
        "accepted endpoint coverage/time differs");
    RestoreAndVerify(e,accepted);
    // Capture exactly what this diagnostic-only process consumed before RHS.
    Snapshot(e,out);
    const auto before_bulk=Records(e,0),before_surface=Records(e,1);
    const auto diagnostic=Diagnose(e,input,out);Kokkos::fence();
    const auto after_bulk=Records(e,0),after_surface=Records(e,1);
    All(before_bulk.size()==after_bulk.size()&&before_surface.size()==after_surface.size()&&
        (before_bulk.empty()||std::memcmp(before_bulk.data(),after_bulk.data(),before_bulk.size()*sizeof(Real))==0)&&
        (before_surface.empty()||std::memcmp(before_surface.data(),after_surface.data(),before_surface.size()*sizeof(Real))==0),
        "diagnostic RHS changed accepted chi history bits");
    All(e.steps==end_cycle&&e.time==expected_time,"diagnostic process advanced time or steps");
    CollectiveLocal([&]{if(e.rank==0) {
      std::ofstream file(out/"results.json");file<<std::setprecision(17)
        <<"{\"schema\":\"legacy-n12-endpoint-diagnostics-v1\",\"passed\":true,\"ranks\":24,"
        <<"\"backend\":\""<<Kokkos::DefaultExecutionSpace::name()<<"\",\"cycle\":"<<e.steps
        <<",\"time\":"<<e.time<<",\"dt\":"<<dt<<",\"bulk_points\":229392,\"scri_points\":864,"
        <<"\"post_load_fields_bitwise_control_endpoint\":true,\"post_load_records_bitwise_control_endpoint\":true,"
        <<"\"diagnostics_preserved_chi_records_bitwise\":true,\"evolution_steps\":0,\"timing_measurement\":false,"
        <<"\"diagnostic_operator\":\"unchanged v8\",\"horizon_seed\":\"original2486 run.in\","
        <<"\"full_and_horizon_exterior_sums\":[";
      for(int i=0;i<10;++i){if(i)file<<',';file<<diagnostic.sums[i];}
      file<<"],\"fixed_outer_sums\":[";
      for(int i=0;i<5;++i){if(i)file<<',';file<<diagnostic.fixed[i];}
      file<<"],\"compatibility_max\":"<<diagnostic.compatibility<<",\"clock_max\":"<<diagnostic.clock
          <<",\"CFL_max\":"<<diagnostic.cfl<<"}\n";
      file.close();Require(bool(file),"endpoint diagnostic result write failed");
    }});
  }catch(const std::exception &error) {
    std::cerr<<"rank "<<global_variable::my_rank<<": "<<error.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);
  }
  Kokkos::finalize();MPI_Finalize();return 0;
}
