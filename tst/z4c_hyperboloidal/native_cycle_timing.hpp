// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Test-only whole-cycle boundaries; no numerical task or state is modified.
#ifndef TST_Z4C_HYPERBOLOIDAL_NATIVE_CYCLE_TIMING_HPP_
#define TST_Z4C_HYPERBOLOIDAL_NATIVE_CYCLE_TIMING_HPP_
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "athena.hpp"
#include "globals.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"
#include "z4c_hyperboloidal/z4c_hyperboloidal.hpp"
#include <impl/Kokkos_Profiling.hpp>
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#else
#error "Complete-cycle timing requires MPI."
#endif

namespace native_cycle_timing {
using Clock=std::chrono::steady_clock;
inline double Seconds(Clock::time_point a,Clock::time_point b) {
  return std::chrono::duration<double>(b-a).count();
}
inline void RequireAll(bool okay,const char *message) {
  int bad=okay?0:1;
  MPI_Allreduce(MPI_IN_PLACE,&bad,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
  if(bad)throw std::runtime_error(message);
}
inline bool PositiveEnvironment(const char *name,int &value) {
  const char *text=std::getenv(name);if(!text||!*text)return false;
  errno=0;char *end=nullptr;const long number=std::strtol(text,&end,10);
  if(errno||!end||*end||number<=0||number>150)return false;
  value=static_cast<int>(number);return true;
}
inline bool Present(const char *name) {
  const char *value=std::getenv(name);return value&&*value;
}
struct Window {
  int first=0,last=0;double time_first=0,time_last=0,local=0,maximum=0;
};
class Timer {
 public:
  Timer(Mesh *mesh,Driver *driver,bool restart):mesh_(mesh),driver_(driver),
      initial_cycle_(mesh->ncycle),initial_time_(mesh->time),restart_(restart) {
    bool valid=PositiveEnvironment("NATIVE_TIMING_WARMUP",warmup_) &&
        PositiveEnvironment("NATIVE_TIMING_STEPS",steps_) &&
        PositiveEnvironment("NATIVE_TIMING_WINDOWS",windows_);
    RequireAll(valid,"timing requires positive warmup/steps/windows environment values");
    int config[3]={warmup_,steps_,windows_},minimum[3],maximum[3];
    MPI_Allreduce(config,minimum,3,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
    MPI_Allreduce(config,maximum,3,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
    RequireAll(std::equal(minimum,minimum+3,maximum),"rank-dependent timing configuration");
    total_=warmup_+steps_*windows_;
    profile_loaded_=Kokkos::Tools::profileLibraryLoaded();
    const char *names[]={"KOKKOS_TOOLS_LIBS","KOKKOS_PROFILE_LIBRARY","KOKKOS_TOOLS_ARGS",
      "SYCL_PI_TRACE","SYCL_UR_TRACE","ZE_ENABLE_TRACING_LAYER","UNITRACE_ENABLE",
      "PTI_ENABLE","VTUNE_PROFILER_ENABLE"};
    for(const char *name:names)if(Present(name))profile_environment_.push_back(name);
    RequireAll(!profile_loaded_&&profile_environment_.empty(),
               "primary timing refuses loaded profiling tools or profiling environment");
    RequireAll(total_<=150 && mesh->pmb_pack->pz4c_hyperboloidal!=nullptr &&
        driver->UsesClassicalRK4() && driver->nexp_stages==4 && !mesh->adaptive &&
        driver->nlim==initial_cycle_+total_ && driver->wall_time<=0 && driver->ndiag>0,
        "timing requires fixed native RK4, exact bounded cycle limit and no wall limit");
    RequireAll(global_variable::my_rank!=0||!std::filesystem::exists("native-cycle-timing.json"),
               "timing refuses an existing report");
    windows_data_.resize(windows_);
  }
  void Initialized(Outputs *outputs) {
    outputs_=outputs;auto &module=*mesh_->pmb_pack->pz4c_hyperboloidal;
    dt_=mesh_->dt;expected_time_=initial_time_;
    bool valid=std::isfinite(dt_)&&dt_>0&&dt_==module.requested_dt&&
        mesh_->ncycle==initial_cycle_&&mesh_->time==initial_time_;
    for(int b=0;b<mesh_->nmb_total;++b)valid=valid&&mesh_->cost_eachmb[b]==1;
    // Match the real native predicate, including float casts. No output schedule
    // is changed here. Conservatively require the entire short run output-free.
    Real prospective=initial_time_;
    for(int c=1;c<=total_;++c) {
      valid=valid&&prospective<driver_->tlim;
      if(c>warmup_ && (initial_cycle_+c-1)%driver_->ndiag==0)valid=false;
      prospective+=dt_;
      for(const auto &out:outputs_->pout_list) {
        const auto &op=out->out_params;
        const bool due=(op.dt>0 && static_cast<float>(prospective)>=
            static_cast<float>(op.last_time+op.dt) &&
            static_cast<float>(prospective)<static_cast<float>(driver_->tlim)) ||
            (op.dcycle>0 && (initial_cycle_+c)%op.dcycle==0);
        valid=valid&&!due;
      }
    }
    RequireAll(valid,"timing preflight found changed dt/weights, insufficient time limit or scheduled output");
    final_time_=prospective;blocks_=mesh_->nmb_total;
    std::uint64_t local[2]={static_cast<std::uint64_t>(module.npoints),
                           static_cast<std::uint64_t>(module.nsurface_points)};
    MPI_Allreduce(local,counts_,2,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
    RequireAll(counts_[0]>0&&counts_[1]>0,"timing requires owned bulk and scri work");
    Kokkos::fence("timing startup completion");
    const double startup=driver_->pwall_clock_->seconds();
    MPI_Allreduce(&startup,&startup_max_,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    initialized_=true;
  }
  void ExecuteBegin(bool watchdog) {
    RequireAll(initialized_&&!watchdog,"timing requires completed initialization and no watchdog");
    execute_start_=Clock::now();
  }
  void BeforeCycle() {
    if(seen_==0)warmup_start_=Clock::now();
    invalid_|=mesh_->ncycle!=initial_cycle_+seen_||mesh_->time!=expected_time_||
        mesh_->dt!=dt_||mesh_->nmb_total!=blocks_;
    if(seen_>=warmup_ && (seen_-warmup_)%steps_==0 && seen_<total_) {
      // All validation collectives occur before the timing barrier/start stamp.
      RequireAll(!invalid_&&!active_,"timing window starts from inconsistent cycle metadata");
      Kokkos::fence("timing window start");
      if(seen_==warmup_)warmup_local_=Seconds(warmup_start_,Clock::now());
      MPI_Barrier(MPI_COMM_WORLD);
      start_=Clock::now();active_=true;
      auto &window=windows_data_[(seen_-warmup_)/steps_];
      window.first=mesh_->ncycle;window.time_first=mesh_->time;
    }
  }
  void AfterCycle() {
    expected_time_+=dt_;++seen_;
    invalid_|=mesh_->ncycle!=initial_cycle_+seen_||mesh_->time!=expected_time_||
        mesh_->dt!=dt_||mesh_->dt_last_completed!=dt_||mesh_->nmb_total!=blocks_;
    if(active_ && (seen_-warmup_)%steps_==0) {
      Kokkos::fence("timing window completion");
      const auto stop=Clock::now();
      auto &window=windows_data_[(seen_-warmup_)/steps_-1];
      window.local=Seconds(start_,stop);active_=false;
      window.last=mesh_->ncycle;window.time_last=mesh_->time;
      // This MAX is outside the measured interval. No per-step reduction exists.
      MPI_Allreduce(&window.local,&window.maximum,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
      invalid_|=!std::isfinite(window.maximum)||window.maximum<=0;
    }
  }
  void Output() {if(active_){++timed_outputs_;invalid_=true;}}
  void Console() {if(active_){++timed_console_;invalid_=true;}}
  void ExecuteEnd() {
    invalid_|=active_||seen_!=total_||mesh_->ncycle!=initial_cycle_+total_||
        mesh_->time!=final_time_||mesh_->dt!=dt_;
    for(const auto &window:windows_data_)invalid_|=window.last-window.first!=steps_;
    RequireAll(!invalid_,"timing coverage, metadata or output exclusion failed");
    const double elapsed=Seconds(execute_start_,Clock::now());
    MPI_Allreduce(&elapsed,&execute_max_,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    MPI_Allreduce(&warmup_local_,&warmup_max_,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    executed_=true;
  }
  void FinalizeBegin() {
    RequireAll(executed_,"timing finalization before complete windows");
    finalize_start_=Clock::now();
  }
  void FinalizeEnd() {
    Kokkos::fence("timing final output completion");
    const double elapsed=Seconds(finalize_start_,Clock::now());
    MPI_Allreduce(&elapsed,&finalize_max_,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    // Export existing samples once, after every measured interval and elapsed stamp.
    const int rank=global_variable::my_rank,nranks=global_variable::nranks;
    std::vector<double> local(windows_),rank_seconds(rank==0?nranks*windows_:0);
    for(int i=0;i<windows_;++i)local[i]=windows_data_[i].local;
    MPI_Gather(local.data(),windows_,MPI_DOUBLE,rank_seconds.data(),windows_,MPI_DOUBLE,0,MPI_COMM_WORLD);
    bool samples_valid=true;
    if(rank==0)for(int i=0;i<windows_;++i) {
      double maximum=0;
      for(int r=0;r<nranks;++r) {
        const double value=rank_seconds[r*windows_+i];
        samples_valid&=std::isfinite(value)&&value>0;maximum=std::max(maximum,value);
      }
      samples_valid&=maximum==windows_data_[i].maximum;
    }
    RequireAll(samples_valid,"gathered window samples differ from original maximum");
    bool written=true;
    if(global_variable::my_rank==0) {
      std::ofstream out("native-cycle-timing.json");out<<std::setprecision(17);
      out<<"{\"schema\":\"native-cycle-timing-v1\",\"passed\":true,"
        <<"\"scope\":\"unadmitted bounded timing; external scientific and campaign gates required\","
        <<"\"backend\":\""<<DevExeSpace::name()<<"\",\"ranks\":"<<global_variable::nranks
        <<",\"restart\":"<<(restart_?"true":"false")<<",\"blocks\":"<<blocks_
        <<",\"bulk_points\":"<<counts_[0]<<",\"scri_points\":"<<counts_[1]
        <<",\"initial_cycle\":"<<initial_cycle_<<",\"final_cycle\":"<<mesh_->ncycle
        <<",\"initial_time\":"<<initial_time_<<",\"final_time\":"<<mesh_->time
        <<",\"dt\":"<<dt_<<",\"complete_steps\":"<<seen_
        <<",\"warmup_steps\":"<<warmup_<<",\"steps_per_window\":"<<steps_
        <<",\"window_count\":"<<windows_<<",\"console_ndiag\":"<<driver_->ndiag
        <<",\"timed_output_count\":"<<timed_outputs_<<",\"timed_console_count\":"<<timed_console_
        <<",\"profiling_library_loaded\":false,\"profiling_environment\":[],"
        <<"\"startup_max_seconds\":"<<startup_max_
        <<",\"startup_scope\":\"original main wall timer after Kokkos/MPI/argument setup through driver initialization\""
        <<",\"warmup_max_seconds\":"<<warmup_max_<<",\"execute_max_seconds\":"<<execute_max_
        <<",\"finalize_max_seconds\":"<<finalize_max_
        <<",\"finalize_scope\":\"original Driver::Finalize including final outputs and final completion fence\""
        <<",\"windows\":[";
      for(int i=0;i<windows_;++i) {
        if(i)out<<',';const auto &w=windows_data_[i];
        out<<"{\"index\":"<<i<<",\"first_cycle\":"<<w.first<<",\"last_cycle\":"<<w.last
          <<",\"first_time\":"<<w.time_first<<",\"last_time\":"<<w.time_last
          <<",\"complete_steps\":"<<steps_<<",\"max_rank_seconds\":"<<w.maximum
          <<",\"seconds_per_complete_step\":"<<w.maximum/steps_
          <<",\"bulk_cell_updates\":"<<counts_[0]*std::uint64_t(steps_)
          <<",\"bulk_cell_updates_per_second\":"<<counts_[0]*std::uint64_t(steps_)/w.maximum
          <<",\"rank_order_seconds\":[";
        for(int r=0;r<nranks;++r){if(r)out<<',';out<<rank_seconds[r*windows_+i];}
        out<<"],\"max_rank_ids\":[";bool first=true;
        for(int r=0;r<nranks;++r)if(rank_seconds[r*windows_+i]==w.maximum) {
          if(!first)out<<',';out<<r;first=false;
        }
        out<<"]}";
      }
      out<<"]}\n";out.close();written=out.good();
    }
    RequireAll(written,"timing result write failed");
  }
 private:
  Mesh *mesh_;Driver *driver_;Outputs *outputs_=nullptr;
  int initial_cycle_,warmup_=0,steps_=0,windows_=0,total_=0,seen_=0,blocks_=0;
  Real initial_time_,dt_=0,expected_time_=0,final_time_=0;
  bool restart_,profile_loaded_=false,initialized_=false,executed_=false,active_=false,invalid_=false;
  std::uint64_t counts_[2]{},timed_outputs_=0,timed_console_=0;
  std::vector<std::string> profile_environment_;
  std::vector<Window> windows_data_;
  Clock::time_point execute_start_,warmup_start_,start_,finalize_start_;
  double startup_max_=0,warmup_local_=0,warmup_max_=0,execute_max_=0,finalize_max_=0;
};
inline std::unique_ptr<Timer> timer;
inline void BeginInitialize(Mesh *m,Driver *d,bool restart){timer=std::make_unique<Timer>(m,d,restart);}
inline void EndInitialize(Outputs *o){timer->Initialized(o);}
inline void BeginExecute(bool watchdog){timer->ExecuteBegin(watchdog);}
inline void BeforeCycle(){timer->BeforeCycle();}
inline void AfterCycle(){timer->AfterCycle();}
inline void Output(){if(timer)timer->Output();}
inline void Console(){if(timer)timer->Console();}
inline void EndExecute(){timer->ExecuteEnd();}
inline void BeginFinalize(){timer->FinalizeBegin();}
inline void EndFinalize(){timer->FinalizeEnd();}
}  // namespace native_cycle_timing
#endif
