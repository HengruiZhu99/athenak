// Kokkos allocation high-water marks, per MPI rank and memory space.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <stdexcept>
#include <map>
#include <mutex>
#include <string>
#include "impl/Kokkos_Profiling_C_Interface.h"

namespace {
struct Counts {
  std::uint64_t live = 0, peak = 0;
};
std::map<std::string, Counts> spaces;
std::mutex counter_lock;
}

extern "C" void kokkosp_init_library(int, std::uint64_t, std::uint32_t,
                                     Kokkos_Profiling_KokkosPDeviceInfo*) {}

extern "C" void kokkosp_request_tool_settings(std::uint32_t, Kokkos_Tools_ToolSettings* settings) {
  settings->requires_global_fencing = false;
}

extern "C" void kokkosp_allocate_data(Kokkos_Profiling_SpaceHandle space, const char*,
                                      const void*, std::uint64_t size) {
  std::lock_guard<std::mutex> guard(counter_lock);
  auto& stats = spaces[space.name];
  stats.live += size;
  stats.peak = std::max(stats.peak, stats.live);
}

extern "C" void kokkosp_deallocate_data(Kokkos_Profiling_SpaceHandle space, const char*,
                                        const void*, std::uint64_t size) {
  std::lock_guard<std::mutex> guard(counter_lock);
  spaces.at(space.name).live -= size;
}

extern "C" void kokkosp_finalize_library() {
  for (const auto& entry : spaces)
    std::printf("MULTIRATE_MEMORY space=%s peak_allocated_bytes=%llu\n", entry.first.c_str(),
                static_cast<unsigned long long>(entry.second.peak));
}

// Optional, benchmark-only unitrace collection window. Leave the library paused
// during warmup, plan construction, diagnostics, output and program shutdown.
namespace {
using Toggle = void (*)();
std::uint64_t region_count=0, skip_steps=0;
std::size_t region_depth=0;
Toggle resume_trace=nullptr, pause_trace=nullptr;
}
extern "C" void kokkosp_push_profile_region(const char* name) {
  if (region_depth) {++region_depth;return;}
  if (std::strcmp(name,"AthenaK integrator tasks")!=0) return;
  if (region_count==0) {
    if (const char* value=std::getenv("MR_TRACE_SKIP_STEPS")) {
      skip_steps=std::strtoull(value,nullptr,10);
      resume_trace=reinterpret_cast<Toggle>(dlsym(RTLD_DEFAULT,"__itt_resume"));
      pause_trace=reinterpret_cast<Toggle>(dlsym(RTLD_DEFAULT,"__itt_pause"));
      if (!resume_trace || !pause_trace) throw std::runtime_error("unitrace ITT entry points unavailable");
    }
  }
  if (++region_count>skip_steps && resume_trace) {resume_trace();region_depth=1;}
}
extern "C" void kokkosp_pop_profile_region() {
  if (region_depth && --region_depth==0) pause_trace();
}
