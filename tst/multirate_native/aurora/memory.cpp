// Kokkos allocation high-water marks, per MPI rank and memory space.
#include <algorithm>
#include <cstdint>
#include <cstdio>
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
