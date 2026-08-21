// Minimal synchronized Kokkos profiling tool for ranked per-kernel timings.
// Kokkos fences before begin/end callbacks when global fencing is requested,
// so the intervals measure completed device work instead of enqueue latency.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct Kokkos_Profiling_KokkosPDeviceInfo;
struct Kokkos_Tools_ToolSettings {
  bool requires_global_fencing;
  bool padding[255];
};

namespace {
using Clock = std::chrono::steady_clock;

struct ActiveKernel {
  std::string name;
  Clock::time_point start;
};

struct KernelStats {
  std::uint64_t calls = 0;
  double total_seconds = 0.0;
};

std::mutex timer_mutex;
std::uint64_t next_id = 1;
std::unordered_map<std::uint64_t, ActiveKernel> active;
std::unordered_map<std::string, KernelStats> totals;

void BeginKernel(const char *name, std::uint64_t *kernel_id) {
  std::lock_guard<std::mutex> lock(timer_mutex);
  *kernel_id = next_id++;
  active.emplace(*kernel_id, ActiveKernel{name, Clock::now()});
}

void EndKernel(const std::uint64_t kernel_id) {
  const auto stop = Clock::now();
  std::lock_guard<std::mutex> lock(timer_mutex);
  const auto found = active.find(kernel_id);
  if (found == active.end()) return;
  const double seconds =
      std::chrono::duration<double>(stop - found->second.start).count();
  auto &stats = totals[found->second.name];
  ++stats.calls;
  stats.total_seconds += seconds;
  active.erase(found);
}
}  // namespace

extern "C" void kokkosp_request_tool_settings(
    const std::uint32_t, Kokkos_Tools_ToolSettings *settings) {
  settings->requires_global_fencing = true;
}

extern "C" void kokkosp_init_library(
    const int, const std::uint64_t, const std::uint32_t,
    Kokkos_Profiling_KokkosPDeviceInfo *) {}

extern "C" void kokkosp_begin_parallel_for(
    const char *name, const std::uint32_t, std::uint64_t *kernel_id) {
  BeginKernel(name, kernel_id);
}

extern "C" void kokkosp_end_parallel_for(const std::uint64_t kernel_id) {
  EndKernel(kernel_id);
}

extern "C" void kokkosp_begin_parallel_reduce(
    const char *name, const std::uint32_t, std::uint64_t *kernel_id) {
  BeginKernel(name, kernel_id);
}

extern "C" void kokkosp_end_parallel_reduce(const std::uint64_t kernel_id) {
  EndKernel(kernel_id);
}

extern "C" void kokkosp_begin_parallel_scan(
    const char *name, const std::uint32_t, std::uint64_t *kernel_id) {
  BeginKernel(name, kernel_id);
}

extern "C" void kokkosp_end_parallel_scan(const std::uint64_t kernel_id) {
  EndKernel(kernel_id);
}

extern "C" void kokkosp_finalize_library() {
  std::vector<std::pair<std::string, KernelStats>> sorted(totals.begin(),
                                                          totals.end());
  std::sort(sorted.begin(), sorted.end(), [](const auto &left, const auto &right) {
    return left.second.total_seconds > right.second.total_seconds;
  });
  double grand_total = 0.0;
  for (const auto &entry : sorted) grand_total += entry.second.total_seconds;

  const char *configured_path = std::getenv("REF_GH_KERNEL_TIMING_FILE");
  const std::string path =
      configured_path == nullptr ? "kokkos_kernel_timing.tsv" : configured_path;
  std::ofstream output(path);
  output << "kernel\tcalls\ttotal_seconds\taverage_seconds\tpercent\n";
  output << std::setprecision(12);
  for (const auto &entry : sorted) {
    const auto &stats = entry.second;
    const double percentage =
        grand_total == 0.0 ? 0.0 : 100.0*stats.total_seconds/grand_total;
    output << entry.first << '\t' << stats.calls << '\t' << stats.total_seconds
           << '\t' << stats.total_seconds/static_cast<double>(stats.calls)
           << '\t' << percentage << '\n';
  }
}
