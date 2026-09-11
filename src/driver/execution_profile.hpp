#ifndef DRIVER_EXECUTION_PROFILE_HPP_
#define DRIVER_EXECUTION_PROFILE_HPP_
// Opt-in, fenced, hierarchical wall-time attribution. Disabled path adds no fences.
// Inclusive parents overlap children; only exclusive seconds may be summed.
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <map>
#include <stdexcept>
#include <string>
#include <Kokkos_Core.hpp>

namespace execution_profile {
struct Entry { std::uint64_t calls = 0; double inclusive = 0, exclusive = 0; };
inline bool enabled = false;
inline std::map<std::string, Entry> entries;
class Scope {
 public:
  explicit Scope(const std::string &name) : active_(enabled), name_(name) {
    if (!active_) return;
    Kokkos::fence("profile begin");
    parent_ = current_;
    current_ = this;
    start_ = Clock::now();
  }
  ~Scope() {
    if (!active_) return;
    Kokkos::fence("profile end");
    const double seconds = std::chrono::duration<double>(Clock::now()-start_).count();
    auto &entry = entries[name_];
    ++entry.calls;
    entry.inclusive += seconds;
    entry.exclusive += seconds-children_;
    current_ = parent_;
    if (parent_) parent_->children_ += seconds;
  }
  Scope(const Scope &) = delete;
  Scope &operator=(const Scope &) = delete;
 private:
  using Clock = std::chrono::steady_clock;
  bool active_;
  std::string name_;
  Clock::time_point start_;
  double children_ = 0;
  Scope *parent_ = nullptr;
  inline static Scope *current_ = nullptr;
};
inline void Write(int rank) {
  if (!enabled) return;
  std::ofstream out("execution_profile.rank"+std::to_string(rank)+".csv");
  out << "region,calls,inclusive_seconds,exclusive_seconds\n" << std::setprecision(17);
  for (const auto &item : entries) {
    out << item.first << ',' << item.second.calls << ',' << item.second.inclusive
        << ',' << item.second.exclusive << '\n';
  }
  if (!out) throw std::runtime_error("cannot write execution profile");
}
}  // namespace execution_profile
#endif  // DRIVER_EXECUTION_PROFILE_HPP_
