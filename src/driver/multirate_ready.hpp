// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef DRIVER_MULTIRATE_READY_HPP_
#define DRIVER_MULTIRATE_READY_HPP_

#include <map>
#include <stdexcept>
#include <vector>

namespace time_integration::multirate {
// Compile record consumers once per placement. Ready work is grouped by the
// caller's stage/kernel key. Only the predecessors of each arrival are touched.
template <class Record, class Batch>
class RecordReadiness {
 public:
  void Add(const Batch& batch, const std::vector<Record>& dependencies) {
    const auto index = batches_.size();
    batches_.push_back(batch);
    initial_.push_back(dependencies.size());
    for (const auto& key : dependencies) consumers_[key].push_back(index);
  }
  void Reset() {
    ready_.clear();
    missing_ = initial_;
    for (std::size_t i = 0; i < missing_.size(); ++i)
      if (!missing_[i]) ready_[batches_[i]].push_back(i);
  }
  void Arrive(const Record& key) {
    const auto found = consumers_.find(key);
    if (found == consumers_.end()) return;
    for (const auto task : found->second) {
      if (!missing_[task]) throw std::logic_error("Repeated multirate record arrival");
      if (--missing_[task] == 0) ready_[batches_[task]].push_back(task);
    }
  }
  std::vector<std::size_t> TakeReady() {
    if (ready_.empty()) return {};
    auto first = ready_.begin();
    auto batch = std::move(first->second);
    ready_.erase(first);
    return batch;
  }

 private:
  std::vector<Batch> batches_;
  std::map<Record, std::vector<std::size_t>> consumers_;
  std::vector<std::size_t> initial_, missing_;
  std::map<Batch, std::vector<std::size_t>> ready_;
};
}  // namespace time_integration::multirate
#endif
