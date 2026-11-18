// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef DRIVER_MULTIRATE_READY_HPP_
#define DRIVER_MULTIRATE_READY_HPP_

#include <map>
#include <chrono>
#include <thread>
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
// Both callbacks receive indices into the fixed block-stage task table. The
// interior callback signals the local completion event after its RHS writes.
// The coupled callback prepares the recipient view, evaluates boundary terms,
// checks validity, publishes its slope and commits completed local steps.
template <class Queue, class Progress, class Interior, class Coupled>
double DispatchReadyBatches(Queue& interior, Queue& coupled, std::size_t task_count,
                            Progress progress, Interior evaluate_interior,
                            Coupled evaluate_coupled) {
  using Clock = std::chrono::steady_clock;
  double waiting_seconds = 0;
  std::size_t completed = 0;
  while (completed < task_count) {
    const auto start = Clock::now();
    for (const auto& record : progress()) {
      coupled.Arrive(record);
      interior.Arrive(record);
    }
    auto batch = coupled.TakeReady();
    if (!batch.empty()) {
      evaluate_coupled(batch);
      completed += batch.size();
      continue;
    }
    batch = interior.TakeReady();
    if (!batch.empty()) {
      evaluate_interior(batch);
      continue;
    }
    std::this_thread::yield();
    waiting_seconds += std::chrono::duration<double>(Clock::now() - start).count();
  }
  return waiting_seconds;
}
}  // namespace time_integration::multirate
#endif
