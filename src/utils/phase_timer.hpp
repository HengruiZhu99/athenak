#ifndef UTILS_PHASE_TIMER_HPP_
#define UTILS_PHASE_TIMER_HPP_
//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file phase_timer.hpp
//! \brief Host wall-clock receipts for start-up, restart and regrid phases.
//!
//! Report() prints one line on rank 0,
//!   PHASE_SECONDS scope=<scope> cycle=<n> total=<max>,<mean> <phase>=<max>,<mean> ...
//! where the maximum and the mean are taken over the MPI ranks. It is collective: every
//! rank must have marked the same phases in the same order. The timer only reads a
//! steady clock. It adds no device fence, so asynchronous kernel time is charged to the
//! phase in which the host first waits for it.

#include <chrono>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"

class PhaseTimer {
 public:
  PhaseTimer(const char *scope, int cycle)
      : scope_(scope), cycle_(cycle), start_(Clock::now()), last_(start_) {}

  //! Charge the time since the previous mark (or construction) to this phase.
  void Mark(const char *phase) {
    const auto now = Clock::now();
    const double seconds = std::chrono::duration<double>(now - last_).count();
    last_ = now;
    for (auto &entry : phases_) {
      if (entry.first == phase) {
        entry.second += seconds;
        return;
      }
    }
    phases_.emplace_back(phase, seconds);
  }

  void Report() {
    std::vector<double> maximum{std::chrono::duration<double>(Clock::now() - start_).count()};
    for (const auto &entry : phases_) maximum.push_back(entry.second);
    std::vector<double> sum = maximum;
#if MPI_PARALLEL_ENABLED
    const int count = static_cast<int>(maximum.size());
    const bool root = global_variable::my_rank == 0;
    MPI_Reduce(root ? MPI_IN_PLACE : maximum.data(), maximum.data(), count, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(root ? MPI_IN_PLACE : sum.data(), sum.data(), count, MPI_DOUBLE, MPI_SUM,
               0, MPI_COMM_WORLD);
#endif
    if (global_variable::my_rank != 0) return;
    std::ostringstream line;
    line.precision(4);
    line << "PHASE_SECONDS scope=" << scope_ << " cycle=" << cycle_;
    for (std::size_t i = 0; i < maximum.size(); ++i) {
      line << ' ' << (i ? phases_[i - 1].first : std::string("total")) << '=' << maximum[i]
           << ',' << sum[i] / global_variable::nranks;
    }
    std::cout << line.str() << std::endl;
  }

 private:
  using Clock = std::chrono::steady_clock;
  std::string scope_;
  int cycle_;
  Clock::time_point start_, last_;
  std::vector<std::pair<std::string, double>> phases_;
};

#endif  // UTILS_PHASE_TIMER_HPP_
