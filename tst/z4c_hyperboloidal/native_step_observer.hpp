// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Test-only observations of the original Driver, including its normal restart
// and output paths. No physics source or existing task dependency is changed.
#ifndef TST_Z4C_HYPERBOLOIDAL_NATIVE_STEP_OBSERVER_HPP_
#define TST_Z4C_HYPERBOLOIDAL_NATIVE_STEP_OBSERVER_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <stdexcept>

#include "athena.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#else
#error "The native stage observer requires an MPI build."
#endif
#include "globals.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "z4c_hyperboloidal/z4c_hyperboloidal.hpp"

namespace native_step_observer {
static_assert(sizeof(Real) == sizeof(double), "stage observer requires float64");

inline void RequireAll(bool condition, const char *message) {
  int failed = condition ? 0 : 1;
  MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (failed) throw std::runtime_error(message);
}

class Observer {
 public:
  Observer(Mesh *mesh, Driver *driver)
      : mesh_(mesh), initial_cycle_(mesh->ncycle), initial_time_(mesh->time),
        nominal_dt_(mesh->dt) {
    RequireAll(mesh->pmb_pack->pz4c_hyperboloidal != nullptr &&
                   driver->UsesClassicalRK4(),
               "stage observer requires native hyperboloidal classical RK4");
    auto &module = *mesh->pmb_pack->pz4c_hyperboloidal;
    RequireAll(nominal_dt_ == module.requested_dt && nominal_dt_ > 0,
               "stage observer requires the prescribed fixed timestep");
    bool opened = true;
    if (global_variable::my_rank == 0) {
      opened = !std::filesystem::exists("native-stages.csv") &&
               !std::filesystem::exists("native-step-observer.json");
      if (opened) {
        stream_.open("native-stages.csv");
        stream_ << "step,stage,rhs_time,state_time,dt,compatibility,clock,CFL\n"
                << std::setprecision(17);
        opened = stream_.good();
      }
    }
    RequireAll(opened, "stage observer refuses existing or unwritable output");
  }

  TaskStatus AfterStage(Driver *driver, int stage) {
    auto &module = *mesh_->pmb_pack->pz4c_hyperboloidal;
    const Real rhs_fraction[4] = {0, .5, .5, 1};
    const Real state_fraction[4] = {.5, .5, 1, 1};
    RequireAll(stage >= 1 && stage <= 4 &&
                   stage == int(rows_ % 4) + 1 &&
                   mesh_->ncycle == initial_cycle_ + int(rows_ / 4) &&
                   mesh_->dt == nominal_dt_,
               "stage observer encountered changed order, cycle or timestep");
    RequireAll(driver->stage_time == mesh_->time + nominal_dt_ * rhs_fraction[stage-1] &&
                   driver->stage_state_time ==
                       mesh_->time + nominal_dt_ * state_fraction[stage-1],
               "stage observer encountered changed RK stage clocks");
    const auto state_version = module.StateVersion();
    const DvceArray1D<const Real> cfl = module.rhs_cfl;
    int invalid = 0;
    Real maximum_cfl = 0;
    // rhs_cfl contains the last, just-consumed RHS at every owned point.
    // ExpRKUpdate and ProjectSurface do not overwrite this array. The pinned
    // predecessor's CFL is the bulk maximum, without an extra cfl_number factor.
    Kokkos::parallel_reduce("test observer owned stage CFL",
        Kokkos::RangePolicy<>(0, module.npoints),
        KOKKOS_LAMBDA(int point, int &bad, Real &maximum) {
          const Real value = cfl(point);
          if (!Kokkos::isfinite(value) || value < 0) ++bad;
          maximum = fmax(maximum, value);
        }, Kokkos::Sum<int>(invalid), Kokkos::Max<Real>(maximum_cfl));
    Real values[3] = {module.StageMaxCompatibility(), module.StageMaxClock(),
                     module.npoints ? maximum_cfl : Real(0)};
    for (Real value : values) invalid += !std::isfinite(value) || value < 0;
    RequireAll(invalid == 0 && state_version == module.StateVersion(),
               "invalid or state-changing stage observation");
    MPI_Allreduce(MPI_IN_PLACE, values, 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    RequireAll(values[0] <= 1e-7, "observed normal compatibility exceeds the existing gate");
    if (global_variable::my_rank == 0) {
      stream_ << mesh_->ncycle + 1 << ',' << stage << ',' << driver->stage_time
              << ',' << driver->stage_state_time << ',' << nominal_dt_
              << ',' << values[0] << ',' << values[1] << ',' << values[2] << '\n';
      if (stage == 4) stream_.flush();
    }
    ++rows_;
    if (stage == 4) {
      RequireAll(global_variable::my_rank != 0 || stream_.good(),
                 "stage observer failed to write a complete step");
    }
    return TaskStatus::complete;
  }

  void Finish() {
    const auto steps = mesh_->ncycle - initial_cycle_;
    RequireAll(!finished_ && steps >= 0 && rows_ == std::uint64_t(steps) * 4 &&
                   mesh_->dt == nominal_dt_ &&
                   (!steps || mesh_->dt_last_completed == nominal_dt_),
               "stage observer missed or repeated an RK stage");
    bool written = true;
    if (global_variable::my_rank == 0) {
      stream_.close();
      std::ofstream report("native-step-observer.json");
      report << std::setprecision(17)
             << "{\"schema\":\"native-step-observer-v1\",\"passed\":true,"
             << "\"backend\":\"" << DevExeSpace::name()
             << "\",\"ranks\":" << global_variable::nranks
             << ",\"initial_cycle\":" << initial_cycle_
             << ",\"final_cycle\":" << mesh_->ncycle
             << ",\"initial_time\":" << initial_time_
             << ",\"final_time\":" << mesh_->time
             << ",\"dt\":" << nominal_dt_ << ",\"stage_rows\":" << rows_
             << ",\"scope\":\"read-only stage observations; reference comparison separate\"}\n";
      report.close();
      written = stream_.good() && report.good();
    }
    RequireAll(written, "stage observer completion record write failed");
    finished_ = true;
  }

 private:
  Mesh *mesh_;
  int initial_cycle_;
  Real initial_time_, nominal_dt_;
  std::uint64_t rows_ = 0;
  bool finished_ = false;
  std::ofstream stream_;
};

inline std::shared_ptr<Observer> Install(Mesh *mesh, Driver *driver) {
  auto observer = std::make_shared<Observer>(mesh, driver);
  auto after = mesh->pmb_pack->tl_map.at("after_stagen");
  TaskID dependencies(0);
  for (int task = 1; task <= after->Size(); ++task)
    dependencies = dependencies | TaskID(task);
  after->AddTask([observer](Driver *d, int stage) {
    return observer->AfterStage(d, stage);
  }, dependencies);
  return observer;
}
}  // namespace native_step_observer
#endif  // TST_Z4C_HYPERBOLOIDAL_NATIVE_STEP_OBSERVER_HPP_
