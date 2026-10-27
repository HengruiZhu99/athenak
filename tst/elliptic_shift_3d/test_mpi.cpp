// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// MPI correctness oracle: distributed owned vectors, allgather halo reference,
// replicated global MG preconditioner. This is NOT a scalable production driver.
#include <mpi.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include "z4c/hyperboloidal/composite_shift.hpp"
#include "manufactured.hpp"

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int rank, ranks, status = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  try {
    namespace h = z4c::hyperboloidal;
    using V = multigrid::Vector;
    const auto start = std::chrono::steady_clock::now();
    h::CompositeShiftStage stage(cubed_sphere::CompositeMesh(.25, {.65, 1}, 1, 6));
    const size_t points = stage.grid.active.size(), begin = points*rank/ranks, end = points*(rank+1)/ranks;
    if (ranks > 4) throw std::invalid_argument("bounded MPI reference uses at most four ranks");
    std::vector<int> counts(ranks), displacements(ranks);
    for (int r = 0; r < ranks; ++r) {
      displacements[r] = 3*(points*r/ranks);
      counts[r] = 3*(points*(r+1)/ranks)-displacements[r];
    }
    std::vector<h::MetricJet<Real>> metric;
    V source, exact;
    for (const auto &point : stage.grid.geometry) {
      const auto u = manufactured::State(point.x, true);
      metric.push_back(u.metric);
      Real rhs[3]; h::EllipticShiftSource(u, h::Geometry(u.metric), 0., rhs);
      for (int i = 0; i < 3; ++i) {source.push_back(rhs[i]); exact.push_back(u.beta.value[i]);}
    }
    stage.SetGeometry(metric);
    const h::CompositeShiftStage::Boundary boundary = [](const auto &x) {
      return std::array<Real, 3>{-x[0], -x[1], -x[2]};
    };
    V affine, zero(3*points, 0), rhs(3*(end-begin)), x(rhs.size(), 0);
    stage.Apply(zero, affine, boundary);
    for (size_t p = 0; p < rhs.size(); ++p) rhs[p] = -source[3*begin+p]-affine[3*begin+p];
    std::uint64_t gathers = 0, reductions = 0;
    const auto gather = [&](const V &local) {
      V global(3*points);
      MPI_Allgatherv(local.data(), local.size(), MPI_DOUBLE, global.data(),
          counts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
      ++gathers; return global;
    };
    const multigrid::Operator apply = [&](const V &local, V &result) {
      V global_result; stage.Apply(gather(local), global_result);
      result.assign(global_result.begin()+3*begin, global_result.begin()+3*end);
    };
    const multigrid::Operator precondition = [&](const V &local, V &result) {
      const auto global = gather(local);
      V full(3*points);
      for (int v = 0; v < 3; ++v) {
        V scalar(points);
        for (size_t p = 0; p < points; ++p) scalar[p] = global[3*p+v];
        const auto solved = stage.preconditioner.Cycle(scalar);
        for (size_t p = 0; p < points; ++p) full[3*p+v] = solved[p];
      }
      result.assign(full.begin()+3*begin, full.begin()+3*end);
    };
    Real volume = 0;
    for (Real value : stage.norm_volume) volume += value;
    const multigrid::InnerProduct dot = [&](const V &a, const V &b) {
      Real local = 0, global;
      for (size_t p = begin; p < end; ++p)
      for (int v = 0; v < 3; ++v) local += stage.norm_volume[p]*a[3*(p-begin)+v]*b[3*(p-begin)+v];
      MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); ++reductions;
      return global/volume;
    };
    const auto report = multigrid::FlexibleGMRES(apply, precondition, dot, rhs, x, 1e-9, 0., 120, 24);
    const auto solution = gather(x);
    V original; stage.Apply(solution, original, boundary);
    Real residual_max = 0, error2 = 0;
    for (size_t p = 0; p < points; ++p)
    for (int v = 0; v < 3; ++v) {
      residual_max = std::max(residual_max, std::abs(original[3*p+v]+source[3*p+v]));
      error2 += stage.norm_volume[p]*std::pow(solution[3*p+v]-exact[3*p+v], 2);
    }
    if (!report.converged || residual_max > 1e-7) status = 1;
    if (rank == 0) std::cout << std::setprecision(17)
        << "{\"ranks\":" << ranks << ",\"points\":" << points << ",\"shells\":2,\"n\":6,"
        << "\"iterations\":" << report.iterations << ",\"original_residual\":" << report.residual.back()
        << ",\"residual_max\":" << residual_max << ",\"solution_error\":" << std::sqrt(error2/volume)
        << ",\"allgathers_per_rank\":" << gathers << ",\"allreductions_per_rank\":" << reductions
        << ",\"seconds\":" << std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()
        << ",\"backend\":\"distributed_vectors_replicated_MG_correctness_oracle\",\"converged\":"
        << report.converged << "}\n";
  } catch (const std::exception &error) {std::cerr << "rank " << rank << ": " << error.what() << '\n'; status = 2;}
  int global_status;
  MPI_Allreduce(&status, &global_status, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  Kokkos::finalize(); MPI_Finalize();
  return global_status;
}
