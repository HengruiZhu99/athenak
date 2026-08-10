//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file pdf_scatter_test.cpp
//! \brief Explicit scatter ownership and repeated hot-bin race test.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>

#include <Kokkos_Core.hpp>

#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"
#include "outputs/pdf_accumulate.hpp"

namespace {

pdf::AllocationPlan SmallPlan(const bool has_second_axis) {
  pdf::ValidationInput input;
  input.block_name = "scatter_test";
  input.has_nbin = input.has_bin_min = input.has_bin_max = true;
  input.nbin = 4;
  input.bin_min = 0.0;
  input.bin_max = 4.0;
  input.logscale = false;
  if (has_second_axis) {
    input.variable_2_specified = true;
    input.has_variable_2 = true;
    input.has_nbin2 = input.has_bin2_min = input.has_bin2_max = true;
    input.has_any_second_axis_key = true;
    input.nbin2 = 3;
    input.bin2_min = 0.0;
    input.bin2_max = 3.0;
    input.logscale2 = false;
  }
  return pdf::Validate(input, sizeof(Real), false);
}

bool RunRace(const bool has_second_axis) {
  using Expected = Kokkos::Experimental::ScatterView<
      Real **, LayoutWrapper, typename DvceArray2D<Real>::device_type,
      Kokkos::Experimental::ScatterSum,
      Kokkos::Experimental::ScatterNonDuplicated,
      Kokkos::Experimental::ScatterAtomic>;
  static_assert(std::is_same_v<PDFData::ScatterResult, Expected>);

  PDFData data(SmallPlan(has_second_axis));
  if (data.scatter_result.subview().data() != data.result_.data() ||
      data.scatter_result.subview().span() != data.result_.span() ||
      data.pdf_dimension != (has_second_axis ? 2 : 1) ||
      data.result_.extent(0) != (has_second_axis ? 5U : 1U) ||
      data.result_.extent(1) != 6U ||
      data.bins2.extent(0) != (has_second_axis ? 4U : 0U)) return false;

  constexpr int updates = 20000;
  const int hot_row = has_second_axis ? 2 : 0;
  const int result_rows = has_second_axis ? 5 : 1;
  for (int repetition = 0; repetition < 20; ++repetition) {
    Kokkos::deep_copy(data.result_, 0.0);
    auto scatter = data.scatter_result;
    const bool use_second_axis = has_second_axis;
    Kokkos::parallel_for(
        "pdf scatter collision", Kokkos::RangePolicy<>(0, updates),
        KOKKOS_LAMBDA(const int index) {
          auto access = scatter.access();
          const int mixed_row = use_second_axis ? index % 3 : 0;
          access(hot_row, 3) += 1.0;
          access(mixed_row, index % 4) += 1.0;
        });
    Kokkos::Experimental::contribute(data.result_, scatter);
    Kokkos::fence();
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data.result_);
    for (int j = 0; j < result_rows; ++j) {
      for (int i = 0; i < 6; ++i) {
        Real expected = (j == hot_row && i == 3) ? updates : 0;
        for (int index = 0; index < updates; ++index) {
          const int mixed_row = has_second_axis ? index % 3 : 0;
          if (j == mixed_row && i == index % 4) expected += 1.0;
        }
        if (host(j, i) != expected) return false;
      }
    }
  }
  return true;
}

bool RunCylindricalHistogramOracle() {
  PDFData data(SmallPlan(false));
  if (data.pdf_dimension != 1 || data.result_.extent(0) != 1 ||
      data.scatter_result.subview().data() != data.result_.data()) return false;

  // This signed-rho set is a nonoverlapping mixed-resolution leaf tiling:
  // one coarse radial cell covers [0,1] and two fine cells cover [1,2],
  // with their negative mirrors and an exact-axis sample.  No parent overlaps
  // the fine leaves.  The independent analytic volume for dz=1/2 is 2*pi.
  constexpr int samples = 7;
  const Real host_rho[samples] = {-1.75, -1.25, -0.5, 0.0, 0.5, 1.25, 1.75};
  const Real host_dx1[samples] = {0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 0.5};
  const int host_bin[samples] = {2, 2, 1, 1, 1, 2, 2};
  Kokkos::View<Real*> rho("signed rho", samples);
  Kokkos::View<Real*> dx1("leaf radial width", samples);
  Kokkos::View<int*> bin("cylindrical histogram bin", samples);
  auto rho_host = Kokkos::create_mirror_view(rho);
  auto dx1_host = Kokkos::create_mirror_view(dx1);
  auto bin_host = Kokkos::create_mirror_view(bin);
  for (int n = 0; n < samples; ++n) {
    rho_host(n) = host_rho[n];
    dx1_host(n) = host_dx1[n];
    bin_host(n) = host_bin[n];
  }
  Kokkos::deep_copy(rho, rho_host);
  Kokkos::deep_copy(dx1, dx1_host);
  Kokkos::deep_copy(bin, bin_host);

  Kokkos::deep_copy(data.result_, 0.0);
  auto scatter = data.scatter_result;
  Kokkos::parallel_for(
      "signed rho cylindrical histogram", Kokkos::RangePolicy<>(0, samples),
      KOKKOS_LAMBDA(const int n) {
        const Real weight =
            pdf::CartoonCylindricalFineCellMeasure(rho(n), dx1(n), 0.5);
        if (weight > 0.0) scatter.access()(0, bin(n)) += weight;
      });
  Kokkos::Experimental::contribute(data.result_, scatter);
  Kokkos::fence();

  auto result =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data.result_);
  constexpr Real oracle_pi = 3.1415926535897932384626433832795;
  const Real expected_coarse = 0.5 * oracle_pi;
  const Real expected_fine = 1.5 * oracle_pi;
  const Real expected_total = 2.0 * oracle_pi;
  const Real tolerance =
      128.0 * std::numeric_limits<Real>::epsilon() * expected_total;
  Real total = 0.0;
  for (int i = 0; i < 6; ++i) total += result(0, i);
  return std::abs(result(0, 1) - expected_coarse) <= tolerance &&
         std::abs(result(0, 2) - expected_fine) <= tolerance &&
         std::abs(total - expected_total) <= tolerance &&
         result(0, 0) == 0.0 && result(0, 3) == 0.0 &&
         result(0, 4) == 0.0 && result(0, 5) == 0.0;
}

bool RunProductionAccumulator(const bool cartoon, const bool has_second_axis) {
  PDFData data(SmallPlan(has_second_axis));
  constexpr int cells = 8;
  const int dimension = has_second_axis ? 2 : 1;
  DvceArray5D<Real> outvars(
      "production pdf staged variables", dimension, 1, 1, 1, cells);
  DvceArray5D<Real> state("production pdf state", 1, 1, 1, 1, cells);
  Kokkos::View<RegionSize*> size("production pdf block size", 1);
  auto outvars_host = Kokkos::create_mirror_view(outvars);
  auto state_host = Kokkos::create_mirror_view(state);
  auto size_host = Kokkos::create_mirror_view(size);
  auto bins_host = Kokkos::create_mirror_view(data.bins);
  auto bins2_host = Kokkos::create_mirror_view(data.bins2);
  for (int i = 0; i < cells; ++i) {
    outvars_host(0, 0, 0, 0, i) = 1.5;
    if (has_second_axis) outvars_host(1, 0, 0, 0, i) = 1.5;
    state_host(0, 0, 0, 0, i) = 1.0;
  }
  size_host(0) = {-2.0, -0.5, -0.5, 2.0, 0.5, 0.5, 0.5, 0.5, 1.0};
  for (int i = 0; i <= data.nbin; ++i) bins_host(i) = static_cast<Real>(i);
  if (has_second_axis) {
    for (int i = 0; i <= data.nbin2; ++i) bins2_host(i) = static_cast<Real>(i);
  }
  Kokkos::deep_copy(outvars, outvars_host);
  Kokkos::deep_copy(state, state_host);
  Kokkos::deep_copy(size, size_host);
  Kokkos::deep_copy(data.bins, bins_host);
  if (has_second_axis) Kokkos::deep_copy(data.bins2, bins2_host);
  Kokkos::deep_copy(data.result_, 0.0);

  if (cartoon) {
    const pdf::CartoonPdfCellMeasure<decltype(size)> measure{size, cells, 0};
    pdf::AccumulatePdfHistogram(
        measure, outvars, state, data.bins, data.bins2, data.scatter_result,
        1, 0, 0, 0, 0, 0, cells - 1, data.nbin, data.nbin2,
        data.pdf_dimension, data.step_size, data.step_size2,
        data.logscale, data.logscale2, data.mass_weighted);
  } else {
    const pdf::CartesianPdfCellMeasure<decltype(size)> measure{size};
    pdf::AccumulatePdfHistogram(
        measure, outvars, state, data.bins, data.bins2, data.scatter_result,
        1, 0, 0, 0, 0, 0, cells - 1, data.nbin, data.nbin2,
        data.pdf_dimension, data.step_size, data.step_size2,
        data.logscale, data.logscale2, data.mass_weighted);
  }
  Kokkos::Experimental::contribute(data.result_, data.scatter_result);
  Kokkos::fence();
  auto result =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data.result_);
  constexpr Real pi = 3.1415926535897932384626433832795;
  const Real expected = cartoon ? 2.0 * pi : 2.0;
  const Real tolerance = 128.0 * std::numeric_limits<Real>::epsilon() * expected;
  for (int row = 0; row < static_cast<int>(result.extent(0)); ++row) {
    for (int column = 0; column < static_cast<int>(result.extent(1)); ++column) {
      const Real oracle = (row == (has_second_axis ? 2 : 0) && column == 2)
                              ? expected : 0.0;
      if (std::abs(result(row, column) - oracle) > tolerance) return false;
    }
  }
  return true;
}

bool MeetsCudaRequirement(const bool require_cuda) {
  if (!require_cuda) return true;
#if defined(KOKKOS_ENABLE_CUDA)
  return std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::Cuda> &&
         std::string(Kokkos::DefaultExecutionSpace::name()) == "Cuda";
#else
  return false;
#endif
}

}  // namespace

int main(int argc, char *argv[]) {
  bool require_cuda = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--require-cuda") require_cuda = true;
  }
  Kokkos::initialize(argc, argv);
  const bool cuda_requirement_met = MeetsCudaRequirement(require_cuda);
  const bool passed = cuda_requirement_met && RunRace(false) && RunRace(true) &&
                      RunCylindricalHistogramOracle() &&
                      RunProductionAccumulator(false, false) &&
                      RunProductionAccumulator(false, true) &&
                      RunProductionAccumulator(true, false) &&
                      RunProductionAccumulator(true, true);
  const std::string backend = Kokkos::DefaultExecutionSpace::name();
  Kokkos::finalize();
  if (!cuda_requirement_met) {
    std::cerr << "CUDA PDF scatter qualification requires actual Kokkos "
              << "DefaultExecutionSpace=Cuda, got " << backend << "\n";
  }
  if (!passed) return EXIT_FAILURE;
  std::cout << "PDF production Cartesian/Cartoon 1-D/2-D accumulation, scatter races, "
            << "and cylindrical oracle passed on " << backend << "\n";
  return EXIT_SUCCESS;
}
