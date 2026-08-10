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

#include "outputs/outputs.hpp"

namespace {

pdf::AllocationPlan SmallPlan() {
  pdf::ValidationInput input;
  input.block_name = "scatter_test";
  input.has_nbin = input.has_bin_min = input.has_bin_max = true;
  input.nbin = 4;
  input.bin_min = 0.0;
  input.bin_max = 4.0;
  input.logscale = false;
  input.variable_2_specified = true;
  input.has_variable_2 = true;
  input.has_nbin2 = input.has_bin2_min = input.has_bin2_max = true;
  input.has_any_second_axis_key = true;
  input.nbin2 = 3;
  input.bin2_min = 0.0;
  input.bin2_max = 3.0;
  input.logscale2 = false;
  return pdf::Validate(input, sizeof(Real), false);
}

bool RunRace() {
  using Expected = Kokkos::Experimental::ScatterView<
      Real **, LayoutWrapper, typename DvceArray2D<Real>::device_type,
      Kokkos::Experimental::ScatterSum,
      Kokkos::Experimental::ScatterNonDuplicated,
      Kokkos::Experimental::ScatterAtomic>;
  static_assert(std::is_same_v<PDFData::ScatterResult, Expected>);

  const Real expected_measure =
      6.283185307179586476925286766559 * 0.5 * 0.25 * 0.125;
  if (std::abs(pdf::CartoonCylindricalFineCellMeasure(0.5, 0.25, 0.125) -
               expected_measure) >
          16.0 * std::numeric_limits<Real>::epsilon() * expected_measure ||
      pdf::CartoonCylindricalFineCellMeasure(0.0, 0.25, 0.125) != 0.0 ||
      pdf::CartoonCylindricalFineCellMeasure(-0.5, 0.25, 0.125) != 0.0) {
    return false;
  }

  PDFData data(SmallPlan());
  if (data.scatter_result.subview().data() != data.result_.data() ||
      data.scatter_result.subview().span() != data.result_.span()) return false;

  constexpr int updates = 20000;
  for (int repetition = 0; repetition < 20; ++repetition) {
    Kokkos::deep_copy(data.result_, 0.0);
    auto scatter = data.scatter_result;
    Kokkos::parallel_for(
        "pdf scatter collision", Kokkos::RangePolicy<>(0, updates),
        KOKKOS_LAMBDA(const int index) {
          auto access = scatter.access();
          access(2, 3) += 1.0;
          access(index % 3, index % 4) += 1.0;
        });
    Kokkos::Experimental::contribute(data.result_, scatter);
    Kokkos::fence();
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data.result_);
    for (int j = 0; j < 5; ++j) {
      for (int i = 0; i < 6; ++i) {
        Real expected = (j == 2 && i == 3) ? updates : 0;
        for (int index = 0; index < updates; ++index) {
          if (j == index % 3 && i == index % 4) expected += 1.0;
        }
        if (host(j, i) != expected) return false;
      }
    }
  }
  return true;
}

}  // namespace

int main(int argc, char *argv[]) {
  bool require_cuda = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--require-cuda") require_cuda = true;
  }
#if !defined(KOKKOS_ENABLE_CUDA)
  if (require_cuda) {
    std::cerr << "CUDA PDF scatter qualification requires KOKKOS_ENABLE_CUDA\n";
    return EXIT_FAILURE;
  }
#endif
  Kokkos::initialize(argc, argv);
  const bool passed = RunRace();
  const std::string backend = Kokkos::DefaultExecutionSpace::name();
  Kokkos::finalize();
  if (!passed) return EXIT_FAILURE;
  std::cout << "PDF scatter race passed on " << backend << "\n";
  return EXIT_SUCCESS;
}
