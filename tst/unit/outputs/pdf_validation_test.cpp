//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file pdf_validation_test.cpp
//! \brief Boundary and checked-arithmetic tests for PDF preallocation.

#include <climits>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include "outputs/pdf_validation.hpp"

namespace {

pdf::ValidationInput Valid(const int nbin = 8) {
  pdf::ValidationInput input;
  input.block_name = "output7";
  input.has_nbin = true;
  input.has_bin_min = true;
  input.has_bin_max = true;
  input.nbin = nbin;
  input.bin_min = 0.125;
  input.bin_max = 8.0;
  input.logscale = true;
  return input;
}

void AddSecond(pdf::ValidationInput *input, const int nbin2 = 8) {
  input->variable_2_specified = true;
  input->has_variable_2 = true;
  input->has_nbin2 = true;
  input->has_bin2_min = true;
  input->has_bin2_max = true;
  input->has_any_second_axis_key = true;
  input->nbin2 = nbin2;
  input->bin2_min = 0.25;
  input->bin2_max = 4.0;
  input->logscale2 = true;
}

bool Rejects(const pdf::ValidationInput &input, const std::string &needle,
             const bool reject_mass = false) {
  const auto result = pdf::Validate(input, sizeof(double), reject_mass);
  return !result.valid && result.error.find(needle) != std::string::npos;
}

bool CheckCountsAndOwnership() {
  for (const int count : {1, 4094}) {
    auto one = pdf::Validate(Valid(count), sizeof(double), false);
    if (!one.valid || one.dimension != 1 || one.result_extent2 != 1 ||
        one.bins_extent != static_cast<std::size_t>(count + 1) ||
        one.result_extent1 != static_cast<std::size_t>(count + 2)) return false;
    auto two_input = Valid(count);
    AddSecond(&two_input, count);
    auto two = pdf::Validate(two_input, sizeof(double), false);
    if (!two.valid || two.dimension != 2 ||
        two.bins2_extent != static_cast<std::size_t>(count + 1) ||
        two.result_extent2 != static_cast<std::size_t>(count + 2)) return false;
  }
  auto maximum = Valid(4094);
  AddSecond(&maximum, 4094);
  const auto plan = pdf::Validate(maximum, sizeof(double), false);
  return plan.valid && plan.result_elements == 16777216ULL &&
         plan.persistent_elements == 16785406ULL &&
         plan.persistent_bytes == 134283248ULL &&
         plan.mirror_peak_elements == 33562622ULL &&
         plan.mirror_peak_bytes == 268500976ULL &&
         plan.persistent_bytes < pdf::kPersistentByteCap &&
         plan.mirror_peak_bytes < pdf::kMirrorPeakByteCap;
}

bool CheckInvalidCounts() {
  for (const int count : {-1, 0, 4095, INT_MAX - 1, INT_MAX}) {
    if (!Rejects(Valid(count), "[1,4094]")) return false;
    auto second = Valid();
    AddSecond(&second, count);
    if (!Rejects(second, "nbin2 in [1,4094]")) return false;
  }
  auto missing = Valid();
  missing.has_nbin = false;
  if (!Rejects(missing, "requires nbin")) return false;
  missing = Valid();
  missing.has_bin_min = false;
  if (!Rejects(missing, "requires bin_min")) return false;
  missing = Valid();
  missing.has_bin_max = false;
  if (!Rejects(missing, "requires bin_max")) return false;
  auto second_missing = Valid();
  second_missing.variable_2_specified = true;
  second_missing.has_variable_2 = true;
  if (!Rejects(second_missing, "explicit nbin2")) return false;
  second_missing = Valid();
  AddSecond(&second_missing);
  second_missing.has_bin2_min = false;
  if (!Rejects(second_missing, "explicit bin2_min")) return false;
  second_missing = Valid();
  AddSecond(&second_missing);
  second_missing.has_bin2_max = false;
  if (!Rejects(second_missing, "explicit bin2_max")) return false;
  auto empty_second = Valid();
  empty_second.variable_2_specified = true;
  if (!Rejects(empty_second, "non-empty")) return false;
  return true;
}

bool CheckBounds() {
  const double specials[] = {std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::infinity(),
                             -std::numeric_limits<double>::infinity()};
  for (const double value : specials) {
    auto input = Valid();
    input.bin_min = value;
    if (!Rejects(input, "finite")) return false;
    input = Valid();
    input.bin_max = value;
    if (!Rejects(input, "finite")) return false;
    input = Valid();
    AddSecond(&input);
    input.bin2_min = value;
    if (!Rejects(input, "finite")) return false;
    input = Valid();
    AddSecond(&input);
    input.bin2_max = value;
    if (!Rejects(input, "finite")) return false;
  }
  auto input = Valid();
  input.bin_max = input.bin_min;
  if (!Rejects(input, "bin_min < bin_max")) return false;
  input = Valid();
  input.bin_max = input.bin_min / 2.0;
  if (!Rejects(input, "bin_min < bin_max")) return false;
  input = Valid();
  AddSecond(&input);
  input.bin2_max = input.bin2_min;
  if (!Rejects(input, "bin_min < bin_max")) return false;
  input = Valid();
  AddSecond(&input);
  input.bin2_max = input.bin2_min / 2.0;
  if (!Rejects(input, "bin_min < bin_max")) return false;
  input = Valid();
  input.bin_min = 0.0;
  if (!Rejects(input, "positive")) return false;
  input = Valid();
  input.bin_min = -0.125;
  if (!Rejects(input, "positive")) return false;
  input = Valid();
  AddSecond(&input);
  input.bin2_min = 0.0;
  if (!Rejects(input, "positive")) return false;
  input = Valid();
  AddSecond(&input);
  input.bin2_min = -0.25;
  if (!Rejects(input, "positive")) return false;

  // Finite ordered linear bounds can still overflow their subtraction.
  input = Valid();
  input.logscale = false;
  input.bin_min = -std::numeric_limits<double>::max();
  input.bin_max = std::numeric_limits<double>::max();
  if (!Rejects(input, "step must be finite and positive")) return false;
  input = Valid();
  AddSecond(&input);
  input.logscale2 = false;
  input.bin2_min = -std::numeric_limits<double>::max();
  input.bin2_max = std::numeric_limits<double>::max();
  if (!Rejects(input, "step must be finite and positive")) return false;

  // Distinct finite logarithmic bounds can map to the same rounded logarithm.
  const double largest = std::numeric_limits<double>::max();
  const double below_largest = std::nextafter(largest, 0.0);
  input = Valid();
  input.bin_min = below_largest;
  input.bin_max = largest;
  if (!Rejects(input, "step must be finite and positive")) return false;
  input = Valid();
  AddSecond(&input);
  input.bin2_min = below_largest;
  input.bin2_max = largest;
  if (!Rejects(input, "step must be finite and positive")) return false;

  // A finite positive linear span can underflow to a zero step after division.
  input = Valid();
  input.logscale = false;
  input.bin_min = 0.0;
  input.bin_max = std::numeric_limits<double>::denorm_min();
  if (!Rejects(input, "step must be finite and positive")) return false;
  input = Valid();
  AddSecond(&input);
  input.logscale2 = false;
  input.bin2_min = 0.0;
  input.bin2_max = std::numeric_limits<double>::denorm_min();
  if (!Rejects(input, "step must be finite and positive")) return false;
  return true;
}

bool CheckPredicatesAndArithmetic() {
  auto input = Valid();
  input.has_nbin2 = true;
  input.nbin2 = 0;
  input.has_any_second_axis_key = true;
  if (!Rejects(input, "require variable_2")) return false;
  input = Valid();
  AddSecond(&input, 1);
  if (!pdf::Validate(input, sizeof(double), false).valid) return false;
  input.mass_weighted = true;
  if (!pdf::Validate(input, sizeof(double), false).valid ||
      !Rejects(input, "mass_weighted=true", true)) return false;

  std::size_t value = 0;
  if (pdf::CheckedAdd(std::numeric_limits<std::size_t>::max(), 1, &value)) return false;
  if (pdf::CheckedMultiply(std::numeric_limits<std::size_t>::max(), 2, &value)) return false;
  if (pdf::Validate(Valid(), 0, false).valid) return false;
  if (pdf::Validate(Valid(), std::numeric_limits<std::size_t>::max(), false).valid) {
    return false;
  }
  return true;
}

}  // namespace

int main() {
  if (!(CheckCountsAndOwnership() && CheckInvalidCounts() && CheckBounds() &&
        CheckPredicatesAndArithmetic())) return EXIT_FAILURE;
  std::cout << "PDF validation and ownership tests passed\n";
  return EXIT_SUCCESS;
}
