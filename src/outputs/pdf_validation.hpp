//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file pdf_validation.hpp
//! \brief Allocation-free checked PDF axis and histogram ownership contract.

#ifndef OUTPUTS_PDF_VALIDATION_HPP_
#define OUTPUTS_PDF_VALIDATION_HPP_

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>

namespace pdf {

inline constexpr int kMaximumBinsPerAxis = 4094;
inline constexpr std::size_t kPersistentByteCap = 129ULL * 1024ULL * 1024ULL;
inline constexpr std::size_t kMirrorPeakByteCap = 257ULL * 1024ULL * 1024ULL;

struct ValidationInput {
  std::string block_name;
  bool variable_2_specified = false;
  bool has_variable_2 = false;
  bool has_nbin = false;
  bool has_bin_min = false;
  bool has_bin_max = false;
  int nbin = 0;
  double bin_min = 0.0;
  double bin_max = 0.0;
  bool logscale = true;
  bool has_nbin2 = false;
  bool has_bin2_min = false;
  bool has_bin2_max = false;
  int nbin2 = 0;
  double bin2_min = 0.0;
  double bin2_max = 0.0;
  bool logscale2 = true;
  bool has_any_second_axis_key = false;
  bool mass_weighted = false;
};

struct AllocationPlan {
  bool valid = false;
  std::string error;
  bool has_second_axis = false;
  int dimension = 1;
  int nbin = 0;
  int nbin2 = 0;
  std::size_t bins_extent = 0;
  std::size_t bins2_extent = 0;
  std::size_t result_extent1 = 0;
  std::size_t result_extent2 = 1;
  std::size_t result_elements = 0;
  std::size_t persistent_elements = 0;
  std::size_t persistent_bytes = 0;
  std::size_t mirror_peak_elements = 0;
  std::size_t mirror_peak_bytes = 0;
  double step_size = 0.0;
  double step_size2 = 0.0;
};

inline bool HasSecondAxis(const bool has_variable_2, const int nbin2) {
  return has_variable_2 && nbin2 > 0;
}

inline AllocationPlan Invalid(const ValidationInput &input,
                              const std::string &message) {
  AllocationPlan plan;
  plan.error = "<" + input.block_name + "> " + message;
  return plan;
}

inline bool CheckedAdd(const std::size_t left, const std::size_t right,
                       std::size_t *result) {
  if (left > std::numeric_limits<std::size_t>::max() - right) return false;
  *result = left + right;
  return true;
}

inline bool CheckedMultiply(const std::size_t left, const std::size_t right,
                            std::size_t *result) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) return false;
  *result = left * right;
  return true;
}

inline bool ValidateAxis(const char *axis, const int count, const double minimum,
                         const double maximum, const bool logarithmic,
                         double *step, std::string *error) {
  if (count < 1 || count > kMaximumBinsPerAxis) {
    *error = std::string(axis) + " nbin must be in [1,4094]";
    return false;
  }
  if (!std::isfinite(minimum) || !std::isfinite(maximum)) {
    *error = std::string(axis) + " bin bounds must be finite";
    return false;
  }
  if (!(minimum < maximum)) {
    *error = std::string(axis) + " requires bin_min < bin_max";
    return false;
  }
  if (logarithmic && !(minimum > 0.0)) {
    *error = std::string(axis) + " logarithmic bin_min must be positive";
    return false;
  }
  *step = logarithmic
              ? (std::log10(maximum) - std::log10(minimum)) / count
              : (maximum - minimum) / count;
  if (!std::isfinite(*step) || !(*step > 0.0)) {
    *error = std::string(axis) + " bin step must be finite and positive";
    return false;
  }
  return true;
}

//! Validate all arithmetic and memory ownership without constructing a Kokkos View.
inline AllocationPlan Validate(const ValidationInput &input,
                               const std::size_t real_bytes,
                               const bool reject_mass_weighting) {
  if (input.block_name.empty()) return Invalid(input, "PDF block name is empty");
  if (real_bytes == 0) return Invalid(input, "PDF Real element size must be positive");
  if (reject_mass_weighting && input.mass_weighted) {
    return Invalid(input, "PDF mass_weighted=true is unsupported for vacuum Z4c");
  }
  if (!input.has_nbin) return Invalid(input, "PDF requires nbin");
  if (!input.has_bin_min) return Invalid(input, "PDF requires bin_min");
  if (!input.has_bin_max) return Invalid(input, "PDF requires bin_max");

  if (input.variable_2_specified && !input.has_variable_2) {
    return Invalid(input, "PDF variable_2 must be non-empty when specified");
  }
  if (!input.has_variable_2 && input.has_any_second_axis_key) {
    return Invalid(input, "PDF second-axis keys require variable_2");
  }
  if (input.has_variable_2 && !input.has_nbin2) {
    return Invalid(input, "PDF variable_2 requires explicit nbin2");
  }
  if (input.has_variable_2 && input.has_nbin2 &&
      (input.nbin2 < 1 || input.nbin2 > kMaximumBinsPerAxis)) {
    return Invalid(input, "PDF variable_2 requires nbin2 in [1,4094]");
  }
  if (input.has_variable_2 && !input.has_bin2_min) {
    return Invalid(input, "PDF variable_2 requires explicit bin2_min");
  }
  if (input.has_variable_2 && !input.has_bin2_max) {
    return Invalid(input, "PDF variable_2 requires explicit bin2_max");
  }

  AllocationPlan plan;
  plan.nbin = input.nbin;
  plan.nbin2 = input.has_variable_2 ? input.nbin2 : 0;
  plan.has_second_axis = HasSecondAxis(input.has_variable_2, plan.nbin2);
  plan.dimension = plan.has_second_axis ? 2 : 1;
  std::string error;
  if (!ValidateAxis("primary axis", input.nbin, input.bin_min,
                    input.bin_max, input.logscale, &plan.step_size, &error)) {
    return Invalid(input, error);
  }
  if (input.has_variable_2 && !plan.has_second_axis) {
    return Invalid(input, "PDF variable_2 requires nbin2 in [1,4094]");
  }
  if (plan.has_second_axis &&
      !ValidateAxis("second axis", input.nbin2, input.bin2_min,
                    input.bin2_max, input.logscale2, &plan.step_size2, &error)) {
    return Invalid(input, error);
  }

  if (!CheckedAdd(static_cast<std::size_t>(plan.nbin), 1, &plan.bins_extent) ||
      !CheckedAdd(static_cast<std::size_t>(plan.nbin), 2, &plan.result_extent1)) {
    return Invalid(input, "primary axis extent arithmetic overflow");
  }
  if (plan.has_second_axis) {
    if (!CheckedAdd(static_cast<std::size_t>(plan.nbin2), 1,
                    &plan.bins2_extent) ||
        !CheckedAdd(static_cast<std::size_t>(plan.nbin2), 2,
                    &plan.result_extent2)) {
      return Invalid(input, "second axis extent arithmetic overflow");
    }
  }
  if (!CheckedMultiply(plan.result_extent1, plan.result_extent2,
                       &plan.result_elements) ||
      plan.result_elements > 4096ULL * 4096ULL) {
    return Invalid(input, "PDF result extent product exceeds checked limit");
  }
  std::size_t bin_elements = 0;
  if (!CheckedAdd(plan.bins_extent, plan.bins2_extent, &bin_elements) ||
      !CheckedAdd(bin_elements, plan.result_elements,
                  &plan.persistent_elements) ||
      !CheckedMultiply(plan.persistent_elements, real_bytes,
                       &plan.persistent_bytes)) {
    return Invalid(input, "PDF persistent ownership arithmetic overflow");
  }
  if (plan.persistent_bytes > kPersistentByteCap) {
    return Invalid(input, "PDF persistent Real payload exceeds 129 MiB");
  }
  if (!CheckedAdd(plan.persistent_elements, plan.result_elements,
                  &plan.mirror_peak_elements) ||
      !CheckedMultiply(plan.mirror_peak_elements, real_bytes,
                       &plan.mirror_peak_bytes)) {
    return Invalid(input, "PDF mirror-peak arithmetic overflow");
  }
  if (plan.mirror_peak_bytes > kMirrorPeakByteCap) {
    return Invalid(input, "PDF histogram plus mirror peak exceeds 257 MiB");
  }
  plan.valid = true;
  return plan;
}

}  // namespace pdf

#endif  // OUTPUTS_PDF_VALIDATION_HPP_
