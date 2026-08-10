//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cartoon_derivatives_test_common.hpp
//! \brief Shared manufactured-field checks for the analytic SO(2) provider tests.

#pragma once

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "athena.hpp"
#include "z4c/cartoon_derivatives.hpp"

namespace {

constexpr int kRho = 0;
constexpr int kZ = 1;
constexpr int kSuppressed = 2;
constexpr int kScalarOffset = 0;
constexpr int kVectorOffset = 1;
constexpr int kTensorOffset = 4;
constexpr int kNumVariables = 10;
constexpr int kNumResults = 75;

struct Jet3 {
  double value = 0.0;
  double first[3] = {0.0, 0.0, 0.0};
  double second[3][3] = {{0.0, 0.0, 0.0},
                         {0.0, 0.0, 0.0},
                         {0.0, 0.0, 0.0}};
};

Jet3 Coordinate(const double value, const int direction) {
  Jet3 result;
  result.value = value;
  result.first[direction] = 1.0;
  return result;
}

Jet3 Constant(const double value) {
  Jet3 result;
  result.value = value;
  return result;
}

Jet3 operator+(const Jet3 &left, const Jet3 &right) {
  Jet3 result;
  result.value = left.value + right.value;
  for (int a = 0; a < 3; ++a) {
    result.first[a] = left.first[a] + right.first[a];
    for (int b = 0; b < 3; ++b) {
      result.second[a][b] = left.second[a][b] + right.second[a][b];
    }
  }
  return result;
}

Jet3 operator-(const Jet3 &left, const Jet3 &right) {
  Jet3 result;
  result.value = left.value - right.value;
  for (int a = 0; a < 3; ++a) {
    result.first[a] = left.first[a] - right.first[a];
    for (int b = 0; b < 3; ++b) {
      result.second[a][b] = left.second[a][b] - right.second[a][b];
    }
  }
  return result;
}

Jet3 operator*(const Jet3 &left, const Jet3 &right) {
  Jet3 result;
  result.value = left.value * right.value;
  for (int a = 0; a < 3; ++a) {
    result.first[a] = left.first[a] * right.value + left.value * right.first[a];
    for (int b = 0; b < 3; ++b) {
      result.second[a][b] = left.second[a][b] * right.value +
                            left.first[a] * right.first[b] +
                            left.first[b] * right.first[a] +
                            left.value * right.second[a][b];
    }
  }
  return result;
}

Jet3 operator*(const double coefficient, const Jet3 &value) {
  return Constant(coefficient) * value;
}

struct ManufacturedFields {
  Jet3 scalar;
  std::array<Jet3, 3> vector;
  std::array<std::array<Jet3, 3>, 3> tensor;
};

ManufacturedFields EvaluateManufacturedFields(const double rho_value,
                                               const double z_value,
                                               const double y_value) {
  const Jet3 x = Coordinate(rho_value, kRho);
  const Jet3 z = Coordinate(z_value, kZ);
  const Jet3 y = Coordinate(y_value, kSuppressed);
  const Jet3 s = x * x + y * y;
  const Jet3 s2 = s * s;
  const Jet3 s3 = s2 * s;
  const Jet3 s4 = s2 * s2;

  // These coefficient functions are deliberately unrelated to the production provider.
  // The odd vector/tensor sectors contain s, s^2, and s^3 terms so first-raw
  // quotients expose their generic transition orders 1, 3, and 5 respectively.
  const Jet3 f = Constant(0.7) + 0.23 * s + 0.11 * s2 + 0.017 * s4 +
                 0.31 * z + 0.09 * z * z + 0.07 * s * z;
  const Jet3 a = Constant(0.4) + 0.13 * s + 0.031 * s2 + 0.006 * s3 +
                 0.021 * s4 + 0.05 * z;
  const Jet3 b = Constant(-0.2) + 0.08 * s - 0.027 * s2 + 0.005 * s3 +
                 0.014 * s4 - 0.03 * z;
  const Jet3 c = Constant(0.6) + 0.19 * s + 0.012 * s4 + 0.04 * z * z;
  const Jet3 p = Constant(1.1) + 0.12 * s + 0.016 * s4 + 0.02 * z;
  const Jet3 q = Constant(0.9) - 0.07 * s + 0.011 * s4 + 0.03 * z * z;
  const Jet3 r = Constant(0.3) + 0.09 * s + 0.023 * s2 - 0.004 * s3 +
                 0.013 * s4 + 0.04 * z;
  const Jet3 u = Constant(-0.17) + 0.06 * s - 0.019 * s2 + 0.003 * s3 +
                 0.009 * s4 - 0.02 * z;
  const Jet3 v = Constant(0.22) + 0.05 * s + 0.008 * s4 + 0.01 * z;
  const Jet3 w = Constant(-0.14) + 0.04 * s + 0.007 * s4 - 0.015 * z;

  const std::array<Jet3, 3> radial = {x, Constant(0.0), y};
  const std::array<Jet3, 3> azimuthal = {Constant(0.0) - y, Constant(0.0), x};
  const std::array<Jet3, 3> axial = {Constant(0.0), Constant(1.0), Constant(0.0)};

  ManufacturedFields fields;
  fields.scalar = f;
  for (int component = 0; component < 3; ++component) {
    fields.vector[component] = a * radial[component] + b * azimuthal[component] +
                               c * axial[component];
  }
  for (int first = 0; first < 3; ++first) {
    for (int second = 0; second < 3; ++second) {
      fields.tensor[first][second] =
          p * ((first == kRho && second == kRho) ||
                       (first == kSuppressed && second == kSuppressed)
                   ? Constant(1.0)
                   : Constant(0.0)) +
          q * axial[first] * axial[second] +
          r * (radial[first] * axial[second] + axial[first] * radial[second]) +
          u * (azimuthal[first] * axial[second] + axial[first] * azimuthal[second]) +
          v * (radial[first] * radial[second] -
               azimuthal[first] * azimuthal[second]) +
          w * (radial[first] * azimuthal[second] +
               azimuthal[first] * radial[second]);
    }
  }
  return fields;
}

KOKKOS_INLINE_FUNCTION
int SymmetricIndex(int first, int second) {
  if (first > second) {
    const int swap = first;
    first = second;
    second = swap;
  }
  return first * 3 - first * (first - 1) / 2 + second - first;
}

struct ScalarField {
  DvceArray5D<Real> data;
  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(const int m, const int k, const int j, const int i) const {
    return data(m, kScalarOffset, k, j, i);
  }
};

struct VectorField {
  DvceArray5D<Real> data;
  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(const int m, const int component, const int k,
                            const int j, const int i) const {
    return data(m, kVectorOffset + component, k, j, i);
  }
};

struct TensorField {
  DvceArray5D<Real> data;
  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(const int m, const int first, const int second,
                            const int k, const int j, const int i) const {
    return data(m, kTensorOffset + SymmetricIndex(first, second), k, j, i);
  }
};

template <int NGHOST>
double OracleDissipation(const double h, const double rho, const double z) {
  constexpr int coefficients2[5] = {1, -4, 6, -4, 1};
  constexpr int coefficients3[7] = {1, -6, 15, -20, 15, -6, 1};
  constexpr int coefficients4[9] = {1, -8, 28, -56, 70, -56, 28, -8, 1};
  double result = 0.0;
  for (int direction = kRho; direction <= kZ; ++direction) {
    for (int offset = -NGHOST; offset <= NGHOST; ++offset) {
      int coefficient = 0;
      if constexpr (NGHOST == 2) coefficient = coefficients2[offset + NGHOST];
      if constexpr (NGHOST == 3) coefficient = coefficients3[offset + NGHOST];
      if constexpr (NGHOST == 4) coefficient = coefficients4[offset + NGHOST];
      const double shifted_rho = rho + (direction == kRho ? offset * h : 0.0);
      const double shifted_z = z + (direction == kZ ? offset * h : 0.0);
      result += coefficient *
                EvaluateManufacturedFields(shifted_rho, shifted_z, 0.0).scalar.value / h;
    }
  }
  return result;
}

struct ErrorSummary {
  double derivative = 0.0;
  double dissipation = 0.0;
  double family[3] = {0.0, 0.0, 0.0};
  int family_worst_result[3] = {-1, -1, -1};
  int worst_result = -1;
  double worst_expected = 0.0;
  double worst_observed = 0.0;
  std::array<double, kNumResults> observed{};
  std::array<double, kNumResults> oracle{};
};

bool IsIsotropySensitiveResult(const int result) {
  if (result < 32 || result >= 74) return false;
  const int tensor_offset = result - 32;
  const int component = tensor_offset / 7;
  const int operation = tensor_offset % 7;
  const bool radial_plane_component =
      component == 0 || component == 2 || component == 5;
  return radial_plane_component && (operation <= 3 || operation == 6);
}

template <int NGHOST>
std::array<int, 2 * (NGHOST + 1)> SignedNearAxisOffsets() {
  std::array<int, 2 * (NGHOST + 1)> offsets{};
  int output = 0;
  for (int layer = 0; layer <= NGHOST; ++layer) offsets[output++] = layer;
  for (int layer = 0; layer <= NGHOST; ++layer) offsets[output++] = -layer - 1;
  return offsets;
}

int LayerFromOffset(const int radial_offset) {
  return radial_offset >= 0 ? radial_offset : -radial_offset - 1;
}

template <int NGHOST>
const char *LayerRegion(const int radial_offset) {
  return LayerFromOffset(radial_offset) < NGHOST ? "fitted" : "raw-transition";
}

template <int NGHOST>
constexpr int ExpectedNearAxisOrder(const int layer) {
  constexpr int fitted_order = 2 * (NGHOST - 1);
  // At fixed rho/h, a raw odd quotient divides the O(h^p) centered-Dx error
  // by rho=O(h), so the generic transition order is p-1. Fitted composites
  // retain p; fixed nonzero physical rho is checked separately at p.
  return layer < NGHOST ? fitted_order : fitted_order - 1;
}

template <int NGHOST>
constexpr double FitDerivativeRowOneNorm() {
  if constexpr (NGHOST == 2) return 1.0;
  if constexpr (NGHOST == 3) return 1.5;
  return 2.5;
}

template <int NGHOST>
constexpr double IndependentNoiseCoefficientNormBound() {
  // At a fitted node, 1/r_l^2 contributes at most 4/h^2. The largest
  // derivative composite contributes 4*(rho/h)^2 times the Lagrange-row
  // one-norm, while a two-component nonderivative difference contributes at
  // most 16. A factor two covers the manufactured advection multiplier and
  // the result is rounded upward to a power-of-two audit bound. The raw
  // transition has smaller reciprocal-radius and Dx coefficient norms.
  constexpr double outer_radius = static_cast<double>(NGHOST) - 0.5;
  constexpr double fitted_bound =
      2.0 * (16.0 * outer_radius * outer_radius *
                 FitDerivativeRowOneNorm<NGHOST>() +
             16.0);
  if constexpr (NGHOST == 2) {
    static_assert(fitted_bound < 128.0);
    return 128.0;
  }
  if constexpr (NGHOST == 3) {
    static_assert(fitted_bound < 512.0);
    return 512.0;
  }
  static_assert(fitted_bound < 1024.0);
  return 1024.0;
}

constexpr double RoundoffNoiseUlps() {
  // Two ulps remain representable in a float field without making its much
  // larger epsilon dominate this stress test. Double precision retains the
  // original 64-ulp probe to expose cancellation below truncation error.
  return sizeof(Real) == sizeof(float) ? 2.0 : 64.0;
}

template <int NGHOST>
constexpr double NoiseCoefficientSafety() {
  // Float gets 50% evaluation slack for short device arithmetic; double gets
  // 25%. This is intentionally distinct from merely scaling the old constant
  // by the much larger float epsilon.
  constexpr double evaluation_slack =
      sizeof(Real) == sizeof(float) ? 1.5 : 1.25;
  return evaluation_slack * IndependentNoiseCoefficientNormBound<NGHOST>();
}

std::string ResultName(const int result) {
  constexpr const char *scalar_names[10] = {
      "scalar.dx", "scalar.dz", "scalar.dy", "scalar.dxx", "scalar.dxz",
      "scalar.dzz", "scalar.dyy", "scalar.dxy", "scalar.dzy", "scalar.advective"};
  constexpr const char *vector_names[7] = {
      "dy", "dyy", "dxy", "dzy", "dx", "dxz", "advective"};
  constexpr const char *tensor_names[7] = {
      "dy", "dyy", "dxy", "dzy", "dx", "dxz", "advective"};
  if (result < 10) return scalar_names[result];
  if (result < 31) {
    const int offset = result - 10;
    return "vector[" + std::to_string(offset / 7) + "]." + vector_names[offset % 7];
  }
  if (result == 31) return "vector.divergence";
  if (result < 74) {
    const int offset = result - 32;
    return "tensor[" + std::to_string(offset / 7) + "]." + tensor_names[offset % 7];
  }
  return "state.dissipation";
}

bool CheckDirectNoiseDeltas(const ErrorSummary &clean, const ErrorSummary &noisy,
                            const bool isotropy_only, const double bound,
                            const int order, const double rho_over_h,
                            const int layer, const char *region,
                            const int noise_phase, const double h,
                            const char *noise_kind) {
  for (int result = 0; result < kNumResults; ++result) {
    if (isotropy_only && !IsIsotropySensitiveResult(result)) continue;
    const double clean_value = clean.observed[result];
    const double noisy_value = noisy.observed[result];
    const double amplification = fabs(noisy_value - clean_value);
    if (!std::isfinite(noisy_value) || amplification > bound) {
      std::cerr << "order " << order << " " << noise_kind
                << " noise amplification failed at rho/h=" << rho_over_h
                << " layer=" << layer << " region=" << region
                << " phase=" << noise_phase << " h=" << h
                << " result=" << ResultName(result) << " index=" << result
                << " clean=" << clean_value << " noisy=" << noisy_value
                << " oracle=" << clean.oracle[result]
                << " amplification=" << amplification << " bound=" << bound
                << '\n';
      return false;
    }
  }
  return true;
}

template <int NGHOST>
ErrorSummary MeasureSample(const double h, const double rho_offset,
                           const int radial_index_offset, const double z_sample,
                           const z4c::CartoonAxisLocation axis_location,
                           const double noise_amplitude = 0.0,
                           const int noise_phase = 0,
                           const bool independent_component_noise = false) {
  constexpr int n = 64;
  constexpr int center = n / 2;
  const int sample_i = center + radial_index_offset;
  const int sample_j = center + static_cast<int>(std::llround(z_sample / h));
  const double rho_sample = rho_offset + radial_index_offset * h;
  const double represented_z = (sample_j - center) * h;

  DvceArray5D<Real> state("cartoon manufactured state", 1, kNumVariables, 1, n, n);
  auto host = Kokkos::create_mirror_view(state);
  for (int j = 0; j < n; ++j) {
    const double z = (j - center) * h;
    for (int i = 0; i < n; ++i) {
      const double rho = rho_offset + (i - center) * h;
      const ManufacturedFields fields = EvaluateManufacturedFields(rho, z, 0.0);
      const int radial_layer = static_cast<int>(std::floor(std::abs(rho / h)));
      const int axial_layer = std::abs(j - center);
      const auto pattern = [&](const int component_key) {
        return static_cast<double>(
                   (17 * radial_layer + 13 * axial_layer + 7 * noise_phase +
                    5 * component_key + 3 * component_key * component_key) %
                       23 -
                   11) /
               11.0;
      };
      const double shared_pattern = pattern(0);
      const double even_noise = noise_amplitude * shared_pattern;
      const double odd_noise = noise_amplitude * rho * shared_pattern;
      const double quadratic_noise = noise_amplitude * rho * rho * shared_pattern;
      host(0, kScalarOffset, 0, j, i) = fields.scalar.value + even_noise;
      for (int component = 0; component < 3; ++component) {
        const double component_pattern =
            independent_component_noise ? pattern(1 + component) : shared_pattern;
        const double component_noise =
            (component == kZ) ? noise_amplitude * component_pattern
                              : noise_amplitude * rho * component_pattern;
        host(0, kVectorOffset + component, 0, j, i) =
            fields.vector[component].value + component_noise;
      }
      for (int first = 0; first < 3; ++first) {
        for (int second = first; second < 3; ++second) {
          double noise = 0.0;
          if (independent_component_noise) {
            const double component_pattern =
                pattern(4 + SymmetricIndex(first, second));
            const bool odd_component =
                (first == kZ) != (second == kZ);
            noise = noise_amplitude * component_pattern *
                    (odd_component ? rho : 1.0);
          } else {
            if (first == 0 && second == 0) noise = even_noise + quadratic_noise;
            if (first == 0 && second == 1) noise = odd_noise;
            if (first == 0 && second == 2) noise = quadratic_noise;
            if (first == 1 && second == 1) noise = even_noise;
            if (first == 1 && second == 2) noise = -odd_noise;
            if (first == 2 && second == 2) noise = even_noise - quadratic_noise;
          }
          host(0, kTensorOffset + SymmetricIndex(first, second), 0, j, i) =
              fields.tensor[first][second].value + noise;
        }
      }
    }
  }
  Kokkos::deep_copy(state, host);

  DvceArray1D<Real> results("cartoon derivative results", kNumResults);
  Kokkos::parallel_for(
      "cartoon manufactured derivatives", Kokkos::RangePolicy<DevExeSpace>(0, 1),
      KOKKOS_LAMBDA(const int) {
        ScalarField scalar{state};
        VectorField vector{state};
        TensorField tensor{state};
        const Real inverse_spacing[3] = {
            static_cast<Real>(1.0 / h), static_cast<Real>(1.0 / h),
            static_cast<Real>(1.0 / h)};
        z4c::DerivativeProvider<z4c::CartoonSO2, NGHOST> derivative(
            inverse_spacing, rho_sample, axis_location, 0, 0, sample_j, sample_i);

        results(0) = derivative.ScalarFirst(kRho, scalar);
        results(1) = derivative.ScalarFirst(kZ, scalar);
        results(2) = derivative.ScalarFirst(kSuppressed, scalar);
        results(3) = derivative.ScalarSecond(kRho, kRho, scalar);
        results(4) = derivative.ScalarSecond(kRho, kZ, scalar);
        results(5) = derivative.ScalarSecond(kZ, kZ, scalar);
        results(6) = derivative.ScalarSecond(kSuppressed, kSuppressed, scalar);
        results(7) = derivative.ScalarSecond(kRho, kSuppressed, scalar);
        results(8) = derivative.ScalarSecond(kZ, kSuppressed, scalar);
        results(9) = derivative.ScalarAdvective(vector, scalar);

        for (int component = 0; component < 3; ++component) {
          const int base = 10 + 7 * component;
          results(base) = derivative.VectorFirst(kSuppressed, component, vector);
          results(base + 1) =
              derivative.VectorSecond(kSuppressed, kSuppressed, component, vector);
          results(base + 2) =
              derivative.VectorSecond(kRho, kSuppressed, component, vector);
          results(base + 3) =
              derivative.VectorSecond(kZ, kSuppressed, component, vector);
          results(base + 4) = derivative.VectorFirst(kRho, component, vector);
          results(base + 5) = derivative.VectorSecond(kRho, kZ, component, vector);
          results(base + 6) = derivative.VectorAdvective(component, vector, vector);
        }
        results(31) = derivative.VectorDivergence(vector);

        const int tensor_first[6] = {0, 0, 0, 1, 1, 2};
        const int tensor_second[6] = {0, 1, 2, 1, 2, 2};
        for (int component = 0; component < 6; ++component) {
          const int first = tensor_first[component];
          const int second = tensor_second[component];
          const int base = 32 + 7 * component;
          results(base) = derivative.template TensorFirst<z4c::TensorVariance::all_lower>(
              kSuppressed, first, second, tensor);
          results(base + 1) =
              derivative.template TensorSecond<z4c::TensorVariance::all_lower>(
                  kSuppressed, kSuppressed, first, second, tensor);
          results(base + 2) =
              derivative.template TensorSecond<z4c::TensorVariance::all_lower>(
                  kRho, kSuppressed, first, second, tensor);
          results(base + 3) =
              derivative.template TensorSecond<z4c::TensorVariance::all_lower>(
                  kZ, kSuppressed, first, second, tensor);
          results(base + 4) = derivative.template TensorFirst<z4c::TensorVariance::all_lower>(
              kRho, first, second, tensor);
          results(base + 5) =
              derivative.template TensorSecond<z4c::TensorVariance::all_lower>(
                  kRho, kZ, first, second, tensor);
          results(base + 6) =
              derivative.template TensorAdvective<z4c::TensorVariance::all_lower>(
                  first, second, vector, tensor);
        }
        results(74) = derivative.ComponentDissipation(kScalarOffset, state);
      });
  Kokkos::fence();

  const auto result_host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), results);
  const ManufacturedFields oracle =
      EvaluateManufacturedFields(rho_sample, represented_z, 0.0);
  std::array<double, kNumResults> expected{};
  expected[0] = oracle.scalar.first[kRho];
  expected[1] = oracle.scalar.first[kZ];
  expected[2] = oracle.scalar.first[kSuppressed];
  expected[3] = oracle.scalar.second[kRho][kRho];
  expected[4] = oracle.scalar.second[kRho][kZ];
  expected[5] = oracle.scalar.second[kZ][kZ];
  expected[6] = oracle.scalar.second[kSuppressed][kSuppressed];
  expected[7] = oracle.scalar.second[kRho][kSuppressed];
  expected[8] = oracle.scalar.second[kZ][kSuppressed];
  for (int direction = 0; direction < 3; ++direction) {
    expected[9] += oracle.vector[direction].value * oracle.scalar.first[direction];
  }

  for (int component = 0; component < 3; ++component) {
    const int base = 10 + 7 * component;
    expected[base] = oracle.vector[component].first[kSuppressed];
    expected[base + 1] = oracle.vector[component].second[kSuppressed][kSuppressed];
    expected[base + 2] = oracle.vector[component].second[kRho][kSuppressed];
    expected[base + 3] = oracle.vector[component].second[kZ][kSuppressed];
    expected[base + 4] = oracle.vector[component].first[kRho];
    expected[base + 5] = oracle.vector[component].second[kRho][kZ];
    for (int direction = 0; direction < 3; ++direction) {
      expected[base + 6] +=
          oracle.vector[direction].value * oracle.vector[component].first[direction];
    }
    expected[31] += oracle.vector[component].first[component];
  }

  const int tensor_first[6] = {0, 0, 0, 1, 1, 2};
  const int tensor_second[6] = {0, 1, 2, 1, 2, 2};
  for (int component = 0; component < 6; ++component) {
    const int first = tensor_first[component];
    const int second = tensor_second[component];
    const int base = 32 + 7 * component;
    const Jet3 &value = oracle.tensor[first][second];
    expected[base] = value.first[kSuppressed];
    expected[base + 1] = value.second[kSuppressed][kSuppressed];
    expected[base + 2] = value.second[kRho][kSuppressed];
    expected[base + 3] = value.second[kZ][kSuppressed];
    expected[base + 4] = value.first[kRho];
    expected[base + 5] = value.second[kRho][kZ];
    for (int direction = 0; direction < 3; ++direction) {
      expected[base + 6] += oracle.vector[direction].value * value.first[direction];
    }
  }
  expected[74] = OracleDissipation<NGHOST>(h, rho_sample, represented_z);

  ErrorSummary summary;
  for (int result = 0; result < kNumResults; ++result) {
    summary.observed[result] = result_host(result);
    summary.oracle[result] = expected[result];
  }
  for (int result = 0; result < kNumResults - 1; ++result) {
    const double error =
        fabs(static_cast<double>(result_host(result)) - expected[result]);
    const int family = (result < 10) ? 0 : ((result < 32) ? 1 : 2);
    if (error > summary.family[family]) {
      summary.family[family] = error;
      summary.family_worst_result[family] = result;
    }
    if (error > summary.derivative) {
      summary.derivative = error;
      summary.worst_result = result;
      summary.worst_expected = expected[result];
      summary.worst_observed = result_host(result);
    }
  }
  summary.dissipation = fabs(static_cast<double>(result_host(74)) - expected[74]);
  return summary;
}

template <int NGHOST>
bool CheckFullApiAndCartesianDelegation(const double rho_sample) {
  constexpr int n = 32;
  constexpr int center = n / 2;
  constexpr int kFullResults = 171;
  const double h = 0.0625;
  const double z_sample = 0.25;
  const int sample_i = center + static_cast<int>(std::llround(rho_sample / h));
  const int sample_j = center + static_cast<int>(std::llround(z_sample / h));
  const int sample_k = center;

  DvceArray5D<Real> state("cartoon full api state", 1, kNumVariables, n, n, n);
  auto host = Kokkos::create_mirror_view(state);
  for (int k = 0; k < n; ++k) {
    const double y = (k - center) * h;
    for (int j = 0; j < n; ++j) {
      const double z = (j - center) * h;
      for (int i = 0; i < n; ++i) {
        const double rho = (i - center) * h;
        const ManufacturedFields fields = EvaluateManufacturedFields(rho, z, y);
        host(0, kScalarOffset, k, j, i) = fields.scalar.value;
        for (int component = 0; component < 3; ++component) {
          host(0, kVectorOffset + component, k, j, i) = fields.vector[component].value;
        }
        for (int first = 0; first < 3; ++first) {
          for (int second = first; second < 3; ++second) {
            host(0, kTensorOffset + SymmetricIndex(first, second), k, j, i) =
                fields.tensor[first][second].value;
          }
        }
      }
    }
  }
  Kokkos::deep_copy(state, host);

  DvceArray1D<Real> results("cartoon full api results", kFullResults);
  DvceArray1D<int> cartesian_mismatches("cartesian delegation mismatches", 1);
  Kokkos::parallel_for(
      "cartoon full api", Kokkos::RangePolicy<DevExeSpace>(0, 1),
      KOKKOS_LAMBDA(const int) {
        ScalarField scalar{state};
        VectorField vector{state};
        TensorField tensor{state};
        const Real inverse_spacing[3] = {
            static_cast<Real>(1.0 / h), static_cast<Real>(1.0 / h),
            static_cast<Real>(1.0 / h)};
        z4c::DerivativeProvider<z4c::CartoonSO2, NGHOST> cartoon(
            inverse_spacing, rho_sample, z4c::CartoonAxisLocation::cell_centered, 0,
            sample_k, sample_j, sample_i);
        z4c::DerivativeProvider<z4c::Cartesian3D, NGHOST> cartesian(
            inverse_spacing, 0, sample_k, sample_j, sample_i);

        int output = 0;
        for (int direction = 0; direction < 3; ++direction) {
          results(output++) = cartoon.ScalarFirst(direction, scalar);
        }
        for (int first_direction = 0; first_direction < 3; ++first_direction) {
          for (int second_direction = 0; second_direction < 3; ++second_direction) {
            results(output++) =
                cartoon.ScalarSecond(first_direction, second_direction, scalar);
          }
        }
        results(output++) = cartoon.ScalarAdvective(vector, scalar);
        for (int component = 0; component < 3; ++component) {
          for (int direction = 0; direction < 3; ++direction) {
            results(output++) = cartoon.VectorFirst(direction, component, vector);
          }
        }
        for (int component = 0; component < 3; ++component) {
          for (int first_direction = 0; first_direction < 3; ++first_direction) {
            for (int second_direction = 0; second_direction < 3; ++second_direction) {
              results(output++) = cartoon.VectorSecond(first_direction, second_direction,
                                                       component, vector);
            }
          }
        }
        results(output++) = cartoon.VectorDivergence(vector);
        for (int component = 0; component < 3; ++component) {
          results(output++) = cartoon.VectorAdvective(component, vector, vector);
        }
        for (int first = 0; first < 3; ++first) {
          for (int second = 0; second < 3; ++second) {
            for (int direction = 0; direction < 3; ++direction) {
              results(output++) =
                  cartoon.template TensorFirst<z4c::TensorVariance::all_lower>(
                      direction, first, second, tensor);
            }
          }
        }
        for (int first = 0; first < 3; ++first) {
          for (int second = 0; second < 3; ++second) {
            for (int first_direction = 0; first_direction < 3; ++first_direction) {
              for (int second_direction = 0; second_direction < 3; ++second_direction) {
                results(output++) =
                    cartoon.template TensorSecond<z4c::TensorVariance::all_lower>(
                        first_direction, second_direction, first, second, tensor);
              }
            }
          }
        }
        for (int first = 0; first < 3; ++first) {
          for (int second = 0; second < 3; ++second) {
            results(output++) =
                cartoon.template TensorAdvective<z4c::TensorVariance::all_lower>(
                    first, second, vector, tensor);
          }
        }
        results(output++) = cartoon.ComponentDissipation(kScalarOffset, state);

        int mismatches = 0;
        for (int direction = 0; direction < 3; ++direction) {
          mismatches += cartesian.ScalarFirst(direction, scalar) !=
                        Dx<NGHOST>(direction, inverse_spacing, scalar, 0, sample_k,
                                   sample_j, sample_i);
        }
        for (int first_direction = 0; first_direction < 3; ++first_direction) {
          for (int second_direction = 0; second_direction < 3; ++second_direction) {
            const Real direct = (first_direction == second_direction)
                                    ? Dxx<NGHOST>(first_direction, inverse_spacing, scalar,
                                                 0, sample_k, sample_j, sample_i)
                                    : Dxy<NGHOST>(first_direction, second_direction,
                                                 inverse_spacing, scalar, 0, sample_k,
                                                 sample_j, sample_i);
            mismatches +=
                cartesian.ScalarSecond(first_direction, second_direction, scalar) != direct;
          }
        }
        Real direct_scalar_advection = 0.0;
        for (int direction = 0; direction < 3; ++direction) {
          direct_scalar_advection += Lx<NGHOST>(direction, inverse_spacing, vector, scalar,
                                               0, direction, sample_k, sample_j, sample_i);
        }
        mismatches += cartesian.ScalarAdvective(vector, scalar) != direct_scalar_advection;
        Real direct_divergence = 0.0;
        for (int component = 0; component < 3; ++component) {
          for (int direction = 0; direction < 3; ++direction) {
            mismatches += cartesian.VectorFirst(direction, component, vector) !=
                          Dx<NGHOST>(direction, inverse_spacing, vector, 0, component,
                                     sample_k, sample_j, sample_i);
          }
          direct_divergence += Dx<NGHOST>(component, inverse_spacing, vector, 0,
                                          component, sample_k, sample_j, sample_i);
          for (int first_direction = 0; first_direction < 3; ++first_direction) {
            for (int second_direction = 0; second_direction < 3; ++second_direction) {
              const Real direct =
                  (first_direction == second_direction)
                      ? Dxx<NGHOST>(first_direction, inverse_spacing, vector, 0, component,
                                   sample_k, sample_j, sample_i)
                      : Dxy<NGHOST>(first_direction, second_direction, inverse_spacing,
                                   vector, 0, component, sample_k, sample_j, sample_i);
              mismatches += cartesian.VectorSecond(first_direction, second_direction,
                                                   component, vector) != direct;
            }
          }
          Real direct_advection = 0.0;
          for (int direction = 0; direction < 3; ++direction) {
            direct_advection += Lx<NGHOST>(direction, inverse_spacing, vector, vector, 0,
                                          direction, component, sample_k, sample_j,
                                          sample_i);
          }
          mismatches +=
              cartesian.VectorAdvective(component, vector, vector) != direct_advection;
        }
        mismatches += cartesian.VectorDivergence(vector) != direct_divergence;
        for (int first = 0; first < 3; ++first) {
          for (int second = 0; second < 3; ++second) {
            for (int direction = 0; direction < 3; ++direction) {
              mismatches +=
                  cartesian.template TensorFirst<z4c::TensorVariance::all_lower>(
                      direction, first, second, tensor) !=
                  Dx<NGHOST>(direction, inverse_spacing, tensor, 0, first, second,
                             sample_k, sample_j, sample_i);
            }
            for (int first_direction = 0; first_direction < 3; ++first_direction) {
              for (int second_direction = 0; second_direction < 3; ++second_direction) {
                const Real direct =
                    (first_direction == second_direction)
                        ? Dxx<NGHOST>(first_direction, inverse_spacing, tensor, 0, first,
                                     second, sample_k, sample_j, sample_i)
                        : Dxy<NGHOST>(first_direction, second_direction, inverse_spacing,
                                     tensor, 0, first, second, sample_k, sample_j,
                                     sample_i);
                mismatches +=
                    cartesian.template TensorSecond<z4c::TensorVariance::all_lower>(
                        first_direction, second_direction, first, second, tensor) != direct;
              }
            }
            Real direct_advection = 0.0;
            for (int direction = 0; direction < 3; ++direction) {
              direct_advection += Lx<NGHOST>(direction, inverse_spacing, vector, tensor, 0,
                                            direction, first, second, sample_k, sample_j,
                                            sample_i);
            }
            mismatches +=
                cartesian.template TensorAdvective<z4c::TensorVariance::all_lower>(
                    first, second, vector, tensor) != direct_advection;
          }
        }
        Real direct_dissipation = 0.0;
        for (int direction = 0; direction < 3; ++direction) {
          direct_dissipation += Diss<NGHOST>(direction, inverse_spacing, state, 0,
                                             kScalarOffset, sample_k, sample_j, sample_i);
        }
        mismatches += cartesian.ComponentDissipation(kScalarOffset, state) !=
                      direct_dissipation;
        mismatches +=
            cartoon.template TensorFirst<z4c::TensorVariance::all_upper>(
                kSuppressed, 0, 2, tensor) !=
            cartoon.template TensorFirst<z4c::TensorVariance::all_lower>(
                kSuppressed, 0, 2, tensor);
        mismatches +=
            cartoon.template TensorSecond<z4c::TensorVariance::all_upper>(
                kRho, kSuppressed, 0, 2, tensor) !=
            cartoon.template TensorSecond<z4c::TensorVariance::all_lower>(
                kRho, kSuppressed, 0, 2, tensor);
        mismatches +=
            cartoon.template TensorAdvective<z4c::TensorVariance::all_upper>(
                0, 2, vector, tensor) !=
            cartoon.template TensorAdvective<z4c::TensorVariance::all_lower>(
                0, 2, vector, tensor);
        cartesian_mismatches(0) = mismatches;
      });
  Kokkos::fence();

  const auto result_host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), results);
  const auto mismatch_host =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), cartesian_mismatches);
  if (mismatch_host(0) != 0) {
    std::cerr << "Cartesian delegation or tensor-variance instantiation had "
              << mismatch_host(0) << " bitwise mismatches at rho=" << rho_sample
              << '\n';
    return false;
  }

  const ManufacturedFields oracle =
      EvaluateManufacturedFields(rho_sample, z_sample, 0.0);
  std::vector<double> expected;
  std::vector<std::string> names;
  auto append = [&](const std::string &name, const double value) {
    names.push_back(name);
    expected.push_back(value);
  };
  for (int direction = 0; direction < 3; ++direction) {
    append("scalar.first." + std::to_string(direction), oracle.scalar.first[direction]);
  }
  for (int first_direction = 0; first_direction < 3; ++first_direction) {
    for (int second_direction = 0; second_direction < 3; ++second_direction) {
      append("scalar.second." + std::to_string(first_direction) + "." +
                 std::to_string(second_direction),
             oracle.scalar.second[first_direction][second_direction]);
    }
  }
  double scalar_advection = 0.0;
  for (int direction = 0; direction < 3; ++direction) {
    scalar_advection += oracle.vector[direction].value * oracle.scalar.first[direction];
  }
  append("scalar.advective", scalar_advection);
  for (int component = 0; component < 3; ++component) {
    for (int direction = 0; direction < 3; ++direction) {
      append("vector." + std::to_string(component) + ".first." +
                 std::to_string(direction),
             oracle.vector[component].first[direction]);
    }
  }
  for (int component = 0; component < 3; ++component) {
    for (int first_direction = 0; first_direction < 3; ++first_direction) {
      for (int second_direction = 0; second_direction < 3; ++second_direction) {
        append("vector." + std::to_string(component) + ".second." +
                   std::to_string(first_direction) + "." +
                   std::to_string(second_direction),
               oracle.vector[component].second[first_direction][second_direction]);
      }
    }
  }
  double divergence = 0.0;
  for (int component = 0; component < 3; ++component) {
    divergence += oracle.vector[component].first[component];
  }
  append("vector.divergence", divergence);
  for (int component = 0; component < 3; ++component) {
    double advection = 0.0;
    for (int direction = 0; direction < 3; ++direction) {
      advection +=
          oracle.vector[direction].value * oracle.vector[component].first[direction];
    }
    append("vector." + std::to_string(component) + ".advective", advection);
  }
  for (int first = 0; first < 3; ++first) {
    for (int second = 0; second < 3; ++second) {
      for (int direction = 0; direction < 3; ++direction) {
        append("tensor." + std::to_string(first) + "." + std::to_string(second) +
                   ".first." + std::to_string(direction),
               oracle.tensor[first][second].first[direction]);
      }
    }
  }
  for (int first = 0; first < 3; ++first) {
    for (int second = 0; second < 3; ++second) {
      for (int first_direction = 0; first_direction < 3; ++first_direction) {
        for (int second_direction = 0; second_direction < 3; ++second_direction) {
          append("tensor." + std::to_string(first) + "." + std::to_string(second) +
                     ".second." + std::to_string(first_direction) + "." +
                     std::to_string(second_direction),
                 oracle.tensor[first][second].second[first_direction][second_direction]);
        }
      }
    }
  }
  for (int first = 0; first < 3; ++first) {
    for (int second = 0; second < 3; ++second) {
      double advection = 0.0;
      for (int direction = 0; direction < 3; ++direction) {
        advection += oracle.vector[direction].value *
                     oracle.tensor[first][second].first[direction];
      }
      append("tensor." + std::to_string(first) + "." + std::to_string(second) +
                 ".advective",
             advection);
    }
  }
  append("state.dissipation", OracleDissipation<NGHOST>(h, rho_sample, z_sample));
  if (expected.size() != kFullResults) {
    std::cerr << "internal full-API result count mismatch\n";
    return false;
  }

  constexpr int order = 2 * (NGHOST - 1);
  for (int result = 0; result < kFullResults; ++result) {
    const double absolute_tolerance =
        (result == kFullResults - 1) ? 2.0e-10 : 100.0 * std::pow(h, order);
    const double tolerance = absolute_tolerance + 2.0e-8 * fabs(expected[result]);
    const double error = fabs(result_host(result) - expected[result]);
    if (error > tolerance) {
      std::cerr << "full API order " << order << " failed at rho=" << rho_sample
                << " result=" << names[result] << " expected=" << expected[result]
                << " observed=" << result_host(result) << " error=" << error
                << " tolerance=" << tolerance << '\n';
      return false;
    }
  }
  return true;
}

template <int NGHOST>
bool CheckMinimalFitReach() {
  using Provider = z4c::DerivativeProvider<z4c::CartoonSO2, NGHOST>;
  constexpr int axial_extent = 2 * NGHOST + 1;
  constexpr int targets = 2 * NGHOST;
  // The radial extent is exactly the fitted-side sample set, with no radial
  // ghosts. Every operation below is a suppressed composite whose fitted
  // branch needs only those samples (and, for z-mixed terms, axial ghosts).
  // Kokkos debug bounds therefore catches any extra or mirrored radial read.
  DvceArray5D<Real> state("minimal Cartoon fit reach state", 2, kNumVariables, 1,
                          axial_extent, NGHOST);
  auto host = Kokkos::create_mirror_view(state);
  for (int block = 0; block < 2; ++block) {
    for (int j = 0; j < axial_extent; ++j) {
      const double z = static_cast<double>(j - NGHOST);
      for (int i = 0; i < NGHOST; ++i) {
        const double rho = block == 0
                               ? static_cast<double>(i) + 0.5
                               : static_cast<double>(i - NGHOST) + 0.5;
        const ManufacturedFields fields = EvaluateManufacturedFields(rho, z, 0.0);
        host(block, kScalarOffset, 0, j, i) = fields.scalar.value;
        for (int component = 0; component < 3; ++component) {
          host(block, kVectorOffset + component, 0, j, i) =
              fields.vector[component].value;
        }
        for (int first = 0; first < 3; ++first) {
          for (int second = first; second < 3; ++second) {
            host(block, kTensorOffset + SymmetricIndex(first, second), 0, j, i) =
                fields.tensor[first][second].value;
          }
        }
      }
    }
  }
  Kokkos::deep_copy(state, host);

  DvceArray1D<Real> results("minimal Cartoon fit reach results", targets);
  Kokkos::parallel_for(
      "minimal Cartoon fit reach", Kokkos::RangePolicy<DevExeSpace>(0, targets),
      KOKKOS_LAMBDA(const int target) {
        const int block = target / NGHOST;
        const int layer = target % NGHOST;
        const int side_sign = block == 0 ? 1 : -1;
        const int sample_i = block == 0 ? layer : NGHOST - 1 - layer;
        const Real rho = side_sign * (static_cast<Real>(layer) + 0.5);
        const Real inverse_spacing[3] = {1.0, 1.0, 1.0};
        ScalarField scalar{state};
        VectorField vector{state};
        TensorField tensor{state};
        Provider derivative(inverse_spacing, rho,
                            z4c::CartoonAxisLocation::cell_centered, block, 0,
                            NGHOST, sample_i);
        Real sum = derivative.ScalarSecond(kSuppressed, kSuppressed, scalar);
        for (int component = 0; component < 3; ++component) {
          sum += derivative.VectorFirst(kSuppressed, component, vector);
          sum += derivative.VectorSecond(kSuppressed, kSuppressed, component, vector);
          sum += derivative.VectorSecond(kRho, kSuppressed, component, vector);
          sum += derivative.VectorSecond(kSuppressed, kRho, component, vector);
          sum += derivative.VectorSecond(kZ, kSuppressed, component, vector);
          sum += derivative.VectorSecond(kSuppressed, kZ, component, vector);
        }
        for (int first = 0; first < 3; ++first) {
          for (int second = 0; second < 3; ++second) {
            sum += derivative.template TensorFirst<
                z4c::TensorVariance::all_lower>(kSuppressed, first, second, tensor);
            sum += derivative.template TensorSecond<
                z4c::TensorVariance::all_lower>(kSuppressed, kSuppressed, first,
                                                second, tensor);
            sum += derivative.template TensorSecond<
                z4c::TensorVariance::all_lower>(kRho, kSuppressed, first, second,
                                                tensor);
            sum += derivative.template TensorSecond<
                z4c::TensorVariance::all_lower>(kSuppressed, kRho, first, second,
                                                tensor);
            sum += derivative.template TensorSecond<
                z4c::TensorVariance::all_lower>(kZ, kSuppressed, first, second,
                                                tensor);
            sum += derivative.template TensorSecond<
                z4c::TensorVariance::all_lower>(kSuppressed, kZ, first, second,
                                                tensor);
          }
        }
        results(target) = sum;
      });
  Kokkos::fence();
  const auto result_host =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), results);
  for (int target = 0; target < targets; ++target) {
    if (!std::isfinite(result_host(target))) {
      std::cerr << "order " << 2 * (NGHOST - 1)
                << " minimal-fit reach returned a non-finite value at side="
                << (target < NGHOST ? "+" : "-")
                << " layer=" << target % NGHOST << '\n';
      return false;
    }
  }
  return true;
}

template <int NGHOST>
bool CheckBlockBoundaryReach() {
  using Provider = z4c::DerivativeProvider<z4c::CartoonSO2, NGHOST>;
  static_assert(Provider::MaximumRegularizationOffset() == NGHOST - 1,
                "near-axis fit must stay within the ordinary stencil reach");
  constexpr int radial_extent = 3 * NGHOST;
  constexpr int axial_extent = 2 * NGHOST + 1;
  constexpr int targets = 2 * NGHOST;
  DvceArray5D<Real> state("block-local Cartoon reach state", 2, kNumVariables, 1,
                          axial_extent, radial_extent);
  auto host = Kokkos::create_mirror_view(state);
  for (int block = 0; block < 2; ++block) {
    for (int j = 0; j < axial_extent; ++j) {
      const double z = static_cast<double>(j - NGHOST);
      for (int i = 0; i < radial_extent; ++i) {
        const double rho = block == 0
                               ? static_cast<double>(i - NGHOST) + 0.5
                               : static_cast<double>(i - 2 * NGHOST) + 0.5;
        const ManufacturedFields fields = EvaluateManufacturedFields(rho, z, 0.0);
        host(block, kScalarOffset, 0, j, i) = fields.scalar.value;
        for (int component = 0; component < 3; ++component) {
          host(block, kVectorOffset + component, 0, j, i) =
              fields.vector[component].value;
        }
        for (int first = 0; first < 3; ++first) {
          for (int second = first; second < 3; ++second) {
            host(block, kTensorOffset + SymmetricIndex(first, second), 0, j, i) =
                fields.tensor[first][second].value;
          }
        }
      }
    }
  }
  Kokkos::deep_copy(state, host);

  DvceArray1D<Real> results("block-local Cartoon reach results", targets);
  Kokkos::parallel_for(
      "block-local Cartoon reach", Kokkos::RangePolicy<DevExeSpace>(0, targets),
      KOKKOS_LAMBDA(const int target) {
        const int block = target / NGHOST;
        const int layer = target % NGHOST;
        const int side_sign = block == 0 ? 1 : -1;
        const int sample_i = block == 0 ? NGHOST + layer
                                        : 2 * NGHOST - 1 - layer;
        const Real rho = side_sign * (static_cast<Real>(layer) + 0.5);
        const Real inverse_spacing[3] = {1.0, 1.0, 1.0};
        ScalarField scalar{state};
        VectorField vector{state};
        TensorField tensor{state};
        Provider derivative(inverse_spacing, rho,
                            z4c::CartoonAxisLocation::cell_centered, block, 0,
                            NGHOST, sample_i);
        Real sum = derivative.ScalarAdvective(vector, scalar);
        for (int direction = 0; direction < 3; ++direction) {
          sum += derivative.ScalarFirst(direction, scalar);
          for (int second_direction = 0; second_direction < 3;
               ++second_direction) {
            sum += derivative.ScalarSecond(direction, second_direction, scalar);
          }
        }
        sum += derivative.VectorDivergence(vector);
        for (int component = 0; component < 3; ++component) {
          sum += derivative.VectorAdvective(component, vector, vector);
          for (int direction = 0; direction < 3; ++direction) {
            sum += derivative.VectorFirst(direction, component, vector);
            for (int second_direction = 0; second_direction < 3;
                 ++second_direction) {
              sum += derivative.VectorSecond(direction, second_direction, component,
                                             vector);
            }
          }
        }
        for (int first = 0; first < 3; ++first) {
          for (int second = 0; second < 3; ++second) {
            sum += derivative.template TensorAdvective<
                z4c::TensorVariance::all_lower>(first, second, vector, tensor);
            for (int direction = 0; direction < 3; ++direction) {
              sum += derivative.template TensorFirst<
                  z4c::TensorVariance::all_lower>(direction, first, second, tensor);
              for (int second_direction = 0; second_direction < 3;
                   ++second_direction) {
                sum += derivative.template TensorSecond<
                    z4c::TensorVariance::all_lower>(direction, second_direction,
                                                    first, second, tensor);
              }
            }
          }
        }
        sum += derivative.ComponentDissipation(kScalarOffset, state);
        results(target) = sum;
      });
  Kokkos::fence();
  const auto result_host =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), results);
  for (int target = 0; target < targets; ++target) {
    if (!std::isfinite(result_host(target))) {
      std::cerr << "order " << 2 * (NGHOST - 1)
                << " block-boundary reach returned a non-finite value at side="
                << (target < NGHOST ? "+" : "-")
                << " layer=" << target % NGHOST << '\n';
      return false;
    }
  }
  return true;
}

template <int NGHOST>
bool CheckFixedRadiusRawConvergence(const double rho_sample) {
  constexpr int order = 2 * (NGHOST - 1);
  constexpr double coarse_h = 0.125;
  constexpr double medium_h = 0.0625;
  constexpr double fine_h = 0.03125;
  const ErrorSummary coarse = MeasureSample<NGHOST>(
      coarse_h, rho_sample, 0, 0.25, z4c::CartoonAxisLocation::cell_centered);
  const ErrorSummary medium = MeasureSample<NGHOST>(
      medium_h, rho_sample, 0, 0.25, z4c::CartoonAxisLocation::cell_centered);
  const ErrorSummary fine = MeasureSample<NGHOST>(
      fine_h, rho_sample, 0, 0.25, z4c::CartoonAxisLocation::cell_centered);
  const double observed_order = std::log2(medium.derivative / fine.derivative);
  const double fine_tolerance =
      200.0 * std::pow(fine_h, order) * (1.0 + fabs(fine.worst_expected));
  if (!(fine.derivative < medium.derivative && medium.derivative < coarse.derivative &&
        observed_order >= order - 0.15 && fine.derivative <= fine_tolerance)) {
    std::cerr << "order " << order
              << " fixed-radius raw convergence failed at rho=" << rho_sample
              << ": coarse=" << coarse.derivative << " medium=" << medium.derivative
              << " fine=" << fine.derivative << " observed order=" << observed_order
              << " worst=" << ResultName(fine.worst_result)
              << " expected=" << fine.worst_expected
              << " observed=" << fine.worst_observed
              << " tolerance=" << fine_tolerance << '\n';
    return false;
  }
  constexpr const char *family_names[3] = {"scalar", "vector", "tensor"};
  for (int family = 0; family < 3; ++family) {
    const double family_order = std::log2(medium.family[family] / fine.family[family]);
    if (!(fine.family[family] < medium.family[family] &&
          medium.family[family] < coarse.family[family] &&
          family_order >= order - 0.15)) {
      std::cerr << "order " << order << " fixed-radius raw " << family_names[family]
                << " family failed at rho=" << rho_sample
                << ": coarse=" << coarse.family[family]
                << " medium=" << medium.family[family]
                << " fine=" << fine.family[family]
                << " observed order=" << family_order << " worst="
                << ResultName(fine.family_worst_result[family]) << '\n';
      return false;
    }
  }
  std::cout << "order=" << order << " rho=" << rho_sample
            << " region=raw-fixed-radius observed_order=" << observed_order
            << " finest_error=" << fine.derivative
            << " worst_result=" << ResultName(fine.worst_result)
            << " expected=" << fine.worst_expected
            << " observed=" << fine.worst_observed << '\n';
  return true;
}

template <int NGHOST>
bool CheckOrder() {
  constexpr int order = 2 * (NGHOST - 1);
  const auto near_axis_offsets = SignedNearAxisOffsets<NGHOST>();
  // The negative sample is the pi-rotated signed-plane image of the positive sample.
  // Both are compared to independently rotated full-Cartesian jet derivatives.
  if (!CheckMinimalFitReach<NGHOST>() || !CheckBlockBoundaryReach<NGHOST>() ||
      !CheckFullApiAndCartesianDelegation<NGHOST>(0.5) ||
      !CheckFullApiAndCartesianDelegation<NGHOST>(-0.5) ||
      !CheckFixedRadiusRawConvergence<NGHOST>(0.5) ||
      !CheckFixedRadiusRawConvergence<NGHOST>(-0.5)) {
    return false;
  }
  const ErrorSummary coarse = MeasureSample<NGHOST>(
      0.125, 0.0, 4, 0.25, z4c::CartoonAxisLocation::cell_centered);
  const ErrorSummary fine = MeasureSample<NGHOST>(
      0.0625, 0.0, 8, 0.25, z4c::CartoonAxisLocation::cell_centered);
  const double observed_order = std::log2(coarse.derivative / fine.derivative);
  if (!(fine.derivative < coarse.derivative && observed_order >= order - 0.15)) {
    std::cerr << "order " << order << " convergence failed: coarse="
              << coarse.derivative << " fine=" << fine.derivative
              << " observed order=" << observed_order << " worst="
              << ResultName(fine.worst_result) << " expected=" << fine.worst_expected
              << " observed=" << fine.worst_observed << '\n';
    return false;
  }
  constexpr const char *family_names[3] = {"scalar", "vector", "tensor"};
  for (int family = 0; family < 3; ++family) {
    const double family_order = std::log2(coarse.family[family] / fine.family[family]);
    if (!(fine.family[family] < coarse.family[family] &&
          family_order >= order - 0.15)) {
      std::cerr << "order " << order << " " << family_names[family]
                << " family convergence failed: coarse=" << coarse.family[family]
                << " fine=" << fine.family[family]
                << " observed order=" << family_order << " worst="
                << ResultName(fine.family_worst_result[family]) << '\n';
      return false;
    }
  }

  constexpr int signed_point_count = 2 * (NGHOST + 1);
  std::array<ErrorSummary, signed_point_count> layer_coarse{};
  std::array<ErrorSummary, signed_point_count> layer_medium{};
  std::array<ErrorSummary, signed_point_count> layer_fine{};
  for (int point = 0; point < signed_point_count; ++point) {
    const int radial_offset = near_axis_offsets[point];
    const int layer = LayerFromOffset(radial_offset);
    const int expected_order = ExpectedNearAxisOrder<NGHOST>(layer);
    const double rho_over_h = radial_offset + 0.5;
    ErrorSummary &near_coarse = layer_coarse[point];
    ErrorSummary &near_medium = layer_medium[point];
    ErrorSummary &near_fine = layer_fine[point];
    near_coarse = MeasureSample<NGHOST>(
        0.125, 0.0625, radial_offset, 0.25,
        z4c::CartoonAxisLocation::cell_centered);
    near_medium = MeasureSample<NGHOST>(
        0.0625, 0.03125, radial_offset, 0.25,
        z4c::CartoonAxisLocation::cell_centered);
    near_fine = MeasureSample<NGHOST>(
        0.03125, 0.015625, radial_offset, 0.25,
        z4c::CartoonAxisLocation::cell_centered);
    const double near_order = std::log2(near_medium.derivative / near_fine.derivative);
    const double near_fine_tolerance =
        200.0 * std::pow(0.03125, expected_order) *
        (1.0 + fabs(near_fine.worst_expected));
    if (!(near_fine.derivative < near_medium.derivative &&
          near_medium.derivative < near_coarse.derivative &&
          near_order >= expected_order - 0.25 &&
          near_fine.derivative <= near_fine_tolerance)) {
      std::cerr << "order " << order << " near-axis convergence failed at rho/h="
                << rho_over_h << " layer=" << layer
                << " region=" << LayerRegion<NGHOST>(radial_offset)
                << " expected order=" << expected_order
                << ": coarse=" << near_coarse.derivative
                << " medium=" << near_medium.derivative
                << " fine=" << near_fine.derivative << " observed order=" << near_order
                << " worst=" << ResultName(near_fine.worst_result)
                << " expected=" << near_fine.worst_expected
                << " observed=" << near_fine.worst_observed
                << " normalized tolerance=" << near_fine_tolerance << '\n';
      return false;
    }
    for (int family = 0; family < 3; ++family) {
      const double family_order =
          std::log2(near_medium.family[family] / near_fine.family[family]);
      if (!(near_fine.family[family] < near_medium.family[family] &&
            near_medium.family[family] < near_coarse.family[family] &&
            family_order >= expected_order - 0.25)) {
        std::cerr << "order " << order << " near-axis " << family_names[family]
                  << " family failed at rho/h=" << rho_over_h
                  << " layer=" << layer
                  << " region=" << LayerRegion<NGHOST>(radial_offset)
                  << " expected order=" << expected_order
                  << ": coarse=" << near_coarse.family[family]
                  << " medium=" << near_medium.family[family]
                  << " fine=" << near_fine.family[family]
                  << " observed order=" << family_order << " worst="
                  << ResultName(near_fine.family_worst_result[family]) << '\n';
        return false;
      }
    }
    std::cout << "order=" << order << " rho/h=" << rho_over_h
              << " layer=" << layer
              << " region=" << LayerRegion<NGHOST>(radial_offset)
              << " expected_order=" << expected_order
              << " clean_observed_order=" << near_order
              << " finest_error=" << near_fine.derivative
              << " worst_result=" << ResultName(near_fine.worst_result)
              << " expected=" << near_fine.worst_expected
              << " observed=" << near_fine.worst_observed << '\n';
  }

  const double roundoff_noise =
      RoundoffNoiseUlps() * std::numeric_limits<Real>::epsilon();
  for (int noise_phase = 0; noise_phase < 8; ++noise_phase) {
    for (int point = 0; point < signed_point_count; ++point) {
      const int radial_offset = near_axis_offsets[point];
      const int layer = LayerFromOffset(radial_offset);
      const int expected_order = ExpectedNearAxisOrder<NGHOST>(layer);
      const double rho_over_h = radial_offset + 0.5;
      const ErrorSummary noisy_coarse = MeasureSample<NGHOST>(
          0.125, 0.0625, radial_offset, 0.25,
          z4c::CartoonAxisLocation::cell_centered, roundoff_noise, noise_phase);
      const ErrorSummary noisy_medium = MeasureSample<NGHOST>(
          0.0625, 0.03125, radial_offset, 0.25,
          z4c::CartoonAxisLocation::cell_centered, roundoff_noise, noise_phase);
      const ErrorSummary noisy_fine = MeasureSample<NGHOST>(
          0.03125, 0.015625, radial_offset, 0.25,
          z4c::CartoonAxisLocation::cell_centered, roundoff_noise, noise_phase);
      const double noisy_order =
          std::log2(noisy_medium.derivative / noisy_fine.derivative);
      const double noise_bound =
          200.0 * std::pow(0.03125, expected_order) *
              (1.0 + fabs(noisy_fine.worst_expected)) +
          NoiseCoefficientSafety<NGHOST>() * roundoff_noise /
              (0.03125 * 0.03125);
      const double coarse_delta_bound =
          NoiseCoefficientSafety<NGHOST>() * roundoff_noise / (0.125 * 0.125);
      const double medium_delta_bound =
          NoiseCoefficientSafety<NGHOST>() * roundoff_noise / (0.0625 * 0.0625);
      const double fine_delta_bound =
          NoiseCoefficientSafety<NGHOST>() * roundoff_noise / (0.03125 * 0.03125);
      if (!CheckDirectNoiseDeltas(
              layer_coarse[point], noisy_coarse, false, coarse_delta_bound, order,
              rho_over_h, layer, LayerRegion<NGHOST>(radial_offset), noise_phase,
              0.125, "shared-parity") ||
          !CheckDirectNoiseDeltas(
              layer_medium[point], noisy_medium, false, medium_delta_bound, order,
              rho_over_h, layer, LayerRegion<NGHOST>(radial_offset), noise_phase,
              0.0625, "shared-parity") ||
          !CheckDirectNoiseDeltas(
              layer_fine[point], noisy_fine, false, fine_delta_bound, order,
              rho_over_h, layer, LayerRegion<NGHOST>(radial_offset), noise_phase,
              0.03125, "shared-parity")) {
        return false;
      }
      if (!std::isfinite(noisy_fine.derivative) ||
          !(noisy_fine.derivative < noisy_medium.derivative &&
            noisy_medium.derivative < noisy_coarse.derivative &&
            noisy_order >= expected_order - 0.5 &&
            noisy_fine.derivative <= noise_bound)) {
        std::cerr << "order " << order << " parity-noise stability failed at rho/h="
                  << rho_over_h << " layer=" << layer
                  << " region=" << LayerRegion<NGHOST>(radial_offset)
                  << " expected order=" << expected_order
                  << " phase=" << noise_phase
                  << " coarse=" << noisy_coarse.derivative
                  << " medium=" << noisy_medium.derivative
                  << " fine=" << noisy_fine.derivative
                  << " observed order=" << noisy_order
                  << " worst=" << ResultName(noisy_fine.worst_result)
                  << " expected=" << noisy_fine.worst_expected
                  << " observed=" << noisy_fine.worst_observed
                  << " noise bound=" << noise_bound << '\n';
        return false;
      }
    }
  }

  // Independent even perturbations in T_xx, T_xy, and T_yy deliberately do not
  // cancel in the isotropy differences. The composite may amplify roundoff by
  // O(h^-2), but it must remain finite and within that explicit bound.
  constexpr double independent_noise_h = 0.03125;
  for (int noise_phase = 0; noise_phase < 8; ++noise_phase) {
    for (int point = 0; point < signed_point_count; ++point) {
      const int radial_offset = near_axis_offsets[point];
      const int layer = LayerFromOffset(radial_offset);
      const double rho_over_h = radial_offset + 0.5;
      const ErrorSummary noisy = MeasureSample<NGHOST>(
          independent_noise_h, 0.5 * independent_noise_h, radial_offset, 0.25,
          z4c::CartoonAxisLocation::cell_centered, roundoff_noise, noise_phase,
          true);
      const double amplification_bound =
          NoiseCoefficientSafety<NGHOST>() * roundoff_noise /
          (independent_noise_h * independent_noise_h);
      if (!CheckDirectNoiseDeltas(
              layer_fine[point], noisy, true, amplification_bound, order,
              rho_over_h, layer, LayerRegion<NGHOST>(radial_offset), noise_phase,
              independent_noise_h, "independent-component")) {
        return false;
      }
    }
  }

  const ErrorSummary axis = MeasureSample<NGHOST>(
      0.03125, 0.0, 0, 0.25, z4c::CartoonAxisLocation::diagnostic_axis);
  const double axis_tolerance = (order == 2) ? 0.02 : ((order == 4) ? 2.0e-4 : 2.0e-6);
  const double dissipation_tolerance = 2.0e-10;
  if (axis.derivative > axis_tolerance ||
      coarse.dissipation > dissipation_tolerance ||
      fine.dissipation > dissipation_tolerance ||
      axis.dissipation > dissipation_tolerance) {
    std::cerr << "order " << order << " axis/dissipation check failed: axis="
              << axis.derivative << " worst=" << ResultName(axis.worst_result)
              << " expected=" << axis.worst_expected
              << " observed=" << axis.worst_observed << " max diss="
              << fmax(coarse.dissipation, fine.dissipation) << '\n';
    return false;
  }
  return true;
}

bool CheckParity() {
  using Provider = z4c::DerivativeProvider<z4c::CartoonSO2, 2>;
  if (Provider::RegularizedHalfCellLayers() != 2 ||
      Provider::MaximumRegularizationOffset() != 1 ||
      Provider::ScalarParity() != 1 ||
      Provider::VectorParity(0) != -1 ||
      Provider::VectorParity(1) != 1 || Provider::VectorParity(2) != -1) {
    return false;
  }
  for (int first = 0; first < 3; ++first) {
    for (int second = 0; second < 3; ++second) {
      if (Provider::TensorParity(first, second) !=
          Provider::VectorParity(first) * Provider::VectorParity(second)) {
        return false;
      }
    }
  }
  return true;
}

bool CheckIndependentPiRotation() {
  constexpr int rotation_sign[3] = {-1, 1, -1};
  const ManufacturedFields positive = EvaluateManufacturedFields(0.5, 0.25, 0.0);
  const ManufacturedFields negative = EvaluateManufacturedFields(-0.5, 0.25, 0.0);
  const double tolerance = 2.0e-14;
  auto agrees = [&](const double actual, const double expected) {
    return fabs(actual - expected) <= tolerance * (1.0 + fabs(expected));
  };

  if (!agrees(negative.scalar.value, positive.scalar.value)) return false;
  for (int direction = 0; direction < 3; ++direction) {
    if (!agrees(negative.scalar.first[direction],
                rotation_sign[direction] * positive.scalar.first[direction])) {
      return false;
    }
    for (int second_direction = 0; second_direction < 3; ++second_direction) {
      if (!agrees(negative.scalar.second[direction][second_direction],
                  rotation_sign[direction] * rotation_sign[second_direction] *
                      positive.scalar.second[direction][second_direction])) {
        return false;
      }
    }
  }
  for (int component = 0; component < 3; ++component) {
    if (!agrees(negative.vector[component].value,
                rotation_sign[component] * positive.vector[component].value)) {
      return false;
    }
    for (int direction = 0; direction < 3; ++direction) {
      if (!agrees(negative.vector[component].first[direction],
                  rotation_sign[component] * rotation_sign[direction] *
                      positive.vector[component].first[direction])) {
        return false;
      }
      for (int second_direction = 0; second_direction < 3; ++second_direction) {
        if (!agrees(negative.vector[component].second[direction][second_direction],
                    rotation_sign[component] * rotation_sign[direction] *
                        rotation_sign[second_direction] *
                        positive.vector[component].second[direction][second_direction])) {
          return false;
        }
      }
    }
  }
  for (int first = 0; first < 3; ++first) {
    for (int second = 0; second < 3; ++second) {
      const int tensor_sign = rotation_sign[first] * rotation_sign[second];
      if (!agrees(negative.tensor[first][second].value,
                  tensor_sign * positive.tensor[first][second].value)) {
        return false;
      }
      for (int direction = 0; direction < 3; ++direction) {
        if (!agrees(negative.tensor[first][second].first[direction],
                    tensor_sign * rotation_sign[direction] *
                        positive.tensor[first][second].first[direction])) {
          return false;
        }
        for (int second_direction = 0; second_direction < 3; ++second_direction) {
          if (!agrees(negative.tensor[first][second].second[direction][second_direction],
                      tensor_sign * rotation_sign[direction] *
                          rotation_sign[second_direction] *
                          positive.tensor[first][second]
                              .second[direction][second_direction])) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

}  // namespace
