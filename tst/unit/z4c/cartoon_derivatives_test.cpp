//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cartoon_derivatives_test.cpp
//! \brief Independent manufactured-field tests for the analytic SO(2) provider.

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

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
  const Jet3 s4 = s2 * s2;

  // These coefficient functions are deliberately unrelated to the production provider.
  // Their high polynomial degree leaves nonzero truncation error at orders 2, 4, and 6.
  const Jet3 f = Constant(0.7) + 0.23 * s + 0.11 * s2 + 0.017 * s4 +
                 0.31 * z + 0.09 * z * z + 0.07 * s * z;
  const Jet3 a = Constant(0.4) + 0.13 * s + 0.021 * s4 + 0.05 * z;
  const Jet3 b = Constant(-0.2) + 0.08 * s + 0.014 * s4 - 0.03 * z;
  const Jet3 c = Constant(0.6) + 0.19 * s + 0.012 * s4 + 0.04 * z * z;
  const Jet3 p = Constant(1.1) + 0.12 * s + 0.016 * s4 + 0.02 * z;
  const Jet3 q = Constant(0.9) - 0.07 * s + 0.011 * s4 + 0.03 * z * z;
  const Jet3 r = Constant(0.3) + 0.09 * s + 0.013 * s4 + 0.04 * z;
  const Jet3 u = Constant(-0.17) + 0.06 * s + 0.009 * s4 - 0.02 * z;
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
};

template <int NGHOST>
ErrorSummary MeasureSample(const double h, const double rho_offset,
                           const int radial_index_offset, const double z_sample,
                           const double axis_tolerance) {
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
      host(0, kScalarOffset, 0, j, i) = fields.scalar.value;
      for (int component = 0; component < 3; ++component) {
        host(0, kVectorOffset + component, 0, j, i) = fields.vector[component].value;
      }
      for (int first = 0; first < 3; ++first) {
        for (int second = first; second < 3; ++second) {
          host(0, kTensorOffset + SymmetricIndex(first, second), 0, j, i) =
              fields.tensor[first][second].value;
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
        const Real inverse_spacing[3] = {1.0 / h, 1.0 / h, 1.0 / h};
        z4c::DerivativeProvider<z4c::CartoonSO2, NGHOST> derivative(
            inverse_spacing, rho_sample, axis_tolerance, 0, 0, sample_j, sample_i);

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
          results(base) = derivative.TensorFirst(kSuppressed, first, second, tensor);
          results(base + 1) =
              derivative.TensorSecond(kSuppressed, kSuppressed, first, second, tensor);
          results(base + 2) =
              derivative.TensorSecond(kRho, kSuppressed, first, second, tensor);
          results(base + 3) =
              derivative.TensorSecond(kZ, kSuppressed, first, second, tensor);
          results(base + 4) = derivative.TensorFirst(kRho, first, second, tensor);
          results(base + 5) =
              derivative.TensorSecond(kRho, kZ, first, second, tensor);
          results(base + 6) = derivative.TensorAdvective(first, second, vector, tensor);
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
  for (int result = 0; result < kNumResults - 1; ++result) {
    summary.derivative =
        fmax(summary.derivative, fabs(static_cast<double>(result_host(result)) -
                                      expected[result]));
  }
  summary.dissipation = fabs(static_cast<double>(result_host(74)) - expected[74]);
  return summary;
}

template <int NGHOST>
bool CheckOrder() {
  constexpr int order = 2 * (NGHOST - 1);
  const ErrorSummary coarse = MeasureSample<NGHOST>(0.125, 0.0, 4, 0.25, 0.0);
  const ErrorSummary fine = MeasureSample<NGHOST>(0.0625, 0.0, 8, 0.25, 0.0);
  const double required_ratio = std::pow(2.0, order - 1.0);
  if (!(fine.derivative < coarse.derivative &&
        coarse.derivative / fine.derivative > required_ratio)) {
    std::cerr << "order " << order << " convergence failed: coarse="
              << coarse.derivative << " fine=" << fine.derivative
              << " ratio=" << coarse.derivative / fine.derivative << '\n';
    return false;
  }

  const ErrorSummary positive_near_axis =
      MeasureSample<NGHOST>(0.0625, 0.03125, 0, 0.25, 0.0);
  const ErrorSummary negative_near_axis =
      MeasureSample<NGHOST>(0.0625, 0.03125, -1, 0.25, 0.0);
  const ErrorSummary axis = MeasureSample<NGHOST>(0.03125, 0.0, 0, 0.25, 1.0e-14);
  const double near_axis_tolerance =
      (order == 2) ? 0.25 : ((order == 4) ? 0.02 : 0.002);
  const double axis_tolerance = (order == 2) ? 0.02 : ((order == 4) ? 2.0e-4 : 2.0e-6);
  const double dissipation_tolerance = 2.0e-10;
  if (positive_near_axis.derivative > near_axis_tolerance ||
      negative_near_axis.derivative > near_axis_tolerance ||
      axis.derivative > axis_tolerance ||
      coarse.dissipation > dissipation_tolerance ||
      fine.dissipation > dissipation_tolerance ||
      positive_near_axis.dissipation > dissipation_tolerance ||
      negative_near_axis.dissipation > dissipation_tolerance ||
      axis.dissipation > dissipation_tolerance) {
    std::cerr << "order " << order << " axis/dissipation check failed: +axis="
              << positive_near_axis.derivative << " -axis="
              << negative_near_axis.derivative << " axis=" << axis.derivative
              << " max diss="
              << fmax(fmax(coarse.dissipation, fine.dissipation),
                      fmax(positive_near_axis.dissipation,
                           negative_near_axis.dissipation))
              << '\n';
    return false;
  }
  return true;
}

bool CheckParity() {
  using Provider = z4c::DerivativeProvider<z4c::CartoonSO2, 2>;
  if (Provider::ScalarParity() != 1 || Provider::VectorParity(0) != -1 ||
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

}  // namespace

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  bool passed = false;
  {
    passed = CheckParity() && CheckOrder<2>() && CheckOrder<3>() && CheckOrder<4>();
  }
  Kokkos::finalize();
  if (!passed) {
    return EXIT_FAILURE;
  }
  std::cout << "Cartoon derivative manufactured-oracle tests passed\n";
  return EXIT_SUCCESS;
}
