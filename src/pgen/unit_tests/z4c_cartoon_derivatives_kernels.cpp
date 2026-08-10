//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c_cartoon_derivatives.cpp
//! \brief Input-selected manufactured-solution check for the production SO(2) provider.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/z4c_cartoon_derivatives_oracle.hpp"
#include "pgen/unit_tests/z4c_cartoon_derivatives.hpp"
#include "z4c/cartoon_derivatives.hpp"
#include "z4c/stored_domain_bounds.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_symmetry.hpp"

namespace {

constexpr int kRho = 0;
constexpr int kZ = 1;
constexpr int kSuppressed = 2;
constexpr int kScalarOffset = 0;
constexpr int kVectorOffset = 1;
constexpr int kTensorOffset = 4;
constexpr int kVariables = 10;
constexpr int kResults = 171;
constexpr int kRegions = 14;
constexpr int kTensorFirst[6] = {0, 0, 0, 1, 1, 2};
constexpr int kTensorSecond[6] = {0, 1, 2, 1, 2, 2};

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

KOKKOS_INLINE_FUNCTION
Real NoisePattern(const int radial_layer, const int axial_index,
                  const int component, const int phase) {
  int value = (17 * radial_layer + 13 * axial_index + 7 * phase +
               5 * component + 3 * component * component) % 23;
  if (value < 0) value += 23;
  return static_cast<Real>(value - 11) / 11.0;
}

KOKKOS_INLINE_FUNCTION
void StoreManufacturedState(const z4c_mms::FieldValues &oracle, const Real rho,
                            const int radial_layer, const int axial_index,
                            const Real amplitude, const int phase,
                            const int noise_kind, const int m,
                            const int k, const int j, const int i,
                            const DvceArray5D<Real> &state) {
  const Real shared = NoisePattern(radial_layer, axial_index, 0, phase);
  const Real even_noise = amplitude * shared;
  const Real odd_noise = amplitude * rho * shared;
  const Real quadratic_noise = amplitude * rho * rho * shared;
  state(m, kScalarOffset, k, j, i) =
      oracle.scalar + (noise_kind == 2 ? 0.0 : even_noise);
  for (int component = 0; component < 3; ++component) {
    const Real parity_scale = component == kZ ? 1.0 : rho;
    const Real pattern = noise_kind == 2 ? 0.0 : shared;
    state(m, kVectorOffset + component, k, j, i) =
        oracle.vector[component] + amplitude * parity_scale * pattern;
  }
  for (int component = 0; component < 6; ++component) {
    Real noise = 0.0;
    if (noise_kind == 2) {
      if (component == 0 || component == 2 || component == 5) {
        noise = amplitude * NoisePattern(radial_layer, axial_index,
                                         4 + component, phase);
      }
    } else {
      if (component == 0) noise = even_noise + quadratic_noise;
      if (component == 1) noise = odd_noise;
      if (component == 2) noise = quadratic_noise;
      if (component == 3) noise = even_noise;
      if (component == 4) noise = -odd_noise;
      if (component == 5) noise = even_noise - quadratic_noise;
    }
    state(m, kTensorOffset + component, k, j, i) =
        oracle.tensor[component] + noise;
  }
}

template <int NGHOST>
KOKKOS_INLINE_FUNCTION Real AnalyticDissipation(
    const int component, const Real rho, const Real z, const Real dx1,
    const Real dx2) {
  constexpr int coefficients2[5] = {1, -4, 6, -4, 1};
  constexpr int coefficients3[7] = {1, -6, 15, -20, 15, -6, 1};
  constexpr int coefficients4[9] = {1, -8, 28, -56, 70, -56, 28, -8, 1};
  Real result = 0.0;
  for (int direction = 0; direction < 2; ++direction) {
    const Real spacing = direction == 0 ? dx1 : dx2;
    for (int offset = -NGHOST; offset <= NGHOST; ++offset) {
      int coefficient = 0;
      if constexpr (NGHOST == 2) coefficient = coefficients2[offset + NGHOST];
      if constexpr (NGHOST == 3) coefficient = coefficients3[offset + NGHOST];
      if constexpr (NGHOST == 4) coefficient = coefficients4[offset + NGHOST];
      z4c_mms::FieldValues shifted;
      z4c_mms::EvaluateFieldValues(
          rho + (direction == 0 ? offset * spacing : 0.0), 0.0,
          z + (direction == 1 ? offset * spacing : 0.0), shifted);
      Real value = shifted.scalar;
      if (component >= kVectorOffset && component < kTensorOffset) {
        value = shifted.vector[component - kVectorOffset];
      } else if (component >= kTensorOffset) {
        value = shifted.tensor[component - kTensorOffset];
      }
      result += coefficient * value / spacing;
    }
  }
  return result;
}

template <int NGHOST>
constexpr Real NoiseCoefficientSafety() {
  constexpr Real row_norm = NGHOST == 2 ? 1.0 : (NGHOST == 3 ? 1.5 : 2.5);
  constexpr Real outer_radius = static_cast<Real>(NGHOST) - 0.5;
  constexpr Real fitted_bound =
      2.0 * (16.0 * outer_radius * outer_radius * row_norm + 16.0);
  constexpr Real rounded = NGHOST == 2 ? 128.0 : (NGHOST == 3 ? 512.0 : 1024.0);
  static_assert(fitted_bound < rounded);
  constexpr Real evaluation_slack = sizeof(Real) == sizeof(float) ? 1.5 : 1.25;
  return evaluation_slack * rounded;
}


KOKKOS_INLINE_FUNCTION
void StoreComparison(const int local, const int output, const Real clean,
                     const Real shared, const Real independent,
                     const Real expected, DvceArray2D<Real> clean_values,
                     DvceArray2D<Real> shared_values,
                     DvceArray2D<Real> independent_values,
                     DvceArray2D<Real> clean_errors,
                     DvceArray2D<Real> shared_errors,
                     DvceArray2D<Real> independent_errors,
                     DvceArray2D<Real> shared_deltas,
                     DvceArray2D<Real> independent_deltas) {
  clean_values(local, output) = clean;
  shared_values(local, output) = shared;
  independent_values(local, output) = independent;
  clean_errors(local, output) = fabs(clean - expected);
  shared_errors(local, output) = fabs(shared - expected);
  independent_errors(local, output) = fabs(independent - expected);
  shared_deltas(local, output) = fabs(shared - clean);
  independent_deltas(local, output) = fabs(independent - clean);
}

template <z4c::TensorVariance Variance, int NGHOST>
KOKKOS_INLINE_FUNCTION void EvaluateTensorComponents(
    const z4c::DerivativeProvider<z4c::CartoonSO2, NGHOST> &derivative,
    VectorField &clean_vector, TensorField &clean_tensor,
    VectorField &shared_vector, TensorField &shared_tensor,
    VectorField &independent_vector, TensorField &independent_tensor,
    const z4c_mms::FieldValues &fields, const Real rho, const Real z,
    const int local, const int base,
    DvceArray2D<Real> clean_values, DvceArray2D<Real> shared_values,
    DvceArray2D<Real> independent_values, DvceArray2D<Real> clean_errors,
    DvceArray2D<Real> shared_errors, DvceArray2D<Real> independent_errors,
    DvceArray2D<Real> shared_deltas, DvceArray2D<Real> independent_deltas) {
  for (int component = 0; component < 6; ++component) {
    z4c_mms::TensorOracle oracle;
    z4c_mms::EvaluateTensorOracle(component, rho, 0.0, z, oracle);
    const int first_component = kTensorFirst[component];
    const int second_component = kTensorSecond[component];
    const int component_base = base + 10 * component;
    for (int direction = 0; direction < 3; ++direction) {
      StoreComparison(
          local, component_base + direction,
          derivative.template TensorFirst<Variance>(
              direction, first_component, second_component, clean_tensor),
          derivative.template TensorFirst<Variance>(
              direction, first_component, second_component, shared_tensor),
          derivative.template TensorFirst<Variance>(
              direction, first_component, second_component, independent_tensor),
          oracle.first[direction], clean_values, shared_values,
          independent_values, clean_errors, shared_errors, independent_errors,
          shared_deltas, independent_deltas);
    }
    for (int direction = 0; direction < 6; ++direction) {
      const int first = kTensorFirst[direction];
      const int second = kTensorSecond[direction];
      StoreComparison(
          local, component_base + 3 + direction,
          derivative.template TensorSecond<Variance>(
              first, second, first_component, second_component, clean_tensor),
          derivative.template TensorSecond<Variance>(
              first, second, first_component, second_component, shared_tensor),
          derivative.template TensorSecond<Variance>(
              first, second, first_component, second_component, independent_tensor),
          oracle.second[first][second], clean_values, shared_values,
          independent_values, clean_errors, shared_errors, independent_errors,
          shared_deltas, independent_deltas);
    }
    Real expected_advection = 0.0;
    for (int direction = 0; direction < 3; ++direction) {
      expected_advection += fields.vector[direction] * oracle.first[direction];
    }
    StoreComparison(
        local, component_base + 9,
        derivative.template TensorAdvective<Variance>(
            first_component, second_component, clean_vector, clean_tensor),
        derivative.template TensorAdvective<Variance>(
            first_component, second_component, shared_vector, shared_tensor),
        derivative.template TensorAdvective<Variance>(
            first_component, second_component, independent_vector,
            independent_tensor),
        expected_advection, clean_values, shared_values, independent_values,
        clean_errors, shared_errors, independent_errors, shared_deltas,
        independent_deltas);
  }
}

template <int NGHOST>
KOKKOS_INLINE_FUNCTION void EvaluateAndStorePoint(
    const z4c::DerivativeProvider<z4c::CartoonSO2, NGHOST> &derivative,
    ScalarField &clean_scalar, VectorField &clean_vector, TensorField &clean_tensor,
    ScalarField &shared_scalar, VectorField &shared_vector, TensorField &shared_tensor,
    ScalarField &independent_scalar, VectorField &independent_vector,
    TensorField &independent_tensor, DvceArray5D<Real> clean_state,
    DvceArray5D<Real> shared_state, DvceArray5D<Real> independent_state,
    const z4c_mms::FieldValues &fields, const Real rho, const Real z,
    const Real dx1, const Real dx2, const int local,
    DvceArray2D<Real> clean_values, DvceArray2D<Real> shared_values,
    DvceArray2D<Real> independent_values, DvceArray2D<Real> clean_errors,
    DvceArray2D<Real> shared_errors, DvceArray2D<Real> independent_errors,
    DvceArray2D<Real> shared_deltas, DvceArray2D<Real> independent_deltas) {
  z4c_mms::ScalarOracle scalar_oracle;
  z4c_mms::EvaluateScalarOracle(rho, 0.0, z, scalar_oracle);
  for (int direction = 0; direction < 3; ++direction) {
    StoreComparison(local, direction, derivative.ScalarFirst(direction, clean_scalar),
                    derivative.ScalarFirst(direction, shared_scalar),
                    derivative.ScalarFirst(direction, independent_scalar),
                    scalar_oracle.first[direction], clean_values, shared_values,
                    independent_values, clean_errors, shared_errors,
                    independent_errors, shared_deltas, independent_deltas);
  }
  for (int direction = 0; direction < 6; ++direction) {
    const int first = kTensorFirst[direction];
    const int second = kTensorSecond[direction];
    StoreComparison(local, 3 + direction,
                    derivative.ScalarSecond(first, second, clean_scalar),
                    derivative.ScalarSecond(first, second, shared_scalar),
                    derivative.ScalarSecond(first, second, independent_scalar),
                    scalar_oracle.second[first][second], clean_values, shared_values,
                    independent_values, clean_errors, shared_errors,
                    independent_errors, shared_deltas, independent_deltas);
  }
  Real scalar_advection = 0.0;
  for (int direction = 0; direction < 3; ++direction) {
    scalar_advection += fields.vector[direction] * scalar_oracle.first[direction];
  }
  StoreComparison(local, 9, derivative.ScalarAdvective(clean_vector, clean_scalar),
                  derivative.ScalarAdvective(shared_vector, shared_scalar),
                  derivative.ScalarAdvective(independent_vector, independent_scalar),
                  scalar_advection, clean_values, shared_values, independent_values,
                  clean_errors, shared_errors, independent_errors, shared_deltas,
                  independent_deltas);

  for (int component = 0; component < 3; ++component) {
    z4c_mms::VectorOracle vector_oracle;
    z4c_mms::EvaluateVectorOracle(component, rho, 0.0, z, vector_oracle);
    for (int direction = 0; direction < 3; ++direction) {
      StoreComparison(
          local, 10 + 3 * component + direction,
          derivative.VectorFirst(direction, component, clean_vector),
          derivative.VectorFirst(direction, component, shared_vector),
          derivative.VectorFirst(direction, component, independent_vector),
          vector_oracle.first[direction], clean_values, shared_values,
          independent_values, clean_errors, shared_errors, independent_errors,
          shared_deltas, independent_deltas);
    }
    for (int direction = 0; direction < 6; ++direction) {
      const int first = kTensorFirst[direction];
      const int second = kTensorSecond[direction];
      StoreComparison(
          local, 19 + 6 * component + direction,
          derivative.VectorSecond(first, second, component, clean_vector),
          derivative.VectorSecond(first, second, component, shared_vector),
          derivative.VectorSecond(first, second, component, independent_vector),
          vector_oracle.second[first][second], clean_values, shared_values,
          independent_values, clean_errors, shared_errors, independent_errors,
          shared_deltas, independent_deltas);
    }
    Real vector_advection = 0.0;
    for (int direction = 0; direction < 3; ++direction) {
      vector_advection += fields.vector[direction] * vector_oracle.first[direction];
    }
    StoreComparison(local, 38 + component,
                    derivative.VectorAdvective(component, clean_vector, clean_vector),
                    derivative.VectorAdvective(component, shared_vector, shared_vector),
                    derivative.VectorAdvective(component, independent_vector,
                                                independent_vector),
                    vector_advection, clean_values, shared_values, independent_values,
                    clean_errors, shared_errors, independent_errors, shared_deltas,
                    independent_deltas);
  }
  Real divergence = 0.0;
  for (int component = 0; component < 3; ++component) {
    z4c_mms::VectorOracle vector_oracle;
    z4c_mms::EvaluateVectorOracle(component, rho, 0.0, z, vector_oracle);
    divergence += vector_oracle.first[component];
  }
  StoreComparison(local, 37, derivative.VectorDivergence(clean_vector),
                  derivative.VectorDivergence(shared_vector),
                  derivative.VectorDivergence(independent_vector), divergence,
                  clean_values, shared_values, independent_values, clean_errors,
                  shared_errors, independent_errors, shared_deltas,
                  independent_deltas);

  EvaluateTensorComponents<z4c::TensorVariance::all_lower>(
      derivative, clean_vector, clean_tensor, shared_vector, shared_tensor,
      independent_vector, independent_tensor, fields, rho, z, local, 41, clean_values,
      shared_values, independent_values, clean_errors, shared_errors,
      independent_errors, shared_deltas, independent_deltas);
  EvaluateTensorComponents<z4c::TensorVariance::all_upper>(
      derivative, clean_vector, clean_tensor, shared_vector, shared_tensor,
      independent_vector, independent_tensor, fields, rho, z, local, 101, clean_values,
      shared_values, independent_values, clean_errors, shared_errors,
      independent_errors, shared_deltas, independent_deltas);

  for (int component = 0; component < kVariables; ++component) {
    const Real expected = AnalyticDissipation<NGHOST>(component, rho, z, dx1, dx2);
    StoreComparison(local, 161 + component,
                    derivative.ComponentDissipation(component, clean_state),
                    derivative.ComponentDissipation(component, shared_state),
                    derivative.ComponentDissipation(component, independent_state),
                    expected, clean_values, shared_values, independent_values,
                    clean_errors, shared_errors, independent_errors, shared_deltas,
                    independent_deltas);
  }
}

std::vector<std::string> ResultNames() {
  std::vector<std::string> names;
  for (int direction = 0; direction < 3; ++direction) {
    names.push_back("scalar.first." + std::to_string(direction));
  }
  for (int derivative = 0; derivative < 6; ++derivative) {
    names.push_back("scalar.second." +
                    std::to_string(kTensorFirst[derivative]) + "." +
                    std::to_string(kTensorSecond[derivative]));
  }
  names.push_back("scalar.advective");
  for (int component = 0; component < 3; ++component) {
    for (int direction = 0; direction < 3; ++direction) {
      names.push_back("vector." + std::to_string(component) + ".first." +
                      std::to_string(direction));
    }
  }
  for (int component = 0; component < 3; ++component) {
    for (int derivative = 0; derivative < 6; ++derivative) {
      names.push_back("vector." + std::to_string(component) + ".second." +
                      std::to_string(kTensorFirst[derivative]) + "." +
                      std::to_string(kTensorSecond[derivative]));
    }
  }
  names.push_back("vector.divergence");
  for (int component = 0; component < 3; ++component) {
    names.push_back("vector." + std::to_string(component) + ".advective");
  }
  for (const char *variance : {"lower", "upper"}) {
    for (int component = 0; component < 6; ++component) {
      const std::string prefix = "tensor." + std::string(variance) + "." +
                                 std::to_string(kTensorFirst[component]) + "." +
                                 std::to_string(kTensorSecond[component]);
      for (int direction = 0; direction < 3; ++direction) {
        names.push_back(prefix + ".first." + std::to_string(direction));
      }
      for (int derivative = 0; derivative < 6; ++derivative) {
        names.push_back(prefix + ".second." +
                        std::to_string(kTensorFirst[derivative]) + "." +
                        std::to_string(kTensorSecond[derivative]));
      }
      names.push_back(prefix + ".advective");
    }
  }
  for (int component = 0; component < kVariables; ++component) {
    names.push_back("state." + std::to_string(component) + ".dissipation");
  }
  return names;
}

std::vector<int> ResultParities() {
  constexpr int direction_parity[3] = {-1, 1, -1};
  constexpr int component_parity[3] = {-1, 1, -1};
  std::vector<int> parity;
  for (int direction = 0; direction < 3; ++direction) {
    parity.push_back(direction_parity[direction]);
  }
  for (int derivative = 0; derivative < 6; ++derivative) {
    parity.push_back(direction_parity[kTensorFirst[derivative]] *
                     direction_parity[kTensorSecond[derivative]]);
  }
  parity.push_back(1);
  for (int component = 0; component < 3; ++component) {
    for (int direction = 0; direction < 3; ++direction) {
      parity.push_back(component_parity[component] * direction_parity[direction]);
    }
  }
  for (int component = 0; component < 3; ++component) {
    for (int derivative = 0; derivative < 6; ++derivative) {
      parity.push_back(component_parity[component] *
                       direction_parity[kTensorFirst[derivative]] *
                       direction_parity[kTensorSecond[derivative]]);
    }
  }
  parity.push_back(1);
  for (int component = 0; component < 3; ++component) {
    parity.push_back(component_parity[component]);
  }
  for (int variance = 0; variance < 2; ++variance) {
    for (int component = 0; component < 6; ++component) {
      const int tensor_parity = component_parity[kTensorFirst[component]] *
                                component_parity[kTensorSecond[component]];
      for (int direction = 0; direction < 3; ++direction) {
        parity.push_back(tensor_parity * direction_parity[direction]);
      }
      for (int derivative = 0; derivative < 6; ++derivative) {
        parity.push_back(tensor_parity *
                         direction_parity[kTensorFirst[derivative]] *
                         direction_parity[kTensorSecond[derivative]]);
      }
      parity.push_back(tensor_parity);
    }
  }
  parity.push_back(1);
  for (int component = 0; component < 3; ++component) {
    parity.push_back(component_parity[component]);
  }
  for (int component = 0; component < 6; ++component) {
    parity.push_back(component_parity[kTensorFirst[component]] *
                     component_parity[kTensorSecond[component]]);
  }
  return parity;
}

struct NormRecord {
  double sum_abs = 0.0;
  double sum_square = 0.0;
  double maximum = 0.0;
  double cylindrical_abs = 0.0;
  double cylindrical_square = 0.0;
  double cylindrical_volume = 0.0;
  double cylindrical_maximum = 0.0;
  long long count = 0;
  long long cylindrical_count = 0;
  long long nonfinite = 0;
  std::uint64_t mask_xor = 0;
};

std::uint64_t HashCellId(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

template <int NGHOST>
double RunDiagnosticAxisProbe() {
  constexpr int n = 2 * NGHOST + 9;
  constexpr int center = n / 2;
  constexpr Real spacing = 0.03125;
  DvceArray5D<Real> state("Cartoon MMS diagnostic axis", 1, kVariables, 1, n, n);
  Kokkos::parallel_for(
      "fill Cartoon MMS diagnostic-axis probe",
      Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<2>>({0, 0}, {n, n}),
      KOKKOS_LAMBDA(const int j, const int i) {
      const Real z = (j - center) * spacing;
      const Real rho = (i - center) * spacing;
      z4c_mms::FieldValues oracle;
      z4c_mms::EvaluateFieldValues(rho, 0.0, z, oracle);
      const int radial_layer = static_cast<int>(floor(fabs(rho / spacing)));
      const int axial_index = static_cast<int>(llround(z / spacing));
      StoreManufacturedState(oracle, rho, radial_layer, axial_index,
                             0.0, 0, 0, 0, 0, j, i, state);
      });
  DvceArray2D<Real> axis_values("Cartoon MMS axis values", 1, kResults);
  DvceArray2D<Real> axis_errors("Cartoon MMS axis errors", 1, kResults);
  DvceArray2D<Real> axis_deltas("Cartoon MMS axis deltas", 1, kResults);
  Kokkos::parallel_for(
      "Cartoon MMS diagnostic-axis probe", Kokkos::RangePolicy<DevExeSpace>(0, 1),
      KOKKOS_LAMBDA(const int) {
        const Real inverse_spacing[3] = {1.0 / spacing, 1.0 / spacing,
                                         1.0 / spacing};
        ScalarField scalar{state};
        VectorField vector{state};
        TensorField tensor{state};
        z4c::DerivativeProvider<z4c::CartoonSO2, NGHOST> derivative(
            inverse_spacing, 0.0, z4c::CartoonAxisLocation::diagnostic_axis,
            0, 0, center, center);
        z4c_mms::FieldValues fields;
        z4c_mms::EvaluateFieldValues(0.0, 0.0, 0.0, fields);
        EvaluateAndStorePoint(
            derivative, scalar, vector, tensor, scalar, vector, tensor, scalar,
            vector, tensor, state, state, state, fields, 0.0, 0.0, spacing,
            spacing, 0, axis_values, axis_values, axis_values, axis_errors,
            axis_errors, axis_errors, axis_deltas, axis_deltas);
      });
  auto errors = Kokkos::create_mirror_view_and_copy(HostMemSpace(), axis_errors);
  double maximum = 0.0;
  for (int result = 0; result < kResults - kVariables; ++result) {
    maximum = std::max(maximum, static_cast<double>(errors(0, result)));
  }
  return maximum;
}

template <int NGHOST>
void RunMmsOrder(ParameterInput *pin, Mesh *mesh) {
  const int initial_cycle = mesh->ncycle;
  const Real initial_time = mesh->time;
  MeshBlockPack *pack = mesh->pmb_pack;
  auto &indcs = mesh->mb_indcs;
  auto &size = pack->pmb->mb_size;
  const int n1 = indcs.nx1 + 2 * indcs.ng;
  const int n2 = indcs.nx2 + 2 * indcs.ng;
  const int n3 = 1;
  const int active_per_block = indcs.nx1 * indcs.nx2;
  const int local_cells = pack->nmb_thispack * active_per_block;
  constexpr Real noise_ulps = sizeof(Real) == sizeof(float) ? 2.0 : 64.0;
  const Real noise_amplitude = noise_ulps * std::numeric_limits<Real>::epsilon();
  const int noise_phase = pin->GetOrAddInteger("problem", "noise_phase", 3);
  if (noise_phase < 0 || noise_phase >= 8) {
    if (global_variable::my_rank == 0) {
      std::cerr << "z4c_cartoon_derivatives requires noise_phase in [0,7]\n";
    }
    std::exit(EXIT_FAILURE);
  }

  DvceArray5D<Real> clean = pack->pz4c->u0;
  if (clean.extent_int(0) != pack->nmb_thispack || clean.extent_int(2) != 1 ||
      clean.extent_int(3) != n2 || clean.extent_int(4) != n1 ||
      clean.extent_int(1) < kVariables) {
    if (global_variable::my_rank == 0) {
      std::cerr << "Cartoon MMS requires actual collapsed Z4c u0 storage n3=1\n";
    }
    std::exit(EXIT_FAILURE);
  }
  DvceArray5D<Real> shared_noisy("Cartoon MMS shared parity noise",
                                 pack->nmb_thispack, kVariables, n3, n2, n1);
  DvceArray5D<Real> independent_noisy("Cartoon MMS independent tensor noise",
                                      pack->nmb_thispack, kVariables, n3, n2, n1);
  par_for(
      "fill Cartoon MMS active and ghost storage", DevExeSpace(), 0,
      pack->nmb_thispack - 1, 0, 0, 0, n2 - 1, 0, n1 - 1,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        const Real rho = CellCenterX(i - indcs.is, indcs.nx1,
                                     size.d_view(m).x1min, size.d_view(m).x1max);
        const Real z = CellCenterX(j - indcs.js, indcs.nx2,
                                   size.d_view(m).x2min, size.d_view(m).x2max);
        z4c_mms::FieldValues oracle;
        z4c_mms::EvaluateFieldValues(rho, 0.0, z, oracle);
        const int radial_layer = static_cast<int>(floor(fabs(rho / size.d_view(m).dx1)));
        const int axial_index = static_cast<int>(llround(z / size.d_view(m).dx2));
        StoreManufacturedState(oracle, rho, radial_layer, axial_index,
                               0.0, noise_phase, 0, m, k, j, i, clean);
        StoreManufacturedState(oracle, rho, radial_layer, axial_index,
                               noise_amplitude, noise_phase,
                               1, m, k, j, i, shared_noisy);
        StoreManufacturedState(oracle, rho, radial_layer, axial_index,
                               noise_amplitude, noise_phase,
                               2, m, k, j, i, independent_noisy);
      });

  DvceArray2D<Real> clean_errors("Cartoon MMS clean errors", local_cells, kResults);
  DvceArray2D<Real> clean_values("Cartoon MMS clean values", local_cells, kResults);
  DvceArray2D<Real> shared_values("Cartoon MMS shared noisy values", local_cells, kResults);
  DvceArray2D<Real> independent_values("Cartoon MMS independent noisy values",
                                       local_cells, kResults);
  DvceArray2D<Real> shared_errors("Cartoon MMS shared noisy errors", local_cells, kResults);
  DvceArray2D<Real> independent_errors("Cartoon MMS independent noisy errors",
                                       local_cells, kResults);
  DvceArray2D<Real> shared_deltas("Cartoon MMS shared noisy deltas", local_cells, kResults);
  DvceArray2D<Real> independent_deltas("Cartoon MMS independent noisy deltas",
                                       local_cells, kResults);
  par_for(
      "evaluate production Cartoon MMS provider", DevExeSpace(), 0,
      pack->nmb_thispack - 1, indcs.js, indcs.je, indcs.is, indcs.ie,
      KOKKOS_LAMBDA(const int m, const int j, const int i) {
        const int local = m * active_per_block +
                          (j - indcs.js) * indcs.nx1 + (i - indcs.is);
        const Real rho = CellCenterX(i - indcs.is, indcs.nx1,
                                     size.d_view(m).x1min, size.d_view(m).x1max);
        const Real z = CellCenterX(j - indcs.js, indcs.nx2,
                                   size.d_view(m).x2min, size.d_view(m).x2max);
        const Real inverse_spacing[3] = {1.0 / size.d_view(m).dx1,
                                         1.0 / size.d_view(m).dx2,
                                         1.0 / size.d_view(m).dx3};
        ScalarField clean_scalar{clean};
        VectorField clean_vector{clean};
        TensorField clean_tensor{clean};
        ScalarField shared_scalar{shared_noisy};
        VectorField shared_vector{shared_noisy};
        TensorField shared_tensor{shared_noisy};
        ScalarField independent_scalar{independent_noisy};
        VectorField independent_vector{independent_noisy};
        TensorField independent_tensor{independent_noisy};
        z4c::DerivativeProvider<z4c::CartoonSO2, NGHOST> derivative(
            inverse_spacing, rho, z4c::CartoonAxisLocation::cell_centered,
            m, 0, j, i);
        z4c_mms::FieldValues fields;
        z4c_mms::EvaluateFieldValues(rho, 0.0, z, fields);
        EvaluateAndStorePoint(
            derivative, clean_scalar, clean_vector, clean_tensor, shared_scalar,
            shared_vector, shared_tensor, independent_scalar, independent_vector,
            independent_tensor, clean, shared_noisy, independent_noisy, fields,
            rho, z, size.d_view(m).dx1, size.d_view(m).dx2, local, clean_values,
            shared_values, independent_values, clean_errors, shared_errors,
            independent_errors, shared_deltas, independent_deltas);
      });
  Kokkos::fence();

  auto clean_host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), clean_errors);
  auto values_host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), clean_values);
  auto shared_values_host =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), shared_values);
  auto independent_values_host =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), independent_values);
  auto shared_errors_host =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), shared_errors);
  auto independent_errors_host =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), independent_errors);
  auto shared_deltas_host =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), shared_deltas);
  auto independent_deltas_host =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), independent_deltas);
  std::vector<NormRecord> full(kResults);
  std::vector<NormRecord> region(kResults * kRegions);
  std::vector<NormRecord> shared_full(kResults);
  std::vector<NormRecord> shared_region(kResults * kRegions);
  std::vector<NormRecord> shared_delta_full(kResults);
  std::vector<NormRecord> shared_delta_region(kResults * kRegions);
  std::vector<NormRecord> independent_full(kResults);
  std::vector<NormRecord> independent_region(kResults * kRegions);
  std::vector<NormRecord> independent_delta_full(kResults);
  std::vector<NormRecord> independent_delta_region(kResults * kRegions);
  std::vector<long long> local_owned_ids;
  local_owned_ids.reserve(local_cells);
  long long owned_cells = local_cells;
  for (int m = 0; m < pack->nmb_thispack; ++m) {
    const Real dx1 = size.h_view(m).dx1;
    const Real dx2 = size.h_view(m).dx2;
    for (int j = indcs.js; j <= indcs.je; ++j) {
      const Real z = CellCenterX(j - indcs.js, indcs.nx2,
                                 size.h_view(m).x2min, size.h_view(m).x2max);
      (void)z;
      for (int i = indcs.is; i <= indcs.ie; ++i) {
        const int local = m * active_per_block +
                          (j - indcs.js) * indcs.nx1 + (i - indcs.is);
        const Real rho = CellCenterX(i - indcs.is, indcs.nx1,
                                     size.h_view(m).x1min, size.h_view(m).x1max);
        const long long global_i = std::llround(
            (rho - mesh->mesh_size.x1min) / mesh->mesh_size.dx1 - 0.5);
        const long long global_j = std::llround(
            (z - mesh->mesh_size.x2min) / mesh->mesh_size.dx2 - 0.5);
        local_owned_ids.push_back(global_j * mesh->mesh_indcs.nx1 + global_i);
        const int layer = static_cast<int>(std::floor(std::abs(rho / dx1)));
        const int side = rho > 0.0 ? 1 : 0;
        int region_index = -1;
        if (layer < NGHOST) {
          region_index = 2 + 2 * layer + side;
        } else if (layer == NGHOST) {
          region_index = 10 + side;
        } else if (std::abs(rho) >= 0.75) {
          region_index = side;
        }
        for (int result = 0; result < kResults; ++result) {
          const double error = clean_host(local, result);
          const double shared_error = shared_errors_host(local, result);
          const double independent_error = independent_errors_host(local, result);
          const double shared_delta = shared_deltas_host(local, result);
          const double independent_delta = independent_deltas_host(local, result);
          auto accumulate = [&](NormRecord &record, const double value) {
            if (!std::isfinite(value)) {
              ++record.nonfinite;
              return;
            }
            record.sum_abs += std::abs(value);
            record.sum_square += value * value;
            record.maximum = std::max(record.maximum, std::abs(value));
            ++record.count;
            record.mask_xor ^= HashCellId(static_cast<std::uint64_t>(
                global_j * mesh->mesh_indcs.nx1 + global_i));
            if (rho > 0.0) {
              const double volume = 2.0 * M_PI * rho * dx1 * dx2;
              record.cylindrical_abs += volume * std::abs(value);
              record.cylindrical_square += volume * value * value;
              record.cylindrical_volume += volume;
              record.cylindrical_maximum =
                  std::max(record.cylindrical_maximum, std::abs(value));
              ++record.cylindrical_count;
            }
          };
          accumulate(full[result], error);
          if (region_index >= 0) {
            accumulate(region[result * kRegions + region_index], error);
          }
          accumulate(shared_full[result], shared_error);
          if (region_index >= 0) {
            accumulate(shared_region[result * kRegions + region_index], shared_error);
          }
          accumulate(shared_delta_full[result], shared_delta);
          if (region_index >= 0) {
            accumulate(shared_delta_region[result * kRegions + region_index], shared_delta);
          }
          accumulate(independent_full[result], independent_error);
          if (region_index >= 0) {
            accumulate(independent_region[result * kRegions + region_index],
                       independent_error);
          }
          accumulate(independent_delta_full[result], independent_delta);
          if (region_index >= 0) {
            accumulate(independent_delta_region[result * kRegions + region_index],
                       independent_delta);
          }
          const int fixed_layer = static_cast<int>(std::floor(0.5 / dx1));
          if (layer == fixed_layer) {
            accumulate(region[result * kRegions + 12 + side], error);
            accumulate(shared_region[result * kRegions + 12 + side], shared_error);
            accumulate(shared_delta_region[result * kRegions + 12 + side], shared_delta);
            accumulate(independent_region[result * kRegions + 12 + side],
                       independent_error);
            accumulate(independent_delta_region[result * kRegions + 12 + side],
                       independent_delta);
          }
        }
      }
    }
  }

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &owned_cells, 1, MPI_LONG_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  auto reduce_records = [](std::vector<NormRecord> &records) {
    std::vector<double> sums(records.size() * 5);
    std::vector<double> maxima(records.size() * 2);
    std::vector<long long> counts(records.size() * 3);
    std::vector<unsigned long long> mask_xors(records.size());
    for (std::size_t i = 0; i < records.size(); ++i) {
      sums[5 * i] = records[i].sum_abs;
      sums[5 * i + 1] = records[i].sum_square;
      sums[5 * i + 2] = records[i].cylindrical_abs;
      sums[5 * i + 3] = records[i].cylindrical_square;
      sums[5 * i + 4] = records[i].cylindrical_volume;
      maxima[2 * i] = records[i].maximum;
      maxima[2 * i + 1] = records[i].cylindrical_maximum;
      counts[3 * i] = records[i].count;
      counts[3 * i + 1] = records[i].cylindrical_count;
      counts[3 * i + 2] = records[i].nonfinite;
      mask_xors[i] = records[i].mask_xor;
    }
    MPI_Allreduce(MPI_IN_PLACE, sums.data(), static_cast<int>(sums.size()),
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, maxima.data(), static_cast<int>(maxima.size()),
                  MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, counts.data(), static_cast<int>(counts.size()),
                  MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, mask_xors.data(), static_cast<int>(mask_xors.size()),
                  MPI_UNSIGNED_LONG_LONG, MPI_BXOR, MPI_COMM_WORLD);
    for (std::size_t i = 0; i < records.size(); ++i) {
      records[i].sum_abs = sums[5 * i];
      records[i].sum_square = sums[5 * i + 1];
      records[i].cylindrical_abs = sums[5 * i + 2];
      records[i].cylindrical_square = sums[5 * i + 3];
      records[i].cylindrical_volume = sums[5 * i + 4];
      records[i].maximum = maxima[2 * i];
      records[i].cylindrical_maximum = maxima[2 * i + 1];
      records[i].count = counts[3 * i];
      records[i].cylindrical_count = counts[3 * i + 1];
      records[i].nonfinite = counts[3 * i + 2];
      records[i].mask_xor = mask_xors[i];
    }
  };
  reduce_records(full);
  reduce_records(region);
  reduce_records(shared_full);
  reduce_records(shared_region);
  reduce_records(shared_delta_full);
  reduce_records(shared_delta_region);
  reduce_records(independent_full);
  reduce_records(independent_region);
  reduce_records(independent_delta_full);
  reduce_records(independent_delta_region);
#endif

  const long long expected_cells =
      static_cast<long long>(mesh->mesh_indcs.nx1) * mesh->mesh_indcs.nx2;
  std::vector<long long> all_owned_ids;
  std::vector<Real> local_values(local_cells * kResults);
  std::vector<Real> local_errors(local_cells * kResults);
  for (int local = 0; local < local_cells; ++local) {
    for (int result = 0; result < kResults; ++result) {
      local_values[local * kResults + result] = values_host(local, result);
      local_errors[local * kResults + result] = clean_host(local, result);
    }
  }
  std::vector<Real> gathered_values;
  std::vector<Real> gathered_errors;
#if MPI_PARALLEL_ENABLED
  std::vector<int> ownership_counts(global_variable::nranks);
  const int local_owned_count = static_cast<int>(local_owned_ids.size());
  MPI_Allgather(&local_owned_count, 1, MPI_INT, ownership_counts.data(), 1,
                MPI_INT, MPI_COMM_WORLD);
  std::vector<int> ownership_offsets(global_variable::nranks, 0);
  for (int rank = 1; rank < global_variable::nranks; ++rank) {
    ownership_offsets[rank] = ownership_offsets[rank - 1] + ownership_counts[rank - 1];
  }
  all_owned_ids.resize(owned_cells);
  MPI_Allgatherv(local_owned_ids.data(), local_owned_count, MPI_LONG_LONG,
                 all_owned_ids.data(), ownership_counts.data(), ownership_offsets.data(),
                 MPI_LONG_LONG, MPI_COMM_WORLD);
  std::vector<int> value_counts(global_variable::nranks);
  std::vector<int> value_offsets(global_variable::nranks);
  for (int rank = 0; rank < global_variable::nranks; ++rank) {
    value_counts[rank] = ownership_counts[rank] * kResults;
    value_offsets[rank] = ownership_offsets[rank] * kResults;
  }
  gathered_values.resize(owned_cells * kResults);
  gathered_errors.resize(owned_cells * kResults);
  MPI_Allgatherv(local_values.data(), local_owned_count * kResults, MPI_ATHENA_REAL,
                 gathered_values.data(), value_counts.data(), value_offsets.data(),
                 MPI_ATHENA_REAL, MPI_COMM_WORLD);
  MPI_Allgatherv(local_errors.data(), local_owned_count * kResults, MPI_ATHENA_REAL,
                 gathered_errors.data(), value_counts.data(), value_offsets.data(),
                 MPI_ATHENA_REAL, MPI_COMM_WORLD);
#else
  all_owned_ids = local_owned_ids;
  gathered_values = local_values;
  gathered_errors = local_errors;
#endif
  const std::vector<long long> gathered_owned_ids = all_owned_ids;
  std::sort(all_owned_ids.begin(), all_owned_ids.end());
  bool ownership_valid = owned_cells == expected_cells &&
                         static_cast<long long>(all_owned_ids.size()) == expected_cells;
  std::uint64_t ownership_hash = 1469598103934665603ULL;
  for (long long id = 0; id < expected_cells && ownership_valid; ++id) {
    ownership_valid = all_owned_ids[id] == id;
    ownership_hash ^= static_cast<std::uint64_t>(all_owned_ids[id]);
    ownership_hash *= 1099511628211ULL;
  }
  if (!ownership_valid) {
    if (global_variable::my_rank == 0) {
      std::cerr << "Cartoon MMS ownership mismatch: included=" << owned_cells
                << " expected=" << expected_cells << '\n';
    }
    std::exit(EXIT_FAILURE);
  }
  std::vector<Real> ordered_values(expected_cells * kResults);
  std::vector<Real> ordered_errors(expected_cells * kResults);
  for (long long entry = 0; entry < expected_cells; ++entry) {
    const long long id = gathered_owned_ids[entry];
    for (int result = 0; result < kResults; ++result) {
      ordered_values[id * kResults + result] =
          gathered_values[entry * kResults + result];
      ordered_errors[id * kResults + result] =
          gathered_errors[entry * kResults + result];
    }
  }
  const std::vector<int> result_parities = ResultParities();
  std::vector<double> rotation_residual(kResults, 0.0);
  for (int global_j = 0; global_j < mesh->mesh_indcs.nx2; ++global_j) {
    for (int global_i = 0; global_i < mesh->mesh_indcs.nx1 / 2; ++global_i) {
      const long long negative_id =
          static_cast<long long>(global_j) * mesh->mesh_indcs.nx1 + global_i;
      const long long positive_id =
          static_cast<long long>(global_j) * mesh->mesh_indcs.nx1 +
          (mesh->mesh_indcs.nx1 - 1 - global_i);
      for (int result = 0; result < kResults; ++result) {
        const double residual = std::abs(
            ordered_values[negative_id * kResults + result] -
            result_parities[result] * ordered_values[positive_id * kResults + result]);
        rotation_residual[result] = std::max(rotation_residual[result], residual);
      }
    }
  }
  const double axis_error = RunDiagnosticAxisProbe<NGHOST>();
  double global_axis_error = axis_error;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &global_axis_error, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);
#endif

  long long nonfinite = 0;
  double maximum_error = 0.0;
  double maximum_noise_delta = 0.0;
  for (int result = 0; result < kResults; ++result) {
    nonfinite += full[result].nonfinite + shared_full[result].nonfinite +
                 shared_delta_full[result].nonfinite +
                 independent_full[result].nonfinite +
                 independent_delta_full[result].nonfinite;
    maximum_error = std::max(maximum_error, full[result].maximum);
    maximum_noise_delta = std::max(
        {maximum_noise_delta, shared_delta_full[result].maximum,
         independent_delta_full[result].maximum});
  }
  const double minimum_spacing = std::min(mesh->mesh_size.dx1, mesh->mesh_size.dx2);
  const double noise_bound = NoiseCoefficientSafety<NGHOST>() * noise_amplitude /
                             (minimum_spacing * minimum_spacing);
  const double rotation_bound = NoiseCoefficientSafety<NGHOST>() *
                                std::numeric_limits<Real>::epsilon() /
                                (minimum_spacing * minimum_spacing);
  const double maximum_rotation_residual =
      *std::max_element(rotation_residual.begin(), rotation_residual.end());
  constexpr double axis_tolerance =
      NGHOST == 2 ? 2.0e-2 : (NGHOST == 3 ? 2.0e-4 : 2.0e-6);
  bool failed = !ownership_valid || nonfinite != 0 ||
                      maximum_noise_delta > noise_bound ||
                      maximum_rotation_residual > rotation_bound ||
                      !std::isfinite(global_axis_error) ||
                      global_axis_error > axis_tolerance ||
                      mesh->ncycle != initial_cycle || mesh->time != initial_time;

  int io_failure = 0;
  if (global_variable::my_rank == 0) {
    const std::vector<std::string> names = ResultNames();
    if (names.size() != kResults || result_parities.size() != kResults) {
      std::cerr << "Cartoon MMS internal result-name count mismatch\n";
      std::exit(EXIT_FAILURE);
    }
    const std::string basename = pin->GetString("job", "basename");
    const std::filesystem::path output_directory =
        pin->GetOrAddString("problem", "output_directory", ".");
    std::error_code directory_error;
    std::filesystem::create_directories(output_directory, directory_error);
    if (directory_error) io_failure = 1;
    const auto csv_path = output_directory / (basename + ".mms.csv");
    const auto probes_path = output_directory / (basename + ".mms.probes.csv");
    const auto json_path = output_directory / (basename + ".mms.json");
    std::ofstream csv(csv_path);
    if (!csv.is_open()) io_failure = 1;
    csv << "operator,mask,count,nonfinite,l1,l2,linfinity,cyl_count,cyl_l1,"
           "cyl_l2,cyl_linfinity,shared_l1,shared_l2,shared_linfinity,"
           "shared_delta_l1,shared_delta_l2,shared_delta_linfinity,"
           "independent_l1,independent_l2,independent_linfinity,"
           "independent_delta_l1,independent_delta_l2,"
           "independent_delta_linfinity,rotation_linfinity,target_abs_rho,"
           "actual_abs_rho,mask_xor\n";
    const char *region_names[kRegions] = {
        "regular_negative", "regular_positive",
        "fitted_layer_0_negative", "fitted_layer_0_positive",
        "fitted_layer_1_negative", "fitted_layer_1_positive",
        "fitted_layer_2_negative", "fitted_layer_2_positive",
        "fitted_layer_3_negative", "fitted_layer_3_positive",
        "raw_transition_negative", "raw_transition_positive",
        "fixed_rho_negative_0.5", "fixed_rho_positive_0.5"};
    auto emit = [&](const int result, const char *mask, const NormRecord &record,
                    const NormRecord &shared_record,
                    const NormRecord &shared_delta_record,
                    const NormRecord &independent_record,
                    const NormRecord &independent_delta_record) {
      const double l1 = record.count > 0 ? record.sum_abs / record.count : 0.0;
      const double l2 = record.count > 0
                            ? std::sqrt(record.sum_square / record.count)
                            : 0.0;
      const double cyl_l1 = record.cylindrical_volume > 0.0
                                ? record.cylindrical_abs / record.cylindrical_volume
                                : 0.0;
      const double cyl_l2 = record.cylindrical_volume > 0.0
                                ? std::sqrt(record.cylindrical_square /
                                            record.cylindrical_volume)
                                : 0.0;
      auto norm_l1 = [](const NormRecord &value) {
        return value.count > 0 ? value.sum_abs / value.count : 0.0;
      };
      auto norm_l2 = [](const NormRecord &value) {
        return value.count > 0 ? std::sqrt(value.sum_square / value.count) : 0.0;
      };
      csv << names[result] << ',' << mask << ',' << record.count << ','
          << (record.nonfinite + shared_record.nonfinite +
              shared_delta_record.nonfinite + independent_record.nonfinite +
              independent_delta_record.nonfinite)
          << ',' << std::setprecision(17) << l1 << ',' << l2
          << ',' << record.maximum << ',' << record.cylindrical_count << ','
          << cyl_l1 << ',' << cyl_l2 << ',' << record.cylindrical_maximum << ','
          << norm_l1(shared_record) << ',' << norm_l2(shared_record) << ','
          << shared_record.maximum << ',' << norm_l1(shared_delta_record) << ','
          << norm_l2(shared_delta_record) << ',' << shared_delta_record.maximum << ','
          << norm_l1(independent_record) << ',' << norm_l2(independent_record) << ','
          << independent_record.maximum << ',' << norm_l1(independent_delta_record)
          << ',' << norm_l2(independent_delta_record) << ','
          << independent_delta_record.maximum << ','
          << rotation_residual[result] << ',';
      if (std::string(mask).find("fixed_rho_") == 0) {
        const int fixed_layer = static_cast<int>(std::floor(0.5 / mesh->mesh_size.dx1));
        csv << 0.5 << ',' << (fixed_layer + 0.5) * mesh->mesh_size.dx1;
      } else {
        csv << "nan,nan";
      }
      csv << ',' << std::hex << record.mask_xor << std::dec << '\n';
    };
    for (int result = 0; result < kResults; ++result) {
      emit(result, "full_signed_plane", full[result], shared_full[result],
           shared_delta_full[result], independent_full[result],
           independent_delta_full[result]);
      for (int region_index = 0; region_index < kRegions; ++region_index) {
        if (region_index >= 2 + 2 * NGHOST && region_index <= 9) continue;
        emit(result, region_names[region_index],
             region[result * kRegions + region_index],
             shared_region[result * kRegions + region_index],
             shared_delta_region[result * kRegions + region_index],
             independent_region[result * kRegions + region_index],
             independent_delta_region[result * kRegions + region_index]);
      }
    }
    csv.close();
    if (!csv) io_failure = 1;
    std::ofstream probes(probes_path);
    if (!probes.is_open()) io_failure = 1;
    probes << "operator,mask,side,layer_index,classification,target_rho,"
              "actual_rho,target_z,actual_z,global_cell_id,raw_error\n";
    const int global_center = mesh->mesh_indcs.nx1 / 2;
    const int probe_j = mesh->mesh_indcs.nx2 / 2;
    const double actual_z = mesh->mesh_size.x2min +
                            (probe_j + 0.5) * mesh->mesh_size.dx2;
    for (int region_index = 0; region_index < kRegions; ++region_index) {
      if (region_index >= 2 + 2 * NGHOST && region_index <= 9) continue;
      const int side = region_index % 2;
      int layer = 0;
      const char *classification = "regular";
      double target_rho = side == 0 ? -1.0 : 1.0;
      if (region_index >= 2 && region_index <= 9) {
        layer = (region_index - 2) / 2;
        classification = "fitted";
        target_rho = std::numeric_limits<double>::quiet_NaN();
      } else if (region_index == 10 || region_index == 11) {
        layer = NGHOST;
        classification = "raw_transition";
        target_rho = std::numeric_limits<double>::quiet_NaN();
      } else if (region_index == 12 || region_index == 13) {
        layer = static_cast<int>(std::floor(0.5 / mesh->mesh_size.dx1));
        classification = "fixed_radius";
        target_rho = side == 0 ? -0.5 : 0.5;
      } else {
        layer = static_cast<int>(std::floor(1.0 / mesh->mesh_size.dx1));
      }
      const int probe_i = side == 0 ? global_center - 1 - layer
                                    : global_center + layer;
      const long long global_id =
          static_cast<long long>(probe_j) * mesh->mesh_indcs.nx1 + probe_i;
      const double actual_rho = mesh->mesh_size.x1min +
                                (probe_i + 0.5) * mesh->mesh_size.dx1;
      for (int result = 0; result < kResults; ++result) {
        probes << names[result] << ',' << region_names[region_index] << ','
               << (side == 0 ? "negative" : "positive") << ',' << layer << ','
               << classification << ',' << std::setprecision(17) << target_rho << ','
               << actual_rho << ',' << 0.0 << ',' << actual_z << ',' << global_id << ','
               << ordered_errors[global_id * kResults + result] << '\n';
      }
    }
    probes.close();
    if (!probes) io_failure = 1;
    std::ofstream json(json_path);
    if (!json.is_open()) io_failure = 1;
    json << std::setprecision(17)
         << "{\n  \"schema\": \"athenak_z4c_cartoon_derivative_mms_v1\",\n"
         << "  \"status\": \"" << (failed ? "fail" : "pass") << "\",\n"
         << "  \"backend\": \"" << Kokkos::DefaultExecutionSpace::name()
         << "\",\n  \"spatial_order\": " << 2 * (NGHOST - 1)
         << ",\n  \"nghost\": " << NGHOST
         << ",\n  \"nx1\": " << mesh->mesh_indcs.nx1
         << ",\n  \"nx2\": " << mesh->mesh_indcs.nx2
         << ",\n  \"nx3\": " << mesh->mesh_indcs.nx3
         << ",\n  \"mpi_ranks\": " << global_variable::nranks
         << ",\n  \"initial_cycle\": " << initial_cycle
         << ",\n  \"initial_time\": " << initial_time
         << ",\n  \"pgen_final_cycle\": " << mesh->ncycle
         << ",\n  \"pgen_final_time\": " << mesh->time
         << ",\n  \"owned_cells\": " << owned_cells
         << ",\n  \"ownership_sequence\": \"[0,N*N) exactly once\""
         << ",\n  \"ownership_fnv1a64\": \"" << std::hex << ownership_hash << std::dec
         << "\""
         << ",\n  \"operator_count\": " << kResults
         << ",\n  \"operator_names\": [";
    for (int result = 0; result < kResults; ++result) {
      if (result != 0) json << ',';
      json << "\n    \"" << names[result] << "\"";
    }
    json << "\n  ]"
         << ",\n  \"noise_amplitude\": " << noise_amplitude
         << ",\n  \"maximum_error\": " << maximum_error
         << ",\n  \"maximum_noise_delta\": " << maximum_noise_delta
         << ",\n  \"noise_delta_bound\": " << noise_bound
         << ",\n  \"maximum_rotation_residual\": " << maximum_rotation_residual
         << ",\n  \"rotation_residual_bound\": " << rotation_bound
         << ",\n  \"diagnostic_axis_linf\": " << global_axis_error
         << ",\n  \"diagnostic_axis_tolerance\": " << axis_tolerance
         << ",\n  \"nonfinite_count\": " << nonfinite
         << ",\n  \"cell_mask\": \"each active cell exactly once; cylindrical rho>0 only\",\n"
         << "  \"csv\": \"" << csv_path.filename().string() << "\",\n"
         << "  \"probes_csv\": \"" << probes_path.filename().string()
         << "\"\n}\n";
    json.close();
    if (!json) io_failure = 1;
    if (failed) {
      std::cerr << "Cartoon MMS failed: nonfinite=" << nonfinite
                << " noise_delta=" << maximum_noise_delta
                << " noise_bound=" << noise_bound
                << " rotation_residual=" << maximum_rotation_residual
                << " rotation_bound=" << rotation_bound
                << " axis_error=" << global_axis_error
                << " axis_tolerance=" << axis_tolerance << '\n';
    }
    if (!failed) {
      std::cout << "Cartoon derivative MMS passed: order=" << 2 * (NGHOST - 1)
                << " cells=" << owned_cells << " max_error=" << maximum_error
                << " axis_error=" << global_axis_error << '\n';
    }
  }
#if MPI_PARALLEL_ENABLED
  MPI_Bcast(&io_failure, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
  failed = failed || io_failure != 0;
  if (failed) std::exit(EXIT_FAILURE);
}

void InitializeMinkowski(MeshBlockPack *pack) {
  auto &u0 = pack->pz4c->u0;
  const int chi = pack->pz4c->I_Z4C_CHI;
  const int gxx = pack->pz4c->I_Z4C_GXX;
  const int gyy = pack->pz4c->I_Z4C_GYY;
  const int gzz = pack->pz4c->I_Z4C_GZZ;
  const int alpha = pack->pz4c->I_Z4C_ALPHA;
  Kokkos::deep_copy(u0, 0.0);
  const auto bounds = z4c::MakeStoredDomainBounds(pack->pmesh->mb_indcs);
  par_for(
      "initialize Cartoon MMS Minkowski sentinel", DevExeSpace(), 0,
      pack->nmb_thispack - 1, bounds.ks, bounds.ke, bounds.js, bounds.je,
      bounds.is, bounds.ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        u0(m, chi, k, j, i) = 1.0;
        u0(m, gxx, k, j, i) = 1.0;
        u0(m, gyy, k, j, i) = 1.0;
        u0(m, gzz, k, j, i) = 1.0;
        u0(m, alpha, k, j, i) = 1.0;
      });
}

struct MmsDispatchContext {
  ParameterInput *pin;
  Mesh *mesh;
};

void RunMmsOrder2(void *opaque) {
  auto *context = static_cast<MmsDispatchContext *>(opaque);
  RunMmsOrder<2>(context->pin, context->mesh);
}

void RunMmsOrder4(void *opaque) {
  auto *context = static_cast<MmsDispatchContext *>(opaque);
  RunMmsOrder<3>(context->pin, context->mesh);
}

void RunMmsOrder6(void *opaque) {
  auto *context = static_cast<MmsDispatchContext *>(opaque);
  RunMmsOrder<4>(context->pin, context->mesh);
}

}  // namespace

void z4c_mms::RunCartoonDerivativeMms(ParameterInput *pin, Mesh *mesh) {
  MeshBlockPack *pack = mesh->pmb_pack;
  MmsDispatchContext context{pin, mesh};
  z4c::DispatchCartoonZ4cKernel(pack->z4c_symmetry, &context, RunMmsOrder2,
                                RunMmsOrder4, RunMmsOrder6);
  InitializeMinkowski(pack);
}
