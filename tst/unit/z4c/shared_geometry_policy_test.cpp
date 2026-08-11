//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file shared_geometry_policy_test.cpp
//! \brief Shared curvature-algebra parity for Cartesian and collapsed SO(2) policies.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include <Kokkos_Core.hpp>

#include "athena.hpp"
#include "z4c/cartoon_meridional_sampler.hpp"
#include "z4c/cartoon_derivatives.hpp"
#include "z4c/curvature_diagnostics.hpp"
#include "z4c/weyl_tetrad.hpp"

namespace {

KOKKOS_INLINE_FUNCTION
int SymmetricIndex(int first, int second) {
  if (first > second) {
    const int swap = first;
    first = second;
    second = swap;
  }
  return first * 3 - first * (first - 1) / 2 + second - first;
}

struct TensorField {
  DvceArray5D<Real> data;
  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(const int m, const int first, const int second,
                            const int k, const int j, const int i) const {
    return data(m, SymmetricIndex(first, second), k, j, i);
  }
};

struct VectorField {
  DvceArray5D<Real> data;
  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(const int m, const int component, const int k,
                            const int j, const int i) const {
    return data(m, component, k, j, i);
  }
};

struct PointVector {
  Real values[3] = {};
  KOKKOS_INLINE_FUNCTION
  Real &operator()(const int component) { return values[component]; }
};

bool NearlyEqual(Real left, Real right, Real tolerance);

bool CheckMeridionalMeasureAndState() {
  Kokkos::View<Real *> results("Cartoon meridional helper results", 5);
  DvceArray5D<Real> scalar("Cartoon bilinear scalar", 1, 1, 1, 4, 4);
  auto scalar_host = Kokkos::create_mirror_view(scalar);
  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 4; ++i) scalar_host(0, 0, 0, j, i) = 2.0 + 3.0 * i + 5.0 * j;
  }
  Kokkos::deep_copy(scalar, scalar_host);
  z4c::CartoonMeridionalStencil stencil;
  stencil.local_block = 0;
  stencil.k = 0;
  stencil.i0 = 1;
  stencil.j0 = 1;
  stencil.wi = 0.25;
  stencil.wj = 0.75;
  Kokkos::parallel_for(
      "Cartoon meridional helper oracle", Kokkos::RangePolicy<>(0, 1),
      KOKKOS_LAMBDA(const int) {
        results(0) = z4c::Z4cDiagnosticCellMeasure(
            z4c::Z4cSymmetryMode::cartoon_so2, 0.5, 0.5, 0.5, 9.0, 4.0);
        results(1) = z4c::Z4cDiagnosticCellMeasure(
            z4c::Z4cSymmetryMode::cartoon_so2, -0.5, 0.5, 0.5, 9.0, 4.0);
        results(2) = z4c::Z4cDiagnosticCellMeasure(
            z4c::Z4cSymmetryMode::cartesian3d, -0.5, 0.5, 0.25, 2.0, 4.0);
        results(3) = z4c::SampleCartoonMeridionalScalar(scalar, 0, stencil);
        results(4) = z4c::Z4cAggregateConstraintNorm(25.0);
      });
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), results);
  if (!NearlyEqual(host(0), 0.25 * z4c::kCartoonTwoPi, 1.0e-15) ||
      host(1) != 0.0 || !NearlyEqual(host(2), 0.5, 1.0e-15) ||
      !NearlyEqual(host(3), 2.0 + 3.0 * 1.25 + 5.0 * 1.75, 1.0e-15) ||
      host(4) != 5.0) {
    return false;
  }

  z4c::Z4cCentralRestartState state;
  if (!z4c::ValidateZ4cCentralRestartState(state).valid ||
      !z4c::UpdateZ4cCentralRestartState(
           &state, 1.0, 0.25, 3.0, 7, 2, 0, 0.0, false).valid ||
      !z4c::UpdateZ4cCentralRestartState(
           &state, 0.5, 0.125, 4.0, 9, 3, 1, 0.5, false).valid ||
      !NearlyEqual(state.proper_time, 0.375, 1.0e-15) ||
      !z4c::UpdateZ4cCentralRestartState(
           &state, 0.5, 0.125, 4.0, 9, 3, 1, 0.5, true).valid ||
      z4c::UpdateZ4cCentralRestartState(
          &state, 0.4, 0.1, 4.0, 9, 3, 3, 1.0, false).valid) {
    return false;
  }
  z4c::Z4cCentralRestartState uninitialized;
  return !z4c::UpdateZ4cCentralRestartState(
              &uninitialized, 1.0, 0.0, 0.0, 0, 0, 0, 0.0, true).valid &&
         !z4c::UpdateZ4cCentralRestartState(
              &uninitialized, std::numeric_limits<double>::infinity(), 0.0,
              0.0, 0, 0, 0, 0.0, false).valid;
}

bool CheckWeylTetradComponentMap() {
  for (const Real rho : {Real(-0.75), Real(0.75)}) {
    constexpr Real z = 0.4;
    PointVector radial;
    PointVector polar;
    PointVector azimuthal;
    z4c::InitializeWeylTetradSeed<z4c::CartoonSO2>(
        rho, z, 0.0, radial, polar, azimuthal);
    const Real expected[9] = {
        rho, z, 0.0, rho * z, -rho * rho, 0.0, 0.0, 0.0, rho};
    for (int component = 0; component < 3; ++component) {
      if (radial(component) != expected[component]) return false;
      if (polar(component) != expected[3 + component]) return false;
      if (azimuthal(component) != expected[6 + component]) return false;
    }
    Real radial_polar = 0.0;
    Real radial_azimuthal = 0.0;
    Real polar_azimuthal = 0.0;
    for (int component = 0; component < 3; ++component) {
      radial_polar += radial(component) * polar(component);
      radial_azimuthal += radial(component) * azimuthal(component);
      polar_azimuthal += polar(component) * azimuthal(component);
    }
    if (!NearlyEqual(radial_polar, 0.0, 1.0e-15) ||
        !NearlyEqual(radial_azimuthal, 0.0, 1.0e-15) ||
        !NearlyEqual(polar_azimuthal, 0.0, 1.0e-15)) {
      return false;
    }
  }
  return true;
}

bool CheckWeylCoordinatePolicy() {
  Kokkos::View<Real *> coordinates("Weyl x3 coordinate policy", 2);
  Kokkos::parallel_for(
      "Weyl x3 coordinate policy", Kokkos::RangePolicy<>(0, 1),
      KOKKOS_LAMBDA(const int) {
        coordinates(0) = z4c::WeylX3Coordinate<z4c::CartoonSO2>(
            37, 11, 1, -9.0, 13.0);
        coordinates(1) = z4c::WeylX3Coordinate<z4c::Cartesian3D>(
            3, 1, 4, -2.0, 2.0);
      });
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coordinates);
  return host(0) == 0.0 && host(1) == 0.5;
}

void FillAxisymmetricTensorPair(DvceArray5D<Real> metric,
                                DvceArray5D<Real> extrinsic,
                                const bool minkowski, const Real spacing,
                                const int radial_axis, const int z_center,
                                const int suppressed_center) {
  auto metric_host = Kokkos::create_mirror_view(metric);
  auto extrinsic_host = Kokkos::create_mirror_view(extrinsic);
  for (int k = 0; k < metric.extent_int(2); ++k) {
    const Real y = (k - suppressed_center) * spacing;
    for (int j = 0; j < metric.extent_int(3); ++j) {
      const Real z = (j - z_center) * spacing;
      for (int i = 0; i < metric.extent_int(4); ++i) {
        const Real x = (i - radial_axis + 0.5) * spacing;
        const Real radius2 = x * x + y * y;
        const Real common = radius2 + z * z;
        Real g[3][3] = {};
        Real K[3][3] = {};
        g[0][0] = g[1][1] = g[2][2] = 1.0;
        if (!minkowski) {
          // Component order is (X,Z,Y)=(x1,x2,x3).  These regular polynomial
          // tensors carry nontrivial diagonal, off-diagonal, radial, axial, and
          // suppressed components; a scalar-only or unpermuted policy cannot pass.
          g[0][0] += 0.010 * common + 0.006 * x * x;
          g[2][2] += 0.010 * common + 0.006 * y * y;
          g[0][2] = g[2][0] = 0.006 * x * y;
          g[0][1] = g[1][0] = 0.004 * x * z;
          g[2][1] = g[1][2] = 0.004 * y * z;
          g[1][1] += 0.008 * radius2 + 0.005 * z * z;

          K[0][0] = 0.003 * common + 0.002 * x * x;
          K[2][2] = 0.003 * common + 0.002 * y * y;
          K[0][2] = K[2][0] = 0.002 * x * y;
          K[0][1] = K[1][0] = -0.0015 * x * z;
          K[2][1] = K[1][2] = -0.0015 * y * z;
          K[1][1] = -0.002 * radius2 + 0.001 * z * z;
        }
        for (int a = 0; a < 3; ++a) {
          for (int b = a; b < 3; ++b) {
            const int component = SymmetricIndex(a, b);
            metric_host(0, component, k, j, i) = g[a][b];
            extrinsic_host(0, component, k, j, i) = K[a][b];
          }
        }
      }
    }
  }
  Kokkos::deep_copy(metric, metric_host);
  Kokkos::deep_copy(extrinsic, extrinsic_host);
}

void FillAxisymmetricVector(DvceArray5D<Real> vector, const Real spacing,
                            const int radial_axis, const int z_center,
                            const int suppressed_center, const int family) {
  auto host = Kokkos::create_mirror_view(vector);
  for (int k = 0; k < vector.extent_int(2); ++k) {
    const Real y = (k - suppressed_center) * spacing;
    for (int j = 0; j < vector.extent_int(3); ++j) {
      const Real z = (j - z_center) * spacing;
      for (int i = 0; i < vector.extent_int(4); ++i) {
        const Real x = (i - radial_axis + 0.5) * spacing;
        const Real radius2 = x * x + y * y;
        // Every family is a different regular SO(2) Cartesian vector field.  Under
        // rotations a lower Euclidean covector has the same component map, but the
        // independent coefficients prevent a beta/Gamma field from accidentally being
        // substituted for the telegraph-lapse B_i field.
        const Real a = 0.11 * (family + 1) - 0.02 * z;
        const Real b = -0.07 * (family + 2) + 0.015 * z;
        const Real c = 0.05 * (family + 3) - 0.025 * radius2 + 0.02 * z * z;
        host(0, 0, k, j, i) = a * x - b * y;
        host(0, 1, k, j, i) = c;
        host(0, 2, k, j, i) = a * y + b * x;
      }
    }
  }
  Kokkos::deep_copy(vector, host);
}

bool NearlyEqual(const Real left, const Real right, const Real tolerance) {
  const Real scale = std::max({Real(1.0), std::abs(left), std::abs(right)});
  return std::abs(left - right) <= tolerance * scale;
}

template <int NGHOST>
bool CheckGaugeVectorFamilies() {
  constexpr int radial_cells = 24;
  constexpr int z_cells = 16;
  constexpr Real spacing = 0.125;
  constexpr int families = 3;  // beta^i, Gamma^i, and the independent B_i covector.
  constexpr int samples = 2;
  constexpr int results_per_family = 30;
  const int radial_axis = NGHOST + radial_cells / 2;
  const int z_center = NGHOST + z_cells / 2;
  const int cartesian_k_center = NGHOST;
  const int n1 = radial_cells + 2 * NGHOST;
  const int n2 = z_cells + 2 * NGHOST;
  const int n3 = 2 * NGHOST + 1;
  DvceArray5D<Real> collapsed[families] = {
      DvceArray5D<Real>("collapsed beta upper", 1, 3, 1, n2, n1),
      DvceArray5D<Real>("collapsed Gamma upper", 1, 3, 1, n2, n1),
      DvceArray5D<Real>("collapsed B lower", 1, 3, 1, n2, n1)};
  DvceArray5D<Real> full[families] = {
      DvceArray5D<Real>("cartesian beta upper", 1, 3, n3, n2, n1),
      DvceArray5D<Real>("cartesian Gamma upper", 1, 3, n3, n2, n1),
      DvceArray5D<Real>("cartesian B lower", 1, 3, n3, n2, n1)};
  for (int family = 0; family < families; ++family) {
    FillAxisymmetricVector(collapsed[family], spacing, radial_axis, z_center, 0,
                           family);
    FillAxisymmetricVector(full[family], spacing, radial_axis, z_center,
                           cartesian_k_center, family);
  }

  DvceArray2D<Real> differences("gauge vector policy differences", samples,
                                families * results_per_family);
  auto sample_indices = Kokkos::View<int*>("gauge sample indices", samples);
  auto sample_host = Kokkos::create_mirror_view(sample_indices);
  sample_host(0) = radial_axis + NGHOST;
  sample_host(1) = radial_axis - NGHOST - 1;
  Kokkos::deep_copy(sample_indices, sample_host);
  VectorField collapsed_beta{collapsed[0]};
  VectorField collapsed_Gamma{collapsed[1]};
  VectorField collapsed_B{collapsed[2]};
  VectorField full_beta{full[0]};
  VectorField full_Gamma{full[1]};
  VectorField full_B{full[2]};
  Kokkos::parallel_for(
      "gauge vector family policy comparison", Kokkos::RangePolicy<>(0, samples),
      KOKKOS_LAMBDA(const int sample) {
        const int i = sample_indices(sample);
        const int j = z_center + 2;
        const Real rho = (i - radial_axis + 0.5) * spacing;
        const Real inverse_spacing[3] = {1.0 / spacing, 1.0 / spacing,
                                         1.0 / spacing};
        z4c::DerivativeProvider<z4c::CartoonSO2, NGHOST> cartoon(
            inverse_spacing, rho, z4c::CartoonAxisLocation::cell_centered,
            0, 0, j, i);
        z4c::DerivativeProvider<z4c::Cartesian3D, NGHOST> cartesian(
            inverse_spacing, 0, cartesian_k_center, j, i);
        const VectorField collapsed_fields[families] = {
            collapsed_beta, collapsed_Gamma, collapsed_B};
        const VectorField full_fields[families] = {full_beta, full_Gamma, full_B};
        for (int family = 0; family < families; ++family) {
          int output = family * results_per_family;
          for (int direction = 0; direction < 3; ++direction) {
            for (int component = 0; component < 3; ++component) {
              differences(sample, output++) =
                  cartoon.VectorFirst(direction, component, collapsed_fields[family]) -
                  cartesian.VectorFirst(direction, component, full_fields[family]);
            }
          }
          for (int first = 0; first < 3; ++first) {
            for (int second = first; second < 3; ++second) {
              for (int component = 0; component < 3; ++component) {
                differences(sample, output++) =
                    cartoon.VectorSecond(first, second, component,
                                           collapsed_fields[family]) -
                    cartesian.VectorSecond(first, second, component,
                                            full_fields[family]);
              }
            }
          }
          differences(sample, output++) =
              cartoon.VectorAdvective(0, collapsed_beta,
                                        collapsed_fields[family]) -
              cartesian.VectorAdvective(0, full_beta, full_fields[family]);
          differences(sample, output++) =
              cartoon.VectorAdvective(1, collapsed_beta,
                                        collapsed_fields[family]) -
              cartesian.VectorAdvective(1, full_beta, full_fields[family]);
          differences(sample, output++) =
              cartoon.VectorAdvective(2, collapsed_beta,
                                        collapsed_fields[family]) -
              cartesian.VectorAdvective(2, full_beta, full_fields[family]);
        }
      });
  Kokkos::fence();
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), differences);
  for (int sample = 0; sample < samples; ++sample) {
    for (int result = 0; result < families * results_per_family; ++result) {
      if (!NearlyEqual(host(sample, result), 0.0, 2.0e-11)) {
        std::cerr << "NGHOST=" << NGHOST << " gauge sample=" << sample
                  << " result=" << result << " difference="
                  << host(sample, result) << "\n";
        return false;
      }
    }
  }
  return true;
}

template <int NGHOST>
bool CheckSharedGeometry(const bool minkowski) {
  constexpr int radial_cells = 24;
  constexpr int z_cells = 16;
  constexpr Real spacing = 0.125;
  const int radial_axis = NGHOST + radial_cells / 2;
  const int z_center = NGHOST + z_cells / 2;
  const int cartesian_k_center = NGHOST;
  const int n1 = radial_cells + 2 * NGHOST;
  const int n2 = z_cells + 2 * NGHOST;
  const int n3 = 2 * NGHOST + 1;

  DvceArray5D<Real> cartoon_metric("collapsed metric", 1, 6, 1, n2, n1);
  DvceArray5D<Real> cartoon_extrinsic("collapsed extrinsic", 1, 6, 1, n2, n1);
  DvceArray5D<Real> cartesian_metric("cartesian metric", 1, 6, n3, n2, n1);
  DvceArray5D<Real> cartesian_extrinsic("cartesian extrinsic", 1, 6, n3, n2, n1);
  FillAxisymmetricTensorPair(cartoon_metric, cartoon_extrinsic, minkowski,
                             spacing, radial_axis, z_center, 0);
  FillAxisymmetricTensorPair(cartesian_metric, cartesian_extrinsic, minkowski,
                             spacing, radial_axis, z_center, cartesian_k_center);

  constexpr int samples = 2;
  DvceArray2D<Real> differences("geometry policy differences", samples, 18);
  const int sample_i[samples] = {radial_axis + NGHOST,
                                 radial_axis - NGHOST - 1};
  auto sample_indices = Kokkos::View<int*>("sample indices", samples);
  auto sample_host = Kokkos::create_mirror_view(sample_indices);
  for (int sample = 0; sample < samples; ++sample) sample_host(sample) = sample_i[sample];
  Kokkos::deep_copy(sample_indices, sample_host);
  TensorField cartoon_g{cartoon_metric};
  TensorField cartoon_K{cartoon_extrinsic};
  TensorField cartesian_g{cartesian_metric};
  TensorField cartesian_K{cartesian_extrinsic};
  Kokkos::parallel_for(
      "shared geometry policy comparison", Kokkos::RangePolicy<>(0, samples),
      KOKKOS_LAMBDA(const int sample) {
        const int i = sample_indices(sample);
        const int j = z_center + 2;
        const Real rho = (i - radial_axis + 0.5) * spacing;
        const Real inverse_spacing[3] = {1.0 / spacing, 1.0 / spacing,
                                         1.0 / spacing};
        z4c::DerivativeProvider<z4c::CartoonSO2, NGHOST> cartoon(
            inverse_spacing, rho, z4c::CartoonAxisLocation::cell_centered,
            0, 0, j, i);
        z4c::DerivativeProvider<z4c::Cartesian3D, NGHOST> cartesian(
            inverse_spacing, 0, cartesian_k_center, j, i);
        const auto collapsed = ComputeZ4cCurvatureDiagnostics<NGHOST>(
            cartoon, cartoon_g, cartoon_K, 0, 0, j, i);
        const auto full = ComputeZ4cCurvatureDiagnostics<NGHOST>(
            cartesian, cartesian_g, cartesian_K, 0, cartesian_k_center, j, i);
        differences(sample, 0) = collapsed.valid && full.valid ? 0.0 : 1.0;
        differences(sample, 1) = collapsed.kretschmann - full.kretschmann;
        for (int component = 0; component < 6; ++component) {
          differences(sample, 2 + component) =
              collapsed.electric[component] - full.electric[component];
          differences(sample, 8 + component) =
              collapsed.magnetic[component] - full.magnetic[component];
        }
        differences(sample, 14) = collapsed.poynting[0] - full.poynting[0];
        differences(sample, 15) = collapsed.poynting[1] - full.poynting[1];
        differences(sample, 16) = collapsed.poynting[2] - full.poynting[2];
        differences(sample, 17) = Kokkos::abs(collapsed.kretschmann);
        for (int component = 0; component < 6; ++component) {
          differences(sample, 17) += Kokkos::abs(collapsed.electric[component]);
          differences(sample, 17) += Kokkos::abs(collapsed.magnetic[component]);
        }
        for (int component = 0; component < 3; ++component) {
          differences(sample, 17) += Kokkos::abs(collapsed.poynting[component]);
        }
      });
  Kokkos::fence();
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), differences);
  const Real tolerance = minkowski ? 5.0e-13 : 2.0e-11;
  for (int sample = 0; sample < samples; ++sample) {
    if (host(sample, 0) != 0.0) return false;
    for (int component = 1; component < 17; ++component) {
      if (!NearlyEqual(host(sample, component), 0.0, tolerance)) return false;
    }
    if (minkowski && !NearlyEqual(host(sample, 17), 0.0, tolerance)) return false;
    if (!minkowski && !(host(sample, 17) > 1.0e-10)) return false;
  }
  return true;
}

}  // namespace

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  const bool passed = CheckWeylCoordinatePolicy() &&
                      CheckWeylTetradComponentMap() &&
                      CheckMeridionalMeasureAndState() &&
      CheckGaugeVectorFamilies<2>() && CheckGaugeVectorFamilies<3>() &&
      CheckGaugeVectorFamilies<4>() && CheckSharedGeometry<2>(true) &&
      CheckSharedGeometry<3>(true) &&
      CheckSharedGeometry<4>(true) && CheckSharedGeometry<2>(false) &&
      CheckSharedGeometry<3>(false) && CheckSharedGeometry<4>(false);
  const std::string backend = Kokkos::DefaultExecutionSpace::name();
  Kokkos::finalize();
  if (!passed) return EXIT_FAILURE;
  std::cout << "Shared Cartesian/Cartoon curvature policy tests passed on "
            << backend << "\n";
  return EXIT_SUCCESS;
}
