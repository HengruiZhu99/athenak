//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c_vertex_derivatives_test.cpp
//! \brief Device tests for Cartesian and Cartoon vertex derivative factories.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>

#include <Kokkos_Core.hpp>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "z4c/cartoon_derivatives.hpp"

namespace {

struct ScalarField {
  DvceArray5D<Real> data;
  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(const int m, const int k, const int j,
                            const int i) const {
    return data(m, 0, k, j, i);
  }
};

bool Near(const Real actual, const Real expected) {
  return std::abs(actual - expected) <=
         2.0e-10 * std::max<Real>(1.0, std::abs(expected));
}

template <int NGHOST>
bool CheckCartoon() {
  constexpr int nx1 = 16;
  constexpr int nx2 = 16;
  constexpr int is = NGHOST;
  constexpr int js = NGHOST;
  constexpr int n1 = nx1 + 1 + 2 * NGHOST;
  constexpr int n2 = nx2 + 1 + 2 * NGHOST;
  DvceArray5D<Real> storage("VC Cartoon scalar", 1, 1, 1, n2, n1);
  auto host = Kokkos::create_mirror_view(storage);
  for (int j = 0; j < n2; ++j) {
    const Real z = VertexX(j - js, nx2, -1.0, 1.0);
    for (int i = 0; i < n1; ++i) {
      const Real rho = VertexX(i - is, nx1, 0.0, 2.0);
      host(0, 0, 0, j, i) = rho * rho + 3.0 * z * z + rho * rho * z * z;
    }
  }
  Kokkos::deep_copy(storage, host);

  Kokkos::View<RegionSize *> size("VC Cartoon size", 1);
  auto size_host = Kokkos::create_mirror_view(size);
  size_host(0).x1min = 0.0;
  size_host(0).x1max = 2.0;
  size_host(0).x2min = -1.0;
  size_host(0).x2max = 1.0;
  size_host(0).dx1 = 2.0 / nx1;
  size_host(0).dx2 = 2.0 / nx2;
  size_host(0).dx3 = 1.0;
  Kokkos::deep_copy(size, size_host);

  Kokkos::View<Real *[6]> result("VC Cartoon derivative results", 2);
  ScalarField field{storage};
  Kokkos::parallel_for(
      "VC Cartoon derivative factories", Kokkos::RangePolicy<DevExeSpace>(0, 2),
      KOKKOS_LAMBDA(const int sample) {
        const int i = sample == 0 ? is : is + 4;
        const int j = js + 5;
        const Real inverse_spacing[3] = {1.0 / size(0).dx1,
                                         1.0 / size(0).dx2, 1.0};
        auto derivatives = z4c::MakeZ4cDerivativeProvider<
            z4c::VertexCenteredZ4c, z4c::CartoonSO2, NGHOST>(
                inverse_spacing, size, nx1, is, 0, 0, j, i);
        result(sample, 0) = derivatives.ScalarFirst(0, field);
        result(sample, 1) = derivatives.ScalarFirst(1, field);
        result(sample, 2) = derivatives.ScalarSecond(0, 0, field);
        result(sample, 3) = derivatives.ScalarSecond(1, 1, field);
        result(sample, 4) = derivatives.ScalarSecond(0, 1, field);
        result(sample, 5) = derivatives.ScalarSecond(2, 2, field);
      });
  Kokkos::fence();
  const auto values =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);
  for (int sample = 0; sample < 2; ++sample) {
    const Real rho = sample == 0 ? 0.0 : VertexX(4, nx1, 0.0, 2.0);
    const Real z = VertexX(5, nx2, -1.0, 1.0);
    const Real expected[6] = {2.0 * rho * (1.0 + z * z),
                              2.0 * z * (3.0 + rho * rho),
                              2.0 * (1.0 + z * z),
                              2.0 * (3.0 + rho * rho),
                              4.0 * rho * z,
                              2.0 * (1.0 + z * z)};
    for (int quantity = 0; quantity < 6; ++quantity) {
      if (!Near(values(sample, quantity), expected[quantity])) return false;
    }
  }
  return true;
}

std::pair<Real, Real> CartoonO4ConvergenceError(const int nx) {
  constexpr int nghost = 3;
  const int is = nghost;
  const int js = nghost;
  const int n1 = nx + 1 + 2 * nghost;
  const int n2 = nx + 1 + 2 * nghost;
  DvceArray5D<Real> storage("VC Cartoon smooth convergence", 1, 1, 1, n2, n1);
  auto host = Kokkos::create_mirror_view(storage);
  for (int j = 0; j < n2; ++j) {
    const Real z = VertexX(j - js, nx, -1.0, 1.0);
    for (int i = 0; i < n1; ++i) {
      const Real rho = VertexX(i - is, nx, 0.0, 2.0);
      host(0, 0, 0, j, i) = std::exp(-rho * rho) * std::cos(z);
    }
  }
  Kokkos::deep_copy(storage, host);
  Kokkos::View<RegionSize *> size("VC Cartoon convergence size", 1);
  auto size_host = Kokkos::create_mirror_view(size);
  size_host(0).x1min = 0.0;
  size_host(0).x1max = 2.0;
  size_host(0).x2min = -1.0;
  size_host(0).x2max = 1.0;
  size_host(0).dx1 = 2.0 / nx;
  size_host(0).dx2 = 2.0 / nx;
  size_host(0).dx3 = 1.0;
  Kokkos::deep_copy(size, size_host);
  ScalarField field{storage};
  const int active_points = nx + 1;
  const int total_points = active_points * active_points;
  Real squared_error = 0.0;
  Kokkos::parallel_reduce(
      "VC Cartoon O4 smooth whole-domain error",
      Kokkos::RangePolicy<DevExeSpace>(0, total_points),
      KOKKOS_LAMBDA(const int index, Real &sum) {
        const int i = is + index % active_points;
        const int j = js + index / active_points;
        const Real rho = VertexX(i - is, nx, 0.0, 2.0);
        const Real z = VertexX(j - js, nx, -1.0, 1.0);
        const Real exponential = Kokkos::exp(-rho * rho);
        const Real cosine = Kokkos::cos(z);
        const Real sine = Kokkos::sin(z);
        const Real exact[6] = {
            -2.0 * rho * exponential * cosine,
            -exponential * sine,
            (4.0 * rho * rho - 2.0) * exponential * cosine,
            -exponential * cosine,
            2.0 * rho * exponential * sine,
            -2.0 * exponential * cosine};
        const Real inverse_spacing[3] = {0.5 * nx, 0.5 * nx, 1.0};
        auto derivatives = z4c::MakeZ4cDerivativeProvider<
            z4c::VertexCenteredZ4c, z4c::CartoonSO2, nghost>(
                inverse_spacing, size, nx, is, 0, 0, j, i);
        const Real actual[6] = {
            derivatives.ScalarFirst(0, field),
            derivatives.ScalarFirst(1, field),
            derivatives.ScalarSecond(0, 0, field),
            derivatives.ScalarSecond(1, 1, field),
            derivatives.ScalarSecond(0, 1, field),
            derivatives.ScalarSecond(2, 2, field)};
        for (int quantity = 0; quantity < 6; ++quantity) {
          const Real difference = actual[quantity] - exact[quantity];
          sum += difference * difference;
        }
      }, squared_error);
  Real axis_squared_error = 0.0;
  Kokkos::parallel_reduce(
      "VC Cartoon O4 smooth axis error",
      Kokkos::RangePolicy<DevExeSpace>(0, active_points),
      KOKKOS_LAMBDA(const int offset, Real &sum) {
        const int i = is;
        const int j = js + offset;
        const Real z = VertexX(offset, nx, -1.0, 1.0);
        const Real exponential = 1.0;
        const Real cosine = Kokkos::cos(z);
        const Real sine = Kokkos::sin(z);
        const Real exact[6] = {0.0, -exponential * sine,
                               -2.0 * exponential * cosine,
                               -exponential * cosine, 0.0,
                               -2.0 * exponential * cosine};
        const Real inverse_spacing[3] = {0.5 * nx, 0.5 * nx, 1.0};
        auto derivatives = z4c::MakeZ4cDerivativeProvider<
            z4c::VertexCenteredZ4c, z4c::CartoonSO2, nghost>(
                inverse_spacing, size, nx, is, 0, 0, j, i);
        const Real actual[6] = {
            derivatives.ScalarFirst(0, field),
            derivatives.ScalarFirst(1, field),
            derivatives.ScalarSecond(0, 0, field),
            derivatives.ScalarSecond(1, 1, field),
            derivatives.ScalarSecond(0, 1, field),
            derivatives.ScalarSecond(2, 2, field)};
        for (int quantity = 0; quantity < 6; ++quantity) {
          const Real difference = actual[quantity] - exact[quantity];
          sum += difference * difference;
        }
      }, axis_squared_error);
  Kokkos::fence();
  return {std::sqrt(squared_error / (6.0 * total_points)),
          std::sqrt(axis_squared_error / (6.0 * active_points))};
}

bool CheckCartoonO4Convergence() {
  const auto coarse = CartoonO4ConvergenceError(16);
  const auto medium = CartoonO4ConvergenceError(32);
  const auto fine = CartoonO4ConvergenceError(64);
  const Real whole_order_1 = std::log2(coarse.first / medium.first);
  const Real whole_order_2 = std::log2(medium.first / fine.first);
  const Real axis_order_1 = std::log2(coarse.second / medium.second);
  const Real axis_order_2 = std::log2(medium.second / fine.second);
  if (std::min(whole_order_1, whole_order_2) < 3.5 ||
      std::min(axis_order_1, axis_order_2) < 3.5) {
    std::cerr << "Cartoon native-VC O4 convergence failure whole="
              << whole_order_1 << "," << whole_order_2 << " axis="
              << axis_order_1 << "," << axis_order_2 << "\n";
    return false;
  }
  std::cout << "Cartoon native-VC O4 convergence whole=" << whole_order_1
            << "," << whole_order_2 << " axis=" << axis_order_1 << ","
            << axis_order_2 << "\n";
  return true;
}

template <int NGHOST>
bool CheckCartesian() {
  constexpr int nx = 12;
  constexpr int start = NGHOST;
  constexpr int extent = nx + 1 + 2 * NGHOST;
  DvceArray5D<Real> storage("VC Cartesian scalar", 1, 1, extent, extent, extent);
  auto host = Kokkos::create_mirror_view(storage);
  for (int k = 0; k < extent; ++k) {
    const Real z = VertexX(k - start, nx, -1.0, 1.0);
    for (int j = 0; j < extent; ++j) {
      const Real y = VertexX(j - start, nx, -1.0, 1.0);
      for (int i = 0; i < extent; ++i) {
        const Real x = VertexX(i - start, nx, -1.0, 1.0);
        host(0, 0, k, j, i) = x*x + 2.0*y*y + 3.0*z*z + x*y + y*z;
      }
    }
  }
  Kokkos::deep_copy(storage, host);
  Kokkos::View<RegionSize *> size("VC Cartesian size", 1);
  auto size_host = Kokkos::create_mirror_view(size);
  size_host(0).x1min = size_host(0).x2min = size_host(0).x3min = -1.0;
  size_host(0).x1max = size_host(0).x2max = size_host(0).x3max = 1.0;
  size_host(0).dx1 = size_host(0).dx2 = size_host(0).dx3 = 2.0 / nx;
  Kokkos::deep_copy(size, size_host);

  Kokkos::View<Real *[6]> result("VC Cartesian derivative results", 1);
  ScalarField field{storage};
  Kokkos::parallel_for(
      "VC Cartesian derivative factory", Kokkos::RangePolicy<DevExeSpace>(0, 1),
      KOKKOS_LAMBDA(const int) {
        const int i = start + 4, j = start + 5, k = start + 6;
        const Real inverse_spacing[3] = {1.0 / size(0).dx1,
                                         1.0 / size(0).dx2,
                                         1.0 / size(0).dx3};
        auto derivatives = z4c::MakeZ4cDerivativeProvider<
            z4c::VertexCenteredZ4c, z4c::Cartesian3D, NGHOST>(
                inverse_spacing, size, nx, start, 0, k, j, i);
        result(0, 0) = derivatives.ScalarFirst(0, field);
        result(0, 1) = derivatives.ScalarFirst(1, field);
        result(0, 2) = derivatives.ScalarFirst(2, field);
        result(0, 3) = derivatives.ScalarSecond(0, 0, field);
        result(0, 4) = derivatives.ScalarSecond(0, 1, field);
        result(0, 5) = derivatives.ScalarSecond(1, 2, field);
      });
  Kokkos::fence();
  const auto values =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);
  const Real x = VertexX(4, nx, -1.0, 1.0);
  const Real y = VertexX(5, nx, -1.0, 1.0);
  const Real z = VertexX(6, nx, -1.0, 1.0);
  const Real expected[6] = {2.0*x + y, 4.0*y + x + z, 6.0*z + y,
                            2.0, 1.0, 1.0};
  for (int quantity = 0; quantity < 6; ++quantity) {
    if (!Near(values(0, quantity), expected[quantity])) return false;
  }
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  if (argc != 2) return EXIT_FAILURE;
  const std::string selection(argv[1]);
  bool passed = false;
  if (selection == "cartesian_o2") passed = CheckCartesian<2>();
  if (selection == "cartesian_o4") passed = CheckCartesian<3>();
  if (selection == "cartesian_o6") passed = CheckCartesian<4>();
  if (selection == "cartoon_o2") passed = CheckCartoon<2>();
  if (selection == "cartoon_o4") {
    passed = CheckCartoon<3>() && CheckCartoonO4Convergence();
  }
  if (selection == "cartoon_o6") passed = CheckCartoon<4>();
  if (!passed) return EXIT_FAILURE;
  std::cout << "Z4c Cartesian/Cartoon VC derivative factories passed on "
            << Kokkos::DefaultExecutionSpace::name() << "\n";
  return EXIT_SUCCESS;
}
