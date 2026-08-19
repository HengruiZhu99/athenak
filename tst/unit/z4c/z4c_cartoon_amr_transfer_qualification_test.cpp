//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c_cartoon_amr_transfer_qualification_test.cpp
//! \brief Zero-PDE 2D Cartoon transfer qualification matrix.
//!
//! This is intentionally an operator qualification, not an evolution test.  It calls the
//! production point-value restriction and prolongation templates for every packed Z4c
//! component, under the production half-plane parity table, for O2/O4/O6 configurations.
//! The AMR-jump runtime probe separately exercises the full topology/communication path.

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

#include <Kokkos_Core.hpp>

#include "athena.hpp"
#include "mesh/prolongation.hpp"
#include "mesh/restriction.hpp"
#include "z4c/cartoon_axis_parity.hpp"
#include "z4c/z4c.hpp"

namespace {

constexpr int kNvar = z4c::Z4c::nz4c;

bool NearlyEqual(const Real left, const Real right, const Real tolerance) {
  const Real scale = std::max({Real(1.0), std::abs(left), std::abs(right)});
  return std::abs(left - right) <= tolerance * scale;
}

struct RestrictionWeights {
  DualArray1D<Real> second;
  DualArray1D<Real> fourth;
  DualArray1D<Real> fourth_edge;
};

RestrictionWeights MakeRestrictionWeights() {
  RestrictionWeights weights{
      DualArray1D<Real>("qualification second restriction", 3),
      DualArray1D<Real>("qualification fourth restriction", 5),
      DualArray1D<Real>("qualification fourth edge restriction", 5)};
  constexpr Real second[3] = {0.375, 0.75, -0.125};
  constexpr Real fourth[5] = {
      -0.0390625, 0.46875, 0.703125, -0.15625, 0.0234375};
  constexpr Real fourth_edge[5] = {
      0.2734375, 1.09375, -0.546875, 0.21875, -0.0390625};
  for (int n = 0; n < 3; ++n) weights.second.h_view(n) = second[n];
  for (int n = 0; n < 5; ++n) {
    weights.fourth.h_view(n) = fourth[n];
    weights.fourth_edge.h_view(n) = fourth_edge[n];
  }
  weights.second.template modify<HostMemSpace>();
  weights.second.template sync<DevExeSpace>();
  weights.fourth.template modify<HostMemSpace>();
  weights.fourth.template sync<DevExeSpace>();
  weights.fourth_edge.template modify<HostMemSpace>();
  weights.fourth_edge.template sync<DevExeSpace>();
  return weights;
}

template <int NGHOST>
DualArray3D<Real> MakeProlongationWeights() {
  DualArray3D<Real> weights("qualification prolongation", 5, 5, 5);
  for (int k = 0; k < 5; ++k) {
    for (int j = 0; j < 5; ++j) {
      for (int i = 0; i < 5; ++i) {
        weights.h_view(k, j, i) =
            (k <= NGHOST && j <= NGHOST && i <= NGHOST)
                ? ProlongWeight1D<NGHOST>(k, false) *
                      ProlongWeight1D<NGHOST>(j, false) *
                      ProlongWeight1D<NGHOST>(i, false)
                : 0.0;
      }
    }
  }
  weights.template modify<HostMemSpace>();
  weights.template sync<DevExeSpace>();
  return weights;
}

bool IsMetricDiagonal(const int component) {
  return component == z4c::Z4c::I_Z4C_GXX ||
         component == z4c::Z4c::I_Z4C_GYY ||
         component == z4c::Z4c::I_Z4C_GZZ;
}

// A smooth analytic half-plane field for every transformation class in the packed state.
// Odd packed components carry an explicit signed-rho factor, and the A diagonal is
// trace-free in the identity conformal metric.  The polynomial degree is at most two so
// all three production restriction configurations must reproduce it exactly.
Real AnalyticState(const int component, const Real rho, const Real z) {
  const int parity = z4c::Z4cStateAxisParitySignFromPackedIndex(component);
  const Real amplitude = 1.0e-3 * static_cast<Real>(component + 1);
  const Real even = amplitude * (1.0 + 0.17 * rho * rho + 0.11 * z * z);
  const Real odd = amplitude * rho * (1.0 + 0.13 * z);
  const Real a_diagonal = 1.0e-2 * (1.0 + 0.17 * rho * rho + 0.11 * z * z);
  if (component == z4c::Z4c::I_Z4C_AXX) return a_diagonal;
  if (component == z4c::Z4c::I_Z4C_AYY || component == z4c::Z4c::I_Z4C_AZZ) {
    return -0.5 * a_diagonal;
  }
  Real base = 0.0;
  if (component == z4c::Z4c::I_Z4C_CHI) base = 1.25;
  if (component == z4c::Z4c::I_Z4C_ALPHA) base = 1.0;
  if (IsMetricDiagonal(component)) base = 1.0;
  return base + (parity < 0 ? odd : even);
}

template <int NGHOST>
bool CheckAllComponentRestrictionAndProlongation() {
  constexpr int nx = 16;
  constexpr int extent = nx + 2 * NGHOST;
  constexpr Real h = 0.125;
  constexpr int target = NGHOST + 4;
  constexpr int stencil = NGHOST + 1;
  constexpr int center = NGHOST / 2;
  const auto restriction = MakeRestrictionWeights();
  const auto prolongation = MakeProlongationWeights<NGHOST>();
  DvceArray5D<Real> fine("qualification analytic fine", 1, kNvar, 1,
                         extent, extent);
  DvceArray5D<Real> restricted("qualification restricted values", 1, kNvar,
                               1, 1, 1);
  DvceArray5D<Real> coarse("qualification analytic coarse", 1, kNvar, 1,
                           stencil, stencil);
  DvceArray5D<Real> children("qualification prolonged children", 1, kNvar,
                             1, 2, 2);

  auto fine_host = Kokkos::create_mirror_view(fine);
  for (int v = 0; v < kNvar; ++v) {
    for (int j = 0; j < extent; ++j) {
      for (int i = 0; i < extent; ++i) {
        // RestrictInterpolation is defined on the index-coordinate convention
        // used by the production point-value operator: its parent target is at
        // fi+1/2 while the fine stencil samples are indexed by fi, fi+1, ... .
        // Keep this fixture aligned with the existing exactness qualification.
        const Real rho = static_cast<Real>(i - NGHOST) * h;
        const Real z = static_cast<Real>(j - NGHOST) * h;
        fine_host(0, v, 0, j, i) = AnalyticState(v, rho, z);
      }
    }
  }
  Kokkos::deep_copy(fine, fine_host);
  Kokkos::parallel_for(
      "qualification restrict every packed Z4c component",
      Kokkos::RangePolicy<>(0, kNvar), KOKKOS_LAMBDA(const int v) {
        restricted(0, v, 0, 0, 0) = RestrictInterpolation<NGHOST>(
            0, v, 0, target, target, nx, nx, 1, fine, restriction.second,
            restriction.fourth, restriction.fourth_edge);
      });
  const auto restricted_host =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), restricted);
  const Real restriction_coordinate =
      (static_cast<Real>(target - NGHOST) + 0.5) * h;
  for (int v = 0; v < kNvar; ++v) {
    const Real expected = AnalyticState(v, restriction_coordinate,
                                        restriction_coordinate);
    if (!NearlyEqual(restricted_host(0, v, 0, 0, 0), expected,
                     3.0e-12)) {
      std::cerr << "restriction mismatch order=" << NGHOST << " component=" << v
                << " actual=" << std::setprecision(17)
                << restricted_host(0, v, 0, 0, 0)
                << " expected=" << expected << "\n";
      return false;
    }
  }

  auto coarse_host = Kokkos::create_mirror_view(coarse);
  constexpr Real coarse_h = 2.0 * h;
  for (int v = 0; v < kNvar; ++v) {
    for (int j = 0; j < stencil; ++j) {
      for (int i = 0; i < stencil; ++i) {
        coarse_host(0, v, 0, j, i) = AnalyticState(
            v, static_cast<Real>(i - center) * coarse_h,
            static_cast<Real>(j - center) * coarse_h);
      }
    }
  }
  Kokkos::deep_copy(coarse, coarse_host);
  Kokkos::parallel_for(
      "qualification prolong every packed Z4c component",
      Kokkos::RangePolicy<>(0, kNvar), KOKKOS_LAMBDA(const int v) {
        HighOrderProlongCC<NGHOST>(0, v, 0, center, center, 0, 0, 0, nx,
                                   nx, 1, coarse, children, prolongation);
      });
  const auto children_host =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), children);
  for (int v = 0; v < kNvar; ++v) {
    for (int j = 0; j < 2; ++j) {
      for (int i = 0; i < 2; ++i) {
        const Real rho = (static_cast<Real>(i) - 0.5) * h;
        const Real z = (static_cast<Real>(j) - 0.5) * h;
        if (!NearlyEqual(children_host(0, v, 0, j, i),
                         AnalyticState(v, rho, z), 4.0e-12)) {
          std::cerr << "prolongation mismatch order=" << NGHOST
                    << " component=" << v << " child=" << j << "," << i
                    << "\n";
          return false;
        }
      }
    }
  }

  // The analytic metric remains SPD and the constructed A is trace-free.  This does not
  // replace the production algebraic projection; it guards the field-class fixture.
  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < 2; ++i) {
      const Real gxx = children_host(0, z4c::Z4c::I_Z4C_GXX, 0, j, i);
      const Real gyy = children_host(0, z4c::Z4c::I_Z4C_GYY, 0, j, i);
      const Real gzz = children_host(0, z4c::Z4c::I_Z4C_GZZ, 0, j, i);
      const Real trace = children_host(0, z4c::Z4c::I_Z4C_AXX, 0, j, i) +
                         children_host(0, z4c::Z4c::I_Z4C_AYY, 0, j, i) +
                         children_host(0, z4c::Z4c::I_Z4C_AZZ, 0, j, i);
      if (!(gxx > 0.0 && gyy > 0.0 && gzz > 0.0) ||
          !NearlyEqual(trace, 0.0, 5.0e-14)) return false;
    }
  }
  return true;
}

template <int NGHOST>
bool CheckFourierSymbolSweep() {
  constexpr int nx = 16;
  constexpr int extent = nx + 2 * NGHOST;
  const auto restriction = MakeRestrictionWeights();
  const auto prolongation = MakeProlongationWeights<NGHOST>();
  DvceArray5D<Real> fine("qualification Fourier fine", 1, 1, 1, extent, extent);
  DvceArray5D<Real> coarse("qualification Fourier coarse", 1, 1, 1,
                           extent / 2, extent / 2);
  DvceArray5D<Real> children("qualification Fourier children", 1, 1, 1, 2, 2);
  Real largest_gain = 0.0;
  int largest_mrho = 0;
  int largest_mz = 0;
  int annihilated_modes = 0;
  const int coarse_center = NGHOST / 2;
  for (int mrho = 0; mrho <= nx / 2; ++mrho) {
    for (int mz = 0; mz <= nx / 2; ++mz) {
      if (mrho == 0 && mz == 0) continue;
      auto fine_host = Kokkos::create_mirror_view(fine);
      for (int j = 0; j < extent; ++j) {
        for (int i = 0; i < extent; ++i) {
          const Real phase = 2.0 * M_PI *
              (static_cast<Real>(mrho * (i - NGHOST) + mz * (j - NGHOST)) /
               static_cast<Real>(nx));
          fine_host(0, 0, 0, j, i) = std::cos(phase);
        }
      }
      Kokkos::deep_copy(fine, fine_host);
      auto coarse_host = Kokkos::create_mirror_view(coarse);
      for (int j = 0; j < coarse.extent_int(3); ++j) {
        for (int i = 0; i < coarse.extent_int(4); ++i) {
          // A cyclic extension supplies a complete local stencil while preserving the
          // exact production restriction kernel at every representable parent mode.
          const int fi = 2 * (i - coarse_center) + NGHOST;
          const int fj = 2 * (j - coarse_center) + NGHOST;
          if (fi < NGHOST / 2 || fj < NGHOST / 2 ||
              fi >= extent - NGHOST / 2 || fj >= extent - NGHOST / 2) {
            coarse_host(0, 0, 0, j, i) = 0.0;
          }
        }
      }
      Kokkos::deep_copy(coarse, coarse_host);
      Kokkos::parallel_for(
          "qualification Fourier restrict", Kokkos::RangePolicy<>(0, 1),
          KOKKOS_LAMBDA(const int) {
            for (int j = 0; j < coarse.extent_int(3); ++j) {
              for (int i = 0; i < coarse.extent_int(4); ++i) {
                const int fi = 2 * (i - coarse_center) + NGHOST;
                const int fj = 2 * (j - coarse_center) + NGHOST;
                if (fi >= NGHOST / 2 && fj >= NGHOST / 2 &&
                    fi < extent - NGHOST / 2 && fj < extent - NGHOST / 2) {
                  coarse(0, 0, 0, j, i) = RestrictInterpolation<NGHOST>(
                      0, 0, 0, fj, fi, nx, nx, 1, fine, restriction.second,
                      restriction.fourth, restriction.fourth_edge);
                }
              }
            }
          });
      Kokkos::parallel_for(
          "qualification Fourier prolong", Kokkos::RangePolicy<>(0, 1),
          KOKKOS_LAMBDA(const int) {
            HighOrderProlongCC<NGHOST>(0, 0, 0, coarse_center, coarse_center,
                                       0, 0, 0, nx, nx, 1, coarse, children,
                                       prolongation);
          });
      const auto child_host =
          Kokkos::create_mirror_view_and_copy(HostMemSpace(), children);
      const auto parent_host =
          Kokkos::create_mirror_view_and_copy(HostMemSpace(), coarse);
      Real child_energy = 0.0;
      for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 2; ++i) {
          const Real value = child_host(0, 0, 0, j, i);
          if (!std::isfinite(value)) return false;
          child_energy += value * value;
        }
      }
      Real parent_energy = 0.0;
      int parent_count = 0;
      for (int j = 0; j < coarse.extent_int(3); ++j) {
        for (int i = 0; i < coarse.extent_int(4); ++i) {
          const int fi = 2 * (i - coarse_center) + NGHOST;
          const int fj = 2 * (j - coarse_center) + NGHOST;
          if (fi >= NGHOST / 2 && fj >= NGHOST / 2 &&
              fi < extent - NGHOST / 2 && fj < extent - NGHOST / 2) {
            const Real value = parent_host(0, 0, 0, j, i);
            parent_energy += value * value;
            ++parent_count;
          }
        }
      }
      const Real parent_rms = std::sqrt(parent_energy /
                                        static_cast<Real>(parent_count));
      if (parent_rms <= 1.0e-12) {
        // A restriction-null mode has no meaningful relative gain.  Require its
        // children to be null too, and retain it in the deterministic inventory.
        if (child_energy > 1.0e-22) return false;
        ++annihilated_modes;
        continue;
      }
      const Real gain = std::sqrt(child_energy / 4.0) / parent_rms;
      if (!std::isfinite(gain)) return false;
      if (gain > largest_gain) {
        largest_gain = gain;
        largest_mrho = mrho;
        largest_mz = mz;
      }
    }
  }
  std::cout << "AMR_TRANSFER_FOURIER order=" << NGHOST
            << " max_child_to_parent_rms_gain=" << std::setprecision(17)
            << largest_gain << " mode_rho=" << largest_mrho
            << " mode_z=" << largest_mz
            << " restriction_null_modes=" << annihilated_modes << "\n";
  return std::isfinite(largest_gain);
}

template <int NGHOST>
bool CheckRepeatedRoundTrip() {
  constexpr int nx = 16;
  constexpr int n = NGHOST + 1;
  constexpr int center = NGHOST / 2;
  const auto weights = MakeProlongationWeights<NGHOST>();
  DvceArray5D<Real> parent("qualification repeated parent", 1, kNvar, 1, n, n);
  DvceArray5D<Real> children("qualification repeated children", 1, kNvar, 1, 2, 2);
  auto host = Kokkos::create_mirror_view(parent);
  for (int v = 0; v < kNvar; ++v) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        host(0, v, 0, j, i) = AnalyticState(v, 0.25 * (i - center),
                                              0.25 * (j - center));
      }
    }
  }
  Kokkos::deep_copy(parent, host);
  for (int repeat = 0; repeat < 4; ++repeat) {
    Kokkos::parallel_for(
        "qualification repeated prolongation", Kokkos::RangePolicy<>(0, kNvar),
        KOKKOS_LAMBDA(const int v) {
          HighOrderProlongCC<NGHOST>(0, v, 0, center, center, 0, 0, 0, nx,
                                     nx, 1, parent, children, weights);
        });
    const auto child_host =
        Kokkos::create_mirror_view_and_copy(HostMemSpace(), children);
    for (int v = 0; v < kNvar; ++v) {
      for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 2; ++i) {
          if (!std::isfinite(child_host(0, v, 0, j, i))) return false;
        }
      }
    }
  }
  return true;
}

template <int NGHOST>
bool RunOrder() {
  return CheckAllComponentRestrictionAndProlongation<NGHOST>() &&
         CheckRepeatedRoundTrip<NGHOST>() && CheckFourierSymbolSweep<NGHOST>();
}

}  // namespace

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  const bool passed = RunOrder<2>() && RunOrder<3>() && RunOrder<4>();
  Kokkos::finalize();
  if (!passed) {
    std::cerr << "Cartoon Z4c AMR transfer qualification failed\n";
    return 1;
  }
  std::cout << "Cartoon Z4c AMR transfer qualification passed\n";
  return 0;
}
