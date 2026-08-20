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
#include "z4c/state_admissibility.hpp"
#include "z4c/z4c.hpp"

namespace {

constexpr int kNvar = z4c::Z4c::nz4c;

template <int STENCIL>
constexpr int AllocatedGhosts() {
  return STENCIL == 3 ? 4 : STENCIL;
}

template <int STENCIL>
constexpr int ProlongationParentExtent() {
  return STENCIL == 3 ? 5 : STENCIL + 1;
}

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
  constexpr int allocated_ng = AllocatedGhosts<NGHOST>();
  constexpr int extent = nx + 2 * allocated_ng;
  constexpr Real h = 0.125;
  constexpr int target = allocated_ng + 4;
  constexpr int stencil = ProlongationParentExtent<NGHOST>();
  constexpr int center = stencil / 2;
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
        const Real rho = static_cast<Real>(i - allocated_ng) * h;
        const Real z = static_cast<Real>(j - allocated_ng) * h;
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
      (static_cast<Real>(target - allocated_ng) + 0.5) * h;
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
  constexpr int allocated_ng = AllocatedGhosts<NGHOST>();
  constexpr int extent = nx + 2 * allocated_ng;
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
  const int coarse_center = NGHOST == 3 ? 2 : NGHOST / 2;
  for (int mrho = 0; mrho <= nx / 2; ++mrho) {
    for (int mz = 0; mz <= nx / 2; ++mz) {
      if (mrho == 0 && mz == 0) continue;
      auto fine_host = Kokkos::create_mirror_view(fine);
      for (int j = 0; j < extent; ++j) {
        for (int i = 0; i < extent; ++i) {
          const Real phase = 2.0 * M_PI *
              (static_cast<Real>(mrho * (i - allocated_ng) +
                                      mz * (j - allocated_ng)) /
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
          const int fi = 2 * (i - coarse_center) + allocated_ng;
          const int fj = 2 * (j - coarse_center) + allocated_ng;
          if (fi < 1 || fj < 1 || fi >= extent - 2 || fj >= extent - 2) {
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
                const int fi = 2 * (i - coarse_center) + allocated_ng;
                const int fj = 2 * (j - coarse_center) + allocated_ng;
                if (fi >= 1 && fj >= 1 && fi < extent - 2 && fj < extent - 2) {
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
          const int fi = 2 * (i - coarse_center) + allocated_ng;
          const int fj = 2 * (j - coarse_center) + allocated_ng;
          if (fi >= 1 && fj >= 1 && fi < extent - 2 && fj < extent - 2) {
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
  constexpr int n = ProlongationParentExtent<NGHOST>();
  constexpr int center = n / 2;
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

bool CheckO4RestrictionEdgesAndGhostPoisoning() {
  constexpr int stencil = 3;
  constexpr int allocated_ng = 4;
  constexpr int nx = 16;
  constexpr int extent = nx + 2 * allocated_ng;
  const auto restriction = MakeRestrictionWeights();
  DvceArray5D<Real> fine("O4 active-only edge restriction", 1, 1, 1,
                         extent, extent);
  DvceArray5D<Real> result("O4 edge restriction results", 1, 1, 1, 3, 3);

  auto fill = [&](const Real poison) {
    auto host = Kokkos::create_mirror_view(fine);
    for (int j = 0; j < extent; ++j) {
      for (int i = 0; i < extent; ++i) host(0, 0, 0, j, i) = poison;
    }
    for (int j = allocated_ng; j < allocated_ng + nx; ++j) {
      const Real y = static_cast<Real>(j - allocated_ng);
      for (int i = allocated_ng; i < allocated_ng + nx; ++i) {
        const Real x = static_cast<Real>(i - allocated_ng);
        host(0, 0, 0, j, i) = 2.0 + 0.3*x - 0.2*y + 0.04*x*x -
                               0.03*x*y + 0.01*y*y + 0.002*x*x*x -
                               0.001*y*y*y;
      }
    }
    Kokkos::deep_copy(fine, host);
  };
  auto evaluate = [&]() {
    constexpr int lower = allocated_ng;
    constexpr int middle = allocated_ng + 6;
    constexpr int upper = allocated_ng + nx - 2;
    Kokkos::parallel_for(
        "O4 evaluate edge restriction", Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0}, {3, 3}), KOKKOS_LAMBDA(const int jcase, const int icase) {
          const int target_j = jcase == 0 ? lower : (jcase == 1 ? middle : upper);
          const int target_i = icase == 0 ? lower : (icase == 1 ? middle : upper);
          result(0, 0, 0, jcase, icase) = RestrictInterpolation<stencil>(
              0, 0, 0, target_j, target_i, nx, nx, 1, fine,
              restriction.second, restriction.fourth,
              restriction.fourth_edge);
        });
    Kokkos::fence();
    return Kokkos::create_mirror_view_and_copy(HostMemSpace(), result);
  };

  fill(9.0e20);
  const auto positive_poison = evaluate();
  fill(-7.0e30);
  const auto negative_poison = evaluate();
  constexpr int targets[3] = {allocated_ng, allocated_ng + 6,
                              allocated_ng + nx - 2};
  for (int jcase = 0; jcase < 3; ++jcase) {
    const Real y = static_cast<Real>(targets[jcase] - allocated_ng) + 0.5;
    for (int icase = 0; icase < 3; ++icase) {
      const Real x = static_cast<Real>(targets[icase] - allocated_ng) + 0.5;
      const Real expected = 2.0 + 0.3*x - 0.2*y + 0.04*x*x -
                            0.03*x*y + 0.01*y*y + 0.002*x*x*x -
                            0.001*y*y*y;
      if (!NearlyEqual(positive_poison(0, 0, 0, jcase, icase), expected,
                       4.0e-12) ||
          positive_poison(0, 0, 0, jcase, icase) !=
              negative_poison(0, 0, 0, jcase, icase)) {
        std::cerr << "O4 edge restriction failed case=" << jcase << ','
                  << icase << "\n";
        return false;
      }
    }
  }
  for (int n = 0; n < 4; ++n) {
    if (O4RestrictionWeight(O4RestrictionStencil1D::active_lower, n) !=
        O4RestrictionWeight(O4RestrictionStencil1D::active_upper, 3 - n)) {
      return false;
    }
  }
  return true;
}

bool CheckO4SymmetricProlongationAndChiUnion() {
  constexpr int n = 7;
  constexpr int center = 3;
  constexpr int nx = 16;
  const auto weights = MakeProlongationWeights<3>();
  DvceArray5D<Real> coarse("O4 symmetric coarse", 1, 1, 1, n, n);
  DvceArray5D<Real> children("O4 symmetric children", 1, 1, 1, 2, 2);
  auto host = Kokkos::create_mirror_view(coarse);
  for (int j = 0; j < n; ++j) {
    const Real y = 2.0 * static_cast<Real>(j - center);
    for (int i = 0; i < n; ++i) {
      const Real x = 2.0 * static_cast<Real>(i - center);
      host(0, 0, 0, j, i) = 20.0 + 0.4*x - 0.3*y + 0.07*x*x +
                             0.02*x*y - 0.03*y*y + 0.004*x*x*x -
                             0.002*y*y*y;
    }
  }
  Kokkos::deep_copy(coarse, host);
  Kokkos::parallel_for(
      "O4 symmetric prolongation", Kokkos::RangePolicy<>(0, 1),
      KOKKOS_LAMBDA(const int) {
        HighOrderProlongCC<3>(0, 0, 0, center, center, 0, 0, 0, nx,
                              nx, 1, coarse, children, weights);
      });
  const auto child =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), children);
  for (int j = 0; j < 2; ++j) {
    const Real y = j == 0 ? -0.5 : 0.5;
    for (int i = 0; i < 2; ++i) {
      const Real x = i == 0 ? -0.5 : 0.5;
      const Real expected = 20.0 + 0.4*x - 0.3*y + 0.07*x*x +
                            0.02*x*y - 0.03*y*y + 0.004*x*x*x -
                            0.002*y*y*y;
      if (!NearlyEqual(child(0, 0, 0, j, i), expected, 4.0e-12)) {
        std::cerr << "O4 child mismatch j=" << j << " i=" << i
                  << " actual=" << std::setprecision(17)
                  << child(0, 0, 0, j, i) << " expected=" << expected << "\n";
        return false;
      }
    }
  }
  for (int q = 0; q < 4; ++q) {
    if (ProlongWeight1D<3>(q, false) !=
        ProlongWeight1D<3>(3 - q, true)) return false;
  }

  // Every one of the 5x5 parents consumed by the complete child group must be
  // covered by the strict chi gate.  A value just outside that union must not
  // affect it.
  for (int dj = -2; dj <= 2; ++dj) {
    for (int di = -2; di <= 2; ++di) {
      auto candidate = Kokkos::create_mirror_view(coarse);
      for (int jj = 0; jj < n; ++jj) {
        const Real y = 2.0 * static_cast<Real>(jj - center);
        for (int ii = 0; ii < n; ++ii) {
          const Real x = 2.0 * static_cast<Real>(ii - center);
          candidate(0, 0, 0, jj, ii) =
              20.0 + 0.4*x - 0.3*y + 0.07*x*x + 0.02*x*y -
              0.03*y*y + 0.004*x*x*x - 0.002*y*y*y;
        }
      }
      candidate(0, 0, 0, center + dj, center + di) = 0.0;
      Kokkos::deep_copy(coarse, candidate);
      Kokkos::View<int *> valid("O4 chi parent gate result", 1);
      Kokkos::parallel_for(
          "O4 chi parent union", Kokkos::RangePolicy<>(0, 1),
          KOKKOS_LAMBDA(const int) {
            valid(0) = ProlongationParentStencilFinitePositive<3>(
                           0, 0, 0, center, center, 1, coarse) ? 1 : 0;
          });
      const auto valid_host =
          Kokkos::create_mirror_view_and_copy(HostMemSpace(), valid);
        if (valid_host(0) != 0) {
          std::cerr << "O4 chi union missed parent dj=" << dj
                    << " di=" << di << "\n";
          return false;
        }
    }
  }
  for (int j = 0; j < n; ++j) {
    const Real y = 2.0 * static_cast<Real>(j - center);
    for (int i = 0; i < n; ++i) {
      const Real x = 2.0 * static_cast<Real>(i - center);
      host(0, 0, 0, j, i) = 20.0 + 0.4*x - 0.3*y + 0.07*x*x +
                             0.02*x*y - 0.03*y*y + 0.004*x*x*x -
                             0.002*y*y*y;
    }
  }
  Kokkos::deep_copy(coarse, host);
  Kokkos::View<int *> valid("O4 positive chi gate result", 1);
  Kokkos::parallel_for(
      "O4 positive chi union", Kokkos::RangePolicy<>(0, 1),
      KOKKOS_LAMBDA(const int) {
        valid(0) = ProlongationParentStencilFinitePositive<3>(
                       0, 0, 0, center, center, 1, coarse) ? 1 : 0;
      });
  const auto valid_host =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), valid);
  if (valid_host(0) != 1) {
    std::cerr << "O4 positive chi union rejected valid stencil\n";
    return false;
  }
  return true;
}

bool CheckO4FourAllocatedGhostLayers() {
  constexpr int n = 7;
  constexpr int first_parent = 2;
  constexpr int nx = 16;
  const auto weights = MakeProlongationWeights<3>();
  DvceArray5D<Real> coarse("O4 four-layer coarse support", 1, 1, 1, n, n);
  DvceArray5D<Real> fine("O4 four allocated fine ghost layers", 1, 1, 1, 4, 4);
  auto host = Kokkos::create_mirror_view(coarse);
  for (int j = 0; j < n; ++j) {
    const Real y = 2.0 * static_cast<Real>(j - first_parent);
    for (int i = 0; i < n; ++i) {
      const Real x = 2.0 * static_cast<Real>(i - first_parent);
      host(0, 0, 0, j, i) = 5.0 + 0.2*x - 0.1*y + 0.03*x*x +
                             0.02*x*y - 0.01*y*y + 0.001*x*x*x;
    }
  }
  Kokkos::deep_copy(coarse, host);
  Kokkos::deep_copy(fine, std::numeric_limits<Real>::quiet_NaN());
  Kokkos::parallel_for(
      "O4 populate four allocated ghost layers",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {2, 2}),
      KOKKOS_LAMBDA(const int parent_j, const int parent_i) {
        HighOrderProlongCC<3>(0, 0, 0, first_parent + parent_j,
                              first_parent + parent_i, 0, 2*parent_j,
                              2*parent_i, nx, nx, 1, coarse, fine, weights);
      });
  const auto result =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), fine);
  for (int j = 0; j < 4; ++j) {
    const Real y = static_cast<Real>(j) - 0.5;
    for (int i = 0; i < 4; ++i) {
      const Real x = static_cast<Real>(i) - 0.5;
      const Real expected = 5.0 + 0.2*x - 0.1*y + 0.03*x*x +
                            0.02*x*y - 0.01*y*y + 0.001*x*x*x;
      if (!std::isfinite(result(0, 0, 0, j, i)) ||
          !NearlyEqual(result(0, 0, 0, j, i), expected, 4.0e-12)) {
        std::cerr << "O4 allocated ghost layer was not populated exactly: j="
                  << j << " i=" << i << "\n";
        return false;
      }
    }
  }
  return true;
}

bool CheckProjectedActiveStateFeedsCacheAndGhosts() {
  constexpr int nx = 16;
  constexpr int ng = 4;
  constexpr int extent = nx + 2 * ng;
  constexpr int coarse_n = nx / 2;
  const auto restriction = MakeRestrictionWeights();
  const auto prolongation = MakeProlongationWeights<3>();
  DvceArray5D<Real> raw("O4 raw accepted active", 1, 1, 1, extent, extent);
  DvceArray5D<Real> projected("O4 projected accepted active", 1, 1, 1,
                              extent, extent);
  DvceArray5D<Real> raw_coarse("O4 raw coarse cache", 1, 1, 1,
                               coarse_n, coarse_n);
  DvceArray5D<Real> projected_coarse("O4 projected coarse cache", 1, 1, 1,
                                     coarse_n, coarse_n);
  DvceArray5D<Real> raw_ghost("O4 raw derived ghosts", 1, 1, 1, 2, 2);
  DvceArray5D<Real> projected_ghost("O4 projected derived ghosts", 1, 1, 1, 2, 2);
  auto raw_host = Kokkos::create_mirror_view(raw);
  auto projected_host = Kokkos::create_mirror_view(projected);
  for (int j = 0; j < extent; ++j) {
    for (int i = 0; i < extent; ++i) {
      raw_host(0, 0, 0, j, i) = std::numeric_limits<Real>::quiet_NaN();
      projected_host(0, 0, 0, j, i) =
          std::numeric_limits<Real>::quiet_NaN();
    }
  }
  for (int j = ng; j < ng + nx; ++j) {
    const Real y = static_cast<Real>(j - ng);
    for (int i = ng; i < ng + nx; ++i) {
      const Real x = static_cast<Real>(i - ng);
      Real metric[6] = {2.0 + 0.01*x + 0.002*y, 0.003, -0.002,
                        1.4 + 0.004*y, 0.001, 0.9 + 0.003*x};
      Real atracefree[6] = {0.08, 0.01, -0.005, -0.03, 0.004, 0.02};
      const Real before = metric[0];
      if (!z4c::ProjectAdmissibleConformalState(metric, atracefree)) return false;
      raw_host(0, 0, 0, j, i) = before;
      projected_host(0, 0, 0, j, i) = metric[0];
    }
  }
  Kokkos::deep_copy(raw, raw_host);
  Kokkos::deep_copy(projected, projected_host);
  Kokkos::parallel_for(
      "O4 restrict projected accepted active state",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {coarse_n, coarse_n}),
      KOKKOS_LAMBDA(const int cj, const int ci) {
        const int fj = ng + 2*cj;
        const int fi = ng + 2*ci;
        raw_coarse(0, 0, 0, cj, ci) = RestrictInterpolation<3>(
            0, 0, 0, fj, fi, nx, nx, 1, raw, restriction.second,
            restriction.fourth, restriction.fourth_edge);
        projected_coarse(0, 0, 0, cj, ci) = RestrictInterpolation<3>(
            0, 0, 0, fj, fi, nx, nx, 1, projected, restriction.second,
            restriction.fourth, restriction.fourth_edge);
      });
  Kokkos::parallel_for(
      "O4 prolong projected coarse cache into ghosts", Kokkos::RangePolicy<>(0, 1),
      KOKKOS_LAMBDA(const int) {
        HighOrderProlongCC<3>(0, 0, 0, 3, 3, 0, 0, 0, nx, nx, 1,
                              raw_coarse, raw_ghost, prolongation);
        HighOrderProlongCC<3>(0, 0, 0, 3, 3, 0, 0, 0, nx, nx, 1,
                              projected_coarse, projected_ghost, prolongation);
      });
  const auto raw_cache =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), raw_coarse);
  const auto projected_cache =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), projected_coarse);
  const auto raw_boundary =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), raw_ghost);
  const auto projected_boundary =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), projected_ghost);
  bool cache_changed = false;
  bool ghost_changed = false;
  for (int j = 0; j < coarse_n; ++j) {
    for (int i = 0; i < coarse_n; ++i) {
      if (!std::isfinite(projected_cache(0, 0, 0, j, i))) return false;
      cache_changed = cache_changed ||
          !NearlyEqual(raw_cache(0, 0, 0, j, i),
                       projected_cache(0, 0, 0, j, i), 1.0e-13);
    }
  }
  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < 2; ++i) {
      if (!std::isfinite(projected_boundary(0, 0, 0, j, i))) return false;
      ghost_changed = ghost_changed ||
          !NearlyEqual(raw_boundary(0, 0, 0, j, i),
                       projected_boundary(0, 0, 0, j, i), 1.0e-13);
    }
  }
  return cache_changed && ghost_changed;
}

template <int NGHOST>
bool RunOrder() {
  if (!CheckAllComponentRestrictionAndProlongation<NGHOST>()) {
    std::cerr << "all-component transfer failed order=" << NGHOST << "\n";
    return false;
  }
  if (!CheckRepeatedRoundTrip<NGHOST>()) {
    std::cerr << "repeated transfer failed order=" << NGHOST << "\n";
    return false;
  }
  if (!CheckFourierSymbolSweep<NGHOST>()) {
    std::cerr << "Fourier transfer failed order=" << NGHOST << "\n";
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  bool passed = RunOrder<2>() && RunOrder<3>() && RunOrder<4>();
  if (!CheckO4RestrictionEdgesAndGhostPoisoning()) {
    std::cerr << "O4 active-edge restriction qualification failed\n";
    passed = false;
  }
  if (!CheckO4SymmetricProlongationAndChiUnion()) {
    std::cerr << "O4 reflection/chi-parent qualification failed\n";
    passed = false;
  }
  if (!CheckO4FourAllocatedGhostLayers()) {
    std::cerr << "O4 four-layer coarse-fine population failed\n";
    passed = false;
  }
  if (!CheckProjectedActiveStateFeedsCacheAndGhosts()) {
    std::cerr << "projected active state did not feed O4 cache/ghosts\n";
    passed = false;
  }
  Kokkos::finalize();
  if (!passed) {
    std::cerr << "Cartoon Z4c AMR transfer qualification failed\n";
    return 1;
  }
  std::cout << "Cartoon Z4c AMR transfer qualification passed\n";
  return 0;
}
