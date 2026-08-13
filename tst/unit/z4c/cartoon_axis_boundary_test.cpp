//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cartoon_axis_boundary_test.cpp
//! \brief Device-capable bitwise tests for exact half-plane axis parity ghosts.

#include <cstdint>
#include <cstring>
#include <iostream>

#include <Kokkos_Core.hpp>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "mesh/prolongation.hpp"
#include "z4c/cartoon_axis_boundary.hpp"
#include "z4c/z4c.hpp"

namespace {

std::uint64_t Bits(const Real value) {
  std::uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

enum class AxisFieldFamily : int { z4c, adm, constraint };

template <AxisFieldFamily FAMILY>
KOKKOS_INLINE_FUNCTION bool FillPackedAxisGhostLine(
    const DvceArray5D<Real> &state, const int component,
    const int active_start, const int ghost_depth) {
  if constexpr (FAMILY == AxisFieldFamily::z4c) {
    return z4c::FillZ4cAxisGhostLine(state, 0, component, 0, 0,
                                    active_start, ghost_depth);
  } else if constexpr (FAMILY == AxisFieldFamily::adm) {
    return z4c::FillAdmAxisGhostLine(state, 0, component, 0, 0,
                                    active_start, ghost_depth);
  } else {
    return z4c::FillConstraintAxisGhostLine(state, 0, component, 0, 0,
                                           active_start, ghost_depth);
  }
}

template <AxisFieldFamily FAMILY>
constexpr int PackedAxisParitySign(const int component) {
  if constexpr (FAMILY == AxisFieldFamily::z4c) {
    return z4c::Z4cStateAxisParitySignFromPackedIndex(component);
  } else if constexpr (FAMILY == AxisFieldFamily::adm) {
    return z4c::AdmStateAxisParitySignFromPackedIndex(component);
  } else {
    return z4c::ConstraintAxisParitySignFromPackedIndex(component);
  }
}

template <int NCOMPONENTS, AxisFieldFamily FAMILY>
bool CheckComponents(const int ghost_depth) {
  const int active_cells = ghost_depth + 2;
  const int radial_cells = active_cells + ghost_depth;
  const int active_start = ghost_depth;
  DvceArray5D<Real> state("axis parity state", 1, NCOMPONENTS, 1, 1,
                          radial_cells);
  auto host = Kokkos::create_mirror_view(state);
  constexpr Real sentinel = -987654.25;
  for (int component = 0; component < NCOMPONENTS; ++component) {
    for (int i = 0; i < radial_cells; ++i) host(0, component, 0, 0, i) = sentinel;
    for (int offset = 0; offset < active_cells; ++offset) {
      host(0, component, 0, 0, active_start + offset) =
          static_cast<Real>(1000 * (component + 1) + 17 * offset) + 0.25;
    }
  }
  Kokkos::deep_copy(state, host);

  Kokkos::parallel_for(
      "fill exact axis parity ghosts", Kokkos::RangePolicy<DevExeSpace>(0, NCOMPONENTS),
      KOKKOS_LAMBDA(const int component) {
        if (!FillPackedAxisGhostLine<FAMILY>(state, component, active_start,
                                             ghost_depth)) {
          Kokkos::abort("valid packed component lacks axis parity");
        }
      });
  Kokkos::fence();
  Kokkos::deep_copy(host, state);

  for (int component = 0; component < NCOMPONENTS; ++component) {
    const int sign = PackedAxisParitySign<FAMILY>(component);
    if (sign != -1 && sign != 1) return false;
    for (int depth = 0; depth < ghost_depth; ++depth) {
      const Real source = host(0, component, 0, 0,
                               z4c::AxisMirrorActiveIndex(active_start, depth));
      const Real expected = static_cast<Real>(sign) * source;
      const Real actual = host(0, component, 0, 0,
                               z4c::AxisGhostIndex(active_start, depth));
      if (Bits(actual) != Bits(expected)) return false;
    }
    for (int offset = 0; offset < active_cells; ++offset) {
      const Real expected =
          static_cast<Real>(1000 * (component + 1) + 17 * offset) + 0.25;
      if (Bits(host(0, component, 0, 0, active_start + offset)) != Bits(expected)) {
        return false;
      }
    }
  }
  return true;
}

bool CheckRestrictionProlongationCommutation(const int parity_sign) {
  constexpr int coarse_active_start = 2;
  constexpr int fine_active_start = 2;
  DvceArray5D<Real> coarse("axis commutation coarse", 1, 1, 1, 1, 7);
  DvceArray5D<Real> fine("axis commutation fine", 1, 1, 1, 1, 8);
  auto coarse_host = Kokkos::create_mirror_view(coarse);
  for (int i = 0; i < 7; ++i) coarse_host(0, 0, 0, 0, i) = -999.0;
  for (int offset = 0; offset < 3; ++offset) {
    const Real x = static_cast<Real>(2 * offset + 1);
    coarse_host(0, 0, 0, 0, coarse_active_start + offset) =
        parity_sign > 0 ? x * x : x;
  }
  Kokkos::deep_copy(coarse, coarse_host);
  Kokkos::deep_copy(fine, -777.0);
  Kokkos::parallel_for(
      "axis restriction prolongation commutation",
      Kokkos::RangePolicy<DevExeSpace>(0, 1), KOKKOS_LAMBDA(const int) {
        if (!z4c::FillAxisGhostLine(coarse, 0, 0, 0, 0,
                                     coarse_active_start, 2, parity_sign)) {
          Kokkos::abort("valid commutation parity rejected");
        }
        // Prolong the innermost coarse ghost and active cell.  Fine ghost
        // children occupy [0,1], while active children occupy [2,3].
        ProlongCC(0, 0, 0, 0, coarse_active_start - 1,
                  0, 0, fine_active_start - 2, false, false, coarse, fine);
        ProlongCC(0, 0, 0, 0, coarse_active_start,
                  0, 0, fine_active_start, false, false, coarse, fine);
      });
  Kokkos::fence();
  auto fine_host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), fine);

  for (int depth = 0; depth < 2; ++depth) {
    const Real ghost = fine_host(0, 0, 0, 0, fine_active_start - depth - 1);
    const Real active = fine_host(0, 0, 0, 0, fine_active_start + depth);
    if (Bits(ghost) != Bits(static_cast<Real>(parity_sign) * active)) return false;
  }
  // Cell-average restriction of the parity-prolonged fine pair must commute
  // exactly with parity extension and recover the parent averages.
  const Real restricted_ghost =
      0.5 * (fine_host(0, 0, 0, 0, 0) + fine_host(0, 0, 0, 0, 1));
  const Real restricted_active =
      0.5 * (fine_host(0, 0, 0, 0, 2) + fine_host(0, 0, 0, 0, 3));
  if (Bits(restricted_ghost) !=
      Bits(static_cast<Real>(parity_sign) * restricted_active)) return false;
  Kokkos::deep_copy(coarse_host, coarse);
  return Bits(restricted_active) ==
         Bits(coarse_host(0, 0, 0, 0, coarse_active_start));
}

}  // namespace

int main(int argc, char **argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  bool passed = true;
  for (const int ghost_depth : {2, 3, 4}) {
    passed &= CheckComponents<z4c::Z4c::nz4c, AxisFieldFamily::z4c>(ghost_depth);
    passed &= CheckComponents<adm::ADM::nadm, AxisFieldFamily::adm>(ghost_depth);
    passed &=
        CheckComponents<z4c::Z4c::ncon, AxisFieldFamily::constraint>(ghost_depth);
  }
  passed &= CheckRestrictionProlongationCommutation(1);
  passed &= CheckRestrictionProlongationCommutation(-1);
  if (!passed) {
    std::cerr << "half-plane axis ghost parity contract failed\n";
    return 1;
  }
  std::cout << "half-plane axis ghost parity contract passed\n";
  return 0;
}
