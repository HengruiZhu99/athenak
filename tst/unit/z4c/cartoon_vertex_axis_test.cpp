//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cartoon_vertex_axis_test.cpp
//! \brief Device-capable tests for a native evolved Cartoon rho=0 vertex.

#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>

#include <Kokkos_Core.hpp>

#include "athena.hpp"
#include "z4c/cartoon_axis_boundary.hpp"
#include "z4c/cartoon_axis_parity.hpp"
#include "z4c/cartoon_vertex_axis.hpp"

namespace {

std::uint64_t Bits(const Real value) {
  std::uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

bool IsScalarComponent(const int component) {
  using z4c::Z4cStateComponent;
  switch (static_cast<Z4cStateComponent>(component)) {
    case Z4cStateComponent::chi:
    case Z4cStateComponent::khat:
    case Z4cStateComponent::theta:
    case Z4cStateComponent::alpha:
      return true;
    default:
      return false;
  }
}

bool IsVectorComponent(const int component) {
  using z4c::Z4cStateComponent;
  switch (static_cast<Z4cStateComponent>(component)) {
    case Z4cStateComponent::gamma_rho:
    case Z4cStateComponent::gamma_z:
    case Z4cStateComponent::gamma_y:
    case Z4cStateComponent::beta_rho:
    case Z4cStateComponent::beta_z:
    case Z4cStateComponent::beta_y:
    case Z4cStateComponent::b_rho:
    case Z4cStateComponent::b_z:
    case Z4cStateComponent::b_y:
      return true;
    default:
      return false;
  }
}

bool IsTensorComponent(const int component) {
  return !IsScalarComponent(component) && !IsVectorComponent(component);
}

enum class VertexAxisFamily : int { adm, constraint };

template <VertexAxisFamily Family>
KOKKOS_INLINE_FUNCTION bool FillVertexAxisFamilyGhosts(
    const DvceArray5D<Real> &state, const int meshblock, const int component,
    const int k, const int j, const int active_start, const int ghost_depth) {
  if constexpr (Family == VertexAxisFamily::adm) {
    return z4c::FillCenteredAdmAxisGhostLine<z4c::VertexCenteredZ4c>(
        state, meshblock, component, k, j, active_start, ghost_depth);
  } else {
    return z4c::FillCenteredConstraintAxisGhostLine<z4c::VertexCenteredZ4c>(
        state, meshblock, component, k, j, active_start, ghost_depth);
  }
}

template <VertexAxisFamily Family>
constexpr int VertexAxisFamilyParity(const int component) {
  if constexpr (Family == VertexAxisFamily::adm) {
    return z4c::AdmStateAxisParitySignFromPackedIndex(component);
  } else {
    return z4c::ConstraintAxisParitySignFromPackedIndex(component);
  }
}

template <int Components, VertexAxisFamily Family>
bool CheckVertexPackedGhostParity(const int ghost_depth) {
  constexpr int meshblocks = 3;
  constexpr int n3 = 2;
  constexpr int n2 = 7;
  const int active_start = ghost_depth;
  const int active_points = ghost_depth + 3;
  const int radial_points = active_points + ghost_depth;
  DvceArray5D<Real> state("vertex packed axis poison", meshblocks, Components,
                          n3, n2, radial_points);
  auto host = Kokkos::create_mirror_view(state);
  const Real poison = std::numeric_limits<Real>::quiet_NaN();
  for (int m = 0; m < meshblocks; ++m) {
    for (int component = 0; component < Components; ++component) {
      for (int k = 0; k < n3; ++k) {
        for (int j = 0; j < n2; ++j) {
          for (int i = 0; i < radial_points; ++i) {
            host(m, component, k, j, i) = poison;
          }
          for (int offset = 0; offset < active_points; ++offset) {
            host(m, component, k, j, active_start + offset) =
                static_cast<Real>(100000 * (m + 1) + 1000 * (component + 1) +
                                  100 * k + 10 * j + offset) + 0.375;
          }
        }
      }
    }
  }
  Kokkos::deep_copy(state, host);
  Kokkos::parallel_for(
      "fill vertex packed axis poison",
      Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<4>>(
          {0, 0, 0, 0}, {meshblocks, Components, n3, n2}),
      KOKKOS_LAMBDA(const int m, const int component, const int k, const int j) {
        if (!FillVertexAxisFamilyGhosts<Family>(
                state, m, component, k, j, active_start, ghost_depth)) {
          Kokkos::abort("valid vertex packed component lacks parity");
        }
      });
  Kokkos::fence();
  Kokkos::deep_copy(host, state);

  for (int m = 0; m < meshblocks; ++m) {
    for (int component = 0; component < Components; ++component) {
      const int sign = VertexAxisFamilyParity<Family>(component);
      if (sign != -1 && sign != 1) return false;
      for (int k = 0; k < n3; ++k) {
        for (int j = 0; j < n2; ++j) {
          for (int depth = 0; depth < ghost_depth; ++depth) {
            const int source =
                z4c::VertexAxisMirrorActiveIndex(active_start, depth);
            const int target = z4c::AxisGhostIndex(active_start, depth);
            const Real expected =
                static_cast<Real>(sign) * host(m, component, k, j, source);
            if (Bits(host(m, component, k, j, target)) != Bits(expected)) {
              return false;
            }
          }
          for (int offset = 0; offset < active_points; ++offset) {
            const Real expected =
                static_cast<Real>(100000 * (m + 1) + 1000 * (component + 1) +
                                  100 * k + 10 * j + offset) + 0.375;
            if (Bits(host(m, component, k, j, active_start + offset)) !=
                Bits(expected)) {
              return false;
            }
          }
        }
      }
    }
  }
  return true;
}

template <typename Predicate>
bool CheckVertexGhostParity(const int ghost_depth, Predicate selected) {
  constexpr int components = static_cast<int>(z4c::Z4cStateComponent::count);
  const int active_start = ghost_depth;
  const int active_points = ghost_depth + 3;
  const int radial_points = active_points + ghost_depth;
  DvceArray5D<Real> state("vertex axis parity", 1, components, 1, 1,
                          radial_points);
  auto host = Kokkos::create_mirror_view(state);
  const Real sentinel = std::numeric_limits<Real>::quiet_NaN();
  for (int component = 0; component < components; ++component) {
    for (int i = 0; i < radial_points; ++i) {
      host(0, component, 0, 0, i) = sentinel;
    }
    for (int offset = 0; offset < active_points; ++offset) {
      host(0, component, 0, 0, active_start + offset) =
          static_cast<Real>(1000 * (component + 1) + 19 * offset) + 0.375;
    }
  }
  Kokkos::deep_copy(state, host);
  Kokkos::parallel_for(
      "fill vertex axis parity ghosts",
      Kokkos::RangePolicy<DevExeSpace>(0, components),
      KOKKOS_LAMBDA(const int component) {
        if (!z4c::FillCenteredZ4cAxisGhostLine<z4c::VertexCenteredZ4c>(
                state, 0, component, 0, 0, active_start, ghost_depth)) {
          Kokkos::abort("valid vertex-centered component lacks parity");
        }
      });
  Kokkos::fence();
  Kokkos::deep_copy(host, state);

  for (int component = 0; component < components; ++component) {
    if (!selected(component)) continue;
    const int sign = z4c::Z4cStateAxisParitySignFromPackedIndex(component);
    for (int depth = 0; depth < ghost_depth; ++depth) {
      const int source =
          z4c::VertexAxisMirrorActiveIndex(active_start, depth);
      const int target = z4c::AxisGhostIndex(active_start, depth);
      if (Bits(host(0, component, 0, 0, target)) !=
          Bits(static_cast<Real>(sign) * host(0, component, 0, 0, source))) {
        return false;
      }
    }
    // The evolved axis node and every positive-rho active vertex remain bitwise intact.
    for (int offset = 0; offset < active_points; ++offset) {
      const Real expected =
          static_cast<Real>(1000 * (component + 1) + 19 * offset) + 0.375;
      if (Bits(host(0, component, 0, 0, active_start + offset)) != Bits(expected)) {
        return false;
      }
    }
  }
  return true;
}

bool CheckAxisStateAndRhsRegularity() {
  constexpr int components = static_cast<int>(z4c::Z4cStateComponent::count);
  constexpr int axis = 4;
  DvceArray5D<Real> state("vertex axis regularity", 1, components, 1, 1, 9);
  auto host = Kokkos::create_mirror_view(state);
  for (int component = 0; component < components; ++component) {
    for (int i = 0; i < 9; ++i) {
      host(0, component, 0, 0, i) =
          static_cast<Real>(100 * (component + 1) + i) + 0.125;
    }
  }
  const auto before = Kokkos::create_mirror_view(state);
  Kokkos::deep_copy(state, host);
  Kokkos::deep_copy(before, state);
  Kokkos::parallel_for(
      "enforce vertex axis regularity", Kokkos::RangePolicy<DevExeSpace>(0, 1),
      KOKKOS_LAMBDA(const int) {
        (void)z4c::EnforceVertexAxisZ4cPoint(state, 0, 0, 0, axis);
      });
  Kokkos::fence();
  Kokkos::deep_copy(host, state);

  using z4c::Z4cStateComponent;
  const int grr = static_cast<int>(Z4cStateComponent::g_rhorho);
  const int gyy = static_cast<int>(Z4cStateComponent::g_yy);
  const int arr = static_cast<int>(Z4cStateComponent::a_rhorho);
  const int ayy = static_cast<int>(Z4cStateComponent::a_yy);
  const Real metric_average =
      0.5 * (before(0, grr, 0, 0, axis) + before(0, gyy, 0, 0, axis));
  const Real atilde_average =
      0.5 * (before(0, arr, 0, 0, axis) + before(0, ayy, 0, 0, axis));
  if (Bits(host(0, grr, 0, 0, axis)) != Bits(metric_average) ||
      Bits(host(0, gyy, 0, 0, axis)) != Bits(metric_average) ||
      Bits(host(0, arr, 0, 0, axis)) != Bits(atilde_average) ||
      Bits(host(0, ayy, 0, 0, axis)) != Bits(atilde_average)) {
    return false;
  }
  constexpr Z4cStateComponent zero_components[] = {
      Z4cStateComponent::g_rhoz, Z4cStateComponent::g_rhoy,
      Z4cStateComponent::g_zy, Z4cStateComponent::a_rhoz,
      Z4cStateComponent::a_rhoy, Z4cStateComponent::a_zy,
      Z4cStateComponent::gamma_rho, Z4cStateComponent::gamma_y,
      Z4cStateComponent::beta_rho, Z4cStateComponent::beta_y,
      Z4cStateComponent::b_rho, Z4cStateComponent::b_y};
  bool is_zero_component[components] = {};
  for (const auto component : zero_components) {
    const int index = static_cast<int>(component);
    is_zero_component[index] = true;
    if (Bits(host(0, index, 0, 0, axis)) != Bits(0.0)) return false;
  }
  for (int component = 0; component < components; ++component) {
    if (component == grr || component == gyy || component == arr ||
        component == ayy || is_zero_component[component]) {
      continue;
    }
    if (Bits(host(0, component, 0, 0, axis)) !=
        Bits(before(0, component, 0, 0, axis))) {
      return false;
    }
  }
  // State/RHS regularity is a projection and must therefore be bitwise idempotent.
  Kokkos::parallel_for(
      "repeat vertex axis regularity", Kokkos::RangePolicy<DevExeSpace>(0, 1),
      KOKKOS_LAMBDA(const int) {
        (void)z4c::EnforceVertexAxisZ4cPoint(state, 0, 0, 0, axis);
      });
  Kokkos::fence();
  const auto repeated =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), state);
  for (int component = 0; component < components; ++component) {
    if (Bits(repeated(0, component, 0, 0, axis)) !=
        Bits(host(0, component, 0, 0, axis))) {
      return false;
    }
  }

  // The register-local projection used by native-VC Cartoon KO must enforce
  // the same representation and leave every unconstrained component intact.
  Kokkos::View<Real *> local_projection("local axis projection", components);
  Kokkos::parallel_for(
      "project local axis values", Kokkos::RangePolicy<DevExeSpace>(0, 1),
      KOKKOS_LAMBDA(const int) {
        Real values[components];
        for (int component = 0; component < components; ++component) {
          values[component] = static_cast<Real>(17 * (component + 1)) + 0.25;
        }
        z4c::ProjectVertexAxisZ4cValues(values);
        for (int component = 0; component < components; ++component) {
          local_projection(component) = values[component];
        }
      });
  Kokkos::fence();
  const auto local_host =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), local_projection);
  const Real local_metric_average =
      0.5 * ((17.0 * (grr + 1) + 0.25) + (17.0 * (gyy + 1) + 0.25));
  const Real local_atilde_average =
      0.5 * ((17.0 * (arr + 1) + 0.25) + (17.0 * (ayy + 1) + 0.25));
  if (Bits(local_host(grr)) != Bits(local_metric_average) ||
      Bits(local_host(gyy)) != Bits(local_metric_average) ||
      Bits(local_host(arr)) != Bits(local_atilde_average) ||
      Bits(local_host(ayy)) != Bits(local_atilde_average)) {
    return false;
  }
  for (int component = 0; component < components; ++component) {
    if (is_zero_component[component]) {
      if (Bits(local_host(component)) != Bits(0.0)) return false;
    } else if (component != grr && component != gyy && component != arr &&
               component != ayy) {
      const Real expected = static_cast<Real>(17 * (component + 1)) + 0.25;
      if (Bits(local_host(component)) != Bits(expected)) return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  if (argc != 2) {
    std::cerr << "expected scalar, vector, tensor, adm, constraint, or "
                 "rhs_regularity\n";
    return 2;
  }
  const std::string mode(argv[1]);
  bool passed = true;
  if (mode == "scalar") {
    for (const int ng : {2, 3, 4}) passed &= CheckVertexGhostParity(ng, IsScalarComponent);
  } else if (mode == "vector") {
    for (const int ng : {2, 3, 4}) passed &= CheckVertexGhostParity(ng, IsVectorComponent);
  } else if (mode == "tensor") {
    for (const int ng : {2, 3, 4}) passed &= CheckVertexGhostParity(ng, IsTensorComponent);
  } else if (mode == "adm") {
    for (const int ng : {2, 3, 4}) {
      passed &= CheckVertexPackedGhostParity<
          static_cast<int>(z4c::AdmStateComponent::count),
          VertexAxisFamily::adm>(ng);
    }
  } else if (mode == "constraint") {
    for (const int ng : {2, 3, 4}) {
      passed &= CheckVertexPackedGhostParity<
          static_cast<int>(z4c::ConstraintComponent::count),
          VertexAxisFamily::constraint>(ng);
    }
  } else if (mode == "rhs_regularity") {
    passed = CheckAxisStateAndRhsRegularity();
  } else {
    std::cerr << "unknown mode " << mode << '\n';
    return 2;
  }
  if (!passed) {
    std::cerr << "vertex Cartoon axis contract failed for " << mode << '\n';
    return 1;
  }
  std::cout << "vertex Cartoon axis contract passed for " << mode << '\n';
  return 0;
}
