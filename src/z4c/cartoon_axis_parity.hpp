//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cartoon_axis_parity.hpp
//! \brief Compile-time geometry, parity, and regularity traits for half-plane SO(2).

#ifndef Z4C_CARTOON_AXIS_PARITY_HPP_
#define Z4C_CARTOON_AXIS_PARITY_HPP_

#include <Kokkos_Macros.hpp>

namespace z4c {

//! Physical component directions.  The explicit names prevent legacy X/Y/Z packing from
//! being mistaken for visual Cartesian ordering: (X,Y,Z)=(rho,axial,suppressed).
enum class AxisDirection : int {
  rho = 0,
  axial = 1,
  suppressed = 2,
};

enum class AxisParity : int {
  odd = -1,
  even = 1,
};

KOKKOS_INLINE_FUNCTION constexpr int AxisParitySign(const AxisParity parity) {
  return static_cast<int>(parity);
}

KOKKOS_INLINE_FUNCTION constexpr AxisParity MultiplyAxisParity(
    const AxisParity left, const AxisParity right) {
  return AxisParitySign(left) * AxisParitySign(right) < 0
             ? AxisParity::odd
             : AxisParity::even;
}

KOKKOS_INLINE_FUNCTION constexpr AxisParity DirectionAxisParity(
    const AxisDirection direction) {
  return direction == AxisDirection::axial ? AxisParity::even : AxisParity::odd;
}

KOKKOS_INLINE_FUNCTION constexpr AxisParity VectorAxisParity(
    const AxisDirection component) {
  return DirectionAxisParity(component);
}

KOKKOS_INLINE_FUNCTION constexpr AxisParity SymmetricTensorAxisParity(
    const AxisDirection first, const AxisDirection second) {
  return MultiplyAxisParity(DirectionAxisParity(first),
                            DirectionAxisParity(second));
}

//! Analytic regularity classes used by the suppressed-direction operators.  Parity is
//! necessary but cannot distinguish an O(1) even field from an O(rho^2) even field.
enum class AxisRegularityClass : int {
  even_scalar,
  odd_linear,
  even_quadratic_zero,
  tensor_planar_pair,
  tensor_swirl_pair,
};

//! Named mirror of the public Z4c packed layout.  A unit test binds every value to Z4c's
//! authoritative enum, so changing either layout without the other fails immediately.
enum class Z4cStateComponent : int {
  chi = 0,
  g_rhorho,
  g_rhoz,
  g_rhoy,
  g_zz,
  g_zy,
  g_yy,
  khat,
  a_rhorho,
  a_rhoz,
  a_rhoy,
  a_zz,
  a_zy,
  a_yy,
  gamma_rho,
  gamma_z,
  gamma_y,
  theta,
  alpha,
  beta_rho,
  beta_z,
  beta_y,
  b_rho,
  b_z,
  b_y,
  count,
};

KOKKOS_INLINE_FUNCTION constexpr bool IsValidZ4cStateComponent(
    const Z4cStateComponent component) {
  const int index = static_cast<int>(component);
  return index >= 0 && index < static_cast<int>(Z4cStateComponent::count);
}

KOKKOS_INLINE_FUNCTION constexpr AxisParity Z4cStateAxisParity(
    const Z4cStateComponent component) {
  switch (component) {
    case Z4cStateComponent::g_rhoz:
    case Z4cStateComponent::g_zy:
    case Z4cStateComponent::a_rhoz:
    case Z4cStateComponent::a_zy:
    case Z4cStateComponent::gamma_rho:
    case Z4cStateComponent::gamma_y:
    case Z4cStateComponent::beta_rho:
    case Z4cStateComponent::beta_y:
    case Z4cStateComponent::b_rho:
    case Z4cStateComponent::b_y:
      return AxisParity::odd;
    case Z4cStateComponent::chi:
    case Z4cStateComponent::g_rhorho:
    case Z4cStateComponent::g_rhoy:
    case Z4cStateComponent::g_zz:
    case Z4cStateComponent::g_yy:
    case Z4cStateComponent::khat:
    case Z4cStateComponent::a_rhorho:
    case Z4cStateComponent::a_rhoy:
    case Z4cStateComponent::a_zz:
    case Z4cStateComponent::a_yy:
    case Z4cStateComponent::gamma_z:
    case Z4cStateComponent::theta:
    case Z4cStateComponent::alpha:
    case Z4cStateComponent::beta_z:
    case Z4cStateComponent::b_z:
      return AxisParity::even;
    case Z4cStateComponent::count:
      break;
  }
  return AxisParity::even;
}

KOKKOS_INLINE_FUNCTION constexpr int Z4cStateAxisParitySignFromPackedIndex(
    const int component) {
  return component >= 0 && component < static_cast<int>(Z4cStateComponent::count)
             ? AxisParitySign(
                   Z4cStateAxisParity(static_cast<Z4cStateComponent>(component)))
             : 0;
}

enum class AdmStateComponent : int {
  g_rhorho = 0,
  g_rhoz,
  g_rhoy,
  g_zz,
  g_zy,
  g_yy,
  k_rhorho,
  k_rhoz,
  k_rhoy,
  k_zz,
  k_zy,
  k_yy,
  psi4,
  alpha,
  beta_rho,
  beta_z,
  beta_y,
  count,
};

KOKKOS_INLINE_FUNCTION constexpr bool IsValidAdmStateComponent(
    const AdmStateComponent component) {
  const int index = static_cast<int>(component);
  return index >= 0 && index < static_cast<int>(AdmStateComponent::count);
}

KOKKOS_INLINE_FUNCTION constexpr AxisParity AdmStateAxisParity(
    const AdmStateComponent component) {
  switch (component) {
    case AdmStateComponent::g_rhoz:
    case AdmStateComponent::g_zy:
    case AdmStateComponent::k_rhoz:
    case AdmStateComponent::k_zy:
    case AdmStateComponent::beta_rho:
    case AdmStateComponent::beta_y:
      return AxisParity::odd;
    case AdmStateComponent::g_rhorho:
    case AdmStateComponent::g_rhoy:
    case AdmStateComponent::g_zz:
    case AdmStateComponent::g_yy:
    case AdmStateComponent::k_rhorho:
    case AdmStateComponent::k_rhoy:
    case AdmStateComponent::k_zz:
    case AdmStateComponent::k_yy:
    case AdmStateComponent::psi4:
    case AdmStateComponent::alpha:
    case AdmStateComponent::beta_z:
      return AxisParity::even;
    case AdmStateComponent::count:
      break;
  }
  return AxisParity::even;
}

KOKKOS_INLINE_FUNCTION constexpr int AdmStateAxisParitySignFromPackedIndex(
    const int component) {
  return component >= 0 && component < static_cast<int>(AdmStateComponent::count)
             ? AxisParitySign(
                   AdmStateAxisParity(static_cast<AdmStateComponent>(component)))
             : 0;
}

enum class ConstraintComponent : int {
  aggregate = 0,
  hamiltonian,
  momentum_norm_squared,
  z_norm_squared,
  momentum_rho,
  momentum_z,
  momentum_y,
  count,
};

KOKKOS_INLINE_FUNCTION constexpr bool IsValidConstraintComponent(
    const ConstraintComponent component) {
  const int index = static_cast<int>(component);
  return index >= 0 && index < static_cast<int>(ConstraintComponent::count);
}

KOKKOS_INLINE_FUNCTION constexpr AxisParity ConstraintAxisParity(
    const ConstraintComponent component) {
  return component == ConstraintComponent::momentum_rho ||
                 component == ConstraintComponent::momentum_y
             ? AxisParity::odd
             : AxisParity::even;
}

KOKKOS_INLINE_FUNCTION constexpr int ConstraintAxisParitySignFromPackedIndex(
    const int component) {
  return component >= 0 && component < static_cast<int>(ConstraintComponent::count)
             ? AxisParitySign(
                   ConstraintAxisParity(static_cast<ConstraintComponent>(component)))
             : 0;
}

static_assert(AxisParitySign(VectorAxisParity(AxisDirection::rho)) == -1);
static_assert(AxisParitySign(VectorAxisParity(AxisDirection::axial)) == 1);
static_assert(AxisParitySign(VectorAxisParity(AxisDirection::suppressed)) == -1);
static_assert(SymmetricTensorAxisParity(AxisDirection::rho,
                                        AxisDirection::suppressed) ==
              AxisParity::even);
static_assert(SymmetricTensorAxisParity(AxisDirection::rho,
                                        AxisDirection::axial) ==
              AxisParity::odd);
static_assert(Z4cStateAxisParitySignFromPackedIndex(-1) == 0);
static_assert(Z4cStateAxisParitySignFromPackedIndex(
                  static_cast<int>(Z4cStateComponent::count)) == 0);

}  // namespace z4c

#endif  // Z4C_CARTOON_AXIS_PARITY_HPP_
