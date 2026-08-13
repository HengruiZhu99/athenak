//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cartoon_derivatives.hpp
//! \brief Tensor-aware Cartesian and SO(2) derivative providers for vacuum Z4c.

#ifndef Z4C_CARTOON_DERIVATIVES_HPP_
#define Z4C_CARTOON_DERIVATIVES_HPP_

#include <math.h>
#include <type_traits>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "utils/finite_diff.hpp"

namespace z4c {

//! Physical directions used by the half-plane SO(2) Cartoon mapping.
enum class CartoonDirection : int {
  rho = 0,        //!< x1: nonnegative cylindrical radius on the active plane
  z = 1,          //!< x2: physical symmetry-axis coordinate
  suppressed = 2 //!< x3: suppressed azimuthal Cartesian direction
};

//! Tags select separately compiled symmetry-policy kernels on the host.
struct Cartesian3D {};
struct CartoonSO2 {};

//! Tensor component tables are valid only when both indices have the same variance.
enum class TensorVariance { all_lower, all_upper };

//! Exact-axis diagnostic samples are distinct from nonzero production cell centers.
enum class CartoonAxisLocation { cell_centered, diagnostic_axis };

template <typename Symmetry, int NGHOST>
class DerivativeProvider;

//! Cartesian derivative policy.
//!
//! This specialization intentionally contains no replacement stencils. Each operation
//! delegates directly to AthenaK's generated finite-difference implementation so the
//! existing three-dimensional path retains exactly the same numerical operators.
template <int NGHOST>
class DerivativeProvider<Cartesian3D, NGHOST> {
 public:
  static_assert(NGHOST >= 2 && NGHOST <= 4,
                "Z4c derivatives support NGHOST=2, 3, or 4");

  KOKKOS_INLINE_FUNCTION
  DerivativeProvider(const Real inverse_spacing[3], const int m, const int k,
                     const int j, const int i)
      : m_(m), k_(k), j_(j), i_(i) {
    for (int d = 0; d < 3; ++d) {
      inverse_spacing_[d] = inverse_spacing[d];
    }
  }

  template <typename ScalarField>
  KOKKOS_INLINE_FUNCTION Real ScalarFirst(const int derivative_direction,
                                          ScalarField &field) const {
    return Dx<NGHOST>(derivative_direction, inverse_spacing_, field, m_, k_, j_, i_);
  }

  template <typename ScalarField>
  KOKKOS_INLINE_FUNCTION Real ScalarSecond(const int first_direction,
                                           const int second_direction,
                                           ScalarField &field) const {
    if (first_direction == second_direction) {
      return Dxx<NGHOST>(first_direction, inverse_spacing_, field, m_, k_, j_, i_);
    }
    return Dxy<NGHOST>(first_direction, second_direction, inverse_spacing_, field,
                       m_, k_, j_, i_);
  }

  template <typename VectorField>
  KOKKOS_INLINE_FUNCTION Real VectorFirst(const int derivative_direction,
                                          const int component,
                                          VectorField &field) const {
    return Dx<NGHOST>(derivative_direction, inverse_spacing_, field, m_, component,
                      k_, j_, i_);
  }

  template <typename VectorField>
  KOKKOS_INLINE_FUNCTION Real VectorSecond(const int first_direction,
                                           const int second_direction,
                                           const int component,
                                           VectorField &field) const {
    if (first_direction == second_direction) {
      return Dxx<NGHOST>(first_direction, inverse_spacing_, field, m_, component,
                         k_, j_, i_);
    }
    return Dxy<NGHOST>(first_direction, second_direction, inverse_spacing_, field,
                       m_, component, k_, j_, i_);
  }

  //! `Variance` must describe an all-lower or all-upper symmetric Cartesian tensor.
  //! Mixed-index tensors require their own generator action and cannot use this table.
  template <TensorVariance Variance, typename TensorField>
  KOKKOS_INLINE_FUNCTION Real TensorFirst(const int derivative_direction,
                                          const int first_component,
                                          const int second_component,
                                          TensorField &field) const {
    return Dx<NGHOST>(derivative_direction, inverse_spacing_, field, m_,
                      first_component, second_component, k_, j_, i_);
  }

  //! `Variance` must describe an all-lower or all-upper symmetric Cartesian tensor.
  template <TensorVariance Variance, typename TensorField>
  KOKKOS_INLINE_FUNCTION Real TensorSecond(const int first_direction,
                                           const int second_direction,
                                           const int first_component,
                                           const int second_component,
                                           TensorField &field) const {
    if (first_direction == second_direction) {
      return Dxx<NGHOST>(first_direction, inverse_spacing_, field, m_,
                         first_component, second_component, k_, j_, i_);
    }
    return Dxy<NGHOST>(first_direction, second_direction, inverse_spacing_, field,
                       m_, first_component, second_component, k_, j_, i_);
  }

  template <typename VectorField>
  KOKKOS_INLINE_FUNCTION Real VectorDivergence(VectorField &field) const {
    Real divergence = 0.0;
    for (int d = 0; d < 3; ++d) {
      divergence += VectorFirst(d, d, field);
    }
    return divergence;
  }

  template <typename VelocityField, typename ScalarField>
  KOKKOS_INLINE_FUNCTION Real ScalarAdvective(const VelocityField &velocity,
                                              const ScalarField &field) const {
    Real derivative = 0.0;
    for (int d = 0; d < 3; ++d) {
      derivative += Lx<NGHOST>(d, inverse_spacing_, velocity, field, m_, d,
                               k_, j_, i_);
    }
    return derivative;
  }

  template <typename VelocityField, typename VectorField>
  KOKKOS_INLINE_FUNCTION Real VectorAdvective(const int component,
                                              const VelocityField &velocity,
                                              const VectorField &field) const {
    Real derivative = 0.0;
    for (int d = 0; d < 3; ++d) {
      derivative += Lx<NGHOST>(d, inverse_spacing_, velocity, field, m_, d,
                               component, k_, j_, i_);
    }
    return derivative;
  }

  //! `Variance` must describe an all-lower or all-upper symmetric Cartesian tensor.
  template <TensorVariance Variance, typename VelocityField, typename TensorField>
  KOKKOS_INLINE_FUNCTION Real TensorAdvective(const int first_component,
                                              const int second_component,
                                              const VelocityField &velocity,
                                              const TensorField &field) const {
    Real derivative = 0.0;
    for (int d = 0; d < 3; ++d) {
      derivative += Lx<NGHOST>(d, inverse_spacing_, velocity, field, m_, d,
                               first_component, second_component, k_, j_, i_);
    }
    return derivative;
  }

  template <typename ComponentField>
  KOKKOS_INLINE_FUNCTION Real DirectionalComponentDissipation(
      const int direction, const int component, ComponentField &field) const {
    return Diss<NGHOST>(direction, inverse_spacing_, field, m_, component,
                        k_, j_, i_);
  }

  template <typename ComponentField>
  KOKKOS_INLINE_FUNCTION Real ComponentDissipation(const int component,
                                                   ComponentField &field) const {
    Real dissipation = 0.0;
    for (int d = 0; d < 3; ++d) {
      dissipation += DirectionalComponentDissipation(d, component, field);
    }
    return dissipation;
  }

 private:
  Real inverse_spacing_[3];
  int m_;
  int k_;
  int j_;
  int i_;
};

//! Analytic SO(2) Cartoon derivative policy on the x1>=0 meridional half-plane.
//!
//! The formulas in this specialization come from the independent, production-code-free
//! derivation in `signed_rho_so2_identity_note.md`. They follow from the Killing relation
//! for xi=-y*d_x+x*d_y with component order (x,z,y)=(x1,x2,x3). Component-sensitive
//! manufactured oracles cover the signs, axis limits, parity, divergence, and suppressed
//! advection before a Cartoon problem generator may be enabled.
//!
//! Ordinary rho/z derivatives always use AthenaK's centered finite differences through
//! exact parity ghosts. Suppressed-direction combinations whose continuum form loses a
//! power of rho use fixed rational regularity functionals only in rows whose centered
//! stencil intersects the axis. The closure samples exactly `NGHOST` positive half-cell
//! radii and differentiates the known regular coefficient in s=rho^2. Its coefficients
//! are hard-coded: there is no runtime fit, matrix solve, allocation, or independently
//! evolved negative side. The first row with a wholly active centered stencil uses the
//! bulk analytic identity, so the legacy outer fitted row is absent.
template <int NGHOST>
class DerivativeProvider<CartoonSO2, NGHOST> {
 public:
  static_assert(NGHOST >= 2 && NGHOST <= 4,
                "Z4c derivatives support NGHOST=2, 3, or 4");

  KOKKOS_INLINE_FUNCTION
  DerivativeProvider(const Real inverse_spacing[3], const Real rho,
                     const CartoonAxisLocation axis_location, const int m, const int k,
                     const int j, const int i)
      : rho_(rho), axis_location_(axis_location), m_(m), k_(k), j_(j), i_(i) {
    for (int d = 0; d < 3; ++d) {
      inverse_spacing_[d] = inverse_spacing[d];
    }
  }

  KOKKOS_INLINE_FUNCTION static constexpr int ScalarParity() { return 1; }

  KOKKOS_INLINE_FUNCTION static constexpr int RegularizedHalfCellLayers() {
    return NGHOST - 1;
  }

  KOKKOS_INLINE_FUNCTION static constexpr int MaximumRegularizationOffset() {
    return NGHOST - 1;
  }

  KOKKOS_INLINE_FUNCTION static constexpr int VectorParity(const int component) {
    return (component == ZDirection()) ? 1 : -1;
  }

  KOKKOS_INLINE_FUNCTION static constexpr int TensorParity(const int first_component,
                                                           const int second_component) {
    return VectorParity(first_component) * VectorParity(second_component);
  }

  template <typename ScalarField>
  KOKKOS_INLINE_FUNCTION Real ScalarFirst(const int derivative_direction,
                                          ScalarField &field) const {
    if (derivative_direction == SuppressedDirection()) {
      return 0.0;
    }
    return ActiveFirst(derivative_direction, field);
  }

  template <typename ScalarField>
  KOKKOS_INLINE_FUNCTION Real ScalarSecond(const int first_direction,
                                           const int second_direction,
                                           ScalarField &field) const {
    if (first_direction != SuppressedDirection() &&
        second_direction != SuppressedDirection()) {
      return ActiveSecond(first_direction, second_direction, field);
    }
    if (first_direction != second_direction) {
      return 0.0;
    }
    if (OnAxis()) {
      return ActiveSecond(RhoDirection(), RhoDirection(), field);
    }
    if (NearAxisCell()) {
      return 2.0 * EvenCoefficientDerivative(field);
    }
    return ActiveFirst(RhoDirection(), field) / rho_;
  }

  template <typename VectorField>
  KOKKOS_INLINE_FUNCTION Real VectorFirst(const int derivative_direction,
                                          const int component,
                                          VectorField &field) const {
    if (derivative_direction != SuppressedDirection()) {
      return ActiveFirst(derivative_direction, component, field);
    }
    if (component == ZDirection()) {
      return 0.0;
    }
    if (OnAxis()) {
      if (component == RhoDirection()) {
        return -ActiveFirst(RhoDirection(), SuppressedDirection(), field);
      }
      return ActiveFirst(RhoDirection(), RhoDirection(), field);
    }
    if (component == RhoDirection()) {
      return -Value(field, SuppressedDirection()) / rho_;
    }
    return Value(field, RhoDirection()) / rho_;
  }

  template <typename VectorField>
  KOKKOS_INLINE_FUNCTION Real VectorSecond(const int first_direction,
                                           const int second_direction,
                                           const int component,
                                           VectorField &field) const {
    const bool first_suppressed = first_direction == SuppressedDirection();
    const bool second_suppressed = second_direction == SuppressedDirection();
    if (!first_suppressed && !second_suppressed) {
      return ActiveSecond(first_direction, second_direction, component, field);
    }
    if (first_suppressed && second_suppressed) {
      return VectorSecondSuppressed(component, field);
    }
    const int active_direction = first_suppressed ? second_direction : first_direction;
    return VectorMixedSuppressed(active_direction, component, field);
  }

  //! `Variance` must describe an all-lower or all-upper symmetric Cartesian tensor.
  //! Mixed-index tensors require their own generator action and cannot use this table.
  template <TensorVariance Variance, typename TensorField>
  KOKKOS_INLINE_FUNCTION Real TensorFirst(const int derivative_direction,
                                          const int first_component,
                                          const int second_component,
                                          TensorField &field) const {
    if (derivative_direction != SuppressedDirection()) {
      return ActiveFirst(derivative_direction, first_component, second_component, field);
    }
    const int a = first_component;
    const int b = second_component;
    if ((a == RhoDirection() && b == RhoDirection()) ||
        (a == SuppressedDirection() && b == SuppressedDirection())) {
      const Real sign = (a == RhoDirection()) ? -2.0 : 2.0;
      if (OnAxis()) {
        return 0.0;
      }
      return sign * Value(field, RhoDirection(), SuppressedDirection()) / rho_;
    }
    if (IsComponentPair(a, b, RhoDirection(), SuppressedDirection())) {
      if (OnAxis()) {
        return 0.0;
      }
      return (Value(field, RhoDirection(), RhoDirection()) -
              Value(field, SuppressedDirection(), SuppressedDirection())) / rho_;
    }
    if (IsComponentPair(a, b, RhoDirection(), ZDirection())) {
      if (OnAxis()) {
        return -ActiveFirst(RhoDirection(), SuppressedDirection(), ZDirection(), field);
      }
      return -Value(field, SuppressedDirection(), ZDirection()) / rho_;
    }
    if (IsComponentPair(a, b, SuppressedDirection(), ZDirection())) {
      if (OnAxis()) {
        return ActiveFirst(RhoDirection(), RhoDirection(), ZDirection(), field);
      }
      return Value(field, RhoDirection(), ZDirection()) / rho_;
    }
    return 0.0;
  }

  //! `Variance` must describe an all-lower or all-upper symmetric Cartesian tensor.
  template <TensorVariance Variance, typename TensorField>
  KOKKOS_INLINE_FUNCTION Real TensorSecond(const int first_direction,
                                           const int second_direction,
                                           const int first_component,
                                           const int second_component,
                                           TensorField &field) const {
    const bool first_suppressed = first_direction == SuppressedDirection();
    const bool second_suppressed = second_direction == SuppressedDirection();
    if (!first_suppressed && !second_suppressed) {
      return ActiveSecond(first_direction, second_direction, first_component,
                          second_component, field);
    }
    if (first_suppressed && second_suppressed) {
      return TensorSecondSuppressed(first_component, second_component, field);
    }
    const int active_direction = first_suppressed ? second_direction : first_direction;
    return TensorMixedSuppressed(active_direction, first_component, second_component,
                                 field);
  }

  template <typename VectorField>
  KOKKOS_INLINE_FUNCTION Real VectorDivergence(VectorField &field) const {
    if (OnAxis()) {
      return 2.0 * ActiveFirst(RhoDirection(), RhoDirection(), field) +
             ActiveFirst(ZDirection(), ZDirection(), field);
    }
    return ActiveFirst(RhoDirection(), RhoDirection(), field) +
           ActiveFirst(ZDirection(), ZDirection(), field) +
           Value(field, RhoDirection()) / rho_;
  }

  template <typename VelocityField, typename ScalarField>
  KOKKOS_INLINE_FUNCTION Real ScalarAdvective(const VelocityField &velocity,
                                              const ScalarField &field) const {
    Real derivative = 0.0;
    for (int d = RhoDirection(); d <= ZDirection(); ++d) {
      derivative += Lx<NGHOST>(d, inverse_spacing_, velocity, field, m_, d,
                               k_, j_, i_);
    }
    return derivative;
  }

  template <typename VelocityField, typename VectorField>
  KOKKOS_INLINE_FUNCTION Real VectorAdvective(const int component,
                                              const VelocityField &velocity,
                                              VectorField &field) const {
    Real derivative = 0.0;
    for (int d = RhoDirection(); d <= ZDirection(); ++d) {
      derivative += Lx<NGHOST>(d, inverse_spacing_, velocity, field, m_, d,
                               component, k_, j_, i_);
    }
    return derivative + Value(velocity, SuppressedDirection()) *
                            VectorFirst(SuppressedDirection(), component, field);
  }

  //! `Variance` must describe an all-lower or all-upper symmetric Cartesian tensor.
  template <TensorVariance Variance, typename VelocityField, typename TensorField>
  KOKKOS_INLINE_FUNCTION Real TensorAdvective(const int first_component,
                                              const int second_component,
                                              const VelocityField &velocity,
                                              TensorField &field) const {
    Real derivative = 0.0;
    for (int d = RhoDirection(); d <= ZDirection(); ++d) {
      derivative += Lx<NGHOST>(d, inverse_spacing_, velocity, field, m_, d,
                               first_component, second_component, k_, j_, i_);
    }
    return derivative + Value(velocity, SuppressedDirection()) *
                            TensorFirst<Variance>(SuppressedDirection(), first_component,
                                                  second_component, field);
  }

  template <typename ComponentField>
  KOKKOS_INLINE_FUNCTION Real DirectionalComponentDissipation(
      const int direction, const int component, ComponentField &field) const {
    if (direction == SuppressedDirection()) return 0.0;
    return Diss<NGHOST>(direction, inverse_spacing_, field, m_, component,
                        k_, j_, i_);
  }

  template <typename ComponentField>
  KOKKOS_INLINE_FUNCTION Real ComponentDissipation(const int component,
                                                   ComponentField &field) const {
    Real dissipation = 0.0;
    for (int d = 0; d < 3; ++d) {
      dissipation += DirectionalComponentDissipation(d, component, field);
    }
    return dissipation;
  }

 private:
  KOKKOS_INLINE_FUNCTION static constexpr int RhoDirection() {
    return static_cast<int>(CartoonDirection::rho);
  }

  KOKKOS_INLINE_FUNCTION static constexpr int ZDirection() {
    return static_cast<int>(CartoonDirection::z);
  }

  KOKKOS_INLINE_FUNCTION static constexpr int SuppressedDirection() {
    return static_cast<int>(CartoonDirection::suppressed);
  }

  KOKKOS_INLINE_FUNCTION bool OnAxis() const {
    return axis_location_ == CartoonAxisLocation::diagnostic_axis;
  }

  KOKKOS_INLINE_FUNCTION int NearestInteger(const Real value) const {
    return static_cast<int>(value + (value < 0.0 ? -0.5 : 0.5));
  }

  KOKKOS_INLINE_FUNCTION bool NearAxisCell() const {
    if (OnAxis()) return false;
    const Real half_cell_index = 2.0 * rho_ * inverse_spacing_[RhoDirection()];
    const int rounded_index = NearestInteger(half_cell_index);
    const Real grid_tolerance = (sizeof(Real) == sizeof(float)) ? 1.0e-4 : 1.0e-10;
    const bool is_half_cell = fabs(half_cell_index - rounded_index) <= grid_tolerance;
    const int absolute_index = rounded_index < 0 ? -rounded_index : rounded_index;
    return is_half_cell && (absolute_index % 2 == 1) &&
           absolute_index <= 2 * (NGHOST - 1) - 1;
  }

  KOKKOS_INLINE_FUNCTION static bool IsComponentPair(const int a, const int b,
                                                     const int c, const int d) {
    return (a == c && b == d) || (a == d && b == c);
  }

  KOKKOS_INLINE_FUNCTION int TargetSideSign() const {
    const Real half_cell_index = 2.0 * rho_ * inverse_spacing_[RhoDirection()];
    return NearestInteger(half_cell_index) < 0 ? -1 : 1;
  }

  KOKKOS_INLINE_FUNCTION int TargetLayer() const {
    const Real half_cell_index = 2.0 * rho_ * inverse_spacing_[RhoDirection()];
    const int rounded_index = NearestInteger(half_cell_index);
    const int absolute_index = rounded_index < 0 ? -rounded_index : rounded_index;
    return (absolute_index - 1) / 2;
  }

  KOKKOS_INLINE_FUNCTION int SideLayerIndex(const int layer) const {
    return i_ + TargetSideSign() * (layer - TargetLayer());
  }

  //! Differentiate a regular coefficient F(s) at the target half-cell.
  //!
  //! Samples are F((l+1/2)^2 h^2), l=0,...,NGHOST-1. These exact rational
  //! rows reproduce degree NGHOST-1 in s and give O2/O4/O6 accuracy for the
  //! singular-looking SO(2) combinations. Only target rows 0,...,NGHOST-2
  //! are reachable; the next row uses the wholly active bulk stencil.
  KOKKOS_INLINE_FUNCTION Real RegularCoefficientDerivative(
      const Real samples[NGHOST]) const {
    const int target = TargetLayer();
    Real derivative = 0.0;
    if constexpr (NGHOST == 2) {
      derivative = -0.5 * samples[0] + 0.5 * samples[1];
    } else if constexpr (NGHOST == 3) {
      if (target == 0) {
        derivative = -2.0 / 3.0 * samples[0] + 3.0 / 4.0 * samples[1] -
                     1.0 / 12.0 * samples[2];
      } else {
        derivative = -1.0 / 3.0 * samples[0] + 1.0 / 4.0 * samples[1] +
                     1.0 / 12.0 * samples[2];
      }
    } else if constexpr (NGHOST == 4) {
      if (target == 0) {
        derivative = -3.0 / 4.0 * samples[0] + 9.0 / 10.0 * samples[1] -
                     1.0 / 6.0 * samples[2] + 1.0 / 60.0 * samples[3];
      } else if (target == 1) {
        derivative = -5.0 / 18.0 * samples[0] + 3.0 / 20.0 * samples[1] +
                     5.0 / 36.0 * samples[2] - 1.0 / 90.0 * samples[3];
      } else {
        derivative = 1.0 / 6.0 * samples[0] - 9.0 / 20.0 * samples[1] +
                     1.0 / 4.0 * samples[2] + 1.0 / 30.0 * samples[3];
      }
    }
    return derivative * inverse_spacing_[RhoDirection()] *
           inverse_spacing_[RhoDirection()];
  }

  template <typename ScalarField>
  KOKKOS_INLINE_FUNCTION Real MappedScalar(const ScalarField &field,
                                           const int layer) const {
    return field(m_, k_, j_, SideLayerIndex(layer));
  }

  template <typename VectorField>
  KOKKOS_INLINE_FUNCTION Real MappedVector(const VectorField &field,
                                           const int component, const int layer) const {
    const int parity = TargetSideSign() < 0 ? VectorParity(component) : 1;
    return parity * field(m_, component, k_, j_, SideLayerIndex(layer));
  }

  template <typename TensorField>
  KOKKOS_INLINE_FUNCTION Real MappedTensor(const TensorField &field, const int a,
                                           const int b, const int layer) const {
    const int parity = TargetSideSign() < 0 ? TensorParity(a, b) : 1;
    return parity * field(m_, a, b, k_, j_, SideLayerIndex(layer));
  }

  template <typename ScalarField>
  KOKKOS_INLINE_FUNCTION Real EvenCoefficientDerivative(
      const ScalarField &field) const {
    Real samples[NGHOST];
    for (int layer = 0; layer < NGHOST; ++layer) {
      samples[layer] = MappedScalar(field, layer);
    }
    return RegularCoefficientDerivative(samples);
  }

  template <typename VectorField>
  KOKKOS_INLINE_FUNCTION Real EvenCoefficientDerivative(
      const VectorField &field, const int component) const {
    Real samples[NGHOST];
    for (int layer = 0; layer < NGHOST; ++layer) {
      samples[layer] = MappedVector(field, component, layer);
    }
    return RegularCoefficientDerivative(samples);
  }

  template <typename TensorField>
  KOKKOS_INLINE_FUNCTION Real EvenCoefficientDerivative(
      const TensorField &field, const int a, const int b) const {
    Real samples[NGHOST];
    for (int layer = 0; layer < NGHOST; ++layer) {
      samples[layer] = MappedTensor(field, a, b, layer);
    }
    return RegularCoefficientDerivative(samples);
  }

  template <typename VectorField>
  KOKKOS_INLINE_FUNCTION Real OddCoefficientDerivative(
      const VectorField &field, const int component) const {
    Real samples[NGHOST];
    const Real spacing = 1.0 / inverse_spacing_[RhoDirection()];
    for (int layer = 0; layer < NGHOST; ++layer) {
      const Real radius = (static_cast<Real>(layer) + 0.5) * spacing;
      samples[layer] = MappedVector(field, component, layer) / radius;
    }
    return RegularCoefficientDerivative(samples);
  }

  template <typename TensorField>
  KOKKOS_INLINE_FUNCTION Real OddCoefficientDerivative(
      const TensorField &field, const int a, const int b) const {
    Real samples[NGHOST];
    const Real spacing = 1.0 / inverse_spacing_[RhoDirection()];
    for (int layer = 0; layer < NGHOST; ++layer) {
      const Real radius = (static_cast<Real>(layer) + 0.5) * spacing;
      samples[layer] = MappedTensor(field, a, b, layer) / radius;
    }
    return RegularCoefficientDerivative(samples);
  }

  template <typename TensorField>
  KOKKOS_INLINE_FUNCTION Real QuadraticCoefficientDerivative(
      const TensorField &field, const int a, const int b) const {
    Real samples[NGHOST];
    const Real spacing = 1.0 / inverse_spacing_[RhoDirection()];
    for (int layer = 0; layer < NGHOST; ++layer) {
      const Real radius = (static_cast<Real>(layer) + 0.5) * spacing;
      samples[layer] = MappedTensor(field, a, b, layer) / (radius * radius);
    }
    return RegularCoefficientDerivative(samples);
  }

  template <typename TensorField>
  KOKKOS_INLINE_FUNCTION Real QuadraticDifferenceCoefficientDerivative(
      const TensorField &field, const int a, const int b, const int c,
      const int d) const {
    Real samples[NGHOST];
    const Real spacing = 1.0 / inverse_spacing_[RhoDirection()];
    for (int layer = 0; layer < NGHOST; ++layer) {
      const Real radius = (static_cast<Real>(layer) + 0.5) * spacing;
      samples[layer] =
          (MappedTensor(field, a, b, layer) - MappedTensor(field, c, d, layer)) /
          (radius * radius);
    }
    return RegularCoefficientDerivative(samples);
  }

  template <typename ScalarField>
  KOKKOS_INLINE_FUNCTION Real Value(const ScalarField &field) const {
    return field(m_, k_, j_, i_);
  }

  template <typename VectorField>
  KOKKOS_INLINE_FUNCTION Real Value(const VectorField &field, const int component) const {
    return field(m_, component, k_, j_, i_);
  }

  template <typename TensorField>
  KOKKOS_INLINE_FUNCTION Real Value(const TensorField &field, const int first_component,
                                    const int second_component) const {
    return field(m_, first_component, second_component, k_, j_, i_);
  }

  template <typename ScalarField>
  KOKKOS_INLINE_FUNCTION Real ActiveFirst(const int direction, ScalarField &field) const {
    return Dx<NGHOST>(direction, inverse_spacing_, field, m_, k_, j_, i_);
  }

  template <typename VectorField>
  KOKKOS_INLINE_FUNCTION Real ActiveFirst(const int direction, const int component,
                                          VectorField &field) const {
    return Dx<NGHOST>(direction, inverse_spacing_, field, m_, component, k_, j_, i_);
  }

  template <typename TensorField>
  KOKKOS_INLINE_FUNCTION Real ActiveFirst(const int direction, const int first_component,
                                          const int second_component,
                                          TensorField &field) const {
    return Dx<NGHOST>(direction, inverse_spacing_, field, m_, first_component,
                      second_component, k_, j_, i_);
  }

  template <typename ScalarField>
  KOKKOS_INLINE_FUNCTION Real ActiveSecond(const int first_direction,
                                           const int second_direction,
                                           ScalarField &field) const {
    if (first_direction == second_direction) {
      return Dxx<NGHOST>(first_direction, inverse_spacing_, field, m_, k_, j_, i_);
    }
    return Dxy<NGHOST>(first_direction, second_direction, inverse_spacing_, field,
                       m_, k_, j_, i_);
  }

  template <typename VectorField>
  KOKKOS_INLINE_FUNCTION Real ActiveSecond(const int first_direction,
                                           const int second_direction,
                                           const int component,
                                           VectorField &field) const {
    if (first_direction == second_direction) {
      return Dxx<NGHOST>(first_direction, inverse_spacing_, field, m_, component,
                         k_, j_, i_);
    }
    return Dxy<NGHOST>(first_direction, second_direction, inverse_spacing_, field,
                       m_, component, k_, j_, i_);
  }

  template <typename TensorField>
  KOKKOS_INLINE_FUNCTION Real ActiveSecond(const int first_direction,
                                           const int second_direction,
                                           const int first_component,
                                           const int second_component,
                                           TensorField &field) const {
    if (first_direction == second_direction) {
      return Dxx<NGHOST>(first_direction, inverse_spacing_, field, m_, first_component,
                         second_component, k_, j_, i_);
    }
    return Dxy<NGHOST>(first_direction, second_direction, inverse_spacing_, field,
                       m_, first_component, second_component, k_, j_, i_);
  }

  template <typename VectorField>
  KOKKOS_INLINE_FUNCTION Real VectorSecondSuppressed(const int component,
                                                     VectorField &field) const {
    if (OnAxis()) {
      if (component == ZDirection()) {
        return ActiveSecond(RhoDirection(), RhoDirection(), component, field);
      }
      return 0.0;
    }
    if (NearAxisCell()) {
      if (component == ZDirection()) {
        return 2.0 * EvenCoefficientDerivative(field, component);
      }
      return 2.0 * rho_ * OddCoefficientDerivative(field, component);
    }
    const Real radial_derivative = ActiveFirst(RhoDirection(), component, field);
    if (component == ZDirection()) {
      return radial_derivative / rho_;
    }
    return radial_derivative / rho_ - Value(field, component) / (rho_ * rho_);
  }

  template <typename VectorField>
  KOKKOS_INLINE_FUNCTION Real VectorMixedSuppressed(const int active_direction,
                                                    const int component,
                                                    VectorField &field) const {
    if (component == ZDirection()) {
      return 0.0;
    }
    if (OnAxis()) {
      if (active_direction == RhoDirection()) {
        return 0.0;
      }
      const int rotated_component = (component == RhoDirection())
                                        ? SuppressedDirection()
                                        : RhoDirection();
      const Real sign = (component == RhoDirection()) ? -1.0 : 1.0;
      return sign * ActiveSecond(RhoDirection(), ZDirection(), rotated_component, field);
    }
    const int rotated_component = (component == RhoDirection())
                                      ? SuppressedDirection()
                                      : RhoDirection();
    const Real sign = (component == RhoDirection()) ? -1.0 : 1.0;
    if (NearAxisCell()) {
      if (active_direction == RhoDirection()) {
        return sign * 2.0 * rho_ *
               OddCoefficientDerivative(field, rotated_component);
      }
      return sign * ActiveFirst(active_direction, rotated_component, field) / rho_;
    }
    const Real derivative = ActiveFirst(active_direction, rotated_component, field);
    Real result = sign * derivative / rho_;
    if (active_direction == RhoDirection()) {
      result -= sign * Value(field, rotated_component) / (rho_ * rho_);
    }
    return result;
  }

  template <typename TensorField>
  KOKKOS_INLINE_FUNCTION Real TensorSecondSuppressed(const int a, const int b,
                                                     TensorField &field) const {
    if (OnAxis()) {
      if (a == RhoDirection() && b == RhoDirection()) {
        return ActiveSecond(RhoDirection(), RhoDirection(),
                            SuppressedDirection(), SuppressedDirection(), field);
      }
      if (a == SuppressedDirection() && b == SuppressedDirection()) {
        return ActiveSecond(RhoDirection(), RhoDirection(),
                            RhoDirection(), RhoDirection(), field);
      }
      if (IsComponentPair(a, b, RhoDirection(), SuppressedDirection())) {
        return -ActiveSecond(RhoDirection(), RhoDirection(), RhoDirection(),
                             SuppressedDirection(), field);
      }
      if (a == ZDirection() && b == ZDirection()) {
        return ActiveSecond(RhoDirection(), RhoDirection(), ZDirection(), ZDirection(),
                            field);
      }
      return 0.0;
    }

    if (NearAxisCell()) {
      if (a == RhoDirection() && b == RhoDirection()) {
        return 2.0 * EvenCoefficientDerivative(field, a, b) -
               2.0 * (Value(field, RhoDirection(), RhoDirection()) -
                      Value(field, SuppressedDirection(), SuppressedDirection())) /
                   (rho_ * rho_);
      }
      if (a == SuppressedDirection() && b == SuppressedDirection()) {
        return 2.0 * EvenCoefficientDerivative(field, a, b) +
               2.0 * (Value(field, RhoDirection(), RhoDirection()) -
                      Value(field, SuppressedDirection(), SuppressedDirection())) /
                   (rho_ * rho_);
      }
      if (IsComponentPair(a, b, RhoDirection(), SuppressedDirection())) {
        return 2.0 * rho_ * rho_ *
                   QuadraticCoefficientDerivative(field, a, b) -
               2.0 * Value(field, a, b) / (rho_ * rho_);
      }
      if (IsComponentPair(a, b, RhoDirection(), ZDirection()) ||
          IsComponentPair(a, b, SuppressedDirection(), ZDirection())) {
        return 2.0 * rho_ * OddCoefficientDerivative(field, a, b);
      }
      return 2.0 * EvenCoefficientDerivative(field, a, b);
    }

    const Real radial_derivative = ActiveFirst(RhoDirection(), a, b, field);
    if (a == RhoDirection() && b == RhoDirection()) {
      return radial_derivative / rho_ -
             2.0 * (Value(field, RhoDirection(), RhoDirection()) -
                    Value(field, SuppressedDirection(), SuppressedDirection())) /
                 (rho_ * rho_);
    }
    if (a == SuppressedDirection() && b == SuppressedDirection()) {
      return radial_derivative / rho_ +
             2.0 * (Value(field, RhoDirection(), RhoDirection()) -
                    Value(field, SuppressedDirection(), SuppressedDirection())) /
                 (rho_ * rho_);
    }
    if (IsComponentPair(a, b, RhoDirection(), SuppressedDirection())) {
      return radial_derivative / rho_ -
             4.0 * Value(field, RhoDirection(), SuppressedDirection()) /
                 (rho_ * rho_);
    }
    if (IsComponentPair(a, b, RhoDirection(), ZDirection()) ||
        IsComponentPair(a, b, SuppressedDirection(), ZDirection())) {
      return radial_derivative / rho_ - Value(field, a, b) / (rho_ * rho_);
    }
    return radial_derivative / rho_;
  }

  template <typename TensorField>
  KOKKOS_INLINE_FUNCTION Real TensorMixedSuppressed(const int active_direction,
                                                    const int a, const int b,
                                                    TensorField &field) const {
    if (OnAxis()) {
      return TensorMixedSuppressedAxis(active_direction, a, b, field);
    }

    if (NearAxisCell()) {
      if (active_direction == RhoDirection()) {
        if (a == RhoDirection() && b == RhoDirection()) {
          return -2.0 *
                     Value(field, RhoDirection(), SuppressedDirection()) /
                     (rho_ * rho_) -
                 4.0 * rho_ * rho_ * QuadraticCoefficientDerivative(
                                               field, RhoDirection(),
                                               SuppressedDirection());
        }
        if (a == SuppressedDirection() && b == SuppressedDirection()) {
          return 2.0 *
                     Value(field, RhoDirection(), SuppressedDirection()) /
                     (rho_ * rho_) +
                 4.0 * rho_ * rho_ * QuadraticCoefficientDerivative(
                                               field, RhoDirection(),
                                               SuppressedDirection());
        }
        if (IsComponentPair(a, b, RhoDirection(), SuppressedDirection())) {
          return (Value(field, RhoDirection(), RhoDirection()) -
                  Value(field, SuppressedDirection(), SuppressedDirection())) /
                     (rho_ * rho_) +
                 2.0 * rho_ * rho_ *
                     QuadraticDifferenceCoefficientDerivative(
                         field, RhoDirection(), RhoDirection(),
                         SuppressedDirection(), SuppressedDirection());
        }
        if (IsComponentPair(a, b, RhoDirection(), ZDirection())) {
          return -2.0 * rho_ * OddCoefficientDerivative(
                                   field, SuppressedDirection(), ZDirection());
        }
        if (IsComponentPair(a, b, SuppressedDirection(), ZDirection())) {
          return 2.0 * rho_ *
                 OddCoefficientDerivative(field, RhoDirection(), ZDirection());
        }
        return 0.0;
      }
    }

    if (a == RhoDirection() && b == RhoDirection()) {
      Real result = -2.0 * ActiveFirst(active_direction, RhoDirection(),
                                       SuppressedDirection(), field) / rho_;
      if (active_direction == RhoDirection()) {
        result += 2.0 * Value(field, RhoDirection(), SuppressedDirection()) /
                  (rho_ * rho_);
      }
      return result;
    }
    if (a == SuppressedDirection() && b == SuppressedDirection()) {
      Real result = 2.0 * ActiveFirst(active_direction, RhoDirection(),
                                      SuppressedDirection(), field) / rho_;
      if (active_direction == RhoDirection()) {
        result -= 2.0 * Value(field, RhoDirection(), SuppressedDirection()) /
                  (rho_ * rho_);
      }
      return result;
    }
    if (IsComponentPair(a, b, RhoDirection(), SuppressedDirection())) {
      Real result = (ActiveFirst(active_direction, RhoDirection(), RhoDirection(), field) -
                     ActiveFirst(active_direction, SuppressedDirection(),
                                 SuppressedDirection(), field)) / rho_;
      if (active_direction == RhoDirection()) {
        result -= (Value(field, RhoDirection(), RhoDirection()) -
                   Value(field, SuppressedDirection(), SuppressedDirection())) /
                  (rho_ * rho_);
      }
      return result;
    }
    if (IsComponentPair(a, b, RhoDirection(), ZDirection())) {
      Real result = -ActiveFirst(active_direction, SuppressedDirection(), ZDirection(),
                                 field) / rho_;
      if (active_direction == RhoDirection()) {
        result += Value(field, SuppressedDirection(), ZDirection()) /
                  (rho_ * rho_);
      }
      return result;
    }
    if (IsComponentPair(a, b, SuppressedDirection(), ZDirection())) {
      Real result = ActiveFirst(active_direction, RhoDirection(), ZDirection(), field) /
                    rho_;
      if (active_direction == RhoDirection()) {
        result -= Value(field, RhoDirection(), ZDirection()) / (rho_ * rho_);
      }
      return result;
    }
    return 0.0;
  }

  template <typename TensorField>
  KOKKOS_INLINE_FUNCTION Real TensorMixedSuppressedAxis(const int active_direction,
                                                        const int a, const int b,
                                                        TensorField &field) const {
    if (active_direction == RhoDirection()) {
      if (a == RhoDirection() && b == RhoDirection()) {
        return -ActiveSecond(RhoDirection(), RhoDirection(), RhoDirection(),
                             SuppressedDirection(), field);
      }
      if (a == SuppressedDirection() && b == SuppressedDirection()) {
        return ActiveSecond(RhoDirection(), RhoDirection(), RhoDirection(),
                            SuppressedDirection(), field);
      }
      if (IsComponentPair(a, b, RhoDirection(), SuppressedDirection())) {
        return 0.5 * (ActiveSecond(RhoDirection(), RhoDirection(),
                                  RhoDirection(), RhoDirection(), field) -
                      ActiveSecond(RhoDirection(), RhoDirection(),
                                   SuppressedDirection(), SuppressedDirection(), field));
      }
      return 0.0;
    }

    if (IsComponentPair(a, b, RhoDirection(), ZDirection())) {
      return -ActiveSecond(RhoDirection(), ZDirection(), SuppressedDirection(),
                           ZDirection(), field);
    }
    if (IsComponentPair(a, b, SuppressedDirection(), ZDirection())) {
      return ActiveSecond(RhoDirection(), ZDirection(), RhoDirection(), ZDirection(),
                          field);
    }
    return 0.0;
  }

  Real inverse_spacing_[3];
  Real rho_;
  CartoonAxisLocation axis_location_;
  int m_;
  int k_;
  int j_;
  int i_;
};

//! Construct the compile-time derivative policy for one cell.
//!
//! The Cartesian instantiation does not evaluate coordinates; it delegates directly to
//! the frozen finite-difference primitives.  The Cartoon instantiation alone evaluates
//! nonnegative half-plane rho. Consequently host-selected Cartesian kernels retain their
//! path and neither policy captures a runtime symmetry mode in a device lambda.
template <typename Symmetry, int NGHOST, typename RegionSizeView>
KOKKOS_INLINE_FUNCTION DerivativeProvider<Symmetry, NGHOST>
MakeCellCenteredDerivativeProvider(const Real inverse_spacing[3],
                                   const RegionSizeView &size, const int nx1,
                                   const int is, const int m, const int k,
                                   const int j, const int i) {
  if constexpr (std::is_same_v<Symmetry, CartoonSO2>) {
    const Real rho = CellCenterX(i - is, nx1, size(m).x1min, size(m).x1max);
    return DerivativeProvider<CartoonSO2, NGHOST>(
        inverse_spacing, rho, CartoonAxisLocation::cell_centered, m, k, j, i);
  } else {
    static_assert(std::is_same_v<Symmetry, Cartesian3D>,
                  "Unknown Z4c derivative symmetry policy");
    return DerivativeProvider<Cartesian3D, NGHOST>(inverse_spacing, m, k, j, i);
  }
}

}  // namespace z4c

#endif  // Z4C_CARTOON_DERIVATIVES_HPP_
