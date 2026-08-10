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

#include "athena.hpp"
#include "utils/finite_diff.hpp"

namespace z4c {

//! Coordinate directions used by the signed-meridional Cartoon mapping.
enum class CartoonDirection : int {
  rho = 0,        //!< x1: signed cylindrical radius on the active plane
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
  KOKKOS_INLINE_FUNCTION Real ComponentDissipation(const int component,
                                                   ComponentField &field) const {
    Real dissipation = 0.0;
    for (int d = 0; d < 3; ++d) {
      dissipation += Diss<NGHOST>(d, inverse_spacing_, field, m_, component,
                                  k_, j_, i_);
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

//! Analytic SO(2) Cartoon derivative policy on the signed x1-x2 meridional plane.
//!
//! The formulas in this specialization come from the independent, production-code-free
//! derivation in `signed_rho_so2_identity_note.md`. They follow from the Killing relation
//! for xi=-y*d_x+x*d_y with component order (x,z,y)=(x1,x2,x3). The note explicitly
//! requires an independent numerical-relativity review of all signs and axis limits before
//! production integration; this provider must remain outside the RHS until that review.
//! Every geometry-dependent formula and regularized return below is therefore
//! reviewer-pending, including parity, divergence, and suppressed advection.
//!
//! Near-axis cell centers use side-local, parity-mapped polynomial fits in s=rho^2. The
//! fit uses `NGHOST` samples on the target side, giving order 2*(NGHOST-1), and covers
//! the innermost `NGHOST` half-cell layers with maximum radial reach `NGHOST-1`. Thus an
//! internal axis at a MeshBlock boundary needs no wider communication than the existing
//! finite-difference stencil. Exact-axis diagnostic limits use a separate location tag;
//! no floating tolerance can silently turn a nonzero production cell into the axis.
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
    return NGHOST;
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
      return 2.0 * PhysicalRadialDerivative(EvenFit(field));
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
    if (NearAxisCell()) {
      if (component == RhoDirection()) {
        return -OddCoefficientFit(field, SuppressedDirection()).value;
      }
      return OddCoefficientFit(field, RhoDirection()).value;
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
      if (NearAxisCell()) {
        return sign * rho_ *
               QuadraticCoefficientFit(field, RhoDirection(), SuppressedDirection())
                   .value;
      }
      return sign * Value(field, RhoDirection(), SuppressedDirection()) / rho_;
    }
    if (IsComponentPair(a, b, RhoDirection(), SuppressedDirection())) {
      if (OnAxis()) {
        return 0.0;
      }
      if (NearAxisCell()) {
        return rho_ * QuadraticDifferenceFit(
                          field, RhoDirection(), RhoDirection(), SuppressedDirection(),
                          SuppressedDirection())
                          .value;
      }
      return (Value(field, RhoDirection(), RhoDirection()) -
              Value(field, SuppressedDirection(), SuppressedDirection())) / rho_;
    }
    if (IsComponentPair(a, b, RhoDirection(), ZDirection())) {
      if (OnAxis()) {
        return -ActiveFirst(RhoDirection(), SuppressedDirection(), ZDirection(), field);
      }
      if (NearAxisCell()) {
        return -OddCoefficientFit(field, SuppressedDirection(), ZDirection()).value;
      }
      return -Value(field, SuppressedDirection(), ZDirection()) / rho_;
    }
    if (IsComponentPair(a, b, SuppressedDirection(), ZDirection())) {
      if (OnAxis()) {
        return ActiveFirst(RhoDirection(), RhoDirection(), ZDirection(), field);
      }
      if (NearAxisCell()) {
        return OddCoefficientFit(field, RhoDirection(), ZDirection()).value;
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
    if (NearAxisCell()) {
      return ActiveFirst(RhoDirection(), RhoDirection(), field) +
             ActiveFirst(ZDirection(), ZDirection(), field) +
             OddCoefficientFit(field, RhoDirection()).value;
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
  KOKKOS_INLINE_FUNCTION Real ComponentDissipation(const int component,
                                                   ComponentField &field) const {
    Real dissipation = 0.0;
    for (int d = RhoDirection(); d <= ZDirection(); ++d) {
      dissipation += Diss<NGHOST>(d, inverse_spacing_, field, m_, component,
                                  k_, j_, i_);
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
           absolute_index <= 2 * NGHOST - 1;
  }

  KOKKOS_INLINE_FUNCTION static bool IsComponentPair(const int a, const int b,
                                                     const int c, const int d) {
    return (a == c && b == d) || (a == d && b == c);
  }

  struct RadialFit {
    Real value;
    Real derivative;
  };

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

  KOKKOS_INLINE_FUNCTION RadialFit FitRadialSamples(const Real samples[NGHOST]) const {
    const Real target = SQR(rho_ * inverse_spacing_[RhoDirection()]);
    RadialFit fit{0.0, 0.0};
    for (int point = 0; point < NGHOST; ++point) {
      const Real point_radius = static_cast<Real>(point) + 0.5;
      const Real point_s = point_radius * point_radius;
      Real basis = 1.0;
      for (int other = 0; other < NGHOST; ++other) {
        if (other == point) continue;
        const Real other_radius = static_cast<Real>(other) + 0.5;
        const Real other_s = other_radius * other_radius;
        basis *= (target - other_s) / (point_s - other_s);
      }
      Real basis_derivative = 0.0;
      for (int differentiated = 0; differentiated < NGHOST; ++differentiated) {
        if (differentiated == point) continue;
        const Real differentiated_radius = static_cast<Real>(differentiated) + 0.5;
        const Real differentiated_s = differentiated_radius * differentiated_radius;
        Real term = 1.0 / (point_s - differentiated_s);
        for (int other = 0; other < NGHOST; ++other) {
          if (other == point || other == differentiated) continue;
          const Real other_radius = static_cast<Real>(other) + 0.5;
          const Real other_s = other_radius * other_radius;
          term *= (target - other_s) / (point_s - other_s);
        }
        basis_derivative += term;
      }
      fit.value += basis * samples[point];
      fit.derivative += basis_derivative * samples[point];
    }
    return fit;
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

  template <typename VectorField>
  KOKKOS_INLINE_FUNCTION Real MappedVectorDerivative(
      const VectorField &field, const int direction, const int component,
      const int layer) const {
    const int parity = TargetSideSign() < 0 ? VectorParity(component) : 1;
    return parity * Dx<NGHOST>(direction, inverse_spacing_, field, m_, component, k_,
                               j_, SideLayerIndex(layer));
  }

  template <typename TensorField>
  KOKKOS_INLINE_FUNCTION Real MappedTensorDerivative(
      const TensorField &field, const int direction, const int a, const int b,
      const int layer) const {
    const int parity = TargetSideSign() < 0 ? TensorParity(a, b) : 1;
    return parity * Dx<NGHOST>(direction, inverse_spacing_, field, m_, a, b, k_, j_,
                               SideLayerIndex(layer));
  }

  template <typename ScalarField>
  KOKKOS_INLINE_FUNCTION RadialFit EvenFit(const ScalarField &field) const {
    Real samples[NGHOST];
    for (int layer = 0; layer < NGHOST; ++layer) {
      samples[layer] = MappedScalar(field, layer);
    }
    return FitRadialSamples(samples);
  }

  template <typename VectorField>
  KOKKOS_INLINE_FUNCTION RadialFit EvenFit(const VectorField &field,
                                          const int component) const {
    Real samples[NGHOST];
    for (int layer = 0; layer < NGHOST; ++layer) {
      samples[layer] = MappedVector(field, component, layer);
    }
    return FitRadialSamples(samples);
  }

  template <typename TensorField>
  KOKKOS_INLINE_FUNCTION RadialFit EvenFit(const TensorField &field, const int a,
                                          const int b) const {
    Real samples[NGHOST];
    for (int layer = 0; layer < NGHOST; ++layer) {
      samples[layer] = MappedTensor(field, a, b, layer);
    }
    return FitRadialSamples(samples);
  }

  template <typename VectorField>
  KOKKOS_INLINE_FUNCTION RadialFit OddCoefficientFit(const VectorField &field,
                                                     const int component) const {
    Real samples[NGHOST];
    const Real spacing = 1.0 / inverse_spacing_[RhoDirection()];
    for (int layer = 0; layer < NGHOST; ++layer) {
      const Real radius = (static_cast<Real>(layer) + 0.5) * spacing;
      samples[layer] = MappedVector(field, component, layer) / radius;
    }
    return FitRadialSamples(samples);
  }

  template <typename TensorField>
  KOKKOS_INLINE_FUNCTION RadialFit OddCoefficientFit(const TensorField &field,
                                                     const int a, const int b) const {
    Real samples[NGHOST];
    const Real spacing = 1.0 / inverse_spacing_[RhoDirection()];
    for (int layer = 0; layer < NGHOST; ++layer) {
      const Real radius = (static_cast<Real>(layer) + 0.5) * spacing;
      samples[layer] = MappedTensor(field, a, b, layer) / radius;
    }
    return FitRadialSamples(samples);
  }

  template <typename VectorField>
  KOKKOS_INLINE_FUNCTION RadialFit OddDerivativeCoefficientFit(
      const VectorField &field, const int direction, const int component) const {
    Real samples[NGHOST];
    const Real spacing = 1.0 / inverse_spacing_[RhoDirection()];
    for (int layer = 0; layer < NGHOST; ++layer) {
      const Real radius = (static_cast<Real>(layer) + 0.5) * spacing;
      samples[layer] =
          MappedVectorDerivative(field, direction, component, layer) / radius;
    }
    return FitRadialSamples(samples);
  }

  template <typename TensorField>
  KOKKOS_INLINE_FUNCTION RadialFit OddDerivativeCoefficientFit(
      const TensorField &field, const int direction, const int a, const int b) const {
    Real samples[NGHOST];
    const Real spacing = 1.0 / inverse_spacing_[RhoDirection()];
    for (int layer = 0; layer < NGHOST; ++layer) {
      const Real radius = (static_cast<Real>(layer) + 0.5) * spacing;
      samples[layer] =
          MappedTensorDerivative(field, direction, a, b, layer) / radius;
    }
    return FitRadialSamples(samples);
  }

  template <typename TensorField>
  KOKKOS_INLINE_FUNCTION RadialFit QuadraticCoefficientFit(
      const TensorField &field, const int a, const int b) const {
    Real samples[NGHOST];
    const Real spacing = 1.0 / inverse_spacing_[RhoDirection()];
    for (int layer = 0; layer < NGHOST; ++layer) {
      const Real radius = (static_cast<Real>(layer) + 0.5) * spacing;
      samples[layer] = MappedTensor(field, a, b, layer) / (radius * radius);
    }
    return FitRadialSamples(samples);
  }

  template <typename TensorField>
  KOKKOS_INLINE_FUNCTION RadialFit QuadraticDifferenceFit(
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
    return FitRadialSamples(samples);
  }

  template <typename TensorField>
  KOKKOS_INLINE_FUNCTION RadialFit QuadraticDerivativeFit(
      const TensorField &field, const int direction, const int a, const int b) const {
    Real samples[NGHOST];
    const Real spacing = 1.0 / inverse_spacing_[RhoDirection()];
    for (int layer = 0; layer < NGHOST; ++layer) {
      const Real radius = (static_cast<Real>(layer) + 0.5) * spacing;
      samples[layer] = MappedTensorDerivative(field, direction, a, b, layer) /
                       (radius * radius);
    }
    return FitRadialSamples(samples);
  }

  template <typename TensorField>
  KOKKOS_INLINE_FUNCTION RadialFit QuadraticDifferenceDerivativeFit(
      const TensorField &field, const int direction, const int a, const int b,
      const int c, const int d) const {
    Real samples[NGHOST];
    const Real spacing = 1.0 / inverse_spacing_[RhoDirection()];
    for (int layer = 0; layer < NGHOST; ++layer) {
      const Real radius = (static_cast<Real>(layer) + 0.5) * spacing;
      samples[layer] =
          (MappedTensorDerivative(field, direction, a, b, layer) -
           MappedTensorDerivative(field, direction, c, d, layer)) /
          (radius * radius);
    }
    return FitRadialSamples(samples);
  }

  KOKKOS_INLINE_FUNCTION Real PhysicalRadialDerivative(const RadialFit &fit) const {
    return fit.derivative * inverse_spacing_[RhoDirection()] *
           inverse_spacing_[RhoDirection()];
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
        return 2.0 * PhysicalRadialDerivative(EvenFit(field, component));
      }
      const RadialFit coefficient = OddCoefficientFit(field, component);
      return 2.0 * rho_ * PhysicalRadialDerivative(coefficient);
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
        const RadialFit coefficient = OddCoefficientFit(field, rotated_component);
        return sign * 2.0 * rho_ * PhysicalRadialDerivative(coefficient);
      }
      return sign *
             OddDerivativeCoefficientFit(field, active_direction, rotated_component).value;
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
        const RadialFit component = EvenFit(field, a, b);
        const RadialFit difference = QuadraticDifferenceFit(
            field, RhoDirection(), RhoDirection(), SuppressedDirection(),
            SuppressedDirection());
        return 2.0 * PhysicalRadialDerivative(component) - 2.0 * difference.value;
      }
      if (a == SuppressedDirection() && b == SuppressedDirection()) {
        const RadialFit component = EvenFit(field, a, b);
        const RadialFit difference = QuadraticDifferenceFit(
            field, RhoDirection(), RhoDirection(), SuppressedDirection(),
            SuppressedDirection());
        return 2.0 * PhysicalRadialDerivative(component) + 2.0 * difference.value;
      }
      if (IsComponentPair(a, b, RhoDirection(), SuppressedDirection())) {
        const RadialFit coefficient = QuadraticCoefficientFit(field, a, b);
        return 2.0 * rho_ * rho_ * PhysicalRadialDerivative(coefficient) -
               2.0 * coefficient.value;
      }
      if (IsComponentPair(a, b, RhoDirection(), ZDirection()) ||
          IsComponentPair(a, b, SuppressedDirection(), ZDirection())) {
        const RadialFit coefficient = OddCoefficientFit(field, a, b);
        return 2.0 * rho_ * PhysicalRadialDerivative(coefficient);
      }
      return 2.0 * PhysicalRadialDerivative(EvenFit(field, a, b));
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
          const RadialFit coefficient = QuadraticCoefficientFit(
              field, RhoDirection(), SuppressedDirection());
          return -2.0 * coefficient.value -
                 4.0 * rho_ * rho_ * PhysicalRadialDerivative(coefficient);
        }
        if (a == SuppressedDirection() && b == SuppressedDirection()) {
          const RadialFit coefficient = QuadraticCoefficientFit(
              field, RhoDirection(), SuppressedDirection());
          return 2.0 * coefficient.value +
                 4.0 * rho_ * rho_ * PhysicalRadialDerivative(coefficient);
        }
        if (IsComponentPair(a, b, RhoDirection(), SuppressedDirection())) {
          const RadialFit difference = QuadraticDifferenceFit(
              field, RhoDirection(), RhoDirection(), SuppressedDirection(),
              SuppressedDirection());
          return difference.value +
                 2.0 * rho_ * rho_ * PhysicalRadialDerivative(difference);
        }
        if (IsComponentPair(a, b, RhoDirection(), ZDirection())) {
          const RadialFit coefficient =
              OddCoefficientFit(field, SuppressedDirection(), ZDirection());
          return -2.0 * rho_ * PhysicalRadialDerivative(coefficient);
        }
        if (IsComponentPair(a, b, SuppressedDirection(), ZDirection())) {
          const RadialFit coefficient =
              OddCoefficientFit(field, RhoDirection(), ZDirection());
          return 2.0 * rho_ * PhysicalRadialDerivative(coefficient);
        }
        return 0.0;
      }

      if (a == RhoDirection() && b == RhoDirection()) {
        return -2.0 * rho_ *
               QuadraticDerivativeFit(field, active_direction, RhoDirection(),
                                      SuppressedDirection())
                   .value;
      }
      if (a == SuppressedDirection() && b == SuppressedDirection()) {
        return 2.0 * rho_ *
               QuadraticDerivativeFit(field, active_direction, RhoDirection(),
                                      SuppressedDirection())
                   .value;
      }
      if (IsComponentPair(a, b, RhoDirection(), SuppressedDirection())) {
        return rho_ * QuadraticDifferenceDerivativeFit(
                          field, active_direction, RhoDirection(), RhoDirection(),
                          SuppressedDirection(), SuppressedDirection())
                          .value;
      }
      if (IsComponentPair(a, b, RhoDirection(), ZDirection())) {
        return -OddDerivativeCoefficientFit(field, active_direction,
                                            SuppressedDirection(), ZDirection())
                    .value;
      }
      if (IsComponentPair(a, b, SuppressedDirection(), ZDirection())) {
        return OddDerivativeCoefficientFit(field, active_direction, RhoDirection(),
                                           ZDirection())
            .value;
      }
      return 0.0;
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

}  // namespace z4c

#endif  // Z4C_CARTOON_DERIVATIVES_HPP_
