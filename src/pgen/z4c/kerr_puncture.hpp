//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE for details
//========================================================================================
//! \file kerr_puncture.hpp
//! \brief Analytic single-Kerr puncture data from arXiv:1001.4077.

#ifndef PGEN_Z4C_KERR_PUNCTURE_HPP_
#define PGEN_Z4C_KERR_PUNCTURE_HPP_

#include <cmath>
#include <limits>

#if defined(KOKKOS_INLINE_FUNCTION)
#define ATHENAK_KERR_PUNCTURE_INLINE KOKKOS_INLINE_FUNCTION
#else
#define ATHENAK_KERR_PUNCTURE_INLINE inline
#endif

class Mesh;
class MeshBlockPack;
class ParameterInput;
class ProblemGenerator;

namespace kerr_puncture {

enum class CoordinateMap {
  cartesian_xyz,
  half_rho_z_suppressed_y_v2,
};

enum class GaugeChoice {
  pre_collapsed,
  stationary,
};

template <typename Scalar>
struct Parameters {
  Scalar mass;
  Scalar chi;
  Scalar axial_center;
};

template <typename Scalar>
struct SymmetricTensor3 {
  Scalar xx;
  Scalar xy;
  Scalar xz;
  Scalar yy;
  Scalar yz;
  Scalar zz;

  ATHENAK_KERR_PUNCTURE_INLINE
  Scalar operator()(const int first, const int second) const {
    const int a = first < second ? first : second;
    const int b = first < second ? second : first;
    if (a == 0 && b == 0) return xx;
    if (a == 0 && b == 1) return xy;
    if (a == 0 && b == 2) return xz;
    if (a == 1 && b == 1) return yy;
    if (a == 1 && b == 2) return yz;
    return zz;
  }
};

template <typename Scalar>
struct PointData {
  bool valid;
  bool physical_adm_available;
  bool at_puncture;
  Scalar isotropic_radius;
  Scalar boyer_lindquist_radius;
  Scalar r_plus;
  Scalar r_minus;
  Scalar horizon_radius;
  Scalar lapse;
  Scalar shift[3];
  Scalar psi4;
  SymmetricTensor3<Scalar> spatial_metric;
  SymmetricTensor3<Scalar> extrinsic_curvature;
  Scalar conformal_chi;
  SymmetricTensor3<Scalar> conformal_metric;
  Scalar trace_extrinsic_curvature;
  SymmetricTensor3<Scalar> conformal_tracefree_curvature;
};

namespace detail {

template <typename Scalar>
ATHENAK_KERR_PUNCTURE_INLINE
bool IsFinite(const Scalar value) {
  return value == value &&
         value <= std::numeric_limits<Scalar>::max() &&
         value >= -std::numeric_limits<Scalar>::max();
}

template <typename Scalar>
ATHENAK_KERR_PUNCTURE_INLINE
Scalar Determinant(const SymmetricTensor3<Scalar> &value) {
  return value.xx * value.yy * value.zz +
         Scalar{2} * value.xy * value.xz * value.yz -
         value.xx * value.yz * value.yz -
         value.yy * value.xz * value.xz -
         value.zz * value.xy * value.xy;
}

template <typename Scalar>
ATHENAK_KERR_PUNCTURE_INLINE
SymmetricTensor3<Scalar> Scale(const SymmetricTensor3<Scalar> &value,
                               const Scalar factor) {
  return {factor * value.xx, factor * value.xy, factor * value.xz,
          factor * value.yy, factor * value.yz, factor * value.zz};
}

template <typename Scalar>
ATHENAK_KERR_PUNCTURE_INLINE
Scalar TraceWithInverse(const SymmetricTensor3<Scalar> &metric,
                        const SymmetricTensor3<Scalar> &tensor,
                        const Scalar inverse_determinant) {
  const Scalar uxx = (metric.yy * metric.zz - metric.yz * metric.yz) *
                     inverse_determinant;
  const Scalar uxy = (metric.xz * metric.yz - metric.xy * metric.zz) *
                     inverse_determinant;
  const Scalar uxz = (metric.xy * metric.yz - metric.xz * metric.yy) *
                     inverse_determinant;
  const Scalar uyy = (metric.xx * metric.zz - metric.xz * metric.xz) *
                     inverse_determinant;
  const Scalar uyz = (metric.xy * metric.xz - metric.xx * metric.yz) *
                     inverse_determinant;
  const Scalar uzz = (metric.xx * metric.yy - metric.xy * metric.xy) *
                     inverse_determinant;
  return uxx * tensor.xx + Scalar{2} * uxy * tensor.xy +
         Scalar{2} * uxz * tensor.xz + uyy * tensor.yy +
         Scalar{2} * uyz * tensor.yz + uzz * tensor.zz;
}

template <CoordinateMap Map, typename Scalar>
ATHENAK_KERR_PUNCTURE_INLINE
SymmetricTensor3<Scalar> TransformFromPhysicalCartesian(
    const SymmetricTensor3<Scalar> &physical) {
  if constexpr (Map == CoordinateMap::cartesian_xyz) return physical;
  // Cartoon stores code components (x1,x2,x3)=(X,Z,Y).
  return {physical.xx, physical.xz, physical.xy,
          physical.zz, physical.yz, physical.yy};
}

template <CoordinateMap Map, typename Scalar>
ATHENAK_KERR_PUNCTURE_INLINE
void TransformVectorFromPhysicalCartesian(const Scalar physical[3],
                                          Scalar code[3]) {
  if constexpr (Map == CoordinateMap::cartesian_xyz) {
    code[0] = physical[0];
    code[1] = physical[1];
    code[2] = physical[2];
    return;
  }
  code[0] = physical[0];
  code[1] = physical[2];
  code[2] = physical[1];
}

template <typename Scalar>
ATHENAK_KERR_PUNCTURE_INLINE
SymmetricTensor3<Scalar> TensorFromBasis(
    const Scalar radial_coefficient, const Scalar theta_coefficient,
    const Scalar phi_coefficient, const Scalar radial[3],
    const Scalar theta[3], const Scalar phi[3]) {
  Scalar result[3][3];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      result[i][j] = radial_coefficient * radial[i] * radial[j] +
                     theta_coefficient * theta[i] * theta[j] +
                     phi_coefficient * phi[i] * phi[j];
    }
  }
  return {result[0][0], result[0][1], result[0][2],
          result[1][1], result[1][2], result[2][2]};
}

template <typename Scalar>
ATHENAK_KERR_PUNCTURE_INLINE
SymmetricTensor3<Scalar> CrossTensorFromBasis(
    const Scalar radial_phi_coefficient,
    const Scalar theta_phi_coefficient, const Scalar radial[3],
    const Scalar theta[3], const Scalar phi[3]) {
  Scalar result[3][3];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      result[i][j] =
          radial_phi_coefficient *
              (radial[i] * phi[j] + phi[i] * radial[j]) +
          theta_phi_coefficient *
              (theta[i] * phi[j] + phi[i] * theta[j]);
    }
  }
  return {result[0][0], result[0][1], result[0][2],
          result[1][1], result[1][2], result[2][2]};
}

}  // namespace detail

//! Evaluate a stationary, axis-aligned single Kerr hole.
//!
//! Equations (11) and (13)-(15) of Liu, Etienne, and Shapiro,
//! arXiv:1001.4077, are evaluated in physical Cartesian coordinates and then
//! permuted into the requested AthenaK coordinate map.  At r=0 the physical
//! ADM metric belongs to the second asymptotic end and diverges.  The function
//! therefore marks physical_adm_available=false and returns only its finite
//! conformal/Z4c limit; it never clips r or substitutes an epsilon.
template <CoordinateMap Map, GaugeChoice Gauge, typename Scalar>
ATHENAK_KERR_PUNCTURE_INLINE
PointData<Scalar> Evaluate(const Scalar code_x1, const Scalar code_x2,
                           const Scalar code_x3,
                           const Parameters<Scalar> &parameters) {
  PointData<Scalar> result{};
  result.valid = detail::IsFinite(parameters.mass) &&
                 detail::IsFinite(parameters.chi) &&
                 detail::IsFinite(parameters.axial_center) &&
                 detail::IsFinite(code_x1) && detail::IsFinite(code_x2) &&
                 detail::IsFinite(code_x3) &&
                 parameters.mass > Scalar{0} &&
                 parameters.chi > Scalar{-1} &&
                 parameters.chi < Scalar{1};
  if (!result.valid) return result;

  Scalar x = code_x1;
  Scalar y = code_x2;
  Scalar z = code_x3;
  if constexpr (Map == CoordinateMap::half_rho_z_suppressed_y_v2) {
    y = code_x3;
    z = code_x2;
  }
  z -= parameters.axial_center;

  const Scalar mass = parameters.mass;
  const Scalar spin = parameters.chi * mass;
  const Scalar horizon_delta =
      mass * std::sqrt((Scalar{1} - parameters.chi) *
                       (Scalar{1} + parameters.chi));
  result.r_plus = mass + horizon_delta;
  result.r_minus = mass - horizon_delta;
  result.horizon_radius = result.r_plus / Scalar{4};

  const Scalar rho2 = x * x + y * y;
  const Scalar radius2 = rho2 + z * z;
  result.isotropic_radius = std::sqrt(radius2);
  result.at_puncture = result.isotropic_radius == Scalar{0};
  if (result.at_puncture) {
    // This is the analytic limit, not a flat-data substitution.  With
    // c=r_+/4, Eqs. (11),(13) give gamma_ij~c^4 r^-4 delta_ij and therefore
    // chi~r^4/c^4, tilde(gamma)_ij->delta_ij.  Equations (14),(15) give
    // K=0 and chi K_ij=O(r^3)->0.  The divergent physical carrier is marked
    // unavailable and is never consumed by the mesh initializer.
    result.physical_adm_available = false;
    result.boyer_lindquist_radius = Scalar{0};
    result.lapse = Gauge == GaugeChoice::stationary ? Scalar{1} : Scalar{0};
    result.shift[0] = result.shift[1] = result.shift[2] = Scalar{0};
    result.psi4 = Scalar{0};
    result.spatial_metric = {};
    result.extrinsic_curvature = {};
    result.conformal_chi = Scalar{0};
    result.conformal_metric = {Scalar{1}, Scalar{0}, Scalar{0},
                               Scalar{1}, Scalar{0}, Scalar{1}};
    result.trace_extrinsic_curvature = Scalar{0};
    result.conformal_tracefree_curvature = {};
    return result;
  }

  result.physical_adm_available = true;
  const Scalar radius = result.isotropic_radius;
  const Scalar cylindrical_radius = std::sqrt(rho2);
  const Scalar inverse_radius = Scalar{1} / radius;
  const Scalar sin_theta = cylindrical_radius * inverse_radius;
  const Scalar cos_theta = z * inverse_radius;
  const Scalar c = result.horizon_radius;
  result.boyer_lindquist_radius =
      radius + Scalar{2} * c + c * c / radius;
  const Scalar r_bl = result.boyer_lindquist_radius;
  const Scalar r_bl2 = r_bl * r_bl;
  const Scalar spin2 = spin * spin;
  const Scalar sigma = r_bl2 + spin2 * cos_theta * cos_theta;
  // Delta=(r_BL-r_+)(r_BL-r_-).  Eq. (11) makes the first factor
  // (r-c)^2/r, which is non-negative and exactly zero at the horizon without
  // a tolerance, clamp, or epsilon.
  const Scalar delta =
      (radius - c) * (radius - c) / radius *
      (r_bl - result.r_minus);
  const Scalar r_bl2_plus_spin2 = r_bl2 + spin2;
  const Scalar capital_a = r_bl2_plus_spin2 * r_bl2_plus_spin2 -
                           delta * spin2 * sin_theta * sin_theta;

  Scalar radial[3] = {x * inverse_radius, y * inverse_radius,
                      z * inverse_radius};
  Scalar theta[3];
  Scalar phi[3];
  if (cylindrical_radius > Scalar{0}) {
    const Scalar inverse_cylindrical_radius = Scalar{1} / cylindrical_radius;
    theta[0] = cos_theta * x * inverse_cylindrical_radius;
    theta[1] = cos_theta * y * inverse_cylindrical_radius;
    theta[2] = -sin_theta;
    phi[0] = -y * inverse_cylindrical_radius;
    phi[1] = x * inverse_cylindrical_radius;
    phi[2] = Scalar{0};
  } else {
    // On the rotation axis the two tangential metric eigenvalues agree and
    // both extrinsic-curvature coefficients vanish.  Any orthonormal pair is
    // therefore equivalent; this fixed pair makes the limit deterministic.
    theta[0] = Scalar{1};
    theta[1] = Scalar{0};
    theta[2] = Scalar{0};
    phi[0] = Scalar{0};
    phi[1] = radial[2] >= Scalar{0} ? Scalar{1} : Scalar{-1};
    phi[2] = Scalar{0};
  }

  const Scalar radial_metric =
      sigma * (radius + c) * (radius + c) /
      (radius * radius * radius * (r_bl - result.r_minus));
  const Scalar theta_metric = sigma / (radius * radius);
  const Scalar phi_metric = capital_a / (sigma * radius * radius);
  const auto physical_metric = detail::TensorFromBasis(
      radial_metric, theta_metric, phi_metric, radial, theta, phi);
  result.spatial_metric =
      detail::TransformFromPhysicalCartesian<Map>(physical_metric);

  const Scalar sqrt_a_sigma = std::sqrt(capital_a * sigma);
  const Scalar polynomial =
      Scalar{3} * r_bl2 * r_bl2 + Scalar{2} * spin2 * r_bl2 -
      spin2 * spin2 -
      spin2 * (r_bl2 - spin2) * sin_theta * sin_theta;
  const Scalar radial_phi =
      mass * spin * sin_theta * sin_theta * polynomial /
      (sigma * sqrt_a_sigma) * (Scalar{1} + c / radius) /
      std::sqrt(radius * (r_bl - result.r_minus));
  const Scalar theta_phi =
      -Scalar{2} * spin * spin2 * mass * r_bl * cos_theta *
      sin_theta * sin_theta * sin_theta /
      (sigma * sqrt_a_sigma) * (radius - c) *
      std::sqrt((r_bl - result.r_minus) / radius);
  const Scalar radial_phi_cartesian =
      cylindrical_radius > Scalar{0}
          ? radial_phi / (radius * sin_theta)
          : Scalar{0};
  const Scalar theta_phi_cartesian =
      cylindrical_radius > Scalar{0}
          ? theta_phi / (radius * radius * sin_theta)
          : Scalar{0};
  const auto physical_curvature = detail::CrossTensorFromBasis(
      radial_phi_cartesian, theta_phi_cartesian, radial, theta, phi);
  result.extrinsic_curvature =
      detail::TransformFromPhysicalCartesian<Map>(physical_curvature);

  const Scalar determinant = detail::Determinant(result.spatial_metric);
  const Scalar inverse_conformal_scale =
      Scalar{1} / std::cbrt(determinant);
  result.psi4 = Scalar{1} / inverse_conformal_scale;
  result.conformal_chi = inverse_conformal_scale;
  result.conformal_metric =
      detail::Scale(result.spatial_metric, inverse_conformal_scale);
  result.trace_extrinsic_curvature = detail::TraceWithInverse(
      result.spatial_metric, result.extrinsic_curvature,
      Scalar{1} / determinant);
  const Scalar one_third_trace =
      result.trace_extrinsic_curvature / Scalar{3};
  result.conformal_tracefree_curvature = {
      inverse_conformal_scale *
          (result.extrinsic_curvature.xx -
           one_third_trace * result.spatial_metric.xx),
      inverse_conformal_scale *
          (result.extrinsic_curvature.xy -
           one_third_trace * result.spatial_metric.xy),
      inverse_conformal_scale *
          (result.extrinsic_curvature.xz -
           one_third_trace * result.spatial_metric.xz),
      inverse_conformal_scale *
          (result.extrinsic_curvature.yy -
           one_third_trace * result.spatial_metric.yy),
      inverse_conformal_scale *
          (result.extrinsic_curvature.yz -
           one_third_trace * result.spatial_metric.yz),
      inverse_conformal_scale *
          (result.extrinsic_curvature.zz -
           one_third_trace * result.spatial_metric.zz)};

  Scalar physical_shift[3] = {Scalar{0}, Scalar{0}, Scalar{0}};
  if constexpr (Gauge == GaugeChoice::stationary) {
    result.lapse = std::sqrt(delta * sigma / capital_a);
    const Scalar beta_phi =
        -Scalar{2} * mass * spin * r_bl / capital_a;
    physical_shift[0] = -beta_phi * y;
    physical_shift[1] = beta_phi * x;
  } else {
    result.lapse = std::sqrt(result.conformal_chi);
  }
  detail::TransformVectorFromPhysicalCartesian<Map>(physical_shift,
                                                     result.shift);
  return result;
}

}  // namespace kerr_puncture

//! Fill a fresh AthenaK mesh with the analytic pgen.  On restart this function
//! returns without modifying evolved state.
void InitializeKerrPuncture(Mesh *mesh, ParameterInput *pin, bool restart);
void KerrPunctureRefinementCondition(MeshBlockPack *pack);
//! Enroll callbacks before InitializeKerrPuncture's restart-safe early return.
void ConfigureKerrPuncture(ProblemGenerator *generator, Mesh *mesh,
                           ParameterInput *pin, bool restart);

#undef ATHENAK_KERR_PUNCTURE_INLINE

#endif  // PGEN_Z4C_KERR_PUNCTURE_HPP_
