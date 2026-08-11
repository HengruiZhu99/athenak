//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE for details
//========================================================================================
//! \file z4c_irisk_coordinate_map.hpp
//! \brief Pure coordinate and component maps for IrisK ADM data.

#ifndef PGEN_Z4C_IRISK_COORDINATE_MAP_HPP_
#define PGEN_Z4C_IRISK_COORDINATE_MAP_HPP_

#include <array>
#include <cstddef>
#include <stdexcept>

#include "z4c/z4c_symmetry.hpp"

namespace z4c_irisk {

enum class AdmMap { cartesian_xyz, signed_rho_z_suppressed_y_v1 };

inline AdmMap SelectAdmMap(const z4c::Z4cSymmetryConfig &config) {
  switch (config.mode) {
    case z4c::Z4cSymmetryMode::cartesian3d:
      if (config.coordinate_map == z4c::Z4cCoordinateMap::cartesian_xyz) {
        return AdmMap::cartesian_xyz;
      }
      break;
    case z4c::Z4cSymmetryMode::cartoon_so2:
      if (config.coordinate_map ==
          z4c::Z4cCoordinateMap::signed_rho_z_suppressed_y_v1) {
        return AdmMap::signed_rho_z_suppressed_y_v1;
      }
      break;
    default:
      break;
  }
  throw std::invalid_argument("invalid IrisK ADM symmetry/coordinate-map pair");
}

template <typename Scalar>
constexpr std::array<Scalar, 3> CartoonIrisInterpolationCoordinates(
    const Scalar code_x1, const Scalar code_x2, const Scalar /*code_x3*/) {
  // Cartoon evolves the physical Y=0 meridional plane with code axes
  // (x1,x2,x3)=(X,Z,suppressed Y).
  return {code_x1, Scalar{0}, code_x2};
}

template <AdmMap Map, typename Scalar>
constexpr std::array<Scalar, 3> IrisInterpolationCoordinates(
    const Scalar code_x1, const Scalar code_x2, const Scalar code_x3) {
  if constexpr (Map == AdmMap::cartesian_xyz) {
    return {code_x1, code_x2, code_x3};
  }
  return CartoonIrisInterpolationCoordinates(code_x1, code_x2, code_x3);
}

template <AdmMap Map, typename Scalar>
constexpr std::array<Scalar, 3> VectorFromPhysicalCartesian(
    const std::array<Scalar, 3> &physical) {
  if constexpr (Map == AdmMap::cartesian_xyz) return physical;
  // Physical (X,Y,Z) -> Cartoon code (X,Z,Y).
  return {physical[0], physical[2], physical[1]};
}

template <AdmMap Map, typename Scalar>
constexpr Scalar ScalarFromPhysicalCartesian(const Scalar physical) {
  return physical;
}

template <AdmMap Map, typename Scalar>
constexpr std::array<Scalar, 6> SymmetricTensorFromPhysicalCartesian(
    const std::array<Scalar, 6> &physical) {
  if constexpr (Map == AdmMap::cartesian_xyz) return physical;
  // Packed physical [XX,XY,XZ,YY,YZ,ZZ] -> code [XX,XZ,XY,ZZ,YZ,YY].
  return {physical[0], physical[2], physical[1],
          physical[5], physical[4], physical[3]};
}

template <AdmMap Map>
constexpr std::array<std::size_t, 3> IrisTensorProductDimensions(
    const std::size_t nx1, const std::size_t nx2, const std::size_t nx3) {
  if constexpr (Map == AdmMap::cartesian_xyz) return {nx1, nx2, nx3};
  // The Iris Cartesian API receives X, Y, Z axes.  Cartoon has one physical
  // Y=0 sample and code x2 supplies the physical Z axis.
  return {nx1, 1, nx2};
}

template <AdmMap Map>
constexpr std::size_t IrisPointIndex(const std::size_t i,
                                     const std::size_t j,
                                     const std::size_t k,
                                     const std::size_t nx1,
                                     const std::size_t nx2) {
  if constexpr (Map == AdmMap::cartesian_xyz) {
    return i + nx1 * (j + nx2 * k);
  }
  // Every stored suppressed-direction layer receives the same Y=0 plane data.
  return i + nx1 * j;
}

}  // namespace z4c_irisk

#endif  // PGEN_Z4C_IRISK_COORDINATE_MAP_HPP_
