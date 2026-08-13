//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE for details
//========================================================================================

#include <array>
#include <cstddef>
#include <stdexcept>

#include "pgen/z4c_irisk_coordinate_map.hpp"

namespace {

template <typename T, std::size_t N>
bool Equal(const std::array<T, N> &actual, const std::array<T, N> &expected) {
  for (std::size_t i = 0; i < N; ++i) {
    if (actual[i] != expected[i]) return false;
  }
  return true;
}

}  // namespace

int main() {
  using z4c_irisk::AdmMap;
  bool passed = true;

  const auto cartoon_positive =
      z4c_irisk::CartoonIrisInterpolationCoordinates(2.0, 3.0, 91.0);
  const auto cartoon_negative =
      z4c_irisk::CartoonIrisInterpolationCoordinates(-2.0, 3.0, -91.0);
  const auto cartoon_axis =
      z4c_irisk::CartoonIrisInterpolationCoordinates(0.0, -4.0, 17.0);
  passed &= Equal(cartoon_positive, std::array<double, 3>{2.0, 0.0, 3.0});
  passed &= Equal(cartoon_negative, std::array<double, 3>{-2.0, 0.0, 3.0});
  passed &= Equal(cartoon_axis, std::array<double, 3>{0.0, 0.0, -4.0});
  passed &= Equal(
      z4c_irisk::IrisInterpolationCoordinates<AdmMap::cartesian_xyz>(
          2.0, 3.0, 91.0),
      std::array<double, 3>{2.0, 3.0, 91.0});

  const std::array<double, 3> vector{101.0, 102.0, 103.0};
  const std::array<double, 6> tensor{11.0, 12.0, 13.0, 22.0, 23.0, 33.0};
  passed &= Equal(z4c_irisk::VectorFromPhysicalCartesian<AdmMap::cartesian_xyz>(
                      vector),
                  vector);
  passed &= Equal(
      z4c_irisk::SymmetricTensorFromPhysicalCartesian<AdmMap::cartesian_xyz>(
          tensor),
      tensor);
  passed &= Equal(z4c_irisk::VectorFromPhysicalCartesian<
                      AdmMap::half_rho_z_suppressed_y_v2>(vector),
                  std::array<double, 3>{101.0, 103.0, 102.0});
  passed &= Equal(z4c_irisk::SymmetricTensorFromPhysicalCartesian<
                      AdmMap::half_rho_z_suppressed_y_v2>(tensor),
                  std::array<double, 6>{11.0, 13.0, 12.0, 33.0, 23.0, 22.0});
  const auto mapped_vector = z4c_irisk::VectorFromPhysicalCartesian<
      AdmMap::half_rho_z_suppressed_y_v2>(vector);
  const auto mapped_tensor = z4c_irisk::SymmetricTensorFromPhysicalCartesian<
      AdmMap::half_rho_z_suppressed_y_v2>(tensor);
  passed &= Equal(z4c_irisk::VectorFromPhysicalCartesian<
                      AdmMap::half_rho_z_suppressed_y_v2>(mapped_vector),
                  vector);
  passed &= Equal(z4c_irisk::SymmetricTensorFromPhysicalCartesian<
                      AdmMap::half_rho_z_suppressed_y_v2>(mapped_tensor),
                  tensor);
  passed &= z4c_irisk::ScalarFromPhysicalCartesian<
                AdmMap::half_rho_z_suppressed_y_v2>(211.0) == 211.0;

  passed &= Equal(z4c_irisk::IrisTensorProductDimensions<AdmMap::cartesian_xyz>(
                      7, 5, 3),
                  std::array<std::size_t, 3>{7, 5, 3});
  passed &= Equal(z4c_irisk::IrisTensorProductDimensions<
                      AdmMap::half_rho_z_suppressed_y_v2>(7, 5, 3),
                  std::array<std::size_t, 3>{7, 1, 5});
  passed &= z4c_irisk::IrisPointIndex<AdmMap::cartesian_xyz>(2, 3, 1, 7, 5) ==
            2 + 7 * (3 + 5);
  passed &= z4c_irisk::IrisPointIndex<
                AdmMap::half_rho_z_suppressed_y_v2>(2, 3, 0, 7, 5) ==
            2 + 7 * 3;
  passed &= z4c_irisk::IrisPointIndex<
                AdmMap::half_rho_z_suppressed_y_v2>(2, 3, 9, 7, 5) ==
            2 + 7 * 3;

  const z4c::Z4cSymmetryConfig cartesian{
      z4c::Z4cSymmetryMode::cartesian3d,
      z4c::Z4cCoordinateMap::cartesian_xyz, 1, 2};
  const z4c::Z4cSymmetryConfig cartoon{
      z4c::Z4cSymmetryMode::cartoon_so2,
      z4c::Z4cCoordinateMap::half_rho_z_suppressed_y_v2, 2, 4};
  passed &= z4c_irisk::SelectAdmMap(cartesian) == AdmMap::cartesian_xyz;
  passed &= z4c_irisk::SelectAdmMap(cartoon) ==
            AdmMap::half_rho_z_suppressed_y_v2;

  for (const auto invalid : {
           z4c::Z4cSymmetryConfig{z4c::Z4cSymmetryMode::cartesian3d,
                                  z4c::Z4cCoordinateMap::
                                      signed_rho_z_suppressed_y_v1,
                                  1, 2},
           z4c::Z4cSymmetryConfig{z4c::Z4cSymmetryMode::cartoon_so2,
                                  z4c::Z4cCoordinateMap::cartesian_xyz, 1, 2},
           z4c::Z4cSymmetryConfig{
               static_cast<z4c::Z4cSymmetryMode>(99),
               z4c::Z4cCoordinateMap::cartesian_xyz, 1, 2}}) {
    bool rejected = false;
    try {
      static_cast<void>(z4c_irisk::SelectAdmMap(invalid));
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    passed &= rejected;
  }

  return passed ? 0 : 1;
}
