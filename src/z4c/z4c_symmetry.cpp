//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c_symmetry.cpp
//! \brief Separately compiled host targets for Z4c symmetry/stencil dispatch.

#include <type_traits>

#include "z4c/cartoon_derivatives.hpp"
#include "z4c/z4c_symmetry.hpp"

namespace z4c {

const char *ToString(const Z4cSymmetryMode mode) {
  switch (mode) {
    case Z4cSymmetryMode::cartesian3d:
      return "cartesian3d";
    case Z4cSymmetryMode::cartoon_so2:
      return "cartoon_so2";
  }
  return "invalid";
}

const char *ToString(const Z4cCoordinateMap coordinate_map) {
  switch (coordinate_map) {
    case Z4cCoordinateMap::cartesian_xyz:
      return "cartesian_xyz";
    case Z4cCoordinateMap::signed_rho_z_suppressed_y_v1:
      return "signed_rho_z_suppressed_y_v1";
  }
  return "invalid";
}

template <typename Symmetry, int NGHOST>
Z4cKernelDispatchTarget InstantiateZ4cKernelTarget() {
  static_assert(NGHOST >= 2 && NGHOST <= 4,
                "Z4c dispatch supports stencil widths 2, 3, and 4");
  constexpr bool is_cartesian = std::is_same_v<Symmetry, Cartesian3D>;
  static_assert(is_cartesian || std::is_same_v<Symmetry, CartoonSO2>,
                "Unknown Z4c symmetry policy tag");
  return {is_cartesian ? Z4cSymmetryMode::cartesian3d
                       : Z4cSymmetryMode::cartoon_so2,
          is_cartesian ? Z4cCoordinateMap::cartesian_xyz
                       : Z4cCoordinateMap::signed_rho_z_suppressed_y_v1,
          NGHOST};
}

template Z4cKernelDispatchTarget InstantiateZ4cKernelTarget<Cartesian3D, 2>();
template Z4cKernelDispatchTarget InstantiateZ4cKernelTarget<Cartesian3D, 3>();
template Z4cKernelDispatchTarget InstantiateZ4cKernelTarget<Cartesian3D, 4>();
template Z4cKernelDispatchTarget InstantiateZ4cKernelTarget<CartoonSO2, 2>();
template Z4cKernelDispatchTarget InstantiateZ4cKernelTarget<CartoonSO2, 3>();
template Z4cKernelDispatchTarget InstantiateZ4cKernelTarget<CartoonSO2, 4>();

}  // namespace z4c
