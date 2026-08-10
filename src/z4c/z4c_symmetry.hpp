//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c_symmetry.hpp
//! \brief Host-only Z4c symmetry and finite-difference dispatch declarations.

#ifndef Z4C_Z4C_SYMMETRY_HPP_
#define Z4C_Z4C_SYMMETRY_HPP_

#include <cstdlib>
#include <iostream>
#include <string>

namespace z4c {

// Policy tags are defined by cartoon_derivatives.hpp. Dispatch only needs their identity.
struct Cartesian3D;
struct CartoonSO2;

//! Public Z4c evolution geometry. Cartesian three-dimensional evolution is the default.
enum class Z4cSymmetryMode { cartesian3d, cartoon_so2 };

//! Coordinate/component map carried by configuration and restart provenance.
enum class Z4cCoordinateMap {
  cartesian_xyz,
  signed_rho_z_suppressed_y_v1
};

//! Immutable host configuration selected before any physics allocation.
struct Z4cSymmetryConfig {
  static constexpr int kCurrentSchema = 1;

  Z4cSymmetryMode mode = Z4cSymmetryMode::cartesian3d;
  Z4cCoordinateMap coordinate_map = Z4cCoordinateMap::cartesian_xyz;
  int schema = kCurrentSchema;
  int stencil_width = 2;
};

//! Concrete target used to verify separately compiled policy/stencil instantiations.
struct Z4cKernelDispatchTarget {
  Z4cSymmetryMode mode;
  Z4cCoordinateMap coordinate_map;
  int stencil_width;
};

const char *ToString(Z4cSymmetryMode mode);
const char *ToString(Z4cCoordinateMap coordinate_map);

//! Return the target associated with a separately compiled policy/stencil pair.
template <typename Symmetry, int NGHOST>
Z4cKernelDispatchTarget InstantiateZ4cKernelTarget();

extern template Z4cKernelDispatchTarget InstantiateZ4cKernelTarget<Cartesian3D, 2>();
extern template Z4cKernelDispatchTarget InstantiateZ4cKernelTarget<Cartesian3D, 3>();
extern template Z4cKernelDispatchTarget InstantiateZ4cKernelTarget<Cartesian3D, 4>();
extern template Z4cKernelDispatchTarget InstantiateZ4cKernelTarget<CartoonSO2, 2>();
extern template Z4cKernelDispatchTarget InstantiateZ4cKernelTarget<CartoonSO2, 3>();
extern template Z4cKernelDispatchTarget InstantiateZ4cKernelTarget<CartoonSO2, 4>();

//! Dispatch once on the host to a compile-time symmetry and stencil pair.
//!
//! `Callable` supplies `template <typename Symmetry, int NGHOST> void Invoke(target)`.
//! Production callers invoke the selected separately compiled kernel from that method;
//! they must never capture `Z4cSymmetryConfig` in a device lambda.
template <typename Callable>
void DispatchZ4cKernel(const Z4cSymmetryConfig &config, Callable &callable) {
  if (config.mode == Z4cSymmetryMode::cartesian3d) {
    switch (config.stencil_width) {
      case 2:
        callable.template Invoke<Cartesian3D, 2>(
            InstantiateZ4cKernelTarget<Cartesian3D, 2>());
        return;
      case 3:
        callable.template Invoke<Cartesian3D, 3>(
            InstantiateZ4cKernelTarget<Cartesian3D, 3>());
        return;
      case 4:
        callable.template Invoke<Cartesian3D, 4>(
            InstantiateZ4cKernelTarget<Cartesian3D, 4>());
        return;
      default:
        break;
    }
  } else if (config.mode == Z4cSymmetryMode::cartoon_so2) {
    switch (config.stencil_width) {
      case 2:
        callable.template Invoke<CartoonSO2, 2>(
            InstantiateZ4cKernelTarget<CartoonSO2, 2>());
        return;
      case 3:
        callable.template Invoke<CartoonSO2, 3>(
            InstantiateZ4cKernelTarget<CartoonSO2, 3>());
        return;
      case 4:
        callable.template Invoke<CartoonSO2, 4>(
            InstantiateZ4cKernelTarget<CartoonSO2, 4>());
        return;
      default:
        break;
    }
  }

  std::cerr << "### FATAL ERROR: invalid Z4c host dispatch target: symmetry="
            << ToString(config.mode) << ", stencil_width=" << config.stencil_width
            << std::endl;
  std::exit(EXIT_FAILURE);
}

}  // namespace z4c

#endif  // Z4C_Z4C_SYMMETRY_HPP_
