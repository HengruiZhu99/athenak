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
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include "outputs/pdf_validation.hpp"
#include "z4c/z4c_grid.hpp"

namespace z4c {

// Policy tags are defined by cartoon_derivatives.hpp. Dispatch only needs their identity.
struct Cartesian3D;
struct CartoonSO2;

//! Public Z4c evolution geometry. Cartesian three-dimensional evolution is the default.
enum class Z4cSymmetryMode { cartesian3d, cartoon_so2 };

//! Coordinate/component map carried by configuration and restart provenance.
enum class Z4cCoordinateMap {
  cartesian_xyz,
  signed_rho_z_suppressed_y_v1,
  half_rho_z_suppressed_y_v2
};

//! Immutable host configuration selected before any physics allocation.
struct Z4cSymmetryConfig {
  static constexpr int kCartesianSchema = 1;
  static constexpr int kHalfPlaneCartoonSchema = 2;
  static constexpr int kCurrentSchema = kHalfPlaneCartoonSchema;

  Z4cSymmetryMode mode = Z4cSymmetryMode::cartesian3d;
  Z4cCoordinateMap coordinate_map = Z4cCoordinateMap::cartesian_xyz;
  int schema = kCartesianSchema;
  int stencil_width = 2;
  Z4cGridCentering grid_centering = Z4cGridCentering::cell;
  int centering_schema = Z4cGridLayout::kCenteringSchema;
  // Effective native-VC midpoint interpolation order.  Zero is the immutable
  // cell-centered value and prevents the VC-only selector from leaking into
  // legacy cell configuration/restart provenance.
  int vertex_prolongation_order = 0;
};

constexpr int ExpectedZ4cSymmetrySchema(const Z4cSymmetryMode mode) {
  return mode == Z4cSymmetryMode::cartoon_so2
             ? Z4cSymmetryConfig::kHalfPlaneCartoonSchema
             : Z4cSymmetryConfig::kCartesianSchema;
}

//! Concrete target used to verify separately compiled policy/stencil instantiations.
struct Z4cKernelDispatchTarget {
  Z4cSymmetryMode mode;
  Z4cCoordinateMap coordinate_map;
  int stencil_width;
  Z4cGridCentering grid_centering = Z4cGridCentering::cell;
};

//! Output facts collected before output wrappers or physics modules are constructed.
struct Z4cOutputValidationRequest {
  std::string block_name;
  std::string file_type;
  pdf::ValidationInput pdf_input;
};

//! Host snapshot consumed by the allocation-free Cartoon validator.
struct Z4cValidationInput {
  std::string requested_symmetry = "cartesian3d";
  std::string requested_grid_centering = "cell";
  bool coordinate_map_specified = false;
  std::string coordinate_map;
  bool schema_specified = false;
  int schema = Z4cSymmetryConfig::kCartesianSchema;

  bool z4c_enabled = false;
  int nghost = 2;
  int requested_spatial_order = 2;
  bool vertex_prolongation_order_specified = false;
  std::string requested_vertex_prolongation_order = "auto";
  int mesh_nx1 = 1;
  int mesh_nx2 = 1;
  int mesh_nx3 = 1;
  int meshblock_nx1 = 1;
  int meshblock_nx3 = 1;
  int root_blocks_x1 = 1;
  double x1min = 0.0;
  double x1max = 1.0;
  std::size_t real_bytes = sizeof(double);
  std::string inner_x1_boundary = "undef";
  std::string outer_x1_boundary = "undef";
  std::string inner_x2_boundary = "undef";
  std::string outer_x2_boundary = "undef";
  std::string inner_x3_boundary = "undef";
  std::string outer_x3_boundary = "undef";

  std::vector<std::string> incompatible_physics;
  std::vector<std::string> incompatible_consumers;
  std::vector<Z4cOutputValidationRequest> outputs;

  bool restart_metadata_present = false;
  bool restart_carrier_present = false;
  bool restart_origin = false;
  std::string restart_symmetry;
  std::string restart_coordinate_map;
  int restart_schema = 0;

  std::string problem_generator;
  bool cartoon_derivative_check_only_present = false;
  bool cartoon_derivative_check_only_valid = false;
  bool cartoon_derivative_check_only = false;
  bool multilevel = false;
};

//! Result returned without throwing, aborting, or allocating device storage.
struct Z4cValidationResult {
  bool valid = false;
  Z4cSymmetryConfig config;
  std::string error;
};

const char *ToString(Z4cSymmetryMode mode);
const char *ToString(Z4cCoordinateMap coordinate_map);
const char *ToString(Z4cGridCentering centering);
int EffectiveZ4cSpatialOrder(int requested_spatial_order, int nghost);
Z4cValidationResult ValidateZ4cSymmetry(const Z4cValidationInput &input);

using Z4cCartoonKernelEntry = void (*)(void *context);
//! Compiled host-only Cartoon stencil dispatch without Cartesian instantiation.
void DispatchCartoonZ4cKernel(const Z4cSymmetryConfig &config, void *context,
                              Z4cCartoonKernelEntry order2,
                              Z4cCartoonKernelEntry order4,
                              Z4cCartoonKernelEntry order6);

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

namespace detail {

template <typename Centering, typename Callable>
struct CenteredZ4cDispatchAdapter {
  Callable &callable;

  template <typename Symmetry, int NGHOST>
  void Invoke(Z4cKernelDispatchTarget target) {
    target.grid_centering = std::is_same_v<Centering, VertexCenteredZ4c>
                                ? Z4cGridCentering::vertex
                                : Z4cGridCentering::cell;
    callable.template Invoke<Centering, Symmetry, NGHOST>(target);
  }
};

}  // namespace detail

//! Dispatch once on the host to a compile-time centering, symmetry, and stencil triple.
//!
//! The centering branch occurs before any device kernel launch. Production callables
//! must not capture the runtime centering enum inside their device lambdas.
template <typename Callable>
void DispatchCenteredZ4cKernel(const Z4cSymmetryConfig &config, Callable &callable) {
  if (config.grid_centering == Z4cGridCentering::cell) {
    detail::CenteredZ4cDispatchAdapter<CellCenteredZ4c, Callable> adapter{callable};
    DispatchZ4cKernel(config, adapter);
    return;
  }
  if (config.grid_centering == Z4cGridCentering::vertex) {
    detail::CenteredZ4cDispatchAdapter<VertexCenteredZ4c, Callable> adapter{callable};
    DispatchZ4cKernel(config, adapter);
    return;
  }
  std::cerr << "### FATAL ERROR: invalid Z4c host centering dispatch target"
            << std::endl;
  std::exit(EXIT_FAILURE);
}

}  // namespace z4c

#endif  // Z4C_Z4C_SYMMETRY_HPP_
