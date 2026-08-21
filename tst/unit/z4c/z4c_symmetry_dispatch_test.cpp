//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c_symmetry_dispatch_test.cpp
//! \brief Host-only tests for Z4c symmetry/stencil dispatch targets.

#include <cstdlib>
#include <iostream>
#include <type_traits>

#include "z4c/z4c_symmetry.hpp"

namespace {

struct DispatchRecorder {
  z4c::Z4cKernelDispatchTarget target{};
  bool invoked = false;
  bool compile_time_target_matches = false;

  template <typename Symmetry, int NGHOST>
  void Invoke(const z4c::Z4cKernelDispatchTarget selected) {
    constexpr bool is_cartesian = std::is_same_v<Symmetry, z4c::Cartesian3D>;
    invoked = true;
    target = selected;
    compile_time_target_matches =
        selected.mode == (is_cartesian ? z4c::Z4cSymmetryMode::cartesian3d
                                       : z4c::Z4cSymmetryMode::cartoon_so2) &&
        selected.stencil_width == NGHOST;
  }
};

struct CenteredDispatchRecorder {
  z4c::Z4cKernelDispatchTarget target{};
  bool invoked = false;
  bool compile_time_target_matches = false;

  template <typename Centering, typename Symmetry, int NGHOST>
  void Invoke(const z4c::Z4cKernelDispatchTarget selected) {
    constexpr bool is_vertex =
        std::is_same_v<Centering, z4c::VertexCenteredZ4c>;
    constexpr bool is_cartesian = std::is_same_v<Symmetry, z4c::Cartesian3D>;
    invoked = true;
    target = selected;
    compile_time_target_matches =
        selected.grid_centering ==
            (is_vertex ? z4c::Z4cGridCentering::vertex
                       : z4c::Z4cGridCentering::cell) &&
        selected.mode == (is_cartesian ? z4c::Z4cSymmetryMode::cartesian3d
                                       : z4c::Z4cSymmetryMode::cartoon_so2) &&
        selected.stencil_width == NGHOST;
  }
};

bool CheckTarget(const z4c::Z4cSymmetryMode mode, const int stencil_width) {
  z4c::Z4cSymmetryConfig config;
  config.mode = mode;
  config.coordinate_map =
      mode == z4c::Z4cSymmetryMode::cartesian3d
          ? z4c::Z4cCoordinateMap::cartesian_xyz
          : z4c::Z4cCoordinateMap::half_rho_z_suppressed_y_v2;
  config.schema = z4c::ExpectedZ4cSymmetrySchema(mode);
  config.stencil_width = stencil_width;

  DispatchRecorder recorder;
  z4c::DispatchZ4cKernel(config, recorder);
  return recorder.invoked && recorder.compile_time_target_matches &&
         recorder.target.mode == mode && recorder.target.stencil_width == stencil_width &&
         recorder.target.coordinate_map == config.coordinate_map;
}

bool CheckCenteredTarget(const z4c::Z4cGridCentering centering,
                         const z4c::Z4cSymmetryMode mode,
                         const int stencil_width) {
  z4c::Z4cSymmetryConfig config;
  config.mode = mode;
  config.coordinate_map =
      mode == z4c::Z4cSymmetryMode::cartesian3d
          ? z4c::Z4cCoordinateMap::cartesian_xyz
          : z4c::Z4cCoordinateMap::half_rho_z_suppressed_y_v2;
  config.schema = z4c::ExpectedZ4cSymmetrySchema(mode);
  config.stencil_width = stencil_width;
  config.grid_centering = centering;

  CenteredDispatchRecorder recorder;
  z4c::DispatchCenteredZ4cKernel(config, recorder);
  return recorder.invoked && recorder.compile_time_target_matches &&
         recorder.target.grid_centering == centering;
}

}  // namespace

int main() {
  z4c::Z4cSymmetryConfig default_config;
  bool passed = default_config.mode == z4c::Z4cSymmetryMode::cartesian3d &&
                default_config.coordinate_map == z4c::Z4cCoordinateMap::cartesian_xyz &&
                default_config.schema == z4c::Z4cSymmetryConfig::kCartesianSchema &&
                default_config.grid_centering == z4c::Z4cGridCentering::cell &&
                default_config.centering_schema ==
                    z4c::Z4cGridLayout::kCenteringSchema;
  for (const auto mode : {z4c::Z4cSymmetryMode::cartesian3d,
                          z4c::Z4cSymmetryMode::cartoon_so2}) {
    for (const int stencil_width : {2, 3, 4}) {
      passed = passed && CheckTarget(mode, stencil_width);
      for (const auto centering : {z4c::Z4cGridCentering::cell,
                                   z4c::Z4cGridCentering::vertex}) {
        passed = passed && CheckCenteredTarget(centering, mode, stencil_width);
      }
    }
  }

  if (!passed) return EXIT_FAILURE;
  std::cout << "Z4c host symmetry/stencil dispatch tests passed\n";
  return EXIT_SUCCESS;
}
