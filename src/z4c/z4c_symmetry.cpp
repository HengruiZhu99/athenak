//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c_symmetry.cpp
//! \brief Separately compiled host targets for Z4c symmetry/stencil dispatch.

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
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
    case Z4cCoordinateMap::half_rho_z_suppressed_y_v2:
      return "half_rho_z_suppressed_y_v2";
  }
  return "invalid";
}

int EffectiveZ4cSpatialOrder(const int requested_spatial_order, const int nghost) {
  return requested_spatial_order > 0 ? requested_spatial_order : 2 * (nghost - 1);
}

namespace {

Z4cValidationResult Invalid(const Z4cSymmetryConfig &config,
                            const std::string &message) {
  return {false, config, message};
}

bool IsSupportedOutput(const std::string &file_type) {
  return file_type == "tab" || file_type == "hst" || file_type == "log" ||
         file_type == "vtk" || file_type == "pdf" || file_type == "bin" ||
         file_type == "rst";
}

bool IsRejectedOutput(const std::string &file_type) {
  return file_type == "cart" || file_type == "sph" || file_type == "cbin" ||
         file_type == "pvtk" || file_type == "trk";
}

}  // namespace

Z4cValidationResult ValidateZ4cSymmetry(const Z4cValidationInput &input) {
  Z4cSymmetryConfig config;
  if (input.requested_symmetry == "cartesian3d") {
    config.mode = Z4cSymmetryMode::cartesian3d;
    config.coordinate_map = Z4cCoordinateMap::cartesian_xyz;
    config.schema = Z4cSymmetryConfig::kCartesianSchema;
  } else if (input.requested_symmetry == "cartoon_so2") {
    config.mode = Z4cSymmetryMode::cartoon_so2;
    config.coordinate_map = Z4cCoordinateMap::half_rho_z_suppressed_y_v2;
    config.schema = Z4cSymmetryConfig::kHalfPlaneCartoonSchema;
  } else {
    return Invalid(config, "<z4c>/symmetry must be cartesian3d or cartoon_so2, not '" +
                               input.requested_symmetry + "'");
  }

  if (input.z4c_enabled) {
    const int spatial_order =
        EffectiveZ4cSpatialOrder(input.requested_spatial_order, input.nghost);
    if (spatial_order != 2 && spatial_order != 4 && spatial_order != 6) {
      return Invalid(config, "<z4c>/spatial_order must be 2, 4, or 6");
    }
    config.stencil_width = spatial_order / 2 + 1;
    if (input.nghost < config.stencil_width) {
      std::ostringstream message;
      message << "effective <z4c>/spatial_order=" << spatial_order
              << " requires at least " << config.stencil_width
              << " ghost cells, but <mesh>/nghost=" << input.nghost;
      return Invalid(config, message.str());
    }
  }

  for (const auto &output : input.outputs) {
    if (output.file_type != "pdf") continue;
    const auto pdf_plan = pdf::Validate(
        output.pdf_input, input.real_bytes,
        config.mode == Z4cSymmetryMode::cartoon_so2);
    if (!pdf_plan.valid) return Invalid(config, pdf_plan.error);
  }

  // This exact pgen is a Cartoon-only, fresh-start, check-only exception.  Enforce
  // those immutable facts before the Cartesian early return as well, so a genuine
  // carrier-free Cartesian restart cannot inject the pgen through -i/CLI and reach
  // Mesh/Z4c allocation before its own defensive restart check.
  const bool derivative_pgen = input.problem_generator == "z4c_cartoon_derivatives";
  if (derivative_pgen && input.cartoon_derivative_check_only_present &&
      !input.cartoon_derivative_check_only_valid) {
    return Invalid(config,
                   "<problem>/check_only must be an explicit boolean for "
                   "z4c_cartoon_derivatives");
  }
  if (derivative_pgen && (input.restart_origin || input.restart_metadata_present ||
                          input.restart_carrier_present)) {
    return Invalid(config,
                   "z4c_cartoon_derivatives check_only rejects restart");
  }
  if (derivative_pgen && config.mode != Z4cSymmetryMode::cartoon_so2) {
    return Invalid(config,
                   "z4c_cartoon_derivatives check_only requires cartoon_so2");
  }

  if (config.mode == Z4cSymmetryMode::cartesian3d) {
    if (input.coordinate_map_specified && input.coordinate_map != "cartesian_xyz") {
      return Invalid(config, "cartesian3d requires coordinate_map=cartesian_xyz");
    }
    if (input.schema_specified &&
        input.schema != Z4cSymmetryConfig::kCartesianSchema) {
      return Invalid(config, "unsupported <z4c>/symmetry_schema for cartesian3d");
    }
    for (const auto *boundary : {&input.inner_x1_boundary, &input.outer_x1_boundary,
                                 &input.inner_x2_boundary, &input.outer_x2_boundary,
                                 &input.inner_x3_boundary, &input.outer_x3_boundary}) {
      if (*boundary == "axis") {
        return Invalid(config,
                       "boundary type axis is reserved for cartoon_so2 inner_x1");
      }
    }
    if (input.restart_metadata_present &&
        (input.restart_symmetry != "cartesian3d" ||
         input.restart_coordinate_map != "cartesian_xyz" ||
         input.restart_schema != Z4cSymmetryConfig::kCartesianSchema)) {
      return Invalid(config, "restart symmetry metadata conflicts with cartesian3d");
    }
    return {true, config, ""};
  }

  if (!input.z4c_enabled) {
    return Invalid(config, "cartoon_so2 requires the <z4c> physics block");
  }
  if (input.coordinate_map_specified &&
      input.coordinate_map != "half_rho_z_suppressed_y_v2") {
    return Invalid(config,
                   "cartoon_so2 requires coordinate_map="
                   "half_rho_z_suppressed_y_v2");
  }
  if (input.schema_specified &&
      input.schema != Z4cSymmetryConfig::kHalfPlaneCartoonSchema) {
    return Invalid(config, "unsupported <z4c>/symmetry_schema for cartoon_so2");
  }

  if (input.mesh_nx3 != 1 || input.meshblock_nx3 != 1) {
    return Invalid(config, "cartoon_so2 requires mesh/nx3=meshblock/nx3=1");
  }
  if (input.mesh_nx2 <= 1) {
    return Invalid(config, "cartoon_so2 requires an active x1-x2 meridional plane");
  }
  if (input.mesh_nx1 <= 0) {
    return Invalid(config, "cartoon_so2 requires positive <mesh>/nx1");
  }
  if (input.meshblock_nx1 < input.nghost) {
    return Invalid(config,
                   "cartoon_so2 requires meshblock/nx1 >= mesh/nghost for local "
                   "axis parity extension");
  }
  if (input.multilevel && input.meshblock_nx1 < 2 * input.nghost) {
    return Invalid(config,
                   "multilevel cartoon_so2 requires meshblock/nx1 >= "
                   "2*mesh/nghost so coarse axis parity ghosts mirror local "
                   "coarse active cells");
  }
  if (input.root_blocks_x1 <= 0) {
    return Invalid(config, "cartoon_so2 requires positive root x1 MeshBlock count");
  }
  if (!std::isfinite(input.x1min) || !std::isfinite(input.x1max) ||
      input.x1min != 0.0 || !(input.x1max > 0.0)) {
    return Invalid(config,
                   "cartoon_so2 requires the exact finite half-plane axis face "
                   "<mesh>/x1min=0 and x1max>0");
  }
  const double dx1 = (input.x1max - input.x1min) /
                     static_cast<double>(input.mesh_nx1);
  const double first_center = input.x1min + 0.5 * dx1;
  if (!std::isfinite(dx1) || !(dx1 > 0.0) || !std::isfinite(first_center) ||
      !(first_center > 0.0)) {
    return Invalid(config,
                   "cartoon_so2 requires a finite positive first active center at "
                   "rho=dx1/2, never rho=0");
  }
  if (input.inner_x1_boundary != "axis") {
    return Invalid(config,
                   "cartoon_so2 half-plane requires <mesh>/ix1_bc=axis");
  }
  for (const auto *boundary : {&input.outer_x1_boundary, &input.inner_x2_boundary,
                               &input.outer_x2_boundary, &input.inner_x3_boundary,
                               &input.outer_x3_boundary}) {
    if (*boundary == "axis") {
      return Invalid(config,
                     "cartoon_so2 permits boundary type axis only at inner_x1");
    }
  }

  if (!input.incompatible_physics.empty()) {
    return Invalid(config, "cartoon_so2 vacuum Z4c forbids <" +
                               input.incompatible_physics.front() + "> physics");
  }
  if (!input.incompatible_consumers.empty()) {
    return Invalid(config, "cartoon_so2 does not support " +
                               input.incompatible_consumers.front());
  }

  for (const auto &output : input.outputs) {
    if (IsRejectedOutput(output.file_type)) {
      return Invalid(config, "cartoon_so2 rejects file_type=" + output.file_type +
                                 " in <" + output.block_name +
                                 "> before output construction");
    }
    if (!IsSupportedOutput(output.file_type)) {
      return Invalid(config, "unknown file_type='" + output.file_type + "' in <" +
                                 output.block_name +
                                 ">; supported Cartoon types are "
                                 "tab,hst,log,vtk,pdf,bin,rst");
    }
    if (output.file_type == "pdf") {
      // The shared allocation-free PDF contract was already checked above.
    }
  }

  if (input.restart_metadata_present &&
      (input.restart_symmetry != "cartoon_so2" ||
       input.restart_coordinate_map != "half_rho_z_suppressed_y_v2" ||
       input.restart_schema != Z4cSymmetryConfig::kHalfPlaneCartoonSchema)) {
    return Invalid(config,
                   "restart symmetry/map/schema metadata conflicts with cartoon_so2");
  }

  // This reusable analytic pgen supplies ordinary production evolution for both
  // Cartesian and Cartoon storage. Its parameter/topology checks remain inside
  // the pgen, while all allocation-free gates above apply unchanged here.
  if (input.problem_generator == "kerr_puncture" ||
      input.problem_generator == "z4c_irisk_xcts") {
    return {true, config, ""};
  }

  if (derivative_pgen &&
      input.cartoon_derivative_check_only_present &&
      !input.cartoon_derivative_check_only_valid) {
    return Invalid(config,
                   "<problem>/check_only must be an explicit boolean for "
                   "z4c_cartoon_derivatives");
  }
  const bool derivative_check = derivative_pgen &&
                                input.cartoon_derivative_check_only_present &&
                                input.cartoon_derivative_check_only_valid &&
                                input.cartoon_derivative_check_only;
  if (!derivative_check) {
    return Invalid(config, "problem generator '" + input.problem_generator +
                               "' is not the staged check_only Cartoon derivative MMS");
  }
  if (input.multilevel) {
    return Invalid(config,
                   "z4c_cartoon_derivatives check_only rejects AMR/SMR");
  }
  if (!input.outputs.empty()) {
    return Invalid(config,
                   "z4c_cartoon_derivatives check_only rejects Athena output blocks");
  }

  return {true, config, ""};
}

void DispatchCartoonZ4cKernel(const Z4cSymmetryConfig &config, void *context,
                              Z4cCartoonKernelEntry order2,
                              Z4cCartoonKernelEntry order4,
                              Z4cCartoonKernelEntry order6) {
  if (config.mode != Z4cSymmetryMode::cartoon_so2) {
    std::cerr << "### FATAL ERROR: Cartoon-only host dispatch received symmetry="
              << ToString(config.mode) << std::endl;
    std::exit(EXIT_FAILURE);
  }
  switch (config.stencil_width) {
    case 2: order2(context); return;
    case 3: order4(context); return;
    case 4: order6(context); return;
    default:
      std::cerr << "### FATAL ERROR: Cartoon-only host dispatch received stencil_width="
                << config.stencil_width << std::endl;
      std::exit(EXIT_FAILURE);
  }
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
                       : Z4cCoordinateMap::half_rho_z_suppressed_y_v2,
          NGHOST};
}

template Z4cKernelDispatchTarget InstantiateZ4cKernelTarget<Cartesian3D, 2>();
template Z4cKernelDispatchTarget InstantiateZ4cKernelTarget<Cartesian3D, 3>();
template Z4cKernelDispatchTarget InstantiateZ4cKernelTarget<Cartesian3D, 4>();
template Z4cKernelDispatchTarget InstantiateZ4cKernelTarget<CartoonSO2, 2>();
template Z4cKernelDispatchTarget InstantiateZ4cKernelTarget<CartoonSO2, 3>();
template Z4cKernelDispatchTarget InstantiateZ4cKernelTarget<CartoonSO2, 4>();

}  // namespace z4c
