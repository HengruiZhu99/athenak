//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c_symmetry_validation_test.cpp
//! \brief Allocation-free negative tests for the Cartoon configuration contract.

#include <cstdlib>
#include <iostream>
#include <string>

#include "pgen/pgen_defaults.hpp"
#include "z4c/z4c_symmetry.hpp"

namespace {

z4c::Z4cValidationInput ValidCartoonInput() {
  z4c::Z4cValidationInput input;
  input.requested_symmetry = "cartoon_so2";
  input.coordinate_map_specified = true;
  input.coordinate_map = "half_rho_z_suppressed_y_v2";
  input.schema_specified = true;
  input.schema = z4c::Z4cSymmetryConfig::kCurrentSchema;
  input.z4c_enabled = true;
  input.nghost = 4;
  input.requested_spatial_order = 6;
  input.mesh_nx1 = 64;
  input.mesh_nx2 = 64;
  input.mesh_nx3 = 1;
  input.meshblock_nx1 = 32;
  input.meshblock_nx3 = 1;
  input.root_blocks_x1 = 2;
  input.x1min = 0.0;
  input.x1max = 24.0;
  input.inner_x1_boundary = "axis";
  input.outer_x1_boundary = "outflow";
  input.inner_x2_boundary = "outflow";
  input.outer_x2_boundary = "outflow";
  input.problem_generator = "z4c_cartoon_derivatives";
  input.cartoon_derivative_check_only_present = true;
  input.cartoon_derivative_check_only_valid = true;
  input.cartoon_derivative_check_only = true;
  return input;
}

bool Rejects(const z4c::Z4cValidationInput &input, const std::string &needle) {
  const auto result = z4c::ValidateZ4cSymmetry(input);
  return !result.valid && result.error.find(needle) != std::string::npos;
}

z4c::Z4cOutputValidationRequest ValidPdfOutput() {
  z4c::Z4cOutputValidationRequest output{"output7", "pdf"};
  output.pdf_input.block_name = output.block_name;
  output.pdf_input.has_nbin = true;
  output.pdf_input.has_bin_min = true;
  output.pdf_input.has_bin_max = true;
  output.pdf_input.nbin = 8;
  output.pdf_input.bin_min = 0.1;
  output.pdf_input.bin_max = 1.0;
  return output;
}

bool CheckDefaultCartesian() {
  z4c::Z4cValidationInput input;
  input.incompatible_physics = {"hydro"};
  input.outputs.push_back({"output1", "cart"});
  const auto result = z4c::ValidateZ4cSymmetry(input);
  return result.valid &&
         result.config.mode == z4c::Z4cSymmetryMode::cartesian3d &&
         result.config.coordinate_map == z4c::Z4cCoordinateMap::cartesian_xyz &&
         result.config.grid_centering == z4c::Z4cGridCentering::cell &&
         result.config.centering_schema == z4c::Z4cGridLayout::kCenteringSchema;
}

bool CheckCenteringSelection() {
  auto input = ValidCartoonInput();
  input.requested_grid_centering = "cell";
  auto result = z4c::ValidateZ4cSymmetry(input);
  if (!result.valid || result.config.grid_centering != z4c::Z4cGridCentering::cell ||
      std::string(z4c::ToString(result.config.grid_centering)) != "cell") {
    return false;
  }
  input.requested_grid_centering = "vertex";
  result = z4c::ValidateZ4cSymmetry(input);
  if (!result.valid ||
      result.config.grid_centering != z4c::Z4cGridCentering::vertex ||
      result.config.centering_schema != z4c::Z4cGridLayout::kCenteringSchema ||
      std::string(z4c::ToString(result.config.grid_centering)) != "vertex") {
    return false;
  }
  input.requested_grid_centering = "nodal";
  if (!Rejects(input, "grid_centering must be cell or vertex")) return false;
  z4c::Z4cValidationInput no_z4c;
  no_z4c.requested_grid_centering = "vertex";
  if (!Rejects(no_z4c, "requires the <z4c>")) return false;
  input = ValidCartoonInput();
  input.requested_grid_centering = "vertex";
  input.incompatible_physics = {"hydro"};
  if (!Rejects(input, "vacuum Z4c")) return false;
  input.incompatible_physics.clear();
  input.incompatible_consumers = {"Z4c wave extraction"};
  return Rejects(input, "without a centering-aware sampler");
}

bool CheckStencilDispatch() {
  for (const int spatial_order : {2, 4, 6}) {
    auto input = ValidCartoonInput();
    input.requested_spatial_order = spatial_order;
    const auto result = z4c::ValidateZ4cSymmetry(input);
    if (!result.valid || result.config.stencil_width != spatial_order / 2 + 1) {
      return false;
    }
  }
  auto input = ValidCartoonInput();
  input.requested_spatial_order = 8;
  if (!Rejects(input, "spatial_order")) return false;
  input = ValidCartoonInput();
  input.nghost = 2;
  if (!Rejects(input, "ghost cells")) return false;
  return true;
}

bool CheckNonpositiveSpatialOrderFallback() {
  for (const int nghost : {2, 3, 4}) {
    const int expected_order = 2 * (nghost - 1);
    if (z4c::EffectiveZ4cSpatialOrder(-1, nghost) != expected_order) return false;

    for (const char *symmetry : {"cartesian3d", "cartoon_so2"}) {
      auto input = ValidCartoonInput();
      input.requested_symmetry = symmetry;
      input.nghost = nghost;
      input.requested_spatial_order = -1;
      if (input.requested_symmetry == "cartesian3d") {
        input.coordinate_map = "cartesian_xyz";
        input.schema = z4c::Z4cSymmetryConfig::kCartesianSchema;
        input.inner_x1_boundary = "outflow";
        input.problem_generator = "none";
        input.cartoon_derivative_check_only_present = false;
      }
      const auto result = z4c::ValidateZ4cSymmetry(input);
      if (!result.valid || result.config.stencil_width != nghost) return false;
    }
  }

  // This is the sentinel used by the shipped one-puncture input deck.
  auto input = z4c::Z4cValidationInput{};
  input.z4c_enabled = true;
  input.nghost = 4;
  input.requested_spatial_order = -1;
  const auto shipped_deck = z4c::ValidateZ4cSymmetry(input);
  if (!shipped_deck.valid || shipped_deck.config.stencil_width != 4) return false;

  input.requested_spatial_order = 0;
  if (!z4c::ValidateZ4cSymmetry(input).valid) return false;
  input.requested_spatial_order = 8;
  return Rejects(input, "spatial_order");
}

bool CheckMeshAndPhysicsFailures() {
  auto allowed = ValidCartoonInput();
  allowed.mesh_nx1 = 63;
  allowed.root_blocks_x1 = 1;
  if (!z4c::ValidateZ4cSymmetry(allowed).valid) return false;

  auto input = ValidCartoonInput();
  input.requested_symmetry = "bad_mode";
  if (!Rejects(input, "cartesian3d or cartoon_so2")) return false;
  input = ValidCartoonInput();
  input.z4c_enabled = false;
  if (!Rejects(input, "requires the <z4c>")) return false;
  input = ValidCartoonInput();
  input.mesh_nx3 = 2;
  if (!Rejects(input, "nx3=meshblock/nx3=1")) return false;
  input = ValidCartoonInput();
  input.meshblock_nx3 = 2;
  if (!Rejects(input, "nx3=meshblock/nx3=1")) return false;
  input = ValidCartoonInput();
  input.mesh_nx2 = 1;
  if (!Rejects(input, "x1-x2")) return false;
  input = ValidCartoonInput();
  input.mesh_nx1 = 0;
  if (!Rejects(input, "positive <mesh>/nx1")) return false;
  input = ValidCartoonInput();
  input.meshblock_nx1 = 3;
  if (!Rejects(input, "meshblock/nx1 >= mesh/nghost")) return false;
  input = ValidCartoonInput();
  input.multilevel = true;
  input.meshblock_nx1 = 7;
  if (!Rejects(input, "2*mesh/nghost")) return false;
  input = ValidCartoonInput();
  input.root_blocks_x1 = 0;
  if (!Rejects(input, "positive root x1 MeshBlock")) return false;
  input = ValidCartoonInput();
  input.x1min = -1.0;
  if (!Rejects(input, "x1min=0")) return false;
  input = ValidCartoonInput();
  input.x1min = 1.0e-300;
  if (!Rejects(input, "x1min=0")) return false;
  input = ValidCartoonInput();
  input.x1max = 0.0;
  if (!Rejects(input, "x1max>0")) return false;
  input = ValidCartoonInput();
  input.inner_x1_boundary = "reflect";
  if (!Rejects(input, "ix1_bc=axis")) return false;
  input = ValidCartoonInput();
  input.outer_x1_boundary = "axis";
  if (!Rejects(input, "only at inner_x1")) return false;
  input = ValidCartoonInput();
  input.inner_x2_boundary = "axis";
  if (!Rejects(input, "only at inner_x1")) return false;
  auto cartesian = z4c::Z4cValidationInput{};
  cartesian.inner_x1_boundary = "axis";
  if (!Rejects(cartesian, "reserved for cartoon_so2 inner_x1")) return false;
  input = ValidCartoonInput();
  input.coordinate_map = "cartesian_xyz";
  if (!Rejects(input, "coordinate_map")) return false;
  input = ValidCartoonInput();
  input.schema = 99;
  if (!Rejects(input, "symmetry_schema")) return false;

  for (const char *block : {"hydro", "mhd", "ion-neutral", "radiation",
                            "turb_driving", "particles"}) {
    input = ValidCartoonInput();
    input.incompatible_physics.emplace_back(block);
    if (!Rejects(input, block)) return false;
  }
  return true;
}

bool CheckConsumerAndOutputFailures() {
  for (const char *consumer : {"compact-object tracker", "Z4c wave extraction",
                               "CCE extraction", "horizon dump", "legacy FastFlow"}) {
    auto input = ValidCartoonInput();
    input.incompatible_consumers.emplace_back(consumer);
    if (!Rejects(input, consumer)) return false;
  }

  for (const char *file_type : {"cart", "sph", "cbin", "pvtk", "trk"}) {
    auto input = ValidCartoonInput();
    input.outputs.push_back({"output7", file_type});
    if (!Rejects(input, std::string("file_type=") + file_type)) return false;
  }
  for (const char *file_type : {"tab", "hst", "log", "vtk", "pdf", "bin", "rst"}) {
    auto input = ValidCartoonInput();
    if (std::string(file_type) == "pdf") {
      input.outputs.push_back(ValidPdfOutput());
    } else {
      input.outputs.push_back({"output7", file_type});
    }
    if (!Rejects(input, "check_only rejects Athena output blocks")) return false;
  }
  auto input = ValidCartoonInput();
  input.outputs.push_back({"output7", "mystery"});
  if (!Rejects(input, "supported Cartoon types")) return false;

  input = ValidCartoonInput();
  auto pdf = ValidPdfOutput();
  pdf.pdf_input.mass_weighted = true;
  input.outputs.push_back(pdf);
  if (!Rejects(input, "mass_weighted=true")) return false;
  input = ValidCartoonInput();
  pdf = ValidPdfOutput();
  pdf.pdf_input.has_any_second_axis_key = true;
  input.outputs.push_back(pdf);
  if (!Rejects(input, "require variable_2")) return false;
  input = ValidCartoonInput();
  pdf = ValidPdfOutput();
  pdf.pdf_input.variable_2_specified = true;
  pdf.pdf_input.has_variable_2 = true;
  input.outputs.push_back(pdf);
  if (!Rejects(input, "explicit nbin2")) return false;
  input = ValidCartoonInput();
  pdf.pdf_input.has_nbin2 = true;
  pdf.pdf_input.nbin2 = 1;
  pdf.pdf_input.has_bin2_min = true;
  pdf.pdf_input.has_bin2_max = true;
  pdf.pdf_input.bin2_min = 0.1;
  pdf.pdf_input.bin2_max = 1.0;
  input.outputs.push_back(pdf);
  if (!Rejects(input, "check_only rejects Athena output blocks")) return false;
  return true;
}

bool CheckRestartAndPgenFailures() {
  auto cartesian = z4c::Z4cValidationInput{};
  cartesian.z4c_enabled = true;
  cartesian.problem_generator = "z4c_cartoon_derivatives";
  cartesian.cartoon_derivative_check_only_present = true;
  cartesian.cartoon_derivative_check_only_valid = true;
  cartesian.cartoon_derivative_check_only = true;
  if (!Rejects(cartesian, "requires cartoon_so2")) return false;
  cartesian.restart_origin = true;
  if (!Rejects(cartesian, "check_only rejects restart")) return false;

  auto input = ValidCartoonInput();
  input.restart_metadata_present = true;
  input.restart_symmetry = "cartesian3d";
  input.restart_coordinate_map = "cartesian_xyz";
  input.restart_schema = z4c::Z4cSymmetryConfig::kCurrentSchema;
  if (!Rejects(input, "restart")) return false;

  input = ValidCartoonInput();
  input.restart_metadata_present = true;
  input.restart_symmetry = "cartoon_so2";
  input.restart_coordinate_map = "wrong";
  input.restart_schema = z4c::Z4cSymmetryConfig::kCurrentSchema;
  if (!Rejects(input, "restart")) return false;

  input = ValidCartoonInput();
  input.restart_metadata_present = true;
  input.restart_symmetry = "cartoon_so2";
  input.restart_coordinate_map = "half_rho_z_suppressed_y_v2";
  input.restart_schema = 99;
  if (!Rejects(input, "restart")) return false;

  input = ValidCartoonInput();
  input.restart_metadata_present = true;
  input.restart_symmetry = "cartoon_so2";
  input.restart_coordinate_map = "half_rho_z_suppressed_y_v2";
  input.restart_schema = z4c::Z4cSymmetryConfig::kCurrentSchema;
  if (!Rejects(input, "check_only rejects restart")) return false;

  input = ValidCartoonInput();
  input.restart_carrier_present = true;
  if (!Rejects(input, "check_only rejects restart")) return false;

  input = ValidCartoonInput();
  input.restart_origin = true;
  if (!Rejects(input, "check_only rejects restart")) return false;

  input = ValidCartoonInput();
  input.cartoon_derivative_check_only = false;
  if (!Rejects(input, "not the staged check_only")) return false;

  input = ValidCartoonInput();
  input.cartoon_derivative_check_only_valid = false;
  if (!Rejects(input, "must be an explicit boolean")) return false;

  input = ValidCartoonInput();
  input.problem_generator = "z4c_linear_wave";
  if (!Rejects(input, "not the staged check_only")) return false;

  input = ValidCartoonInput();
  input.multilevel = true;
  if (!Rejects(input, "rejects AMR/SMR")) return false;
  return true;
}

bool CheckKerrPunctureProductionAdmission() {
  auto input = ValidCartoonInput();
  input.problem_generator = "kerr_puncture";
  input.cartoon_derivative_check_only_present = false;
  input.cartoon_derivative_check_only_valid = false;
  input.cartoon_derivative_check_only = false;
  input.multilevel = true;
  input.outputs = {{"output1", "hst"}, {"output2", "rst"}};
  auto result = z4c::ValidateZ4cSymmetry(input);
  if (!result.valid ||
      result.config.mode != z4c::Z4cSymmetryMode::cartoon_so2) {
    return false;
  }

  input.restart_origin = true;
  input.restart_metadata_present = true;
  input.restart_symmetry = "cartoon_so2";
  input.restart_coordinate_map = "half_rho_z_suppressed_y_v2";
  input.restart_schema = z4c::Z4cSymmetryConfig::kCurrentSchema;
  result = z4c::ValidateZ4cSymmetry(input);
  if (!result.valid) return false;

  input.restart_coordinate_map = "cartesian_xyz";
  if (!Rejects(input, "restart")) return false;

  input = ValidCartoonInput();
  input.problem_generator = "unreviewed_production_pgen";
  input.cartoon_derivative_check_only_present = false;
  input.cartoon_derivative_check_only_valid = false;
  input.cartoon_derivative_check_only = false;
  return Rejects(input, "not the staged check_only");
}

bool CheckIrisImporterProductionAdmission() {
  auto input = ValidCartoonInput();
  // Historical PROBLEM=z4c_irisk_xcts builds omit problem/pgen_name. Both
  // preallocation validation and runtime dispatch consume this shared default.
  if (std::string(DefaultInputSelectedPgen("z4c_irisk_xcts")) !=
          "z4c_irisk_xcts" ||
      std::string(DefaultInputSelectedPgen("none")) != "none") {
    return false;
  }
  input.problem_generator = DefaultInputSelectedPgen("z4c_irisk_xcts");
  input.cartoon_derivative_check_only_present = false;
  input.cartoon_derivative_check_only_valid = false;
  input.cartoon_derivative_check_only = false;
  input.multilevel = true;
  input.outputs = {{"output1", "hst"}, {"output2", "rst"},
                   {"output3", "bin"}};
  auto result = z4c::ValidateZ4cSymmetry(input);
  if (!result.valid ||
      result.config.mode != z4c::Z4cSymmetryMode::cartoon_so2) {
    return false;
  }

  // A restart uses persisted Z4c fields, but must still match every immutable
  // Cartoon carrier before the importer can re-enroll its AMR criterion.
  input.restart_origin = true;
  input.restart_metadata_present = true;
  input.restart_symmetry = "cartoon_so2";
  input.restart_coordinate_map = "half_rho_z_suppressed_y_v2";
  input.restart_schema = z4c::Z4cSymmetryConfig::kCurrentSchema;
  if (!z4c::ValidateZ4cSymmetry(input).valid) return false;
  input.restart_coordinate_map = "cartesian_xyz";
  if (!Rejects(input, "restart")) return false;

  input = ValidCartoonInput();
  input.problem_generator = "z4c_irisk_xcts";
  input.cartoon_derivative_check_only_present = false;
  input.incompatible_physics = {"hydro"};
  if (!Rejects(input, "hydro")) return false;
  input.incompatible_physics.clear();
  input.incompatible_consumers = {"legacy FastFlow"};
  if (!Rejects(input, "legacy FastFlow")) return false;
  input.incompatible_consumers.clear();
  input.meshblock_nx1 = 3;
  return Rejects(input, "meshblock/nx1 >= mesh/nghost");
}

}  // namespace

int main() {
  const bool passed = CheckDefaultCartesian() && CheckCenteringSelection() &&
                      CheckStencilDispatch() &&
                      CheckNonpositiveSpatialOrderFallback() &&
                      CheckMeshAndPhysicsFailures() &&
                      CheckConsumerAndOutputFailures() &&
                      CheckRestartAndPgenFailures() &&
                      CheckKerrPunctureProductionAdmission() &&
                      CheckIrisImporterProductionAdmission();
  if (!passed) return EXIT_FAILURE;
  std::cout << "Z4c Cartoon preallocation validation tests passed\n";
  return EXIT_SUCCESS;
}
