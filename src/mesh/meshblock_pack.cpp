//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file meshblock_pack.cpp
//  \brief implementation of constructor and functions in MeshBlockPack class

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <utility>
#include <memory>
#include <string>
#include <vector>

#include "athena.hpp"
#include "config.hpp"
#include "parameter_input.hpp"
#include "mesh.hpp"
#include "driver/driver.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "ion-neutral/ion-neutral.hpp"
#include "coordinates/adm.hpp"
#include "z4c/tmunu.hpp"
#include "tasklist/numerical_relativity.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_restart.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "z4c/cce/cce.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/resistivity.hpp"
#include "radiation/radiation.hpp"
#include "srcterms/turb_driver.hpp"
#include "particles/particles.hpp"
#include "pgen/pgen_defaults.hpp"
#include "units/units.hpp"
#include "meshblock_pack.hpp"
#include "z4c/z4c_symmetry.hpp"

namespace {

bool IsActiveOutput(ParameterInput *pin, const std::string &block_name) {
  if (pin->DoesParameterExist(block_name, "dcycle")) {
    return pin->GetInteger(block_name, "dcycle") != 0;
  }
  return pin->DoesParameterExist(block_name, "dt") &&
         pin->GetReal(block_name, "dt") > 0.0;
}

bool HasAnySecondPdfKey(ParameterInput *pin, const std::string &block_name) {
  for (const char *key : {"bin2_min", "bin2_max", "nbin2", "logscale2"}) {
    if (pin->DoesParameterExist(block_name, key)) return true;
  }
  return false;
}

z4c::Z4cValidationInput CollectZ4cValidationInput(ParameterInput *pin,
                                                  const Mesh &mesh) {
  z4c::Z4cValidationInput input;
  input.z4c_enabled = pin->DoesBlockExist("z4c");
  if (input.z4c_enabled) {
    input.requested_grid_centering =
        pin->DoesParameterExist("z4c", "grid_centering")
            ? pin->GetString("z4c", "grid_centering")
            : "cell";
    input.requested_symmetry = pin->DoesParameterExist("z4c", "symmetry")
                                   ? pin->GetString("z4c", "symmetry")
                                   : "cartesian3d";
    input.coordinate_map_specified =
        pin->DoesParameterExist("z4c", "coordinate_map");
    if (input.coordinate_map_specified) {
      input.coordinate_map = pin->GetString("z4c", "coordinate_map");
    }
    input.schema_specified = pin->DoesParameterExist("z4c", "symmetry_schema");
    if (input.schema_specified) {
      input.schema = pin->GetInteger("z4c", "symmetry_schema");
    }
  }

  input.nghost = mesh.mb_indcs.ng;
  const int default_spatial_order = 2 * (input.nghost - 1);
  input.requested_spatial_order =
      input.z4c_enabled && pin->DoesParameterExist("z4c", "spatial_order")
          ? pin->GetInteger("z4c", "spatial_order")
          : default_spatial_order;
  input.mesh_nx1 = mesh.mesh_indcs.nx1;
  input.mesh_nx2 = mesh.mesh_indcs.nx2;
  input.mesh_nx3 = mesh.mesh_indcs.nx3;
  input.meshblock_nx1 = mesh.mb_indcs.nx1;
  input.meshblock_nx3 = mesh.mb_indcs.nx3;
  input.root_blocks_x1 = mesh.nmb_rootx1;
  input.x1min = mesh.mesh_size.x1min;
  input.x1max = mesh.mesh_size.x1max;
  input.real_bytes = sizeof(Real);
  input.inner_x1_boundary = mesh.GetBoundaryString(
      mesh.mesh_bcs[BoundaryFace::inner_x1]);
  input.outer_x1_boundary = mesh.GetBoundaryString(
      mesh.mesh_bcs[BoundaryFace::outer_x1]);
  input.inner_x2_boundary = mesh.GetBoundaryString(
      mesh.mesh_bcs[BoundaryFace::inner_x2]);
  input.outer_x2_boundary = mesh.GetBoundaryString(
      mesh.mesh_bcs[BoundaryFace::outer_x2]);
  input.inner_x3_boundary = mesh.GetBoundaryString(
      mesh.mesh_bcs[BoundaryFace::inner_x3]);
  input.outer_x3_boundary = mesh.GetBoundaryString(
      mesh.mesh_bcs[BoundaryFace::outer_x3]);

  for (const char *block : {"hydro", "mhd", "ion-neutral", "radiation",
                            "turb_driving", "particles"}) {
    if (pin->DoesBlockExist(block)) input.incompatible_physics.emplace_back(block);
  }

  if (input.z4c_enabled) {
    for (const auto &line : pin->block) {
      if (line.block_name != "z4c") continue;
      for (const auto &parameter : line.line) {
        const std::string &name = parameter.param_name;
        if (name.rfind("co_", 0) == 0 &&
            name.size() > 5 && name.compare(name.size() - 5, 5, "_type") == 0) {
          input.incompatible_consumers.emplace_back("compact-object tracker " + name);
        }
        if (name.rfind("dump_horizon_", 0) == 0 &&
            pin->GetBoolean("z4c", name)) {
          input.incompatible_consumers.emplace_back("horizon dump " + name);
        }
      }
    }
    if (pin->DoesParameterExist("z4c", "nrad_wave_extraction") &&
        pin->GetReal("z4c", "nrad_wave_extraction") > 0.0) {
      input.incompatible_consumers.emplace_back("Z4c wave extraction");
    }
  }
  if (pin->DoesParameterExist("cce", "num_radii") &&
      pin->GetInteger("cce", "num_radii") > 0) {
    input.incompatible_consumers.emplace_back("CCE extraction");
  }
  if (input.requested_symmetry == "cartoon_so2" &&
      pin->DoesBlockExist("fastflow")) {
    const int horizon_count = pin->DoesParameterExist("fastflow", "num_horizons")
                                  ? pin->GetInteger("fastflow", "num_horizons") : 0;
    if (horizon_count != 0 && horizon_count != 1) {
      input.incompatible_consumers.emplace_back(
          "Cartoon m=0 FastFlow requires num_horizons=1");
    }
    const std::vector<std::string> supported_keys = {
        "num_horizons", "lmax", "ntheta", "flow_iterations_0",
        "find_interval_0", "start_time_0", "stop_time_0", "initial_radius_0",
        "flow_alpha_beta_const_0", "dimensionless_hrms_tol_0",
        "mass_relative_tol_0", "cartoon_surface_mode_0",
        "cartoon_direct_residual_tol_0", "cartoon_pair_relative_tol_0",
        "cartoon_center_z_0", "cartoon_axis_search_bound_0",
        "cartoon_axis_search_samples_0"};
    for (const auto &block : pin->block) {
      if (block.block_name != "fastflow") continue;
      for (const auto &parameter : block.line) {
        const std::string &name = parameter.param_name;
        if (std::find(supported_keys.begin(), supported_keys.end(), name) ==
            supported_keys.end()) {
          input.incompatible_consumers.emplace_back(
              "unsupported Cartoon m=0 FastFlow key " + name);
        }
      }
    }
    if (horizon_count == 1 &&
        pin->DoesParameterExist("fastflow", "cartoon_surface_mode_0") &&
        pin->GetString("fastflow", "cartoon_surface_mode_0") == "mirror_pair" &&
        pin->DoesParameterExist("coord", "excision_scheme") &&
        pin->GetString("coord", "excision_scheme") == "horizon") {
      input.incompatible_consumers.emplace_back(
          "Cartoon mirror_pair cannot feed single-surface horizon excision");
    }
  }
  if (input.requested_grid_centering == "vertex" &&
      input.requested_symmetry == "cartesian3d" &&
      pin->DoesParameterExist("fastflow", "num_horizons") &&
      pin->GetInteger("fastflow", "num_horizons") > 0) {
    input.incompatible_consumers.emplace_back("Cartesian FastFlow");
  }

  for (const auto &block : pin->block) {
    if (block.block_name.rfind("output", 0) != 0 ||
        !IsActiveOutput(pin, block.block_name)) {
      continue;
    }
    z4c::Z4cOutputValidationRequest output;
    output.block_name = block.block_name;
    output.file_type = pin->DoesParameterExist(block.block_name, "file_type")
                           ? pin->GetString(block.block_name, "file_type")
                           : "";
    if (output.file_type == "pdf") {
      auto &pdf = output.pdf_input;
      pdf.block_name = block.block_name;
      pdf.mass_weighted =
          pin->DoesParameterExist(block.block_name, "mass_weighted") &&
          pin->GetBoolean(block.block_name, "mass_weighted");
      pdf.variable_2_specified =
          pin->DoesParameterExist(block.block_name, "variable_2");
      pdf.has_variable_2 =
          pdf.variable_2_specified &&
          !pin->GetString(block.block_name, "variable_2").empty();
      pdf.has_nbin = pin->DoesParameterExist(block.block_name, "nbin");
      pdf.has_bin_min = pin->DoesParameterExist(block.block_name, "bin_min");
      pdf.has_bin_max = pin->DoesParameterExist(block.block_name, "bin_max");
      if (pdf.has_nbin) pdf.nbin = pin->GetInteger(block.block_name, "nbin");
      if (pdf.has_bin_min) pdf.bin_min = pin->GetReal(block.block_name, "bin_min");
      if (pdf.has_bin_max) pdf.bin_max = pin->GetReal(block.block_name, "bin_max");
      pdf.logscale = !pin->DoesParameterExist(block.block_name, "logscale") ||
                     pin->GetBoolean(block.block_name, "logscale");
      pdf.has_nbin2 = pin->DoesParameterExist(block.block_name, "nbin2");
      pdf.has_bin2_min = pin->DoesParameterExist(block.block_name, "bin2_min");
      pdf.has_bin2_max = pin->DoesParameterExist(block.block_name, "bin2_max");
      if (pdf.has_nbin2) pdf.nbin2 = pin->GetInteger(block.block_name, "nbin2");
      if (pdf.has_bin2_min) pdf.bin2_min = pin->GetReal(block.block_name, "bin2_min");
      if (pdf.has_bin2_max) pdf.bin2_max = pin->GetReal(block.block_name, "bin2_max");
      pdf.logscale2 = !pin->DoesParameterExist(block.block_name, "logscale2") ||
                      pin->GetBoolean(block.block_name, "logscale2");
      pdf.has_any_second_axis_key = HasAnySecondPdfKey(pin, block.block_name);
    }
    input.outputs.push_back(output);
  }

  const bool has_restart_symmetry =
      pin->DoesParameterExist("z4c", "restart_symmetry");
  const bool has_restart_map =
      pin->DoesParameterExist("z4c", "restart_coordinate_map");
  const bool has_restart_schema =
      pin->DoesParameterExist("z4c", "restart_symmetry_schema");
  input.restart_metadata_present =
      has_restart_symmetry || has_restart_map || has_restart_schema;
  input.restart_carrier_present = pin->DoesBlockExist(z4c::kZ4cRestartBlock);
  input.restart_origin = mesh.restart_origin;
  if (input.restart_metadata_present) {
    input.restart_symmetry = has_restart_symmetry
                                 ? pin->GetString("z4c", "restart_symmetry")
                                 : "";
    input.restart_coordinate_map = has_restart_map
                                       ? pin->GetString("z4c", "restart_coordinate_map")
                                       : "";
    input.restart_schema = has_restart_schema
                               ? pin->GetInteger("z4c", "restart_symmetry_schema")
                               : 0;
  }

#if USER_PROBLEM_ENABLED
  input.problem_generator = PROBLEM_GENERATOR;
#else
  const char *default_pgen = DefaultInputSelectedPgen(PROBLEM_GENERATOR);
  input.problem_generator = pin->DoesParameterExist("problem", "pgen_name")
                                ? pin->GetString("problem", "pgen_name")
                                : default_pgen;
#endif
  input.cartoon_derivative_check_only_present =
      pin->DoesParameterExist("problem", "check_only");
  if (input.cartoon_derivative_check_only_present) {
    std::string raw_check_only = pin->GetString("problem", "check_only");
    std::transform(raw_check_only.begin(), raw_check_only.end(),
                   raw_check_only.begin(), [](const unsigned char character) {
                     return static_cast<char>(std::tolower(character));
                   });
    input.cartoon_derivative_check_only_valid =
        raw_check_only == "true" || raw_check_only == "false" ||
        raw_check_only == "1" || raw_check_only == "0";
    input.cartoon_derivative_check_only =
        raw_check_only == "true" || raw_check_only == "1";
  }
  input.multilevel = mesh.multilevel;
  return input;
}

void ValidateAndStoreZ4cSymmetry(ParameterInput *pin, MeshBlockPack *pack) {
  const auto validation =
      z4c::ValidateZ4cSymmetry(CollectZ4cValidationInput(pin, *pack->pmesh));
  if (!validation.valid) {
    std::cerr << "### FATAL ERROR in " << __FILE__ << ": Z4c preallocation "
              << "validation failed: " << validation.error << std::endl;
    std::exit(EXIT_FAILURE);
  }
  pack->z4c_symmetry = validation.config;
  if (pin->DoesBlockExist(z4c::kZ4cRestartBlock)) {
    z4c::Z4cRestartSnapshot snapshot;
    const auto restart = z4c::CaptureZ4cRestartSnapshot(pin, &snapshot);
    if (!restart.valid || !snapshot.present) {
      std::cerr << "### FATAL ERROR in " << __FILE__
                << ": invalid Z4c restart carrier: " << restart.error << std::endl;
      std::exit(EXIT_FAILURE);
    }
    pack->z4c_restart_state = snapshot.state;
  } else {
    const int requested_order = pin->DoesParameterExist("z4c", "spatial_order")
                                    ? pin->GetInteger("z4c", "spatial_order")
                                    : 2 * (pack->pmesh->mb_indcs.ng - 1);
    pack->z4c_restart_state = z4c::MakeDefaultZ4cRestartState(
        validation.config, requested_order, pack->pmesh->mb_indcs.ng,
        pack->pmesh->mesh_indcs.nx1, pack->pmesh->mesh_indcs.nx2,
        pack->pmesh->mesh_indcs.nx3, pack->pmesh->mb_indcs.nx1,
        pack->pmesh->mb_indcs.nx2, pack->pmesh->mb_indcs.nx3);
  }
}

}  // namespace

//----------------------------------------------------------------------------------------
// MeshBlockPack constructor:

MeshBlockPack::MeshBlockPack(Mesh *pm, int igids, int igide) :
  pmesh(pm),
  gids(igids),
  gide(igide),
  nmb_thispack(igide - igids + 1) {
  // create map for task lists
  tl_map.insert(std::make_pair("before_timeintegrator",std::make_shared<TaskList>()));
  tl_map.insert(std::make_pair("after_timeintegrator",std::make_shared<TaskList>()));
  tl_map.insert(std::make_pair("before_stagen",std::make_shared<TaskList>()));
  tl_map.insert(std::make_pair("stagen",std::make_shared<TaskList>()));
  tl_map.insert(std::make_pair("after_stagen",std::make_shared<TaskList>()));
}

//----------------------------------------------------------------------------------------
// MeshBlock destructor

MeshBlockPack::~MeshBlockPack() {
  if (ppart  != nullptr) {delete ppart;}
  if (pnr    != nullptr) {delete pnr;}
  if (pdyngr != nullptr) {delete pdyngr;}
  if (ptmunu != nullptr) {delete ptmunu;}
  if (padm   != nullptr) {delete padm;}
  if (pz4c   != nullptr) {
    delete pz4c;
    // cce dump
    for (auto cce : pz4c_cce) {
      delete cce;
    }
    pz4c_cce.resize(0);
  }
  if (pturb  != nullptr) {delete pturb;}
  if (prad   != nullptr) {delete prad;}
  if (pmhd   != nullptr) {delete pmhd;}
  if (phydro != nullptr) {delete phydro;}
  if (punit  != nullptr) {delete punit;}
  delete pcoord;
  delete pmb;
}

//----------------------------------------------------------------------------------------
//! \fn MeshBlockPack::AddMeshBlocks(ParameterInput *pin)
//! \brief Wrapper function for calling MeshBlock constructor inside MeshBlockPack.
//! Allows for passing of pointer to 'this' pack.

void MeshBlockPack::AddMeshBlocks(ParameterInput *pin) {
  pmb = new MeshBlock(this, gids, nmb_thispack);
}

//----------------------------------------------------------------------------------------
//! \fn MeshBlockPack::AddCoordinates(ParameterInput *pin)
//! \brief Wrapper function for calling Coordinates constructor inside MeshBlockPack.
//! Allows for passing of pointer to 'this' pack. Must be called BEFORE AddPhysics()
//! function, since latter uses data inside Coordinates class.

void MeshBlockPack::AddCoordinates(ParameterInput *pin) {
  pcoord = new Coordinates(pin, this);
}

//----------------------------------------------------------------------------------------
// \fn MeshBlockPack::AddPhysics()
// \brief construct physics modules and tasks lists in this MeshBlockPack, based on which
// <blocks> are present in the input file.  Called from main().

void MeshBlockPack::AddPhysics(ParameterInput *pin) {
  ValidateAndStoreZ4cSymmetry(pin, this);

  int nphysics = 0;
  TaskID none(0);

  // (1) Units.  Create first so that they can be used in other physics constructors
  // Default units are simply code units
  if (pin->DoesBlockExist("units")) {
    punit = new units::Units(pin);
  } else {
    punit = nullptr;
  }

  // (2) HYDRODYNAMICS
  // Create Hydro physics module.  Create TaskLists only for single-fluid hydro
  // (Note TaskLists stored in MeshBlockPack)
  if (pin->DoesBlockExist("hydro")) {
    phydro = new hydro::Hydro(this, pin);
    nphysics++;
    if (!(pin->DoesBlockExist("mhd")) && !(pin->DoesBlockExist("radiation")) &&
        !(pin->DoesBlockExist("adm")) && !(pin->DoesBlockExist("z4c")) ) {
      phydro->AssembleHydroTasks(tl_map);
    }
  } else {
    phydro = nullptr;
  }

  // (3) MHD
  // Create MHD physics module.  Create TaskLists only for single-fluid MHD
  if (pin->DoesBlockExist("mhd")) {
    pmhd = new mhd::MHD(this, pin);
    nphysics++;
    if (!(pin->DoesBlockExist("hydro")) && !(pin->DoesBlockExist("radiation")) &&
        !(pin->DoesBlockExist("adm")) && !(pin->DoesBlockExist("z4c")) ) {
      pmhd->AssembleMHDTasks(tl_map);
    }
  } else {
    pmhd = nullptr;
  }

  // (4) ION_NEUTRAL (two-fluid) MHD
  // Create Ion-Neutral physics module and TaskLists. Error if <hydro> and <mhd> are not
  // both defined as well.
  if (pin->DoesBlockExist("ion-neutral")) {
    pionn = new ion_neutral::IonNeutral(this, pin);   // construct new MHD object
    if (pin->DoesBlockExist("hydro") && pin->DoesBlockExist("mhd") &&
        !(pin->DoesBlockExist("adm")) && !(pin->DoesBlockExist("z4c")) ) {
      pionn->AssembleIonNeutralTasks(tl_map);
      nphysics++;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<ion-neutral> block detected in input file, but either"
                << " <hydro> or <mhd> block missing" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  } else {
    // Error if both <hydro> and <mhd> defined, but not <ion-neutral>
    if (pin->DoesBlockExist("hydro") && pin->DoesBlockExist("mhd")) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Both <hydro> and <mhd> blocks detected in input file, "
                << "but <ion-neutral> block missing" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    pionn = nullptr;
  }

  // (5) RADIATION
  // Create radiation physics module.  Create tasklist.
  if (pin->DoesBlockExist("radiation")) {
    prad = new radiation::Radiation(this, pin);
    nphysics++;
    prad->AssembleRadTasks(tl_map);
  } else {
    prad = nullptr;
  }

  // (6) TURBULENCE DRIVER
  // This is a special module to drive turbulence in hydro, MHD, or both. Cannot be
  // included as a source term since it requires evolving force array via O-U process.
  // Instead, TurbulenceDriver object is stored in MeshBlockPack and tasks for evolving
  // force and adding force to fluid are included in operator_split and stage_run
  // task lists respectively.
  if (pin->DoesBlockExist("turb_driving")) {
    pturb = new TurbulenceDriver(this, pin);
    pturb->IncludeInitializeModesTask(tl_map["before_timeintegrator"], none);
    pturb->IncludeAddForcingTask(tl_map["stagen"], none);
  } else {
    pturb = nullptr;
  }

  // (7) Z4c and ADM
  // Create Z4c and ADM physics module.
  if (pin->DoesBlockExist("z4c")) {
    pz4c = new z4c::Z4c(this, pin);
    padm = new adm::ADM(this, pin);
    ptmunu = nullptr;
    // init cce dump
    pz4c_cce.reserve(0);
    int ncce = pin->GetOrAddInteger("cce", "num_radii", 0);
    pz4c_cce.reserve(ncce);// 10 different components for each radius
    for(int n = 0; n < ncce; ++n) {
      // NOTE: these names are used for pittnull code, so DON'T change the convention
      pz4c_cce.push_back(new z4c::CCE(pmesh, pin,n));
    }
    nphysics++;
  } else {
    pz4c = nullptr;
    if (pin->DoesBlockExist("adm")) {
      padm = new adm::ADM(this, pin);
    } else {
      padm = nullptr;
    }
  }

  // (8) Dynamical Spacetime and Matter (MHD TODO)
  if ((pin->DoesBlockExist("z4c") || pin->DoesBlockExist("adm")) &&
      (pin->DoesBlockExist("hydro")) ) {
    std::cout << "Dynamical metric and hydro not compatible; use MHD instead  "
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if ((pin->DoesBlockExist("z4c") || pin->DoesBlockExist("adm")) &&
      (pin->DoesBlockExist("mhd")) ) {
    pdyngr = dyngr::BuildDynGRMHD(this, pin);
    ptmunu = new Tmunu(this, pin);
  }

  if (pz4c != nullptr || padm != nullptr) {
    pnr = new numrel::NumericalRelativity(this, pin);
    pnr->AssembleNumericalRelativityTasks(tl_map);
  }

  // (9) PARTICLES
  // Create particles module.  Create tasklist.
  if (pin->DoesBlockExist("particles")) {
    ppart = new particles::Particles(this, pin);
    ppart->AssembleTasks(tl_map);
    nphysics++;
  } else {
    ppart = nullptr;
  }

  // Check that at least ONE is requested and initialized.
  // Error if there are no physics blocks in the input file.
  if (nphysics == 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "At least one physics module must be specified in input file." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  return;
}
