//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file meshblock_pack.cpp
//  \brief implementation of constructor and functions in MeshBlockPack class

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
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "z4c/cce/cce.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/resistivity.hpp"
#include "radiation/radiation.hpp"
#include "srcterms/turb_driver.hpp"
#include "particles/particles.hpp"
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
  input.meshblock_nx3 = mesh.mb_indcs.nx3;
  input.root_blocks_x1 = mesh.nmb_rootx1;
  input.x1min = mesh.mesh_size.x1min;
  input.x1max = mesh.mesh_size.x1max;

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
  if (pin->DoesParameterExist("fastflow", "num_horizons") &&
      pin->GetInteger("fastflow", "num_horizons") > 0) {
    input.incompatible_consumers.emplace_back(
        "FastFlow before the m=0 Cartoon adapter is integrated");
  }
  for (const auto &block : pin->block) {
    if (block.block_name != "fastflow") continue;
    for (const auto &parameter : block.line) {
      const std::string &name = parameter.param_name;
      if (name.rfind("center_", 0) == 0 || name.rfind("use_puncture_", 0) == 0 ||
          name.rfind("wait_until_punc_are_close_", 0) == 0 ||
          name.rfind("use_puncture_massweighted_center_", 0) == 0) {
        input.incompatible_consumers.emplace_back("legacy FastFlow key " + name);
      }
    }
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
      output.mass_weighted =
          pin->DoesParameterExist(block.block_name, "mass_weighted") &&
          pin->GetBoolean(block.block_name, "mass_weighted");
      output.has_variable_2 =
          pin->DoesParameterExist(block.block_name, "variable_2");
      output.has_nbin2 = pin->DoesParameterExist(block.block_name, "nbin2");
      if (output.has_nbin2) output.nbin2 = pin->GetInteger(block.block_name, "nbin2");
      output.has_any_second_axis_key = HasAnySecondPdfKey(pin, block.block_name);
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
  input.problem_generator = pin->DoesParameterExist("problem", "pgen_name")
                                ? pin->GetString("problem", "pgen_name")
                                : "none";
#endif
  // The pgen/Kerr slice changes this only after its stored bounds and tensor map pass.
  input.accepted_cartoon_problem_generator = false;
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
