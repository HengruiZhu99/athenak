//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c_deterministic_amr.hpp
//! \brief Deterministic refine/derefine schedule shared by bounded Z4c test pgens.

#ifndef PGEN_TESTS_Z4C_DETERMINISTIC_AMR_HPP_
#define PGEN_TESTS_Z4C_DETERMINISTIC_AMR_HPP_

#include <cstdlib>
#include <iostream>

#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"

namespace z4c_test {

struct DeterministicAmrTarget {
  int lx1 = 0;
  int lx2 = 0;
  int lx3 = 0;
};

inline DeterministicAmrTarget deterministic_amr_target;
inline bool deterministic_amr_uses_time = false;
inline Real deterministic_amr_refine_time = 0.0;
inline Real deterministic_amr_derefine_time = 0.0;
inline int deterministic_amr_phase = 0;

inline void DeterministicRefinementSchedule(MeshBlockPack *pack) {
  Mesh *mesh = pack->pmesh;
  auto &flags = mesh->pmr->refine_flag;
  const int first_gid = mesh->gids_eachrank[global_variable::my_rank];
  const auto target = deterministic_amr_target;
  const bool refine_event = deterministic_amr_uses_time
      ? deterministic_amr_phase == 0 &&
            mesh->time >= deterministic_amr_refine_time
      : mesh->ncycle == 1;
  const bool derefine_event = deterministic_amr_uses_time
      ? deterministic_amr_phase == 1 &&
            mesh->time >= deterministic_amr_derefine_time
      : mesh->ncycle == 2;
  for (int m = 0; m < pack->nmb_thispack; ++m) {
    const int gid = first_gid + m;
    const auto &location = mesh->lloc_eachmb[gid];
    int flag = 0;
    if (refine_event && location.level == mesh->root_level &&
        location.lx1 == target.lx1 && location.lx2 == target.lx2 &&
        location.lx3 == target.lx3) {
      flag = 1;
    } else if (derefine_event &&
               location.level == mesh->root_level + 1 &&
               (location.lx1 >> 1) == target.lx1 &&
               (location.lx2 >> 1) == target.lx2 &&
               (location.lx3 >> 1) == target.lx3) {
      flag = -1;
    }
    flags.h_view(gid) = flag;
  }
  flags.template modify<HostMemSpace>();
  flags.template sync<DevExeSpace>();
  if (deterministic_amr_uses_time && refine_event) {
    deterministic_amr_phase = 1;
  } else if (deterministic_amr_uses_time && derefine_event) {
    deterministic_amr_phase = 2;
  }
}

inline bool ConfigureDeterministicRefinementSchedule(ParameterInput *pin,
                                                     const Mesh *mesh,
                                                     const char *pgen_name) {
  bool enabled = false;
  if (pin->DoesParameterExist("problem", "exercise_deterministic_amr")) {
    enabled = pin->GetBoolean("problem", "exercise_deterministic_amr");
  } else if (pin->DoesParameterExist("problem", "exercise_dynamic_vc_amr")) {
    // Preserve the original Minkowski fixture spelling without materializing it
    // in unrelated CC or linear-wave inputs.
    enabled = pin->GetBoolean("problem", "exercise_dynamic_vc_amr");
  }
  if (!enabled) return false;

  const bool has_refine_time =
      pin->DoesParameterExist("problem", "amr_refine_time");
  const bool has_derefine_time =
      pin->DoesParameterExist("problem", "amr_derefine_time");
  if (has_refine_time != has_derefine_time) {
    std::cerr << "### FATAL ERROR: " << pgen_name
              << " deterministic AMR requires both amr_refine_time and "
                 "amr_derefine_time when either is specified" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  deterministic_amr_uses_time = has_refine_time;
  deterministic_amr_phase = 0;
  if (deterministic_amr_uses_time) {
    deterministic_amr_refine_time =
        pin->GetReal("problem", "amr_refine_time");
    deterministic_amr_derefine_time =
        pin->GetReal("problem", "amr_derefine_time");
    if (!(deterministic_amr_refine_time >= 0.0 &&
          deterministic_amr_derefine_time > deterministic_amr_refine_time)) {
      std::cerr << "### FATAL ERROR: " << pgen_name
                << " requires 0 <= amr_refine_time < amr_derefine_time"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  deterministic_amr_target.lx1 =
      pin->GetOrAddInteger("problem", "amr_target_lx1", 0);
  deterministic_amr_target.lx2 = pin->GetOrAddInteger(
      "problem", "amr_target_lx2", mesh->nmb_rootx2 > 1 ? 1 : 0);
  deterministic_amr_target.lx3 = pin->GetOrAddInteger(
      "problem", "amr_target_lx3", mesh->nmb_rootx3 > 1 ? 1 : 0);
  const auto target = deterministic_amr_target;
  const bool valid_target =
      target.lx1 >= 0 && target.lx1 < mesh->nmb_rootx1 &&
      target.lx2 >= 0 && target.lx2 < mesh->nmb_rootx2 &&
      target.lx3 >= 0 && target.lx3 < mesh->nmb_rootx3;
  if (!valid_target) {
    std::cerr << "### FATAL ERROR: " << pgen_name << " AMR target ("
              << target.lx1 << "," << target.lx2 << "," << target.lx3
              << ") lies outside the root MeshBlock lattice ("
              << mesh->nmb_rootx1 << "," << mesh->nmb_rootx2 << ","
              << mesh->nmb_rootx3 << ")" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return true;
}

}  // namespace z4c_test

#endif  // PGEN_TESTS_Z4C_DETERMINISTIC_AMR_HPP_
