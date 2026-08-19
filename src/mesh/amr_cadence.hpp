//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file amr_cadence.hpp
//! \brief Small dependency-free AMR cadence contract.

#ifndef MESH_AMR_CADENCE_HPP_
#define MESH_AMR_CADENCE_HPP_

//! \brief AMR cadence is defined in integral cycles, never in a real-valued unit.
inline constexpr bool IsValidAMRCadence(const int ncycle_check,
                                        const int refinement_interval) {
  return ncycle_check > 0 && refinement_interval > 0;
}

#endif  // MESH_AMR_CADENCE_HPP_
