#ifndef Z4C_HYPERBOLOIDAL_RHS_SUBSET_HPP_
#define Z4C_HYPERBOLOIDAL_RHS_SUBSET_HPP_
//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

namespace z4c_hyperboloidal {

// MappedHalo classifies each owned evolved point by its complete numerical
// dependency, not by distance from a coordinate face alone. The surface class
// includes points requiring current-stage endpoint reconstruction/normal lift.
// Dependent joins the three non-interior classes only for batched execution.
enum class RHSSubset {Interior, LocalHalo, RemoteHalo, Surface, Dependent};

}  // namespace z4c_hyperboloidal
#endif  // Z4C_HYPERBOLOIDAL_RHS_SUBSET_HPP_
