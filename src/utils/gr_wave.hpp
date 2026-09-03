//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gr_wave.hpp
//! \brief formulation-independent ADM waveform diagnostics

#ifndef UTILS_GR_WAVE_HPP_
#define UTILS_GR_WAVE_HPP_

#include <memory>
#include <string>
#include <vector>

#include "athena.hpp"

class MeshBlockPack;
class SphericalGrid;

namespace gr_wave {

// ADM-based finite-radius Psi4 diagnostics shared by all vacuum spacetime systems.
template <int FD_STENCIL>
void CalculateWeyl(MeshBlockPack *pmbp, DvceArray5D<Real> u_weyl);

void ExtractWaveform(MeshBlockPack *pmbp,
                     std::vector<std::unique_ptr<SphericalGrid>> &spherical_grids,
                     DvceArray5D<Real> &u_weyl, Real *psi_out,
                     const std::string &output_directory);

}  // namespace gr_wave

#endif  // UTILS_GR_WAVE_HPP_
