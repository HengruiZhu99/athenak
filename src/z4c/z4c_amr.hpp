//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

#ifndef Z4C_Z4C_AMR_HPP_
#define Z4C_Z4C_AMR_HPP_

#include <algorithm>
#include <string>
#include <vector>

#include "athena.hpp"
#include "z4c/z4c_grid.hpp"

class ParameterInput;
class MeshBlockPack;

namespace z4c {
class Z4c;

//! \brief Squared distance from a point to an axis-aligned MeshBlock box.
//!
//! A corner-only minimum is wrong whenever the closest point lies on a face, an
//! edge, or in the interior.  This helper is host-only because the Z4c radius
//! refinement policy is evaluated while refining the mesh tree.
inline Real SquaredDistanceToAABB(const Real x, const Real y, const Real z,
                                  const Real x1min, const Real x1max,
                                  const Real x2min, const Real x2max,
                                  const Real x3min, const Real x3max) {
  const Real cx = std::max(x1min, std::min(x, x1max));
  const Real cy = std::max(x2min, std::min(y, x2max));
  const Real cz = std::max(x3min, std::min(z, x3max));
  const Real dx = x - cx;
  const Real dy = y - cy;
  const Real dz = z - cz;
  return dx * dx + dy * dy + dz * dz;
}

//! \brief Recalibrate the dimensionless dchi trigger between root resolutions.
//!
//! The centered dchi sensor is 2*h*|grad chi| on an isotropic root mesh.  A
//! threshold calibrated at reference_nx1 therefore scales inversely with the
//! current root resolution when the physical radial domain is held fixed.
inline Real ResolutionScaledDchiThreshold(const Real reference_threshold,
                                          const int reference_nx1,
                                          const int current_nx1) {
  return reference_threshold * static_cast<Real>(reference_nx1) /
         static_cast<Real>(current_nx1);
}

//! \class Z4c_AMR
//  \brief managing AMR for Z4c simulations
class Z4c_AMR {
  enum RefinementMethod { Trivial, Tracker, Chi, dChi };

 public:
  explicit Z4c_AMR(ParameterInput *pin);
  ~Z4c_AMR() noexcept = default;

  void Refine(MeshBlockPack *pmbp);             // call the AMR method
  void RefineTracker(MeshBlockPack *pmbp);      // Refine based on trackers
  void RefineChiMin(MeshBlockPack *pmbp);       // Refine based on min{chi}
  void RefineDchiMax(MeshBlockPack *pmbp);      // Refine based on max{dchi}
  template <typename Centering>
  void RefineChiMinImpl(MeshBlockPack *pmbp);
  template <typename Centering>
  void RefineDchiMaxImpl(MeshBlockPack *pmbp);
  void RefineRadii(MeshBlockPack *pmbp);        // Refine based on the radii
  void WriteDchiShadow(MeshBlockPack *pmbp);    // Diagnostic-only Nyquist sensor

  RefinementMethod method;

  // Optinally set the minimum refinement level inside different radial shells
  std::vector<Real> radius;
  std::vector<int> reflevel;

  Real chi_thresh;     // chi threshold for chi refinement method
  Real dchi_thresh;    // dchi threshold for dchi refinement method
  Real dchi_reference_thresh = 0.0;  // unscaled value supplied by the input deck
  int dchi_reference_nx1 = 0;        // zero keeps the legacy unscaled behavior
  int dchi_current_nx1 = 0;
  Real dchi_derefine_factor = 0.25;  // derefine below this fraction of dchi threshold
  bool dchi_shadow_nyquist = false;  // default-off; never changes refinement flags
  bool capture_replay_dchi = false;  // diagnostic-only native criterion values in replay
  std::vector<Real> last_dchi_max;   // local MeshBlock maxima from the latest dchi check
  std::vector<int> last_dchi_ordinal;  // first active cell attaining each local maximum
  int max_ref_lev;   // maximum level of refinement for chi and dchi
};

} // namespace z4c
#endif // Z4C_Z4C_AMR_HPP_
