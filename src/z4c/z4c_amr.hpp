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
  void RefineRadii(MeshBlockPack *pmbp);        // Refine based on the radii
  void WriteDchiShadow(MeshBlockPack *pmbp);    // Diagnostic-only Nyquist sensor

  RefinementMethod method;

  // Optinally set the minimum refinement level inside different radial shells
  std::vector<Real> radius;
  std::vector<int> reflevel;

  Real chi_thresh;     // chi threshold for chi refinement method
  Real dchi_thresh;    // dchi threshold for dchi refinement method
  Real dchi_derefine_factor = 0.25;  // derefine below this fraction of dchi threshold
  bool dchi_shadow_nyquist = false;  // default-off; never changes refinement flags
  int max_ref_lev;   // maximum level of refinement for chi and dchi
};

} // namespace z4c
#endif // Z4C_Z4C_AMR_HPP_
