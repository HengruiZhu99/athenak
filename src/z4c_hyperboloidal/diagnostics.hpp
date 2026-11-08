// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_DIAGNOSTICS_HPP_
#define Z4C_HYPERBOLOIDAL_DIAGNOSTICS_HPP_

#include <array>
#include <cstdint>
#include <memory>

#include "athena.hpp"
#include "z4c/fastflow.hpp"
#include "z4c_hyperboloidal/conformal_rhs.hpp"
#include "z4c_hyperboloidal/fields.hpp"

class MeshBlockPack;
class ParameterInput;
namespace z4c_hyperboloidal {
class Z4cHyperboloidal;
class SurfaceExchange;
class HorizonSampler;

struct ConstraintPoint {
  EvolvedConstraintResult<Real> constraints{};
  Real chi=0, lapse=0, theta=0;
  PointStatus status=PointStatus::invalid_input;
};

// Accepted-state consumers share one read-only refresh. Owned fields, RK
// accumulators, chi histories and the actual stage observations are untouched.
// As with native history output, all ranks enter these methods together.
class Diagnostics {
 public:
  struct Statistics {
    std::uint64_t refreshes=0, searches=0;
    Real startup_seconds=0, preparation_seconds=0, constraint_seconds=0;
    Real horizon_seconds=0, reduction_seconds=0, io_seconds=0;
  };
  Diagnostics(MeshBlockPack *pack,Z4cHyperboloidal *module,ParameterInput *pin);
  ~Diagnostics();
  void Refresh();
  const FastFlow::Result &FindHorizon();
  std::array<Real,10> HistorySums();
  const DvceArray1D<ConstraintPoint> &Points() const {return points_;}
  const DvceArray1D<Real> &Weights() const {return weights_;}
  const DvceArray1D<int> &ExteriorMask() const {return exterior_;}
  const FastFlowSurface &Surface() const {return surface_;}
  const Statistics &GetStatistics() const {return statistics_;}

 private:
  MeshBlockPack *pack_;
  Z4cHyperboloidal *module_;
  ParameterInput *pin_;
  // Independent scri workspace preserves the last true RK stage observations.
  std::unique_ptr<SurfaceExchange> boundary_;
  std::unique_ptr<HorizonSampler> sampler_;
  std::unique_ptr<FastFlow> finder_;
  DvceArray1D<ConstraintPoint> points_;
  DvceArray1D<Real> weights_;
  DvceArray1D<int> exterior_;
  HostArray2D<Real> positions_;
  FastFlowSurface surface_;
  FastFlow::Result horizon_{};
  Statistics statistics_;
  std::uint64_t version_=0;
  int mesh_sequence_=-1, cycle_=-1;
  Real time_=0;
  bool ready_=false, horizon_ready_=false, sums_ready_=false, written_=false;
  std::array<Real,10> sums_{};
  void Rebuild();
  void Prepare();
  void EvaluateConstraints();
};
}  // namespace z4c_hyperboloidal
#endif  // Z4C_HYPERBOLOIDAL_DIAGNOSTICS_HPP_
