// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_HORIZON_SAMPLING_HPP_
#define Z4C_HYPERBOLOIDAL_HORIZON_SAMPLING_HPP_

#include <cstddef>
#include <memory>

#include "athena.hpp"
#include "z4c/horizon_geometry.hpp"

class MeshBlockPack;
namespace z4c_hyperboloidal {
class Z4cHyperboloidal;

// A collective batch callback for the existing FastFlow solver. Only touched
// six-point tensor nodes and their evolved residual donors are staged on host.
// The caller first prepares the module's accepted-state residual workspace.
class HorizonSampler {
 public:
  struct Statistics {
    std::size_t batches=0, owned_queries=0, nodes=0, ghost_nodes=0;
    std::size_t exchanges=0, requested_donors=0, remote_donors=0;
  };
  // Construction and SampleBatch are collective, including empty-owner ranks.
  HorizonSampler(MeshBlockPack *pack,Z4cHyperboloidal *module);
  ~HorizonSampler();
  HorizonSampler(const HorizonSampler &)=delete;
  HorizonSampler &operator=(const HorizonSampler &)=delete;
  void SampleBatch(int count,const Real *positions,
                   z4c::HorizonGeometry<Real> *geometry,int *owners);
  // Optional explicit invalidation; StateVersion and the native mesh sequence
  // are also checked automatically before every batch.
  void Invalidate();
  const Statistics &GetStatistics() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace z4c_hyperboloidal
#endif  // Z4C_HYPERBOLOIDAL_HORIZON_SAMPLING_HPP_
