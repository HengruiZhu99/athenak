#ifndef UTILS_FLUX_GENERALIZED_HPP_
#define UTILS_FLUX_GENERALIZED_HPP_
#include <array>
#include <vector>
#include "athena.hpp"
#include "outputs/outputs.hpp"
#include "utils/surface_grid.hpp"

constexpr int kSurfaceChannels = 44;
struct SurfaceFluxResult {
  std::array<Real, kSurfaceChannels> values{};
  bool valid = false;
  int reason = 0;
};
// Global result, reduced outside task lists. Contribute to history on rank 0.
SurfaceFluxResult IntegrateSurface(MeshBlockPack* pack, SphericalSurfaceGrid* surface);
void TorusFluxes_General(HistoryData* pdata, MeshBlockPack* pack,
                         const std::vector<SphericalSurfaceGrid*>& surfaces);
#endif  // UTILS_FLUX_GENERALIZED_HPP_
