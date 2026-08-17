#ifndef BVALS_COARSE_CACHE_OWNERSHIP_HPP_
#define BVALS_COARSE_CACHE_OWNERSHIP_HPP_

// Z4c communicates its owner-computed coarse representation explicitly through
// isame_z4c.  Generic finite-volume fields do not, so they retain the historical
// receiver-local same-level coarse refresh.
constexpr bool ShouldLocallyRefreshSameLevelCoarseCache(const bool is_z4c) {
  return !is_z4c;
}

#endif  // BVALS_COARSE_CACHE_OWNERSHIP_HPP_
