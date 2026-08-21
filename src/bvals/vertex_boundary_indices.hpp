//========================================================================================
// AthenaK astrophysical plasma code
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file vertex_boundary_indices.hpp
//! \brief Exact same-level vertex-centered boundary index contracts.

#ifndef BVALS_VERTEX_BOUNDARY_INDICES_HPP_
#define BVALS_VERTEX_BOUNDARY_INDICES_HPP_

namespace vertex_bvals {

struct VertexIndexRange {
  int lower = 0;
  int upper = -1;

  constexpr int count() const { return upper - lower + 1; }
};

// A VC same-level send includes the shared boundary vertex plus ng interior vertices.
constexpr VertexIndexRange VertexSendRange(const int start, const int end,
                                           const int ng, const int offset,
                                           const bool collapsed) {
  if (collapsed) return {0, 0};
  if (offset < 0) return {start, start + ng};
  if (offset > 0) return {end - ng, end};
  return {start, end};
}

// A VC same-level receive includes the shared boundary vertex plus ng ghost vertices.
constexpr VertexIndexRange VertexRecvRange(const int start, const int end,
                                           const int ng, const int offset,
                                           const bool collapsed) {
  if (collapsed) return {0, 0};
  if (offset < 0) return {start - ng, start};
  if (offset > 0) return {end, end + ng};
  return {start, end};
}

// AthenaK orders f1/f2 as the ascending list of noncollapsed tangential axes.
// Return the child selector associated with one Cartesian direction, or -1 when
// that direction is normal to the neighbor or collapsed.
constexpr int TangentialSelector(const int direction, const int ox1,
                                 const int ox2, const int ox3,
                                 const int f1, const int f2,
                                 const bool collapse_x2,
                                 const bool collapse_x3) {
  const int offsets[3] = {ox1, ox2, ox3};
  const bool collapsed[3] = {false, collapse_x2, collapse_x3};
  int slot = 0;
  for (int d = 0; d < 3; ++d) {
    if (offsets[d] != 0 || collapsed[d]) continue;
    if (d == direction) return slot == 0 ? f1 : f2;
    ++slot;
  }
  return -1;
}

// A fine block injects coincident vertices from its coarse cache.  At a
// tangential child split the lower child owns the shared midpoint buffer slot,
// so the upper child skips its lower endpoint.
constexpr VertexIndexRange FineToCoarseSendRange(
    const int start, const int end, const int ng, const int offset,
    const int selector, const bool collapsed) {
  if (collapsed) return {0, 0};
  if (offset != 0) return VertexSendRange(start, end, ng, offset, false);
  return selector == 0 ? VertexIndexRange{start, end}
                       : VertexIndexRange{start + 1, end};
}

// Matching half of the coarse receiver's native active vertex interval.
constexpr VertexIndexRange FineToCoarseRecvRange(
    const int start, const int end, const int ng, const int offset,
    const int selector, const bool collapsed) {
  if (collapsed) return {0, 0};
  if (offset != 0) return VertexRecvRange(start, end, ng, offset, false);
  const int midpoint = start + (end - start) / 2;
  return selector == 0 ? VertexIndexRange{start, midpoint}
                       : VertexIndexRange{midpoint + 1, end};
}

// A coarse block supplies one child half plus the coarse halo needed for
// high-order vertex interpolation in every tangential direction.
constexpr VertexIndexRange CoarseToFineSendRange(
    const int start, const int end, const int coarse_ng, const int offset,
    const int selector, const bool collapsed) {
  if (collapsed) return {0, 0};
  if (offset != 0) {
    return VertexSendRange(start, end, coarse_ng, offset, false);
  }
  const int midpoint = start + (end - start) / 2;
  return selector == 0
      ? VertexIndexRange{start, midpoint + coarse_ng}
      : VertexIndexRange{midpoint - coarse_ng, end};
}

// Matching coarse-cache interval on the fine receiver.
constexpr VertexIndexRange CoarseToFineRecvRange(
    const int start, const int end, const int coarse_ng, const int offset,
    const int selector, const bool collapsed) {
  if (collapsed) return {0, 0};
  if (offset != 0) {
    return VertexRecvRange(start, end, coarse_ng, offset, false);
  }
  return selector == 0
      ? VertexIndexRange{start, end + coarse_ng}
      : VertexIndexRange{start - coarse_ng, end};
}

constexpr bool IsValidRange(const VertexIndexRange range) {
  return range.lower <= range.upper;
}

constexpr bool IsActiveVertex(const int i, const int j, const int k,
                              const int is, const int ie,
                              const int js, const int je,
                              const int ks, const int ke) {
  return i >= is && i <= ie && j >= js && j <= je && k >= ks && k <= ke;
}

}  // namespace vertex_bvals

#endif  // BVALS_VERTEX_BOUNDARY_INDICES_HPP_
