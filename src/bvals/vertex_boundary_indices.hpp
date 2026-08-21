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

constexpr bool IsActiveVertex(const int i, const int j, const int k,
                              const int is, const int ie,
                              const int js, const int je,
                              const int ks, const int ke) {
  return i >= is && i <= ie && j >= js && j <= je && k >= ks && k <= ke;
}

}  // namespace vertex_bvals

#endif  // BVALS_VERTEX_BOUNDARY_INDICES_HPP_
