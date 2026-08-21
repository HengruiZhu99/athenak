//========================================================================================
// AthenaK astrophysical plasma code
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

#include <cstdlib>
#include <iostream>

#include "bvals/vertex_boundary_indices.hpp"

namespace {

void Require(const bool condition, const char *message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

void RequireEqualCount(const vertex_bvals::VertexIndexRange left,
                       const vertex_bvals::VertexIndexRange right,
                       const char *message) {
  if (!(vertex_bvals::IsValidRange(left) &&
        vertex_bvals::IsValidRange(right) &&
        left.count() == right.count())) {
    std::cerr << "FAIL: " << message << " left=[" << left.lower << ','
              << left.upper << "] right=[" << right.lower << ','
              << right.upper << "]" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

void CheckRelation(const int ox1, const int ox2, const int ox3,
                   const int f1, const int f2,
                   const bool collapse_x2, const bool collapse_x3,
                   const int ng) {
  constexpr int fine_start = 4;
  constexpr int fine_end = 12;
  constexpr int coarse_start = 2;
  constexpr int coarse_end = 6;
  const int offsets[3] = {ox1, ox2, ox3};
  const bool collapsed[3] = {false, collapse_x2, collapse_x3};
  for (int direction = 0; direction < 3; ++direction) {
    const int selector = vertex_bvals::TangentialSelector(
        direction, ox1, ox2, ox3, f1, f2, collapse_x2, collapse_x3);
    const int fine_s = collapsed[direction] ? 0 : fine_start;
    const int fine_e = collapsed[direction] ? 0 : fine_end;
    const int coarse_s = collapsed[direction] ? 0 : coarse_start;
    const int coarse_e = collapsed[direction] ? 0 : coarse_end;

    const auto fine_to_coarse_send = vertex_bvals::FineToCoarseSendRange(
        coarse_s, coarse_e, ng, offsets[direction], selector,
        collapsed[direction]);
    const auto fine_to_coarse_recv = vertex_bvals::FineToCoarseRecvRange(
        fine_s, fine_e, ng, offsets[direction], selector,
        collapsed[direction]);
    RequireEqualCount(fine_to_coarse_send, fine_to_coarse_recv,
                      "fine-to-coarse relation counts must match");

    const auto coarse_to_fine_send = vertex_bvals::CoarseToFineSendRange(
        fine_s, fine_e, ng, offsets[direction], selector,
        collapsed[direction]);
    const auto coarse_to_fine_recv = vertex_bvals::CoarseToFineRecvRange(
        coarse_s, coarse_e, ng, offsets[direction], selector,
        collapsed[direction]);
    RequireEqualCount(coarse_to_fine_send, coarse_to_fine_recv,
                      "coarse-to-fine relation counts must match");

    const int fine_stored_lower = collapsed[direction] ? 0 : fine_start - ng;
    const int fine_stored_upper = collapsed[direction] ? 0 : fine_end + ng;
    const int coarse_stored_lower = collapsed[direction] ? 0 : coarse_start - ng;
    const int coarse_stored_upper = collapsed[direction] ? 0 : coarse_end + ng;
    Require(fine_to_coarse_send.lower >= coarse_stored_lower &&
                fine_to_coarse_send.upper <= coarse_stored_upper,
            "fine-to-coarse send must stay inside coarse cache storage");
    Require(fine_to_coarse_recv.lower >= fine_stored_lower &&
                fine_to_coarse_recv.upper <= fine_stored_upper,
            "fine-to-coarse receive must stay inside coarse active storage");
    Require(coarse_to_fine_send.lower >= fine_stored_lower &&
                coarse_to_fine_send.upper <= fine_stored_upper,
            "coarse-to-fine send must stay inside coarse block storage");
    Require(coarse_to_fine_recv.lower >= coarse_stored_lower &&
                coarse_to_fine_recv.upper <= coarse_stored_upper,
            "coarse-to-fine receive must stay inside fine coarse-cache storage");
  }
}

}  // namespace

int main() {
  constexpr int start = 4;
  constexpr int end = 12;  // eight cells, nine active vertices
  constexpr int ng = 4;

  const auto send_lower =
      vertex_bvals::VertexSendRange(start, end, ng, -1, false);
  const auto send_upper =
      vertex_bvals::VertexSendRange(start, end, ng, +1, false);
  const auto send_tangent =
      vertex_bvals::VertexSendRange(start, end, ng, 0, false);
  Require(send_lower.lower == 4 && send_lower.upper == 8,
          "lower send must contain boundary plus ng interior vertices");
  Require(send_upper.lower == 8 && send_upper.upper == 12,
          "upper send must contain boundary plus ng interior vertices");
  Require(send_tangent.lower == 4 && send_tangent.upper == 12,
          "tangential send must contain every active vertex");
  Require(send_lower.count() == ng + 1 && send_upper.count() == ng + 1,
          "normal send count must be ng+1");

  const auto recv_lower =
      vertex_bvals::VertexRecvRange(start, end, ng, -1, false);
  const auto recv_upper =
      vertex_bvals::VertexRecvRange(start, end, ng, +1, false);
  Require(recv_lower.lower == 0 && recv_lower.upper == 4,
          "lower receive must contain ghosts plus shared boundary vertex");
  Require(recv_upper.lower == 12 && recv_upper.upper == 16,
          "upper receive must contain shared boundary vertex plus ghosts");
  Require(recv_lower.count() == ng + 1 && recv_upper.count() == ng + 1,
          "normal receive count must be ng+1");

  const auto collapsed_send =
      vertex_bvals::VertexSendRange(start, end, ng, +1, true);
  const auto collapsed_recv =
      vertex_bvals::VertexRecvRange(start, end, ng, -1, true);
  Require(collapsed_send.lower == 0 && collapsed_send.upper == 0 &&
              collapsed_recv.lower == 0 && collapsed_recv.upper == 0,
          "collapsed direction must remain a singleton");

  Require(vertex_bvals::IsActiveVertex(4, 4, 0, 4, 12, 4, 12, 0, 0),
          "shared lower corner is active and must not be overwritten");
  Require(!vertex_bvals::IsActiveVertex(3, 4, 0, 4, 12, 4, 12, 0, 0),
          "ghost vertex must be unpacked");

  // Exhaust all face/edge/corner orientations and child selectors in full 3D,
  // Cartoon 2D, and 1D for the O2/O4/O6 native ghost widths (2,3,4).
  const int ghost_widths[3] = {2, 3, 4};
  for (const int native_ng : ghost_widths) {
    for (int ox3 = -1; ox3 <= 1; ++ox3) {
      for (int ox2 = -1; ox2 <= 1; ++ox2) {
        for (int ox1 = -1; ox1 <= 1; ++ox1) {
          if (ox1 == 0 && ox2 == 0 && ox3 == 0) continue;
          for (int f2 = 0; f2 <= 1; ++f2) {
            for (int f1 = 0; f1 <= 1; ++f1) {
              CheckRelation(ox1, ox2, ox3, f1, f2, false, false, native_ng);
              if (ox3 == 0) {
                CheckRelation(ox1, ox2, ox3, f1, f2, false, true, native_ng);
              }
              if (ox2 == 0 && ox3 == 0) {
                CheckRelation(ox1, ox2, ox3, f1, f2, true, true, native_ng);
              }
            }
          }
        }
      }
    }
  }

  std::cout << "PASS: native VC same/coarse/fine boundary index contract"
            << std::endl;
  return EXIT_SUCCESS;
}
