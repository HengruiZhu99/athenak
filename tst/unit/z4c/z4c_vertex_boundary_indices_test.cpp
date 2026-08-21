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

  std::cout << "PASS: native VC same-level boundary index contract" << std::endl;
  return EXIT_SUCCESS;
}
