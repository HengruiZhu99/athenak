// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_ACTIVITY_POLICY_HPP_
#define Z4C_HYPERBOLOIDAL_ACTIVITY_POLICY_HPP_

#include <array>
#include <set>
#include <vector>
#include "mesh/chart_atlas.hpp"

namespace z4c_hyperboloidal {
enum class PointActivity {inactive, fringe, evolved};

// The union of the actual bulk numerical dependencies, including mixed
// derivatives. A bounding cube would unnecessarily populate tensor corners.
// Mixed derivatives reach two nodes along each of two axes, or three with the
// seven-node angular stencils (angular_stencil_points = 7). Host-side setting, fixed
// by the module before any plan is built.
inline int &BulkStencilCornerRadius() {
  static int radius=2;
  return radius;
}
inline const std::vector<std::array<int,3>> &BulkStencilOffsets() {
  static std::vector<std::array<int,3>> offsets[2];
  const int corner=BulkStencilCornerRadius();
  auto &cached=offsets[corner==3];
  if (cached.empty()) {
    std::set<std::array<int,3>> out;
    for (int a=0;a<3;++a) for (int d=-4;d<=4;++d) {
      std::array<int,3> q{};q[a]=d;out.insert(q);
    }
    for (int a=0;a<3;++a) for (int b=a+1;b<3;++b)
      for (int da=-corner;da<=corner;++da) for (int db=-corner;db<=corner;++db) {
        std::array<int,3> q{};q[a]=da;q[b]=db;out.insert(q);
      }
    cached.assign(out.begin(),out.end());
  }
  return cached;
}

// Module-owned immutable geometry policy. Indices here are relative to origin.
// Rectangular storage and native block ownership are unaffected by this policy.
struct ActivityPolicy {
  KOKKOS_INLINE_FUNCTION bool Evolved(const chart_atlas::BlockMap &map,
                                     int n,int i,int j,int k) const {
    const int index[3]={i,j,k};
    for (int a=0;a<3;++a)
      if (index[a]<-map.overlap_sides[2*a] ||
          index[a]>=n+map.overlap_sides[2*a+1]) return false;
    return true;
  }
  PointActivity Classify(const chart_atlas::BlockMap &map,int n,
                         int i,int j,int k) const {
    if (Evolved(map,n,i,j,k)) return PointActivity::evolved;
    for (const auto &d:BulkStencilOffsets())
      if (Evolved(map,n,i-d[0],j-d[1],k-d[2])) return PointActivity::fringe;
    return PointActivity::inactive;
  }
};
}  // namespace z4c_hyperboloidal
#endif  // Z4C_HYPERBOLOIDAL_ACTIVITY_POLICY_HPP_
