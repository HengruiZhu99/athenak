// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_ACTIVITY_POLICY_HPP_
#define Z4C_HYPERBOLOIDAL_ACTIVITY_POLICY_HPP_

#include <array>
#include <set>
#include <vector>
#include "mesh/chart_atlas.hpp"

namespace z4c_hyperboloidal {
enum class OverlapMode {legacy, masked, agreement};
enum class PointActivity {inactive, fringe, evolved};

// The union of the actual bulk numerical dependencies, including mixed
// derivatives. A bounding cube would unnecessarily populate tensor corners.
inline const std::vector<std::array<int,3>> &BulkStencilOffsets() {
  static const auto offsets=[] {
    std::set<std::array<int,3>> out;
    for (int a=0;a<3;++a) for (int d=-4;d<=4;++d) {
      std::array<int,3> q{};q[a]=d;out.insert(q);
    }
    for (int a=0;a<3;++a) for (int b=a+1;b<3;++b)
      for (int da=-2;da<=2;++da) for (int db=-2;db<=2;++db) {
        std::array<int,3> q{};q[a]=da;q[b]=db;out.insert(q);
      }
    return std::vector<std::array<int,3>>(out.begin(),out.end());
  }();
  return offsets;
}

// Module-owned immutable geometry policy. Indices here are relative to origin.
// Rectangular storage and native block ownership are unaffected by this policy.
struct ActivityPolicy {
  OverlapMode mode=OverlapMode::legacy;
  Real shell_inner=0, cartesian_cutoff=0;
  KOKKOS_INLINE_FUNCTION bool Masked() const {return mode!=OverlapMode::legacy;}
  KOKKOS_INLINE_FUNCTION bool CartesianActive(const Real x[3]) const {
    return !Masked() || x[0]*x[0]+x[1]*x[1]+x[2]*x[2]
        <cartesian_cutoff*cartesian_cutoff;
  }
  KOKKOS_INLINE_FUNCTION bool Evolved(const chart_atlas::BlockMap &map,
                                     int n,int i,int j,int k) const {
    const int index[3]={i,j,k};
    for (int a=0;a<3;++a)
      if (index[a]<-map.overlap_sides[2*a] ||
          index[a]>=n+map.overlap_sides[2*a+1]) return false;
    if (!Masked()) return true;
    const Real q[3]={(i+map.node_offset[0])/n,(j+map.node_offset[1])/n,
                     (k+map.node_offset[2])/n};
    if (map.face<0) {Real x[3];map.Forward(q,x);return CartesianActive(x);}
    const Real r=map.inner_radius+(map.lo[0]+q[0]*(map.hi[0]-map.lo[0]))
        *(map.outer_radius-map.inner_radius);
    return r>=shell_inner;
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
