// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_REMESH_HPP_
#define Z4C_HYPERBOLOIDAL_REMESH_HPP_

#include <memory>
#include <vector>

#include "mesh/chart_atlas.hpp"
#include "z4c_hyperboloidal/z4c_hyperboloidal.hpp"

class MeshRefinement;

namespace z4c_hyperboloidal {
class NativeDonorSelector;
struct WeightedDonor;

// Metadata only: native old/new leaf identities and owners. No global field
// snapshot is permitted in this context. Indices in every vector are GIDs.
struct RemeshLayout {
  int cells=0, origin=0, old_first=0, old_count=0, new_first=0, new_count=0;
  int mesh_sequence=0;
  std::vector<chart_atlas::BlockMap> old_maps, new_maps;
  std::vector<int> old_owner, new_owner, new_to_old, old_to_new, old_level, new_level;
  std::vector<unsigned char> changed;
};

std::vector<int> ReplacementFamily(const RemeshLayout &layout, int old_gid);

// Geometry-only raw-field row; target indices are logical (without origin).
// The optional flag distinguishes extrapolated coarse overlap which needs the
// ordinary physical-space selector from a same-family tensor stencil.
std::vector<WeightedDonor> BuildBulkTransferRow(const RemeshLayout &layout,
    const NativeDonorSelector &selector,const std::vector<int> &old_family,
    int new_gid,int i,int j,int k,bool *used_global=nullptr);

KOKKOS_INLINE_FUNCTION
void PackChiRecord(const ChiRecord &r, Real out[8]) {
  out[0]=r.minimum; out[1]=r.accepted_minimum; out[2]=r.floor_count;
  out[3]=r.nonpositive_count; out[4]=r.nonfinite_count; out[5]=r.max_correction;
  out[6]=r.sum_correction; out[7]=r.first_floor;
}

KOKKOS_INLINE_FUNCTION
ChiRecord UnpackChiRecord(const Real in[8]) {
  ChiRecord r;
  r.minimum=in[0]; r.accepted_minimum=in[1]; r.floor_count=in[2];
  r.nonpositive_count=in[3]; r.nonfinite_count=in[4]; r.max_correction=in[5];
  r.sum_correction=in[6]; r.first_floor=in[7];
  return r;
}

// A history packet is folded once, in stable old-GID/k/j/i order. Counts and
// accumulated corrections are additive; extrema and first-event time are not.
KOKKOS_INLINE_FUNCTION
void MergeChiRecord(ChiRecord &target, const ChiRecord &source) {
  target.minimum=fmin(target.minimum,source.minimum);
  target.accepted_minimum=fmin(target.accepted_minimum,source.accepted_minimum);
  target.floor_count+=source.floor_count;
  target.nonpositive_count+=source.nonpositive_count;
  target.nonfinite_count+=source.nonfinite_count;
  target.max_correction=fmax(target.max_correction,source.max_correction);
  target.sum_correction+=source.sum_correction;
  target.first_floor=fmin(target.first_floor,source.first_floor);
}

class RemeshState {
 public:
  RemeshState();
  ~RemeshState();
  RemeshState(const RemeshState &)=delete;
  RemeshState &operator=(const RemeshState &)=delete;
  void Prepare(MeshBlockPack &pack, const Z4cHyperboloidal &module,
               MeshRefinement &refinement);
  void Install(Z4cHyperboloidal &module) const;
  DvceArray5D<Real> &Nominal();
  DvceArray5D<Real> &Coarse();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace z4c_hyperboloidal
#endif
