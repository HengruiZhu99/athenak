// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_SURFACE_REMESH_HPP_
#define Z4C_HYPERBOLOIDAL_SURFACE_REMESH_HPP_

#include <vector>
#include "bvals/mapped_exchange.hpp"
#include "z4c_hyperboloidal/remesh.hpp"

namespace z4c_hyperboloidal {

struct SurfaceTransferTerm {bvals::DonorPoint donor; Real weight;};
struct SurfaceTransferRow {
  int new_gid, face, k, j;
  bool copy;
  std::vector<SurfaceTransferTerm> fields;
  // Each old history packet appears once globally, in old-GID/k/j order.
  std::vector<bvals::DonorPoint> history;
};

// Geometry/lineage metadata only; independently testable without any field data.
std::vector<SurfaceTransferRow> BuildSurfaceTransferRows(const RemeshLayout &layout);

class SurfaceRemesh {
 public:
  void Prepare(const Z4cHyperboloidal &old_module, const RemeshLayout &layout);
  void Install(Z4cHyperboloidal &new_module) const;
  const DvceArray4D<Real> &PreparedState() const {return state_;}
  const DvceArray1D<ChiRecord> &PreparedRecords() const {return records_;}
  int RemoteDonors() const {return remote_donors_;}

 private:
  DvceArray4D<Real> state_;
  DvceArray1D<ChiRecord> records_;
  std::vector<int> face_blocks_;
  int cells_=0, blocks_=0, remote_donors_=0;
  bool ready_=false;
};
}  // namespace z4c_hyperboloidal
#endif  // Z4C_HYPERBOLOIDAL_SURFACE_REMESH_HPP_
