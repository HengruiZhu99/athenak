//========================================================================================
// AthenaK astrophysical plasma code
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file buffs_vc.cpp
//! \brief Native vertex-centered boundary buffer geometry.

#include "bvals.hpp"

#include "vertex_boundary_indices.hpp"

namespace {

void SetRange(MeshBufferIndcs &indices,
              const vertex_bvals::VertexIndexRange &x1,
              const vertex_bvals::VertexIndexRange &x2,
              const vertex_bvals::VertexIndexRange &x3) {
  indices.bis = x1.lower;
  indices.bie = x1.upper;
  indices.bjs = x2.lower;
  indices.bje = x2.upper;
  indices.bks = x3.lower;
  indices.bke = x3.upper;
}

int Count(const MeshBufferIndcs &indices) {
  return (indices.bie - indices.bis + 1) *
         (indices.bje - indices.bjs + 1) *
         (indices.bke - indices.bks + 1);
}

}  // namespace

MeshBoundaryValuesVC::MeshBoundaryValuesVC(
    MeshBlockPack *ppack, ParameterInput *pin,
    const VertexBoundaryLayout &vertex_layout)
    : MeshBoundaryValues(ppack, pin, false), layout(vertex_layout) {}

void MeshBoundaryValuesVC::InitSendIndices(MeshBoundaryBuffer &buf,
                                            int ox1, int ox2, int ox3,
                                            int f1, int f2) {
  if (f1 != 0 || f2 != 0) return;
  SetRange(buf.isame[0],
      vertex_bvals::VertexSendRange(layout.is, layout.ie, layout.ng, ox1, false),
      vertex_bvals::VertexSendRange(layout.js, layout.je, layout.ng, ox2,
                                    layout.collapse_x2),
      vertex_bvals::VertexSendRange(layout.ks, layout.ke, layout.ng, ox3,
                                    layout.collapse_x3));
  buf.isame_ndat = Count(buf.isame[0]);
  buf.isame_z4c = buf.isame[0];
  buf.isame_z4c_ndat = buf.isame_ndat;
  // Reserve valid storage geometry for all neighbor relations, but production methods
  // reject level-mismatched communication until native VC AMR transfer is installed.
  buf.icoar[0] = buf.isame[0];
  buf.ifine[0] = buf.isame[0];
  buf.icoar_ndat = buf.isame_ndat;
  buf.ifine_ndat = buf.isame_ndat;
}

void MeshBoundaryValuesVC::InitRecvIndices(MeshBoundaryBuffer &buf,
                                            int ox1, int ox2, int ox3,
                                            int f1, int f2) {
  if (f1 != 0 || f2 != 0) return;
  SetRange(buf.isame[0],
      vertex_bvals::VertexRecvRange(layout.is, layout.ie, layout.ng, ox1, false),
      vertex_bvals::VertexRecvRange(layout.js, layout.je, layout.ng, ox2,
                                    layout.collapse_x2),
      vertex_bvals::VertexRecvRange(layout.ks, layout.ke, layout.ng, ox3,
                                    layout.collapse_x3));
  buf.isame_ndat = Count(buf.isame[0]);
  buf.isame_z4c = buf.isame[0];
  buf.isame_z4c_ndat = buf.isame_ndat;
  buf.icoar[0] = buf.isame[0];
  buf.ifine[0] = buf.isame[0];
  buf.icoar_ndat = buf.isame_ndat;
  buf.ifine_ndat = buf.isame_ndat;
}

TaskStatus MeshBoundaryValuesVC::InitFluxRecv(const int) {
  return TaskStatus::complete;
}
