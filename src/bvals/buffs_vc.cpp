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

vertex_bvals::VertexIndexRange SameSend(const int start, const int end,
                                        const int ng, const int offset,
                                        const bool collapsed) {
  return vertex_bvals::VertexSendRange(start, end, ng, offset, collapsed);
}

vertex_bvals::VertexIndexRange SameRecv(const int start, const int end,
                                        const int ng, const int offset,
                                        const bool collapsed) {
  return vertex_bvals::VertexRecvRange(start, end, ng, offset, collapsed);
}

}  // namespace

MeshBoundaryValuesVC::MeshBoundaryValuesVC(
    MeshBlockPack *ppack, ParameterInput *pin,
    const VertexBoundaryLayout &vertex_layout)
    : MeshBoundaryValues(ppack, pin, true), layout(vertex_layout) {}

void MeshBoundaryValuesVC::InitializeBuffers(const int nvar) {
  MeshBoundaryValues::InitializeBuffers(nvar);
  const int nnghbr = pmy_pack->pmb->nnghbr;
  Kokkos::realloc(prolongation_bounds, nnghbr);
  for (int n = 0; n < nnghbr; ++n) {
    prolongation_bounds.h_view(n) = recvbuf[n].iprol[0];
  }
  prolongation_bounds.template modify<HostMemSpace>();
  prolongation_bounds.template sync<DevExeSpace>();
}

void MeshBoundaryValuesVC::InitSendIndices(MeshBoundaryBuffer &buf,
                                            int ox1, int ox2, int ox3,
                                            int f1, int f2) {
  const int sx = vertex_bvals::TangentialSelector(0, ox1, ox2, ox3, f1, f2,
                                    layout.collapse_x2, layout.collapse_x3);
  const int sy = vertex_bvals::TangentialSelector(1, ox1, ox2, ox3, f1, f2,
                                    layout.collapse_x2, layout.collapse_x3);
  const int sz = vertex_bvals::TangentialSelector(2, ox1, ox2, ox3, f1, f2,
                                    layout.collapse_x2, layout.collapse_x3);
  SetRange(buf.isame[0], SameSend(layout.is, layout.ie, layout.ng, ox1, false),
      SameSend(layout.js, layout.je, layout.ng, ox2, layout.collapse_x2),
      SameSend(layout.ks, layout.ke, layout.ng, ox3, layout.collapse_x3));
  buf.isame_ndat = Count(buf.isame[0]);
  SetRange(buf.isame_z4c,
      SameSend(layout.cis, layout.cie, layout.coarse_ng, ox1, false),
      SameSend(layout.cjs, layout.cje, layout.coarse_ng, ox2,
               layout.collapse_x2),
      SameSend(layout.cks, layout.cke, layout.coarse_ng, ox3,
               layout.collapse_x3));
  buf.isame_z4c_ndat = buf.isame_ndat + Count(buf.isame_z4c);
  SetRange(buf.icoar[0],
      vertex_bvals::FineToCoarseSendRange(
          layout.cis, layout.cie, layout.coarse_ng, ox1, sx, false),
      vertex_bvals::FineToCoarseSendRange(
          layout.cjs, layout.cje, layout.coarse_ng, ox2, sy,
                       layout.collapse_x2),
      vertex_bvals::FineToCoarseSendRange(
          layout.cks, layout.cke, layout.coarse_ng, ox3, sz,
                       layout.collapse_x3));
  buf.icoar_ndat = Count(buf.icoar[0]);
  SetRange(buf.ifine[0],
      vertex_bvals::CoarseToFineSendRange(
          layout.is, layout.ie, layout.coarse_ng, ox1, sx, false),
      vertex_bvals::CoarseToFineSendRange(
          layout.js, layout.je, layout.coarse_ng, ox2, sy,
                       layout.collapse_x2),
      vertex_bvals::CoarseToFineSendRange(
          layout.ks, layout.ke, layout.coarse_ng, ox3, sz,
                       layout.collapse_x3));
  buf.ifine_ndat = Count(buf.ifine[0]);
}

void MeshBoundaryValuesVC::InitRecvIndices(MeshBoundaryBuffer &buf,
                                            int ox1, int ox2, int ox3,
                                            int f1, int f2) {
  const int sx = vertex_bvals::TangentialSelector(0, ox1, ox2, ox3, f1, f2,
                                    layout.collapse_x2, layout.collapse_x3);
  const int sy = vertex_bvals::TangentialSelector(1, ox1, ox2, ox3, f1, f2,
                                    layout.collapse_x2, layout.collapse_x3);
  const int sz = vertex_bvals::TangentialSelector(2, ox1, ox2, ox3, f1, f2,
                                    layout.collapse_x2, layout.collapse_x3);
  SetRange(buf.isame[0], SameRecv(layout.is, layout.ie, layout.ng, ox1, false),
      SameRecv(layout.js, layout.je, layout.ng, ox2, layout.collapse_x2),
      SameRecv(layout.ks, layout.ke, layout.ng, ox3, layout.collapse_x3));
  buf.isame_ndat = Count(buf.isame[0]);
  SetRange(buf.isame_z4c,
      SameRecv(layout.cis, layout.cie, layout.coarse_ng, ox1, false),
      SameRecv(layout.cjs, layout.cje, layout.coarse_ng, ox2,
               layout.collapse_x2),
      SameRecv(layout.cks, layout.cke, layout.coarse_ng, ox3,
               layout.collapse_x3));
  buf.isame_z4c_ndat = buf.isame_ndat + Count(buf.isame_z4c);
  SetRange(buf.icoar[0],
      vertex_bvals::CoarseToFineRecvRange(
          layout.cis, layout.cie, layout.coarse_ng, ox1, sx, false),
      vertex_bvals::CoarseToFineRecvRange(
          layout.cjs, layout.cje, layout.coarse_ng, ox2, sy,
                       layout.collapse_x2),
      vertex_bvals::CoarseToFineRecvRange(
          layout.cks, layout.cke, layout.coarse_ng, ox3, sz,
                       layout.collapse_x3));
  buf.icoar_ndat = Count(buf.icoar[0]);
  SetRange(buf.ifine[0],
      vertex_bvals::FineToCoarseRecvRange(
          layout.is, layout.ie, layout.ng, ox1, sx, false),
      vertex_bvals::FineToCoarseRecvRange(
          layout.js, layout.je, layout.ng, ox2, sy,
                       layout.collapse_x2),
      vertex_bvals::FineToCoarseRecvRange(
          layout.ks, layout.ke, layout.ng, ox3, sz,
                       layout.collapse_x3));
  buf.ifine_ndat = Count(buf.ifine[0]);
  buf.iprol[0] = buf.icoar[0];
}

TaskStatus MeshBoundaryValuesVC::InitFluxRecv(const int) {
  return TaskStatus::complete;
}
