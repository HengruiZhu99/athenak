#ifndef MESH_MESH_REFINEMENT_HPP_
#define MESH_MESH_REFINEMENT_HPP_

#include <memory>
#include "mesh/amr_cadence.hpp"
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mesh_refinement.hpp
//! \brief defines MeshRefinement class containing data and functions controlling SMR/AMR

//----------------------------------------------------------------------------------------
//! \fn int CreateAMR_MPI_Tag(int lid, int ox1, int ox2, int ox3)
//! \brief calculate an MPI tag for AMR communications.  Note maximum size of
//! lid that can be encoded is set by (NUM_BITS_LID) macro.
//! The convention in Athena++ is lid is for the *receiving* process.
//! The MPI standard requires signed int tag, with MPI_TAG_UB>=2^15-1 = 32,767 (inclusive)
inline int CreateAMR_MPI_Tag(int lid, int ox1, int ox2, int ox3) {
  return (ox1<<(NUM_BITS_LID+2)) | (ox2<<(NUM_BITS_LID+1))| (ox3<<(NUM_BITS_LID)) | lid;
}

//----------------------------------------------------------------------------------------
//! \struct AMRBufferData
//! \brief container for index ranges, storage, and flags for AMR buffers used with load
//! balancing.

#if MPI_PARALLEL_ENABLED
struct AMRBufferData {
  int bis, bie, bjs, bje, bks, bke;  // start/end indices of data to be packed/unpacked
  int vbis, vbie, vbjs, vbje, vbks, vbke;  // separate native VC bounds
  int cntcc, cntvc, cntfc;   // CC, VC, and FC elements sent/recv per variable
  int cnt;                   // total number of elements stored in buffer incl all vars
  int offset=0;              // starting index of data for this buffer
  int lid;                   // local ID (gid - gids) of MeshBlock on this rank
  bool use_coarse=false;     // pack/unpack from coarse array when true
  // Receive-side native-VC metadata.  Nonnegative values identify the logical
  // child of a derefined parent; -1 denotes same-level or refinement traffic.
  int vc_derefine_child=-1;
};
#endif

// Forward declaration
class RefinementCriteria;
class AMRHistory;

//----------------------------------------------------------------------------------------
//! \class MeshRefinement
//! \brief data/functions associated with SMR/AMR

class MeshRefinement {
 public:
  MeshRefinement(Mesh *pm, ParameterInput *pin);
  ~MeshRefinement();

  // data
  int nmb_created;           // # of MeshBlocks created via AMR across all ranks
  int nmb_deleted;           // # of MeshBlocks deleted via AMR across all ranks
  int nmb_sent_thisrank;     // # of MeshBlocks sent during load balancing on this rank
  int ncyc_check_amr;        // # of cycles between checking mesh for ref/derefinement
  int refinement_interval;   // # of cycles between allowing successive ref/derefinement
  bool prolong_prims;        // flag to enable prolongation of primitive vars
  bool clean_stop_on_max_nmb_per_rank;  // retain accepted mesh and finalize before OOM
  RefinementCriteria* pmrc=nullptr;   // object to control various refinement criteria
  std::unique_ptr<AMRHistory> amr_history;  // optional deterministic hierarchy history

  // following 2x Views are dimensioned [nmb_total]
  DualArray1D<int> refine_flag;    // refinement flag for each MeshBlock
  HostArray1D<int> ncyc_since_ref; // # of cycles since MB last refined/derefined

  // following 4x arrays allocated with length [nranks] only with AMR
  int *nref_eachrank;     // number of MBs refined per rank
  int *nderef_eachrank;   // number of MBs de-refined per rank
  int *nref_rsum;         // running sum of number of MBs refined per rank
  int *nderef_rsum;       // running sum of number of MBs de-refined per rank
  // following 2x arrays allocated with length [nmb_new] and [nmb_old]] only with AMR
  int *newtoold;          // mapping of new gid (index n) to old gid
  int *oldtonew;          // mapping of old gid (index n) to new gid

  // arrays in Mesh class created for new MB heirarchy with AMR
  // following 3x arrays allocated with length [new_nmb_total]
  float *new_cost_eachmb;            // cost of each MeshBlock
  int *new_rank_eachmb;              // rank of each MeshBlock
  LogicalLocation *new_lloc_eachmb;  // LogicalLocations for each MeshBlock
  // following 2x arrays allocated with length [nranks]
  int *new_gids_eachrank;      // starting global ID of MeshBlocks in each rank
  int *new_nmb_eachrank;       // number of MeshBlocks on each rank

  // Lagrange Interpolation weights for prolongation and restriction operators
  // naming convention: {prolong/restrict}_{order of interpolation}_{optional index}
  struct InterpWeight {
    DualArray3D<Real> prolong_2nd;
    DualArray1D<Real> restrict_2nd;
    DualArray3D<Real> prolong_4th;
    DualArray1D<Real> restrict_4th_edge;
    DualArray1D<Real> restrict_4th;
  };
  InterpWeight weights;

#if MPI_PARALLEL_ENABLED
  int nmb_send, nmb_recv;
  MPI_Comm amr_comm;                           // unique communicator for AMR
  DualArray1D<AMRBufferData> sendbuf, recvbuf; // send/recv buffer metadata
  MPI_Request *send_req, *recv_req;
  DvceArray1D<Real> send_data, recv_data;      // send/recv device data
  // Immutable local-child snapshots used by the one-writer split-family VC
  // derefinement assembly.  A7 may reuse coarse-array slots for newly refined
  // blocks before A8 consumes the derefinement receives, so A8 must not read
  // local child contributions directly from coarse_u0.
  DualArray2D<int> vc_derefine_child_sources;
  DvceArray5D<Real> vc_derefine_local_data;
#endif

  // functions
  void CheckForRefinement(MeshBlockPack* pmbp);
  void AdaptiveMeshRefinement(Driver *pdrive, ParameterInput *pin);
  void UpdateMeshBlockTree(int &nnew, int &ndel);
  void RedistAndRefineMeshBlocks(ParameterInput *pin, int nnew, int ndel);

  // Default-off native-VC AMR lifecycle diagnostics.  Selection is controlled by
  // ATHENA_Z4C_VC_AMR_LIFECYCLE and therefore does not materialize a CC input/restart
  // parameter.  A completion marker is emitted only after a device fence and MPI barrier.
  bool VCAMRLifecycleDiagnosticEnabled() const;
  void VCAMRLifecycleMark(int phase, const char *name) const;
  void VCAMRLifecycleMarkState(const char *checkpoint, const char *name,
                               const DvceArray5D<Real> &state, int stage,
                               int event_cycle) const;
  void VCAMRLifecycleArmPostEvent();
  void VCAMRLifecycleMarkFirstPostEventRHS(const DvceArray5D<Real> &rhs,
                                           int stage);
  void VCAMRLifecycleMarkFirstPostEventUpdate(const DvceArray5D<Real> &state,
                                              int stage);
  void VCAMRWriterCheckpoint(const char *checkpoint,
                             const DvceArray5D<Real> &state, int stage) const;
  void ValidateVCAMRMaps(int old_nmb, int new_nmb) const;

  void DerefineCCSameRank(DvceArray5D<Real> &a, DvceArray5D<Real> &ca);
  void DerefineVCSameRank(DvceArray5D<Real> &a, DvceArray5D<Real> &ca);
  void DerefineFCSameRank(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb);

  void CopyCC(DvceArray5D<Real> &a);
  void CopyVC(DvceArray5D<Real> &a);
  void CopyFC(DvceFaceFld4D<Real> &b);

  void CopyForRefinementCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca);
  void CopyForRefinementVC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca);
  void CopyForRefinementFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb);

  void RefineCC(DualArray1D<int> &n2o, DvceArray5D<Real> &a, DvceArray5D<Real> &ca,
                bool is_z4c=false);
  void RefineVC(DualArray1D<int> &n2o, DvceArray5D<Real> &a,
                DvceArray5D<Real> &ca);
  void RefineFC(DualArray1D<int> &n2o, DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb);

  void RestrictCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca, bool is_z4c=false);
  void RestrictVC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca);
  void RestrictFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb);
  void HighOrderRestrictCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca);

  // functions for load balancing (in file load_balance.cpp)
  void InitRecvAMR(int nleaf);
  void PackAndSendAMR(int nleaf);
  void PackAMRBuffersCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca, int ncc, int nfc);
  void PackAMRBuffersVC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca,
                        int ncc, int nvc, int nfc);
  void PackAMRBuffersFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb,
                        int ncc, int nvc, int nfc);
  void ClearRecvAndUnpackAMR();
  void UnpackAMRBuffersCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca, int ncc,int nfc);
  void UnpackAMRBuffersVC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca,
                          int ncc, int nvc, int nfc);
  void UnpackAMRBuffersFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb,
                          int ncc, int nvc, int nfc);
  void ClearSendAMR();

  // initialize interpolation weights
  void InitInterpWghts();

 private:
  // data
  Mesh *pmy_mesh;
  bool vc_lifecycle_waiting_rhs = false;
  bool vc_lifecycle_waiting_update = false;
  int vc_lifecycle_event_cycle = -1;
};
#endif // MESH_MESH_REFINEMENT_HPP_
