#ifndef BVALS_MAPPED_EXCHANGE_HPP_
#define BVALS_MAPPED_EXCHANGE_HPP_
//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mapped_exchange.hpp
//! \brief Native block donor transport for mapped interpolation boundaries.

#include <vector>

#include "athena.hpp"
#include "tasklist/task_list.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

class MeshBlockPack;

namespace bvals {

// Native global block identity and absolute indices in the module's local
// (block,field,k,j,i) array. Field selection is shared by all donors in a plan.
struct DonorPoint {
  int gid, k, j, i;
  bool operator<(const DonorPoint &other) const;
};

class MappedExchange {
 public:
  // Construction duplicates the communicator collectively on every rank.
#if MPI_PARALLEL_ENABLED
  explicit MappedExchange(MPI_Comm comm = MPI_COMM_WORLD);
#else
  MappedExchange();
#endif
  ~MappedExchange();
  MappedExchange(const MappedExchange &) = delete;
  MappedExchange &operator=(const MappedExchange &) = delete;

  // Collective only when the topology/donor set changes. Every rank must call,
  // including ranks with no donors. One native pack per rank is currently used.
  void Rebuild(MeshBlockPack *pack, const std::vector<DonorPoint> &requested,
               const std::vector<int> &fields);
  void Rebuild(int first_gid, int local_count, const std::vector<int> &rank_eachmb,
               int mesh_sequence, const std::vector<DonorPoint> &requested,
               const std::vector<int> &fields);
  bool IsCurrent(const MeshBlockPack *pack) const;

  // Persistent stage tasks: no global collectives and no blocking MPI waits.
  // Post receives before packing; Receive polls and then scatters remote values.
  // Clear both sends and receives before reusing or rebuilding the plan.
  TaskStatus StartReceives();
  TaskStatus PackAndSend(const DvceArray5D<Real> &state);
  TaskStatus Receive();
  TaskStatus ClearSend();
  TaskStatus ClearRecv();

  using ConstValues = Kokkos::View<const Real **, LayoutWrapper, DevMemSpace>;
  ConstValues Values() const {return values_;}
  const std::vector<DonorPoint> &UniqueDonors() const {return unique_donors_;}
  const std::vector<int> &RequestOrdinals() const {return request_ordinals_;}
  const std::vector<int> &Fields() const {return fields_host_;}
  int MeshSequence() const {return mesh_sequence_;}
  int RemoteDonors() const {return remote_count_;}
  int SendDonors() const {return send_count_;}
  int ReceivePeers() const {return recv_peers_.size();}
  int SendPeers() const {return send_peers_.size();}

 private:
  struct LocalPoint {int block, k, j, i, ordinal;};
  struct Peer {int rank, offset, count;};  // scalar Real payload offsets/counts

  MeshBlockPack *pack_ = nullptr;
  int rank_ = 0, nranks_ = 1, first_gid_ = 0, local_count_ = 0;
  int mesh_sequence_ = -1, remote_count_ = 0, send_count_ = 0;
  int max_field_ = -1, max_k_ = -1, max_j_ = -1, max_i_ = -1;
  bool built_ = false, posted_ = false, packed_ = false, received_ = false;
  bool sends_cleared_ = false, recvs_cleared_ = false;
  bool recv_buffer_in_use_ = false;
  std::vector<DonorPoint> unique_donors_;
  std::vector<int> request_ordinals_, fields_host_;
  std::vector<Peer> send_peers_, recv_peers_;
  DvceArray1D<LocalPoint> pack_points_;
  DvceArray1D<int> fields_, recv_ordinals_;
  DvceArray1D<Real> send_buffer_, recv_buffer_;
  DvceArray2D<Real> values_;
  int direct_count_ = 0;
#if MPI_PARALLEL_ENABLED
  MPI_Comm comm_ = MPI_COMM_NULL;
  std::vector<MPI_Request> send_requests_, recv_requests_;
#endif

  void CheckCurrent() const;
  void FinishStage();
  void CheckCollective(bool condition, const char *message) const;
};

}  // namespace bvals
#endif  // BVALS_MAPPED_EXCHANGE_HPP_
