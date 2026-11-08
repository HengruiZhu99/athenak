//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mapped_exchange.cpp
//! \brief Persistent peer-packed transport of native block interpolation donors.

#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "bvals/mapped_exchange.hpp"
#include "mesh/mesh.hpp"

namespace bvals {
namespace {
int CheckedCount(std::size_t count) {
  if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("Mapped exchange exceeds MPI integer count limit");
  }
  return static_cast<int>(count);
}
#if MPI_PARALLEL_ENABLED
void CheckMPI(int status, const char *message) {
  if (status != MPI_SUCCESS) throw std::runtime_error(message);
}
#endif

template<typename T>
void CopyMetadata(const std::vector<T> &host, DvceArray1D<T> &device,
                  const char *name) {
  device = DvceArray1D<T>(std::string(name),host.size());
  auto mirror = Kokkos::create_mirror_view(device);
  for (std::size_t n=0; n<host.size(); ++n) mirror(n) = host[n];
  Kokkos::deep_copy(device,mirror);
}
}  // namespace

bool DonorPoint::operator<(const DonorPoint &other) const {
  return std::tie(gid,k,j,i) < std::tie(other.gid,other.k,other.j,other.i);
}

#if MPI_PARALLEL_ENABLED
MappedExchange::MappedExchange(MPI_Comm comm) {
  CheckMPI(MPI_Comm_dup(comm,&comm_),"Cannot duplicate mapped-boundary communicator");
  CheckMPI(MPI_Comm_rank(comm_,&rank_),"Cannot query mapped-boundary rank");
  CheckMPI(MPI_Comm_size(comm_,&nranks_),"Cannot query mapped-boundary rank count");
}
#else
MappedExchange::MappedExchange() = default;
#endif

MappedExchange::~MappedExchange() {
#if MPI_PARALLEL_ENABLED
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (!finalized && comm_ != MPI_COMM_NULL) {
    // Normal task execution clears requests through nonblocking polling. The
    // destructor only drains any still-owned requests before releasing buffers.
    if (!send_requests_.empty()) {
      MPI_Waitall(send_requests_.size(),send_requests_.data(),MPI_STATUSES_IGNORE);
    }
    if (!recv_requests_.empty()) {
      MPI_Waitall(recv_requests_.size(),recv_requests_.data(),MPI_STATUSES_IGNORE);
    }
    MPI_Comm_free(&comm_);
  }
#endif
  if (built_ && Kokkos::is_initialized() && !Kokkos::is_finalized()) {
    DevExeSpace().fence("Mapped donor buffer destruction");
  }
}

void MappedExchange::CheckCollective(bool condition, const char *message) const {
  int invalid = condition ? 0 : 1;
#if MPI_PARALLEL_ENABLED
  int any_invalid = 0;
  CheckMPI(MPI_Allreduce(&invalid,&any_invalid,1,MPI_INT,MPI_MAX,comm_),
           "Mapped-boundary validation reduction failed");
  invalid = any_invalid;
#endif
  if (invalid) throw std::invalid_argument(message);
}

bool MappedExchange::IsCurrent(const MeshBlockPack *pack) const {
  return built_ && pack != nullptr && first_gid_ == pack->gids &&
         local_count_ == pack->nmb_thispack &&
         mesh_sequence_ == pack->pmesh->GetAMRLoadBalanceUpdateSeq();
}

void MappedExchange::CheckCurrent() const {
  if (!built_ || (pack_ != nullptr && !IsCurrent(pack_))) {
    throw std::logic_error("Mapped donor plan must be rebuilt after mesh redistribution");
  }
}

void MappedExchange::Rebuild(MeshBlockPack *pack,
    const std::vector<DonorPoint> &requested, const std::vector<int> &fields) {
  CheckCollective(pack != nullptr && pack->pmesh->nmb_packs_thisrank == 1,
                   "Mapped exchange requires one native MeshBlockPack per rank");
  auto *mesh = pack->pmesh;
  std::vector<int> owners(mesh->rank_eachmb,mesh->rank_eachmb+mesh->nmb_total);
  Rebuild(pack->gids,pack->nmb_thispack,owners,mesh->GetAMRLoadBalanceUpdateSeq(),
           requested,fields);
  pack_ = pack;
}

void MappedExchange::Rebuild(int first_gid, int local_count,
    const std::vector<int> &rank_eachmb, int mesh_sequence,
    const std::vector<DonorPoint> &requested, const std::vector<int> &fields) {
  CheckCollective(!posted_,"Cannot rebuild an active mapped-boundary exchange");
  bool valid = first_gid >= 0 && local_count >= 0 &&
      static_cast<std::size_t>(first_gid)+local_count <= rank_eachmb.size() &&
      !fields.empty();
  for (std::size_t gid=0; gid<rank_eachmb.size(); ++gid) {
    const bool local = gid >= static_cast<std::size_t>(first_gid) &&
                       gid < static_cast<std::size_t>(first_gid)+local_count;
    valid = valid && rank_eachmb[gid] >= 0 && rank_eachmb[gid] < nranks_ &&
            ((rank_eachmb[gid] == rank_) == local);
  }
  std::vector<int> sorted_fields = fields;
  std::sort(sorted_fields.begin(),sorted_fields.end());
  valid = valid && !sorted_fields.empty() && sorted_fields.front() >= 0 &&
      std::adjacent_find(sorted_fields.begin(),sorted_fields.end()) == sorted_fields.end();
  for (const auto &point : requested) {
    valid = valid && point.gid >= 0 &&
        static_cast<std::size_t>(point.gid) < rank_eachmb.size() &&
        point.i >= 0 && point.j >= 0 && point.k >= 0;
  }
  CheckCollective(valid,"Invalid native ownership, fields, or donor point in mapped plan");
  const int nfield = CheckedCount(fields.size());
#if MPI_PARALLEL_ENABLED
  int min_fields=0,max_fields=0,min_blocks=0,max_blocks=0;
  const int nblocks = CheckedCount(rank_eachmb.size());
  CheckMPI(MPI_Allreduce(&nfield,&min_fields,1,MPI_INT,MPI_MIN,comm_),
           "Mapped field-count check failed");
  CheckMPI(MPI_Allreduce(&nfield,&max_fields,1,MPI_INT,MPI_MAX,comm_),
           "Mapped field-count check failed");
  CheckMPI(MPI_Allreduce(&nblocks,&min_blocks,1,MPI_INT,MPI_MIN,comm_),
           "Mapped ownership-size check failed");
  CheckMPI(MPI_Allreduce(&nblocks,&max_blocks,1,MPI_INT,MPI_MAX,comm_),
           "Mapped ownership-size check failed");
  if (min_fields != max_fields || min_blocks != max_blocks) {
    throw std::invalid_argument("Mapped plan fields/ownership differ across ranks");
  }
  std::vector<int> min_field(nfield),max_field(nfield);
  CheckMPI(MPI_Allreduce(fields.data(),min_field.data(),nfield,MPI_INT,MPI_MIN,comm_),
           "Mapped field-list check failed");
  CheckMPI(MPI_Allreduce(fields.data(),max_field.data(),nfield,MPI_INT,MPI_MAX,comm_),
           "Mapped field-list check failed");
  if (min_field != max_field) {
    throw std::invalid_argument("Mapped donor field selection differs across ranks");
  }
#endif

  if (built_) DevExeSpace().fence("Mapped donor plan rebuild");
  recv_buffer_in_use_ = false;
  pack_ = nullptr;
  first_gid_ = first_gid;
  local_count_ = local_count;
  mesh_sequence_ = mesh_sequence;
  fields_host_ = fields;
  unique_donors_.clear();
  request_ordinals_.clear();
  std::map<DonorPoint,int> ordinals;
  for (const auto &point : requested) {
    auto result = ordinals.emplace(point,CheckedCount(unique_donors_.size()));
    if (result.second) unique_donors_.push_back(point);
    request_ordinals_.push_back(result.first->second);
  }

  std::vector<LocalPoint> pack_points;
  std::vector<std::vector<int>> request_by_rank(nranks_), ordinal_by_rank(nranks_);
  for (std::size_t ordinal=0; ordinal<unique_donors_.size(); ++ordinal) {
    const auto &point = unique_donors_[ordinal];
    const int owner = rank_eachmb[point.gid];
    if (owner == rank_) {
      pack_points.push_back({point.gid-first_gid,point.k,point.j,point.i,
                             CheckedCount(ordinal)});
    } else {
      auto &request = request_by_rank[owner];
      request.insert(request.end(),{point.gid,point.k,point.j,point.i});
      ordinal_by_rank[owner].push_back(CheckedCount(ordinal));
    }
  }
  direct_count_ = CheckedCount(pack_points.size());
  std::vector<int> send_counts(nranks_),recv_counts(nranks_);
  std::vector<int> send_displacements(nranks_),recv_displacements(nranks_);
  std::vector<int> outgoing,remote_ordinals;
  for (int rank=0; rank<nranks_; ++rank) {
    send_displacements[rank] = CheckedCount(outgoing.size());
    send_counts[rank] = CheckedCount(request_by_rank[rank].size());
    outgoing.insert(outgoing.end(),request_by_rank[rank].begin(),request_by_rank[rank].end());
    remote_ordinals.insert(remote_ordinals.end(),ordinal_by_rank[rank].begin(),
                            ordinal_by_rank[rank].end());
  }
#if MPI_PARALLEL_ENABLED
  CheckMPI(MPI_Alltoall(send_counts.data(),1,MPI_INT,recv_counts.data(),1,MPI_INT,comm_),
           "Mapped donor request count exchange failed");
#endif
  std::size_t incoming_count=0;
  for (int rank=0; rank<nranks_; ++rank) {
    recv_displacements[rank] = CheckedCount(incoming_count);
    incoming_count += recv_counts[rank];
  }
  CheckedCount(incoming_count);
  std::vector<int> incoming(incoming_count);
#if MPI_PARALLEL_ENABLED
  CheckMPI(MPI_Alltoallv(outgoing.data(),send_counts.data(),send_displacements.data(),MPI_INT,
                         incoming.data(),recv_counts.data(),recv_displacements.data(),
                         MPI_INT,comm_),"Mapped donor request exchange failed");
#endif
  valid = incoming.size()%4 == 0;
  for (std::size_t n=0; n<incoming.size(); n+=4) {
    const int gid=incoming[n],k=incoming[n+1],j=incoming[n+2],i=incoming[n+3];
    valid = valid && gid >= first_gid && gid-first_gid < local_count &&
            k >= 0 && j >= 0 && i >= 0;
    pack_points.push_back({gid-first_gid,k,j,i,-1});
  }
  CheckCollective(valid,"A mapped peer requested a donor outside its owner's native pack");

  max_field_ = sorted_fields.back();
  max_k_ = max_j_ = max_i_ = -1;
  for (const auto &point : pack_points) {
    max_k_ = std::max(max_k_,point.k);
    max_j_ = std::max(max_j_,point.j);
    max_i_ = std::max(max_i_,point.i);
  }
  remote_count_ = CheckedCount(remote_ordinals.size());
  send_count_ = CheckedCount(incoming.size()/4);
  const int send_values = CheckedCount(static_cast<std::size_t>(send_count_)*nfield);
  const int recv_values = CheckedCount(static_cast<std::size_t>(remote_count_)*nfield);
  send_peers_.clear();
  recv_peers_.clear();
  for (int rank=0; rank<nranks_; ++rank) {
    if (recv_counts[rank] > 0) send_peers_.push_back(
        {rank,recv_displacements[rank]/4*nfield,recv_counts[rank]/4*nfield});
    if (send_counts[rank] > 0) recv_peers_.push_back(
        {rank,send_displacements[rank]/4*nfield,send_counts[rank]/4*nfield});
  }
  CopyMetadata(pack_points,pack_points_,"mapped donor pack points");
  CopyMetadata(remote_ordinals,recv_ordinals_,"mapped donor receive ordinals");
  CopyMetadata(fields,fields_,"mapped donor fields");
  send_buffer_ = DvceArray1D<Real>("mapped donor send buffer",send_values);
  recv_buffer_ = DvceArray1D<Real>("mapped donor receive buffer",recv_values);
  values_ = DvceArray2D<Real>("mapped donor values",unique_donors_.size(),nfield);
#if MPI_PARALLEL_ENABLED
  send_requests_.assign(send_peers_.size(),MPI_REQUEST_NULL);
  recv_requests_.assign(recv_peers_.size(),MPI_REQUEST_NULL);
#endif
  built_ = true;
}

TaskStatus MappedExchange::StartReceives() {
  CheckCurrent();
  if (posted_) throw std::logic_error("Mapped donor stage started before requests cleared");
  if (recv_buffer_in_use_) {
    DevExeSpace().fence("Mapped donor receive buffer reuse");
    recv_buffer_in_use_ = false;
  }
  posted_ = true;
  packed_ = received_ = sends_cleared_ = recvs_cleared_ = false;
#if MPI_PARALLEL_ENABLED
  for (std::size_t p=0; p<recv_peers_.size(); ++p) {
    const auto &peer = recv_peers_[p];
    CheckMPI(MPI_Irecv(recv_buffer_.data()+peer.offset,peer.count,MPI_ATHENA_REAL,
                       peer.rank,0,comm_,&recv_requests_[p]),
             "Mapped donor receive posting failed");
  }
#endif
  return TaskStatus::complete;
}

TaskStatus MappedExchange::PackAndSend(const DvceArray5D<Real> &state) {
  CheckCurrent();
  if (!posted_) throw std::logic_error("Mapped donor receives must be posted before packing");
  if (packed_) return TaskStatus::complete;
  if (state.extent_int(0) < local_count_ || state.extent_int(1) <= max_field_ ||
      state.extent_int(2) <= max_k_ || state.extent_int(3) <= max_j_ ||
      state.extent_int(4) <= max_i_) {
    throw std::invalid_argument("Mapped donor point exceeds the local state array");
  }
  auto points = pack_points_;
  auto fields = fields_;
  auto send = send_buffer_;
  auto values = values_;
  const int direct = direct_count_, nf = fields_host_.size();
  const int count = CheckedCount(points.extent(0)*nf);
  Kokkos::parallel_for("MappedDonorPack",Kokkos::RangePolicy<DevExeSpace>(0,count),
    KOKKOS_LAMBDA(int n) {
      const int p = n/nf, f = n%nf;
      const auto point = points(p);
      const Real value = state(point.block,fields(f),point.k,point.j,point.i);
      if (p < direct) values(point.ordinal,f) = value;
      else send((p-direct)*nf+f) = value;
    });
#if MPI_PARALLEL_ENABLED
  // Native boundary transport passes device buffers to GPU-aware MPI. Fence
  // only the outgoing pack dependency before handing those buffers to MPI.
  if (!send_peers_.empty()) DevExeSpace().fence("Mapped donor pack before MPI");
  for (std::size_t p=0; p<send_peers_.size(); ++p) {
    const auto &peer = send_peers_[p];
    CheckMPI(MPI_Isend(send_buffer_.data()+peer.offset,peer.count,MPI_ATHENA_REAL,
                       peer.rank,0,comm_,&send_requests_[p]),
             "Mapped donor send posting failed");
  }
#endif
  packed_ = true;
  return TaskStatus::complete;
}

TaskStatus MappedExchange::Receive() {
  CheckCurrent();
  if (!posted_ || !packed_) return TaskStatus::incomplete;
  if (received_) return TaskStatus::complete;
#if MPI_PARALLEL_ENABLED
  int complete=1;
  CheckMPI(MPI_Testall(recv_requests_.size(),recv_requests_.data(),&complete,
                       MPI_STATUSES_IGNORE),"Mapped donor receive polling failed");
  if (!complete) return TaskStatus::incomplete;
#endif
  auto received = recv_buffer_;
  auto ordinals = recv_ordinals_;
  auto values = values_;
  const int nf = fields_host_.size();
  const int count = CheckedCount(static_cast<std::size_t>(remote_count_)*nf);
  Kokkos::parallel_for("MappedDonorUnpack",Kokkos::RangePolicy<DevExeSpace>(0,count),
    KOKKOS_LAMBDA(int n) {values(ordinals(n/nf),n%nf) = received(n);});
  recv_buffer_in_use_ = remote_count_ > 0;
  received_ = true;
  return TaskStatus::complete;
}

void MappedExchange::FinishStage() {
  if (sends_cleared_ && recvs_cleared_) posted_ = false;
}

TaskStatus MappedExchange::ClearSend() {
  if (!posted_) return TaskStatus::complete;
  if (!packed_) return TaskStatus::incomplete;
#if MPI_PARALLEL_ENABLED
  int complete=1;
  CheckMPI(MPI_Testall(send_requests_.size(),send_requests_.data(),&complete,
                       MPI_STATUSES_IGNORE),"Mapped donor send completion polling failed");
  if (!complete) return TaskStatus::incomplete;
#endif
  sends_cleared_ = true;
  FinishStage();
  return TaskStatus::complete;
}

TaskStatus MappedExchange::ClearRecv() {
  if (!posted_) return TaskStatus::complete;
  if (!received_) return TaskStatus::incomplete;
  recvs_cleared_ = true;
  FinishStage();
  return TaskStatus::complete;
}

}  // namespace bvals
