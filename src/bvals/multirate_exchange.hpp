// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef BVALS_MULTIRATE_EXCHANGE_HPP_
#define BVALS_MULTIRATE_EXCHANGE_HPP_

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "athena.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

class MeshBlockPack;
namespace bvals::multirate {
struct Point {
  int gid = -1, k = 0, j = 0, i = 0, kind = 0;  // kind: 0 bulk, 1 edge
  bool operator<(const Point &other) const;
  bool operator==(const Point &other) const;
};
struct RecordKey {
  int gid = -1;
  std::uint32_t step = 0;
  int stage = -1;  // -1 accepted state; 0..3 RHS
  bool operator<(const RecordKey &other) const;
};

// Sparse, immutable derivative/accepted-state records. Metadata setup is
// collective; Begin/Publish/Poll/Release/Finish contain no MPI collectives.
// Publish queues GPU packing; Poll fences and posts all queued packets as one
// batch. Call Poll regularly while numerical tasks are in flight.
// Publish's point-major values must follow OwnedPoints(gid), with fields
// contiguous within each point. Local publications are retained by View alias:
// do not overwrite their allocation while Has(key) remains true. The exchange
// also retains source Views until their asynchronous device gathers complete.
class RecordExchange {
 public:
  RecordExchange(MeshBlockPack *pack, std::vector<Point> owned,
                 std::vector<Point> requested, int fields = 31);
  RecordExchange(const std::vector<int> &owner_by_gid, std::vector<Point> owned,
                 std::vector<Point> requested, int fields = 31
#if MPI_PARALLEL_ENABLED
                 , MPI_Comm communicator = MPI_COMM_WORLD
#endif
                 );
  ~RecordExchange();
  RecordExchange(const RecordExchange &) = delete;
  RecordExchange &operator=(const RecordExchange &) = delete;
  const std::vector<Point> &OwnedPoints(int gid) const;
  // Prepare before the first epoch. A held view keeps its record immutable.
  void PreparePublicationBuffers();
  DvceArray1D<Real> PublicationBuffer(const RecordKey& key);
  // Local records contain all owned points; remote records contain this rank's
  // sorted sparse request set for the GID. PointIndex refers to that layout.
  std::size_t PointIndex(const Point &point) const;
  // Collective once per placement. Each peer requests only records read by its stages.
  // Accepted states retain their original width; slopes may encode fewer fields.
  void SetSlopeFields(int fields);
  // Blocks with the same owner and publication group produce a given step/stage
  // together. An empty grouping keeps individual record packets.
  void CompileReceives(const std::map<RecordKey,std::size_t> &consumer_counts,
                       const std::vector<int>& publication_groups = {});
  void Begin(std::uint64_t epoch, const std::vector<int> &steps_by_gid,
             const std::map<RecordKey, std::size_t> &consumer_counts, bool profile_peers=false);
  void Publish(const RecordKey &key, const DvceArray1D<Real> &values);
  using Publication = std::pair<RecordKey,DvceArray1D<Real>>;
  // Pack into individually owned GPU-aware MPI buffers for the destination peers.
  void PublishBatch(const std::vector<Publication> &batch);
  // Reuse completed sends of the same length up to a rank-local byte budget.
  void SetSendCacheBytes(std::size_t bytes);
  // Reuse posted-receive allocations only after MPI and all record aliases release them.
  void SetReceiveCacheBytes(std::size_t bytes);
  bool Has(const RecordKey &key) const;
  DvceArray1D<Real> Data(const RecordKey &key) const;
  bool Poll();  // true if transport made progress
  void Release(const RecordKey &key);
  bool Finish();  // nonblocking drain; call until true after all publications
  std::size_t ExpectedReceives() const;
  std::size_t CompletedReceives() const;
  std::size_t CachedRecords() const;
  struct Statistics {
    std::uint64_t peak_record_bytes=0, peak_packet_bytes=0;
    std::uint64_t sent_bytes=0, received_bytes=0, send_packets=0, pack_kernels=0;
    std::uint64_t reused_send_packets=0, reused_receive_packets=0;
    std::vector<std::uint64_t> sent_bytes_by_peer;
  };
  Statistics Stats() const;
  // Notifications identify newly retained records. Payload ownership remains
  // with the consumer counts registered by Begin.
  std::vector<RecordKey> TakeArrivals();
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace bvals::multirate
#endif
