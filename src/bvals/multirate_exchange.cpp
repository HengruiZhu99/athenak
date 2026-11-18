// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include "bvals/multirate_exchange.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include "mesh/mesh.hpp"

namespace bvals::multirate {
namespace {
constexpr std::size_t kHeader = 5;
[[maybe_unused]] int Count(std::size_t n) {
  if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error("multirate transport exceeds MPI count limit");
  return static_cast<int>(n);
}
bool ExactInteger(std::uint64_t value) {
  return value <= (std::uint64_t{1} << std::numeric_limits<Real>::digits);
}
std::vector<int> Owners(MeshBlockPack *pack) {
  if (pack == nullptr || pack->pmesh == nullptr)
    throw std::invalid_argument("multirate exchange requires a mesh pack");
  return {pack->pmesh->rank_eachmb, pack->pmesh->rank_eachmb+pack->pmesh->nmb_total};
}
#if MPI_PARALLEL_ENABLED
void CheckMPI(int status, const char *message) {
  if (status != MPI_SUCCESS) throw std::runtime_error(message);
}
std::vector<int> ExchangeRequests(const std::vector<std::vector<int>>& requests, MPI_Comm comm,
                                  std::vector<int>& recv_counts, std::vector<int>& recv_offsets) {
  const int ranks=Count(requests.size());
  std::vector<int> send_counts(ranks), send_offsets(ranks);
  recv_counts.resize(ranks); recv_offsets.resize(ranks);
  std::size_t send_total = 0, recv_total = 0;
  for (int peer = 0; peer < ranks; ++peer) {
    send_offsets[peer] = Count(send_total);
    send_counts[peer] = Count(requests[peer].size()); send_total += requests[peer].size();
  }
  Count(send_total);
  CheckMPI(MPI_Alltoall(send_counts.data(),1,MPI_INT,recv_counts.data(),1,MPI_INT,comm),
           "multirate request count exchange failed");
  for (int peer = 0; peer < ranks; ++peer) {
    recv_offsets[peer] = Count(recv_total); recv_total += recv_counts[peer];
  }
  Count(recv_total);
  std::vector<int> send(send_total), receive(recv_total);
  for (int peer = 0; peer < ranks; ++peer)
    std::copy(requests[peer].begin(),requests[peer].end(),send.begin()+send_offsets[peer]);
  CheckMPI(MPI_Alltoallv(send.data(),send_counts.data(),send_offsets.data(),MPI_INT,
                        receive.data(),recv_counts.data(),recv_offsets.data(),MPI_INT,comm),
           "multirate sparse request exchange failed");
  return receive;
}
#endif
}  // namespace

bool Point::operator<(const Point &other) const {
  return std::tie(gid,kind,k,j,i) < std::tie(other.gid,other.kind,other.k,other.j,other.i);
}
bool Point::operator==(const Point &other) const {
  return gid==other.gid && kind==other.kind && k==other.k && j==other.j && i==other.i;
}
bool RecordKey::operator<(const RecordKey &other) const {
  return std::tie(gid,step,stage) < std::tie(other.gid,other.step,other.stage);
}

struct RecordExchange::Impl {
  int rank = 0, ranks = 1, fields = 31, slope_fields = 31;
  int Width(const RecordKey &key) const { return key.stage<0 ? fields : slope_fields; }
  std::vector<int> owners, steps;
  std::map<int,std::vector<Point>> owned, layout;
  std::map<Point,std::size_t> point_index;
  std::map<RecordKey,std::size_t> consumers;
  std::map<RecordKey,DvceArray1D<Real>> records;
  std::set<RecordKey> published, received;
  std::uint64_t epoch = 0;
  bool active = false, finished = true, used_epoch = false;
  std::size_t expected_receives = 0, completed_receives = 0, expected_publications = 0;
  RecordExchange::Statistics stats;
  std::uint64_t record_bytes=0, packet_bytes=0;
  std::vector<RecordKey> arrivals;
  void AllocatePacket(std::size_t scalars) {
    packet_bytes+=scalars*sizeof(Real);
    stats.peak_packet_bytes=std::max(stats.peak_packet_bytes,packet_bytes);
  }
  struct Export {
    int rank = 0;
    DvceArray1D<int> indices;
  };
  std::map<int,std::vector<Export>> exports;
#if MPI_PARALLEL_ENABLED
  MPI_Comm comm = MPI_COMM_NULL;
  int tag_upper_bound = 0, epoch_tag = 0;
  using Channel = std::pair<int,int>;  // source rank and publication group
  struct ReceiveSlot {
    int tag;
    std::vector<RecordKey> keys;
    std::size_t scalars = 0;
  };
  struct Pending {
    MPI_Request request = MPI_REQUEST_NULL;
    DvceArray1D<Real> packet;
    int peer = 0, tag = 0;
    Channel channel{};
    const ReceiveSlot* receive_slot = nullptr;
  };
  std::vector<Pending> sends, receives;
  std::vector<DvceArray1D<Real>> pack_sources;
  std::map<std::size_t,std::vector<DvceArray1D<Real>>> send_cache;
  std::size_t send_cache_limit=0, cached_bytes=0;
  DvceArray1D<Real> SendBuffer(std::size_t size) {
    auto found=send_cache.find(size);
    if (found==send_cache.end())
      return DvceArray1D<Real>(Kokkos::view_alloc(Kokkos::WithoutInitializing,
          "multirate immutable send packet"),size);
    auto buffer=std::move(found->second.back());
    found->second.pop_back();
    if (found->second.empty()) send_cache.erase(found);
    cached_bytes-=size*sizeof(Real);
    ++stats.reused_send_packets;
    return buffer;
  }
  void RetireSend(DvceArray1D<Real>& buffer) {
    const auto size=buffer.extent(0), bytes=size*sizeof(Real);
    if (bytes && bytes<=send_cache_limit-cached_bytes) {
      send_cache[size].push_back(std::move(buffer));
      cached_bytes+=bytes;
    }
  }
  // Each MPI source/destination pair has a dense tag namespace. Two banks
  // distinguish adjacent epochs without copying a header back from the GPU.
  bool planned = false;
  int tag_stride = 0;
  struct Window { std::vector<ReceiveSlot> slots; std::size_t next = 0; };
  std::map<Channel,Window> windows;
  struct SendRoute { int tag, offset, packet_size; };
  std::map<std::pair<int,RecordKey>,SendRoute> send_routes;
  struct OpenPacket { Pending send; std::size_t written=0; };
  std::map<std::pair<int,int>,OpenPacket> open_packets;
  void PostNext(Channel channel) {
    auto &window=windows.at(channel);
    if (window.next==window.slots.size()) return;
    const auto& slot=window.slots[window.next++];
    Pending pending;
    pending.receive_slot=&slot; pending.channel=channel; pending.peer=channel.first;
    pending.tag=slot.tag+static_cast<int>(epoch%2)*tag_stride;
    const auto scalars=slot.scalars;
    pending.packet=DvceArray1D<Real>(Kokkos::view_alloc(Kokkos::WithoutInitializing,
        "multirate posted donor record"),scalars);
    AllocatePacket(scalars); stats.received_bytes+=scalars*sizeof(Real);
    CheckMPI(MPI_Irecv(pending.packet.data(),Count(scalars),MPI_ATHENA_REAL,pending.peer,
                       pending.tag,comm,&pending.request),"multirate preposted receive failed");
    receives.push_back(std::move(pending));
  }

#endif

  ~Impl() {
#if MPI_PARALLEL_ENABLED
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized && comm != MPI_COMM_NULL) {
      // Also covers exceptions during collective metadata construction.
      for (auto &pending : receives) if (pending.request != MPI_REQUEST_NULL) {
        MPI_Cancel(&pending.request); MPI_Wait(&pending.request,MPI_STATUS_IGNORE);
      }
      for (auto &pending : sends) if (pending.request != MPI_REQUEST_NULL) {
        MPI_Cancel(&pending.request); MPI_Wait(&pending.request,MPI_STATUS_IGNORE);
      }
      if (!pack_sources.empty() && Kokkos::is_initialized() && !Kokkos::is_finalized())
        DevExeSpace().fence("multirate unpublished pack cleanup");
      MPI_Comm_free(&comm);
    }
#endif
  }

  void CollectiveRequire(bool valid, const char *message) const {
    int invalid = valid ? 0 : 1;
#if MPI_PARALLEL_ENABLED
    int any_invalid = 0;
    CheckMPI(MPI_Allreduce(&invalid,&any_invalid,1,MPI_INT,MPI_MAX,comm),
             "multirate metadata validation failed");
    invalid = any_invalid;
#endif
    if (invalid) throw std::invalid_argument(message);
  }
  bool ValidKey(const RecordKey &key) const {
    if (key.gid < 0 || static_cast<std::size_t>(key.gid) >= steps.size()) return false;
    const auto n = static_cast<std::uint32_t>(steps[key.gid]);
    return (key.stage == -1 && key.step <= n) ||
        (key.stage >= 0 && key.stage < 4 && key.step < n);
  }
  void Accept(const RecordKey &key, const DvceArray1D<Real> &values) {
    const auto count = consumers.find(key);
    if (count != consumers.end() && count->second != 0) {
      if (!records.emplace(key,values).second)
        throw std::logic_error("duplicate multirate record cache insertion");
      arrivals.push_back(key);
      record_bytes+=values.extent(0)*sizeof(Real);
      stats.peak_record_bytes=std::max(stats.peak_record_bytes,record_bytes);
    }
  }
};

RecordExchange::RecordExchange(MeshBlockPack *pack, std::vector<Point> owned,
                               std::vector<Point> requested, int fields)
    : RecordExchange(Owners(pack),std::move(owned),std::move(requested),fields) {}

RecordExchange::RecordExchange(const std::vector<int> &owners, std::vector<Point> owned,
                               std::vector<Point> requested, int fields
#if MPI_PARALLEL_ENABLED
                               , MPI_Comm communicator
#endif
                               ) : impl_(std::make_unique<Impl>()) {
  auto &p = *impl_;
  p.owners = owners; p.fields = p.slope_fields = fields;
#if MPI_PARALLEL_ENABLED
  CheckMPI(MPI_Comm_dup(communicator,&p.comm),"cannot duplicate multirate communicator");
  CheckMPI(MPI_Comm_rank(p.comm,&p.rank),"cannot query multirate rank");
  CheckMPI(MPI_Comm_size(p.comm,&p.ranks),"cannot query multirate rank count");
  int *tag_bound = nullptr, has_tag_bound = 0;
  CheckMPI(MPI_Comm_get_attr(p.comm,MPI_TAG_UB,&tag_bound,&has_tag_bound),"cannot query multirate MPI tags");
  if (!has_tag_bound || tag_bound == nullptr || *tag_bound < 1)
    throw std::runtime_error("multirate communicator has insufficient MPI tags");
  p.tag_upper_bound = *tag_bound;
#endif
  bool valid = fields > 0;
  for (int owner : owners) valid = valid && owner >= 0 && owner < p.ranks;
  auto valid_point = [&](const Point &point) {
    return point.gid >= 0 && static_cast<std::size_t>(point.gid) < owners.size() &&
        point.k >= 0 && point.j >= 0 && point.i >= 0 && (point.kind == 0 || point.kind == 1) &&
        ExactInteger(static_cast<std::uint64_t>(point.gid));
  };
  for (const auto &point : owned)
    valid = valid && valid_point(point) && owners[point.gid] == p.rank;
  for (const auto &point : requested) valid = valid && valid_point(point);
  p.CollectiveRequire(valid,"invalid multirate transport point/owner metadata");
  std::sort(owned.begin(),owned.end());
  p.CollectiveRequire(std::adjacent_find(owned.begin(),owned.end()) == owned.end(),
                      "duplicate owned multirate point");
  std::sort(requested.begin(),requested.end());
  requested.erase(std::unique(requested.begin(),requested.end()),requested.end());
  for (const auto &point : owned) p.owned[point.gid].push_back(point);
  p.layout = p.owned;
  std::vector<std::vector<int>> requests(p.ranks);
  for (const auto &point : requested) {
    if (owners[point.gid] == p.rank) {
      valid = valid && std::binary_search(owned.begin(),owned.end(),point);
    } else {
      p.layout[point.gid].push_back(point);
      auto &buffer = requests[owners[point.gid]];
      buffer.insert(buffer.end(),{point.gid,point.k,point.j,point.i,point.kind});
    }
  }
  p.CollectiveRequire(valid,"requested multirate point is not owned/evolved");
  for (const auto &entry : p.layout)
    for (std::size_t i = 0; i < entry.second.size(); ++i) p.point_index.emplace(entry.second[i],i);
#if MPI_PARALLEL_ENABLED
  std::vector<int> recv_counts,recv_offsets;
  auto receive=ExchangeRequests(requests,p.comm,recv_counts,recv_offsets);
  for (int peer = 0; peer < p.ranks; ++peer) {
    valid = valid && recv_counts[peer]%5 == 0;
    std::map<int,std::vector<int>> indices;
    for (int j = 0; j+4 < recv_counts[peer]; j += 5) {
      const int offset = recv_offsets[peer]+j;
      Point point{receive[offset],receive[offset+1],receive[offset+2],receive[offset+3],receive[offset+4]};
      const auto found = p.point_index.find(point);
      const bool exists = valid_point(point) && owners[point.gid] == p.rank && found != p.point_index.end();
      valid = valid && exists;
      if (exists) indices[point.gid].push_back(Count(found->second));
    }
    for (const auto &entry : indices) {
      Impl::Export target;
      target.rank = peer;
      target.indices = DvceArray1D<int>("multirate sparse export indices",entry.second.size());
      auto mirror = Kokkos::create_mirror_view(target.indices);
      for (std::size_t i = 0; i < entry.second.size(); ++i) mirror(i) = entry.second[i];
      Kokkos::deep_copy(target.indices,mirror);
      p.exports[entry.first].push_back(std::move(target));
    }
  }
  p.CollectiveRequire(valid,"remote multirate request names an unowned point");
#endif
}

RecordExchange::~RecordExchange() = default;

const std::vector<Point> &RecordExchange::OwnedPoints(int gid) const {
  return impl_->owned.at(gid);
}
std::size_t RecordExchange::PointIndex(const Point &point) const { return impl_->point_index.at(point); }

void RecordExchange::SetSlopeFields(int fields) {
#if MPI_PARALLEL_ENABLED
  if (impl_->planned) throw std::logic_error("Configure slope width before compiling receive layouts");
#endif
  if (impl_->used_epoch || fields<1 || fields>impl_->fields)
    throw std::invalid_argument("invalid immutable slope record width");
  impl_->slope_fields=fields;
}

void RecordExchange::CompileReceives(const std::map<RecordKey,std::size_t> &consumers,
                                     const std::vector<int>& groups) {
#if MPI_PARALLEL_ENABLED
  auto &p=*impl_;
  if (p.used_epoch || p.planned) throw std::logic_error("compile multirate receives before Begin");
  using PacketKey = std::tuple<int,int,std::uint32_t,int>; // peer, group, step, stage
  std::map<PacketKey,std::vector<RecordKey>> packets;
  bool valid=groups.empty() || groups.size()==p.owners.size();
  for (const auto& [key,count] : consumers) if (count)
    valid=valid && p.layout.count(key.gid) && key.stage>=-1 && key.stage<4 &&
        key.step<=static_cast<std::uint32_t>(std::numeric_limits<int>::max());
  p.CollectiveRequire(valid,"invalid multirate publication group or receive layout");
  for (const auto &[key,count] : consumers) if (count && p.owners.at(key.gid)!=p.rank) {
    const int group = groups.empty() ? key.gid : groups[key.gid];
    packets[{p.owners[key.gid],group,key.step,key.stage}].push_back(key);
  }
  std::vector<std::vector<int>> requests(p.ranks);
  std::vector<int> next_tag(p.ranks);
  for (const auto& [descriptor,keys] : packets) {
    const auto [peer,group,step,stage]=descriptor;
    const int tag=next_tag[peer]++;
    auto& window=p.windows[{peer,group}];
    window.slots.push_back({tag,keys});
    auto& slot=window.slots.back();
    for (const auto& key : keys) slot.scalars+=p.layout.at(key.gid).size()*p.Width(key);
    const int packet_size=Count(slot.scalars);
    int offset=0;
    for (const auto& key : keys) {
      requests[peer].insert(requests[peer].end(),{key.gid,static_cast<int>(step),stage,
                                                  tag,offset,packet_size});
      offset+=Count(p.layout.at(key.gid).size()*p.Width(key));
    }
  }
  p.tag_stride=*std::max_element(next_tag.begin(),next_tag.end());
  CheckMPI(MPI_Allreduce(MPI_IN_PLACE,&p.tag_stride,1,MPI_INT,MPI_MAX,p.comm),
           "multirate tag capacity validation failed");
  if (2LL*p.tag_stride>1LL+p.tag_upper_bound) {p.windows.clear();return;}
  std::vector<int> counts,offsets;
  const auto received=ExchangeRequests(requests,p.comm,counts,offsets);
  for (int peer=0;peer<p.ranks;++peer) for (int j=0;j<counts[peer];j+=6) {
    const int i=offsets[peer]+j;
    RecordKey key{received[i],static_cast<std::uint32_t>(received[i+1]),received[i+2]};
    p.send_routes[{peer,key}]={received[i+3],received[i+4],received[i+5]};
  }
  p.planned=true;
#else
  (void)consumers; (void)groups;
#endif
}

void RecordExchange::Begin(std::uint64_t epoch, const std::vector<int> &steps,
                           const std::map<RecordKey,std::size_t> &consumers, bool profile_peers) {
  auto &p = *impl_;
  if (!p.finished || !p.records.empty())
    throw std::logic_error("multirate Begin requires a drained epoch and released consumers");
  if (!ExactInteger(epoch) || (p.used_epoch && epoch <= p.epoch))
    throw std::invalid_argument("multirate epoch must increase and be exactly representable in Real");
  if (steps.size() != p.owners.size()) throw std::invalid_argument("multirate step table has wrong size");
  for (int n : steps)
    if (n <= 0 || (n & (n-1)) != 0 || !ExactInteger(static_cast<std::uint64_t>(n)))
      throw std::invalid_argument("multirate steps must be exactly representable positive powers of two");
  p.steps = steps;
  for (const auto &entry : consumers)
    if (!p.ValidKey(entry.first) || p.layout.count(entry.first.gid) == 0)
      throw std::invalid_argument("multirate consumer names an unavailable record/point layout");
  p.epoch = epoch; p.used_epoch = true; p.active = true; p.finished = false;
#if MPI_PARALLEL_ENABLED
  // Isolate a neighboring rank that has reached the next common endpoint
  // first. Full headers still reject stale data if tags eventually wrap.
  p.epoch_tag = static_cast<int>(epoch%(std::uint64_t(p.tag_upper_bound)+1));
#endif
  p.stats={};
  if (profile_peers) p.stats.sent_bytes_by_peer.assign(p.ranks, 0);
  p.arrivals.clear();p.record_bytes=0;p.packet_bytes=0;
  p.consumers = consumers; p.published.clear(); p.received.clear();
  p.expected_receives = p.completed_receives = p.expected_publications = 0;
  for (const auto &entry : p.owned)
    p.expected_publications += 5*std::size_t(steps[entry.first])+1;
#if MPI_PARALLEL_ENABLED
  if (p.planned) {
    for (auto &[channel,window]:p.windows) {
      window.next=0;
      for (const auto& slot:window.slots) p.expected_receives+=slot.keys.size();
      p.PostNext(channel); p.PostNext(channel);
    }
    return;
  }
#endif
  for (const auto &entry : p.layout)
    if (p.owners[entry.first] != p.rank)
      p.expected_receives += 5*std::size_t(steps[entry.first])+1;
}

void RecordExchange::PublishBatch(const std::vector<Publication> &batch) {
  for (const auto &[key,values]:batch) Publish(key,values);
}

void RecordExchange::Publish(const RecordKey &key, const DvceArray1D<Real> &values) {
  auto &p = *impl_;
  if (!p.active || p.finished || !p.ValidKey(key) || p.owned.count(key.gid) == 0)
    throw std::invalid_argument("invalid multirate local publication");
  if (values.extent(0) != p.owned.at(key.gid).size()*p.Width(key))
    throw std::invalid_argument("multirate publication has wrong point-major payload size");
  if (!p.published.insert(key).second) throw std::logic_error("multirate record published twice");
  p.Accept(key,values);
#if MPI_PARALLEL_ENABLED
  bool retain_source=true;
  for (const auto &target : p.exports[key.gid]) {
    Impl::Pending pending;
    pending.peer = target.rank;
    pending.tag = p.epoch_tag;
    const std::size_t header = p.planned ? 0 : kHeader;
    const auto indices = target.indices;
    const int fields = p.Width(key);
    const auto size = indices.extent(0)*fields;
    std::size_t offset=0,packet_size=size+header;
    if (p.planned) {
      const auto route=p.send_routes.find({target.rank,key});
      if (route==p.send_routes.end()) continue;
      pending.tag=static_cast<int>(p.epoch%2)*p.tag_stride+route->second.tag;
      offset=route->second.offset;
      packet_size=route->second.packet_size;
    }
    Count(packet_size);
    const bool shared=packet_size!=size+header;
    const auto packet_key=std::make_pair(target.rank,pending.tag);
    Impl::OpenPacket* open=nullptr;
    if (shared) {
      open=&p.open_packets[packet_key];
      if (!open->written) {
        open->send=std::move(pending);
        open->send.packet=p.SendBuffer(packet_size);
        p.AllocatePacket(packet_size);
      }
      open->written+=size;
    } else {
      pending.packet=p.SendBuffer(packet_size);
      p.AllocatePacket(packet_size);
    }
    const auto bytes=(size+header)*sizeof(Real);
    p.stats.sent_bytes+=bytes;
    if (!p.stats.sent_bytes_by_peer.empty()) p.stats.sent_bytes_by_peer[target.rank]+=bytes;
    auto packet = shared ? open->send.packet : pending.packet;
    const Real epoch = static_cast<Real>(p.epoch), gid = key.gid, step = key.step,
               stage = key.stage, points = indices.extent(0);
    Kokkos::parallel_for("multirate sparse derivative pack",Kokkos::RangePolicy<>(0,size+header),
      KOKKOS_LAMBDA(std::size_t i) {
        if (i < header) {
          packet(i) = i == 0 ? epoch : (i == 1 ? gid : (i == 2 ? step : (i == 3 ? stage : points)));
        } else {
          const auto scalar = i-header;
          packet(offset+i) = values(std::size_t(indices(scalar/fields))*fields+scalar%fields);
        }
      });
    // Poll fences one batch of queued packs before posting their MPI sends.
    if (retain_source) {p.pack_sources.push_back(values);retain_source=false;}
    ++p.stats.pack_kernels;
    if (shared) {
      if (open->written!=packet_size) continue;
      pending=std::move(open->send);
      p.open_packets.erase(packet_key);
    }
    ++p.stats.send_packets;
    p.sends.push_back(std::move(pending));
  }
#endif
}

void RecordExchange::SetSendCacheBytes(std::size_t bytes) {
  if (!impl_->finished) throw std::logic_error("Resize send cache only at synchronized endpoints");
#if MPI_PARALLEL_ENABLED
  impl_->send_cache.clear();
  impl_->cached_bytes=0;
  impl_->send_cache_limit=bytes;
#else
  (void)bytes;
#endif
}

bool RecordExchange::Has(const RecordKey &key) const { return impl_->records.count(key) != 0; }
DvceArray1D<Real> RecordExchange::Data(const RecordKey &key) const { return impl_->records.at(key); }

bool RecordExchange::Poll() {
  auto &p = *impl_;
  if (!p.active) throw std::logic_error("multirate Poll requires Begin");
  bool progressed = false;
#if MPI_PARALLEL_ENABLED
  const bool queued = std::any_of(p.sends.begin(),p.sends.end(),
      [](const Impl::Pending &item) { return item.request == MPI_REQUEST_NULL; });
  if (queued) {
    DevExeSpace().fence("multirate batched publication packs before MPI");
    p.pack_sources.clear();
    for (auto &pending : p.sends) if (pending.request == MPI_REQUEST_NULL) {
      CheckMPI(MPI_Isend(pending.packet.data(),Count(pending.packet.extent(0)),MPI_ATHENA_REAL,
                         pending.peer,pending.tag,p.comm,&pending.request),"multirate record send failed");
    }
    progressed = true;
  }
  for (std::size_t i = 0; i < p.sends.size();) {
    int done = 0;
    CheckMPI(MPI_Test(&p.sends[i].request,&done,MPI_STATUS_IGNORE),"multirate send poll failed");
    if (done) {
      p.packet_bytes-=p.sends[i].packet.extent(0)*sizeof(Real);
      p.RetireSend(p.sends[i].packet);
      p.sends.erase(p.sends.begin()+i);
      progressed=true;
    } else ++i;
  }
  // Bound one call's admission work so ready GPU tasks can continue to run.
  for (int admitted = 0; !p.planned && admitted < 32; ++admitted) {
    int available = 0;
    MPI_Status status;
    CheckMPI(MPI_Iprobe(MPI_ANY_SOURCE,p.epoch_tag,p.comm,&available,&status),"multirate probe failed");
    if (!available) break;
    int count = 0;
    CheckMPI(MPI_Get_count(&status,MPI_ATHENA_REAL,&count),"multirate packet count failed");
    if (count < static_cast<int>(kHeader)) throw std::runtime_error("truncated multirate packet");
    Impl::Pending pending;
    pending.peer = status.MPI_SOURCE;
    pending.packet = DvceArray1D<Real>(Kokkos::view_alloc(Kokkos::WithoutInitializing,"multirate incoming record"),count);
    p.AllocatePacket(count);p.stats.received_bytes+=std::uint64_t(count)*sizeof(Real);
    CheckMPI(MPI_Irecv(pending.packet.data(),count,MPI_ATHENA_REAL,pending.peer,p.epoch_tag,p.comm,&pending.request),
             "multirate record receive failed");
    p.receives.push_back(std::move(pending)); progressed = true;
  }
  for (std::size_t i = 0; i < p.receives.size();) {
    auto &incoming = p.receives[i];
    int done = 0;
    CheckMPI(MPI_Test(&incoming.request,&done,MPI_STATUS_IGNORE),"multirate receive poll failed");
    if (!done) { ++i; continue; }
    if (p.planned) {
      const auto channel=incoming.channel;
      const auto& slot=*incoming.receive_slot;
      std::size_t offset=0;
      for (const auto& key : slot.keys) {
        const std::size_t size=p.layout.at(key.gid).size()*p.Width(key);
        if (size==slot.scalars) p.Accept(key,incoming.packet);
        else p.Accept(key,Kokkos::subview(incoming.packet,std::make_pair(offset,offset+size)));
        offset+=size;
      }
      p.completed_receives+=slot.keys.size();
      p.packet_bytes-=slot.scalars*sizeof(Real);
      p.receives.erase(p.receives.begin()+i);
      p.PostNext(channel);
      progressed=true;
      continue;
    }
    auto header = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
        Kokkos::subview(incoming.packet,std::make_pair(std::size_t(0),kHeader)));
    for (std::size_t h = 0; h < kHeader; ++h)
      if (!std::isfinite(header(h)) || std::floor(header(h)) != header(h))
        throw std::runtime_error("nonintegral multirate packet header");
    if (header(0) != static_cast<Real>(p.epoch) || header(1) < 0 ||
        header(1) >= static_cast<Real>(p.owners.size()) || header(2) < 0 ||
        header(2) > std::numeric_limits<std::uint32_t>::max() || header(3) < -1 || header(3) > 3)
      throw std::runtime_error("stale or invalid multirate packet header");
    RecordKey key{static_cast<int>(header(1)),static_cast<std::uint32_t>(header(2)),static_cast<int>(header(3))};
    const auto layout = p.layout.find(key.gid);
    if (!p.ValidKey(key) || layout == p.layout.end() || p.owners[key.gid] != incoming.peer ||
        incoming.peer == p.rank || header(4) != static_cast<Real>(layout->second.size()) ||
        incoming.packet.extent(0) != kHeader+layout->second.size()*p.Width(key) || !p.received.insert(key).second)
      throw std::runtime_error("unexpected or duplicate multirate record");
    DvceArray1D<Real> payload = Kokkos::subview(incoming.packet,
        std::make_pair(kHeader,incoming.packet.extent(0)));
    p.Accept(key,payload); ++p.completed_receives;
    p.packet_bytes-=incoming.packet.extent(0)*sizeof(Real);
    p.receives.erase(p.receives.begin()+i); progressed = true;
  }
#endif
  return progressed;
}

void RecordExchange::Release(const RecordKey &key) {
  auto &p = *impl_;
  auto count = p.consumers.find(key);
  if (count == p.consumers.end() || count->second == 0 || p.records.count(key) == 0)
    throw std::logic_error("multirate record released without a live consumer");
  if (--count->second == 0) {
    p.record_bytes-=p.records.at(key).extent(0)*sizeof(Real);
    p.records.erase(key);
  }
}

bool RecordExchange::Finish() {
  auto &p = *impl_;
  if (p.finished) return true;
  Poll();
  if (p.published.size() != p.expected_publications)
    throw std::logic_error("multirate Finish requires all local states and RHS records published");
#if MPI_PARALLEL_ENABLED
  if (!p.open_packets.empty()) throw std::logic_error("incomplete multirate publication group at endpoint");
#endif
  if (p.completed_receives != p.expected_receives) return false;
#if MPI_PARALLEL_ENABLED
  if (!p.receives.empty() || !p.sends.empty()) return false;
#endif
  p.finished = true;
  return true;
}
std::size_t RecordExchange::ExpectedReceives() const { return impl_->expected_receives; }
std::size_t RecordExchange::CompletedReceives() const { return impl_->completed_receives; }
std::vector<RecordKey> RecordExchange::TakeArrivals() {
  std::vector<RecordKey> arrivals;
  arrivals.swap(impl_->arrivals);
  return arrivals;
}
RecordExchange::Statistics RecordExchange::Stats() const {return impl_->stats;}
std::size_t RecordExchange::CachedRecords() const { return impl_->records.size(); }
}  // namespace bvals::multirate
