#include "bvals/multirate_exchange.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <stdexcept>
#include <thread>
#include <vector>

namespace mt = bvals::multirate;
namespace {
void Require(bool valid, const char *message) {
  if (!valid) throw std::runtime_error(message);
}
Real Expected(int epoch, const mt::RecordKey &key, const mt::Point &point, int field) {
  return Real(epoch*10000000+key.gid*100000+key.step*1000+(key.stage+1)*100+
              point.kind*40+point.k*10+point.j*3+point.i+field);
}
std::vector<mt::Point> Points(int gid) {
  return {{gid,1,2,3,0},{gid,2,1,0,0},{gid,1,2,0,1}};
}
void Exercise(int rank, int ranks, bool reverse_ownership) {
  constexpr int fields = 7;
  const int own_rank = reverse_ownership ? ranks-1-rank : rank;
  const int first_gid = own_rank, second_gid = own_rank+ranks;
  std::vector<int> owners(2*ranks), steps(2*ranks);
  for (int gid = 0; gid < 2*ranks; ++gid) {
    owners[gid] = reverse_ownership ? ranks-1-gid%ranks : gid%ranks;
    steps[gid] = 1 << (gid%3);
  }
  auto owned = Points(first_gid);
  const auto second = Points(second_gid);
  owned.insert(owned.end(),second.begin(),second.end());
  // Ranks >=2 are deliberately zero-peer ranks. They still participate in
  // setup and publish/release their own records, without waiting on peers.
  const int peer_rank = ranks>1 && rank<2 ? 1-rank : rank;
  const int peer_gid = reverse_ownership ? ranks-1-peer_rank : peer_rank;
  auto requested = std::vector<mt::Point>{Points(peer_gid)[2],Points(peer_gid)[0],Points(peer_gid)[2]};
  if (rank >= 2) requested.clear();
  mt::RecordExchange exchange(owners,owned,requested,fields);
  Require(exchange.OwnedPoints(first_gid).size() == 3,"owned record layout missing points");
  for (int epoch = 1; epoch <= 2; ++epoch) {
    std::map<mt::RecordKey,std::size_t> consumers;
    consumers[{first_gid,0,-1}] = 2;
    consumers[{first_gid,0,3}] = 1;
    consumers[{first_gid,static_cast<std::uint32_t>(steps[first_gid]),-1}] = 1;
    if (!requested.empty()) {
      consumers[{peer_gid,0,-1}] += 2;
      consumers[{peer_gid,0,3}] += 1;
      consumers[{peer_gid,static_cast<std::uint32_t>(steps[peer_gid]),-1}] += 1;
    }
    if (reverse_ownership && epoch==1) exchange.CompileReceives(consumers);
    exchange.Begin(epoch,steps,consumers);
    if (reverse_ownership) {
      const std::size_t wanted=(!requested.empty() && owners[peer_gid]!=rank)?3:0;
      Require(exchange.ExpectedReceives()==wanted,"planned transport sends unused stage records");
    }
    auto publish = [&](const mt::RecordKey &key) {
      const auto &points = exchange.OwnedPoints(key.gid);
      DvceArray1D<Real> values("multirate test local immutable record",points.size()*fields);
      auto host = Kokkos::create_mirror_view(values);
      for (std::size_t point = 0; point < points.size(); ++point)
        for (int field = 0; field < fields; ++field)
          host(point*fields+field) = Expected(epoch,key,points[point],field);
      Kokkos::deep_copy(values,host);
      exchange.Publish(key,values);
      exchange.Poll();
    };
    // Publish in reverse time order to expose an accidental ordering assumption.
    for (int gid : {second_gid,first_gid}) {
      for (int step = steps[gid]-1; step >= 0; --step) {
        publish({gid,static_cast<std::uint32_t>(step+1),-1});
        for (int stage = 3; stage >= 0; --stage)
          publish({gid,static_cast<std::uint32_t>(step),stage});
      }
      publish({gid,0,-1});
    }
    // Let another rank start its next epoch while this one still drains the
    // previous epoch. Header validation must not consume a future packet.
    if (rank == 1 && epoch == 1) std::this_thread::sleep_for(std::chrono::milliseconds(25));
    const auto start = std::chrono::steady_clock::now();
    while (!exchange.Finish())
      Require(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<10,
              "multirate sparse record drain timed out");
    Require(exchange.CompletedReceives() == exchange.ExpectedReceives(),"wrong receive record count");
    for (const auto &entry : consumers) {
      const auto &key = entry.first;
      Require(exchange.Has(key),"requested record was discarded");
      const auto data = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),exchange.Data(key));
      const auto points = owners[key.gid] == rank ? exchange.OwnedPoints(key.gid) :
          std::vector<mt::Point>{Points(key.gid)[0],Points(key.gid)[2]};
      for (const auto &point : points) {
        const auto index = exchange.PointIndex(point);
        for (int field = 0; field < fields; ++field)
          Require(data(index*fields+field) == Expected(epoch,key,point,field),"sparse payload mismatch");
      }
      for (std::size_t use = 0; use < entry.second; ++use) {
        Require(exchange.Has(key),"record lifetime retired too early");
        exchange.Release(key);
      }
      Require(!exchange.Has(key),"record lifetime did not retire");
    }
    Require(exchange.CachedRecords() == 0,"unused arrivals remained cached");
  }
}
// Every owner publishes five records of each requested kind in one batch.
// Initial and final records fan out to both peers on three ranks. The slope
// has a narrower destination set and each peer selects different sparse rows.
void ExerciseBatches(int rank,int ranks,bool grouped) {
  constexpr int fields=7, per_rank=5;
  std::vector<int> owners(per_rank*ranks),steps(owners.size(),1);
  std::vector<mt::Point> owned,requested;
  std::map<mt::RecordKey,std::size_t> consumers;
  for (int gid=0; gid<static_cast<int>(owners.size()); ++gid) {
    owners[gid]=gid%ranks;
    const auto points=Points(gid);
    if (owners[gid]==rank) {
      owned.insert(owned.end(),points.begin(),points.end());
    } else {
      requested.push_back(points[0]);
      if (rank%2==0) requested.push_back(points[2]);
      consumers[{gid,0,-1}]=1;
      consumers[{gid,1,-1}]=1;
      if ((owners[gid]+1)%ranks==rank) consumers[{gid,0,2}]=1;
    }
  }
  mt::RecordExchange exchange(owners,owned,requested,fields);
  exchange.SetSendCacheBytes(4096);
  exchange.SetReceiveCacheBytes(4096);
  DvceArray1D<Real> held_record;
  Real held_first=0;
  exchange.SetSlopeFields(fields-2);
  exchange.CompileReceives(consumers,grouped ? std::vector<int>(owners.size(),0) : std::vector<int>{});
  for (int epoch=1; epoch<=2; ++epoch) {
    exchange.Begin(epoch,steps,consumers,true);
    std::vector<mt::RecordExchange::Publication> batch;
    for (int gid=rank;gid<static_cast<int>(owners.size());gid+=ranks) {
      for (const mt::RecordKey key : {mt::RecordKey{gid,1,-1}, {gid,0,2}, {gid,0,-1},
                                      {gid,0,0}, {gid,0,1}, {gid,0,3}}) {
        const auto &points=exchange.OwnedPoints(gid);
        const int width=key.stage<0 ? fields : fields-2;
        DvceArray1D<Real> values("batched test record",points.size()*width);
        auto host=Kokkos::create_mirror_view(values);
        for (std::size_t q=0; q<points.size(); ++q)
          for (int f=0; f<width; ++f) host(q*width+f)=Expected(epoch,key,points[q],f);
        Kokkos::deep_copy(values,host);
        batch.emplace_back(key,values);
      }
    }
    exchange.PublishBatch(batch);
    batch.clear();  // No local consumer owns these input arrays.
    const auto queued=exchange.Stats();
    std::uint64_t wire=0,packets=0;
    for (int peer=0; peer<ranks; ++peer) {
      std::size_t count=peer==rank ? 0 : per_rank*(2+(peer==(rank+1)%ranks));
      std::size_t scalars=count*fields;
      if (peer!=rank && peer==(rank+1)%ranks) scalars-=2*per_rank;
      std::size_t bytes=scalars*(1+(peer%2==0))*sizeof(Real);
      Require(queued.sent_bytes_by_peer.at(peer)==bytes,"incorrect batched peer volume");
      wire+=bytes;
      packets+=count;
    }
    Require(queued.sent_bytes==wire && queued.send_packets==(grouped ? packets/per_rank : packets),"batched wire counters disagree");
    Require(queued.peak_packet_bytes==wire+queued.received_bytes,"batched packet peak differs from allocations");
    if (epoch>1 && packets) Require(queued.reused_send_packets>0,"completed GPU send buffers were not reused");
    Require(queued.pack_kernels==packets,"one sparse gather per MPI send");
    const auto start=std::chrono::steady_clock::now();
    while (!exchange.Finish())
      Require(std::chrono::steady_clock::now()-start<std::chrono::seconds(10),
              "batched multirate records failed to drain");
    Require(exchange.CompletedReceives()==consumers.size(),"unexpected stage in batched receive plan");
    if (epoch>1 && !consumers.empty()) {
      Require(exchange.Stats().reused_receive_packets>0,"no completed receive buffers were reused");
      auto held=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),held_record);
      Require(held(0)==held_first,"receive cache overwrote a retained record from the previous epoch");
    }
    for (const auto &[key,uses] : consumers) {
      auto data=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),exchange.Data(key));
      if (epoch==1 && !held_record.data()) {
        held_record=exchange.Data(key);
        held_first=data(0);
      }
      const int width=key.stage<0 ? fields : fields-2;
      for (const auto &point : requested) if (point.gid==key.gid)
        for (int f=0; f<width; ++f)
          Require(data(exchange.PointIndex(point)*width+f)==Expected(epoch,key,point,f),
                  "batched sparse payload differs");
      exchange.Release(key);
    }
  }
}
}  // namespace

int main(int argc, char **argv) {
  int rank = 0, ranks = 1;
#if MPI_PARALLEL_ENABLED
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&ranks);
#endif
  Kokkos::initialize(argc,argv);
  int result = 0;
  try {
    Exercise(rank,ranks,false);
    Exercise(rank,ranks,true);
    ExerciseBatches(rank,ranks,false);
    ExerciseBatches(rank,ranks,true);
    if (rank == 0) std::cout << "multirate sparse records passed on " << ranks << " ranks\n";
  } catch (const std::exception &error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n'; result = 1;
#if MPI_PARALLEL_ENABLED
    MPI_Abort(MPI_COMM_WORLD,1);
#endif
  }
  Kokkos::finalize();
#if MPI_PARALLEL_ENABLED
  MPI_Finalize();
#endif
  return result;
}
