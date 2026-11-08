// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Test-only plan metadata. No physics allocation, field fill, or evolution.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "mesh/nghbr_index.hpp"
#include "bvals/bvals.hpp"
#include "z4c_hyperboloidal/mapped_halo.hpp"
#include "z4c_hyperboloidal/fields.hpp"

namespace native=z4c_hyperboloidal;
using Count=std::uint64_t;
using UseKey=std::array<std::int32_t,5>;  // target GID, donor GID,k,j,i
enum Reason {eligible,not_single,nonunit,different_chart,different_layer,different_level,
  different_offset,not_adjacent,missing_neighbor,neighbor_metadata,target_range,
  donor_range,index_translation,nreason};
const char *names[nreason]={"eligible","not_single","nonunit_weight","different_chart",
  "different_layer","different_level","different_node_offset","not_adjacent",
  "missing_native_neighbor","native_neighbor_metadata","outside_native_receive_box",
  "outside_native_send_box","index_translation"};
void Require(bool ok,const char *why) {if(!ok)throw std::runtime_error(why);}
bool Contains(const MeshBufferIndcs &b,const std::array<int,3>&p) {
  return p[0]>=b.bis&&p[0]<=b.bie&&p[1]>=b.bjs&&p[1]<=b.bje&&p[2]>=b.bks&&p[2]<=b.bke;
}
Count Volume(const MeshBufferIndcs &b) {
  return Count(b.bie-b.bis+1)*(b.bje-b.bjs+1)*(b.bke-b.bks+1);
}
struct BlockCounts {
  std::array<Count,nreason> reason{};
  Count rows=0,entries=0,single_nonunit_near_one=0,exterior=0;
};
struct Route {
  int target,donor,neighbor;
  Count eligible_rows=0,plan_rows=0,evolved=0,slab=0;
};

void Capture(Mesh &mesh,ParameterInput &pin) {
  const int rank=global_variable::my_rank,ranks=global_variable::nranks;
  auto *pack=mesh.pmb_pack;
  const int n=mesh.mb_indcs.nx1,ng=mesh.mb_indcs.ng;
  const int overlap=pin.GetInteger("mesh_atlas","evolved_overlap"),origin=ng+overlap;
  Require(n==12&&ng==4&&overlap==3&&mesh.nmb_total==106,"requires fixed N12 topology");
  native::NativeDonorSelector selector(mesh,origin);
  native::MappedHalo halo;halo.Rebuild(pack,origin);
  // Constructor allocates request metadata only. Do not InitializeBuffers:
  // these public methods populate lightweight native same-level index ranges.
  MeshBoundaryValuesCC cc(pack,&pin,false);
  std::array<MeshBoundaryBuffer,56> receives,sends;
  std::array<std::array<int,3>,56> offsets{};
  std::set<int> indices;
  for(int z=-1;z<=1;++z)for(int y=-1;y<=1;++y)for(int x=-1;x<=1;++x) {
    if(x==0&&y==0&&z==0)continue;
    const int index=NeighborIndex(x,y,z,0,0);indices.insert(index);offsets[index]={x,y,z};
    cc.InitRecvIndices(receives[index],x,y,z,0,0);
    cc.InitSendIndices(sends[index],-x,-y,-z,0,0);
  }
  const std::array<int,3> start={mesh.mb_indcs.is,mesh.mb_indcs.js,mesh.mb_indcs.ks};
  auto native_target=[&](native::HaloPoint c) {
    return std::array<int,3>{c.i-origin+start[0],c.j-origin+start[1],c.k-origin+start[2]};
  };
  std::map<int,BlockCounts> blocks;
  std::map<UseKey,int> uses;
  std::map<std::pair<int,int>,Route> routes;
  for(int m=0;m<pack->nmb_thispack;++m)blocks[pack->gids+m]=BlockCounts{};
  for(const auto &row:halo.ExteriorTargets())++blocks.at(pack->gids+row.target.m).exterior;
  for(const auto &row:halo.Rows()) {
    const int gid=pack->gids+row.target.m;auto &counts=blocks.at(gid);
    ++counts.rows;counts.entries+=row.donors.size();
    for(const auto &d:row.donors)Require(std::isfinite(d.weight),"nonfinite plan weight");
    Reason why=eligible;int index=-1;
    if(row.donors.size()!=1)why=not_single;
    else {
      const auto &d=row.donors.front();const auto &a=selector.Map(gid),&b=selector.Map(d.point.gid);
      const auto &la=mesh.lloc_eachmb[gid],&lb=mesh.lloc_eachmb[d.point.gid];
      if(d.weight!=Real(1)) {
        why=nonunit;
        if(std::abs(d.weight-1)<=1e-12)++counts.single_nonunit_near_one;
      } else if(a.chart_id!=b.chart_id)why=different_chart;
      else if(a.layer!=b.layer)why=different_layer;
      else if(la.level!=lb.level)why=different_level;
      else if(a.node_offset[0]!=b.node_offset[0]||a.node_offset[1]!=b.node_offset[1]
          ||a.node_offset[2]!=b.node_offset[2])why=different_offset;
      else {
        const std::array<int,3> delta={lb.lx1-la.lx1,lb.lx2-la.lx2,lb.lx3-la.lx3};
        if(std::abs(delta[0])>1||std::abs(delta[1])>1||std::abs(delta[2])>1
            ||(delta[0]==0&&delta[1]==0&&delta[2]==0))why=not_adjacent;
        else {
          index=NeighborIndex(delta[0],delta[1],delta[2],0,0);
          const auto &nb=pack->pmb->nghbr.h_view(row.target.m,index);
          if(nb.gid!=d.point.gid)why=missing_neighbor;
          else if(nb.lev!=lb.level||nb.rank!=mesh.rank_eachmb[d.point.gid]
              ||nb.dest!=NeighborIndex(-delta[0],-delta[1],-delta[2],0,0))why=neighbor_metadata;
          else {
            const auto t=native_target(row.target);
            const std::array<int,3> p={d.point.i-origin+start[0],
              d.point.j-origin+start[1],d.point.k-origin+start[2]};
            if(!Contains(receives[index].isame[0],t))why=target_range;
            else if(!Contains(sends[index].isame[0],p))why=donor_range;
            else for(int a=0;a<3;++a)if(p[a]!=t[a]-delta[a]*n)why=index_translation;
          }
        }
      }
    }
    ++counts.reason[why];
    for(const auto &d:row.donors)
      uses[{gid,d.point.gid,d.point.k,d.point.j,d.point.i}]|=why==eligible?1:2;
    if(why==eligible) {
      const auto key=std::make_pair(gid,index);
      if(routes.count(key)==0)routes.emplace(key,Route{gid,row.donors[0].point.gid,index});
      ++routes.at(key).eligible_rows;
    }
  }
  // Count complete ordinary slabs separately: they can include targets that
  // were not eligible, were never requested, or are independently evolved.
  for(const auto &row:halo.Rows()) {
    const int gid=pack->gids+row.target.m;const auto t=native_target(row.target);
    for(int index:indices) {
      auto route=routes.find({gid,index});
      if(route!=routes.end()&&Contains(receives[index].isame[0],t))++route->second.plan_rows;
    }
  }
  for(auto &[key,r]:routes) {
    const auto &b=receives[r.neighbor].isame[0];r.slab=Volume(b);
    const int low[3]={b.bis,b.bjs,b.bks},high[3]={b.bie,b.bje,b.bke};
    r.evolved=1;
    for(int a=0;a<3;++a)r.evolved*=std::max(0,std::min(high[a],start[a]+selector.Upper(r.target,a)-1)
        -std::max(low[a],start[a]+selector.Lower(r.target,a))+1);
    Require(r.plan_rows>=r.eligible_rows&&r.slab>=r.plan_rows+r.evolved,"slab accounting error");
  }
  std::set<bvals::DonorPoint> unique;
  for(const auto &[key,flags]:uses)unique.insert({key[1],key[2],key[3],key[4]});
  std::vector<int> receives_by_rank(ranks),sends_by_rank(ranks);
  Count remote=0;int recv_peers=0,send_peers=0;
  for(const auto &p:unique)if(mesh.rank_eachmb[p.gid]!=rank) {
    ++remote;++receives_by_rank[mesh.rank_eachmb[p.gid]];
  }
  MPI_Alltoall(receives_by_rank.data(),1,MPI_INT,sends_by_rank.data(),1,MPI_INT,MPI_COMM_WORLD);
  Count send=0;
  for(int r=0;r<ranks;++r){send+=sends_by_rank[r];recv_peers+=receives_by_rank[r]>0;send_peers+=sends_by_rank[r]>0;}
  const auto &residual=halo.ResidualExchange(),&physical=halo.PhysicalExchange();
  for(const auto *exchange:{&residual,&physical}) {
    const std::set<bvals::DonorPoint> actual(exchange->UniqueDonors().begin(),exchange->UniqueDonors().end());
    Require(actual.size()==unique.size()&&std::equal(actual.begin(),actual.end(),unique.begin(),
      [](const auto&a,const auto&b){return !(a<b)&&!(b<a);}),"unique donor statistics mismatch");
    Require(exchange->RemoteDonors()==remote&&exchange->SendDonors()==send
        &&exchange->ReceivePeers()==recv_peers&&exchange->SendPeers()==send_peers,
        "public exchange traffic statistics mismatch");
  }
  Require(residual.Fields().size()==26&&physical.Fields().size()==31,"changed field payload");
  const std::string suffix="-rank"+std::to_string(rank);
  std::ofstream block_file("blocks"+suffix+".tsv");
  block_file<<"gid\towner\tchart\tlayer\tlevel\trows\tentries\texterior\tsingle_nonunit_near_one";
  for(const char *name:names)block_file<<'\t'<<name;block_file<<'\n';
  for(const auto &[gid,c]:blocks) {
    const auto &map=selector.Map(gid);
    block_file<<gid<<'\t'<<rank<<'\t'<<map.chart_id<<'\t'<<map.layer<<'\t'
      <<mesh.lloc_eachmb[gid].level<<'\t'<<c.rows<<'\t'<<c.entries<<'\t'<<c.exterior<<'\t'<<c.single_nonunit_near_one;
    for(Count value:c.reason)block_file<<'\t'<<value;block_file<<'\n';
  }
  block_file.close();Require(bool(block_file),"block metadata write failed");
  std::ofstream route_file("routes"+suffix+".tsv");
  route_file<<"target_gid\tdonor_gid\tneighbor\teligible_rows\tplan_rows_in_slab\tevolved_in_slab\tslab_points\n";
  for(const auto &[key,r]:routes)route_file<<r.target<<'\t'<<r.donor<<'\t'<<r.neighbor<<'\t'
    <<r.eligible_rows<<'\t'<<r.plan_rows<<'\t'<<r.evolved<<'\t'<<r.slab<<'\n';
  route_file.close();Require(bool(route_file),"route metadata write failed");
  std::ofstream use_file("uses"+suffix+".bin",std::ios::binary);
  const char magic[16]="BVALS_USES_V1";const Count count=uses.size();const std::uint32_t shape[2]={6,1};
  use_file.write(magic,16);use_file.write(reinterpret_cast<const char*>(&count),8);
  use_file.write(reinterpret_cast<const char*>(shape),8);
  for(const auto &[key,flags]:uses) {
    const std::int32_t record[6]={key[0],key[1],key[2],key[3],key[4],flags};
    use_file.write(reinterpret_cast<const char*>(record),sizeof(record));
  }
  use_file.close();Require(bool(use_file),"donor metadata write failed");
  std::ofstream stats("stats"+suffix+".json");
  stats<<"{\n  \"schema\":\"native-bvals-route-capture-v1\",\n  \"passed\":true,\n"
    <<"  \"backend\":\""<<Kokkos::DefaultExecutionSpace::name()<<"\",\n"
    <<"  \"rank\":"<<rank<<",\n  \"ranks\":"<<ranks<<",\n  \"origin\":"<<origin
    <<",\n  \"cells\":"<<n<<",\n  \"real_bytes\":"<<sizeof(Real)
    <<",\n  \"blocks\":"<<pack->nmb_thispack<<",\n  \"use_records\":"<<count
    <<",\n  \"unique_donors\":"<<unique.size()<<",\n  \"remote_donors\":"<<remote
    <<",\n  \"send_donors\":"<<send<<",\n  \"receive_peers\":"<<recv_peers
    <<",\n  \"send_peers\":"<<send_peers
    <<",\n  \"residual_fields\":26,\n  \"physical_fields\":31,\n"
    <<"  \"public_exchange_statistics_match\":true,\n  \"field_transport_called\":false\n}\n";
  stats.close();Require(bool(stats),"statistics write failed");
}

int main(int argc,char **argv) {
  MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);
  if(argc!=2){MPI_Abort(MPI_COMM_WORLD,2);return 2;}
  const std::string input=argv[1];Kokkos::initialize(argc,argv);
  try {{
    ParameterInput pin;IOWrapper file;file.Open(input.c_str(),IOWrapper::FileMode::read);
    pin.LoadFromFile(file);file.Close();Mesh mesh(&pin);mesh.BuildTreeFromScratch(&pin);
    mesh.pmb_pack->pcoord=nullptr;  // BuildTree-only fixture does not AddCoordinates.
    Capture(mesh,pin);
  }
    if(global_variable::my_rank==0)std::cout<<"PASS metadata-only native bvals route capture\n";
  }catch(const std::exception&e){std::cerr<<"rank "<<global_variable::my_rank<<": "<<e.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);}
  Kokkos::finalize();MPI_Finalize();return 0;
}
