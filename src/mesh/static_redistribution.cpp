// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause.
#include <algorithm>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "globals.hpp"
#include "coordinates/coordinates.hpp"
#include "mesh/mesh.hpp"
#include "mesh/refinement_criteria.hpp"
#include "utils/phase_timer.hpp"
#include "z4c_hyperboloidal/restart.hpp"
#include "z4c_hyperboloidal/z4c_hyperboloidal.hpp"

namespace {
void RequireRedistribution(bool good,const char *message) {
  int failed=good?0:1;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,&failed,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
#endif
  if (failed) throw std::invalid_argument(message);
}
}

void Mesh::ConfigureStaticOwnership(ParameterInput *pin) {
  if (!pin->DoesParameterExist("multirate","block_owners")) return;
  const auto text=pin->GetString("multirate","block_owners");
  if (text.empty() || text=="auto") return;
  if ((adaptive && pin->GetString("time","integrator")!="mrgark4") || !pin->DoesBlockExist("z4c_hyperboloidal"))
    throw std::invalid_argument("Explicit block ownership requires static native hyperboloidal evolution");
  for (const auto *other:{"hydro","mhd","radiation","z4c","adm","particles",
                          "ion-neutral","gravity","cyclic_zoom"})
    if (pin->DoesBlockExist(other))
      throw std::invalid_argument("Explicit block ownership is supported only for standalone native hyperboloidal evolution");
  std::vector<int> owners;
  std::istringstream input(text);std::string token;
  while (std::getline(input,token,',')) {
    std::istringstream value(token);int rank=-1;
    if (!(value>>rank) || rank<0 || rank>=global_variable::nranks)
      throw std::invalid_argument("Invalid multirate block owner rank");
    value>>std::ws;
    if (!value.eof()) throw std::invalid_argument("Malformed multirate block ownership list");
    owners.push_back(rank);
  }
  if (owners.size()!=static_cast<std::size_t>(nmb_total) || text.back()==',')
    throw std::invalid_argument("Multirate block ownership list must contain one rank per stable block ID");
  std::fill(nmb_eachrank,nmb_eachrank+global_variable::nranks,0);
  std::fill(gids_eachrank,gids_eachrank+global_variable::nranks,nmb_total);
  for (int gid=0;gid<nmb_total;++gid) {
    rank_eachmb[gid]=owners[gid];
    ++nmb_eachrank[owners[gid]];
    gids_eachrank[owners[gid]]=std::min(gids_eachrank[owners[gid]],gid);
  }
  for (int rank=0;rank<global_variable::nranks;++rank)
    if (!nmb_eachrank[rank])
      throw std::invalid_argument("Static multirate ownership requires at least one block per MPI rank");
}

void Mesh::RedistributeStaticBlocks(const std::vector<int> &owners) {
  auto *pack=pmb_pack;
  auto *state=pack?pack->pz4c_hyperboloidal:nullptr;
  RequireRedistribution(state && state->IsInitialized() && (!adaptive || state->UsesMultirate()) && !pzoom &&
      !pack->phydro && !pack->pmhd && !pack->pz4c && !pack->padm && !pack->prad &&
      !pack->ppart && !pack->pgrav && !pack->pionn && !pack->pdyngr &&
      owners.size()==static_cast<std::size_t>(nmb_total),
      "Static redistribution requires an initialized standalone native hyperboloidal mesh");
  std::vector<int> counts(global_variable::nranks,0);
  bool valid=true;
  for (int rank:owners) {
    if (rank<0 || rank>=global_variable::nranks) valid=false;
    else ++counts[rank];
  }
  valid=valid && std::none_of(counts.begin(),counts.end(),[](int n){return n==0;});
  if (adaptive) valid = valid && *std::max_element(counts.begin(), counts.end()) <= nmb_maxperrank;
  RequireRedistribution(valid,"Invalid or over-capacity static block ownership or empty MPI rank");
#if MPI_PARALLEL_ENABLED
  auto agreed=owners;
  MPI_Bcast(agreed.data(),nmb_total,MPI_INT,0,MPI_COMM_WORLD);
  RequireRedistribution(agreed==owners,"MPI ranks disagree on static block ownership");
#endif
  if (std::equal(owners.begin(),owners.end(),rank_eachmb)) return;

  const int rank=global_variable::my_rank;
  std::vector<int> new_ids,new_slot(nmb_total,-1);
  for (int gid=0;gid<nmb_total;++gid) if (owners[gid]==rank) {
    new_slot[gid]=static_cast<int>(new_ids.size());new_ids.push_back(gid);
  }
  PhaseTimer phases("static_redistribution",ncycle);
  HostArray2D<Real> old_records;
  z4c_hyperboloidal::restart::Pack(*state,old_records);
  phases.Mark("pack");
  const auto scalars=old_records.extent(1);
  RequireRedistribution(scalars<=static_cast<std::size_t>(std::numeric_limits<int>::max()),
      "Static migration block record exceeds MPI count limit");
  HostArray2D<Real> new_records("static ownership accepted records",new_ids.size(),scalars);
  // One bounded message per block, sorted by stable identity on both peers.
  // This avoids an aggregate MPI count limit and never replicates global fields.
#if MPI_PARALLEL_ENABLED
  MPI_Comm migration=MPI_COMM_NULL;
  MPI_Comm_dup(MPI_COMM_WORLD,&migration);
  std::vector<MPI_Request> requests;
  requests.reserve(pack->nmb_thispack+new_ids.size());
#endif
  for (int gid=0;gid<nmb_total;++gid) {
    const int before=rank_eachmb[gid],after=owners[gid];
    if (before==rank && after==rank) {
      std::memcpy(&new_records(new_slot[gid],0),&old_records(pack->LocalId(gid),0),
                  scalars*sizeof(Real));
    }
#if MPI_PARALLEL_ENABLED
    else if (before==rank) {
      requests.emplace_back(MPI_REQUEST_NULL);
      MPI_Isend(&old_records(pack->LocalId(gid),0),static_cast<int>(scalars),MPI_ATHENA_REAL,
                after,0,migration,&requests.back());
    } else if (after==rank) {
      requests.emplace_back(MPI_REQUEST_NULL);
      MPI_Irecv(&new_records(new_slot[gid],0),static_cast<int>(scalars),MPI_ATHENA_REAL,
                before,0,migration,&requests.back());
    }
#endif
  }
#if MPI_PARALLEL_ENABLED
  MPI_Waitall(static_cast<int>(requests.size()),requests.data(),MPI_STATUSES_IGNORE);
  MPI_Comm_free(&migration);
#endif
  phases.Mark("transfer");
  std::copy(owners.begin(),owners.end(),rank_eachmb);
  std::copy(counts.begin(),counts.end(),nmb_eachrank);
  std::fill(gids_eachrank,gids_eachrank+global_variable::nranks,nmb_total);
  for (int gid=0;gid<nmb_total;++gid)
    gids_eachrank[owners[gid]]=std::min(gids_eachrank[owners[gid]],gid);
  nmb_thisrank=counts[rank];nmb_maxperrank=std::max(nmb_maxperrank,nmb_thisrank);
  pack->SetBlockIds(std::move(new_ids));
  delete pack->pmb;pack->pmb=nullptr;
  delete pack->pcoord;pack->pcoord=nullptr;
  pack->AddMeshBlocks(parameter_input_);pack->AddCoordinates(parameter_input_);
  pack->pmb->SetNeighbors(ptree,rank_eachmb);
  MarkMeshUpdated();
  phases.Mark("new_blocks");
  state->RebuildAfterRedistribution(new_records);
  phases.Mark("rebuild_state");
  if (adaptive) pmr->pmrc->SetRefinementData(pack,false,false);
  phases.Mark("refinement_data");
  phases.Report();
  std::ostringstream saved;
  for (std::size_t gid=0;gid<owners.size();++gid) {if(gid)saved<<',';saved<<owners[gid];}
  parameter_input_->SetString("multirate","block_owners",saved.str());
}
