//========================================================================================
// AthenaK astrophysical plasma code
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
// Optional raw same-cell state increments. No spatial derivatives or ghost-validity
// claims are made here. Large output is intended for targeted operation fixtures.
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "pc_gh/pc_gh.hpp"

namespace pc_gh {

void PcGh::BeginStateBudget(int operation) {
  if (!opt.state_budget
      || (pmy_pack->pmesh->ncycle+1)%opt.state_budget_dcycle != 0) return;
  if (state_budget_pending) throw std::runtime_error("nested PC-GH state budget");
  // create_mirror (not create_mirror_view) must own distinct storage on Serial.
  state_budget_before = Kokkos::create_mirror(u0);
  Kokkos::deep_copy(state_budget_before,u0);
  state_budget_operation = operation;
  state_budget_pending = true;
}

void PcGh::EndStateBudget(int operation) {
  if (!opt.state_budget
      || (pmy_pack->pmesh->ncycle+1)%opt.state_budget_dcycle != 0) return;
  if (!state_budget_pending || state_budget_operation != operation) {
    throw std::runtime_error("unpaired PC-GH state budget");
  }
  auto after = Kokkos::create_mirror_view_and_copy(HostMemSpace(),u0);
  auto *mesh = pmy_pack->pmesh;
  auto ind = mesh->mb_indcs;
  const int nv=u0.extent_int(1), nk=u0.extent_int(2);
  const int nj=u0.extent_int(3), ni=u0.extent_int(4);
  std::ofstream file(opt.reduction_monitor_file+".state-budget.rank"
      +std::to_string(global_variable::my_rank)+".bin",std::ios::binary|std::ios::app);
  if (!file) throw std::runtime_error("cannot open PC-GH state budget");
  std::vector<double> payload(static_cast<std::size_t>(nv)*nk*nj*ni*3);
  for (int m=0; m<pmy_pack->nmb_thispack; ++m) {
    // Header and payload are native-endian int64/double; the marker permits
    // independent readers to detect byte order. No Kokkos layout is serialized.
    const std::int64_t header[20] = {mesh->ncycle,reduction_monitor_stage,operation,
      state_budget_event,global_variable::my_rank,pmy_pack->pmb->mb_gid.h_view(m),
      pmy_pack->pmb->mb_lev.h_view(m),nv,nk,nj,ni,
      ind.is,ind.ie,ind.js,ind.je,ind.ks,ind.ke,0,0,0x0102030405060708LL};
    auto size=pmy_pack->pmb->mb_size.h_view(m);
    const double geometry[8] = {mesh->time,mesh->dt,size.x1min,size.x1max,
      size.x2min,size.x2max,size.x3min,size.x3max};
    std::size_t q=0;
    for (int n=0; n<nv; ++n) for (int k=0; k<nk; ++k) {
      for (int j=0; j<nj; ++j) for (int i=0; i<ni; ++i) {
        double before=state_budget_before(m,n,k,j,i);
        double value=after(m,n,k,j,i);
        payload[q++]=before;
        payload[q++]=value;
        payload[q++]=value-before;
      }
    }
    file.write("PCGHBUD1",8);
    file.write(reinterpret_cast<const char *>(header),sizeof(header));
    file.write(reinterpret_cast<const char *>(geometry),sizeof(geometry));
    file.write(reinterpret_cast<const char *>(payload.data()),
               payload.size()*sizeof(double));
  }
  file.flush();
  if (!file) throw std::runtime_error("incomplete PC-GH state budget write");
  state_budget_pending=false;
  ++state_budget_event;
}

}  // namespace pc_gh
