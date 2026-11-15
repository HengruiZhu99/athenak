// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <cmath>
#include <stdexcept>
#include "driver/driver.hpp"
#include "globals.hpp"
#include "z4c/z4c_amr.hpp"
#include "mesh/mesh.hpp"
#include "z4c/compact_object_tracker.hpp"
#include "z4c_hyperboloidal/z4c_hyperboloidal.hpp"
#include "z4c_hyperboloidal/mapped_halo.hpp"
namespace z4c_hyperboloidal {
void Z4cHyperboloidal::RefineBinaryChi() {
  const Real threshold=pinput_->GetOrAddReal("z4c_amr","chi_min",.2);
  const int cap=pinput_->GetInteger("z4c_amr","max_ref_lev");
  const bool coarsen=pinput_->GetOrAddBoolean("z4c_amr","allow_coarsening",true);
  if(!(threshold>0) || !std::isfinite(threshold) || cap<0)
    throw std::invalid_argument("Invalid binary chi AMR policy");
  // Match ordinary isotropic Cartesian chi: x=scale*X and bar chi=scale^2*chi/Omega^2.
  const Real omega=parameters.profile.plateau;
  z4c::RefineChiMinimum(pmy_pack,u0,Fields::chi,overlap,
                       threshold*parameters.inner_gamma/(omega*omega));
  auto &mesh=*pmy_pack->pmesh;auto &flags=mesh.pmr->refine_flag;
  const Real radius=parameters.residual_weight.inner;
  for(int m=0;m<nblocks;++m) {
    const auto map=pmy_pack->pmb->mb_map.h_view(m);
    Real far2=0;for(int a=0;a<3;++a)far2+=std::pow(std::max(std::abs(map.lo[a]),std::abs(map.hi[a])),2);
    const int gid=pmy_pack->gids+m,level=mesh.lloc_eachmb[gid].level-mesh.root_level;
    if(map.chart_id!=0 || far2>radius*radius)flags.h_view(gid)=0;
    else {
      if(level>=cap && flags.h_view(gid)>0)flags.h_view(gid)=0;
      if(!coarsen && flags.h_view(gid)<0)flags.h_view(gid)=0;
    }
  }
  flags.template modify<HostMemSpace>();flags.template sync<DevExeSpace>();
}
void Z4cHyperboloidal::InitializeBinaryTracking() {
  if(!trackers_.empty())return;
  if(pinput_->GetOrAddString("z4c_hyperboloidal","tracker_mode","ode")!="ode")
    throw std::invalid_argument("Native binary tracker requires the existing ODE algorithm");
  for(int h=0;h<2;++h) {
    const auto suffix=std::to_string(h),prefix="co_"+suffix+"_";
    pinput_->GetOrAddString("z4c_hyperboloidal",prefix+"type","BH");
    pinput_->GetOrAddReal("z4c_hyperboloidal",prefix+"mass",1./6.);
    for(int a=0;a<3;++a) {
      const std::string axis(1,"xyz"[a]);
      pinput_->GetOrAddReal("z4c_hyperboloidal",prefix+axis,
        pinput_->GetReal("fastflow","center_"+axis+"_"+suffix));
    }
    auto tracker=std::make_unique<CompactObjectTracker>(pmy_pack->pmesh,pinput_,h,"z4c_hyperboloidal");
    tracker->SetVelocitySampler([this](const Real *x,Real *v) {
      NativeDonorSelector selector(*pmy_pack->pmesh,origin);
      const int gid=selector.Owner(x);
      int valid=gid>=0;
      std::vector<WeightedDonor> row;
      const bool owned=valid && pmy_pack->pmesh->rank_eachmb[gid]==global_variable::my_rank;
      try {if(owned)row=selector.Interpolate(x);}catch(const std::exception &){valid=0;}
#if MPI_PARALLEL_ENABLED
      MPI_Allreduce(MPI_IN_PLACE,&valid,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
#endif
      if(!valid)throw std::runtime_error("Binary tracker point lacks native mapped donors");
      std::vector<bvals::DonorPoint> donors;for(const auto &d:row)donors.push_back(d.point);
      bvals::MappedExchange exchange;exchange.Rebuild(pmy_pack,donors,{22,23,24});
      exchange.StartReceives();exchange.PackAndSend(u0);
      while(exchange.Receive()!=TaskStatus::complete){}
      while(exchange.ClearSend()!=TaskStatus::complete){}
      while(exchange.ClearRecv()!=TaskStatus::complete){}
      const auto values=Kokkos::create_mirror_view_and_copy(HostMemSpace(),exchange.Values());
      if(owned)for(int a=0;a<3;++a) {
        Real beta=0;for(std::size_t d=0;d<row.size();++d)
          beta+=row[d].weight*values(exchange.RequestOrdinals()[d],a);
        v[a]=x[a]-beta; // stored beta+x -> physical dx/dt=-beta
      }
      return owned;
    });
    trackers_.push_back(std::move(tracker));
  }
}
const Real *Z4cHyperboloidal::ComponentCenter(int h) const {
  return trackers_.at(h)->GetPos();
}
TaskStatus Z4cHyperboloidal::TrackBinary(Driver *driver,int stage) {
  if(stage!=driver->nexp_stages)return TaskStatus::complete;
  for(std::size_t h=0;h<trackers_.size();++h) {
    auto &t=trackers_[h];t->InterpolateVelocity(pmy_pack);t->EvolveTracker(pmy_pack);t->WriteTracker();
    for(int a=0;a<3;++a)pinput_->SetRealExact("z4c_hyperboloidal",
      "co_"+std::to_string(h)+"_"+std::string(1,"xyz"[a]),t->GetPos(a));
  }
  return TaskStatus::complete;
}
}
