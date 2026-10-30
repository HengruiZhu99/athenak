// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_HYPERBOLIC_MESH_PLANS_HPP_
#define Z4C_HYPERBOLOIDAL_HYPERBOLIC_MESH_PLANS_HPP_
#include <cstring>
#include <functional>
#include <memory>
#include "z4c/hyperboloidal/hyperbolic_composite_exchange.hpp"
#include "z4c/hyperboloidal/hyperbolic_scri_device.hpp"
#include "z4c/hyperboloidal/hyperbolic_sampled_normal_device.hpp"
#include "z4c/hyperboloidal/hyperbolic_angular_dissipation_device.hpp"
#include "z4c/hyperboloidal/sparse_stage_exchange.hpp"

namespace z4c::hyperboloidal {
// Owns a complete, immovable host/device plan generation for the sampled-normal
// donor-seven scheme. Rebuild is collective and must occur at a synchronization
// point. It does not prolong/restrict physical fields or evolve an AMR mesh.
class HyperbolicMeshPlans {
 public:
  using Model=HyperbolicComposite;
  using Buffer=HyperbolicCompositeDevice::Buffer;
  struct Options {
    Real angular_epsilon=2;
    bool inner_logical=true,positive=true,puncture_reference=false;
    bool boundary_reuse=true,selective_halos=true;
  };
  struct Layout {
    int rank=0,size=0,total=0,first=0,local=0,first_block=0,end_block=0;
    int surface_total=0,surface_first=0,surface_local=0;
    std::vector<int> cuts,counts,displacements,surface_counts,surface_displacements;
    std::vector<int> residual_counts,residual_displacements;
  };
  class Generation {
    friend class HyperbolicMeshPlans;
    // Declared first and destroyed last: surface/radial host plans reference it.
    std::unique_ptr<Model> model_;
   public:
    Layout layout;
    std::unique_ptr<HyperbolicCompositeDevice> bulk;
    std::unique_ptr<HyperbolicCompositeExchange> halo,normal_halo;
    std::unique_ptr<HyperbolicScriDevice> boundary,global_projector;
    std::unique_ptr<HyperbolicSampledNormalDevice> normal;
    std::unique_ptr<HyperbolicAngularDissipationDevice> angular;
    std::unique_ptr<SparseStageExchange> sparse;
    Buffer reference_lapse,angular_residual;
    Real angular_bound=0;
    const Model& Host()const{return *model_;}
  };
 private:
  MPI_Comm comm_;
  int rank_=0,size_=0;
  bool ready_=false,published_=false;
  MeshPlanEpoch epoch_{};
  uint64_t builds_=0;
  std::vector<char> descriptor_;
  std::unique_ptr<Generation> current_;
  void Require(bool okay,const char*reason)const {
    int bad=!okay;MPI_Allreduce(MPI_IN_PLACE,&bad,1,MPI_INT,MPI_MAX,comm_);
    if(bad)throw std::invalid_argument(reason);
  }
  template<class Action> void Local(Action action)const {
    int bad=0;char message[512]{};
    try{action();}catch(const std::exception&e){bad=1;std::strncpy(message,e.what(),sizeof(message)-1);}
    catch(...){bad=1;std::strncpy(message,"unknown mesh-plan construction failure",sizeof(message)-1);}
    int any=bad;MPI_Allreduce(MPI_IN_PLACE,&any,1,MPI_INT,MPI_MAX,comm_);
    if(any){int owner=bad?rank_:size_;MPI_Allreduce(MPI_IN_PLACE,&owner,1,MPI_INT,MPI_MIN,comm_);
      MPI_Bcast(message,sizeof(message),MPI_CHAR,owner,comm_);throw std::invalid_argument(message);}
  }
  template<class T> static void Append(std::vector<char>&out,const T&x){
    const char*p=reinterpret_cast<const char*>(&x);out.insert(out.end(),p,p+sizeof(T));
  }
  // Exact descriptors, not hashes: same-shaped geometry/ownership changes must
  // not be mistaken for the old plan. This work occurs only at rebuild events.
  static std::vector<char> Descriptor(const cubed_sphere::CompositeMesh&m,
                                     const Options&o,const std::vector<int>&cuts){
    std::vector<char> out;
    Append(out,m.half_width);Append(out,m.face_subdivisions);Append(out,m.cells);Append(out,m.ghosts);
    Append(out,uint64_t(m.revision));Append(out,m.evolved_overlap);
    for(bool b:{m.spherical_inner_boundary,m.equiangular,m.continuous_cartesian_core})Append(out,int(b));
    Append(out,m.inner_radius);Append(out,uint64_t(m.radii.size()));for(Real r:m.radii)Append(out,r);
    Append(out,uint64_t(m.blocks.size()));for(const auto&b:m.blocks){
      Append(out,b.face);Append(out,b.layer);Append(out,b.refinement);
      for(int a=0;a<3;++a){Append(out,b.lo[a]);Append(out,b.hi[a]);}}
    Append(out,o.angular_epsilon);
    for(bool b:{o.inner_logical,o.positive,o.puncture_reference,o.boundary_reuse,o.selective_halos})Append(out,int(b));
    Append(out,uint64_t(cuts.size()));for(int cut:cuts)Append(out,cut);
    return out;
  }
  void Consistent(const std::vector<char>&bytes,MeshPlanEpoch epoch)const {
    Require(bytes.size()<=INT_MAX,"mesh descriptor exceeds MPI count");
    int n=bytes.size(),root_n=n;MPI_Bcast(&root_n,1,MPI_INT,0,comm_);
    Require(n==root_n,"inconsistent mesh descriptor lengths");
    std::vector<char> root;Local([&]{root=bytes;});
    MPI_Bcast(root.data(),n,MPI_CHAR,0,comm_);
    uint64_t key[3]={epoch.topology,epoch.geometry,epoch.partition};
    MPI_Bcast(key,3,MPI_UINT64_T,0,comm_);
    Require(root==bytes&&key[0]==epoch.topology&&key[1]==epoch.geometry&&key[2]==epoch.partition,
            "inconsistent mesh descriptors or generations");
  }
 public:
  explicit HyperbolicMeshPlans(MPI_Comm comm=MPI_COMM_WORLD):comm_(comm){
    MPI_Comm_rank(comm_,&rank_);MPI_Comm_size(comm_,&size_);
  }
  HyperbolicMeshPlans(const HyperbolicMeshPlans&)=delete;
  HyperbolicMeshPlans&operator=(const HyperbolicMeshPlans&)=delete;
  // after_local is a LOCAL staging/validation hook (no collectives). A caller
  // may prepare new-layout fields here, but must publish them only on success.
  // A hook exception on one rank invalidates the replacement on every rank.
  void Rebuild(const cubed_sphere::CompositeMesh&mesh,const Options&options,
               const std::vector<int>&cuts,MeshPlanEpoch epoch,
               const std::function<void(const Generation&)>&after_local={}) {
    Kokkos::fence();const bool was_ready=ready_;ready_=false;
    std::vector<char> descriptor;
    Local([&]{descriptor=Descriptor(mesh,options,cuts);});Consistent(descriptor,epoch);
    const bool same=published_&&epoch==epoch_;
    Require(!published_||(epoch.topology>=epoch_.topology&&epoch.geometry>=epoch_.geometry&&epoch.partition>=epoch_.partition),
            "mesh generation moved backwards");
    Require(!same||descriptor==descriptor_,"mesh descriptor changed without a new generation");
    // All ranks either reuse the complete generation or rebuild it together.
    int reuse=was_ready&&same,all_reuse=reuse;MPI_Allreduce(MPI_IN_PLACE,&all_reuse,1,MPI_INT,MPI_MIN,comm_);
    if(all_reuse){ready_=true;return;}
    std::unique_ptr<Generation> next;
    Local([&]{
      if((!cuts.empty()&&(cuts.size()!=size_t(size_+1)||cuts.front()!=0))||!std::isfinite(options.angular_epsilon)||options.angular_epsilon<0)
        throw std::invalid_argument("invalid mesh-plan options or ownership");
      for(int r=0;!cuts.empty()&&r<size_;++r)if(cuts[r+1]<=cuts[r])throw std::invalid_argument("empty mesh-plan ownership");
      next=std::make_unique<Generation>();
      next->model_=std::make_unique<Model>(mesh,false,true,options.angular_epsilon,
          Model::NormalDerivative::sampled_numerator,Model::RadialClosure::donor_seven);
      auto&m=*next->model_;if(options.inner_logical)m.grid.SetInnerTransitionLogical(true);
      m.positive_dissipation=options.positive;m.puncture_reference=options.puncture_reference;
      auto&l=next->layout;l.rank=rank_;l.size=size_;l.cuts=cuts;
      if(m.grid.active.size()>size_t(INT_MAX/31)||m.scri.rows.size()>size_t(INT_MAX/31))
        throw std::invalid_argument("mesh-plan field size exceeds MPI count");
      l.total=m.grid.active.size();
      // Empty cuts select deterministic equal-cell ownership after constructing
      // the host grid once. Explicit cuts remain part of the exact descriptor.
      if(l.cuts.empty()){l.cuts.resize(size_+1);for(int r=0;r<=size_;++r)l.cuts[r]=int(int64_t(l.total)*r/size_);}
      const auto&cuts=l.cuts;
      l.first=cuts[rank_];l.local=cuts[rank_+1]-l.first;
      if(l.local==0)throw std::invalid_argument("empty mesh-plan ownership");
      if(cuts.back()!=l.total)throw std::invalid_argument("ownership does not cover rebuilt mesh");
      l.surface_total=m.scri.rows.size();l.surface_first=l.surface_total*rank_/size_;
      l.surface_local=l.surface_total*(rank_+1)/size_-l.surface_first;
      if(l.surface_local==0)throw std::invalid_argument("empty scri ownership");
      l.first_block=m.grid.mesh.blocks.size();l.end_block=0;std::set<int> selected;
      for(int p=l.first;p<l.first+l.local;++p){const int b=m.grid.cells[p].block;
        selected.insert(b);l.first_block=std::min(l.first_block,b);l.end_block=std::max(l.end_block,b+1);}
      std::vector<int> blocks(selected.begin(),selected.end());
      l.counts.resize(size_);l.displacements.resize(size_);l.surface_counts.resize(size_);l.surface_displacements.resize(size_);
      l.residual_counts.resize(size_);l.residual_displacements.resize(size_);
      for(int r=0;r<size_;++r){l.counts[r]=31*(cuts[r+1]-cuts[r]);l.displacements[r]=31*cuts[r];
        l.surface_counts[r]=31*(l.surface_total*(r+1)/size_-l.surface_total*r/size_);l.surface_displacements[r]=31*(l.surface_total*r/size_);
        l.residual_counts[r]=21*(l.surface_counts[r]/31);l.residual_displacements[r]=21*(l.surface_displacements[r]/31);}
      next->bulk=std::make_unique<HyperbolicCompositeDevice>(m.grid,l.first,l.local,m.radial_plan.get());
      next->halo=std::make_unique<HyperbolicCompositeExchange>(m,l.first_block,l.end_block,blocks,options.selective_halos);
      next->boundary=std::make_unique<HyperbolicScriDevice>(m,l.surface_first,l.surface_local);
      if(options.boundary_reuse)next->global_projector=std::make_unique<HyperbolicScriDevice>(m,0,l.surface_total);
      next->normal=std::make_unique<HyperbolicSampledNormalDevice>(m,l.surface_first,l.surface_local);
      next->normal_halo=std::make_unique<HyperbolicCompositeExchange>(m,next->normal->FirstBlock(),next->normal->EndBlock(),std::vector<int>{},options.selective_halos,false);
      if(options.angular_epsilon>0){
        next->angular=std::make_unique<HyperbolicAngularDissipationDevice>(*m.angular_plan,
            options.boundary_reuse?0:l.surface_first,options.boundary_reuse?l.surface_total:l.surface_local);
        next->angular_bound=m.angular_plan->SpectralUpperBound();next->angular_residual=Buffer("generation angular residual",21*l.surface_total);
      }
      next->reference_lapse=Buffer("generation reference lapse",l.local);
      auto lapse=Kokkos::create_mirror_view(next->reference_lapse);
      for(int p=0;p<l.local;++p){const Real r=m.Radius(m.grid.geometry[l.first+p].x);lapse(p)=(1+r*r)/2;}
      Kokkos::deep_copy(next->reference_lapse,lapse);
      next->sparse=std::make_unique<SparseStageExchange>(comm_);
      if(after_local)after_local(*next);
    });
    std::vector<int> needed;
    Local([&]{needed=next->halo->ActiveDependencies();
      for(const auto*readset:{&next->normal_halo->ActiveDependencies(),&next->boundary->ActiveDependencies()})
        needed.insert(needed.end(),readset->begin(),readset->end());});
    next->sparse->Rebuild(epoch,next->layout.cuts,31,needed);
    current_.swap(next);descriptor_.swap(descriptor);epoch_=epoch;published_=ready_=true;++builds_;
  }
  // Call collectively before every stage/diagnostic use. Do not retain borrowed
  // operators or their stage buffers across Rebuild/Invalidate. The action must
  // contain only local work: its exceptions are synchronized before returning.
  // Use Exchange below for the collective sparse application. No per-stage
  // descriptor scan or donor search is performed here.
  template<class Action> void Use(MeshPlanEpoch epoch,Action action){
    Require(ready_&&epoch==epoch_,"stale or incomplete hyperbolic mesh-plan generation");
    Local([&]{action(*current_);});
  }
  // For local operator calls already enclosed by a caller's collective error
  // boundary. Re-fetch on every call; never retain the returned reference over
  // Rebuild/Invalidate. This check itself performs no MPI communication.
  Generation& Borrow(MeshPlanEpoch epoch){
    if(!ready_||!(epoch==epoch_))throw std::invalid_argument("stale or incomplete hyperbolic mesh-plan generation");
    return *current_;
  }
  SparseStageExchange::Timing Exchange(MeshPlanEpoch epoch,Buffer owned,Buffer global,bool poison=false){
    Require(ready_&&epoch==epoch_,"stale or incomplete hyperbolic mesh-plan generation");
    return current_->sparse->Exchange(owned,global,epoch,poison);
  }
  void Gather(MeshPlanEpoch epoch,Buffer owned,Buffer global,bool surface=false){
    Require(ready_&&epoch==epoch_,"stale or incomplete hyperbolic mesh-plan generation");
    const auto&l=current_->layout;const auto&counts=surface?l.surface_counts:l.counts;
    const auto&offsets=surface?l.surface_displacements:l.displacements;
    const auto ob=reinterpret_cast<uintptr_t>(owned.data()),gb=reinterpret_cast<uintptr_t>(global.data());
    const bool separate=ob+owned.extent(0)*sizeof(Real)<=gb||gb+global.extent(0)*sizeof(Real)<=ob;
    Require(owned.extent(0)==size_t(counts[rank_])&&global.extent(0)==size_t(31)*(surface?l.surface_total:l.total)&&separate,
            "incompatible mesh-generation gather storage");
    Kokkos::fence();MPI_Allgatherv(owned.data(),counts[rank_],MPI_ATHENA_REAL,
        global.data(),counts.data(),offsets.data(),MPI_ATHENA_REAL,comm_);
  }
  void Invalidate(){Kokkos::fence();ready_=false;}
  uint64_t Builds()const{return builds_;}
};
} // namespace z4c::hyperboloidal
#endif
