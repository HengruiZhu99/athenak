#ifndef Z4C_LIVE_HIERARCHY_EVOLUTION_HPP_
#define Z4C_LIVE_HIERARCHY_EVOLUTION_HPP_
#include "z4c/synchronized_hierarchy_evolution.hpp"
namespace z4c {
class LiveHierarchyEvolution {
 public:
  virtual ~LiveHierarchyEvolution()=default;
  virtual subcycling::AcceptedInterval Advance(Real cap,unsigned ratio)=0;
  virtual void ImportAcceptedNativeState()=0;
  virtual std::uint64_t LeafBlockSteps(unsigned ratio) const=0;
};
template<int NG> class LiveHierarchyImplementation final : public LiveHierarchyEvolution {
 public:
  LiveHierarchyImplementation(Mesh *mesh,Z4c *z,int maximum_passes):evolution(mesh,z),native(z) {
    evolution.control.maximum_passes=maximum_passes;
    evolution.control.Validate();
  }
  subcycling::AcceptedInterval Advance(Real cap,unsigned ratio) override {
    const auto result=evolution.Advance(cap,ratio);
    evolution.CopyAcceptedLeavesTo(native->u0);
    return result.interval;
  }
  void ImportAcceptedNativeState() override {
    evolution.ImportSynchronizedLeaves(native->u0);
  }
  std::uint64_t LeafBlockSteps(unsigned ratio) const override {
    const auto &nodes=evolution.tree->Nodes();
    int first=nodes.front().key[0],last=first;
    for(const auto &node:nodes) last=std::max(last,node.key[0]);
    const subcycling::Schedule schedule(first,last,ratio);
    std::uint64_t count=0;
    for(const auto &node:nodes) if(!node.Covered()) count+=schedule.Substeps(node.key[0]);
    return count;
  }
 private:
  SynchronizedHierarchyEvolution<NG> evolution;
  Z4c *native;
};
inline std::unique_ptr<LiveHierarchyEvolution> MakeLiveHierarchy(Mesh *mesh,Z4c *z,int maximum_passes=8) {
  if(maximum_passes<3 || maximum_passes>64)
    throw std::invalid_argument("live corrector pass limit must be between 3 and 64");
  switch(z->opt.fd_stencil) {
    case 2:return std::make_unique<LiveHierarchyImplementation<2>>(mesh,z,maximum_passes);
    case 3:return std::make_unique<LiveHierarchyImplementation<3>>(mesh,z,maximum_passes);
    case 4:return std::make_unique<LiveHierarchyImplementation<4>>(mesh,z,maximum_passes);
    default:throw std::invalid_argument("unsupported live hierarchy stencil");
  }
}
}
#endif
