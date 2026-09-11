#ifndef DRIVER_HIERARCHY_PHYSICAL_MAXIMUM_HPP_
#define DRIVER_HIERARCHY_PHYSICAL_MAXIMUM_HPP_
#include "driver/rk4_physical_maximum.hpp"
#include "driver/subcycle_hierarchy.hpp"
namespace subcycling {
// One fixed hierarchy, one rank. Sources are complete levels including covered
// parents; only physical leaves participate. No extrapolation across intervals.
class HierarchyPhysicalMaximum {
 public:
  void Build(const Hierarchy &tree,int first,int second,Real a=1,Real b=2) {
    levels_.clear();
    std::map<int,std::vector<int>> sources,leaves;
    for(int n=0;n<static_cast<int>(tree.Nodes().size());++n) {
      const auto &node=tree.Nodes()[n];sources[node.key[0]].push_back(n);
      if(!node.Covered()) leaves[node.key[0]].push_back(n);
    }
    for(const auto &entry:leaves)
      levels_[entry.first].Build(sources.at(entry.first),entry.second,first,second,a,b);
  }
  template<class Histories> Real Evaluate(const Histories &histories,double time) {
    if(levels_.empty() || !std::isfinite(time))
      throw std::invalid_argument("invalid hierarchy maximum request");
    Real maximum=0;
    for(auto &level:levels_) {
      const RK4PredictorStates *chosen=nullptr;
      // At a shared endpoint choose the later history, whose initial state
      // includes synchronization/projection, rather than the preceding dense endpoint.
      for(const auto &entry:histories) if(entry.first.first==level.first) {
        const auto &h=entry.second;
        const double tol=32*std::numeric_limits<double>::epsilon()*
                         std::max(1.,std::max(std::abs(time),std::abs(h.StartTime())));
        if(time>=h.StartTime()-tol && time<=h.StartTime()+h.Dt()+tol &&
           (!chosen || h.StartTime()>chosen->StartTime())) chosen=&h;
      }
      if(!chosen) throw std::invalid_argument("missing common-time leaf history");
      const double f=std::max(0.,std::min(1.,(time-chosen->StartTime())/chosen->Dt()));
      maximum=std::max(maximum,level.second.Evaluate(*chosen,f));
    }
    return maximum;
  }
 private:
  std::map<int,RK4PhysicalMaximum> levels_;
};
}
#endif
