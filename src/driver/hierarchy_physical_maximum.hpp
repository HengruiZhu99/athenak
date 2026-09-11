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
  // Verify the feedback at every nominal stage time used by these histories.
  // Max-location switches are recomputed from fields, not interpolated maxima.
  template<class Histories> Real Difference(const Histories &a,const Histories &b,
                                             const CorrectorControl &control) {
    control.Validate();
    if(a.empty() || a.size()!=b.size())
      throw std::invalid_argument("incompatible gauge histories");
    std::set<double> times;
    for(const auto &entry:a) {
      const auto found=b.find(entry.first);
      const auto &h=entry.second;
      if(found==b.end() || found->second.StartTime()!=h.StartTime() ||
         found->second.Dt()!=h.Dt())
        throw std::invalid_argument("changed gauge history intervals");
      for(double f:{0.,.5,1.}) times.insert(h.StartTime()+f*h.Dt());
    }
    Real error=0;
    for(double time:times) {
      const Real x=Evaluate(a,time),y=Evaluate(b,time);
      error=std::max(error,std::abs(x-y)/(control.absolute_tolerance+
                     control.relative_tolerance*std::max(std::abs(x),std::abs(y))));
    }
    return error;
  }
 private:
  std::map<int,RK4PhysicalMaximum> levels_;
};
}
#endif
