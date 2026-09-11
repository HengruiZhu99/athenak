#ifndef DRIVER_COVERED_STAGE_CORRECTION_HPP_
#define DRIVER_COVERED_STAGE_CORRECTION_HPP_
#include "driver/rk4_predictor_states.hpp"
#include "driver/subcycle_hierarchy.hpp"
namespace subcycling {
// Corrector transfer: exact native point injection of reconstructed
// fine-stage vectors/RHS onto covered coarse vertices. No physical leaves/ghosts
// on the parent level are destinations.
class CoveredStageCorrection {
 public:
  void Build(const Hierarchy &tree,int level,const std::vector<int> &fine_sources) {
    ready_=false;
    std::map<int,int> local;
    for(int n=0;n<static_cast<int>(fine_sources.size());++n) {
      const int id=fine_sources[n];
      if(id<0 || id>=static_cast<int>(tree.Nodes().size()) ||
         tree.Nodes()[id].key[0]!=level+1 || !local.emplace(id,n).second)
        throw std::invalid_argument("invalid covered correction donor");
    }
    std::vector<std::array<int,5>> rows;
    for(int n=0;n<static_cast<int>(tree.Nodes().size());++n) {
      const auto node=tree.Nodes()[n];if(node.key[0]!=level || !node.Covered()) continue;
      rows.push_back({n,local.at(node.children[0]),local.at(node.children[1]),
                       local.at(node.children[2]),local.at(node.children[3])});
    }
    copies_=DvceArray2D<int>("covered correction indices",rows.size(),5);
    auto h=Kokkos::create_mirror_view(copies_);
    for(int n=0;n<static_cast<int>(rows.size());++n) for(int q=0;q<5;++q) h(n,q)=rows[n][q];
    Kokkos::deep_copy(copies_,h);sources_=fine_sources;
    nodes_=tree.Nodes().size();ready_=true;
  }
  void Apply(const RK4PredictorStates &history,double dt,int stage,bool rhs,
             const z4c::Z4cGridLayout &l,const DvceArray5D<Real> &target) {
    if(!ready_) throw std::logic_error("covered correction not initialized");
    if(history.SourceBlocks()!=sources_) throw std::invalid_argument("changed corrector donors");
    history.EvaluateParentStage(dt,stage,rhs,values_);
    if(target.extent_int(0)!=nodes_ || target.extent_int(1)!=values_.extent_int(1) ||
       l.nx1<=0 || l.nx2<=0 || l.nx1%2 || l.nx2%2 || l.ks!=l.ke ||
       l.is<0 || l.js<0 || l.ks<0 || l.ie-l.is!=l.nx1 || l.je-l.js!=l.nx2 ||
       target.extent_int(4)<=l.ie || target.extent_int(3)<=l.je ||
       target.extent_int(2)<=l.ke || values_.extent_int(2)!=1 ||
       values_.extent_int(3)!=l.nx2+1 || values_.extent_int(4)!=l.nx1+1)
      throw std::invalid_argument("covered correction layout mismatch");
    const auto ids=copies_;const auto values=values_;const int nx=l.nx1,ny=l.nx2,is=l.is,js=l.js;
    if(ids.extent_int(0)>0) par_for("inject covered RK correction",DevExeSpace(),0,ids.extent_int(0)-1,
        0,target.extent_int(1)-1,l.ks,l.ke,l.js,l.je,l.is,l.ie,KOKKOS_LAMBDA(int n,int v,int k,int j,int i) {
      const int cx=i-is>nx/2,cy=j-js>ny/2;
      target(ids(n,0),v,k,j,i)=values(ids(n,1+cx+2*cy),v,0,2*(j-js)-cy*ny,2*(i-is)-cx*nx);
    });
  }
 private:
  bool ready_=false;
  int nodes_=0;
  std::vector<int> sources_;
  DvceArray2D<int> copies_;
  DvceArray5D<Real> values_;
};
}
#endif
