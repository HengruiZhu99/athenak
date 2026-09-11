#ifndef DRIVER_COVERED_STAGE_CORRECTION_HPP_
#define DRIVER_COVERED_STAGE_CORRECTION_HPP_
#include "driver/rk4_predictor_states.hpp"
#include "driver/subcycle_hierarchy.hpp"
#ifdef ATHENA_SUBCYCLE_DIAGNOSTICS
namespace subcycling {
// Experimental corrector transfer: exact native point injection of reconstructed
// fine-stage vectors/RHS onto covered coarse vertices. No physical leaves/ghosts
// on the parent level are destinations. Compiled only into diagnostic tests.
class CoveredStageCorrection {
 public:
  void Build(const Hierarchy &tree,int level,const std::vector<int> &fine_sources) {
    std::map<int,int> local;for(int n=0;n<static_cast<int>(fine_sources.size());++n) local[fine_sources[n]]=n;
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
  }
  void Apply(const RK4PredictorStates &history,double dt,int stage,bool rhs,
             const z4c::Z4cGridLayout &l,const DvceArray5D<Real> &target) {
    if(history.SourceBlocks()!=sources_) throw std::invalid_argument("changed corrector donors");
    history.EvaluateParentStage(dt,stage,rhs,values_);
    const auto ids=copies_;const auto values=values_;const int nx=l.nx1,ny=l.nx2,is=l.is,js=l.js;
    if(ids.extent_int(0)>0) par_for("inject covered RK correction",DevExeSpace(),0,ids.extent_int(0)-1,
        0,target.extent_int(1)-1,l.ks,l.ke,l.js,l.je,l.is,l.ie,KOKKOS_LAMBDA(int n,int v,int k,int j,int i) {
      const int cx=i-is>nx/2,cy=j-js>ny/2;
      target(ids(n,0),v,k,j,i)=values(ids(n,1+cx+2*cy),v,0,2*(j-js)-cy*ny,2*(i-is)-cx*nx);
    });
  }
 private:
  std::vector<int> sources_;
  DvceArray2D<int> copies_;
  DvceArray5D<Real> values_;
};
}
#endif
#endif
