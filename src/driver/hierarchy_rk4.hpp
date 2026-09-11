#ifndef DRIVER_HIERARCHY_RK4_HPP_
#define DRIVER_HIERARCHY_RK4_HPP_
#include "driver/subcycle_schedule.hpp"
#include "driver/covered_stage_correction.hpp"
#include "driver/classical_rk4_update.hpp"
#include "driver/vertex_parent_states.hpp"
#include "driver/hierarchy_vertex_exchange.hpp"
#include "driver/hierarchy_temporal_ghosts.hpp"
namespace subcycling {
// Assembled fixed-hierarchy recursive RK4 engine. Physics supplies Prepare
// (physical ghosts/stage geometry), RHS, and Project (post-update admissibility).
// Gauge histories, rollback, and live regridding are owned by the outer driver.
// Coarser levels can form a synchronous group. Its internal interfaces use
// current-stage spatial data, with stage restriction before RHS evaluation.
class HierarchyRK4 {
 public:
#ifdef ATHENA_SUBCYCLE_DIAGNOSTICS
  bool test_skip_restriction=false;
  int test_corrector_passes=1;
#endif
  template<int ORDER>
  void Initialize(const Hierarchy &tree,const z4c::Z4cGridLayout &layout,
                  int root,int root_x,int root_y,const std::vector<int> &parities,
                  int extrapolation_order,const std::array<bool,4> &faces) {
    ready_=false;layout_=layout;root_=root;maximum_=root;
    sources_.clear();ghosts_.clear();predictors_.clear();contexts_.clear();
    for(int n=0;n<static_cast<int>(tree.Nodes().size());++n) {
      const int level=tree.Nodes()[n].key[0];
      if(level<root) throw std::invalid_argument("hierarchy RK root mismatch");
      maximum_=std::max(maximum_,level);sources_[level].push_back(n);
    }
    if(maximum_-root>20) throw std::invalid_argument("hierarchy RK ratio exceeds scheduler limit");
    for(int level=root;level<=maximum_;++level)
      if(!sources_.count(level)) throw std::invalid_argument("missing hierarchy RK level");
    exchange_.Build(tree,layout);
    for(int level=root+1;level<=maximum_;++level) {
      ghosts_[level].Build<ORDER>(tree,layout,level,root,root_x,root_y,parities,
                                 extrapolation_order,faces);
      if(ghosts_[level].SourceBlocks()!=sources_.at(level-1))
        throw std::logic_error("hierarchy predictor source order mismatch");
    }
#ifdef ATHENA_SUBCYCLE_DIAGNOSTICS
    corrections_.clear();
    for(int level=root;level<maximum_;++level) corrections_[level].Build(tree,level,sources_.at(level+1));
#endif
    nodes_=tree.Nodes().size();ready_=true;
  }
  template<class Physics>
  void Run(double time,double dt,VertexParentStates &storage,Physics &physics,
           unsigned maximum_ratio=(1U<<20)) {
    if(!ready_ || storage.Values().extent_int(0)!=nodes_ ||
       storage.AllLevels().extent_int(0)!=nodes_)
      throw std::invalid_argument("hierarchy RK storage mismatch");
    for(const auto &entry:sources_) for(int n:entry.second)
      if(storage.AllLevels().h_view(n)!=entry.first)
        throw std::invalid_argument("hierarchy RK level order mismatch");
    const auto state=storage.Values();
#ifdef ATHENA_SUBCYCLE_DIAGNOSTICS
    if(test_corrector_passes>1 && !inside_corrector_) {
      DvceArray5D<Real> saved("corrector interval rollback",state.extent(0),state.extent(1),
                            state.extent(2),state.extent(3),state.extent(4));
      Kokkos::deep_copy(saved,state);previous_history_.clear();inside_corrector_=true;
      try {
        for(int pass=0;pass<test_corrector_passes;++pass) {
          Kokkos::deep_copy(state,saved);current_history_.clear();
          Run(time,dt,storage,physics,maximum_ratio);
          previous_history_=std::move(current_history_);
        }
      } catch(...) {inside_corrector_=false;Kokkos::deep_copy(state,saved);throw;}
      inside_corrector_=false;return;
    }
#endif
    for(auto *scratch:{&initial_,&rhs_,&sum_}) {
      bool resize=false;for(int d=0;d<5;++d) resize|=scratch->extent(d)!=state.extent(d);
      if(resize) Kokkos::realloc(*scratch,state.extent(0),state.extent(1),state.extent(2),
                               state.extent(3),state.extent(4));
    }
    // Lambdas avoid a separate callback type while retaining the actual numeric
    // state machinery in one path used by manufactured and Z4c consumers.
    auto advance=[&](const StepContext &step) {
      const int level=step.maximum_level;
      blocks_.Update(true,storage.AllLevels(),nodes_,step.minimum_level,level);
      for(int stage=1;stage<=4;++stage) {
#ifdef ATHENA_SUBCYCLE_DIAGNOSTICS
        const auto correction=previous_history_.find({level+1,step.begin_tick});
        const bool correct=inside_corrector_ && correction!=previous_history_.end();
        if(correct) corrections_.at(level).Apply(correction->second,step.Dt(),stage,false,layout_,state);
#endif
        for(int parent=level-1;parent>=step.minimum_level;--parent)
          storage.RestrictLevel(layout_,parent);
        exchange_.Apply(state,step.minimum_level,level);
        for(int fine=step.minimum_level+1;fine<=level;++fine)
          ghosts_.at(fine).ApplySpatial(state,layout_);
        if(step.minimum_level>root_) {
          const auto parent=contexts_.at(level-1);
          const double fraction=static_cast<double>(step.begin_tick-parent.begin_tick)/
                                (parent.end_tick-parent.begin_tick);
          ghosts_.at(level).Apply(predictors_.at(level-1),fraction,step.Dt(),stage,state);
        }
        physics.Prepare(step,stage,blocks_,state);
        if(stage==1) {
          const auto initial=initial_;const auto l=layout_;
          blocks_.For5("save hierarchy RK initial state",0,state.extent_int(1)-1,
              l.ks,l.ke,l.js,l.je,l.is,l.ie,KOKKOS_LAMBDA(int m,int v,int k,int j,int i) {
            initial(m,v,k,j,i)=state(m,v,k,j,i);
          });
#ifdef ATHENA_SUBCYCLE_DIAGNOSTICS
          if(inside_corrector_) current_history_[{level,step.begin_tick}].Begin(
              state,layout_,sources_.at(level),step.StartTime(),step.Dt());
#endif
          if(level<maximum_) {
            predictors_[level].Begin(state,layout_,sources_.at(level),step.StartTime(),step.Dt());
            contexts_[level]=step;
          }
        }
        physics.RHS(step,stage,blocks_,state,rhs_);
#ifdef ATHENA_SUBCYCLE_DIAGNOSTICS
        if(correct) corrections_.at(level).Apply(correction->second,step.Dt(),stage,true,layout_,rhs_);
        if(inside_corrector_) current_history_.at({level,step.begin_tick}).Capture(rhs_,stage);
#endif
        if(level<maximum_) predictors_.at(level).Capture(rhs_,stage);
        classical_rk4::Update(blocks_,layout_,step.Dt(),stage,state,initial_,rhs_,sum_);
        physics.Project(step,stage,blocks_,state);
      }
    };
    auto synchronize=[&](const StepContext &step) {
#ifdef ATHENA_SUBCYCLE_DIAGNOSTICS
      if(!test_skip_restriction)
#endif
      for(int parent=step.maximum_level;parent>=step.minimum_level;--parent)
        storage.RestrictLevel(layout_,parent);
      exchange_.Apply(state,step.minimum_level,step.maximum_level);
    };
    struct Callbacks {
      decltype(advance) &advance;
      decltype(synchronize) &synchronize;
      void Advance(const StepContext &s) {advance(s);}
      void Synchronize(const StepContext &s) {synchronize(s);}
    } callbacks{advance,synchronize};
    Schedule(root_,maximum_,maximum_ratio).Run(time,dt,callbacks);
    Kokkos::fence("hierarchy RK synchronized interval complete");
  }
 private:
#ifdef ATHENA_SUBCYCLE_DIAGNOSTICS
  bool inside_corrector_=false;
  std::map<std::pair<int,std::uint64_t>,RK4PredictorStates> previous_history_,current_history_;
  std::map<int,CoveredStageCorrection> corrections_;
#endif
  bool ready_=false;
  int root_=0,maximum_=0,nodes_=0;
  z4c::Z4cGridLayout layout_;
  BlockBatches blocks_;
  HierarchyVertexExchange exchange_;
  std::map<int,std::vector<int>> sources_;
  std::map<int,HierarchyTemporalGhosts> ghosts_;
  std::map<int,RK4PredictorStates> predictors_;
  std::map<int,StepContext> contexts_;
  DvceArray5D<Real> initial_,rhs_,sum_;
};
} // namespace subcycling
#endif
