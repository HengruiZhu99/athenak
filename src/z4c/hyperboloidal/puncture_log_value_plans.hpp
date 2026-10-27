// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_PUNCTURE_LOG_VALUE_PLANS_HPP_
#define Z4C_HYPERBOLOIDAL_PUNCTURE_LOG_VALUE_PLANS_HPP_
#include <numeric>
#include <queue>
#include <cstdint>
#include <map>
#include "mesh/cubed_sphere/composite_difference.hpp"
#include "z4c/hyperboloidal/enriched_log_value.hpp"

namespace z4c { namespace hyperboloidal {
// Optional HOST prototype. Only Cartesian coarse/fine scalar-log ghost values
// needed by the existing first/second stencils are changed. Do not use for beta,
// metric components, signed curvature, flux, or physical (non-log) positive data.
class PunctureLogValuePlans {
 public:
  struct Row {int target;std::vector<int> targets;std::vector<size_t> donors;EnrichedLogValue plan;};
  std::vector<Row> rows;
  size_t halo_assignments=0;
  size_t enlarged_support_rows=0;int maximum_donors_used=0;
  std::map<int,size_t> support_counts;
  std::map<std::string,size_t> rejected_supports;
  Real maximum_condition=0,maximum_value_l1=0,maximum_reproduction=0,maximum_alias_distance=0;
  static std::vector<int> SelectTargets(const cubed_sphere::CompositeDifference &grid,
                                       const PuncturePosition &position,Real radius){
    position.Validate();if(!(radius>0)||!std::isfinite(radius))throw std::invalid_argument("positive finite value-enrichment radius required");
    const int block_size=grid.padded*grid.padded*grid.padded;std::vector<int> targets;
    for(const auto &halo:grid.halos){const auto &receiving=grid.mesh.blocks[halo.target/block_size];if(receiving.face>=0)continue;
      const int i=halo.target%grid.padded-grid.ng,j=halo.target/grid.padded%grid.padded-grid.ng,k=halo.target/(grid.padded*grid.padded)%grid.padded-grid.ng;
      // D_iD_j reads at most two logical directions. Triple-corner padding is
      // never read by the original first/second stencils; no active cell is cut.
      if(int(i<0||i>=grid.n)+int(j<0||j>=grid.n)+int(k<0||k>=grid.n)>2)continue;
      const Real q[3]={(i+.5)/grid.n,(j+.5)/grid.n,(k+.5)/grid.n};Real x[3];grid.mesh.Forward(halo.target/block_size,q,x);
      if(position.RadiusSquared(x)>=radius*radius)continue;bool different=false;
      for(const auto &entry:halo.entries){const auto &donor=grid.mesh.blocks[entry.point/block_size];
        if(donor.face>=0)throw std::runtime_error("core Cartesian value halo reaches shell donors; unsupported chart location");
        different=different||donor.refinement!=receiving.refinement;}
      if(different)targets.push_back(halo.target);
    }return targets;
  }
  explicit PunctureLogValuePlans(const cubed_sphere::CompositeDifference &grid,
      const PuncturePosition &position={},Real radius=.12,size_t maximum_rows=10000,Real maximum_l1=8,int maximum_donors=125)
      :grid_(&grid),descriptor_(grid.mesh),active_(grid.active),order_(grid.order),stamp_(position.Capture()){
    if(maximum_rows>10000||maximum_rows<1)throw std::invalid_argument("value-plan row budget must lie in[1,10000]");
    if(!(maximum_l1>=1&&maximum_l1<=16))throw std::invalid_argument("value weight L1 cap must lie in[1,16]");
    if(maximum_donors!=125&&maximum_donors!=150&&maximum_donors!=175&&maximum_donors!=256)throw std::invalid_argument("value support cap must be125,150,175,or256");
    const auto targets=SelectTargets(grid,position,radius);
    halo_assignments=targets.size();
    // Several receiving blocks request the same physical Cartesian query and
    // spacing. Share its immutable VALUE row; do not discard any assignment.
    // This dyadic key is anchored at -half_width and checked against the actual
    // map, not inferred from puncture position or a field's radial power.
    Real minimum_spacing=std::numeric_limits<Real>::infinity();
    for(const auto &block:grid.mesh.blocks)if(block.face<0)for(int a=0;a<3;++a)minimum_spacing=std::min(minimum_spacing,(block.hi[a]-block.lo[a])/grid.n);
    using Key=std::array<std::int64_t,4>;std::map<Key,std::vector<int>> groups;
    for(int index:targets){const auto x=Location(index);const auto &block=grid.mesh.blocks[index/(grid.padded*grid.padded*grid.padded)];Key key;
      for(int a=0;a<3;++a){const Real coordinate=(x[a]+grid.mesh.half_width)*2/minimum_spacing;key[a]=std::llround(coordinate);
        if(!std::isfinite(coordinate)||std::abs(coordinate-key[a])>1e-8)throw std::runtime_error("noncommensurate Cartesian value-query lattice");}
      const Real spacing_units=(block.hi[0]-block.lo[0])/(grid.n*minimum_spacing);key[3]=std::llround(spacing_units);
      if(std::abs(spacing_units-key[3])>1e-8)throw std::runtime_error("noncommensurate Cartesian value-query spacing");groups[key].push_back(index);
    }
    if(groups.size()>maximum_rows)throw std::length_error("unique core Cartesian value plans exceed row budget: "+std::to_string(groups.size()));
    const size_t per=size_t(grid.n)*grid.n*grid.n;std::vector<size_t> ids;
    for(size_t p=0;p<grid.active.size();++p)if(grid.mesh.blocks[p/per].face<0)ids.push_back(p);
    if(!targets.empty()&&ids.size()<size_t(maximum_donors))throw std::runtime_error("value enrichment lacks requested Cartesian active donors");
    tree_.reserve(ids.size());BuildTree(ids,0,ids.size(),0);rows.reserve(groups.size());
    for(const auto &group:groups){const int index=group.second.front();const auto x=Location(index);
      for(int alias:group.second){const auto y=Location(alias);Real distance=0;for(int a=0;a<3;++a)distance+=std::pow(y[a]-x[a],2);distance=std::sqrt(distance);maximum_alias_distance=std::max(maximum_alias_distance,distance);
        if(distance>1e-11*minimum_spacing)throw std::runtime_error("Cartesian query alias exceeds roundoff bound");}
      std::priority_queue<Distance> nearest;Nearest(0,x,nearest,maximum_donors);std::vector<Distance> ordered;
      while(!nearest.empty()){ordered.push_back(nearest.top());nearest.pop();}std::reverse(ordered.begin(),ordered.end());
      Row row;row.target=index;row.targets=group.second;
      const auto &block=grid.mesh.blocks[index/(grid.padded*grid.padded*grid.padded)];const Real spacing=(block.hi[0]-block.lo[0])/grid.n;
      bool ready=false;
      for(int count:{125,150,175,256})if(count<=maximum_donors){std::vector<EnrichedLogValue::Point> points;row.donors.clear();
        for(int i=0;i<count;++i){row.donors.push_back(ordered[i].second);const auto &p=grid.geometry[ordered[i].second].x;points.push_back({p[0],p[1],p[2]});}
        if(row.plan.Build(x,points,spacing,position,grid.mesh.revision,maximum_l1)){ready=true;++support_counts[count];maximum_donors_used=std::max(maximum_donors_used,count);enlarged_support_rows+=count>125;break;}
        ++rejected_supports[row.plan.diagnostics().status];
      }
      if(!ready)throw std::runtime_error("value halo "+std::to_string(index)+" rejected every requested support through"+std::to_string(maximum_donors)+": "+row.plan.diagnostics().status);
      maximum_condition=std::max(maximum_condition,row.plan.diagnostics().basis_condition);maximum_value_l1=std::max(maximum_value_l1,row.plan.diagnostics().value_weight_l1);maximum_reproduction=std::max(maximum_reproduction,row.plan.diagnostics().value_reproduction);rows.push_back(std::move(row));
    }
    // No KD tree/basis factors are needed at stages: retain only value rows,
    // active donor ordinals and immutable geometry/position stamps.
    tree_.clear();tree_.shrink_to_fit();
  }
  void FillPositiveLogs(std::vector<Real> &log_chi_padded,std::vector<Real> &log_alpha_reference_padded,
                        const PuncturePosition &position)const{
    Check(position);if(log_chi_padded.size()!=size_t(grid_->count)||log_alpha_reference_padded.size()!=size_t(grid_->count))throw std::invalid_argument("positive log value padded shape");
    std::vector<std::array<Real,2>> results;results.reserve(rows.size());
    for(const auto &row:rows){std::vector<Real> chi,lapse;chi.reserve(125);lapse.reserve(125);
      for(size_t p:row.donors){chi.push_back(log_chi_padded[grid_->active[p]]);lapse.push_back(log_alpha_reference_padded[grid_->active[p]]);}
      std::array<Real,2> result;if(!row.plan.Evaluate(chi,position,grid_->mesh.revision,result[0])||!row.plan.Evaluate(lapse,position,grid_->mesh.revision,result[1]))throw std::runtime_error("stale/nonfinite positive log value data");results.push_back(result);}
    // Commit only after every input and row passed; a rejection leaves both
    // caller buffers unchanged. All donors are ACTIVE, so rows have no cycles.
    for(size_t r=0;r<rows.size();++r)for(int target:rows[r].targets){log_chi_padded[target]=results[r][0];log_alpha_reference_padded[target]=results[r][1];}
  }
  PunctureLogValuePlans(cubed_sphere::CompositeDifference&&,const PuncturePosition& = {},Real=.12,size_t=10000,Real=8,int=125)=delete;
 private:
  const cubed_sphere::CompositeDifference *grid_;cubed_sphere::CompositeMesh descriptor_;std::vector<int> active_;int order_;PuncturePosition::Stamp stamp_;
  using Distance=std::pair<Real,size_t>;struct Node{size_t point;int left=-1,right=-1,axis=0;};std::vector<Node> tree_;
  EnrichedLogValue::Point Location(int index)const{const int i=index%grid_->padded-grid_->ng,j=index/grid_->padded%grid_->padded-grid_->ng,k=index/(grid_->padded*grid_->padded)%grid_->padded-grid_->ng;
    const Real q[3]={(i+.5)/grid_->n,(j+.5)/grid_->n,(k+.5)/grid_->n};EnrichedLogValue::Point x;grid_->mesh.Forward(index/(grid_->padded*grid_->padded*grid_->padded),q,x.data());return x;}
  int BuildTree(std::vector<size_t>&ids,size_t begin,size_t end,int depth){if(begin==end)return -1;const int axis=depth%3;const size_t middle=(begin+end)/2;
    std::nth_element(ids.begin()+begin,ids.begin()+middle,ids.begin()+end,[&](size_t a,size_t b){const Real x=grid_->geometry[a].x[axis],y=grid_->geometry[b].x[axis];return x<y||(x==y&&a<b);});
    const int node=tree_.size();tree_.push_back({ids[middle],-1,-1,axis});const int left=BuildTree(ids,begin,middle,depth+1),right=BuildTree(ids,middle+1,end,depth+1);tree_[node].left=left;tree_[node].right=right;return node;}
  void Nearest(int node,const EnrichedLogValue::Point &x,std::priority_queue<Distance>&heap,size_t wanted)const{if(node<0)return;const auto &n=tree_[node];Real distance=0;for(int a=0;a<3;++a)distance+=std::pow(grid_->geometry[n.point].x[a]-x[a],2);
    const Distance candidate{distance,n.point};if(heap.size()<wanted)heap.push(candidate);else if(candidate<heap.top()){heap.pop();heap.push(candidate);}const Real split=x[n.axis]-grid_->geometry[n.point].x[n.axis];
    Nearest(split<0?n.left:n.right,x,heap,wanted);if(heap.size()<wanted||split*split<=heap.top().first)Nearest(split<0?n.right:n.left,x,heap,wanted);}
  void Check(const PuncturePosition &position)const{
    if(!position.Matches(stamp_)||grid_->mesh.revision!=descriptor_.revision||grid_->order!=order_||grid_->n!=descriptor_.cells||grid_->ng!=descriptor_.ghosts
       ||grid_->mesh.cells!=descriptor_.cells||grid_->mesh.ghosts!=descriptor_.ghosts||grid_->mesh.half_width!=descriptor_.half_width||grid_->mesh.radii!=descriptor_.radii
       ||grid_->mesh.face_subdivisions!=descriptor_.face_subdivisions||grid_->active!=active_||grid_->mesh.blocks.size()!=descriptor_.blocks.size())throw std::runtime_error("stale positive log value plans; rebuild after mesh/position change");
    for(size_t p=0;p<descriptor_.blocks.size();++p){const auto &a=descriptor_.blocks[p],&b=grid_->mesh.blocks[p];if(a.face!=b.face||a.layer!=b.layer||a.refinement!=b.refinement)throw std::runtime_error("stale positive log value topology");
      for(int d=0;d<3;++d)if(a.lo[d]!=b.lo[d]||a.hi[d]!=b.hi[d])throw std::runtime_error("stale positive log value map");}
  }
};
}}  // namespace z4c::hyperboloidal
#endif
