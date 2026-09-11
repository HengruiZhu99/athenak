#ifndef DRIVER_HIERARCHY_TEMPORAL_GHOSTS_HPP_
#define DRIVER_HIERARCHY_TEMPORAL_GHOSTS_HPP_
#include "driver/vertex_temporal_boundary.hpp"
namespace subcycling {
// Cached coarse-predictor to fine-ghost plan in populated hierarchy node order.
// Physical ghosts are filled separately. Missing spatial stencil support causes
// Build to fail, including support crossing an axis/outer face; no downgrade.
class HierarchyTemporalGhosts {
 public:
  template<int ORDER>
  void Build(const Hierarchy &tree,const z4c::Z4cGridLayout &l,
             int fine_level,int root,int root_x,int root_y) {
    ready_=false;
    if(tree.Dimension()!=2 || root<0 || fine_level<=root || fine_level-root>30 ||
       root_x<=0 || root_y<=0 || l.centering!=z4c::Z4cGridCentering::vertex ||
       l.nx1<2 || l.nx2<2 || l.nx3!=1 || l.n3!=1 || l.ks!=0 || l.ke!=0 ||
       l.ie-l.is!=l.nx1 || l.je-l.js!=l.nx2 || l.is<0 || l.js<0 ||
       l.ie>=l.n1 || l.je>=l.n2)
      throw std::invalid_argument("invalid hierarchy temporal ghost layout");
    const std::int64_t bx=std::int64_t(root_x)<<(fine_level-root);
    const std::int64_t by=std::int64_t(root_y)<<(fine_level-root);
    // Integer global vertex coordinates also need headroom for multiplication.
    if(bx>std::numeric_limits<std::int64_t>::max()/l.nx1 ||
       by>std::numeric_limits<std::int64_t>::max()/l.nx2)
      throw std::invalid_argument("temporal ghost coordinate overflow");
    std::vector<BlockKey> coarse;
    std::vector<int> sources;
    std::vector<FineVertex2D> targets;
    std::vector<std::array<int,3>> destinations;
    for(int n=0;n<static_cast<int>(tree.Nodes().size());++n) {
      const auto key=tree.Nodes()[n].key;
      if(key[0]==fine_level-1) {coarse.push_back(key);sources.push_back(n);}
      if(key[0]!=fine_level) continue;
      for(int j=0;j<l.n2;++j) for(int i=0;i<l.n1;++i) {
        if(i>=l.is && i<=l.ie && j>=l.js && j<=l.je) continue;
        const std::int64_t x=std::int64_t(key[1])*l.nx1+i-l.is;
        const std::int64_t y=std::int64_t(key[2])*l.nx2+j-l.js;
        if(x<0 || y<0 || x>bx*l.nx1 || y>by*l.nx2) continue;
        // Same-level exchange owns any point with an active same-level donor.
        bool same=false;const auto ix=x/l.nx1,iy=y/l.nx2;
        for(auto a=ix-(x%l.nx1==0);a<=ix && !same;++a)
          for(auto b=iy-(y%l.nx2==0);b<=iy && !same;++b) {
            if(a<0 || b<0 || a>std::numeric_limits<int>::max() ||
               b>std::numeric_limits<int>::max()) continue;
            same=tree.Find({fine_level,static_cast<int>(a),static_cast<int>(b),0})>=0;
          }
        if(same) continue;
        targets.push_back({x,y});destinations.push_back({n,j,i});
      }
    }
    if(coarse.empty()) throw std::invalid_argument("missing coarse predictor level");
    interpolation_.Build<ORDER>(coarse,sources,l.nx1,l.nx2,targets);
    destinations_=DvceArray2D<int>("temporal ghost destinations",destinations.size(),3);
    auto h=Kokkos::create_mirror_view(destinations_);
    for(int p=0;p<static_cast<int>(destinations.size());++p)
      for(int q=0;q<3;++q) h(p,q)=destinations[p][q];
    Kokkos::deep_copy(destinations_,h);
    sources_=sources;nodes_=tree.Nodes().size();n1_=l.n1;n2_=l.n2;ready_=true;
  }
  const std::vector<int> &SourceBlocks() const {return sources_;}
  int TargetCount() const {return ready_ ? destinations_.extent_int(0) : 0;}
  void Apply(const RK4PredictorStates &predictor,double fraction,double dt,int stage,
             const DvceArray5D<Real> &state) {
    if(!ready_ || state.extent_int(0)!=nodes_ || state.extent_int(1)<=0 ||
       state.extent_int(2)!=1 || state.extent_int(3)!=n2_ || state.extent_int(4)!=n1_)
      throw std::invalid_argument("invalid temporal ghost destination state");
    interpolation_.Evaluate(predictor,fraction,dt,stage,values_);
    if(values_.extent_int(1)!=state.extent_int(1))
      throw std::invalid_argument("temporal ghost component mismatch");
    const auto ids=destinations_;const auto values=values_;
    if(ids.extent_int(0)>0) par_for("scatter stage-consistent fine ghosts",DevExeSpace(),
        0,ids.extent_int(0)-1,0,state.extent_int(1)-1,KOKKOS_LAMBDA(int p,int v) {
      state(ids(p,0),v,0,ids(p,1),ids(p,2))=values(p,v);
    });
  }
 private:
  bool ready_=false;
  int nodes_=0,n1_=0,n2_=0;
  std::vector<int> sources_;
  VertexTemporalBoundary interpolation_;
  DvceArray2D<int> destinations_;
  DvceArray2D<Real> values_;
};
} // namespace subcycling
#endif
