// Read-only verification of the optional inner-transition interpolation control.
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include "mesh/cubed_sphere/composite_difference.hpp"
using Grid=cubed_sphere::CompositeDifference;
static bool Same(const Grid::Row&a,const Grid::Row&b){
  if(a.target!=b.target||a.boundary_weight!=b.boundary_weight||a.entries.size()!=b.entries.size())return false;
  for(size_t i=0;i<a.entries.size();++i)if(a.entries[i].point!=b.entries[i].point||a.entries[i].weight!=b.entries[i].weight)return false;
  return true;
}
static Real Wave(const Real*x){return std::sin(3*x[0]+2*x[1]-4*x[2]);}
int main(){try{
  std::cout<<"n,changed_rows,unchanged_outer_rows,max_constant_error,default_l1,hybrid_l1,default_wave_rms,hybrid_wave_rms\n";
  Real previous=0;
  for(int n:{8,16,24}){
    Grid grid(cubed_sphere::CompositeMesh(.125,{.45,.75,1},1,n,3),4,true);
    const auto baseline=grid.halos;grid.SetInnerTransitionLogical(true);
    std::vector<Real> values(grid.count,0);
    for(size_t p=0;p<grid.active.size();++p)values[grid.active[p]]=Wave(grid.geometry[p].x);
    size_t changed=0,outer=0;Real constant=0,l1[2]{},error[2]{};
    for(size_t r=0;r<baseline.size();++r){
      const auto&a=baseline[r];const auto&b=grid.halos[r];
      int index=a.target;const int i=index%grid.padded-grid.ng;index/=grid.padded;
      const int j=index%grid.padded-grid.ng;index/=grid.padded;
      const int k=index%grid.padded-grid.ng,m=index/grid.padded;
      const Real q[3]={(i+.5)/n,(j+.5)/n,(k+.5)/n};Real x[3];grid.mesh.Forward(m,q,x);
      const int owner=grid.mesh.Owner(x);
      if(owner<0||grid.mesh.blocks[owner].layer>0){
        if(!Same(a,b))throw std::runtime_error("outer/scri donor row changed");++outer;
      }
      if(Same(a,b))continue;
      ++changed;
      for(int variant=0;variant<2;++variant){
        const auto&row=variant?b:a;Real sum=0,absolute=0,interpolated=0;
        for(const auto&e:row.entries){sum+=e.weight;absolute+=std::abs(e.weight);interpolated+=e.weight*values[e.point];}
        constant=std::max(constant,std::abs(sum-1));l1[variant]=std::max(l1[variant],absolute);
        error[variant]+=std::pow(interpolated-Wave(x),2);
      }
    }
    if(!changed||constant>1e-11)throw std::runtime_error("constant or changed-row check failed");
    for(auto&e:error)e=std::sqrt(e/changed);
    if(previous&&!(error[1]<previous))throw std::runtime_error("smooth interpolation error fails to decrease");
    previous=error[1];
    std::cout<<std::setprecision(17)<<n<<','<<changed<<','<<outer<<','<<constant<<','<<l1[0]<<','<<l1[1]<<','<<error[0]<<','<<error[1]<<'\n';
    grid.SetInnerTransitionLogical(false);
    for(size_t r=0;r<baseline.size();++r)if(!Same(baseline[r],grid.halos[r]))throw std::runtime_error("reset differs");
  }
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}return 0;}
