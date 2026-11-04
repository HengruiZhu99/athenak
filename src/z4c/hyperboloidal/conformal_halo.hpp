// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_CONFORMAL_HALO_HPP_
#define Z4C_HYPERBOLOIDAL_CONFORMAL_HALO_HPP_
#include "mesh/cubed_sphere/composite_difference.hpp"
#include "z4c/hyperboloidal/conformal_profile.hpp"

namespace z4c::hyperboloidal {
// Interpolate through the legacy conformal frame only at Cartesian-to-shell
// interfaces. Geometry/evolved variables and the interpolation weights do not
// change. Ratios are exactly one outside the flattened-Omega modification.
struct ConformalHaloPoint {
  Real ratio=1, reference_lapse=0;
  KOKKOS_INLINE_FUNCTION Real ToLegacy(int f,Real value)const {
    if(f==0)return ratio*ratio*value;
    if(f==18)return (value+reference_lapse)/ratio;
    return ratio*value; // tracefree BSSN A, components 7..12 only
  }
  KOKKOS_INLINE_FUNCTION Real FromLegacy(int f,Real value)const {
    if(f==0)return value/(ratio*ratio);
    if(f==18)return ratio*value-reference_lapse;
    return value/ratio;
  }
  KOKKOS_INLINE_FUNCTION static bool Transformed(int f){return f==0||f==18||(f>=7&&f<=12);}
};
struct ConformalHaloPlan {
  std::vector<ConformalHaloPoint> points;
  std::vector<unsigned char> enabled;
  size_t corrected_rows=0;
  ConformalHaloPlan(const cubed_sphere::CompositeDifference&g,const ConformalProfile&profile)
      :points(g.count),enabled(g.count,0){
    if(!profile.flat)throw std::invalid_argument("frame-aware halos require flat Omega");
    auto point=[&](const Real*x){const Real r2=x[0]*x[0]+x[1]*x[1]+x[2]*x[2],r=std::sqrt(r2);
      return ConformalHaloPoint{r>=profile.outer_radius?1:profile.Radial(r).value/((1-r2)/2),(1+r2)/2};};
    for(size_t p=0;p<g.active.size();++p)points[g.active[p]]=point(g.geometry[p].x);
    const int w=g.padded,blocksize=w*w*w,o=g.storage_halo;
    for(const auto&row:g.halos){
      const int block=row.target/blocksize;
      if(g.mesh.blocks[block].face<0||row.entries.empty()||row.boundary_weight!=0)continue;
      bool all_cartesian=true,changed=false;
      for(const auto&entry:row.entries){
        if(g.inverse_active[entry.point]<0)throw std::runtime_error("frame halo donor is not active");
        all_cartesian=all_cartesian&&g.mesh.blocks[entry.point/blocksize].face<0;
        changed=changed||points[entry.point].ratio!=1;
      }
      if(!all_cartesian)continue;
      const int offset=row.target%blocksize;
      const Real q[3]={(offset%w-o+.5)/g.n,((offset/w)%w-o+.5)/g.n,(offset/(w*w)-o+.5)/g.n};
      Real x[3];g.mesh.Forward(block,q,x);points[row.target]=point(x);
      if(changed||points[row.target].ratio!=1){enabled[row.target]=1;++corrected_rows;}
    }
  }
  void Fill(std::vector<Real>&out,const cubed_sphere::CompositeDifference&g,int first,int last)const{
    for(const auto&row:g.halos)if(enabled[row.target])for(int f=first;f<last;++f){
      if(!ConformalHaloPoint::Transformed(f))continue;
      const int base=f*g.count;const auto&start=row.entries.front();
      const Real anchor=points[start.point].ToLegacy(f,out[base+start.point]);Real value=anchor;
      for(const auto&entry:row.entries)value+=entry.weight*(points[entry.point].ToLegacy(f,out[base+entry.point])-anchor);
      out[base+row.target]=points[row.target].FromLegacy(f,value);
    }
  }
};
}
#endif
