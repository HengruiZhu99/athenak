// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_NATIVE_COMPOSITE_FIELDS_HPP_
#define Z4C_HYPERBOLOIDAL_NATIVE_COMPOSITE_FIELDS_HPP_

#include "z4c/z4c.hpp"
#include "z4c/hyperboloidal/composite_state.hpp"
#include "mesh/cubed_sphere/composite_difference.hpp"

namespace z4c { namespace hyperboloidal {
// Host adapter to the ACTUAL native Z4c field layout. This is deliberately
// separate from the physics and mesh topology. It does not schedule tasks or
// allocate a shell pack. The caller must reserve native B slots for the new
// reference-vector flux, instead of simultaneously using a Gamma driver.
// Mirrors preserve Kokkos view compatibility; the elliptic backend is host-only.
class NativeCompositeFields {
 public:
  struct Block {int native, composite;};
  const cubed_sphere::CompositeDifference &grid;
  bool puncture;
  int reference;
  NativeCompositeFields(const cubed_sphere::CompositeDifference &g,bool chart=false,
      int chart_reference=0,int chi_psi_power=-4)
      :grid(g),puncture(chart),reference(chart_reference){
    static_assert(Z4c::nz4c==25,"native composite field contract changed");
    if(chi_psi_power!=-4||reference<0||reference>2||(!puncture&&reference))
      throw std::invalid_argument("native composite chart/chi convention");
  }
  static constexpr int scalar_native[22]={Z4c::I_Z4C_CHI,
    Z4c::I_Z4C_GXX,Z4c::I_Z4C_GXY,Z4c::I_Z4C_GXZ,Z4c::I_Z4C_GYY,Z4c::I_Z4C_GYZ,Z4c::I_Z4C_GZZ,
    Z4c::I_Z4C_AXX,Z4c::I_Z4C_AXY,Z4c::I_Z4C_AXZ,Z4c::I_Z4C_AYY,Z4c::I_Z4C_AYZ,Z4c::I_Z4C_AZZ,
    Z4c::I_Z4C_KHAT,Z4c::I_Z4C_THETA,Z4c::I_Z4C_GAMX,Z4c::I_Z4C_GAMY,Z4c::I_Z4C_GAMZ,
    Z4c::I_Z4C_ALPHA,Z4c::I_Z4C_BX,Z4c::I_Z4C_BY,Z4c::I_Z4C_BZ};
  void Read(const DvceArray5D<Real> &native,const std::vector<Block> &blocks,
      std::vector<Real> &coefficients,std::vector<Real> &beta_delta) const {
    Validate(native,blocks);
    const auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),native);
    const size_t points=grid.active.size();
    if(coefficients.empty())coefficients.assign(22*points,0);
    if(beta_delta.empty())beta_delta.assign(3*points,0);
    if(coefficients.size()!=22*points||beta_delta.size()!=3*points)
      throw std::invalid_argument("native stage output vector shape");
    const int n=grid.n,ng=grid.ng;
    for(const auto &block:blocks)for(int k=0;k<n;++k)for(int j=0;j<n;++j)for(int i=0;i<n;++i){
      const size_t p=((block.composite*n+k)*n+j)*n+i;const auto &x=grid.geometry[p].x;
      Real q[22];for(int f=0;f<22;++f)q[f]=host(block.native,scalar_native[f],k+ng,j+ng,i+ng);
      if(!(q[0]>0&&q[18]>0))throw std::runtime_error("native stage nonpositive chi/lapse");
      if(puncture){const auto c=WormholeChart::At(x,reference);
        q[0]=std::log(q[0]/c.chi.f);q[18]=std::log(q[18]/c.alpha.f);
        q[13]=3+q[13]/c.weight.f;q[14]/=c.weight.f;
        for(int t=0;t<6;++t)q[7+t]/=c.weight.f;
      }else{q[0]-=1;q[13]+=3;q[18]-=(1+x[0]*x[0]+x[1]*x[1]+x[2]*x[2])/2;}
      for(int t=0;t<6;++t)if(kCompositeTensorI[t]==kCompositeTensorJ[t])q[1+t]-=1;
      for(int f=0;f<22;++f){if(!std::isfinite(q[f]))throw std::runtime_error("native stage nonfinite coefficient");coefficients[22*p+f]=q[f];}
      for(int v=0;v<3;++v)beta_delta[3*p+v]=host(block.native,Z4c::I_Z4C_BETAX+v,k+ng,j+ng,i+ng)+x[v];
    }
  }
  // Write reconstructed physical state or physical rates. All untouched blocks
  // and every ghost cell retain their original values. Beta is algebraic stage
  // data; its time derivative is zero here and must not be evolved as a driver.
  void Write(DvceArray5D<Real> &native,const std::vector<Block> &blocks,
      const std::vector<Real> &coefficients,const std::vector<Real> &beta_delta,
      const std::vector<Real> *coefficient_rate=nullptr) const {
    Validate(native,blocks);const size_t points=grid.active.size();
    if(coefficients.size()!=22*points||beta_delta.size()!=3*points||
        (coefficient_rate&&coefficient_rate->size()!=coefficients.size()))
      throw std::invalid_argument("native stage input vector shape");
    auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),native);
    const int n=grid.n,ng=grid.ng;
    for(const auto &block:blocks)for(int k=0;k<n;++k)for(int j=0;j<n;++j)for(int i=0;i<n;++i){
      const size_t p=((block.composite*n+k)*n+j)*n+i;const auto &x=grid.geometry[p].x;
      Real q[22];for(int f=0;f<22;++f)q[f]=(coefficient_rate?*coefficient_rate:coefficients)[22*p+f];
      if(puncture){const auto c=WormholeChart::At(x,reference);
        if(coefficient_rate){q[0]*=c.chi.f*std::exp(coefficients[22*p]);q[18]*=c.alpha.f*std::exp(coefficients[22*p+18]);
          q[13]*=c.weight.f;q[14]*=c.weight.f;
        }else{q[0]=c.chi.f*std::exp(q[0]);q[18]=c.alpha.f*std::exp(q[18]);q[13]=c.weight.f*(q[13]-3);q[14]*=c.weight.f;}
        for(int t=0;t<6;++t)q[7+t]*=c.weight.f;
      }else if(!coefficient_rate){q[0]+=1;q[13]-=3;q[18]+=(1+x[0]*x[0]+x[1]*x[1]+x[2]*x[2])/2;}
      if(!coefficient_rate)for(int t=0;t<6;++t)if(kCompositeTensorI[t]==kCompositeTensorJ[t])q[1+t]+=1;
      for(int f=0;f<22;++f){if(!std::isfinite(q[f]))throw std::runtime_error("nonfinite native stage write");host(block.native,scalar_native[f],k+ng,j+ng,i+ng)=q[f];}
      for(int v=0;v<3;++v)host(block.native,Z4c::I_Z4C_BETAX+v,k+ng,j+ng,i+ng)=coefficient_rate?0:beta_delta[3*p+v]-x[v];
    }
    Kokkos::deep_copy(native,host);
  }
 private:
  void Validate(const DvceArray5D<Real> &native,const std::vector<Block> &blocks) const {
    if(native.extent(1)!=Z4c::nz4c)throw std::invalid_argument("native Z4c field count");
    for(int a=2;a<5;++a)if(native.extent(a)!=static_cast<size_t>(grid.n+2*grid.ng))
      throw std::invalid_argument("native/composite active shape or ghost contract");
    std::vector<bool> seen(grid.mesh.blocks.size(),false),native_seen(native.extent(0),false);
    for(const auto &b:blocks){if(b.native<0||b.native>=static_cast<int>(native.extent(0))||b.composite<0||
        b.composite>=static_cast<int>(grid.mesh.blocks.size())||seen[b.composite]||native_seen[b.native])
        throw std::invalid_argument("native/composite block map");seen[b.composite]=true;native_seen[b.native]=true;}
  }
};
}}  // namespace z4c::hyperboloidal
#endif
