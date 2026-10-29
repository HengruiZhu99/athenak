// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_HYPERBOLIC_COMPOSITE_EXCHANGE_HPP_
#define Z4C_HYPERBOLOIDAL_HYPERBOLIC_COMPOSITE_EXCHANGE_HPP_
#include <map>
#include <set>
#include "z4c/hyperboloidal/hyperbolic_composite.hpp"
#include "z4c/hyperboloidal/hyperbolic_composite_device.hpp"
namespace z4c::hyperboloidal {
// Persistent device halo plans for a block partition. Active values are gathered
// once per RK stage by the caller; each rank computes only its own ghost cells.
// Scri's small boundary atlas is shared, and its nonlinear null relation is
// reconstructed after angular interpolation at every ghost evaluation.
class HyperbolicCompositeExchange {
 public:
  using Device=HyperbolicCompositeDevice;
  using Buffer=Device::Buffer;using ConstBuffer=Device::ConstBuffer;
  using IntBuffer=Device::IntBuffer;
  using HostBuffer=Kokkos::View<Real*,Kokkos::HostSpace>;
  static constexpr int nf=31;
  struct Boundary {Real x[3];int point[36];Real weight[36];};
  struct Halo {int target,begin,end,boundary;Real boundary_weight;};
  struct Entry {int point;Real weight;};
  template<class T>using View=Kokkos::View<T*,Device::MemorySpace>;
  int count,total;
  Buffer padded,surface;
 private:
  IntBuffer active,scri_active;
  View<Halo> rows;View<Entry> terms;View<Boundary> boundary;
  Buffer boundary_values,null_before,null_after,scri_samples;
  IntBuffer boundary_bad;
  HostBuffer host_surface,host_samples;
  std::vector<int> scri_padded_indices;
  template<class T>static View<T> Upload(const std::string&name,const std::vector<T>&data){
    View<T> view(name,data.size());auto host=Kokkos::create_mirror_view(view);
    for(size_t i=0;i<data.size();++i)host(i)=data[i];Kokkos::deep_copy(view,host);return view;
  }
 public:
  HyperbolicCompositeExchange(const HyperbolicComposite&e,int first_block,int end_block)
    :count(e.grid.count),total(e.grid.active.size()),
     padded("hyperbolic padded",nf*count),surface("hyperbolic scri",e.scri.values.size()),
     host_surface("host scri",e.scri.values.size()) {
    if(first_block<0||end_block<first_block||size_t(end_block)>e.grid.mesh.blocks.size())
      throw std::invalid_argument("invalid device halo block range");
    active=Upload("active indices",e.grid.active);
    std::vector<Halo> hs;std::vector<Entry> es;std::vector<Boundary> bs;
    std::map<std::array<Real,3>,int> boundary_ids;
    const int blocksize=e.grid.padded*e.grid.padded*e.grid.padded;
    for(const auto&row:e.grid.halos){const int block=row.target/blocksize;
      if(block<first_block||block>=end_block)continue;
      Halo h{row.target,int(es.size()),0,-1,row.boundary_weight};
      for(const auto&t:row.entries)es.push_back({t.point,t.weight});h.end=es.size();
      if(row.boundary_weight!=0){auto it=boundary_ids.find(row.boundary);
        if(it==boundary_ids.end()){
          Boundary b{};for(int d=0;d<3;++d)b.x[d]=row.boundary[d];
          const int owner=e.grid.mesh.Owner(b.x),n=e.grid.n;
          if(owner<0||e.scri.base[owner]<0)throw std::runtime_error("no scri donor");
          Real q[3];e.grid.mesh.Inverse(owner,b.x,q);
          int first[2];Real w[2][6];
          for(int a=0;a<2;++a){const Real t=n*q[a+1]-.5;first[a]=std::clamp(int(std::floor(t))-2,0,n-6);
            for(int i=0;i<6;++i){w[a][i]=1;for(int j=0;j<6;++j)if(i!=j)w[a][i]*=(t-first[a]-j)/Real(i-j);}}
          for(int k=0;k<6;++k)for(int j=0;j<6;++j){int i=6*k+j;
            b.point[i]=e.scri.base[owner]+first[0]+j+n*(first[1]+k);b.weight[i]=w[0][j]*w[1][k];}
          it=boundary_ids.emplace(row.boundary,bs.size()).first;bs.push_back(b);
        }h.boundary=it->second;
      }hs.push_back(h);
    }
    rows=Upload("owned halo rows",hs);terms=Upload("owned halo entries",es);boundary=Upload("boundary plans",bs);
    boundary_values=Buffer("boundary interpolants",bs.size()*nf);
    null_before=Buffer("boundary null before",bs.size());null_after=Buffer("boundary null after",bs.size());
    boundary_bad=IntBuffer("boundary errors",bs.size());
    std::set<int> selected;for(const auto&row:e.scri.rows)for(const auto&t:row.terms)if(!t.boundary)selected.insert(t.point);
    std::vector<int> inverse(count,-1);for(int p=0;p<total;++p)inverse[e.grid.active[p]]=p;
    std::vector<int> points;for(int p:selected){
      if(p<0||p>=count||inverse[p]<0)throw std::runtime_error("scri volume donor is not active");
      scri_padded_indices.push_back(p);points.push_back(inverse[p]);}
    scri_active=Upload("scri active donors",points);
    scri_samples=Buffer("scri samples",points.size()*nf);host_samples=HostBuffer("host scri samples",points.size()*nf);
  }
  void SetSurface(const std::vector<Real>&values){
    if(values.size()!=host_surface.extent(0))throw std::runtime_error("surface size mismatch");
    for(size_t i=0;i<values.size();++i)host_surface(i)=values[i];Kokkos::deep_copy(surface,host_surface);
  }
  void DownloadScriVolume(ConstBuffer state,std::vector<Real>&host_padded){
    if(state.extent(0)!=size_t(nf)*total)throw std::invalid_argument("active device state size mismatch");
    const auto ids=scri_active;const auto samples=scri_samples;
    Kokkos::parallel_for("gather scri donors",Kokkos::RangePolicy<>(0,ids.extent(0)*nf),KOKKOS_LAMBDA(int i){samples(i)=state(ids(i/nf)*nf+i%nf);});
    Kokkos::deep_copy(host_samples,scri_samples);
    if(host_padded.size()!=size_t(nf*count))host_padded.assign(nf*count,0);
    for(size_t p=0;p<scri_padded_indices.size();++p)for(int f=0;f<nf;++f)
      host_padded[f*count+scri_padded_indices[p]]=host_samples(p*nf+f);
  }
  struct Summary {Real before=0,after=0;int invalid=0;};
  Summary Fill(ConstBuffer state){
    if(state.extent(0)!=size_t(nf)*total)throw std::invalid_argument("active device state size mismatch");
    const auto a=active;const auto out=padded;const int size=count;
    Kokkos::parallel_for("active transpose",Kokkos::RangePolicy<>(0,total*nf),KOKKOS_LAMBDA(int i){out((i%nf)*size+a(i/nf))=state(i);});
    const auto b=boundary;const auto s=surface;const auto vals=boundary_values;
    const auto before=null_before,after=null_after;const auto bad=boundary_bad;
    Kokkos::parallel_for("scri nonlinear interpolation",Kokkos::RangePolicy<>(0,b.extent(0)),KOKKOS_LAMBDA(int p){
      Real v[nf];const auto plan=b(p);bad(p)=0;bool finite=true;
      for(int f=0;f<nf;++f){const Real anchor=s(plan.point[0]*nf+f);v[f]=anchor;
        for(int k=0;k<36;++k)v[f]+=plan.weight[k]*(s(plan.point[k]*nf+f)-anchor);
        finite=finite&&Kokkos::isfinite(v[f]);}
      MetricJet<Real> metric{};const int ti[6]={0,0,0,1,1,2},tj[6]={0,1,2,1,2,2};
      for(int f=0;f<6;++f)metric.g[ti[f]][tj[f]]=metric.g[tj[f]][ti[f]]=v[1+f]+(ti[f]==tj[f]);
      const auto g=Geometry(metric);Real q=0,normal=0;
      for(int i=0;i<3;++i){normal+=plan.x[i]*(v[22+i]-plan.x[i]);for(int j=0;j<3;++j)q+=g.inverse[i][j]*plan.x[i]*plan.x[j];}
      if(!finite||!g.valid||!(v[0]>0&&q>0)||!Kokkos::isfinite(v[0]*q)){
        bad(p)=1;before(p)=after(p)=0;
        // A rejected boundary must not leave a previous stage's interpolant.
        for(int f=0;f<nf;++f)vals(p*nf+f)=std::numeric_limits<Real>::quiet_NaN();
        return;}
      before(p)=fabs(normal*normal-v[0]*q)/(v[0]*q);const Real speed=sqrt(v[0]*q);v[18]=0;
      for(int i=0;i<3;++i){v[22+i]=(1-speed)*plan.x[i];v[28+i]=0;}
      normal=0;for(int i=0;i<3;++i)normal+=plan.x[i]*(v[22+i]-plan.x[i]);
      after(p)=fabs(normal*normal-v[0]*q)/(v[0]*q);
      for(int f=0;f<nf;++f)vals(p*nf+f)=v[f];
    });
    const auto h=rows;const auto e=terms;
    Kokkos::parallel_for("owned halo lift",Kokkos::RangePolicy<>(0,h.extent(0)*nf),KOKKOS_LAMBDA(int index){
      const auto row=h(index/nf);const int f=index%nf,base=f*size;const Real anchor=out(base+e(row.begin).point);Real value=anchor;
      for(int i=row.begin;i<row.end;++i)value+=e(i).weight*(out(base+e(i).point)-anchor);
      if(row.boundary>=0)value+=row.boundary_weight*(vals(row.boundary*nf+f)-anchor);out(base+row.target)=value;
    });
    Summary result;
    if(b.extent(0)==0){Kokkos::fence();return result;}
    Kokkos::parallel_reduce("null before max",Kokkos::RangePolicy<>(0,b.extent(0)),KOKKOS_LAMBDA(int p,Real&v){v=fmax(v,before(p));},Kokkos::Max<Real>(result.before));
    Kokkos::parallel_reduce("null after max",Kokkos::RangePolicy<>(0,b.extent(0)),KOKKOS_LAMBDA(int p,Real&v){v=fmax(v,after(p));},Kokkos::Max<Real>(result.after));
    Kokkos::parallel_reduce("null invalid",Kokkos::RangePolicy<>(0,b.extent(0)),KOKKOS_LAMBDA(int p,int&v){v+=bad(p);},result.invalid);
    return result;
  }
};
}
#endif
