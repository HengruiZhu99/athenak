//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

#include "z4c_hyperboloidal/mapped_halo.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "z4c_hyperboloidal/fields.hpp"

namespace z4c_hyperboloidal {
namespace {
Real Radius(const Real x[3]) {
  return std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
}
bool Inside(const Real q[3]) {
  for (int a=0; a<3; ++a) if (q[a]<-2e-13 || q[a]>1+2e-13) return false;
  return true;
}
using CellKey = std::array<int, 4>;  // m,k,j,i
CellKey Key(HaloPoint c) {return {c.m,c.k,c.j,c.i};}
HaloPoint Point(const CellKey &c) {return {c[0],c[1],c[2],c[3]};}

template<class T>
void CopyDevice(const std::vector<T> &host, DvceArray1D<T> &device,
                const char *label) {
  device=DvceArray1D<T>(std::string(label),host.size());
  auto mirror=Kokkos::create_mirror_view(device);
  for (std::size_t i=0;i<host.size();++i) mirror(i)=host[i];
  Kokkos::deep_copy(device,mirror);
}
void LagrangeWeights(Real t, int first, int length, Real *weights) {
  for (int i = 0; i < length; ++i) {
    weights[i] = 1;
    for (int j = 0; j < length; ++j) {
      if (j != i) weights[i] *= (t - first - j) / Real(i - j);
    }
  }
}

// Recover only full native tensor stencils. The stored donor order and every
// coefficient must agree with the original interpolation, including its cutoff.
bool TensorAxes(const MappedHalo::Row &row, const NativeDonorSelector &selector,
                const Real x[3], std::array<std::array<Real, 6>, 3> &weights) {
  if (row.donors.size() != 216) return false;
  const auto p = row.donors.front().point;
  const auto &map = selector.Map(p.gid);
  const int origin = selector.Origin();
  const int first[] = {p.i - origin, p.j - origin, p.k - origin};
  Real q[3];
  map.Inverse(x, q);
  for (int a = 0; a < 3; ++a) {
    LagrangeWeights(selector.Cells()*q[a] - map.node_offset[a], first[a], 6,
                    weights[a].data());
  }
  for (int k = 0; k < 6; ++k) for (int j = 0; j < 6; ++j) for (int i = 0; i < 6; ++i) {
    const auto &entry = row.donors[36*k + 6*j + i];
    const auto d = entry.point;
    if (d.gid != p.gid || d.k != p.k+k || d.j != p.j+j || d.i != p.i+i ||
        entry.weight != weights[0][i]*weights[1][j]*weights[2][k]) return false;
  }
  return true;
}
}  // namespace

NativeDonorSelector::NativeDonorSelector(const Mesh &mesh, int origin, ActivityPolicy activity)
    : activity_(activity), n_(mesh.mb_indcs.nx1), origin_(origin) {
  if (!mesh.atlas || n_<6 || n_!=mesh.mb_indcs.nx2 || n_!=mesh.mb_indcs.nx3) {
    throw std::invalid_argument("mapped donor selection requires a cubic atlas mesh");
  }
  half_width_=mesh.atlas->HalfWidth();
  outer_radius_=mesh.atlas->RadialKnots().back();
  first_radius_=mesh.atlas->RadialKnots().front();
  radii_=mesh.atlas->RadialKnots();
  maps_.reserve(mesh.nmb_total);levels_.reserve(mesh.nmb_total);
  for (int gid=0;gid<mesh.nmb_total;++gid) {
    maps_.push_back(mesh.atlas->GetBlockMap(mesh.lloc_eachmb[gid]));
    levels_.push_back(mesh.lloc_eachmb[gid].level);
  }
  spherical_=true;
  for (const auto &map:maps_) if (map.cube_transition) spherical_=false;
  order_.resize(mesh.nmb_total);std::iota(order_.begin(),order_.end(),0);
  // Geometry ties must not depend on rank ownership or native Morton ordering.
  // This order preserves the N12 reference's coarse-core/children and shell
  // layer/face order while still retaining native GIDs in all donor records.
  std::sort(order_.begin(),order_.end(),[&](int a,int b) {
    const auto &x=maps_[a],&y=maps_[b];
    if ((x.face<0)!=(y.face<0)) return x.face<0;
    if (x.face<0) {
      return std::tie(levels_[a],x.lo[2],x.lo[1],x.lo[0])
          <std::tie(levels_[b],y.lo[2],y.lo[1],y.lo[0]);
    }
    return std::tie(x.layer,x.face,x.lo[2],x.lo[1],x.lo[0])
        <std::tie(y.layer,y.face,y.lo[2],y.lo[1],y.lo[0]);
  });
}

int NativeDonorSelector::Lower(int gid,int axis) const {
  return -maps_.at(gid).overlap_sides[2*axis];
}
int NativeDonorSelector::Upper(int gid,int axis) const {
  return n_+maps_.at(gid).overlap_sides[2*axis+1];
}
bool NativeDonorSelector::IsEvolved(const bvals::DonorPoint &p) const {
  if (p.gid<0 || p.gid>=static_cast<int>(maps_.size())) return false;
  const int q[3]={p.i-origin_,p.j-origin_,p.k-origin_};
  for (int a=0;a<3;++a) if (q[a]<Lower(p.gid,a)||q[a]>=Upper(p.gid,a)) return false;
  return activity_.Evolved(maps_[p.gid],n_,q[0],q[1],q[2]);
}

int NativeDonorSelector::Owner(const Real x[3],bool shell) const {
  const Real radius=Radius(x);
  if (radius>outer_radius_*(1+2e-14)) return -1;
  const bool core=!shell && std::max({std::abs(x[0]),std::abs(x[1]),std::abs(x[2])})
      <=half_width_*(1+2e-14);
  int axis=0;
  for (int a=1;a<3;++a) if (std::abs(x[a])>std::abs(x[axis])) axis=a;
  const int face=core ? -1 : 2*axis+(x[axis]<0);
  const int layer=core ? -1 : static_cast<int>(std::lower_bound(radii_.begin(),
      radii_.end(),std::min(radius,outer_radius_))-radii_.begin());
  int selected=-1;
  for (int gid:order_) {
    const auto &map=maps_[gid];
    if (map.face!=face||map.layer!=layer) continue;
    Real q[3];map.Inverse(x,q);
    if (!Inside(q)) continue;
    if (selected<0 || levels_[gid]>levels_[selected]) selected=gid;
  }
  if (selected<0) throw std::runtime_error("no native atlas leaf contains donor point");
  return selected;
}

std::vector<WeightedDonor> NativeDonorSelector::Tensor(
    int gid,const Real q[3],const int first[3],const int length[3]) const {
  Real weights[3][6]{};
  for (int a=0; a<3; ++a)
    LagrangeWeights(n_*q[a]-maps_[gid].node_offset[a], first[a], length[a], weights[a]);
  std::vector<WeightedDonor> out;
  for (int k=0;k<length[2];++k) for (int j=0;j<length[1];++j)
    for (int i=0;i<length[0];++i) {
      const Real weight=weights[0][i]*weights[1][j]*weights[2][k];
      if (std::abs(weight)<=1e-18) continue;
      bvals::DonorPoint p{gid,first[2]+k+origin_,first[1]+j+origin_,first[0]+i+origin_};
      if (!IsEvolved(p)) throw std::runtime_error("mapped interpolation reads unevolved donor");
      out.push_back({p,weight});
    }
  return out;
}

std::vector<WeightedDonor> NativeDonorSelector::Cartesian(const Real x[3],int width) const {
  int owner=-1;Real best_h=std::numeric_limits<Real>::infinity();
  for (int gid:order_) {
    const auto &map=maps_[gid];if (map.face>=0) continue;
    Real q[3];map.Inverse(x,q);bool inside=true;
    for (int a=0;a<3;++a)
      inside=inside&&n_*q[a]>=Lower(gid,a)-1e-11&&n_*q[a]<=Upper(gid,a)+1e-11;
    const Real h=(map.hi[0]-map.lo[0])/n_;
    if (inside&&h<best_h) {owner=gid;best_h=h;}
  }
  if (owner<0) return {};
  Real q[3];maps_[owner].Inverse(x,q);int first[3],length[3];
  for (int a=0;a<3;++a) {
    const Real t=n_*q[a]-.5;
    const bool exact=std::abs(t-std::round(t))<1e-10;
    length[a]=exact?1:width;
    first[a]=exact?int(std::round(t)):std::clamp(int(std::floor(t))-width/2+1,
                                               Lower(owner,a),Upper(owner,a)-width);
    if (first[a]<Lower(owner,a)||first[a]+length[a]>Upper(owner,a)) return {};
  }

  return Tensor(owner,q,first,length);
}

std::vector<WeightedDonor> NativeDonorSelector::Overlap(const Real x[3],int width) const {
  const int preferred=Owner(x);int selected=-1,best_first[3]{},best_length[3]{};
  Real best_q[3]{},best_margin=-1e300;
  for (int gid:order_) {
    const auto &map=maps_[gid];if (map.face<0||map.layer>0) continue;
    if ((map.face%2?-x[map.face/2]:x[map.face/2])<=0) continue;
    Real q[3];map.Inverse(x,q);int first[3],length[3];Real margin=1e300;
    bool valid=true,aligned=true;
    for (int a=0;a<3;++a) {
      const Real t=n_*q[a]-map.node_offset[a];
      if (!std::isfinite(t)) {valid=false;break;}
      const bool exact=spherical_&&std::abs(t-std::round(t))<=64*
          std::numeric_limits<Real>::epsilon()*std::max(Real(1),std::abs(t));
      length[a]=exact?1:width;aligned=aligned&&exact;
      first[a]=exact?int(std::round(t)):int(std::floor(t))-width/2+1;
      if (first[a]<Lower(gid,a)||first[a]+length[a]>Upper(gid,a)) {valid=false;break;}
      margin=std::min(margin,std::min(t-Lower(gid,a),Upper(gid,a)-1-t));
    }
    if (!valid) continue;
    if (gid==preferred) margin+=1e6;
    if (aligned&&spherical_) margin+=1e9;
    if (margin>best_margin) {
      selected=gid;best_margin=margin;
      for (int a=0;a<3;++a) {best_first[a]=first[a];best_length[a]=length[a];best_q[a]=q[a];}
    }
  }
  if (selected<0) {
    // A refined first-shell child can contain the target without containing a
    // centered six-point stencil across its internal radial/angular boundary.
    // Keep the established centered search above unchanged. Only if it finds
    // no donor, use a one-sided stencil entirely in a covering evolved leaf.
    for (int gid:order_) {
      const auto &map=maps_[gid];if (map.face<0||map.layer>0) continue;
      if ((map.face%2?-x[map.face/2]:x[map.face/2])<=0) continue;
      Real q[3];map.Inverse(x,q);int first[3],length[3];Real margin=1e300;
      bool valid=true,aligned=true;
      for (int a=0;a<3;++a) {
        const Real t=n_*q[a]-map.node_offset[a];
        if (!std::isfinite(t)||n_*q[a]<Lower(gid,a)-1e-11
            ||n_*q[a]>Upper(gid,a)+1e-11) {valid=false;break;}
        const bool exact=spherical_&&std::abs(t-std::round(t))<=64*
            std::numeric_limits<Real>::epsilon()*std::max(Real(1),std::abs(t));
        length[a]=exact?1:width;aligned=aligned&&exact;
        first[a]=exact?int(std::round(t)):std::clamp(int(std::floor(t))-width/2+1,
            Lower(gid,a),Upper(gid,a)-width);
        if (first[a]<Lower(gid,a)||first[a]+length[a]>Upper(gid,a)) {valid=false;break;}
        margin=std::min(margin,std::min(t-Lower(gid,a),Upper(gid,a)-1-t));
      }
      if (!valid)continue;
      if (gid==preferred)margin+=1e6;
      if (aligned&&spherical_)margin+=1e9;
      if (margin>best_margin) {
        selected=gid;best_margin=margin;
        for (int a=0;a<3;++a) {best_first[a]=first[a];best_length[a]=length[a];best_q[a]=q[a];}
      }
    }
  }
  if (selected<0) throw std::runtime_error("no covered native evolved-overlap donor at ("
      +std::to_string(x[0])+","+std::to_string(x[1])+","+std::to_string(x[2])+")");
  return Tensor(selected,best_q,best_first,best_length);
}

std::vector<WeightedDonor> NativeDonorSelector::Shell(
    int owner,const Real x[3],const Real local[3],int width) const {
  const auto &block=maps_[owner];const Real radius=Radius(x);
  const Real angle[2]={block.lo[1]+local[1]*(block.hi[1]-block.lo[1]),
                       block.lo[2]+local[2]*(block.hi[2]-block.lo[2])};
  std::vector<int> chain;std::vector<Real> coverage(radii_.size(),0);
  for (int gid:order_) {
    const auto &map=maps_[gid];if (map.face!=block.face) continue;
    if (map.lo[1]==block.lo[1]&&map.hi[1]==block.hi[1]
        &&map.lo[2]==block.lo[2]&&map.hi[2]==block.hi[2]) {
      chain.push_back(gid);coverage[map.layer]+=map.hi[0]-map.lo[0];
    }
  }
  const bool mixed=std::any_of(coverage.begin(),coverage.end(),[](Real x) {
    return std::abs(x-1)>64*std::numeric_limits<Real>::epsilon();
  });
  std::vector<WeightedDonor> out;
  if (mixed) {
    // Each radial plane uses its own native angular leaf. This handles angular
    // and radial refinement without inventing a uniform shell-layer index.
    if (!spherical_) throw std::runtime_error("mixed angular cube-transition donors unsupported");
    struct Plane {Real radius;int gid,i;};std::vector<Plane> planes;
    for (int gid:order_) {
      const auto &map=maps_[gid];if (map.face!=block.face) continue;
      if (angle[0]<map.lo[1]-2e-13||angle[0]>map.hi[1]+2e-13
          ||angle[1]<map.lo[2]-2e-13||angle[1]>map.hi[2]+2e-13) continue;
      for (int i=0;i<n_;++i) {
        const Real s=map.lo[0]+(i+map.node_offset[0])/n_*(map.hi[0]-map.lo[0]);
        const Real r=map.inner_radius+s*(map.outer_radius-map.inner_radius);
        int selected=gid;
        for (int other:order_) {
          const auto &b=maps_[other];if (b.face!=block.face) continue;
          if (angle[0]<b.lo[1]-2e-13||angle[0]>b.hi[1]+2e-13
              ||angle[1]<b.lo[2]-2e-13||angle[1]>b.hi[2]+2e-13) continue;
          const Real radial=(r-b.inner_radius)/(b.outer_radius-b.inner_radius);
          if (radial<b.lo[0]-2e-13||radial>b.hi[0]+2e-13) continue;
          if (levels_[other]>levels_[selected]) selected=other;
        }
        if (selected==gid) planes.push_back({r,gid,i});
      }
    }
    std::stable_sort(planes.begin(),planes.end(),[](const Plane&a,const Plane&b) {
      return a.radius<b.radius;
    });
    planes.erase(std::unique(planes.begin(),planes.end(),[](const Plane&a,const Plane&b) {
      return a.radius==b.radius;
    }),planes.end());
    if (planes.size()<static_cast<std::size_t>(width)) throw std::runtime_error("short mapped radial chain");
    const auto pos=std::lower_bound(planes.begin(),planes.end(),radius,
                                   [](const Plane&a,Real r) {return a.radius<r;});
    const int begin=std::clamp(int(pos-planes.begin())-width/2,0,int(planes.size())-width);
    for (int t=0;t<width;++t) {
      const auto &plane=planes[begin+t];const auto &map=maps_[plane.gid];
      int first[2];Real weights[2][6]{};
      for (int a=0;a<2;++a) {
        const Real u=n_*(angle[a]-map.lo[a+1])/(map.hi[a+1]-map.lo[a+1])-.5;
        first[a]=std::clamp(int(std::floor(u))-width/2+1,0,n_-width);
        for (int j=0;j<width;++j) {weights[a][j]=1;for (int k=0;k<width;++k)
          if (k!=j) weights[a][j]*=(u-first[a]-k)/Real(j-k);}
      }
      Real radial=1;for (int k=0;k<width;++k) if (k!=t)
        radial*=(radius-planes[begin+k].radius)/(plane.radius-planes[begin+k].radius);
      for (int k=0;k<width;++k) for (int j=0;j<width;++j) {
        const Real weight=radial*weights[0][j]*weights[1][k];
        if (std::abs(weight)>1e-18) out.push_back({{plane.gid,k+first[1]+origin_,
            j+first[0]+origin_,plane.i+origin_},weight});
      }
    }
    return out;
  }
  int first[2];Real weights[2][6]{};
  for (int a=0;a<2;++a) {
    const Real t=n_*local[a+1]-.5;
    first[a]=std::clamp(int(std::floor(t))-width/2+1,0,n_-width);
    for (int i=0;i<width;++i) {weights[a][i]=1;for (int j=0;j<width;++j)
      if (i!=j) weights[a][i]*=(t-first[a]-j)/Real(i-j);}
  }
  struct Node {Real radius;bvals::DonorPoint point;};
  for (int k=0;k<width;++k) for (int j=0;j<width;++j) {
    const Real u=block.lo[1]+(first[0]+j+.5)/n_*(block.hi[1]-block.lo[1]);
    const Real v=block.lo[2]+(first[1]+k+.5)/n_*(block.hi[2]-block.lo[2]);
    std::vector<Node> nodes;
    for (int gid:chain) {
      const auto &map=maps_[gid];
      const Real lower=map.cube_transition ? half_width_*std::sqrt(1+u*u+v*v) :map.inner_radius;
      for (int i=0;i<n_;++i) {
        const Real s=map.lo[0]+(i+map.node_offset[0])/n_*(map.hi[0]-map.lo[0]);
        nodes.push_back({lower+s*(map.outer_radius-lower),
                         {gid,k+first[1]+origin_,j+first[0]+origin_,i+origin_}});
      }
    }
    std::sort(nodes.begin(),nodes.end(),[](const Node&a,const Node&b) {return a.radius<b.radius;});
    if (nodes.size()<static_cast<std::size_t>(width)) throw std::runtime_error("short mapped radial chain");
    auto pos=std::lower_bound(nodes.begin(),nodes.end(),radius,
                              [](const Node&a,Real r) {return a.radius<r;});
    const int begin=std::clamp(int(pos-nodes.begin())-width/2,0,int(nodes.size())-width);
    for (int i=0;i<width;++i) {
      Real weight=weights[0][j]*weights[1][k];
      for (int b=0;b<width;++b) if (i!=b)
        weight*=(radius-nodes[begin+b].radius)/(nodes[begin+i].radius-nodes[begin+b].radius);
      if (std::abs(weight)>1e-18) out.push_back({nodes[begin+i].point,weight});
    }
  }
  return out;
}

std::vector<WeightedDonor> NativeDonorSelector::Interpolate(const Real x[3],int width) const {
  if (width<1||width>6) throw std::invalid_argument("mapped interpolation width must be 1..6");
  for (int a=0;a<3;++a) if (!std::isfinite(x[a]))
    throw std::invalid_argument("mapped donor coordinates must be finite");
  auto core=Cartesian(x,width);if (!core.empty()) return core;
  return Spherical(x,width);
}

std::vector<WeightedDonor> NativeDonorSelector::Spherical(const Real x[3],int width,
                                                       bool force_shell) const {
  if (width<1||width>6) throw std::invalid_argument("mapped interpolation width must be 1..6");
  if (Radius(x)<first_radius_) return Overlap(x,width);
  const int owner=Owner(x,force_shell);
  if (owner<0) throw std::runtime_error("mapped interpolation outside scri");
  Real q[3];maps_[owner].Inverse(x,q);
  const int radial_begin=int(std::floor(n_*q[0]-maps_[owner].node_offset[0]))-width/2+1;
  const bool logical_inner=maps_[owner].layer==0&&radial_begin+width<=n_;
  if (maps_[owner].face>=0&&!logical_inner) return Shell(owner,x,q,width);
  int first[3],length[3]={width,width,width};
  for (int a=0;a<3;++a) first[a]=std::clamp(
      int(std::floor(n_*q[a]-maps_[owner].node_offset[a]))-width/2+1,0,n_-width);
  return Tensor(owner,q,first,length);
}

void MappedHalo::BuildRows(MeshBlockPack *pack,int origin,
                          std::vector<bvals::DonorPoint> &requests) {
  NativeDonorSelector selector(*pack->pmesh,origin,activity_);
  auto *mesh=pack->pmesh;const int ng=mesh->mb_indcs.ng,n=selector.Cells();
  if (ng!=4) throw std::invalid_argument("mapped halo requires four difference ghosts");
  for (int gid=0;gid<mesh->nmb_total;++gid) for (int a=0;a<3;++a) {
    if (origin<ng-selector.Lower(gid,a)||origin<ng+selector.Upper(gid,a)-n)
      throw std::invalid_argument("mapped halo origin does not include overlap and ghosts");
  }
  tensor_weights_.clear();
  rows_.clear();exterior_.clear();interior_points_.clear();local_points_.clear();
  remote_points_.clear();surface_points_.clear();
  const auto &offsets=BulkStencilOffsets();std::map<CellKey,int> dependencies;
  std::vector<HaloPoint> evolved;
  auto is_evolved=[&](HaloPoint c) {return selector.IsEvolved(
      {pack->GlobalId(c.m),c.k,c.j,c.i});};
  for (int m=0;m<pack->nmb_thispack;++m) {
    const int gid=pack->GlobalId(m);
    for (int k=origin+selector.Lower(gid,2);k<origin+selector.Upper(gid,2);++k)
      for (int j=origin+selector.Lower(gid,1);j<origin+selector.Upper(gid,1);++j)
        for (int i=origin+selector.Lower(gid,0);i<origin+selector.Upper(gid,0);++i) {
          if (!is_evolved({m,k,j,i})) continue;
          evolved.push_back({m,k,j,i});
          for (const auto &d:offsets) {
            HaloPoint c{m,k+d[2],j+d[1],i+d[0]};
            if (!is_evolved(c)) dependencies.emplace(Key(c),0);
          }
        }
  }
  for (auto &[key,dependency]:dependencies) {
    const auto c=Point(key);const int gid=pack->GlobalId(c.m);const auto &map=selector.Map(gid);
    const Real q[3]={(c.i-origin+map.node_offset[0])/n,
                      (c.j-origin+map.node_offset[1])/n,
                      (c.k-origin+map.node_offset[2])/n};
    Real x[3];map.Forward(q,x);const Real radius=Radius(x);
    if (radius>=mesh->atlas->RadialKnots().back()*(1-2e-14)) {
      exterior_.push_back({c,{x[0],x[1],x[2]}});dependency=4;continue;
    }
    Row row;row.target=c;
    // Coalesce repeated donor coordinates in stable native GID/index order.
    // All retained rows read evolved state; ghosts are never interpolation donors.
    std::map<bvals::DonorPoint,Real> weights;
    for (const auto &d:selector.Interpolate(x)) weights[d.point]+=d.weight;
    for (const auto &[point,weight]:weights) if (weight!=0) {
      if (!selector.IsEvolved(point)) throw std::runtime_error("unevolved mapped halo donor");
      row.donors.push_back({point,weight});requests.push_back(point);
      row.remote=row.remote||mesh->rank_eachmb[point.gid]!=global_variable::my_rank;
    }
    if (row.donors.empty()) throw std::runtime_error("empty mapped halo row");
    AxisWeights axes{};
    if (TensorAxes(row, selector, x, axes)) tensor_weights_.emplace(rows_.size(), axes);
    dependency=row.remote?2:1;rows_.push_back(std::move(row));
  }
  for (auto c:evolved) {
    int dependency=0;
    for (const auto &d:offsets) {
      const auto found=dependencies.find({c.m,c.k+d[2],c.j+d[1],c.i+d[0]});
      if (found!=dependencies.end()) dependency|=found->second;
    }
    if (dependency&4) surface_points_.push_back(c);
    else if (dependency&2) remote_points_.push_back(c);
    else if (dependency&1) local_points_.push_back(c);
    else interior_points_.push_back(c);
  }
  if (requests.size()>std::size_t(std::numeric_limits<int>::max())
      ||rows_.size()>std::size_t(std::numeric_limits<int>::max()/Fields::count))
    throw std::overflow_error("mapped halo plan exceeds device integer indexing");
}

void MappedHalo::Rebuild(MeshBlockPack *pack,int origin,ActivityPolicy activity) {
  activity_=activity;
  ready_=false;
  std::vector<bvals::DonorPoint> requests;
  std::string error;
  try {
    if (pack==nullptr || pack->pmesh==nullptr)
      throw std::invalid_argument("mapped halo requires a native MeshBlockPack");
    BuildRows(pack,origin,requests);
  } catch (const std::exception &e) {error=e.what();}
  int failed=!error.empty();
#if MPI_PARALLEL_ENABLED
  int any_failed=0;
  MPI_Allreduce(&failed,&any_failed,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
  failed=any_failed;
#endif
  // Geometry construction may fail on only one ownership partition. Fail all
  // ranks before entering the transport's collective metadata rebuild.
  if (failed) throw std::runtime_error(error.empty()
      ? "mapped halo geometry failed on another rank" : error);
  std::vector<int> fields;
  for (int f=0;f<Fields::count;++f) if (Fields::Independent(f)) fields.push_back(f);
  residual_exchange_.Rebuild(pack,requests,fields);
  std::vector<int> physical_fields(Fields::count);std::iota(physical_fields.begin(),physical_fields.end(),0);
  physical_exchange_.Rebuild(pack,requests,physical_fields);
  if (residual_exchange_.RequestOrdinals()!=physical_exchange_.RequestOrdinals())
    throw std::logic_error("mapped residual/physical donor ordinals disagree");
  std::vector<DeviceRow> local_rows,remote_rows;std::vector<DeviceEntry> entries;
  const auto &ordinals=residual_exchange_.RequestOrdinals();std::size_t request=0;
  // BuildRows visits targets in (local slot,k,j,i) order. Both exchange
  // ordinals and interpolation weights use this same flat entry order.
  slot_offsets_.assign(pack->nmb_thispack+1,{});
  batches_.clear();
  line_values_ = DvceArray2D<Real>();
  all_slots_.resize(pack->nmb_thispack);
  std::iota(all_slots_.begin(), all_slots_.end(), 0);
  for (const auto &row:rows_) {
    DeviceRow d{row.target,int(entries.size()),0};
    for (const auto &entry:row.donors) entries.push_back({ordinals.at(request++),entry.weight});
    d.end=entries.size();(row.remote?remote_rows:local_rows).push_back(d);
    auto &offset=slot_offsets_[row.target.m+1];
    ++offset.row;
    offset.entry+=d.end-d.begin;
  }
  for (std::size_t i=1;i<slot_offsets_.size();++i) {
    slot_offsets_[i].row+=slot_offsets_[i-1].row;
    slot_offsets_[i].entry+=slot_offsets_[i-1].entry;
  }
  CopyDevice(local_rows,local_rows_,"mapped_local_rows");
  CopyDevice(remote_rows,remote_rows_,"mapped_remote_rows");
  CopyDevice(entries,entries_,"mapped_halo_entries");
  packed_=received_=false;
  ready_=true;
}

TaskStatus MappedHalo::StartReceives() {
  if (!ready_) throw std::logic_error("mapped halo plan is not valid");
  packed_=received_=false;
  residual_exchange_.StartReceives();physical_exchange_.StartReceives();
  return TaskStatus::complete;
}
TaskStatus MappedHalo::PackAndSend(const DvceArray5D<Real> &residual,
                                  const DvceArray5D<Real> &physical) {
  residual_exchange_.PackAndSend(residual);physical_exchange_.PackAndSend(physical);
  packed_=true;return TaskStatus::complete;
}
TaskStatus MappedHalo::Receive() {
  const auto residual=residual_exchange_.Receive(),physical=physical_exchange_.Receive();
  if (residual!=TaskStatus::complete||physical!=TaskStatus::complete) return TaskStatus::incomplete;
  received_=true;return TaskStatus::complete;
}
void MappedHalo::Fill(const DvceArray1D<DeviceRow> &rows,
                       const DvceArray5D<Real> &residual,
                       const DvceArray5D<Real> &physical,
                       bvals::MappedExchange::ConstValues residual_values,
                       bvals::MappedExchange::ConstValues physical_values,
                       DvceArray1D<int> active) const {
  const auto entries=entries_;
  if(!residual_values.data())residual_values=residual_exchange_.Values();
  if(!physical_values.data())physical_values=physical_exchange_.Values();
  const int count=rows.extent_int(0);
  Kokkos::parallel_for("mapped halo interpolation",Kokkos::RangePolicy<>(0,count*Fields::count),
    KOKKOS_LAMBDA(int index) {
      const auto row=rows(index/Fields::count);const int f=index%Fields::count;
      const auto c=row.target; if(active.extent(0) && !active(c.m))return;
      const int first=entries(row.begin).ordinal;
      const Real anchor=physical_values(first,f);
      const bool independent=Fields::Independent(f);
      const int slot=Fields::IndependentSlot(f);
      const Real base=independent ? residual_values(first,slot) : 0;
      Real value=anchor,interpolated=base;
      for (int e=row.begin;e<row.end;++e) {
        const auto entry=entries(e);
        value+=entry.weight*(physical_values(entry.ordinal,f)-anchor);
        if (independent) interpolated+=entry.weight*(residual_values(entry.ordinal,slot)-base);
      }
      physical(c.m,f,c.k,c.j,c.i)=value;
      residual(c.m,f,c.k,c.j,c.i)=interpolated;
    });
}
// Keep complete recipient blocks together. All uses of a line in this plan
// see the same stage donor values. A topology rebuild discards the plan.
bool MappedHalo::AppendTensorBlock(const SlotOffset &begin, const SlotOffset &end,
                                   std::vector<TensorLine> &lines,
                                   std::vector<TensorRow> &tensors) const {
  const auto &ordinals = residual_exchange_.RequestOrdinals();
  std::map<std::pair<int, std::array<Real, 6>>, int> indices;
  const size_t first_line = lines.size(), first_tensor = tensors.size();
  int entry = begin.entry;
  for (int r = begin.row; r < end.row; ++r) {
    const auto &row = rows_[r];
    const int offset = entry;
    entry += static_cast<int>(row.donors.size());
    const auto found = tensor_weights_.find(r);
    if (found == tensor_weights_.end()) continue;
    const auto &w = found->second;
    TensorRow result{};
    result.target = row.target;
    for (int i = 0; i < 6; ++i) {
      result.y[i] = w[1][i];
      result.z[i] = w[2][i];
      result.sum_x += w[0][i];
    }
    for (int l = 0; l < 36; ++l) {
      const int e = offset + 6 * l;
      // The verified line starts here and covers the next five native i coordinates.
      const auto key = std::make_pair(ordinals[e], w[0]);
      const auto [it, fresh] = indices.try_emplace(key, static_cast<int>(lines.size()));
      if (fresh) {
        TensorLine line{};
        line.block = row.target.m;
        for (int i = 0; i < 6; ++i) {
          line.ordinal[i] = ordinals[e + i];
          line.weight[i] = w[0][i];
        }
        lines.push_back(line);
      }
      result.line[l] = it->second;
    }
    tensors.push_back(result);
  }
  const size_t count = tensors.size() - first_tensor;
  if (count && 4 * (lines.size() - first_line) <= 36 * count) return true;
  lines.resize(first_line);
  tensors.resize(first_tensor);
  return false;
}

void MappedHalo::FillTensors(const BatchPlan &plan, const DvceArray5D<Real> &residual,
                             const DvceArray5D<Real> &physical,
                             bvals::MappedExchange::ConstValues rv,
                             bvals::MappedExchange::ConstValues pv,
                             DvceArray1D<int> active) const {
  const auto lines = plan.lines;
  const auto rows = plan.tensors;
  if (rows.extent(0) == 0) return;
  if (!rv.data()) rv = residual_exchange_.Values();
  if (!pv.data()) pv = physical_exchange_.Values();
  const auto sums = line_values_;
  constexpr int fields = Fields::count;
  using Policy = Kokkos::RangePolicy<Kokkos::IndexType<std::int64_t>>;
  Kokkos::parallel_for(
      "mapped halo first-axis interpolation", Policy(0, lines.extent(0) * fields),
      KOKKOS_LAMBDA(std::int64_t index) {
        const int l = index / fields, f = index % fields;
        const auto &line = lines(l);
        if (active.extent(0) && !active(line.block)) return;
        const int first = line.ordinal[0], slot = Fields::IndependentSlot(f);
        const bool independent = Fields::Independent(f);
        const Real anchor = pv(first, f), base = independent ? rv(first, slot) : 0;
        Real value = 0, delta = 0;
        for (int i = 0; i < 6; ++i) {
          const int q = line.ordinal[i];
          value += line.weight[i] * (pv(q, f) - anchor);
          if (independent) delta += line.weight[i] * (rv(q, slot) - base);
        }
        sums(l, f) = value;
        if (independent) sums(l, fields + slot) = delta;
      });
  Kokkos::parallel_for(
      "mapped halo tensor interpolation", Policy(0, rows.extent(0) * fields),
      KOKKOS_LAMBDA(std::int64_t index) {
        const auto &row = rows(index / fields);
        if (active.extent(0) && !active(row.target.m)) return;
        const int f = index % fields, slot = Fields::IndependentSlot(f);
        const bool independent = Fields::Independent(f);
        const int first = lines(row.line[0]).ordinal[0];
        const Real anchor = pv(first, f), base = independent ? rv(first, slot) : 0;
        Real value = anchor, delta = base;
        for (int k = 0; k < 6; ++k)
          for (int j = 0; j < 6; ++j) {
            const int l = row.line[6 * k + j], q = lines(l).ordinal[0];
            const Real weight = row.y[j] * row.z[k];
            value += weight * (sums(l, f) + row.sum_x * (pv(q, f) - anchor));
            if (independent)
              delta +=
                  weight * (sums(l, fields + slot) + row.sum_x * (rv(q, slot) - base));
          }
        const auto c = row.target;
        physical(c.m, f, c.k, c.j, c.i) = value;
        residual(c.m, f, c.k, c.j, c.i) = delta;
      });
}
const MappedHalo::BatchPlan &MappedHalo::PlanForBatch(const std::vector<int> &slots) {
  const auto found = batches_.find(slots);
  if (found != batches_.end()) return found->second;
  BatchPlan plan;
  std::vector<DeviceRow> selected;
  std::vector<TensorLine> lines;
  std::vector<TensorRow> tensors;
  const auto &ordinals=residual_exchange_.RequestOrdinals();
  std::vector<bool> used(residual_exchange_.UniqueDonors().size());
  for (int m:slots) {
    const auto begin=slot_offsets_.at(m),end=slot_offsets_.at(m+1);
    const bool factored = AppendTensorBlock(begin, end, lines, tensors);
    int entry=begin.entry;
    for (int r=begin.row;r<end.row;++r) {
      const auto &row=rows_[r];
      const int next=entry+static_cast<int>(row.donors.size());
      if (!factored || tensor_weights_.find(r) == tensor_weights_.end())
        selected.push_back({row.target,entry,next});
      entry=next;
    }
    for (int e=begin.entry;e<end.entry;++e) used[ordinals[e]]=true;
  }
  for (std::size_t q=0;q<used.size();++q)
    if (used[q]) plan.donor_ordinals.push_back(q);
  CopyDevice(lines, plan.lines, "mapped halo tensor donor lines");
  CopyDevice(tensors, plan.tensors, "mapped halo tensor targets");
  if (lines.size() > line_values_.extent(0)) {
    if (line_values_.data()) Kokkos::fence("mapped halo scratch resize");
    line_values_ = DvceArray2D<Real>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "mapped halo stage line values"),
        lines.size(), Fields::count + Fields::independent_count);
  }
  CopyDevice(selected, plan.rows, "mapped halo active recipient rows");
  return batches_.emplace(slots,std::move(plan)).first->second;
}

const std::vector<int> &MappedHalo::DonorOrdinalsForBatch(const std::vector<int> &slots) {
  return PlanForBatch(slots).donor_ordinals;
}

void MappedHalo::FillExternal(const DvceArray5D<Real> &residual,
    const DvceArray5D<Real> &physical,
    bvals::MappedExchange::ConstValues rv,
    bvals::MappedExchange::ConstValues pv,DvceArray1D<int> active,
    const std::vector<int> *fixed_slots) {
  const auto &plan = PlanForBatch(fixed_slots ? *fixed_slots : all_slots_);
  if (fixed_slots) active = DvceArray1D<int>();
  Fill(plan.rows,residual,physical,rv,pv,active);
  FillTensors(plan,residual,physical,rv,pv,active);
}
TaskStatus MappedHalo::FillLocal(const DvceArray5D<Real> &residual,
                                 const DvceArray5D<Real> &physical) {
  if (!packed_) throw std::logic_error("mapped local fill precedes donor packing");
  Fill(local_rows_,residual,physical);return TaskStatus::complete;
}
TaskStatus MappedHalo::FillRemote(const DvceArray5D<Real> &residual,
                                  const DvceArray5D<Real> &physical) {
  if (!received_) throw std::logic_error("mapped remote fill precedes receive completion");
  Fill(remote_rows_,residual,physical);return TaskStatus::complete;
}
TaskStatus MappedHalo::ClearSend() {
  const auto a=residual_exchange_.ClearSend(),b=physical_exchange_.ClearSend();
  return a==TaskStatus::complete&&b==TaskStatus::complete?TaskStatus::complete:TaskStatus::incomplete;
}
TaskStatus MappedHalo::ClearRecv() {
  const auto a=residual_exchange_.ClearRecv(),b=physical_exchange_.ClearRecv();
  return a==TaskStatus::complete&&b==TaskStatus::complete?TaskStatus::complete:TaskStatus::incomplete;
}
}  // namespace z4c_hyperboloidal
