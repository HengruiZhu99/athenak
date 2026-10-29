// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef MESH_CUBED_SPHERE_COMPOSITE_DIFFERENCE_HPP_
#define MESH_CUBED_SPHERE_COMPOSITE_DIFFERENCE_HPP_

#include <functional>
#include <map>
#include <utility>
#include "mesh/cubed_sphere/composite.hpp"

namespace cubed_sphere {

// Bounded CPU reference implementation. Plans read active leaf values only,
// including corner/edge halos; no ordering dependency between ghost exchanges.
// The descriptor and exact physics kernels are independent of this host backend.
class CompositeDifference {
 public:
  struct Entry {int point; Real weight;};
  struct Row {
    int target;
    std::vector<Entry> entries;
    Real boundary_weight = 0;
    std::array<Real, 3> boundary{};
  };
  CompositeMesh mesh;
  int n, ng, padded, count, order;
  std::vector<int> active;
  std::vector<Point> geometry;
  std::vector<Real> volumes;
  std::vector<Row> halos;
  std::vector<Row> extrapolated_halos;
  bool radial_interface_stencils = false;

  CompositeDifference(CompositeMesh descriptor, int derivative_order = 4,bool radial_stencils=false)
      : mesh(std::move(descriptor)), n(mesh.cells), ng(mesh.ghosts),
        padded(n+2*ng), count(static_cast<int>(mesh.blocks.size())*padded*padded*padded),
        order(derivative_order),radial_interface_stencils(radial_stencils) {
    if ((order != 2 && order != 4) || ng < order/2 || n < order)
      throw std::invalid_argument("composite derivative order/halo contract");
    for (int m = 0; m < static_cast<int>(mesh.blocks.size()); ++m)
    for (int k = 0; k < n; ++k)
    for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const Real q[3] = {(i+.5)/n, (j+.5)/n, (k+.5)/n};
      active.push_back(Index(m, i, j, k));
      geometry.push_back(mesh.Geometry(m, q));
      volumes.push_back(PhysicalCellVolume(geometry.back(), n));
    }
    BuildHalos();
  }
  int Index(int m, int i, int j, int k) const {
    return ((m*padded+k+ng)*padded+j+ng)*padded+i+ng;
  }
  int Offset(int a, int amount) const {
    return amount*(a == 0 ? 1 : (a == 1 ? padded : padded*padded));
  }
  std::vector<Entry> Interpolate(const Real x[3], int width) const {
    const int owner = mesh.Owner(x);
    if (owner < 0) throw std::runtime_error("interpolation outside scri");
    Real q[3]; mesh.Inverse(owner, x, q);
    if(radial_interface_stencils&&mesh.blocks[owner].face>=0)
      return InterpolateShellRadius(owner,x,q,width);
    std::vector<Entry> entries;
    int first[3];
    Real w[3][6]{};
    width = std::min(width, n);
    for (int a = 0; a < 3; ++a) {
      const Real t = n*q[a]-.5;
      first[a] = std::clamp(static_cast<int>(std::floor(t))-width/2+1, 0, n-width);
      for (int i = 0; i < width; ++i) {
        w[a][i] = 1;
        for (int j = 0; j < width; ++j) if (j != i)
          w[a][i] *= (t-first[a]-j)/Real(i-j);
      }
    }
    for (int k = 0; k < width; ++k)
    for (int j = 0; j < width; ++j)
    for (int i = 0; i < width; ++i)
      if (std::abs(w[0][i]*w[1][j]*w[2][k]) > 1e-18)
        entries.push_back({Index(owner, first[0]+i, first[1]+j, first[2]+k),
                           w[0][i]*w[1][j]*w[2][k]});
    return entries;
  }
  // A radial shell seam is an interior interpolation point, not an artificial
  // boundary of a one-sided donor stencil. Keep Cartesian field components and
  // use physical radius along each angular donor line, including transition-map
  // radius dependence. Every donor is an active cell; no ghost cycles enter.
  void SetRadialInterfaceStencils(bool enabled){
    if(enabled==radial_interface_stencils)return;radial_interface_stencils=enabled;BuildHalos();
  }
  void Fill(std::vector<Real> &field, int components,
      const std::function<Real(int, const std::array<Real, 3> &)> &boundary = {}, int first_component=0, int last_component=-1) const {
    if (field.size() != static_cast<size_t>(components*count))
      throw std::invalid_argument("composite field storage shape");
    if(last_component<0)last_component=components;
    for (const auto &row : halos) for (int v = first_component; v < last_component; ++v) {
      const int base = v*count;
      const Real anchor = field[base+row.entries[0].point];
      Real value = anchor;
      for (const auto &entry : row.entries)
        value += entry.weight*(field[base+entry.point]-anchor);
      if (row.boundary_weight != 0)
        value += row.boundary_weight*((boundary ? boundary(v, row.boundary) : 0)-anchor);
      field[base+row.target] = value;
    }
  }
  // Research hyperbolic closure: active-only polynomial extrapolation at scri.
  // This is an experimental outflow closure, not an admitted regularity theorem.
  void FillExtrapolated(std::vector<Real> &field, int components) const {
    for (const auto &row : extrapolated_halos) for (int v = 0; v < components; ++v) {
      const int base = v*count;
      const Real anchor = field[base+row.entries[0].point];
      Real value = anchor;
      for (const auto &entry : row.entries)
        value += entry.weight*(field[base+entry.point]-anchor);
      field[base+row.target] = value;
    }
  }
  // One-sided tensor-product polynomial jets on the physical boundary. This
  // independently measures pole/null/clock defects without using solver ghosts.
  void SampleJets(const std::vector<Real> &field, int components, const Real x[3],
      std::vector<Real> &values, std::vector<std::array<Real, 3>> &first,
      std::vector<std::array<std::array<Real, 3>, 3>> &second, bool use_halos=false) const {
    const int m = mesh.Owner(x), width = std::min(n, 6);
    if (m < 0) throw std::invalid_argument("sample outside composite domain");
    Real q[3]; mesh.Inverse(m, x, q);
    int begin[3]; Jet weights[3][6];
    for (int a = 0; a < 3; ++a) {
      const Real t = n*q[a]-.5;
      begin[a] = std::clamp(static_cast<int>(std::floor(t))-width/2+1, use_halos ? -ng : 0, use_halos ? n+ng-width : n-width);
      for (int i = 0; i < width; ++i) {
        weights[a][i] = Jet(1);
        for (int j = 0; j < width; ++j) if (i != j)
          weights[a][i] = weights[a][i]*(Jet::Variable(t, a)-Jet(begin[a]+j))/Jet(i-j);
      }
    }
    values.assign(components, 0); first.assign(components, {}); second.assign(components, {});
    std::vector<Jet> jets(components);
    for (int k = 0; k < width; ++k)
    for (int j = 0; j < width; ++j)
    for (int i = 0; i < width; ++i) {
      const Jet w = weights[0][i]*weights[1][j]*weights[2][k];
      const int point = Index(m, begin[0]+i, begin[1]+j, begin[2]+k);
      for (int v = 0; v < components; ++v) jets[v] = jets[v]+w*Jet(field[v*count+point]);
    }
    const auto p = mesh.Geometry(m, q);
    for (int v = 0; v < components; ++v) {
      values[v] = jets[v].f;
      for (int a = 0; a < 3; ++a)
      for (int A = 0; A < 3; ++A) {
        first[v][a] += p.J[A][a]*n*jets[v].d[A];
        for (int b = 0; b < 3; ++b) {
          second[v][a][b] += p.H[A][a][b]*n*jets[v].d[A];
          for (int B = 0; B < 3; ++B)
            second[v][a][b] += p.J[A][a]*p.J[B][b]*n*n*jets[v].h[A][B];
        }
      }
    }
  }
  Real First(const std::vector<Real> &field, int point, int a) const {
    if (order == 2) return n*(field[point+Offset(a, 1)]-field[point+Offset(a, -1)])/2;
    return n*(field[point+Offset(a, -2)]-8*field[point+Offset(a, -1)]
        +8*field[point+Offset(a, 1)]-field[point+Offset(a, 2)])/12;
  }
  Real Second(const std::vector<Real> &field, int point, int a, int b) const {
    if (a == b) {
      if (order == 2) return n*n*(field[point+Offset(a, 1)]-2*field[point]
                                            +field[point+Offset(a, -1)]);
      return n*n*(-field[point+Offset(a, -2)]+16*field[point+Offset(a, -1)]
          -30*field[point]+16*field[point+Offset(a, 1)]-field[point+Offset(a, 2)])/12;
    }
    const int radius = order/2;
    Real result = 0;
    for (int i = -radius; i <= radius; ++i) if (i)
    for (int j = -radius; j <= radius; ++j) if (j) {
      const Real wi = order == 2 ? Real(i)/2 : (std::abs(i) == 1 ? Real(2*i)/3 : Real(-i)/24);
      const Real wj = order == 2 ? Real(j)/2 : (std::abs(j) == 1 ? Real(2*j)/3 : Real(-j)/24);
      result += wi*wj*field[point+Offset(a, i)+Offset(b, j)];
    }
    return result*n*n;
  }
  // Cartesian derivative directions, retaining the full second chain rule.
  void Derivatives(const std::vector<Real> &field, int component, size_t p,
                    Real d[3], Real dd[3][3], bool need_second = true) const {
    const int point = component*count+active[p];
    Real first[3], second[3][3];
    for (int a = 0; a < 3; ++a) {
      first[a] = First(field, point, a);
      for (int b = 0; b < 3; ++b) second[a][b] = need_second ? Second(field, point, a, b) : 0;
    }
    const auto &g = geometry[p];
    for (int a = 0; a < 3; ++a) {
      d[a] = 0;
      for (int b = 0; b < 3; ++b) dd[a][b] = 0;
      for (int A = 0; A < 3; ++A) {
        d[a] += g.J[A][a]*first[A];
        for (int b = 0; b < 3; ++b) {
          if (!need_second) continue;
          dd[a][b] += g.H[A][a][b]*first[A];
          for (int B = 0; B < 3; ++B)
            dd[a][b] += g.J[A][a]*g.J[B][b]*second[A][B];
        }
      }
    }
  }

 private:
  std::vector<Entry> InterpolateShellRadius(int owner,const Real x[3],const Real local[3],int width) const {
    const auto &block=mesh.blocks[owner];width=std::min(width,n);
    std::vector<int> chain;
    for(size_t m=0;m<mesh.blocks.size();++m){const auto &b=mesh.blocks[m];
      if(b.face==block.face&&b.lo[1]==block.lo[1]&&b.hi[1]==block.hi[1]&&b.lo[2]==block.lo[2]&&b.hi[2]==block.hi[2])chain.push_back(m);}
    const Real radius=std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
    int first[2];Real weights[2][6]{};
    for(int a=0;a<2;++a){Real t=n*local[a+1]-.5;first[a]=std::clamp(int(std::floor(t))-width/2+1,0,n-width);
      for(int i=0;i<width;++i){weights[a][i]=1;for(int j=0;j<width;++j)if(i!=j)weights[a][i]*=(t-first[a]-j)/Real(i-j);}}
    std::vector<Entry> out;
    struct Node{Real radius;int point;};
    for(int k=0;k<width;++k)for(int j=0;j<width;++j){
      const Real u=block.lo[1]+(first[0]+j+.5)/n*(block.hi[1]-block.lo[1]);
      const Real v=block.lo[2]+(first[1]+k+.5)/n*(block.hi[2]-block.lo[2]);
      std::vector<Node> nodes;
      for(int m:chain){const auto &b=mesh.blocks[m];const Real lower=b.layer==0?mesh.half_width*std::sqrt(1+u*u+v*v):mesh.radii[b.layer-1];
        const Real upper=mesh.radii[b.layer];
        for(int i=0;i<n;++i){const Real s=b.lo[0]+(i+.5)/n*(b.hi[0]-b.lo[0]);nodes.push_back({lower+s*(upper-lower),Index(m,i,first[0]+j,first[1]+k)});}}
      std::sort(nodes.begin(),nodes.end(),[](const Node&a,const Node&b){return a.radius<b.radius;});
      auto pos=std::lower_bound(nodes.begin(),nodes.end(),radius,[](const Node&a,Real r){return a.radius<r;});
      const int begin=std::clamp(int(pos-nodes.begin())-width/2,0,int(nodes.size())-width);
      for(int i=0;i<width;++i){Real weight=weights[0][j]*weights[1][k];for(int b=0;b<width;++b)if(i!=b)
          weight*=(radius-nodes[begin+b].radius)/(nodes[begin+i].radius-nodes[begin+b].radius);
        if(std::abs(weight)>1e-18)out.push_back({nodes[begin+i].point,weight});}
    }
    return out;
  }
  void BuildHalos() {
    halos.clear();extrapolated_halos.clear();
    for (int m = 0; m < static_cast<int>(mesh.blocks.size()); ++m)
    for (int k = -ng; k < n+ng; ++k)
    for (int j = -ng; j < n+ng; ++j)
    for (int i = -ng; i < n+ng; ++i) {
      if (i >= 0 && i < n && j >= 0 && j < n && k >= 0 && k < n) continue;
      const Real q[3] = {(i+.5)/n, (j+.5)/n, (k+.5)/n};
      Real x[3]; mesh.Forward(m, q, x);
      Row row; row.target = Index(m, i, j, k);
      Row extrapolated; extrapolated.target = row.target;
      const Real radius = std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
      const int width = std::min(n, order+2);
      if (radius <= mesh.radii.back()) {
        row.entries = Interpolate(x, width);
        extrapolated.entries = row.entries;
      }
      else {
        const Real R = mesh.radii.back();
        for (int a = 0; a < 3; ++a) row.boundary[a] = R*x[a]/radius;
        // Coarse Cartesian corner halos can cross the entire thin transition
        // layer. Classify by physical position, never by the target block kind.
        const Real maxcoord = std::max({std::abs(x[0]), std::abs(x[1]), std::abs(x[2])});
        const Real inner = mesh.radii.size() == 1 ? mesh.half_width*radius/maxcoord
                                                : mesh.radii[mesh.radii.size()-2];
        const Real dr = (R-inner)/n;
        if (!(dr > 0)) throw std::runtime_error("outer halo map folds");
        const Real t = (radius-R)/dr;
        // Polynomial through the physical boundary and `order` interior nodes.
        // Boundary value is supplied explicitly; homogeneous corrections use 0.
        row.boundary_weight = 1;
        for (int a = 0; a < width; ++a)
          row.boundary_weight *= (t+a+.5)/(a+.5);
        for (int a = 0; a < width; ++a) {
          const Real node = -a-.5;
          Real weight = t/node;
          for (int b = 0; b < width; ++b) if (b != a)
            weight *= (t+b+.5)/(node+b+.5);
          Real sample[3];
          for (int b = 0; b < 3; ++b) sample[b] = (R+node*dr)*x[b]/radius;
          Real outflow_weight = 1;
          for (int b = 0; b < width; ++b) if (b != a)
            outflow_weight *= (t+b+.5)/(node+b+.5);
          for (auto entry : Interpolate(sample, width)) {
            extrapolated.entries.push_back({entry.point, entry.weight*outflow_weight});
            entry.weight *= weight;
            row.entries.push_back(entry);
          }
        }
      }
      // Collapse repeated active donors generated by radial extrapolation.
      // This changes only summation order, not the interpolation polynomial.
      auto coalesce=[](Row&r){std::map<int,Real> weights;
        for(const auto&e:r.entries)weights[e.point]+=e.weight;
        r.entries.clear();for(const auto&e:weights)if(e.second!=0)r.entries.push_back({e.first,e.second});};
      coalesce(row);coalesce(extrapolated);
      halos.push_back(std::move(row));
      extrapolated_halos.push_back(std::move(extrapolated));
    }
  }
};

}  // namespace cubed_sphere
#endif  // MESH_CUBED_SPHERE_COMPOSITE_DIFFERENCE_HPP_
