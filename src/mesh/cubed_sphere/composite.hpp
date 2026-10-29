// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef MESH_CUBED_SPHERE_COMPOSITE_HPP_
#define MESH_CUBED_SPHERE_COMPOSITE_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>
#include "mesh/cubed_sphere/map.hpp"

namespace cubed_sphere {

// Geometry/topology descriptor independent of field storage or physics. Every
// block uses the SAME local [0,1]^3 active shape and halo contract. The first
// layer joins a cube to a sphere; subsequent layers join concentric spheres.
struct CompositeBlock {
  int face = -1, layer = -1, refinement = 0;
  Real lo[3]{}, hi[3]{};
};

class CompositeMesh {
 public:
  Real half_width;
  std::vector<Real> radii;
  int face_subdivisions, cells, ghosts;
  std::vector<CompositeBlock> blocks;
  unsigned long revision = 0;
  // Optional research overlap: nominal ownership and maps remain unchanged.
  // The Cartesian core and cube-to-sphere layer evolve this many extra cells
  // on each side; interpolation ghosts lie outside that evolved region.
  int evolved_overlap = 0;
  // Experimental core-only extension. -1 preserves every legacy descriptor.
  // Shell angular/radial overlap stays evolved_overlap; geometric overlap of
  // the cube and the spherical annulus is independent of either layer count.
  int core_evolved_overlap = -1;
  bool shifted_overlap_donors = false;
  int CoreOverlap() const {
    return core_evolved_overlap < 0 ? evolved_overlap : core_evolved_overlap;
  }
  // Optional overlapping spherical annulus. Angles are equiangular face
  // coordinates; tensor components remain Cartesian. The core covers r<inner_radius.
  bool spherical_inner_boundary = false, equiangular = false;
  Real inner_radius = 0;
  bool continuous_cartesian_core = false;
  Real AngularRatio(Real q) const {return equiangular?std::tan(std::acos(-1.)*q/4):q;}


  CompositeMesh(Real width, std::vector<Real> shell_radii, int subdivisions,
                int active_cells, int halo = 2, int bulk_subdivisions = 1)
      : half_width(width), radii(std::move(shell_radii)),
        face_subdivisions(subdivisions), cells(active_cells), ghosts(halo) {
    if (!(width > 0) || radii.empty() || !(radii[0] > std::sqrt(3.)*width)
        || subdivisions < 1 || active_cells < 2 || halo < 1 || bulk_subdivisions < 1)
      throw std::invalid_argument("invalid composite block geometry/shape");
    for (size_t i = 0; i < radii.size(); ++i)
      if (!std::isfinite(radii[i]) || (i && !(radii[i] > radii[i-1])))
        throw std::invalid_argument("shell radii must be finite and increasing");
    for (int k = 0; k < bulk_subdivisions; ++k)
    for (int j = 0; j < bulk_subdivisions; ++j)
    for (int i = 0; i < bulk_subdivisions; ++i) {
      CompositeBlock block;
      const int index[3] = {i, j, k};
      for (int a = 0; a < 3; ++a) {
        block.lo[a] = -width+2*width*index[a]/bulk_subdivisions;
        block.hi[a] = -width+2*width*(index[a]+1)/bulk_subdivisions;
      }
      blocks.push_back(block);
    }
    for (int layer = 0; layer < static_cast<int>(radii.size()); ++layer)
    for (int face = 0; face < 6; ++face)
    for (int v = 0; v < subdivisions; ++v)
    for (int u = 0; u < subdivisions; ++u) {
      CompositeBlock block;
      block.face = face; block.layer = layer;
      block.lo[0] = 0; block.hi[0] = 1;
      block.lo[1] = -1+Real(2*u)/subdivisions;
      block.hi[1] = -1+Real(2*(u+1))/subdivisions;
      block.lo[2] = -1+Real(2*v)/subdivisions;
      block.hi[2] = -1+Real(2*(v+1))/subdivisions;
      blocks.push_back(block);
    }
  }

  void Forward(int id, const Real local[3], Real x[3]) const {
    const auto &b = blocks.at(id);
    Real q[3];
    for (int a = 0; a < 3; ++a) q[a] = b.lo[a]+local[a]*(b.hi[a]-b.lo[a]);
    if (b.face < 0) {for (int a = 0; a < 3; ++a) x[a] = q[a]; return;}
    if (b.layer == 0 && !spherical_inner_boundary) {Map{half_width, radii[0], 0}.Forward(b.face, q, x); return;}
    const int a = b.face/2, c = (a+1)%3, d = (a+2)%3;
    q[1]=AngularRatio(q[1]);q[2]=AngularRatio(q[2]);
    const Real lower=b.layer==0?inner_radius:radii[b.layer-1];
    const Real radius = lower+q[0]*(radii[b.layer]-lower);
    const Real scale = radius/std::sqrt(1+q[1]*q[1]+q[2]*q[2]);
    x[a] = b.face%2 ? -scale : scale;
    x[c] = scale*q[1]; x[d] = scale*q[2];
  }

  void Inverse(int id, const Real x[3], Real local[3]) const {
    const auto &b = blocks.at(id);
    Real q[3];
    if (b.face < 0) {for (int a = 0; a < 3; ++a) q[a] = x[a];}
    else if (b.layer == 0 && !spherical_inner_boundary) Map{half_width, radii[0], 0}.Inverse(b.face, x, q);
    else {
      const int a = b.face/2;
      const Real scale = b.face%2 ? -x[a] : x[a];
      const Real radius = std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
      const Real lower=b.layer==0?inner_radius:radii[b.layer-1];
      q[0] = (radius-lower)/(radii[b.layer]-lower);
      q[1] = x[(a+1)%3]/scale; q[2] = x[(a+2)%3]/scale;
    }
    if(b.face>=0&&equiangular)for(int a=1;a<3;++a)q[a]=4*std::atan(q[a])/std::acos(-1.);
    for (int a = 0; a < 3; ++a) local[a] = (q[a]-b.lo[a])/(b.hi[a]-b.lo[a]);
  }

  Point Geometry(int id, const Real local[3]) const {
    const auto &b = blocks.at(id);
    Point out;
    Forward(id, local, out.x);
    if (b.face < 0) {
      for (int a = 0; a < 3; ++a) out.J[a][a] = 1/(b.hi[a]-b.lo[a]);
      return out;
    }
    if (b.layer == 0 && !spherical_inner_boundary) {
      Real q[3];
      for (int a = 0; a < 3; ++a) q[a] = b.lo[a]+local[a]*(b.hi[a]-b.lo[a]);
      out = Map{half_width, radii[0], 0}.Geometry(b.face, q);
      for (int A = 0; A < 3; ++A)
      for (int i = 0; i < 3; ++i) {
        out.J[A][i] /= b.hi[A]-b.lo[A];
        for (int j = 0; j < 3; ++j) out.H[A][i][j] /= b.hi[A]-b.lo[A];
      }
      return out;
    }
    Jet x[3];
    for (int a = 0; a < 3; ++a) x[a] = Jet::Variable(out.x[a], a);
    const int a = b.face/2;
    const Jet scale = x[a]*Jet(b.face%2 ? -1 : 1);
    const Real lower=b.layer==0?inner_radius:radii[b.layer-1];
    Jet q[3] = {(Power(x[0]*x[0]+x[1]*x[1]+x[2]*x[2], .5)
        -Jet(lower))/Jet(radii[b.layer]-lower),
        x[(a+1)%3]/scale, x[(a+2)%3]/scale};
    if(equiangular)for(int A=1;A<3;++A)q[A]=Jet(4/std::acos(-1.))*Atan(q[A]);
    for (int A = 0; A < 3; ++A)
    for (int i = 0; i < 3; ++i) {
      out.J[A][i] = q[A].d[i]/(b.hi[A]-b.lo[A]);
      for (int j = 0; j < 3; ++j) out.H[A][i][j] = q[A].h[i][j]/(b.hi[A]-b.lo[A]);
    }
    return out;
  }

  // Active leaf owner; deterministic ties choose the first block. No shell
  // neighbor relation is inferred from the Cartesian octree.
  int Owner(const Real x[3]) const {
    const Real maxcoord = std::max({std::abs(x[0]), std::abs(x[1]), std::abs(x[2])});
    const Real radius = std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
    if (radius > radii.back()*(1+2e-14)) return -1;
    const bool bulk = maxcoord <= half_width*(1+2e-14);
    const int face = bulk ? -1 : Map{}.Owner(x);
    const int layer = bulk ? -1 : static_cast<int>(std::lower_bound(radii.begin(), radii.end(),
        std::min(radius, radii.back()))-radii.begin());
    for (int m = 0; m < static_cast<int>(blocks.size()); ++m) {
      const auto &block = blocks[m];
      if (block.face != face || block.layer != layer) continue;
      Real q[3]; Inverse(m, x, q);
      if (q[0] >= -2e-13 && q[0] <= 1+2e-13 && q[1] >= -2e-13
          && q[1] <= 1+2e-13 && q[2] >= -2e-13 && q[2] <= 1+2e-13) return m;
    }
    throw std::runtime_error("no active composite donor");
  }

  // Cartesian leaf refinement; topology and all solver levels must be rebuilt
  // after this revision changes. No solver state is stored in this descriptor.
  void RefineCartesian(int id) {
    const auto parent = blocks.at(id);
    if (parent.face >= 0) throw std::invalid_argument("shell AMR not implemented");
    blocks.erase(blocks.begin()+id);
    for (int child = 0; child < 8; ++child) {
      auto b = parent;
      ++b.refinement;
      for (int a = 0; a < 3; ++a) {
        const Real mid = (parent.lo[a]+parent.hi[a])/2;
        if (child & (1<<a)) b.lo[a] = mid; else b.hi[a] = mid;
      }
      blocks.push_back(b);
    }
    ++revision;
  }
  void CoarsenCartesian(const CompositeBlock &parent) {
    if (parent.face >= 0) throw std::invalid_argument("shell AMR not implemented");
    std::vector<size_t> children;
    unsigned octants = 0;
    for (size_t m = 0; m < blocks.size(); ++m) {
      const auto &b = blocks[m];
      if (b.face >= 0 || b.refinement != parent.refinement+1) continue;
      bool inside = true;
      int octant = 0;
      for (int a = 0; a < 3; ++a) {
        const Real mid = (parent.lo[a]+parent.hi[a])/2;
        const bool lower = b.lo[a] == parent.lo[a] && b.hi[a] == mid;
        const bool upper = b.lo[a] == mid && b.hi[a] == parent.hi[a];
        if (!lower && !upper) inside = false;
        if (upper) octant |= 1<<a;
      }
      if (inside) {children.push_back(m); octants |= 1u<<octant;}
    }
    if (children.size() != 8 || octants != 255)
      throw std::invalid_argument("coarsening requires eight active sibling leaves");
    for (auto i = children.rbegin(); i != children.rend(); ++i) blocks.erase(blocks.begin()+*i);
    blocks.push_back(parent); ++revision;
  }
};

inline Real PhysicalCellVolume(const Point &p, int cells) {
  const auto &a = p.J;
  const Real det = a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])
      -a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])
      +a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]);
  if (!(std::abs(det) > 0) || !std::isfinite(det))
    throw std::runtime_error("degenerate composite map");
  return 1/(std::abs(det)*cells*cells*cells);
}

}  // namespace cubed_sphere
#endif  // MESH_CUBED_SPHERE_COMPOSITE_HPP_
