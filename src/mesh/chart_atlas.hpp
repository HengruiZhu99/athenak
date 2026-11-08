#ifndef MESH_CHART_ATLAS_HPP_
#define MESH_CHART_ATLAS_HPP_
//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file chart_atlas.hpp
//! \brief Stateless geometry for a Cartesian core and six cubed-sphere charts.

#include <array>
#include <string>
#include <vector>

#include "athena.hpp"

struct LogicalLocation;

namespace chart_atlas {

// Components of evolved tensors remain in the common Cartesian basis. Only
// derivative directions are transformed. J/H differentiate local [0,1]^3
// block coordinates with respect to the common Cartesian coordinates.
struct Point {
  Real x[3]{};
  Real J[3][3]{};
  Real H[3][3][3]{};
};

struct Bounds {
  Real lo[3]{}, hi[3]{};
};

namespace detail {
// A scalar second-order jet evaluates the analytic inverse-map derivatives.
// It carries no field state and is usable in a device geometry kernel.
struct Jet {
  Real f = 0, d[3]{}, h[3][3]{};
  KOKKOS_INLINE_FUNCTION Jet() = default;
  KOKKOS_INLINE_FUNCTION Jet(Real value) : f(value) {}
  KOKKOS_INLINE_FUNCTION static Jet Variable(Real value, int a) {
    Jet out(value);
    out.d[a] = 1;
    return out;
  }
};

KOKKOS_INLINE_FUNCTION Jet operator+(const Jet &a, const Jet &b) {
  Jet out(a.f + b.f);
  for (int i=0; i<3; ++i) {
    out.d[i] = a.d[i] + b.d[i];
    for (int j=0; j<3; ++j) out.h[i][j] = a.h[i][j] + b.h[i][j];
  }
  return out;
}
KOKKOS_INLINE_FUNCTION Jet operator-(const Jet &a, const Jet &b) {
  Jet out(a.f - b.f);
  for (int i=0; i<3; ++i) {
    out.d[i] = a.d[i] - b.d[i];
    for (int j=0; j<3; ++j) out.h[i][j] = a.h[i][j] - b.h[i][j];
  }
  return out;
}
KOKKOS_INLINE_FUNCTION Jet operator*(const Jet &a, const Jet &b) {
  Jet out(a.f*b.f);
  for (int i=0; i<3; ++i) {
    out.d[i] = a.d[i]*b.f + a.f*b.d[i];
    for (int j=0; j<3; ++j) {
      out.h[i][j] = a.h[i][j]*b.f + a.d[i]*b.d[j]
          + a.d[j]*b.d[i] + a.f*b.h[i][j];
    }
  }
  return out;
}
KOKKOS_INLINE_FUNCTION Jet Power(const Jet &a, Real p) {
  Jet out(pow(a.f, p));
  const Real df = p*pow(a.f, p-1), ddf = p*(p-1)*pow(a.f, p-2);
  for (int i=0; i<3; ++i) {
    out.d[i] = df*a.d[i];
    for (int j=0; j<3; ++j) {
      out.h[i][j] = df*a.h[i][j] + ddf*a.d[i]*a.d[j];
    }
  }
  return out;
}
KOKKOS_INLINE_FUNCTION Jet operator/(const Jet &a, const Jet &b) {
  return a*Power(b, -1);
}
KOKKOS_INLINE_FUNCTION Jet Atan(const Jet &a) {
  Jet out(atan(a.f));
  const Real df = 1/(1+a.f*a.f), ddf = -2*a.f*df*df;
  for (int i=0; i<3; ++i) {
    out.d[i] = df*a.d[i];
    for (int j=0; j<3; ++j) {
      out.h[i][j] = df*a.h[i][j] + ddf*a.d[i]*a.d[j];
    }
  }
  return out;
}
}  // namespace detail

// One small, trivially copyable map per native leaf, suitable for a packed
// device view. No atlas, tree, allocation, or ownership lookup is needed on
// device. Chart 0 is Cartesian; charts 1..6 are +X,-X,+Y,-Y,+Z,-Z.
struct BlockMap {
  int chart_id = 0, face = -1, layer = -1;
  Real lo[3]{}, hi[3]{};
  Real half_width = 0, inner_radius = 0, outer_radius = 0;
  bool cube_transition = false, equiangular = false, outer_surface = false;
  Real node_offset[3]{0.5, 0.5, 0.5};
  // Number of independently evolved cells beyond each nominal block face.
  // These cells are distinct from interpolation/difference ghosts.
  int overlap_sides[6]{};

  KOKKOS_INLINE_FUNCTION void Forward(const Real local[3], Real x[3]) const {
    Real q[3];
    for (int a=0; a<3; ++a) q[a] = lo[a] + local[a]*(hi[a]-lo[a]);
    if (face < 0) {
      for (int a=0; a<3; ++a) x[a] = q[a];
      return;
    }
    const int a = face/2, b = (a+1)%3, c = (a+2)%3;
    if (equiangular) {
      q[1] = tan(acos(Real(-1))*q[1]/4);
      q[2] = tan(acos(Real(-1))*q[2]/4);
    }
    const Real norm = sqrt(1+q[1]*q[1]+q[2]*q[2]);
    const Real scale = cube_transition
        ? (1-q[0])*half_width + q[0]*outer_radius/norm
        : (inner_radius+q[0]*(outer_radius-inner_radius))/norm;
    x[a] = face%2 ? -scale : scale;
    x[b] = scale*q[1];
    x[c] = scale*q[2];
  }

  KOKKOS_INLINE_FUNCTION void Inverse(const Real x[3], Real local[3]) const {
    Real q[3];
    if (face < 0) {
      for (int a=0; a<3; ++a) q[a] = x[a];
    } else {
      const int a = face/2, b = (a+1)%3, c = (a+2)%3;
      const Real scale = face%2 ? -x[a] : x[a];
      q[1] = x[b]/scale;
      q[2] = x[c]/scale;
      if (cube_transition) {
        q[0] = (scale-half_width)
            /(outer_radius/sqrt(1+q[1]*q[1]+q[2]*q[2])-half_width);
      } else {
        const Real radius = sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
        q[0] = (radius-inner_radius)/(outer_radius-inner_radius);
      }
      if (equiangular) {
        q[1] = 4*atan(q[1])/acos(Real(-1));
        q[2] = 4*atan(q[2])/acos(Real(-1));
      }
    }
    for (int a=0; a<3; ++a) local[a] = (q[a]-lo[a])/(hi[a]-lo[a]);
  }

  KOKKOS_INLINE_FUNCTION Point Geometry(const Real local[3]) const {
    using detail::Jet;
    Point out;
    Forward(local, out.x);
    if (face < 0) {
      for (int a=0; a<3; ++a) out.J[a][a] = 1/(hi[a]-lo[a]);
      return out;
    }
    Jet x[3];
    for (int a=0; a<3; ++a) x[a] = Jet::Variable(out.x[a], a);
    const int a = face/2, b = (a+1)%3, c = (a+2)%3;
    const Jet scale = x[a]*Jet(face%2 ? -1 : 1);
    Jet q[3];
    q[1] = x[b]/scale;
    q[2] = x[c]/scale;
    q[0] = cube_transition
        ? (scale-Jet(half_width))/(Jet(outer_radius)
            *detail::Power(Jet(1)+q[1]*q[1]+q[2]*q[2], -0.5)-Jet(half_width))
        : (detail::Power(x[0]*x[0]+x[1]*x[1]+x[2]*x[2], 0.5)-Jet(inner_radius))
            /Jet(outer_radius-inner_radius);
    if (equiangular) {
      q[1] = Jet(4/acos(Real(-1)))*detail::Atan(q[1]);
      q[2] = Jet(4/acos(Real(-1)))*detail::Atan(q[2]);
    }
    for (int A=0; A<3; ++A) {
      for (int i=0; i<3; ++i) {
        out.J[A][i] = q[A].d[i]/(hi[A]-lo[A]);
        for (int j=0; j<3; ++j) out.H[A][i][j] = q[A].h[i][j]/(hi[A]-lo[A]);
      }
    }
    return out;
  }
};

// Geometry and root-grid metadata only. The native MeshBlockTree leaf list is
// the sole block topology/ownership authority. Logical shell x1 runs across
// radial intervals; physical radii need not be uniformly spaced.
class Atlas {
 public:
  Atlas(Real half_width, std::vector<Real> radial_knots,
        std::array<int, 3> core_blocks = {1, 1, 1},
        std::array<int, 2> angular_blocks = {1, 1},
        Real spherical_inner_radius = 0, bool equiangular = false,
        int evolved_overlap = 0, bool outer_endpoint_grid = false);

  int RootLevel() const {return root_level_;}
  static constexpr int NumCharts() {return 7;}
  std::array<int, 3> RootBlocks(int chart_id) const;
  const std::vector<Real> &RadialKnots() const {return radial_knots_;}
  Real HalfWidth() const {return half_width_;}
  std::string Identity() const;
  BlockMap GetBlockMap(const LogicalLocation &loc) const;
  Bounds PhysicalBounds(const LogicalLocation &loc) const;

 private:
  Real half_width_, spherical_inner_radius_;
  std::vector<Real> radial_knots_;
  std::array<int, 3> core_blocks_;
  std::array<int, 2> angular_blocks_;
  int root_level_ = 0, evolved_overlap_;
  bool equiangular_, outer_endpoint_grid_;
};

}  // namespace chart_atlas
#endif  // MESH_CHART_ATLAS_HPP_
