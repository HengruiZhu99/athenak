// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#ifndef Z4C_HYPERBOLOIDAL_SPHERICAL_GHOSTS_HPP_
#define Z4C_HYPERBOLOIDAL_SPHERICAL_GHOSTS_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>
#include <Kokkos_Core.hpp>

namespace z4c {
namespace hyperboloidal {

// Uniform Cartesian patch containing the entire ball, centered at the origin.
// first[] is the coordinate of stored index zero (including allocated halos).
// This host-side planner is not an AMR interpolation or a characteristic BC.
struct SphericalGhostGrid {
  int n[3];
  double first[3], h[3], radius;
  bool Allocated(int i, int j, int k) const {
    return i >= 0 && i < n[0] && j >= 0 && j < n[1] && k >= 0 && k < n[2];
  }
  bool Interior(int i, int j, int k) const {
    if (!Allocated(i,j,k)) return false;
    const double x = first[0]+i*h[0], y = first[1]+j*h[1], z = first[2]+k*h[2];
    return x*x+y*y+z*z < radius*radius;
  }
  int Index(int i, int j, int k) const { return i+n[0]*(j+n[1]*k); }
};

struct SphericalGhostStencil {
  int target, donors[216], count;
  double weights[216];
};

// Interpolate on interior coordinate planes intersecting the true sphere normal,
// then extrapolate along that normal to each required exterior node. Every donor
// MUST be strictly inside the ball, including for mixed-derivative corners whose
// Cartesian coordinate lines may not intersect the physical domain.
// No donor is another ghost: fills are independent and can run in parallel.
// Total-degree-p polynomials are reproduced exactly. This consistency property
// does NOT establish stability for Z4c or prescribe incoming characteristic data.
// halo is AthenaK NGHOST: mixed derivative radius halo-1, upwind/KO radius halo.
// The normal-plane construction follows the smooth-data idea in Baeza et al.,
// https://doi.org/10.1007/s10915-015-0043-2, extended here to two transverse axes.
inline std::vector<SphericalGhostStencil> PlanSphericalGhosts(
    const SphericalGhostGrid &g, int halo, int degree) {
  if (halo < 2 || halo > 4 || degree < 2 || degree > 5 || !std::isfinite(g.radius)
      || g.radius <= 0) throw std::invalid_argument("invalid spherical ghost policy");
  size_t cells = 1;
  for (int d = 0; d < 3; ++d) {
    if (g.n[d] < 2*halo+1 || !std::isfinite(g.first[d])
        || !std::isfinite(g.h[d]) || g.h[d] <= 0) {
      throw std::invalid_argument("invalid spherical ghost grid");
    }
    // Require the physical ball and its full stencil halo in this patch.
    if (g.first[d]+halo*g.h[d] > -g.radius
        || g.first[d]+(g.n[d]-1-halo)*g.h[d] < g.radius) {
      throw std::invalid_argument("spherical ghost patch does not contain ball/halo");
    }
    cells *= g.n[d];
    if (cells > static_cast<size_t>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument("spherical ghost indexing overflow");
    }
  }
  std::vector<unsigned char> needed(cells, 0);
  int interior = 0;
  for (int k = halo; k < g.n[2]-halo; ++k)
  for (int j = halo; j < g.n[1]-halo; ++j)
  for (int i = halo; i < g.n[0]-halo; ++i) {
    if (!g.Interior(i,j,k)) continue;
    ++interior;
    // Mixed second derivatives use two axes to radius halo-1. KO adds an
    // axis-only stencil of radius halo. There are no three-axis mixed terms.
    for (int c = -halo; c <= halo; ++c)
    for (int b = -halo; b <= halo; ++b)
    for (int a = -halo; a <= halo; ++a) {
      const int axes = (a != 0)+(b != 0)+(c != 0);
      if (axes > 2 || (axes == 2 &&
          std::max({std::abs(a),std::abs(b),std::abs(c)}) == halo)) continue;
      if (!g.Interior(i+a,j+b,k+c)) needed[g.Index(i+a,j+b,k+c)] = 1;
    }
  }
  if (interior == 0) throw std::invalid_argument("sphere has no interior nodes");
  std::vector<SphericalGhostStencil> plans;
  for (int k = 0; k < g.n[2]; ++k)
  for (int j = 0; j < g.n[1]; ++j)
  for (int i = 0; i < g.n[0]; ++i) {
    if (!needed[g.Index(i,j,k)]) continue;
    const int index[3] = {i,j,k};
    double x[3];
    int axis = 0;
    for (int d = 0; d < 3; ++d) {
      x[d] = g.first[d]+index[d]*g.h[d];
      if (std::abs(x[d]/g.h[d]) > std::abs(x[axis]/g.h[axis])) axis = d;
    }
    const int b = (axis+1)%3, c = (axis+2)%3;
    const int sign = x[axis] > 0 ? -1 : 1;
    SphericalGhostStencil best{};
    bool found = false;
    // A normal ray intersects coordinate planes, with two-dimensional tensor
    // interpolation inside each plane. The normal spacing is increased until
    // all transverse interpolation rectangles lie strictly within the sphere.
    for (int step = 1; step < g.n[axis] && !found; ++step) {
      SphericalGhostStencil candidate{};
      candidate.target = g.Index(i,j,k);
      bool valid = true;
      for (int l = 1; l <= degree+1 && valid; ++l) {
        const int plane = index[axis]+sign*l*step;
        if (plane < 0 || plane >= g.n[axis]) {
          valid = false;
          break;
        }
        const double scale = (g.first[axis]+plane*g.h[axis])/x[axis];
        const double tb = (scale*x[b]-g.first[b])/g.h[b];
        const double tc = (scale*x[c]-g.first[c])/g.h[c];
        int first_b = -1, first_c = -1;
        double best_distance = std::numeric_limits<double>::infinity();
        // Only interpolation rectangles containing the normal-ray point are
        // admissible. No transverse extrapolation or exterior donor is allowed.
        for (int sb = static_cast<int>(std::ceil(tb))-degree;
             sb <= static_cast<int>(std::floor(tb)); ++sb)
        for (int sc = static_cast<int>(std::ceil(tc))-degree;
             sc <= static_cast<int>(std::floor(tc)); ++sc) {
          bool rectangle = true;
          for (int db : {0,degree}) for (int dc : {0,degree}) {
            int donor[3];
            donor[axis] = plane; donor[b] = sb+db; donor[c] = sc+dc;
            if (!g.Interior(donor[0],donor[1],donor[2])) rectangle = false;
          }
          if (!rectangle) continue;
          const double distance = std::pow(sb+0.5*degree-tb,2)
                                  +std::pow(sc+0.5*degree-tc,2);
          if (distance < best_distance) {
            best_distance = distance; first_b = sb; first_c = sc;
          }
        }
        if (first_b < 0) {
          valid = false;
          break;
        }
        double normal_weight = 1;
        for (int m = 1; m <= degree+1; ++m) {
          if (m != l) normal_weight *= -static_cast<double>(m)/(l-m);
        }
        normal_weight = std::round(normal_weight);
        for (int db = 0; db <= degree; ++db)
        for (int dc = 0; dc <= degree; ++dc) {
          double weight = normal_weight;
          for (int m = 0; m <= degree; ++m) {
            if (m != db) weight *= (tb-first_b-m)/(db-m);
            if (m != dc) weight *= (tc-first_c-m)/(dc-m);
          }
          int donor[3];
          donor[axis] = plane; donor[b] = first_b+db; donor[c] = first_c+dc;
          const int slot = candidate.count++;
          candidate.donors[slot] = g.Index(donor[0],donor[1],donor[2]);
          candidate.weights[slot] = weight;
        }
      }
      if (valid) {
        best = candidate;
        found = true;
      }
    }
    if (!found) throw std::invalid_argument("no interior normal-ray rectangles");
    plans.push_back(best);
  }
  return plans;
}

// Optional cube-equivariant version of the same strictly interior normal-ray
// extrapolation. Plan one representative in the positive sorted-coordinate
// sector and transport its donor map to every reflected/permuted target. Average
// the representative's stabilizer when coordinates coincide; choosing just one
// dominant axis or tied rectangle would break that symmetry. Polynomial
// consistency is retained, but neither an energy estimate nor a reduction of
// the worst extrapolation amplification follows from this symmetrization.
// The original PlanSphericalGhosts behavior remains unchanged. This variant
// requires a centered isotropic cube and rejects a non-invariant mask or a donor
// union exceeding the existing fixed stencil capacity instead of degrading it.
inline std::vector<SphericalGhostStencil> PlanSymmetricSphericalGhosts(
    const SphericalGhostGrid &g, int halo, int degree) {
  for (int a = 0; a < 3; ++a) {
    const double scale = std::max({1.0,std::abs(g.first[a]),
                                  std::abs((g.n[a]-1)*g.h[a])});
    if (g.n[a] != g.n[0] || g.h[a] != g.h[0] || !std::isfinite(g.first[a])
        || !std::isfinite(g.h[a])
        || std::abs(2*g.first[a]+(g.n[a]-1)*g.h[a])
            > 64*std::numeric_limits<double>::epsilon()*scale) {
      throw std::invalid_argument(
          "symmetric spherical ghosts need a centered isotropic cube");
    }
  }
  const auto old = PlanSphericalGhosts(g,halo,degree);
  std::map<int,const SphericalGhostStencil *> lookup;
  for (const auto &plan : old) lookup[plan.target] = &plan;
  const auto transform = [&g](int s, const std::array<int,3> &perm, int reflect) {
    const int index[3] = {s%g.n[0],s/g.n[0]%g.n[1],s/(g.n[0]*g.n[1])};
    int out[3];
    for (int a = 0; a < 3; ++a) {
      out[a] = reflect & (1 << a) ? g.n[a]-1-index[perm[a]] : index[perm[a]];
    }
    return g.Index(out[0],out[1],out[2]);
  };
  const std::array<int,3> identity = {0,1,2}, xy = {1,0,2}, yz = {0,2,1};
  for (const auto &plan : old) {
    for (const auto &tr : {std::make_pair(identity,1),std::make_pair(identity,2),
                          std::make_pair(identity,4),std::make_pair(xy,0),
                          std::make_pair(yz,0)}) {
      if (lookup.find(transform(plan.target,tr.first,tr.second)) == lookup.end()) {
        throw std::invalid_argument("spherical ghost target mask is not cube invariant");
      }
    }
  }
  std::vector<SphericalGhostStencil> plans;
  plans.reserve(old.size());
  for (const auto &raw : old) {
    std::array<int,3> canonical = {raw.target%g.n[0],raw.target/g.n[0]%g.n[1],
                                  raw.target/(g.n[0]*g.n[1])};
    for (int a = 0; a < 3; ++a) {
      canonical[a] = std::max(canonical[a],g.n[a]-1-canonical[a]);
    }
    std::sort(canonical.begin(),canonical.end(),std::greater<int>());
    const int representative = g.Index(canonical[0],canonical[1],canonical[2]);
    const auto found = lookup.find(representative);
    if (found == lookup.end()) {
      throw std::invalid_argument("spherical ghost mask is not cube invariant");
    }
    const auto &base = *found->second;
    std::map<int,double> weights;
    int multiplicity = 0;
    std::array<int,3> perm = {0,1,2};
    do {
      for (int reflect = 0; reflect < 8; ++reflect) {
        if (transform(representative,perm,reflect) != raw.target) continue;
        ++multiplicity;
        for (int l = 0; l < base.count; ++l) {
          weights[transform(base.donors[l],perm,reflect)] += base.weights[l];
        }
      }
    } while (std::next_permutation(perm.begin(),perm.end()));
    SphericalGhostStencil plan{};
    plan.target = raw.target;
    for (const auto &item : weights) {
      if (item.second == 0) continue;
      const int s = item.first;
      if (!g.Interior(s%g.n[0],s/g.n[0]%g.n[1],s/(g.n[0]*g.n[1]))) {
        throw std::invalid_argument("spherical donor mask is not cube invariant");
      }
      if (plan.count == 216) {
        throw std::invalid_argument(
            "symmetric spherical ghost donor union exceeds capacity");
      }
      plan.donors[plan.count] = s;
      plan.weights[plan.count++] = item.second/multiplicity;
    }
    plans.push_back(plan);
  }
  return plans;
}

// Scalar flattened view. For tensors, apply the same plans to each Cartesian
// component; do not rotate components into the extrapolation-line direction.
// Prefer deviations from the analytic reference so reference values are exact.
template <typename ScalarView, typename PlanView>
void FillSphericalGhosts(const ScalarView &q, const PlanView &plans) {
  Kokkos::parallel_for("spherical Cartesian ghosts", plans.extent(0),
      KOKKOS_LAMBDA(const int p) {
    const auto &s = plans(p);
    double value = 0;
    for (int l = 0; l < s.count; ++l) value += s.weights[l]*q(s.donors[l]);
    q(s.target) = value;
  });
}

}  // namespace hyperboloidal
}  // namespace z4c
#endif  // Z4C_HYPERBOLOIDAL_SPHERICAL_GHOSTS_HPP_
