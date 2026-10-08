// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#ifndef Z4C_HYPERBOLOIDAL_CARTESIAN_HAWKING_HPP_
#define Z4C_HYPERBOLOIDAL_CARTESIAN_HAWKING_HPP_

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include "z4c/hyperboloidal/cartesian_patch.hpp"

namespace z4c {
namespace hyperboloidal {

// Densities per Euclidean solid angle on the coordinate sphere through x.
// Null normals are n+s and n-s, with inner product -2. The correction is
// theta_plus*theta_minus*dA/domega + 4, so its sphere integral is I+16*pi.
// This shifts an exactly known constant, not any evolved or Schwarzschild data.
template <typename T>
struct HawkingDensity {
  bool valid = false;
  T area = 0, correction = 0;
};

template <typename T>
KOKKOS_INLINE_FUNCTION
HawkingDensity<T> CoordinateSphereDensity(const Z4cJet<T> &u,
    const OmegaJet<T> &o, const T x[3]) {
  HawkingDensity<T> out;
  T r2 = 0;
  for (int i = 0; i < 3; ++i) r2 += x[i]*x[i];
  if (!(r2 > 0 && o.omega > 0 && u.chi.value > 0)) return out;
  const auto gb = Geometry(PenroseMetric(u.metric,u.chi));
  if (!gb.valid) return out;
  T norm2 = 0, normal[3]{};
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) {
    normal[i] += gb.inverse[i][j]*x[j];
    norm2 += gb.inverse[i][j]*x[i]*x[j];
  }
  if (!(norm2 > 0)) return out;
  const T norm = Kokkos::sqrt(norm2);
  for (int i = 0; i < 3; ++i) normal[i] /= norm;
  T divergence = 0, a_difference = 0, normal_omega = 0;
  for (int i = 0; i < 3; ++i) {
    normal_omega += normal[i]*o.gradient[i];
    for (int j = 0; j < 3; ++j) {
      T hessian = i == j ? 1 : 0;
      for (int k = 0; k < 3; ++k) hessian -= gb.connection[k][i][j]*x[k];
      divergence += (gb.inverse[i][j]-normal[i]*normal[j])*hessian/norm;
      // Use the full physical ADM curvature even if the algebraic A trace
      // constraint has a numerical residual.
      a_difference += u.a.k[i][j]*(normal[i]*normal[j]-gb.inverse[i][j]);
    }
  }
  const T mean_curvature = o.omega*divergence-2*normal_omega;
  const T trace_surface = o.omega*a_difference/u.chi.value
      -T(2)/3*(u.trace.value+2*u.theta.value);
  out.area = Kokkos::sqrt(r2*gb.determinant*norm2)/(o.omega*o.omega);
  out.correction = (trace_surface*trace_surface-mean_curvature*mean_curvature)
      *out.area+4;
  out.valid = out.area > 0 && Kokkos::isfinite(out.area)
      && Kokkos::isfinite(out.correction);
  return out;
}

struct HawkingMassResult {
  double coordinate_radius, area, areal_radius, mass;
};

// Gauss-Legendre in cos(theta), periodic trapezoid in phi. Sampling does not
// assume spherical symmetry of the metric, curvature or surface densities.
template <typename Sampler>
HawkingMassResult IntegrateHawkingMass(double radius, int nmu, const Sampler &sample) {
  if (!std::isfinite(radius) || radius <= 0 || nmu < 4 || nmu > 128) {
    throw std::invalid_argument("invalid Hawking sphere/quadrature");
  }
  const double pi = std::acos(-1.0);
  long double area = 0, correction = 0;
  for (int i = 0; i < nmu; ++i) {
    double mu = std::cos(pi*(i+0.75)/(nmu+0.5)), derivative = 0;
    bool converged = false;
    for (int iteration = 0; iteration < 64; ++iteration) {
      double p = 1, previous = 0;
      for (int l = 1; l <= nmu; ++l) {
        const double old = p;
        p = ((2*l-1)*mu*p-(l-1)*previous)/l;
        previous = old;
      }
      derivative = nmu*(mu*p-previous)/(mu*mu-1);
      const double delta = p/derivative;
      if (std::abs(delta) < 2e-15) {
        converged = true;
        break;
      }
      mu -= delta;
    }
    if (!converged) throw std::runtime_error("Hawking quadrature did not converge");
    const double weight = 2/((1-mu*mu)*derivative*derivative)*pi/nmu;
    for (int j = 0; j < 2*nmu; ++j) {
      const double phi = pi*(j+0.5)/nmu, rho = radius*std::sqrt(1-mu*mu);
      const double x[3] = {rho*std::cos(phi),rho*std::sin(phi),radius*mu};
      const auto density = sample(x);
      if (!density.valid) throw std::runtime_error("invalid Hawking surface density");
      area += weight*density.area;
      correction += weight*density.correction;
    }
  }
  const double areal_radius = std::sqrt(static_cast<double>(area)/(4*pi));
  const double mass = areal_radius/2*static_cast<double>(correction)/(16*pi);
  if (!(area > 0) || !std::isfinite(areal_radius) || !std::isfinite(mass)) {
    throw std::runtime_error("invalid Hawking mass integral");
  }
  return {radius,static_cast<double>(area),areal_radius,mass};
}

// Compute surface densities from the same reconstructed jets as the constraints,
// then interpolate them with interior-only tricubic stencils. The interpolation
// error is separately measurable on exact trumpet data; this is not a scri limit.
inline std::vector<HawkingMassResult> CartesianHawkingMasses(
    const CartesianConformalPatch &patch, const DvceArray5D<Real> &data,
    const std::vector<double> &radii, int nmu = 32) {
  for (const double r : radii) {
    if (!(r > 0 && r < patch.grid.radius) || !std::isfinite(r)) {
      throw std::invalid_argument("Hawking sphere must be strictly inside scri");
    }
  }
  if (radii.empty()) return {};
  patch.Prepare(data);
  const auto g = patch.grid;
  const auto ref = patch.reference;
  const auto nodes = patch.active;
  const auto jets = patch.reconstruction_jets;
  const auto q = BindCartesianFields(patch.deviations);
  Kokkos::View<Real **> densities("Hawking densities",g.n[0]*g.n[1]*g.n[2],2);
  Kokkos::deep_copy(densities,std::numeric_limits<Real>::quiet_NaN());
  int failures = 0;
  Kokkos::parallel_reduce("Hawking grid densities",nodes.extent(0),
      KOKKOS_LAMBDA(const int point, int &bad) {
    const int s = nodes(point), i = s%g.n[0], j = s/g.n[0]%g.n[1];
    const int k = s/(g.n[0]*g.n[1]);
    const Real x[3] = {g.first[0]+i*g.h[0],g.first[1]+j*g.h[1],g.first[2]+k*g.h[2]};
    if (x[0] == 0 && x[1] == 0 && x[2] == 0) return;
    const Real idx[3] = {1/g.h[0],1/g.h[1],1/g.h[2]};
    const auto p = ref.At(x[0],x[1],x[2]);
    auto u = LoadMeshJet<3>(q,idx,0,k,j,i);
    if (jets.extent(0)) AddBackgroundJet(u,jets(point));
    else AddReferenceJet(u,p,ref);
    const auto density = CoordinateSphereDensity(u,CartesianOmega(u,p),x);
    if (!density.valid) {
      ++bad;
      return;
    }
    densities(s,0) = density.area;
    densities(s,1) = density.correction;
  },failures);
  if (failures) throw std::runtime_error("invalid Hawking grid state");
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),densities);
  const auto sample = [g,host](const double x[3]) {
    int first[3];
    double weights[3][4];
    for (int d = 0; d < 3; ++d) {
      const double t = (x[d]-g.first[d])/g.h[d];
      first[d] = static_cast<int>(std::floor(t))-1;
      for (int a = 0; a < 4; ++a) {
        weights[d][a] = 1;
        for (int b = 0; b < 4; ++b) {
          if (a != b) weights[d][a] *= (t-first[d]-b)/(a-b);
        }
      }
    }
    HawkingDensity<double> out;
    for (int c = 0; c < 4; ++c)
    for (int b = 0; b < 4; ++b)
    for (int a = 0; a < 4; ++a) {
      const int i = first[0]+a, j = first[1]+b, k = first[2]+c;
      if (!g.Interior(i,j,k)) {
        throw std::invalid_argument("Hawking interpolation touches inactive cells");
      }
      const int s = g.Index(i,j,k);
      const double w = weights[0][a]*weights[1][b]*weights[2][c];
      out.area += w*host(s,0);
      out.correction += w*host(s,1);
    }
    out.valid = out.area > 0 && std::isfinite(out.area) && std::isfinite(out.correction);
    return out;
  };
  std::vector<HawkingMassResult> result;
  for (const double radius : radii) {
    result.push_back(IntegrateHawkingMass(radius,nmu,sample));
  }
  return result;
}

}  // namespace hyperboloidal
}  // namespace z4c
#endif  // Z4C_HYPERBOLOIDAL_CARTESIAN_HAWKING_HPP_
