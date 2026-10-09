// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#ifndef Z4C_HYPERBOLOIDAL_CARTESIAN_PATCH_HPP_
#define Z4C_HYPERBOLOIDAL_CARTESIAN_PATCH_HPP_

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "z4c/hyperboloidal/athenak_bridge.hpp"
#include "z4c/hyperboloidal/layer_gauge.hpp"
#include "z4c/hyperboloidal/interior_dissipation.hpp"
#include "z4c/hyperboloidal/spherical_ghosts.hpp"
#include "inner_conformal_trace.hpp"

namespace z4c {
namespace hyperboloidal {

inline Z4c::Z4c_vars BindCartesianFields(const DvceArray5D<Real> &data) {
  Z4c::Z4c_vars q;
  q.chi.InitWithShallowSlice(data,Z4c::I_Z4C_CHI);
  q.vKhat.InitWithShallowSlice(data,Z4c::I_Z4C_KHAT);
  q.vTheta.InitWithShallowSlice(data,Z4c::I_Z4C_THETA);
  q.alpha.InitWithShallowSlice(data,Z4c::I_Z4C_ALPHA);
  q.g_dd.InitWithShallowSlice(data,Z4c::I_Z4C_GXX,Z4c::I_Z4C_GZZ);
  q.vA_dd.InitWithShallowSlice(data,Z4c::I_Z4C_AXX,Z4c::I_Z4C_AZZ);
  q.vGam_u.InitWithShallowSlice(data,Z4c::I_Z4C_GAMX,Z4c::I_Z4C_GAMZ);
  q.beta_u.InitWithShallowSlice(data,Z4c::I_Z4C_BETAX,Z4c::I_Z4C_BETAZ);
  q.vB_d.InitWithShallowSlice(data,Z4c::I_Z4C_BX,Z4c::I_Z4C_BZ);
  return q;
}

// Non-owning flattened field access. The calling patch retains the parent view
// until its kernels finish. Keep a strided fallback for padded/subview storage.
struct CartesianComponent {
  Real *data;
  int nx, ny;
  size_t sx, sy, sz;
  bool contiguous;
  template <typename View>
  KOKKOS_INLINE_FUNCTION
  CartesianComponent(const View &v, int field, int x, int y)
      : data(v.data()+field*v.stride_1()), nx(x), ny(y), sx(v.stride_4()),
        sy(v.stride_3()), sz(v.stride_2()),
        contiguous(sx == 1 && sy == static_cast<size_t>(nx)
                   && sz == static_cast<size_t>(nx)*ny) {}
  KOKKOS_INLINE_FUNCTION
  Real &operator()(int s) const {
    if (contiguous) return data[s];
    return data[(s/(nx*ny))*sz+(s/nx%ny)*sy+(s%nx)*sx];
  }
};

KOKKOS_INLINE_FUNCTION
Real ReferenceComponent(int f, const CMCPoint<Real> &p) {
  if (f == Z4c::I_Z4C_CHI || f == Z4c::I_Z4C_GXX
      || f == Z4c::I_Z4C_GYY || f == Z4c::I_Z4C_GZZ) return 1;
  if (f == Z4c::I_Z4C_KHAT) return p.k_physical;
  if (f == Z4c::I_Z4C_ALPHA) return p.alpha;
  if (f >= Z4c::I_Z4C_BETAX && f <= Z4c::I_Z4C_BETAZ) return p.beta[f-Z4c::I_Z4C_BETAX];
  return 0;
}

KOKKOS_INLINE_FUNCTION
void AddReferenceJet(Z4cJet<Real> &u, const CMCPoint<Real> &p,
                     const CMCReference<Real> &ref) {
  u.chi.value += 1;
  u.alpha.value += p.alpha;
  u.trace.value += p.k_physical;
  for (int a = 0; a < 3; ++a) {
    u.metric.g[a][a] += 1;
    u.alpha.d[a] += p.dalpha[a];
    u.alpha.dd[a][a] += p.hessian_alpha;
    u.beta.value[a] += p.beta[a];
    u.beta.d[a][a] -= 1/ref.curvature_radius;
  }
}

KOKKOS_INLINE_FUNCTION
OmegaJet<Real> CartesianOmega(const Z4cJet<Real> &u, const CMCPoint<Real> &p) {
  OmegaJet<Real> o{};
  o.omega = p.omega;
  for (int a = 0; a < 3; ++a) {
    o.gradient[a] = p.domega[a];
    o.hessian[a][a] = p.hessian_omega;
  }
  SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,o);
  return o;
}

// Layer overloads retain the same physical storage and Cartesian basis.
KOKKOS_INLINE_FUNCTION
Real ReferenceComponent(int f, const LayerPoint<Real> &p) {
  const auto &u = p.state;
  if (f == Z4c::I_Z4C_CHI) return u.chi.value;
  if (f == Z4c::I_Z4C_KHAT) return u.trace.value;
  if (f == Z4c::I_Z4C_ALPHA) return u.alpha.value;
  if (f >= Z4c::I_Z4C_BETAX && f <= Z4c::I_Z4C_BETAZ) {
    return u.beta.value[f-Z4c::I_Z4C_BETAX];
  }
  if (f >= Z4c::I_Z4C_GAMX && f <= Z4c::I_Z4C_GAMZ) {
    return u.lambda.value[f-Z4c::I_Z4C_GAMX];
  }
  const int ti[6] = {0,0,0,1,1,2}, tj[6] = {0,1,2,1,2,2};
  if (f >= Z4c::I_Z4C_GXX && f <= Z4c::I_Z4C_GZZ) {
    const int k = f-Z4c::I_Z4C_GXX;
    return u.metric.g[ti[k]][tj[k]];
  }
  if (f >= Z4c::I_Z4C_AXX && f <= Z4c::I_Z4C_AZZ) {
    const int k = f-Z4c::I_Z4C_AXX;
    return u.a.k[ti[k]][tj[k]];
  }
  return 0;
}

KOKKOS_INLINE_FUNCTION
void AddReferenceJet(Z4cJet<Real> &u, const LayerPoint<Real> &p,
                     const LayerReference<Real> &) {
  AddBackgroundJet(u,p.state);
}

KOKKOS_INLINE_FUNCTION
OmegaJet<Real> CartesianOmega(const Z4cJet<Real> &u, const LayerPoint<Real> &p) {
  OmegaJet<Real> o{};
  o.omega = p.omega;
  for (int i = 0; i < 3; ++i) {
    o.gradient[i] = p.domega[i];
    for (int j = 0; j < 3; ++j) o.hessian[i][j] = p.omega_hessian[i][j];
  }
  SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,o);
  return o;
}

struct CartesianDiagnostics {
  double h_l2 = 0, m_l2 = 0, z_l2 = 0, theta_l2 = 0;
  double max_h = 0, max_m = 0, max_det = 0, max_trace = 0, max_deviation = 0;
  double max_h_radius = 0, max_m_radius = 0;
  double shell_max_pole = 0, shell_max_pole_deviation = 0;
  double shell_max_null_deviation = 0;
  double min_alpha = std::numeric_limits<double>::infinity();
  double min_chi = std::numeric_limits<double>::infinity();
};

// A uniform, one-MeshBlock conformal evolution adapter. It operates on AthenaK's
// actual 25-field arrays; it does not own an RK integrator or alter Cauchy tasks.
// Only physical r<S nodes evolve. Ghosts are rebuilt from reference deviations.
// No ADM conversion, Weyl extraction or AMR is implied by this adapter.
class CartesianConformalPatch {
 public:
  const SphericalGhostGrid grid;
  const LayerReference<Real> reference;
  const int ghost_degree;
  const bool symmetric_ghosts;
  Kokkos::View<int *> active;
  Kokkos::View<unsigned char *> mask;
  Kokkos::View<SphericalGhostStencil *> ghosts;
  DvceArray5D<Real> deviations;
  DvceArray5D<Real> reconstruction_values;
  Kokkos::View<Z4cJet<Real> *> reconstruction_jets;
  Real min_omega = std::numeric_limits<Real>::infinity();
  Real kappa1 = 5, dissipation = 0.1;
  GaugeParameters<Real> gauge_parameters{2,0.1,1.5,1};
  LayerGaugeParameters layer_gauge;

  explicit CartesianConformalPatch(SphericalGhostGrid g, Real curvature_radius = 1,
                                     int degree = 3, LayerParameters layer = {},
                                     LayerGaugeParameters gauge = {},
                                     bool symmetric = false)
      : grid(g), reference{g.radius,curvature_radius,layer}, ghost_degree(degree),
        symmetric_ghosts(symmetric), layer_gauge(gauge) {
    reference.Validate();
    if (layer.enabled || layer_gauge.physical_trace_lapse) {
      layer_gauge.Validate(g.radius);
      if (layer.enabled && layer_gauge.preferred_source
          && !(layer_gauge.r0 > layer.r0)) {
        throw std::invalid_argument(
            "preferred projection must begin beyond the Cauchy interior");
      }
    }
    const auto plans = symmetric_ghosts
        ? PlanSymmetricSphericalGhosts(grid,3,ghost_degree)
        : PlanSphericalGhosts(grid,3,ghost_degree);
    const int cells = grid.n[0]*grid.n[1]*grid.n[2];
    std::vector<int> nodes;
    mask = Kokkos::View<unsigned char *>("conformal active mask",cells);
    auto mh = Kokkos::create_mirror_view(mask);
    for (int k = 0; k < grid.n[2]; ++k)
    for (int j = 0; j < grid.n[1]; ++j)
    for (int i = 0; i < grid.n[0]; ++i) {
      const int s = grid.Index(i,j,k);
      mh(s) = grid.Interior(i,j,k);
      if (!mh(s)) continue;
      nodes.push_back(s);
      const auto p = reference.At(grid.first[0]+i*grid.h[0],grid.first[1]+j*grid.h[1],
                                  grid.first[2]+k*grid.h[2]);
      if (!(p.omega > 0) || !(p.L > 0) || !(p.alpha > 0)
          || !std::isfinite(p.omega) || !std::isfinite(p.L)
          || !Geometry(p.state.metric).valid) {
        throw std::invalid_argument("invalid geometry on active layer grid");
      }
      min_omega = std::min(min_omega,p.omega);
    }
    Kokkos::deep_copy(mask,mh);
    active = Kokkos::View<int *>("conformal active nodes",nodes.size());
    auto ah = Kokkos::create_mirror_view(active);
    for (size_t s = 0; s < nodes.size(); ++s) ah(s) = nodes[s];
    Kokkos::deep_copy(active,ah);
    ghosts = Kokkos::View<SphericalGhostStencil *>("conformal ghost plans",plans.size());
    auto gh = Kokkos::create_mirror_view(ghosts);
    for (size_t s = 0; s < plans.size(); ++s) gh(s) = plans[s];
    Kokkos::deep_copy(ghosts,gh);
    deviations = Allocate("conformal deviations");
  }

  DvceArray5D<Real> Allocate(const std::string &label) const {
    return DvceArray5D<Real>(label,1,Z4c::nz4c,grid.n[2],grid.n[1],grid.n[0]);
  }

  void CheckShape(const DvceArray5D<Real> &q) const {
    if (q.extent_int(0) != 1 || q.extent_int(1) != Z4c::nz4c
        || q.extent_int(2) != grid.n[2] || q.extent_int(3) != grid.n[1]
        || q.extent_int(4) != grid.n[0]) {
      throw std::invalid_argument("conformal patch shape");
    }
  }

  void InitializeReference(const DvceArray5D<Real> &data) const {
    CheckShape(data);
    const auto g = grid;
    const auto ref = reference;
    Kokkos::parallel_for("Cartesian reference initialization",
        grid.n[0]*grid.n[1]*grid.n[2],
        KOKKOS_LAMBDA(const int index) {
      const int s = index;
      const int i = s%g.n[0], j = s/g.n[0]%g.n[1], k = s/(g.n[0]*g.n[1]);
      const auto p = ref.At(g.first[0]+i*g.h[0],g.first[1]+j*g.h[1],g.first[2]+k*g.h[2]);
      for (int f = 0; f < Z4c::nz4c; ++f) data(0,f,k,j,i) = ReferenceComponent(f,p);
    });
  }

  // Cache a fixed initial profile and its exact derivatives. The CMC reference
  // remains the gauge target and the only RHS whose roundoff is subtracted.
  void SetReconstruction(const DvceArray5D<Real> &data,
                         const Kokkos::View<Z4cJet<Real> *> &jets) {
    if (jets.extent(0) != active.extent(0) || reconstruction_jets.extent(0)) {
      throw std::invalid_argument("invalid/repeated Cartesian reconstruction setup");
    }
    Prepare(data);  // extend the initial profile using the validated sphere plan
    reconstruction_values = Allocate("analytic reconstruction values");
    Kokkos::deep_copy(reconstruction_values,data);
    reconstruction_jets = Kokkos::View<Z4cJet<Real> *>(
        "analytic initial jets",jets.extent(0));
    Kokkos::deep_copy(reconstruction_jets,jets);
  }

  void Prepare(const DvceArray5D<Real> &data) const {
    CheckShape(data);
    if (data.data() == deviations.data()) {
      throw std::invalid_argument("conformal state aliases scratch storage");
    }
    const auto g = grid;
    const auto ref = reference;
    const auto dev = deviations;
    const auto baseline = reconstruction_values;
    const bool reconstructed = baseline.size() != 0;
    const auto inside = mask;
    Kokkos::parallel_for("Cartesian reference deviations",grid.n[0]*grid.n[1]*grid.n[2],
        KOKKOS_LAMBDA(const int index) {
      const int s = index;
      const int i = s%g.n[0], j = s/g.n[0]%g.n[1], k = s/(g.n[0]*g.n[1]);
      const auto p = ref.At(g.first[0]+i*g.h[0],g.first[1]+j*g.h[1],g.first[2]+k*g.h[2]);
      for (int f = 0; f < Z4c::nz4c; ++f) {
        dev(0,f,k,j,i) = inside(s) ? data(0,f,k,j,i)
            -(reconstructed ? baseline(0,f,k,j,i) : ReferenceComponent(f,p)) : 0;
      }
    });
    for (int f = 0; f < Z4c::nz4c; ++f) {
      FillSphericalGhosts(CartesianComponent{dev,f,g.n[0],g.n[1]},ghosts);
    }
    const auto plans = ghosts;
    Kokkos::parallel_for("Cartesian full ghost fields",ghosts.extent(0),
        KOKKOS_LAMBDA(const int point) {
      const int s = plans(point).target;
      const int i = s%g.n[0], j = s/g.n[0]%g.n[1], k = s/(g.n[0]*g.n[1]);
      const auto p = ref.At(g.first[0]+i*g.h[0],g.first[1]+j*g.h[1],g.first[2]+k*g.h[2]);
      for (int f = 0; f < Z4c::nz4c; ++f) {
        data(0,f,k,j,i) = dev(0,f,k,j,i)
            +(reconstructed ? baseline(0,f,k,j,i) : ReferenceComponent(f,p));
      }
    });
  }

  void RHS(const DvceArray5D<Real> &data, const DvceArray5D<Real> &result) const {
    CheckShape(result);
    gauge_parameters.Validate();
    if (!std::isfinite(kappa1) || kappa1 < 0
        || !std::isfinite(dissipation) || dissipation < 0) {
      throw std::invalid_argument("invalid conformal damping/dissipation");
    }
    if (data.data() == result.data() || result.data() == deviations.data()) {
      throw std::invalid_argument("aliased conformal RHS");
    }
    Prepare(data);
    Kokkos::deep_copy(result,Real(0));
    const auto g = grid;
    const auto ref = reference;
    const auto nodes = active;
    const auto inside = mask;
    const auto dev = deviations;
    const auto udev = BindCartesianFields(dev), full = BindCartesianFields(data);
    const auto output = BindCartesianFields(result);
    const auto initial_jets = reconstruction_jets;
    const auto gauge = gauge_parameters;
    const auto lg = layer_gauge;
    const Real damping = kappa1, epsilon = dissipation;
    int failures = 0;
    Kokkos::parallel_reduce("Cartesian conformal Z4c evolution",nodes.extent(0),
        KOKKOS_LAMBDA(const int point, int &bad) {
      const int s = nodes(point), i = s%g.n[0], j = s/g.n[0]%g.n[1];
      const int k = s/(g.n[0]*g.n[1]);
      const Real spacing[3] = {g.h[0],g.h[1],g.h[2]};
      const Real idx[3] = {1/spacing[0],1/spacing[1],1/spacing[2]};
      const auto p = ref.At(g.first[0]+i*g.h[0],g.first[1]+j*g.h[1],g.first[2]+k*g.h[2]);
      auto u = LoadMeshJet<3>(udev,idx,0,k,j,i);
      if (initial_jets.extent(0)) AddBackgroundJet(u,initial_jets(point));
      else AddReferenceJet(u,p,ref);
      if (!(u.alpha.value > 0) || !Kokkos::isfinite(u.alpha.value)) {
        ++bad;
        return;
      }
      Z4cJet<Real> background{};
      AddReferenceJet(background,p,ref);
      const auto omega = CartesianOmega(u,p), omega0 = CartesianOmega(background,p);
      Z4cRHS<Real> rhs{}, rhs0{};
      GaugeRHS<Real> gauge_rhs{};
      if (!(u.metric.g[0][0] > 0)
          || !(u.metric.g[0][0]*u.metric.g[1][1]-u.metric.g[0][1]*u.metric.g[0][1] > 0)
          || !AssembleInterior(ConformalRHS(u,omega,damping/u.alpha.value,Real(0)),
                                omega.omega,rhs)
          || !AssembleInterior(ConformalRHS(background,omega0,damping,Real(0)),
                                omega.omega,rhs0)
          || !AssembleGaugeInterior((ref.layer.enabled || lg.physical_trace_lapse)
              ? ResearchInnerTraceGauge(p,u,lg,InteriorLayerGauge(p,u,lg),false)
              : UnfactoredReferenceGauge(ref,p,u,gauge,true,true),
                                     omega.omega,gauge_rhs)) {
        ++bad;
        return;
      }
      // Subtract only the analytic Minkowski RHS's floating-point residual.
      rhs.chi -= rhs0.chi; rhs.trace -= rhs0.trace; rhs.theta -= rhs0.theta;
      for (int a = 0; a < 3; ++a) {
        rhs.lambda[a] -= rhs0.lambda[a];
        for (int b = 0; b < 3; ++b) {
          rhs.metric[a][b] -= rhs0.metric[a][b]; rhs.a[a][b] -= rhs0.a[a][b];
        }
      }
      AddMeshUpwindAdvectionWithVelocity<3>(udev,full.beta_u,idx,0,k,j,i,rhs,gauge_rhs);
      StoreMeshRHS(output,0,k,j,i,rhs,gauge_rhs);
      const int stride[3] = {1,g.n[0],g.n[0]*g.n[1]};
      for (int f = 0; f < Z4c::I_Z4C_BX; ++f) {
        const CartesianComponent component{dev,f,g.n[0],g.n[1]};
        result(0,f,k,j,i) += epsilon*InteriorKOSixth(component,inside,s,stride,spacing);
        if (!Kokkos::isfinite(result(0,f,k,j,i))) ++bad;
      }
    },failures);
    if (failures) throw std::runtime_error("invalid Cartesian conformal RHS");
  }

  // Match AthenaK's final-stage algebraic projection, restricted to active
  // cells. Invalid metrics are rejected rather than replaced by a determinant
  // floor. This does not project the differential Einstein/Z4 constraints.
  void ProjectAlgebraic(const DvceArray5D<Real> &data) const {
    CheckShape(data);
    const auto g = grid;
    const auto nodes = active;
    const auto q = BindCartesianFields(data);
    int failures = 0;
    Kokkos::parallel_reduce("masked conformal algebraic projection",nodes.extent(0),
        KOKKOS_LAMBDA(const int point, int &bad) {
      const int s = nodes(point), i = s%g.n[0], j = s/g.n[0]%g.n[1];
      const int k = s/(g.n[0]*g.n[1]);
      MetricJet<Real> metric{};
      for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) metric.g[a][b] = q.g_dd(0,a,b,k,j,i);
      const auto geo = Geometry(metric);
      if (!geo.valid) {
        ++bad;
        return;
      }
      const Real scale = 1/Kokkos::cbrt(geo.determinant);
      Real trace = 0;
      for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) trace += geo.inverse[a][b]*q.vA_dd(0,a,b,k,j,i);
      if (!Kokkos::isfinite(trace) || !Kokkos::isfinite(scale)) {
        ++bad;
        return;
      }
      for (int a = 0; a < 3; ++a)
      for (int b = a; b < 3; ++b) {
        q.g_dd(0,a,b,k,j,i) = scale*metric.g[a][b];
        q.vA_dd(0,a,b,k,j,i) -= metric.g[a][b]*trace/3;
      }
    },failures);
    if (failures) throw std::runtime_error("invalid metric in conformal projection");
  }

  Real MaxGaugeSpeed(const DvceArray5D<Real> &data) const {
    CheckShape(data);
    const auto g = grid;
    const auto nodes = active;
    const auto q = BindCartesianFields(data);
    const auto lg = layer_gauge;
    Real maximum = 0;
    Kokkos::parallel_reduce("layer gauge speed bound",nodes.extent(0),
        KOKKOS_LAMBDA(const int p, Real &speed) {
      const int s = nodes(p), i = s%g.n[0], j = s/g.n[0]%g.n[1], k = s/(g.n[0]*g.n[1]);
      MetricJet<Real> metric{};
      for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) metric.g[a][b] = q.g_dd(0,a,b,k,j,i);
      const auto geo = Geometry(metric);
      const Real x = g.first[0]+i*g.h[0];
      const Real y = g.first[1]+j*g.h[1], z = g.first[2]+k*g.h[2];
      const Real alpha = q.alpha(0,k,j,i), chi = q.chi(0,k,j,i);
      const auto c = LayerCoefficients(Kokkos::sqrt(x*x+y*y+z*z),alpha,lg);
      if (!geo.valid || !(alpha > 0) || !(chi > 0)) {
        speed = std::numeric_limits<Real>::infinity();
        return;
      }
      for (int a = 0; a < 3; ++a) {
        const Real v = Kokkos::abs(q.beta_u(0,a,k,j,i))
            +Kokkos::sqrt(c.alpha2f*chi*geo.inverse[a][a]);
        speed = Kokkos::fmax(speed,
            Kokkos::isfinite(v) ? v : std::numeric_limits<Real>::infinity());
      }
    },Kokkos::Max<Real>(maximum));
    if (!(maximum > 0) || !std::isfinite(maximum)) {
      throw std::runtime_error("invalid live layer gauge speed");
    }
    return maximum;
  }

  CartesianDiagnostics Diagnose(const DvceArray5D<Real> &data) const {
    Prepare(data);
    const auto g = grid;
    const auto ref = reference;
    const auto nodes = active;
    const auto q = BindCartesianFields(deviations);
    const auto initial_jets = reconstruction_jets;
    const Real damping = kappa1;
    Kokkos::View<Real **> diagnostics("Cartesian constraints",active.extent(0),12);
    int failures = 0;
    Kokkos::parallel_reduce("Cartesian physical constraint diagnostics",active.extent(0),
        KOKKOS_LAMBDA(const int point, int &bad) {
      const int s = nodes(point), i = s%g.n[0], j = s/g.n[0]%g.n[1];
      const int k = s/(g.n[0]*g.n[1]);
      const Real idx[3] = {1/g.h[0],1/g.h[1],1/g.h[2]};
      const auto p = ref.At(g.first[0]+i*g.h[0],g.first[1]+j*g.h[1],g.first[2]+k*g.h[2]);
      auto u = LoadMeshJet<3>(q,idx,0,k,j,i);
      if (initial_jets.extent(0)) AddBackgroundJet(u,initial_jets(point));
      else AddReferenceJet(u,p,ref);
      if (!(u.alpha.value > 0) || !Kokkos::isfinite(u.alpha.value)) {
        ++bad;
        return;
      }
      const auto c = EvolvedConstraints(u,CartesianOmega(u,p));
      if (!c.valid || !c.z4.valid) {
        ++bad;
        return;
      }
      diagnostics(point,0) = c.hamiltonian;
      diagnostics(point,1) = Kokkos::sqrt(c.momentum_conformal_norm2);
      diagnostics(point,2) = Kokkos::sqrt(c.z4.z_conformal_norm2);
      diagnostics(point,3) = u.theta.value;
      diagnostics(point,4) = Kokkos::abs(c.z4.determinant_residual);
      diagnostics(point,5) = Kokkos::abs(c.z4.tracefree_residual);
      diagnostics(point,6) = u.alpha.value;
      diagnostics(point,7) = u.chi.value;
      Real deviation = 0;
      for (int f = 0; f < Z4c::nz4c; ++f) deviation = Kokkos::fmax(deviation,
          Kokkos::abs(data(0,f,k,j,i)-ReferenceComponent(f,p)));
      diagnostics(point,8) = deviation;
      diagnostics(point,9) = diagnostics(point,10) = diagnostics(point,11) = 0;
      // An outer interior shell, not an evaluation of the continuum scri limit.
      const Real radius = p.radius;
      const Real width = 2*Kokkos::fmax(g.h[0],Kokkos::fmax(g.h[1],g.h[2]));
      if (radius > g.radius-width) {
        const auto parts = ConformalRHS(u,CartesianOmega(u,p),
                                        damping/u.alpha.value,Real(0));
        if (!parts.valid) {
          ++bad;
          return;
        }
        const auto pole = parts.pole;
        Z4cJet<Real> background{};
        AddReferenceJet(background,p,ref);
        const auto pole0 = ConformalRHS(background,CartesianOmega(background,p),
                                        damping,Real(0)).pole;
        Real delta = Kokkos::fmax(Kokkos::abs(pole.chi-pole0.chi),
                                  Kokkos::abs(pole.trace-pole0.trace));
        delta = Kokkos::fmax(delta,Kokkos::abs(pole.theta-pole0.theta));
        Real maximum = Kokkos::fmax(Kokkos::abs(pole.chi),Kokkos::abs(pole.trace));
        maximum = Kokkos::fmax(maximum,Kokkos::abs(pole.theta));
        for (int a = 0; a < 3; ++a) {
          maximum = Kokkos::fmax(maximum,Kokkos::abs(pole.lambda[a]));
          delta = Kokkos::fmax(delta,Kokkos::abs(pole.lambda[a]-pole0.lambda[a]));
          for (int b = 0; b < 3; ++b) {
            maximum = Kokkos::fmax(maximum,Kokkos::abs(pole.metric[a][b]));
            maximum = Kokkos::fmax(maximum,Kokkos::abs(pole.a[a][b]));
            delta = Kokkos::fmax(delta,Kokkos::abs(pole.metric[a][b]-pole0.metric[a][b]));
            delta = Kokkos::fmax(delta,Kokkos::abs(pole.a[a][b]-pole0.a[a][b]));
          }
        }
        diagnostics(point,9) = maximum;
        diagnostics(point,11) = delta;
        diagnostics(point,10) = Kokkos::abs(c.null_residual
            -p.omega*p.omega*p.n_residue);
      }
    },failures);
    if (failures) throw std::runtime_error("invalid Cartesian constraints");
    const auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),diagnostics);
    const auto host_nodes = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(),nodes);
    CartesianDiagnostics out;
    for (size_t p = 0; p < active.extent(0); ++p) {
      for (int f = 0; f < 12; ++f) if (!std::isfinite(h(p,f))) {
        throw std::runtime_error("nonfinite Cartesian diagnostic");
      }
      out.h_l2 += h(p,0)*h(p,0); out.m_l2 += h(p,1)*h(p,1);
      out.z_l2 += h(p,2)*h(p,2); out.theta_l2 += h(p,3)*h(p,3);
      const int s = host_nodes(p);
      const double x = g.first[0]+(s%g.n[0])*g.h[0];
      const double y = g.first[1]+(s/g.n[0]%g.n[1])*g.h[1];
      const double z = g.first[2]+(s/(g.n[0]*g.n[1]))*g.h[2];
      const double radius = std::sqrt(x*x+y*y+z*z);
      if (std::abs(h(p,0)) > out.max_h) {
        out.max_h = std::abs(h(p,0));
        out.max_h_radius = radius;
      }
      if (std::abs(h(p,1)) > out.max_m) {
        out.max_m = std::abs(h(p,1));
        out.max_m_radius = radius;
      }
      out.max_det = std::max(out.max_det,h(p,4));
      out.max_trace = std::max(out.max_trace,h(p,5));
      out.min_alpha = std::min(out.min_alpha,h(p,6));
      out.min_chi = std::min(out.min_chi,h(p,7));
      out.max_deviation = std::max(out.max_deviation,h(p,8));
      out.shell_max_pole = std::max(out.shell_max_pole,h(p,9));
      out.shell_max_pole_deviation = std::max(out.shell_max_pole_deviation,h(p,11));
      out.shell_max_null_deviation = std::max(out.shell_max_null_deviation,h(p,10));
    }
    out.h_l2 = std::sqrt(out.h_l2/active.extent(0));
    out.m_l2 = std::sqrt(out.m_l2/active.extent(0));
    out.z_l2 = std::sqrt(out.z_l2/active.extent(0));
    out.theta_l2 = std::sqrt(out.theta_l2/active.extent(0));
    return out;
  }
};

}  // namespace hyperboloidal
}  // namespace z4c
#endif  // Z4C_HYPERBOLOIDAL_CARTESIAN_PATCH_HPP_
