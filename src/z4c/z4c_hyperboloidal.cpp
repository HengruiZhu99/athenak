// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include "globals.hpp"  // NOLINT(build/include_subdir)
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "z4c/z4c.hpp"
#include "z4c/hyperboloidal/cartesian_trumpet.hpp"

namespace z4c {
namespace hyp = hyperboloidal;

void Z4c::SetupHyperboloidal(ParameterInput *pin) {
  if (!pin->GetOrAddBoolean("z4c","hyperboloidal",false)) return;
  const auto &ind = pmy_pack->pmesh->mb_indcs;
  if (global_variable::nranks != 1 || pmy_pack->nmb_thispack != 1
      || u0.extent(0) != 1 || !pmy_pack->pmesh->three_d
      || pmy_pack->pmesh->multilevel || ind.ng != 3 || opt.fd_stencil != 3
      || opt.chi_psi_power != -4 || opt.floor_chi || opt.enable_driftcontrol
      || nrad != 0 || !ptracker.empty()
      || pin->DoesBlockExist("hydro") || pin->DoesBlockExist("mhd")
      || pin->DoesBlockExist("radiation") || pin->DoesBlockExist("ion-neutral")
      || pin->DoesBlockExist("gravity") || pin->DoesBlockExist("particles")
      || pin->GetOrAddInteger("fastflow","num_horizons",0) != 0
      || pin->GetOrAddInteger("cce","num_radii",0) != 0
      || pin->GetString("time","integrator") != "rk3"
      || pin->GetString("problem","pgen_name") != "z4c_hyperboloidal") {
    throw std::invalid_argument("hyperboloidal prototype requires one uniform 3D block, "
        "ng=3, fourth-order Z4c, chi power -4, rk3, and its dedicated vacuum pgen; "
        "matter, MPI, AMR, floors, trackers and extraction are unsupported");
  }
  hyp::SphericalGhostGrid grid;
  grid.radius = 1;
  const auto size = pmy_pack->pmb->mb_size.h_view(0);
  const Real lower[3] = {size.x1min,size.x2min,size.x3min};
  const Real upper[3] = {size.x1max,size.x2max,size.x3max};
  const Real h[3] = {size.dx1,size.dx2,size.dx3};
  for (int d = 0; d < 3; ++d) {
    if (lower[d] > -1 || upper[d] < 1) {
      throw std::invalid_argument("hyperboloidal sphere must fit inside physical mesh");
    }
    grid.n[d] = u0.extent_int(4-d);
    grid.h[d] = h[d];
    grid.first[d] = lower[d]+(0.5-ind.ng)*h[d];
  }
  hyperboloidal_patch = std::make_unique<hyp::CartesianConformalPatch>(grid,1,
      pin->GetOrAddInteger("z4c","hyperboloidal_ghost_degree",3));
  hyperboloidal_mass_diagnostics =
      pin->GetOrAddBoolean("z4c","hyperboloidal_mass_diagnostics",false);
  hyperboloidal_mass_nmu = pin->GetOrAddInteger("z4c","hyperboloidal_mass_nmu",32);
  if (hyperboloidal_mass_nmu < 4 || hyperboloidal_mass_nmu > 128) {
    throw std::invalid_argument("invalid Hawking quadrature");
  }
  auto &patch = *hyperboloidal_patch;
  patch.kappa1 = pin->GetOrAddReal("z4c","hyperboloidal_kappa1",5);
  patch.dissipation = pin->GetOrAddReal("z4c","hyperboloidal_dissipation",0.1);
  hyperboloidal_pole_cfl = pin->GetOrAddReal("z4c","hyperboloidal_pole_cfl",0.04);
  if (!std::isfinite(patch.kappa1) || patch.kappa1 < 0
      || !std::isfinite(patch.dissipation) || patch.dissipation < 0
      || !std::isfinite(pmy_pack->pmesh->cfl_no) || pmy_pack->pmesh->cfl_no <= 0
      || !std::isfinite(hyperboloidal_pole_cfl) || hyperboloidal_pole_cfl <= 0
      || hyperboloidal_pole_cfl > 0.2) {
    throw std::invalid_argument(
        "invalid hyperboloidal damping, dissipation or timestep coefficient");
  }
  const Real mass = pin->GetOrAddReal("problem","mass",0);
  if (!std::isfinite(mass) || mass < 0) {
    throw std::invalid_argument("invalid trumpet mass");
  }
  // Construct the immutable analytic initial profile before restart loading.
  // A restart must not pair its evolved values with the initial analytic jets.
  if (mass > 0) hyp::InitializeCartesianTrumpet(patch,u0,mass);
  else patch.InitializeReference(u0);
  pin->SetBoolean("adm","separate_z4c_gauge",true);
  hyperboloidal_active = DvceArray5D<Real>("hyperboloidal active domain",1,1,
                                         grid.n[2],grid.n[1],grid.n[0]);
  const auto output = hyperboloidal_active;
  const auto mask = patch.mask;
  Kokkos::parallel_for("export hyperboloidal mask",mask.extent(0),
      KOKKOS_LAMBDA(const int s) {
    output(0,0,s/(grid.n[0]*grid.n[1]),s/grid.n[0]%grid.n[1],s%grid.n[0]) = mask(s);
  });
}

void Z4c::InitializeHyperboloidal(ParameterInput *pin, bool restart) {
  if (!restart) {
    const Real amplitude = pin->GetOrAddReal("problem","lapse_pulse",0);
    if (!std::isfinite(amplitude)) throw std::invalid_argument("invalid lapse pulse");
    const auto q = z4c;
    const auto grid = hyperboloidal_patch->grid;
    const auto nodes = hyperboloidal_patch->active;
    Kokkos::parallel_for("hyperboloidal lapse pulse",nodes.extent(0),
        KOKKOS_LAMBDA(const int p) {
      const int s = nodes(p), i = s%grid.n[0], j = s/grid.n[0]%grid.n[1];
      const int k = s/(grid.n[0]*grid.n[1]);
      const Real x = grid.first[0]+i*grid.h[0], y = grid.first[1]+j*grid.h[1];
      const Real z = grid.first[2]+k*grid.h[2], r2 = x*x+y*y+z*z;
      q.alpha(0,k,j,i) += amplitude*Kokkos::pow(1-r2,4)*Kokkos::exp(-4*r2)
          *(1+0.2*x+0.3*y*z);
    });
  }
  HyperboloidalADM();
  HyperboloidalConstraints();
}

void Z4c::HyperboloidalADM() {
  auto *padm = pmy_pack->padm;
  padm->EnsureSeparateGaugeStorage();
  if (!padm->separate_z4c_gauge || padm->gauge_is_shared) {
    throw std::runtime_error("conformal ADM gauge alias");
  }
  Kokkos::deep_copy(padm->u_adm,std::numeric_limits<Real>::quiet_NaN());
  const auto a = padm->adm;
  const auto q = z4c;
  const auto g = hyperboloidal_patch->grid;
  const auto ref = hyperboloidal_patch->reference;
  const auto nodes = hyperboloidal_patch->active;
  int failures = 0;
  Kokkos::parallel_reduce("interior physical ADM conversion",nodes.extent(0),
      KOKKOS_LAMBDA(const int p, int &bad) {
    const int s = nodes(p), i = s%g.n[0], j = s/g.n[0]%g.n[1], k = s/(g.n[0]*g.n[1]);
    hyp::Z4cJet<Real> u{};
    u.chi.value = q.chi(0,k,j,i); u.alpha.value = q.alpha(0,k,j,i);
    u.trace.value = q.vKhat(0,k,j,i); u.theta.value = q.vTheta(0,k,j,i);
    for (int d = 0; d < 3; ++d) {
      u.beta.value[d] = q.beta_u(0,d,k,j,i);
      for (int e = 0; e < 3; ++e) {
        u.metric.g[d][e] = q.g_dd(0,d,e,k,j,i);
        u.a.k[d][e] = q.vA_dd(0,d,e,k,j,i);
      }
    }
    const auto point = ref.At(g.first[0]+i*g.h[0],g.first[1]+j*g.h[1],
                              g.first[2]+k*g.h[2]);
    const auto physical = hyp::ToPhysicalADM(u,point.omega);
    if (!physical.valid) {
      ++bad;
      return;
    }
    a.psi4(0,k,j,i) = physical.psi4; a.alpha(0,k,j,i) = physical.alpha;
    for (int d = 0; d < 3; ++d) {
      a.beta_u(0,d,k,j,i) = physical.beta[d];
      for (int e = d; e < 3; ++e) {
        a.g_dd(0,d,e,k,j,i) = physical.metric[d][e];
        a.vK_dd(0,d,e,k,j,i) = physical.curvature[d][e];
      }
    }
  },failures);
  if (failures) throw std::runtime_error("invalid hyperboloidal physical ADM state");
}

void Z4c::HyperboloidalConstraints() {
  auto &patch = *hyperboloidal_patch;
  patch.Prepare(u0);
  Kokkos::deep_copy(u_con,Real(0));
  const auto q = hyp::BindCartesianFields(patch.deviations);
  const auto jets = patch.reconstruction_jets;
  const auto g = patch.grid;
  const auto ref = patch.reference;
  const auto nodes = patch.active;
  const auto dst = con;
  int failures = 0;
  Kokkos::parallel_reduce("hyperboloidal physical constraints",nodes.extent(0),
      KOKKOS_LAMBDA(const int p, int &bad) {
    const int s = nodes(p), i = s%g.n[0], j = s/g.n[0]%g.n[1], k = s/(g.n[0]*g.n[1]);
    const Real idx[3] = {1/g.h[0],1/g.h[1],1/g.h[2]};
    const auto point = ref.At(g.first[0]+i*g.h[0],g.first[1]+j*g.h[1],
                              g.first[2]+k*g.h[2]);
    auto u = hyp::LoadMeshJet<3>(q,idx,0,k,j,i);
    if (jets.extent(0)) hyp::AddBackgroundJet(u,jets(p));
    else hyp::AddReferenceJet(u,point,ref);
    const auto c = hyp::EvolvedConstraints(u,hyp::CartesianOmega(u,point));
    if (!c.valid || !c.z4.valid) {
      ++bad;
      return;
    }
    dst.H(0,k,j,i) = c.hamiltonian;
    dst.M(0,k,j,i) = c.momentum_conformal_norm2;
    dst.Z(0,k,j,i) = c.z4.z_conformal_norm2;
    dst.C(0,k,j,i) = c.hamiltonian*c.hamiltonian+c.momentum_conformal_norm2
        +c.z4.z_conformal_norm2+u.theta.value*u.theta.value;
    for (int d = 0; d < 3; ++d) dst.M_d(0,d,k,j,i) = c.momentum[d];
  },failures);
  if (failures) throw std::runtime_error("invalid native hyperboloidal constraints");
}
}  // namespace z4c
