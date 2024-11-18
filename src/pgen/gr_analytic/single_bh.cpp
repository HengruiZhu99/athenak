//========================================================================================
// Athena++ astrophysical MHD code, Kokkos version
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dynbbh.cpp
//! \brief Problem generator for superimposed Kerr-Schild black holes

#include <math.h>
#include <algorithm>
#include <sstream>
#include <string>
#include <iostream>

#include "parameter_input.hpp"
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_amr.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"

#define h 5e-7
#define D2(comp, h) ((met_p1.g).comp - (met_m1.g).comp) / (2*h)

namespace {

enum {
  TT, XX, YY, ZZ, NDIM
};

enum {
  X1, Y1, Z1,
  VX1, VY1, VZ1,
  AX1, AY1, AZ1,
  M1T, NPARAM
};

struct dd_sym {
  Real tt;
  Real tx;
  Real ty;
  Real tz;
  Real xx;
  Real xy;
  Real xz;
  Real yy;
  Real yz;
  Real zz;
};

// spacetime metric and its derivatives
struct four_metric {
  struct dd_sym g;
  // partial derivatives
  struct dd_sym g_t;
  struct dd_sym g_x;
  struct dd_sym g_y;
  struct dd_sym g_z;
};

// adm quantities
struct adm_var {
  Real gxx;
  Real gxy;
  Real gxz;
  Real gyy;
  Real gyz;
  Real gzz;
  Real alpha;
  Real betax;
  Real betay;
  Real betaz;
  Real kxx;
  Real kxy;
  Real kxz;
  Real kyy;
  Real kyz;
  Real kzz;
};

// parameters for bh
struct bh_pgen {
  Real x1, y1, z1;
  Real vx1, vy1, vz1;
  Real m1, ax1, ay1, az1;
};

struct bh_pgen bh;

/* Declare functions */

// spacetime metric and its partial derivative, used to evaluate ADM quantities
KOKKOS_INLINE_FUNCTION
void numerical_4metric(const Real t, const Real x, const Real y,
    const Real z, struct four_metric &outmet, const bh_pgen& bh_);

// 3+1 decomposition of the spacetime variables
KOKKOS_INLINE_FUNCTION
int four_metric_to_three_metric(const Real t, const Real x, const Real y,
    const Real z, const struct four_metric &met, struct adm_var &gam,
    const bh_pgen& bh_);

// Change slicing to boosted slice
// t0,x0,y0,z0 are the corresponding coordinate in the unboosted frame
KOKKOS_INLINE_FUNCTION
void BoostSlice(const Real beta[3], const Real x_in[4], Real x_out[4]);

// Boost Tensor
KOKKOS_INLINE_FUNCTION
void BoostMetric(const Real beta[3], const Real gcov[4][4], Real g_out[4][4]);

// construct boosted spacetime metric
KOKKOS_INLINE_FUNCTION
void SpaceTimeMetric(const Real t, const Real x, const Real y, const Real z,
                struct four_metric &met, const bh_pgen& bh_);

KOKKOS_INLINE_FUNCTION
void BHPuncture(const Real x0[4], const Real mass, Real gcov[NDIM][NDIM]);

KOKKOS_INLINE_FUNCTION
Real EvaluateLapse(const Real x0[4], const bh_pgen& bh_);

// call this to set the u_adm array
void SetADMVariables(MeshBlockPack *pmbp);

// how to perform AMR
void RefinementCondition(MeshBlockPack* pmbp);
} // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::ShockTube_()
//! \brief Problem Generator for the shock tube (Riemann problem) tests

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  user_ref_func  = RefinementCondition;

  if (restart) return;

  // location
  bh.x1 = pin->GetOrAddReal("problem", "bh_x", -10.0);
  bh.y1 = pin->GetOrAddReal("problem", "bh_y", 0.0);
  bh.z1 = pin->GetOrAddReal("problem", "bh_z", 0.0);

  // boost
  bh.vx1 = pin->GetOrAddReal("problem", "bh_vx", 0.0);
  bh.vy1 = pin->GetOrAddReal("problem", "bh_vy", 0.0);
  bh.vz1 = pin->GetOrAddReal("problem", "bh_vz", 0.0);

  // mass and spin
  bh.m1 = pin->GetOrAddReal("problem", "bh_bare_mass", 1.0);
  bh.ax1 = pin->GetOrAddReal("problem", "bh_ax", 0.0);
  bh.ay1 = pin->GetOrAddReal("problem", "bh_ay", 0.0);
  bh.az1 = pin->GetOrAddReal("problem", "bh_az", 0.0);

  // capture variables for the kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  int nmb = pmbp->nmb_thispack;
  auto &bh_ = bh;

  // Initialize ADM variables -------------------------------
  SetADMVariables(pmbp);
  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin);
            break;
    case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin);
            break;
    case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin);
            break;
  }
  pmbp->pz4c->Z4cToADM(pmbp);
  pmbp->pz4c->GaugePreCollapsedLapse(pmbp, pin);
  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMConstraints<2>(pmbp);
            break;
    case 3: pmbp->pz4c->ADMConstraints<3>(pmbp);
            break;
    case 4: pmbp->pz4c->ADMConstraints<4>(pmbp);
            break;
  }
  std::cout<<"Single Black Hole initialized."<<std::endl;

  return;
}

namespace {

void SetADMVariables(MeshBlockPack *pmbp) {
  const Real tt = pmbp->pmesh->time;
  auto &adm = pmbp->padm->adm;
  auto &size = pmbp->pmb->mb_size;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;

  auto& bh_ = bh;

  par_for("update_adm_vars", DevExeSpace(), 0,nmb-1,0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    struct four_metric met4;
    struct adm_var met3;
    numerical_4metric(tt, x1v, x2v, x3v, met4, bh_);

    /* Transform 4D metric to 3+1 variables*/
    four_metric_to_three_metric(tt, x1v, x2v, x3v, met4, met3, bh_);

    /* Load (Cartesian) components of the metric and curvature */
    // g_ab
    adm.g_dd(m,0,0,k,j,i) = met3.gxx;
    adm.g_dd(m,0,1,k,j,i) = met3.gxy;
    adm.g_dd(m,0,2,k,j,i) = met3.gxz;
    adm.g_dd(m,1,1,k,j,i) = met3.gyy;
    adm.g_dd(m,1,2,k,j,i) = met3.gyz;
    adm.g_dd(m,2,2,k,j,i) = met3.gzz;

    adm.vK_dd(m,0,0,k,j,i) = met3.kxx;
    adm.vK_dd(m,0,1,k,j,i) = met3.kxy;
    adm.vK_dd(m,0,2,k,j,i) = met3.kxz;
    adm.vK_dd(m,1,1,k,j,i) = met3.kyy;
    adm.vK_dd(m,1,2,k,j,i) = met3.kyz;
    adm.vK_dd(m,2,2,k,j,i) = met3.kzz;

    adm.alpha(m,k,j,i) = met3.alpha;
    adm.beta_u(m,0,k,j,i) = met3.betax;
    adm.beta_u(m,1,k,j,i) = met3.betay;
    adm.beta_u(m,2,k,j,i) = met3.betaz;
  });
  return;
}

KOKKOS_INLINE_FUNCTION
void numerical_4metric(const Real t, const Real x, const Real y,
    const Real z, struct four_metric &outmet, const bh_pgen& bh_) {
  struct four_metric met_m1;
  struct four_metric met_p1;

  // Time
  SpaceTimeMetric(t-1*h, x, y, z, met_m1, bh_);
  SpaceTimeMetric(t+1*h, x, y, z, met_p1, bh_);
  SpaceTimeMetric(t, x, y, z, outmet, bh_);

  outmet.g_t.tt = D2(tt, h);
  outmet.g_t.tx = D2(tx, h);
  outmet.g_t.ty = D2(ty, h);
  outmet.g_t.tz = D2(tz, h);
  outmet.g_t.xx = D2(xx, h);
  outmet.g_t.xy = D2(xy, h);
  outmet.g_t.xz = D2(xz, h);
  outmet.g_t.yy = D2(yy, h);
  outmet.g_t.yz = D2(yz, h);
  outmet.g_t.zz = D2(zz, h);

  // X
  SpaceTimeMetric(t, x-1*h, y, z, met_m1, bh_);
  SpaceTimeMetric(t, x+1*h, y, z, met_p1, bh_);

  outmet.g_x.tt = D2(tt, h);
  outmet.g_x.tx = D2(tx, h);
  outmet.g_x.ty = D2(ty, h);
  outmet.g_x.tz = D2(tz, h);
  outmet.g_x.xx = D2(xx, h);
  outmet.g_x.xy = D2(xy, h);
  outmet.g_x.xz = D2(xz, h);
  outmet.g_x.yy = D2(yy, h);
  outmet.g_x.yz = D2(yz, h);
  outmet.g_x.zz = D2(zz, h);

  // Y
  SpaceTimeMetric(t, x, y-1*h, z, met_m1, bh_);
  SpaceTimeMetric(t, x, y+1*h, z, met_p1, bh_);

  outmet.g_y.tt = D2(tt, h);
  outmet.g_y.tx = D2(tx, h);
  outmet.g_y.ty = D2(ty, h);
  outmet.g_y.tz = D2(tz, h);
  outmet.g_y.xx = D2(xx, h);
  outmet.g_y.xy = D2(xy, h);
  outmet.g_y.xz = D2(xz, h);
  outmet.g_y.yy = D2(yy, h);
  outmet.g_y.yz = D2(yz, h);
  outmet.g_y.zz = D2(zz, h);

  // Z
  SpaceTimeMetric(t, x, y, z-1*h, met_m1, bh_);
  SpaceTimeMetric(t, x, y, z+1*h, met_p1, bh_);

  outmet.g_z.tt = D2(tt, h);
  outmet.g_z.tx = D2(tx, h);
  outmet.g_z.ty = D2(ty, h);
  outmet.g_z.tz = D2(tz, h);
  outmet.g_z.xx = D2(xx, h);
  outmet.g_z.xy = D2(xy, h);
  outmet.g_z.xz = D2(xz, h);
  outmet.g_z.yy = D2(yy, h);
  outmet.g_z.yz = D2(yz, h);
  outmet.g_z.zz = D2(zz, h);

  return;
}

KOKKOS_INLINE_FUNCTION
int four_metric_to_three_metric(const Real t, const Real x, const Real y,
    const Real z, const struct four_metric &met, struct adm_var &gam,
    const bh_pgen& bh_) {
  /* Check determinant first */
  gam.gxx = met.g.xx;
  gam.gxy = met.g.xy;
  gam.gxz = met.g.xz;
  gam.gyy = met.g.yy;
  gam.gyz = met.g.yz;
  gam.gzz = met.g.zz;

  Real det = adm::SpatialDet(gam.gxx, gam.gxy, gam.gxz,
                                   gam.gyy, gam.gyz, gam.gzz);

  /* Compute components if detg is not <0 */
  Real betadownx = met.g.tx;
  Real betadowny = met.g.ty;
  Real betadownz = met.g.tz;

  Real dbetadownxx = met.g_x.tx;
  Real dbetadownyx = met.g_x.ty;
  Real dbetadownzx = met.g_x.tz;

  Real dbetadownxy = met.g_y.tx;
  Real dbetadownyy = met.g_y.ty;
  Real dbetadownzy = met.g_y.tz;

  Real dbetadownxz = met.g_z.tx;
  Real dbetadownyz = met.g_z.ty;
  Real dbetadownzz = met.g_z.tz;

  Real dtgxx = met.g_t.xx;
  Real dtgxy = met.g_t.xy;
  Real dtgxz = met.g_t.xz;
  Real dtgyy = met.g_t.yy;
  Real dtgyz = met.g_t.yz;
  Real dtgzz = met.g_t.zz;

  Real dgxxx = met.g_x.xx;
  Real dgxyx = met.g_x.xy;
  Real dgxzx = met.g_x.xz;
  Real dgyyx = met.g_x.yy;
  Real dgyzx = met.g_x.yz;
  Real dgzzx = met.g_x.zz;

  Real dgxxy = met.g_y.xx;
  Real dgxyy = met.g_y.xy;
  Real dgxzy = met.g_y.xz;
  Real dgyyy = met.g_y.yy;
  Real dgyzy = met.g_y.yz;
  Real dgzzy = met.g_y.zz;

  Real dgxxz = met.g_z.xx;
  Real dgxyz = met.g_z.xy;
  Real dgxzz = met.g_z.xz;
  Real dgyyz = met.g_z.yy;
  Real dgyzz = met.g_z.yz;
  Real dgzzz = met.g_z.zz;

  Real idetgxx = -gam.gyz * gam.gyz + gam.gyy * gam.gzz;
  Real idetgxy = gam.gxz * gam.gyz - gam.gxy * gam.gzz;
  Real idetgxz = -(gam.gxz * gam.gyy) + gam.gxy * gam.gyz;
  Real idetgyy = -gam.gxz * gam.gxz + gam.gxx * gam.gzz;
  Real idetgyz = gam.gxy * gam.gxz - gam.gxx * gam.gyz;
  Real idetgzz = -gam.gxy * gam.gxy + gam.gxx * gam.gyy;

  Real invgxx = idetgxx / det;
  Real invgxy = idetgxy / det;
  Real invgxz = idetgxz / det;
  Real invgyy = idetgyy / det;
  Real invgyz = idetgyz / det;
  Real invgzz = idetgzz / det;

  gam.betax =
    betadownx * invgxx + betadowny * invgxy + betadownz * invgxz;

  gam.betay =
    betadownx * invgxy + betadowny * invgyy + betadownz * invgyz;

  gam.betaz =
    betadownx * invgxz + betadowny * invgyz + betadownz * invgzz;

  Real b2 =
    betadownx * gam.betax + betadowny * gam.betay +
    betadownz * gam.betaz;

  // specify alpha by hand if using isotropic coordinate
  // otherwise cannot specify the sign to guarantee smoothness
  if (true) {
    // black hole location
    Real xi1x = bh_.x1;
    Real xi1y = bh_.y1;
    Real xi1z = bh_.z1;

    // velocity
    Real vel[3];
    vel[0] = bh_.vx1;
    vel[1] = bh_.vy1;
    vel[2] = bh_.vz1;

    // coordinate frame where BH is at the origin at t=0
    Real xboosted[4];
    xboosted[0] = t;
    xboosted[1] = x-xi1x;
    xboosted[2] = y-xi1y;
    xboosted[3] = z-xi1z;

    // rest frame where the analytical metric is evaluated
    Real x0[4];
    
    // evaluate coordinate in the rest frame
    BoostSlice(vel, xboosted, x0);
    gam.alpha = EvaluateLapse(x0, bh_);
  } else {
    gam.alpha = sqrt(fabs(b2 - met.g.tt));
  }

  gam.kxx = -(-2 * dbetadownxx - gam.betax * dgxxx - gam.betay * dgxxy -
    gam.betaz * dgxxz + 2 * (gam.betax * dgxxx + gam.betay * dgxyx +
      gam.betaz * dgxzx) + dtgxx) / (2. * gam.alpha);

  gam.kxy = -(-dbetadownxy - dbetadownyx + gam.betax * dgxxy -
    gam.betaz * dgxyz + gam.betaz * dgxzy + gam.betay * dgyyx +
    gam.betaz * dgyzx + dtgxy) / (2. * gam.alpha);

  gam.kxz = -(-dbetadownxz - dbetadownzx + gam.betax * dgxxz +
    gam.betay * dgxyz - gam.betay * dgxzy + gam.betay * dgyzx +
    gam.betaz * dgzzx + dtgxz) / (2. * gam.alpha);

  gam.kyy = -(-2 * dbetadownyy - gam.betax * dgyyx - gam.betay * dgyyy -
    gam.betaz * dgyyz + 2 * (gam.betax * dgxyy + gam.betay * dgyyy +
      gam.betaz * dgyzy) + dtgyy) / (2. * gam.alpha);

  gam.kyz = -(-dbetadownyz - dbetadownzy + gam.betax * dgxyz +
    gam.betax * dgxzy + gam.betay * dgyyz - gam.betax * dgyzx +
    gam.betaz * dgzzy + dtgyz) / (2. * gam.alpha);

  gam.kzz = -(-2 * dbetadownzz - gam.betax * dgzzx - gam.betay * dgzzy -
    gam.betaz * dgzzz + 2 * (gam.betax * dgxzz + gam.betay * dgyzz +
      gam.betaz * dgzzz) + dtgzz) / (2. * gam.alpha);
  return 0;
}

KOKKOS_INLINE_FUNCTION
void BHPuncture(const Real x0[4], const Real mass, Real gcov[NDIM][NDIM]) {
  Real r0 = std::pow(x0[1]*x0[1]+x0[2]*x0[2]+x0[3]*x0[3],0.5);
  Real psi0 = 1 + 0.5*mass/r0;
  Real alpha0 = (1 - 0.5*mass/r0)/psi0;
  gcov[0][0] = -std::pow(alpha0,2);
  gcov[0][1] =  0.;
  gcov[0][2] =  0.;
  gcov[0][3] =  0.;

  gcov[1][0] =  0.;
  gcov[1][1] =  std::pow(psi0,4);
  gcov[1][2] =  0.;
  gcov[1][3] =  0.;

  gcov[2][0] =  0.;
  gcov[2][1] =  0.;
  gcov[2][2] =  std::pow(psi0,4);
  gcov[2][3] =  0.;

  gcov[3][0] =  0.;
  gcov[3][1] =  0.;
  gcov[3][2] =  0.;
  gcov[3][3] =  std::pow(psi0,4);
}

KOKKOS_INLINE_FUNCTION
Real EvaluateLapse(const Real x0[4], const bh_pgen& bh_) {
  Real mass = bh_.m1;
  Real r0 = std::pow(x0[1]*x0[1]+x0[2]*x0[2]+x0[3]*x0[3],0.5);
  Real psi0 = 1 + 0.5*mass/r0;
  Real alpha0 = (1 - 0.5*mass/r0)/psi0;
  Real v2 = bh_.vx1*bh_.vx1 + bh_.vy1*bh_.vy1 + bh_.vz1*bh_.vz1;
  Real gamma2 = 1/(1-v2);
  Real B0 = std::pow(fabs(gamma2*(1-v2*std::pow(alpha0,2)*std::pow(psi0,-4))),0.5);
  return alpha0/B0;
}

KOKKOS_INLINE_FUNCTION
void SpaceTimeMetric(const Real t,
                const Real x,
                const Real y,
                const Real z,
                struct four_metric &met,
                const bh_pgen& bh_) {
  // black hole location
  Real xi1x = bh_.x1;
  Real xi1y = bh_.y1;
  Real xi1z = bh_.z1;

  Real mass = bh_.m1;

  // velocity
  Real vel[3];
  vel[0] = bh_.vx1;
  vel[1] = bh_.vy1;
  vel[2] = bh_.vz1;

  // coordinate frame where BH is at the origin at t=0
  Real xboosted[4];
  xboosted[0] = t;
  xboosted[1] = x-xi1x;
  xboosted[2] = y-xi1y;
  xboosted[3] = z-xi1z;

  // rest frame where the analytical metric is evaluated
  Real x0[4];
  
  // evaluate coordinate in the rest frame
  BoostSlice(vel, xboosted, x0);

  // Two functions, one to evaluate the new Lorentz boosted coordinate, another to perform the lorentz boost
  // for arbitrary four velocity. For now just do the boost along x direction. 

  Real grest[NDIM][NDIM];
  Real gcov[NDIM][NDIM];

  // change between puncture and KerrSchild coordinate, for now only puncture implemented
  if (true) {
    BHPuncture(x0, mass, grest);
  }

  BoostMetric(vel, grest, gcov);
  // Lorentz Boost the metric

  met.g.tt = gcov[TT][TT];
  met.g.tx = gcov[TT][XX];
  met.g.ty = gcov[TT][YY];
  met.g.tz = gcov[TT][ZZ];
  met.g.xx = gcov[XX][XX];
  met.g.xy = gcov[XX][YY];
  met.g.xz = gcov[XX][ZZ];
  met.g.yy = gcov[YY][YY];
  met.g.yz = gcov[YY][ZZ];
  met.g.zz = gcov[ZZ][ZZ];

  return;
}

KOKKOS_INLINE_FUNCTION
void BoostSlice(const Real beta[3], const Real x_in[4], Real x_out[4]) {
  // Compute beta^2
  Real beta2 = beta[0]*beta[0] + beta[1]*beta[1] + beta[2]*beta[2];

  // If beta^2 is zero (no boost), copy x_in to x_out
  if (beta2 == 0.0) {
    for (int i = 0; i < 4; ++i) {
      x_out[i] = x_in[i];
    }
    return;
  }

  // Compute gamma factor
  Real gamma = 1.0 / sqrt(1.0 - beta2);

  // Compute beta dot x (spatial components)
  Real beta_dot_x = beta[0]*x_in[1] + beta[1]*x_in[2] + beta[2]*x_in[3];

  // Compute the time component x'^0
  x_out[0] = gamma * (x_in[0] - beta_dot_x);

  // Compute the common factor for spatial components
  Real factor = ((gamma - 1.0) * beta_dot_x / beta2) - gamma * x_in[0];

  // Compute the spatial components x'^i
  for (int i = 0; i < 3; ++i) {
    x_out[i+1] = x_in[i+1] + beta[i] * factor;
  }
}

KOKKOS_INLINE_FUNCTION
void BoostMetric(const Real beta[3], const Real gcov[4][4], Real g_out[4][4]) {
  // Compute beta^2
  Real beta2 = beta[0]*beta[0] + beta[1]*beta[1] + beta[2]*beta[2];

  // Check for unphysical beta (beta^2 >= 1)
  if (beta2 >= 1.0) {
    // Handle error: set g_out to NaN (Not a Number)
    for (int mu = 0; mu < 4; ++mu) {
      for (int nu = 0; nu < 4; ++nu) {
        g_out[mu][nu] = NAN;
      }
    }
    return;
  }

  // If beta^2 is zero (no boost), copy gcov to g_out
  if (beta2 == 0.0) {
    for (int mu = 0; mu < 4; ++mu) {
      for (int nu = 0; nu < 4; ++nu) {
        g_out[mu][nu] = gcov[mu][nu];
      }
    }
    return;
  }

  // Compute gamma factor
  Real gamma = 1.0 / sqrt(1.0 - beta2);

  // Initialize Lorentz transformation matrix Lambda^\rho_\mu
  Real Lambda[4][4];
  for (int mu = 0; mu < 4; ++mu) {
    for (int nu = 0; nu < 4; ++nu) {
      Lambda[mu][nu] = 0.0;
    }
  }

  // Compute Lambda^0_0
  Lambda[0][0] = gamma;

  // Compute Lambda^0_i and Lambda^i_0
  for (int i = 0; i < 3; ++i) {
    Lambda[0][i+1] = -gamma * beta[i];
    Lambda[i+1][0] = -gamma * beta[i];
  }

  // Compute Lambda^i_j
  Real gamma_minus_1_over_beta2 = (gamma - 1.0) / beta2;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      Lambda[i+1][j+1] = (i == j ? 1.0 : 0.0) + gamma_minus_1_over_beta2 * beta[i] * beta[j];
    }
  }

  // Compute the transformed metric tensor g_out
  // First, compute the intermediate matrix H = Lambda^T * gcov
  Real H[4][4];
  for (int mu = 0; mu < 4; ++mu) {
    for (int sigma = 0; sigma < 4; ++sigma) {
      Real sum = 0.0;
      for (int rho = 0; rho < 4; ++rho) {
        sum += Lambda[rho][mu] * gcov[rho][sigma];
      }
      H[mu][sigma] = sum;
    }
  }

  // Then, compute g_out = H * Lambda
  for (int mu = 0; mu < 4; ++mu) {
    for (int nu = 0; nu < 4; ++nu) {
      Real sum = 0.0;
      for (int sigma = 0; sigma < 4; ++sigma) {
        sum += H[mu][sigma] * Lambda[sigma][nu];
      }
      g_out[mu][nu] = sum;
    }
  }
}

// how decide the refinement
void RefinementCondition(MeshBlockPack* pmbp) {
  pmbp->pz4c->pamr->Refine(pmbp);
}

} // namespace
