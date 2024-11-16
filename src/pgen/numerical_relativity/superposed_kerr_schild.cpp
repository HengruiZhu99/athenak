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

#define h 5e-5
#define D2(comp, h) ((met_p1.g).comp - (met_m1.g).comp) / (2*h)

namespace {

enum {
  TT, XX, YY, ZZ, NDIM
};

enum {
  X1, Y1, Z1, X2, Y2, Z2,
  VX1, VY1, VZ1, VX2, VY2, VZ2,
  AX1, AY1, AZ1, AX2, AY2, AZ2,
  M1T, M2T, NTRAJ
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

struct four_metric {
  struct dd_sym g;
  struct dd_sym g_t;
  struct dd_sym g_x;
  struct dd_sym g_y;
  struct dd_sym g_z;
};

struct three_metric {
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

struct bbh_pgen {
  Real x1, y1, z1;
  Real x2, y2, z2;
  Real vx1, vy1, vz1;
  Real vx2, vy2, vz2;
  Real m1, m2;
  Real a1, a2;
  Real th_a1, th_a2;
  Real ph_a1, ph_a2;
};

struct bbh_pgen bbh;

/* Declare functions */
void find_traj_t(Real tt, Real traj_array[NTRAJ]);

KOKKOS_INLINE_FUNCTION
void numerical_4metric(const Real t, const Real x, const Real y,
    const Real z, struct four_metric &outmet,
    const Real nz_m1[NTRAJ], const Real nz_0[NTRAJ], const Real nz_p1[NTRAJ],
    const bbh_pgen& bbh_);
KOKKOS_INLINE_FUNCTION
int four_metric_to_three_metric(const struct four_metric &met, struct three_metric &gam);
KOKKOS_INLINE_FUNCTION
void get_metric(const Real t, const Real x, const Real y, const Real z,
                struct four_metric &met, const Real bbh_traj_loc[NTRAJ],
                const bbh_pgen& bbh_);
KOKKOS_INLINE_FUNCTION
void SuperposedBBH(const Real time, const Real x, const Real y, const Real z,
                   Real gcov[][NDIM], const Real traj_array[NTRAJ], const bbh_pgen& bbh_);
void SetADMVariablesToBBH(MeshBlockPack *pmbp);
void RefinementCondition(MeshBlockPack* pmbp);
} // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::ShockTube_()
//! \brief Problem Generator for the shock tube (Riemann problem) tests

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  user_ref_func  = RefinementCondition;

  if (restart) return;

  bbh.a1 = pin->GetOrAddReal("problem", "a1", 0.0);
  bbh.a2 = pin->GetOrAddReal("problem", "a2", 0.0);
  bbh.th_a1 = pin->GetOrAddReal("problem", "th_a1", 0.0);
  bbh.th_a2 = pin->GetOrAddReal("problem", "th_a2", 0.0);
  bbh.ph_a1 = pin->GetOrAddReal("problem", "ph_a1", 0.0);
  bbh.ph_a2 = pin->GetOrAddReal("problem", "ph_a2", 0.0);


  bbh.x1 = pin->GetOrAddReal("problem", "x1", -10.0);
  bbh.y1 = pin->GetOrAddReal("problem", "y1", 0.0);
  bbh.z1 = pin->GetOrAddReal("problem", "z1", 0.0);
  bbh.x2 = pin->GetOrAddReal("problem", "x2", 10.0);
  bbh.y2 = pin->GetOrAddReal("problem", "y2", 0.0);
  bbh.z2 = pin->GetOrAddReal("problem", "z2", 0.0);

  bbh.vx1 = pin->GetOrAddReal("problem", "vx1", 0.0);
  bbh.vy1 = pin->GetOrAddReal("problem", "vy1", 0.0);
  bbh.vz1 = pin->GetOrAddReal("problem", "vz1", 0.0);
  bbh.vx2 = pin->GetOrAddReal("problem", "vx2", 0.0);
  bbh.vy2 = pin->GetOrAddReal("problem", "vy2", 0.0);
  bbh.vz2 = pin->GetOrAddReal("problem", "vz2", 0.0);

  bbh.m1 = pin->GetOrAddReal("problem", "m1", 1.0);
  bbh.m2 = pin->GetOrAddReal("problem", "m2", 1.0);
  // capture variables for the kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  int nmb = pmbp->nmb_thispack;
  auto &bbh_ = bbh;

  // Initialize ADM variables -------------------------------
  SetADMVariablesToBBH(pmbp);
  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin);
            break;
    case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin);
            break;
    case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin);
            break;
  }
  // pmbp->pz4c->Z4cToADM(pmbp);
  // pmbp->pz4c->GaugePreCollapsedLapse(pmbp, pin);
  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMConstraints<2>(pmbp);
            break;
    case 3: pmbp->pz4c->ADMConstraints<3>(pmbp);
            break;
    case 4: pmbp->pz4c->ADMConstraints<4>(pmbp);
            break;
  }
  std::cout<<"Superposed Kerr-Schild initialized."<<std::endl;

  return;
}

namespace {

void SetADMVariablesToBBH(MeshBlockPack *pmbp) {
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

  Real bbh_traj_p1[NTRAJ];
  Real bbh_traj_0[NTRAJ];
  Real bbh_traj_m1[NTRAJ];
  auto& bbh_ = bbh;

  /* Load trajectories */

  /* Whether we load traj from a table or we compute analytical trajectories */
  find_traj_t(tt+h, bbh_traj_p1);
  find_traj_t(tt, bbh_traj_0);
  find_traj_t(tt-h, bbh_traj_m1);


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
    struct three_metric met3;
    numerical_4metric(tt, x1v, x2v, x3v, met4, bbh_traj_m1, bbh_traj_0, bbh_traj_p1,
                      bbh_);

    /* Transform 4D metric to 3+1 variables*/
    four_metric_to_three_metric(met4, met3);

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
    const Real z, struct four_metric &outmet,
    const Real nz_m1[NTRAJ], const Real nz_0[NTRAJ], const Real nz_p1[NTRAJ],
    const bbh_pgen& bbh_) {
  struct four_metric met_m1;
  struct four_metric met_p1;

  // Time
  get_metric(t-1*h, x, y, z, met_m1, nz_m1, bbh_);
  get_metric(t+1*h, x, y, z, met_p1, nz_p1, bbh_);
  get_metric(t, x, y, z, outmet, nz_0, bbh_);

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
  get_metric(t, x-1*h, y, z, met_m1, nz_0, bbh_);
  get_metric(t, x+1*h, y, z, met_p1, nz_0, bbh_);

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
  get_metric(t, x, y-1*h, z, met_m1, nz_0, bbh_);
  get_metric(t, x, y+1*h, z, met_p1, nz_0, bbh_);

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
  get_metric(t, x, y, z-1*h, met_m1, nz_0, bbh_);
  get_metric(t, x, y, z+1*h, met_p1, nz_0, bbh_);

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
int four_metric_to_three_metric(const struct four_metric &met,
                                struct three_metric &gam) {
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


  gam.alpha = sqrt(fabs(b2 - met.g.tt));

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

// Function to calculate the position and velocity of m1 and m2 at time t
void find_traj_t(Real t, Real bbh_t[NTRAJ]) {
  bbh_t[X1] = bbh.x1 + t * bbh.vx1;
  bbh_t[Y1] = bbh.y1 + t * bbh.vy1;
  bbh_t[Z1] = bbh.z1 + t * bbh.vz1;
  bbh_t[X2] = bbh.x2 + t * bbh.vx2;
  bbh_t[Y2] = bbh.y2 + t * bbh.vy2;
  bbh_t[Z2] = bbh.z2 + t * bbh.vz2;
  bbh_t[VX1] = bbh.vx1;
  bbh_t[VY1] = bbh.vy1;
  bbh_t[VZ1] = bbh.vz1;
  bbh_t[VX2] = bbh.vx2;
  bbh_t[VY2] = bbh.vy2;
  bbh_t[VZ2] = bbh.vz2;
  bbh_t[AX1] = bbh.a1*std::sin(bbh.th_a1)*std::cos(bbh.ph_a1);
  bbh_t[AY1] = bbh.a1*std::sin(bbh.th_a1)*std::sin(bbh.ph_a1);
  bbh_t[AZ1] = bbh.a1*std::cos(bbh.th_a1);
  bbh_t[AX2] = bbh.a1*std::sin(bbh.th_a2)*std::cos(bbh.ph_a2);
  bbh_t[AY2] = bbh.a1*std::sin(bbh.th_a2)*std::sin(bbh.ph_a2);
  bbh_t[AZ2] = bbh.a1*std::cos(bbh.th_a2);
  bbh_t[M1T] = bbh.m1;
  bbh_t[M2T] = bbh.m2;
}

KOKKOS_INLINE_FUNCTION
void SuperposedBBH(const Real time, const Real x, const Real y, const Real z,
                  Real gcov[][NDIM], const Real traj_array[NTRAJ], const bbh_pgen& bbh_) {
  /* Superposition components*/
  Real KS1[NDIM][NDIM];
  Real KS2[NDIM][NDIM];
  Real J1[NDIM][NDIM];
  Real J2[NDIM][NDIM];

  /* Load trajectories */
  Real xi1x = traj_array[X1];
  Real xi1y = traj_array[Y1];
  Real xi1z = traj_array[Z1];
  Real xi2x = traj_array[X2];
  Real xi2y = traj_array[Y2];
  Real xi2z = traj_array[Z2];
  Real v1x  = traj_array[VX1];
  // Real v1y  = traj_array[VY1];
  // Real v1z  = traj_array[VZ1];
  Real v2x =  traj_array[VX2];
  // Real v2y =  traj_array[VY2];
  // Real v2z =  traj_array[VZ2];

  // Real v2  =  sqrt( v2x * v2x + v2y * v2y + v2z * v2z );
  // Real v1  =  sqrt( v1x * v1x + v1y * v1y + v1z * v1z );

  // replace this with the superposed boosted puncture solution
  // metric for the first black hole
  Real gamma = 1/sqrt(1-v1x*v1x);
  Real x0 = gamma*((x-xi1x)-v*time);
  Real y0 = y - xi1y;
  Real z0 = z - xi1z;
  Real r0 = pow(x0*x0 + y0*y0 + z0*z0,0.5);
  Real psi0 = 1 + 0.5/r0;
  Real alpha0 = (1-0.5/r0)/psi0;
  Real B02 = gamma*gamma*(1-v*v*alpha0*alpha0*pow(psi0,-4));

  Real eta[4][4] = {
    {-gamma*gamma*(pow(alpha0,2)-pow(psi0,4)*pow(v,2)),gamma*gamma*v*(pow(alpha0,2)-pow(psi0,4)),0,0},
    {gamma*gamma*v*(pow(alpha0,2)-pow(psi0,4)),pow(psi0,4)*B02,0,0},
    {0,0,pow(psi0,4),0},
    {0,0,0,pow(psi0,4)}
  };
  for (int i=0; i < 4; i++ ) {
    for (int j=0; j < 4; j++ ) {
      gcov[i][j] = eta[i][j];
    }
  }

  // metric for the second black hole

  // subtract minkowski

  return;
}

KOKKOS_INLINE_FUNCTION
void get_metric(const Real t,
                const Real x,
                const Real y,
                const Real z,
                struct four_metric &met,
                const Real bbh_traj_loc[NTRAJ],
                const bbh_pgen& bbh_) {
  Real gcov[NDIM][NDIM];

  SuperposedBBH(t, x, y, z, gcov, bbh_traj_loc, bbh_);

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


// how decide the refinement
void RefinementCondition(MeshBlockPack* pmbp) {
  pmbp->pz4c->pamr->Refine(pmbp);
}

} // namespace