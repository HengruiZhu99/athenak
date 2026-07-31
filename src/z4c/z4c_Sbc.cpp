//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_Sbc.cpp
//! \brief Sommerfeld and residual-characteristic Z4c boundary RHS treatments

#include <math.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <utility>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "driver/driver.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "z4c/tmunu.hpp"
#include "z4c/z4c.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace z4c {

namespace {

enum CharacteristicDiag {
  CPBC_GAUGE_AMPLITUDE = 0,
  CPBC_CONSTRAINT_AMPLITUDE = 1,
  CPBC_RADIATION_AMPLITUDE = 2,
  CPBC_ENFORCEMENT_ERROR = 3,
  CPBC_MAX_CORRECTION = 4,
  CPBC_MAX_SHIFT_RATIO = 5,
  CPBC_MAX_CHARACTERISTIC = 6,
  CPBC_OUTGOING_GAUGE_AMPLITUDE = 7,
  CPBC_OUTGOING_CONSTRAINT_AMPLITUDE = 8,
  CPBC_OUTGOING_RADIATION_AMPLITUDE = 9,
  CPBC_MAX_OUTGOING_CHARACTERISTIC = 10,
  CPBC_NDIAG = 11
};

enum CharacteristicInvalid {
  CPBC_VALID = 0,
  CPBC_INVALID_METRIC_DETERMINANT = 1,
  CPBC_INVALID_METRIC_NORMAL = 2,
  CPBC_INVALID_METRIC_TANGENT1 = 3,
  CPBC_INVALID_METRIC_TANGENT2 = 4,
  CPBC_INVALID_MATTER = 5,
  CPBC_INVALID_COEFFICIENT = 6,
  CPBC_INVALID_SPEED = 7,
  CPBC_INVALID_CONE_SEPARATION = 8,
  CPBC_INVALID_SPACING = 9,
  CPBC_INVALID_SCALAR_MAP = 10,
  CPBC_NINVALID = 11
};

template <bool Value>
struct CharacteristicSourceTag {
  static constexpr bool value = Value;
};

KOKKOS_INLINE_FUNCTION
Real BoundaryRoundoffTolerance() {
  return sizeof(Real) == sizeof(float) ? 1.0e-5 : 1.0e-12;
}

KOKKOS_INLINE_FUNCTION
int SymmetricOffset(int a, int b) {
  if (a > b) {
    int t = a;
    a = b;
    b = t;
  }
  if (a == 0) return b;
  if (a == 1) return b + 2;
  return 5;
}

KOKKOS_INLINE_FUNCTION
bool UsesBoundaryRHS(BoundaryFlag flag, bool user_sbc) {
  return flag == BoundaryFlag::outflow || flag == BoundaryFlag::diode ||
         (flag == BoundaryFlag::user && user_sbc);
}

KOKKOS_INLINE_FUNCTION
Real CoordinateDerivative(const DvceArray5D<Real> &u, int m, int n,
                          int k, int j, int i, int dir, int side,
                          const Real idx[3]) {
  if (side != 0) {
    const int inward = -side;
    Real f0 = u(m,n,k,j,i);
    Real f1;
    Real f2;
    if (dir == 0) {
      f1 = u(m,n,k,j,i + inward);
      f2 = u(m,n,k,j,i + 2*inward);
    } else if (dir == 1) {
      f1 = u(m,n,k,j + inward,i);
      f2 = u(m,n,k,j + 2*inward,i);
    } else {
      f1 = u(m,n,k + inward,j,i);
      f2 = u(m,n,k + 2*inward,j,i);
    }
    return 0.5*side*idx[dir]*(3.0*f0 - 4.0*f1 + f2);
  }

  if (dir == 0) {
    return 0.5*idx[0]*(u(m,n,k,j,i+1) - u(m,n,k,j,i-1));
  } else if (dir == 1) {
    return 0.5*idx[1]*(u(m,n,k,j+1,i) - u(m,n,k,j-1,i));
  }
  return 0.5*idx[2]*(u(m,n,k+1,j,i) - u(m,n,k-1,j,i));
}

KOKKOS_INLINE_FUNCTION
Real NormalDerivative(const DvceArray5D<Real> &u, int m, int n,
                      int k, int j, int i, const Real normal_u[3],
                      const int side[3], const Real idx[3]) {
  Real derivative = 0.0;
  for (int a = 0; a < 3; ++a) {
    if (fabs(normal_u[a]) > BoundaryRoundoffTolerance()) {
      derivative += normal_u[a] *
          CoordinateDerivative(u, m, n, k, j, i, a, side[a], idx);
    }
  }
  return derivative;
}

KOKKOS_INLINE_FUNCTION
Real CenteredCoordinateDerivative4(DvceArray5D<Real> u, int m, int n,
                                   int k, int j, int i, int dir,
                                   const Real idx[3]) {
  return Dx<4>(dir,idx,u,m,n,k,j,i);
}

KOKKOS_INLINE_FUNCTION
Real CenteredCoordinateSecondDerivative4(
    DvceArray5D<Real> u, int m, int n, int k, int j, int i,
    int first, int second, const Real idx[3]) {
  return first == second ? Dxx<4>(first,idx,u,m,n,k,j,i) :
                           Dxy<4>(first,second,idx,u,m,n,k,j,i);
}

KOKKOS_INLINE_FUNCTION
Real CenteredNormalDerivative4(DvceArray5D<Real> u, int m, int n,
                               int k, int j, int i,
                               const Real normal_u[3], const Real idx[3]) {
  Real derivative = 0.0;
  for (int a = 0; a < 3; ++a) {
    if (fabs(normal_u[a]) > BoundaryRoundoffTolerance()) {
      derivative += normal_u[a]*CenteredCoordinateDerivative4(
          u,m,n,k,j,i,a,idx);
    }
  }
  return derivative;
}

KOKKOS_INLINE_FUNCTION
Real CenteredNormalSecondDerivative4(
    DvceArray5D<Real> u, int m, int n, int k, int j, int i,
    const Real normal_u[3], const Real idx[3]) {
  Real derivative = 0.0;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      const Real projection = normal_u[a]*normal_u[b];
      if (fabs(projection) <= BoundaryRoundoffTolerance()) continue;
      derivative += projection*CenteredCoordinateSecondDerivative4(
          u,m,n,k,j,i,a,b,idx);
    }
  }
  return derivative;
}

KOKKOS_INLINE_FUNCTION
int MakeBoundaryFrame(const DvceArray5D<Real> &full, int m, int k, int j, int i,
                       const int side[3], Real g_dd[3][3], Real g_uu[3][3],
                       Real normal_d[3], Real normal_u[3],
                       Real tangent1_d[3], Real tangent1_u[3],
                       Real tangent2_d[3], Real tangent2_u[3]) {
  for (int a = 0; a < 3; ++a) {
    for (int b = a; b < 3; ++b) {
      Real value = full(m, Z4c::I_Z4C_GXX + SymmetricOffset(a,b), k,j,i);
      g_dd[a][b] = value;
      g_dd[b][a] = value;
    }
  }

  // Sylvester's criterion is the inexpensive exact positive-definiteness
  // check for this symmetric 3x3 metric.  A positive determinant alone would
  // also accept a signature with two negative eigenvalues.
  const Real leading_minor1 = g_dd[0][0];
  const Real leading_minor2 =
      g_dd[0][0]*g_dd[1][1] - SQR(g_dd[0][1]);
  Real detg = adm::SpatialDet(g_dd[0][0], g_dd[0][1], g_dd[0][2],
                              g_dd[1][1], g_dd[1][2], g_dd[2][2]);
  if (!(isfinite(leading_minor1) && isfinite(leading_minor2) &&
        isfinite(detg)) ||
      leading_minor1 <= 0.0 || leading_minor2 <= 0.0 || detg <= 0.0) {
    return CPBC_INVALID_METRIC_DETERMINANT;
  }
  adm::SpatialInv(1.0/detg,
                  g_dd[0][0], g_dd[0][1], g_dd[0][2],
                  g_dd[1][1], g_dd[1][2], g_dd[2][2],
                  &g_uu[0][0], &g_uu[0][1], &g_uu[0][2],
                  &g_uu[1][1], &g_uu[1][2], &g_uu[2][2]);
  g_uu[1][0] = g_uu[0][1];
  g_uu[2][0] = g_uu[0][2];
  g_uu[2][1] = g_uu[1][2];

  Real norm2 = 0.0;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      norm2 += side[a]*g_uu[a][b]*side[b];
    }
  }
  if (!(isfinite(norm2)) || norm2 <= 0.0) {
    return CPBC_INVALID_METRIC_NORMAL;
  }
  Real inv_norm = 1.0/sqrt(norm2);
  for (int a = 0; a < 3; ++a) {
    normal_d[a] = side[a]*inv_norm;
  }
  for (int a = 0; a < 3; ++a) {
    normal_u[a] = 0.0;
    for (int b = 0; b < 3; ++b) {
      normal_u[a] += g_uu[a][b]*normal_d[b];
    }
  }

  // Pick the coordinate vector with the largest projection into the tangent
  // plane.  This avoids a polar-axis special case and is deterministic.
  int best_axis = 0;
  Real best_norm2 = -1.0;
  for (int axis = 0; axis < 3; ++axis) {
    Real candidate[3];
    for (int a = 0; a < 3; ++a) {
      candidate[a] = (a == axis ? 1.0 : 0.0) -
                     normal_u[a]*normal_d[axis];
    }
    Real candidate_norm2 = 0.0;
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        candidate_norm2 += g_dd[a][b]*candidate[a]*candidate[b];
      }
    }
    if (candidate_norm2 > best_norm2) {
      best_norm2 = candidate_norm2;
      best_axis = axis;
    }
  }
  if (!(isfinite(best_norm2)) || best_norm2 <= 0.0) {
    return CPBC_INVALID_METRIC_TANGENT1;
  }
  Real inv_tangent_norm = 1.0/sqrt(best_norm2);
  for (int a = 0; a < 3; ++a) {
    tangent1_u[a] =
        ((a == best_axis ? 1.0 : 0.0) -
         normal_u[a]*normal_d[best_axis])*inv_tangent_norm;
  }
  for (int a = 0; a < 3; ++a) {
    tangent1_d[a] = 0.0;
    for (int b = 0; b < 3; ++b) {
      tangent1_d[a] += g_dd[a][b]*tangent1_u[b];
    }
  }

  // The metric cross product of the two unit covectors gives the second
  // tangent.  Normalize once more to absorb roundoff and det(g) drift.
  Real sqrt_detg = sqrt(detg);
  tangent2_u[0] = (normal_d[1]*tangent1_d[2] -
                   normal_d[2]*tangent1_d[1])/sqrt_detg;
  tangent2_u[1] = (normal_d[2]*tangent1_d[0] -
                   normal_d[0]*tangent1_d[2])/sqrt_detg;
  tangent2_u[2] = (normal_d[0]*tangent1_d[1] -
                   normal_d[1]*tangent1_d[0])/sqrt_detg;
  Real tangent2_norm2 = 0.0;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      tangent2_norm2 += g_dd[a][b]*tangent2_u[a]*tangent2_u[b];
    }
  }
  if (!(isfinite(tangent2_norm2)) || tangent2_norm2 <= 0.0) {
    return CPBC_INVALID_METRIC_TANGENT2;
  }
  Real inv_tangent2_norm = 1.0/sqrt(tangent2_norm2);
  for (int a = 0; a < 3; ++a) {
    tangent2_u[a] *= inv_tangent2_norm;
  }
  for (int a = 0; a < 3; ++a) {
    tangent2_d[a] = 0.0;
    for (int b = 0; b < 3; ++b) {
      tangent2_d[a] += g_dd[a][b]*tangent2_u[b];
    }
  }
  return CPBC_VALID;
}

KOKKOS_INLINE_FUNCTION
Real ProjectTensor(const Real tensor_dd[3][3],
                   const Real left_u[3], const Real right_u[3]) {
  Real value = 0.0;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      value += tensor_dd[a][b]*left_u[a]*right_u[b];
    }
  }
  return value;
}

KOKKOS_INLINE_FUNCTION
Real TensorTrace(const Real tensor_dd[3][3], const Real metric_uu[3][3]) {
  Real value = 0.0;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) value += metric_uu[a][b]*tensor_dd[a][b];
  }
  return value;
}

KOKKOS_INLINE_FUNCTION
void AtomicDiagnosticMax(const DvceArray1D<Real> &diag, int index, Real value,
                         bool collect_diagnostics) {
  if (collect_diagnostics && isfinite(value)) {
    Kokkos::atomic_max(&diag(index), fabs(value));
  }
}

template <bool TangentialPrincipal>
KOKKOS_INLINE_FUNCTION
int ApplyResidualCharacteristicBC(
    const DvceArray5D<Real> &u, const DvceArray5D<Real> &u_full,
    const DvceArray5D<Real> &u_bg,
    const DvceArray5D<Real> &u_rhs, const DvceArray5D<Real> &matter,
    bool has_matter, const Z4c::Options &opt,
    const DvceArray1D<Real> &diag, Real time,
    int m, int k, int j, int i, const int side[3], const Real idx[3],
    bool collect_diagnostics) {
  Real g_dd[3][3], g_uu[3][3];
  Real normal_d[3], normal_u[3];
  Real tangent1_d[3], tangent1_u[3];
  Real tangent2_d[3], tangent2_u[3];
  const int frame_status =
      MakeBoundaryFrame(u_full, m,k,j,i, side, g_dd,g_uu,
                        normal_d,normal_u, tangent1_d,tangent1_u,
                        tangent2_d,tangent2_u);
  if (frame_status != CPBC_VALID) {
    return frame_status;
  }
  if (has_matter) {
    for (int n = 0; n < Tmunu::N_Tmunu; ++n) {
      const Real component = matter(m,n,k,j,i);
      if (!(isfinite(component)) ||
          fabs(component) > opt.characteristic_bc_max_energy_density) {
        return CPBC_INVALID_MATTER;
      }
    }
  }

  const Real alpha = u_full(m,Z4c::I_Z4C_ALPHA,k,j,i);
  const Real chi = u_full(m,Z4c::I_Z4C_CHI,k,j,i);
  const Real alpha_bg = u_bg(m,Z4c::I_Z4C_ALPHA,k,j,i);
  const Real f_bg = opt.lapse_oplog*opt.lapse_harmonicf +
                    opt.lapse_harmonic*alpha_bg;
  const Real lapse_driver = opt.residual_lapse_f*f_bg*alpha_bg;
  const Real shift_driver =
      (1.0 - opt.sss_damping_amp *
       exp(-0.5*SQR(time/opt.sss_damping_time)))*opt.shift_ggamma;
  Real beta_full_normal = 0.0;
  Real beta_bg_normal = 0.0;
  for (int a = 0; a < 3; ++a) {
    beta_full_normal +=
        normal_d[a]*u_full(m,Z4c::I_Z4C_BETAX+a,k,j,i);
    beta_bg_normal +=
        normal_d[a]*u_bg(m,Z4c::I_Z4C_BETAX+a,k,j,i);
  }

  if (!(isfinite(alpha) && isfinite(alpha_bg) && isfinite(chi) &&
        isfinite(lapse_driver) && isfinite(shift_driver) &&
        isfinite(beta_full_normal) && isfinite(beta_bg_normal)) ||
      alpha <= 0.0 || alpha_bg <= 0.0 || chi <= 0.0 ||
      lapse_driver <= 0.0 || shift_driver <= 0.0) {
    return CPBC_INVALID_COEFFICIENT;
  }

  const Real sqrt_chi = sqrt(chi);
  const Real c_light = alpha*sqrt_chi;
  const Real beta_difference = beta_full_normal-beta_bg_normal;
  const Real lapse_discriminant =
      SQR(beta_difference)+4.0*chi*lapse_driver;
  const Real shift_long_discriminant =
      SQR(beta_difference)+(16.0/3.0)*shift_driver;
  const Real shift_trans_discriminant =
      SQR(beta_difference)+4.0*shift_driver;
  if (!(isfinite(c_light) && isfinite(lapse_discriminant) &&
        isfinite(shift_long_discriminant) &&
        isfinite(shift_trans_discriminant)) ||
      c_light <= 0.0 || lapse_discriminant <= 0.0 ||
      shift_long_discriminant <= 0.0 ||
      shift_trans_discriminant <= 0.0) {
    return CPBC_INVALID_SPEED;
  }
  const Real lambda_lapse[2] = {
    0.5*(beta_full_normal+beta_bg_normal+sqrt(lapse_discriminant)),
    0.5*(beta_full_normal+beta_bg_normal-sqrt(lapse_discriminant))};
  const Real lambda_shift_long[2] = {
    0.5*(beta_full_normal+beta_bg_normal+sqrt(shift_long_discriminant)),
    0.5*(beta_full_normal+beta_bg_normal-sqrt(shift_long_discriminant))};
  const Real lambda_shift_trans[2] = {
    0.5*(beta_full_normal+beta_bg_normal+sqrt(shift_trans_discriminant)),
    0.5*(beta_full_normal+beta_bg_normal-sqrt(shift_trans_discriminant))};
  const Real lambda_light[2] = {
    beta_full_normal+c_light,beta_full_normal-c_light};
  if (!(lambda_lapse[0] > 0.0 && lambda_lapse[1] < 0.0 &&
        lambda_shift_long[0] > 0.0 && lambda_shift_long[1] < 0.0 &&
        lambda_shift_trans[0] > 0.0 && lambda_shift_trans[1] < 0.0 &&
        lambda_light[0] > 0.0 && lambda_light[1] < 0.0)) {
    return CPBC_INVALID_SPEED;
  }

  // The closed scalar rows are singular when the longitudinal-shift root
  // coincides with either the lapse or the light root.  These are the exact
  // finite-residual separations for the hybrid full/background advection
  // symbol, not the zero-residual comoving approximations.
  const Real shift_mu = lambda_shift_long[0]-beta_full_normal;
  const Real shift_delta_bg =
      lambda_shift_long[0]-beta_bg_normal;
  const Real lapse_shift_separation =
      chi*lapse_driver-shift_mu*shift_delta_bg;
  const Real light_shift_separation =
      chi*SQR(alpha)-SQR(shift_mu);
  const Real shift_mu_out =
      lambda_shift_long[1]-beta_full_normal;
  const Real light_shift_separation_out =
      chi*SQR(alpha)-SQR(shift_mu_out);
  const Real separation_scale =
      fmax(1.0,fmax(fabs(chi*lapse_driver),
                    fmax(fabs(shift_mu*shift_delta_bg),
                         fabs(chi*SQR(alpha)))));
  if (fabs(lapse_shift_separation) <=
          BoundaryRoundoffTolerance()*separation_scale ||
      fabs(light_shift_separation) <=
          BoundaryRoundoffTolerance()*separation_scale ||
      fabs(light_shift_separation_out) <=
          BoundaryRoundoffTolerance()*separation_scale) {
    return CPBC_INVALID_CONE_SEPARATION;
  }

  Real inv_h = 0.0;
  for (int a = 0; a < 3; ++a) inv_h += fabs(normal_u[a])*idx[a];
  if (!(isfinite(inv_h)) || inv_h <= 0.0) return CPBC_INVALID_SPACING;

  Real derivative_metric[3][3], derivative_rhs_metric[3][3];
  Real centered_derivative_A[3][3] = {};
  Real centered_second_metric[3][3] = {};
  Real residual_A[3][3], rhs_A[3][3];
  for (int a = 0; a < 3; ++a) {
    for (int b = a; b < 3; ++b) {
      int goffset = Z4c::I_Z4C_GXX + SymmetricOffset(a,b);
      int aoffset = Z4c::I_Z4C_AXX + SymmetricOffset(a,b);
      Real dg = NormalDerivative(u,m,goffset,k,j,i,normal_u,side,idx);
      Real drg = NormalDerivative(u_rhs,m,goffset,k,j,i,normal_u,side,idx);
      Real avalue = u(m,aoffset,k,j,i);
      Real arhs = u_rhs(m,aoffset,k,j,i);
      derivative_metric[a][b] = derivative_metric[b][a] = dg;
      derivative_rhs_metric[a][b] = derivative_rhs_metric[b][a] = drg;
      if constexpr (TangentialPrincipal) {
        const Real da = CenteredNormalDerivative4(
            u,m,aoffset,k,j,i,normal_u,idx);
        const Real ddg = CenteredNormalSecondDerivative4(
            u,m,goffset,k,j,i,normal_u,idx);
        centered_derivative_A[a][b] = centered_derivative_A[b][a] = da;
        centered_second_metric[a][b] = centered_second_metric[b][a] = ddg;
      }
      residual_A[a][b] = residual_A[b][a] = avalue;
      rhs_A[a][b] = rhs_A[b][a] = arhs;
    }
  }
  Real derivative_beta[3], derivative_rhs_beta[3];
  Real centered_derivative_Gamma[3] = {};
  Real centered_second_beta[3] = {};
  for (int a = 0; a < 3; ++a) {
    derivative_beta[a] = NormalDerivative(
        u,m,Z4c::I_Z4C_BETAX+a,k,j,i,normal_u,side,idx);
    derivative_rhs_beta[a] = NormalDerivative(
        u_rhs,m,Z4c::I_Z4C_BETAX+a,k,j,i,normal_u,side,idx);
    if constexpr (TangentialPrincipal) {
      centered_derivative_Gamma[a] = CenteredNormalDerivative4(
          u,m,Z4c::I_Z4C_GAMX+a,k,j,i,normal_u,idx);
      centered_second_beta[a] = CenteredNormalSecondDerivative4(
          u,m,Z4c::I_Z4C_BETAX+a,k,j,i,normal_u,idx);
    }
  }

  const Real derivative_metric_trace = TensorTrace(derivative_metric,g_uu);
  const Real derivative_rhs_metric_trace =
      TensorTrace(derivative_rhs_metric,g_uu);
  const Real centered_derivative_A_trace =
      TensorTrace(centered_derivative_A,g_uu);
  const Real centered_second_metric_trace =
      TensorTrace(centered_second_metric,g_uu);
  const Real residual_A_trace = TensorTrace(residual_A,g_uu);
  const Real rhs_A_trace = TensorTrace(rhs_A,g_uu);

  Real scalar_p[4], scalar_d[4], scalar_p_rhs[4], scalar_d_rhs[4];
  scalar_p[0] = u(m,Z4c::I_Z4C_KHAT,k,j,i);
  scalar_p[1] = u(m,Z4c::I_Z4C_THETA,k,j,i);
  scalar_p[2] = ProjectTensor(residual_A,normal_u,normal_u) -
                residual_A_trace/3.0;
  scalar_p[3] = 0.0;
  scalar_p_rhs[0] = u_rhs(m,Z4c::I_Z4C_KHAT,k,j,i);
  scalar_p_rhs[1] = u_rhs(m,Z4c::I_Z4C_THETA,k,j,i);
  scalar_p_rhs[2] = ProjectTensor(rhs_A,normal_u,normal_u) -
                    rhs_A_trace/3.0;
  scalar_p_rhs[3] = 0.0;
  for (int a = 0; a < 3; ++a) {
    scalar_p[3] += normal_d[a]*u(m,Z4c::I_Z4C_GAMX+a,k,j,i);
    scalar_p_rhs[3] += normal_d[a]*u_rhs(m,Z4c::I_Z4C_GAMX+a,k,j,i);
  }
  scalar_d[0] =
      NormalDerivative(u,m,Z4c::I_Z4C_CHI,k,j,i,normal_u,side,idx);
  scalar_d[1] = ProjectTensor(derivative_metric,normal_u,normal_u) -
                derivative_metric_trace/3.0;
  scalar_d[2] =
      NormalDerivative(u,m,Z4c::I_Z4C_ALPHA,k,j,i,normal_u,side,idx);
  scalar_d[3] = 0.0;
  scalar_d_rhs[0] =
      NormalDerivative(u_rhs,m,Z4c::I_Z4C_CHI,k,j,i,normal_u,side,idx);
  scalar_d_rhs[1] =
      ProjectTensor(derivative_rhs_metric,normal_u,normal_u) -
      derivative_rhs_metric_trace/3.0;
  scalar_d_rhs[2] =
      NormalDerivative(u_rhs,m,Z4c::I_Z4C_ALPHA,k,j,i,normal_u,side,idx);
  scalar_d_rhs[3] = 0.0;
  for (int a = 0; a < 3; ++a) {
    scalar_d[3] += normal_d[a]*derivative_beta[a];
    scalar_d_rhs[3] += normal_d[a]*derivative_rhs_beta[a];
  }

  Real scalar_p_normal[4] = {};
  Real scalar_d_normal[4] = {};
  if constexpr (TangentialPrincipal) {
    scalar_p_normal[0] = CenteredNormalDerivative4(
        u,m,Z4c::I_Z4C_KHAT,k,j,i,normal_u,idx);
    scalar_p_normal[1] = CenteredNormalDerivative4(
        u,m,Z4c::I_Z4C_THETA,k,j,i,normal_u,idx);
    scalar_p_normal[2] =
        ProjectTensor(centered_derivative_A,normal_u,normal_u) -
        centered_derivative_A_trace/3.0;
    scalar_d_normal[0] = CenteredNormalSecondDerivative4(
        u,m,Z4c::I_Z4C_CHI,k,j,i,normal_u,idx);
    scalar_d_normal[1] =
        ProjectTensor(centered_second_metric,normal_u,normal_u) -
        centered_second_metric_trace/3.0;
    scalar_d_normal[2] = CenteredNormalSecondDerivative4(
        u,m,Z4c::I_Z4C_ALPHA,k,j,i,normal_u,idx);
    for (int a = 0; a < 3; ++a) {
      scalar_p_normal[3] += normal_d[a]*centered_derivative_Gamma[a];
      scalar_d_normal[3] += normal_d[a]*centered_second_beta[a];
    }
  }

  Real principal_A_rhs[3][3] = {};
  Real principal_Gamma_rhs[3] = {};
  Real principal_metric_normal_rhs[3][3] = {};
  Real principal_beta_normal_rhs[3] = {};
  Real scalar_p_principal[4] = {};
  Real scalar_d_principal[4] = {};
  if constexpr (TangentialPrincipal) {
  // Build the complete frozen principal RHS from the same NGHOST=4 centered
  // differences as the volume operator.  Subtracting its face-normal part
  // below leaves exactly the tangential principal boundary datum; nonlinear,
  // damping, source, and KO terms are not reclassified as incoming data.
  Real beta_full_u[3], beta_bg_u[3];
  Real gradient_k[3], gradient_theta[3];
  Real gradient_Gamma[3][3], gradient_A[3][3][3];
  Real hessian_alpha[3][3], hessian_chi[3][3];
  Real hessian_beta[3][3][3];
  Real laplacian_metric[3][3], normal_advection_metric[3][3];
  for (int a = 0; a < 3; ++a) {
    beta_full_u[a] = u_full(m,Z4c::I_Z4C_BETAX+a,k,j,i);
    beta_bg_u[a] = u_bg(m,Z4c::I_Z4C_BETAX+a,k,j,i);
    gradient_k[a] = CenteredCoordinateDerivative4(
        u,m,Z4c::I_Z4C_KHAT,k,j,i,a,idx);
    gradient_theta[a] = CenteredCoordinateDerivative4(
        u,m,Z4c::I_Z4C_THETA,k,j,i,a,idx);
    for (int b = 0; b < 3; ++b) {
      gradient_Gamma[a][b] = CenteredCoordinateDerivative4(
          u,m,Z4c::I_Z4C_GAMX+b,k,j,i,a,idx);
      hessian_alpha[a][b] = CenteredCoordinateSecondDerivative4(
          u,m,Z4c::I_Z4C_ALPHA,k,j,i,a,b,idx);
      hessian_chi[a][b] = CenteredCoordinateSecondDerivative4(
          u,m,Z4c::I_Z4C_CHI,k,j,i,a,b,idx);
      for (int c = 0; c < 3; ++c) {
        hessian_beta[a][b][c] =
            CenteredCoordinateSecondDerivative4(
                u,m,Z4c::I_Z4C_BETAX+c,k,j,i,a,b,idx);
      }
    }
  }
  for (int a = 0; a < 3; ++a) {
    for (int b = a; b < 3; ++b) {
      const int aoffset = Z4c::I_Z4C_AXX + SymmetricOffset(a,b);
      const int goffset = Z4c::I_Z4C_GXX + SymmetricOffset(a,b);
      Real lap_g = 0.0;
      Real normal_adv_g = 0.0;
      for (int c = 0; c < 3; ++c) {
        gradient_A[c][a][b] = CenteredCoordinateDerivative4(
            u,m,aoffset,k,j,i,c,idx);
        gradient_A[c][b][a] = gradient_A[c][a][b];
        for (int d = 0; d < 3; ++d) {
          const Real ddg = CenteredCoordinateSecondDerivative4(
              u,m,goffset,k,j,i,c,d,idx);
          lap_g += g_uu[c][d]*ddg;
          normal_adv_g += normal_u[c]*beta_full_u[d]*ddg;
        }
      }
      laplacian_metric[a][b] = laplacian_metric[b][a] = lap_g;
      normal_advection_metric[a][b] =
          normal_advection_metric[b][a] = normal_adv_g;
    }
  }

  Real laplacian_alpha = 0.0;
  Real laplacian_chi = 0.0;
  Real normal_advection_alpha = 0.0;
  Real normal_advection_chi = 0.0;
  Real gradient_divergence_beta[3] = {0.0,0.0,0.0};
  Real divergence_Gamma = 0.0;
  for (int a = 0; a < 3; ++a) {
    divergence_Gamma += gradient_Gamma[a][a];
    for (int b = 0; b < 3; ++b) {
      laplacian_alpha += g_uu[a][b]*hessian_alpha[a][b];
      laplacian_chi += g_uu[a][b]*hessian_chi[a][b];
      normal_advection_alpha +=
          normal_u[a]*beta_bg_u[b]*hessian_alpha[a][b];
      normal_advection_chi +=
          normal_u[a]*beta_full_u[b]*hessian_chi[a][b];
      gradient_divergence_beta[a] += hessian_beta[a][b][b];
    }
  }
  const Real laplacian_metric_trace =
      TensorTrace(laplacian_metric,g_uu);

  Real principal_k_rhs = -chi*laplacian_alpha;
  Real principal_theta_rhs =
      0.5*alpha*(chi*(divergence_Gamma -
                     0.5*laplacian_metric_trace) +
                 2.0*laplacian_chi);
  for (int a = 0; a < 3; ++a) {
    principal_k_rhs += beta_full_u[a]*gradient_k[a];
    principal_theta_rhs += beta_full_u[a]*gradient_theta[a];
  }

  Real ricci_principal[3][3];
  Real lapse_ricci_tf_argument[3][3];
  for (int a = 0; a < 3; ++a) {
    for (int b = a; b < 3; ++b) {
      Real gamma_derivative = 0.0;
      for (int c = 0; c < 3; ++c) {
        gamma_derivative +=
            g_dd[c][a]*gradient_Gamma[b][c] +
            g_dd[c][b]*gradient_Gamma[a][c];
      }
      ricci_principal[a][b] = ricci_principal[b][a] =
          0.5*(gamma_derivative-laplacian_metric[a][b]) +
          0.5/chi*(hessian_chi[a][b] +
                   g_dd[a][b]*laplacian_chi);
      lapse_ricci_tf_argument[a][b] =
          lapse_ricci_tf_argument[b][a] =
          -hessian_alpha[a][b] + alpha*ricci_principal[a][b];
    }
  }
  const Real lapse_ricci_trace =
      TensorTrace(lapse_ricci_tf_argument,g_uu);
  for (int a = 0; a < 3; ++a) {
    for (int b = a; b < 3; ++b) {
      Real value = chi*(lapse_ricci_tf_argument[a][b] -
                        g_dd[a][b]*lapse_ricci_trace/3.0);
      for (int c = 0; c < 3; ++c) {
        value += beta_full_u[c]*gradient_A[c][a][b];
      }
      principal_A_rhs[a][b] = principal_A_rhs[b][a] = value;
    }
  }
  for (int a = 0; a < 3; ++a) {
    Real value = 0.0;
    Real laplacian_beta = 0.0;
    Real raised_gradient_k = 0.0;
    Real raised_gradient_theta = 0.0;
    Real raised_gradient_divergence_beta = 0.0;
    for (int b = 0; b < 3; ++b) {
      value += beta_full_u[b]*gradient_Gamma[b][a];
      raised_gradient_k += g_uu[a][b]*gradient_k[b];
      raised_gradient_theta += g_uu[a][b]*gradient_theta[b];
      raised_gradient_divergence_beta +=
          g_uu[a][b]*gradient_divergence_beta[b];
      for (int c = 0; c < 3; ++c) {
        laplacian_beta += g_uu[b][c]*hessian_beta[b][c][a];
      }
    }
    principal_Gamma_rhs[a] =
        value + laplacian_beta +
        raised_gradient_divergence_beta/3.0 -
        (4.0/3.0)*alpha*raised_gradient_k -
        (2.0/3.0)*alpha*raised_gradient_theta;
  }

  Real principal_chi_normal_rhs = normal_advection_chi;
  Real principal_alpha_normal_rhs = normal_advection_alpha;
  Real normal_gradient_divergence_beta = 0.0;
  for (int a = 0; a < 3; ++a) {
    normal_gradient_divergence_beta +=
        normal_u[a]*gradient_divergence_beta[a];
    principal_chi_normal_rhs +=
        normal_u[a]*((2.0/3.0)*chi*alpha*gradient_k[a] +
                     (4.0/3.0)*chi*alpha*gradient_theta[a] -
                     (2.0/3.0)*chi*gradient_divergence_beta[a]);
    principal_alpha_normal_rhs -=
        lapse_driver*normal_u[a]*gradient_k[a];
    Real beta_value = 0.0;
    for (int b = 0; b < 3; ++b) {
      for (int c = 0; c < 3; ++c) {
        beta_value += normal_u[b]*beta_bg_u[c]*
                      hessian_beta[b][c][a];
      }
    }
    for (int b = 0; b < 3; ++b) {
      beta_value += shift_driver*normal_u[b]*gradient_Gamma[b][a];
    }
    principal_beta_normal_rhs[a] = beta_value;
  }
  for (int a = 0; a < 3; ++a) {
    for (int b = a; b < 3; ++b) {
      Real value = normal_advection_metric[a][b] -
                   2.0*alpha*centered_derivative_A[a][b] -
                   (2.0/3.0)*g_dd[a][b]*
                       normal_gradient_divergence_beta;
      for (int c = 0; c < 3; ++c) {
        for (int d = 0; d < 3; ++d) {
          value += normal_u[c]*
              (g_dd[b][d]*hessian_beta[c][a][d] +
               g_dd[a][d]*hessian_beta[c][b][d]);
        }
      }
      principal_metric_normal_rhs[a][b] =
          principal_metric_normal_rhs[b][a] = value;
    }
  }
  const Real principal_A_rhs_trace =
      TensorTrace(principal_A_rhs,g_uu);
  const Real principal_metric_normal_rhs_trace =
      TensorTrace(principal_metric_normal_rhs,g_uu);
  scalar_p_principal[0] = principal_k_rhs;
  scalar_p_principal[1] = principal_theta_rhs;
  scalar_p_principal[2] =
      ProjectTensor(principal_A_rhs,normal_u,normal_u) -
      principal_A_rhs_trace/3.0;
  scalar_d_principal[0] = principal_chi_normal_rhs;
  scalar_d_principal[1] =
      ProjectTensor(principal_metric_normal_rhs,normal_u,normal_u) -
      principal_metric_normal_rhs_trace/3.0;
  scalar_d_principal[2] = principal_alpha_normal_rhs;
  for (int a = 0; a < 3; ++a) {
    scalar_p_principal[3] += normal_d[a]*principal_Gamma_rhs[a];
    scalar_d_principal[3] += normal_d[a]*principal_beta_normal_rhs[a];
  }
  }

  Real left_p[4][4] = {};

  // Lapse roots of
  //   (lambda-beta_full)(lambda-beta_bg)=chi*lapse_driver.
  left_p[0][0] =
      -(lambda_lapse[0]-beta_bg_normal)/chi;

  // Longitudinal-shift roots of
  //   (lambda-beta_full)(lambda-beta_bg)=4 shift_driver/3.
  // Multiply the unit-d_s-beta row by both cone separations.  This is an
  // eigenvector rescaling, but avoids ill-conditioned large coefficients
  // close to either supported (nondegenerate) cone.
  const Real shift_q = (4.0/3.0)*shift_driver;
  left_p[1][0] =
      alpha*SQR(shift_delta_bg)*light_shift_separation;
  left_p[1][1] =
      0.5*alpha*shift_q*lapse_shift_separation;
  left_p[1][3] =
      0.25*shift_delta_bg*
      (4.0*chi*SQR(alpha)-3.0*SQR(shift_mu))*lapse_shift_separation;
  const Real shift_left_d0 =
      0.5*SQR(alpha)*shift_delta_bg*lapse_shift_separation;
  const Real shift_left_d2 =
      -chi*alpha*shift_delta_bg*light_shift_separation;
  const Real shift_left_d3 =
      lapse_shift_separation*light_shift_separation;

  // The two light-speed scalar rows are unchanged by the hybrid gauge
  // advection because they have no lapse/shift derivative component.
  left_p[2][1] = sqrt_chi;
  left_p[2][3] = 0.5*chi;

  left_p[3][0] = 4.0/(3.0*sqrt_chi);
  left_p[3][1] = 2.0/(3.0*sqrt_chi);
  left_p[3][2] = -2.0/sqrt_chi;
  left_p[3][3] = -1.0;

  Real scalar_residual[4];
  Real target_rate[4];
  const Real scalar_lambda_in[4] = {
    lambda_lapse[0],lambda_shift_long[0],lambda_light[0],lambda_light[0]};
  for (int mode = 0; mode < 4; ++mode) {
    Real amplitude = 0.0;
    Real rate = 0.0;
    Real normal_principal_rate = 0.0;
    Real full_principal_rate = 0.0;
    for (int n = 0; n < 4; ++n) {
      amplitude += left_p[mode][n]*scalar_p[n];
      rate += left_p[mode][n]*scalar_p_rhs[n];
      normal_principal_rate +=
          left_p[mode][n]*scalar_p_normal[n];
      full_principal_rate +=
          left_p[mode][n]*scalar_p_principal[n];
    }
    if (mode == 0) {
      amplitude += scalar_d[2];
      rate += scalar_d_rhs[2];
      normal_principal_rate += scalar_d_normal[2];
      full_principal_rate += scalar_d_principal[2];
    } else if (mode == 1) {
      amplitude += shift_left_d0*scalar_d[0] +
                   shift_left_d2*scalar_d[2] +
                   shift_left_d3*scalar_d[3];
      rate += shift_left_d0*scalar_d_rhs[0] +
              shift_left_d2*scalar_d_rhs[2] +
              shift_left_d3*scalar_d_rhs[3];
      normal_principal_rate +=
          shift_left_d0*scalar_d_normal[0] +
          shift_left_d2*scalar_d_normal[2] +
          shift_left_d3*scalar_d_normal[3];
      full_principal_rate +=
          shift_left_d0*scalar_d_principal[0] +
          shift_left_d2*scalar_d_principal[2] +
          shift_left_d3*scalar_d_principal[3];
    } else if (mode == 2) {
      amplitude += scalar_d[0];
      rate += scalar_d_rhs[0];
      normal_principal_rate += scalar_d_normal[0];
      full_principal_rate += scalar_d_principal[0];
    } else {
      amplitude += scalar_d[1];
      rate += scalar_d_rhs[1];
      normal_principal_rate += scalar_d_normal[1];
      full_principal_rate += scalar_d_principal[1];
    }
    if constexpr (TangentialPrincipal) {
      normal_principal_rate *= scalar_lambda_in[mode];
      target_rate[mode] = full_principal_rate-normal_principal_rate;
    } else {
      // Byte-recovered zero-rate target used by the validated cd7cefef...
      // executable: homogeneous incoming characteristic data have zero rate.
      target_rate[mode] = 0.0;
    }
    scalar_residual[mode] = target_rate[mode] - rate;
    AtomicDiagnosticMax(diag, mode < 2 ? CPBC_GAUGE_AMPLITUDE :
                        CPBC_CONSTRAINT_AMPLITUDE,
                        amplitude, collect_diagnostics);
    AtomicDiagnosticMax(diag, CPBC_MAX_CHARACTERISTIC,
                        amplitude, collect_diagnostics);
    if (collect_diagnostics) {
      Real outgoing_amplitude = 0.0;
      if (mode == 0) {
        outgoing_amplitude =
            -(lambda_lapse[1]-beta_bg_normal)*scalar_p[0]/chi +
            scalar_d[2];
      } else if (mode == 1) {
        const Real lambda = lambda_shift_long[1];
        const Real mu = lambda-beta_full_normal;
        const Real delta_bg = lambda-beta_bg_normal;
        outgoing_amplitude =
            alpha*SQR(delta_bg)*light_shift_separation_out*scalar_p[0] +
            0.5*alpha*shift_q*lapse_shift_separation*scalar_p[1] +
            0.25*delta_bg*(4.0*chi*SQR(alpha)-3.0*SQR(mu))*
                lapse_shift_separation*scalar_p[3] +
            0.5*SQR(alpha)*delta_bg*lapse_shift_separation*scalar_d[0] -
            chi*alpha*delta_bg*light_shift_separation_out*scalar_d[2] +
            lapse_shift_separation*light_shift_separation_out*scalar_d[3];
      } else if (mode == 2) {
        outgoing_amplitude =
            -sqrt_chi*scalar_p[1] + 0.5*chi*scalar_p[3] +
            scalar_d[0];
      } else {
        outgoing_amplitude =
            -4.0*scalar_p[0]/(3.0*sqrt_chi) -
            2.0*scalar_p[1]/(3.0*sqrt_chi) +
            2.0*scalar_p[2]/sqrt_chi - scalar_p[3] + scalar_d[1];
      }
      AtomicDiagnosticMax(
          diag, mode < 2 ? CPBC_OUTGOING_GAUGE_AMPLITUDE :
                           CPBC_OUTGOING_CONSTRAINT_AMPLITUDE,
          outgoing_amplitude,true);
      AtomicDiagnosticMax(diag,CPBC_MAX_OUTGOING_CHARACTERISTIC,
                          outgoing_amplitude,true);
    }
  }

  // The incoming p-map is sparse.  Solve it analytically to avoid a
  // per-thread augmented 4x5 matrix in the fused GPU face kernel.
  Real scalar_correction[4];
  const Real scalar_map_scale =
      fmax(1.0,fmax(fabs(left_p[1][3]),fabs(left_p[1][1]*sqrt_chi)));
  const Real scalar_gamma_denominator =
      left_p[1][3]-0.5*sqrt_chi*left_p[1][1];
  if (fabs(left_p[0][0]) <=
          BoundaryRoundoffTolerance()*scalar_map_scale ||
      fabs(scalar_gamma_denominator) <=
          BoundaryRoundoffTolerance()*scalar_map_scale) {
    return CPBC_INVALID_SCALAR_MAP;
  }
  scalar_correction[0] = scalar_residual[0]/left_p[0][0];
  scalar_correction[3] =
      (scalar_residual[1] -
       left_p[1][0]*scalar_correction[0] -
       left_p[1][1]*scalar_residual[2]/sqrt_chi)/
      scalar_gamma_denominator;
  scalar_correction[1] =
      (scalar_residual[2] -
       0.5*chi*scalar_correction[3])/sqrt_chi;
  scalar_correction[2] =
      (scalar_residual[3] -
       left_p[3][0]*scalar_correction[0] -
       left_p[3][1]*scalar_correction[1] -
       left_p[3][3]*scalar_correction[3])/left_p[3][2];
  for (int n = 0; n < 4; ++n) {
    scalar_p_rhs[n] += scalar_correction[n];
    AtomicDiagnosticMax(diag,CPBC_MAX_CORRECTION,scalar_correction[n],
                        collect_diagnostics);
  }
  for (int mode = 0; mode < 4; ++mode) {
    Real corrected_rate = 0.0;
    for (int n = 0; n < 4; ++n) {
      corrected_rate += left_p[mode][n]*scalar_p_rhs[n];
    }
    if (mode == 0) {
      corrected_rate += scalar_d_rhs[2];
    } else if (mode == 1) {
      corrected_rate += shift_left_d0*scalar_d_rhs[0] +
                        shift_left_d2*scalar_d_rhs[2] +
                        shift_left_d3*scalar_d_rhs[3];
    } else if (mode == 2) {
      corrected_rate += scalar_d_rhs[0];
    } else {
      corrected_rate += scalar_d_rhs[1];
    }
    AtomicDiagnosticMax(diag,CPBC_ENFORCEMENT_ERROR,
                        corrected_rate-target_rate[mode],collect_diagnostics);
  }

  Real vector_correction_A[2];
  Real vector_correction_Gamma[2];
  const Real *tangent_d[2] = {tangent1_d,tangent2_d};
  const Real *tangent_u[2] = {tangent1_u,tangent2_u};
  for (int tangent = 0; tangent < 2; ++tangent) {
    Real vector_p[2] = {
      ProjectTensor(residual_A,normal_u,tangent_u[tangent]), 0.0};
    Real vector_p_rhs[2] = {
      ProjectTensor(rhs_A,normal_u,tangent_u[tangent]), 0.0};
    for (int a = 0; a < 3; ++a) {
      vector_p[1] += tangent_d[tangent][a] *
                     u(m,Z4c::I_Z4C_GAMX+a,k,j,i);
      vector_p_rhs[1] += tangent_d[tangent][a] *
                         u_rhs(m,Z4c::I_Z4C_GAMX+a,k,j,i);
    }
    Real vector_d[2] = {
      ProjectTensor(derivative_metric,normal_u,tangent_u[tangent]), 0.0};
    Real vector_d_rhs[2] = {
      ProjectTensor(derivative_rhs_metric,normal_u,tangent_u[tangent]), 0.0};
    Real vector_p_normal[2] = {
      ProjectTensor(centered_derivative_A,normal_u,tangent_u[tangent]), 0.0};
    Real vector_d_normal[2] = {
      ProjectTensor(centered_second_metric,normal_u,tangent_u[tangent]), 0.0};
    Real vector_p_principal[2] = {
      ProjectTensor(principal_A_rhs,normal_u,tangent_u[tangent]), 0.0};
    Real vector_d_principal[2] = {
      ProjectTensor(principal_metric_normal_rhs,
                    normal_u,tangent_u[tangent]), 0.0};
    for (int a = 0; a < 3; ++a) {
      vector_d[1] += tangent_d[tangent][a]*derivative_beta[a];
      vector_d_rhs[1] +=
          tangent_d[tangent][a]*derivative_rhs_beta[a];
      vector_p_normal[1] +=
          tangent_d[tangent][a]*centered_derivative_Gamma[a];
      vector_d_normal[1] +=
          tangent_d[tangent][a]*centered_second_beta[a];
      vector_p_principal[1] +=
          tangent_d[tangent][a]*principal_Gamma_rhs[a];
      vector_d_principal[1] +=
          tangent_d[tangent][a]*principal_beta_normal_rhs[a];
    }

    // Exact incoming transverse-gauge row for the hybrid full/background
    // advection root.  The coefficient of d_s beta_A is normalized to one.
    const Real transverse_delta_bg =
        lambda_shift_trans[0]-beta_bg_normal;
    Real gauge_amplitude =
        transverse_delta_bg*vector_p[1] + vector_d[1];
    Real gauge_rate =
        transverse_delta_bg*vector_p_rhs[1] + vector_d_rhs[1];
    Real gauge_normal_principal_rate =
        lambda_shift_trans[0]*
        (transverse_delta_bg*vector_p_normal[1] + vector_d_normal[1]);
    Real gauge_full_principal_rate =
        transverse_delta_bg*vector_p_principal[1] +
        vector_d_principal[1];
    Real gauge_target = TangentialPrincipal ?
        gauge_full_principal_rate-gauge_normal_principal_rate : 0.0;
    Real gauge_residual = gauge_target-gauge_rate;
    Real delta_gamma = gauge_residual/transverse_delta_bg;

    Real constraint_amplitude =
        -2.0*vector_p[0]/sqrt_chi - vector_p[1] + vector_d[0];
    Real outgoing_gauge_amplitude =
        (lambda_shift_trans[1]-beta_bg_normal)*vector_p[1] +
        vector_d[1];
    Real outgoing_constraint_amplitude =
        2.0*vector_p[0]/sqrt_chi - vector_p[1] + vector_d[0];
    Real constraint_rate =
        -2.0*vector_p_rhs[0]/sqrt_chi -
        vector_p_rhs[1] + vector_d_rhs[0];
    Real constraint_normal_principal_rate =
        lambda_light[0]*
        (-2.0*vector_p_normal[0]/sqrt_chi -
         vector_p_normal[1] + vector_d_normal[0]);
    Real constraint_full_principal_rate =
        -2.0*vector_p_principal[0]/sqrt_chi -
        vector_p_principal[1] + vector_d_principal[0];
    Real constraint_target = TangentialPrincipal ?
        constraint_full_principal_rate-constraint_normal_principal_rate : 0.0;
    Real constraint_residual = constraint_target-constraint_rate;
    Real delta_A =
        -0.5*sqrt_chi*(constraint_residual + delta_gamma);

    vector_correction_A[tangent] = delta_A;
    vector_correction_Gamma[tangent] = delta_gamma;
    vector_p_rhs[0] += delta_A;
    vector_p_rhs[1] += delta_gamma;
    AtomicDiagnosticMax(diag,CPBC_GAUGE_AMPLITUDE,gauge_amplitude,
                        collect_diagnostics);
    AtomicDiagnosticMax(diag,CPBC_CONSTRAINT_AMPLITUDE,constraint_amplitude,
                        collect_diagnostics);
    AtomicDiagnosticMax(diag,CPBC_MAX_CHARACTERISTIC,
                        fmax(fabs(gauge_amplitude),fabs(constraint_amplitude)),
                        collect_diagnostics);
    AtomicDiagnosticMax(diag,CPBC_OUTGOING_GAUGE_AMPLITUDE,
                        outgoing_gauge_amplitude,collect_diagnostics);
    AtomicDiagnosticMax(diag,CPBC_OUTGOING_CONSTRAINT_AMPLITUDE,
                        outgoing_constraint_amplitude,collect_diagnostics);
    AtomicDiagnosticMax(
        diag,CPBC_MAX_OUTGOING_CHARACTERISTIC,
        fmax(fabs(outgoing_gauge_amplitude),
             fabs(outgoing_constraint_amplitude)),collect_diagnostics);
    AtomicDiagnosticMax(diag,CPBC_MAX_CORRECTION,
                        fmax(fabs(delta_A),fabs(delta_gamma)),
                        collect_diagnostics);
    Real gauge_error =
        transverse_delta_bg*vector_p_rhs[1] + vector_d_rhs[1] -
        gauge_target;
    Real constraint_error =
        -2.0*vector_p_rhs[0]/sqrt_chi -
        vector_p_rhs[1] + vector_d_rhs[0] - constraint_target;
    AtomicDiagnosticMax(diag,CPBC_ENFORCEMENT_ERROR,
                        fmax(fabs(gauge_error),fabs(constraint_error)),
                        collect_diagnostics);
  }

  Real tensor_correction[2];
  for (int polarization = 0; polarization < 2; ++polarization) {
    Real A_component;
    Real A_rhs_component;
    Real metric_derivative;
    Real metric_rhs_derivative;
    Real A_normal_derivative;
    Real metric_normal_second_derivative;
    Real A_principal_component;
    Real metric_principal_normal_derivative;
    if (polarization == 0) {
      A_component = 0.5*(ProjectTensor(residual_A,tangent1_u,tangent1_u) -
                         ProjectTensor(residual_A,tangent2_u,tangent2_u));
      A_rhs_component = 0.5*(ProjectTensor(rhs_A,tangent1_u,tangent1_u) -
                             ProjectTensor(rhs_A,tangent2_u,tangent2_u));
      metric_derivative =
          0.5*(ProjectTensor(derivative_metric,tangent1_u,tangent1_u) -
               ProjectTensor(derivative_metric,tangent2_u,tangent2_u));
      metric_rhs_derivative =
          0.5*(ProjectTensor(derivative_rhs_metric,tangent1_u,tangent1_u) -
               ProjectTensor(derivative_rhs_metric,tangent2_u,tangent2_u));
      A_normal_derivative =
          0.5*(ProjectTensor(centered_derivative_A,tangent1_u,tangent1_u) -
               ProjectTensor(centered_derivative_A,tangent2_u,tangent2_u));
      metric_normal_second_derivative =
          0.5*(ProjectTensor(centered_second_metric,tangent1_u,tangent1_u) -
               ProjectTensor(centered_second_metric,tangent2_u,tangent2_u));
      A_principal_component =
          0.5*(ProjectTensor(principal_A_rhs,tangent1_u,tangent1_u) -
               ProjectTensor(principal_A_rhs,tangent2_u,tangent2_u));
      metric_principal_normal_derivative =
          0.5*(ProjectTensor(principal_metric_normal_rhs,
                             tangent1_u,tangent1_u) -
               ProjectTensor(principal_metric_normal_rhs,
                             tangent2_u,tangent2_u));
    } else {
      A_component = ProjectTensor(residual_A,tangent1_u,tangent2_u);
      A_rhs_component = ProjectTensor(rhs_A,tangent1_u,tangent2_u);
      metric_derivative =
          ProjectTensor(derivative_metric,tangent1_u,tangent2_u);
      metric_rhs_derivative =
          ProjectTensor(derivative_rhs_metric,tangent1_u,tangent2_u);
      A_normal_derivative =
          ProjectTensor(centered_derivative_A,tangent1_u,tangent2_u);
      metric_normal_second_derivative =
          ProjectTensor(centered_second_metric,tangent1_u,tangent2_u);
      A_principal_component =
          ProjectTensor(principal_A_rhs,tangent1_u,tangent2_u);
      metric_principal_normal_derivative =
          ProjectTensor(principal_metric_normal_rhs,
                        tangent1_u,tangent2_u);
    }
    Real radiation_amplitude =
        -2.0*A_component/sqrt_chi + metric_derivative;
    Real outgoing_radiation_amplitude =
        2.0*A_component/sqrt_chi + metric_derivative;
    Real radiation_rate =
        -2.0*A_rhs_component/sqrt_chi + metric_rhs_derivative;
    Real radiation_normal_principal_rate =
        lambda_light[0]*
        (-2.0*A_normal_derivative/sqrt_chi +
         metric_normal_second_derivative);
    Real radiation_full_principal_rate =
        -2.0*A_principal_component/sqrt_chi +
        metric_principal_normal_derivative;
    Real radiation_target = TangentialPrincipal ?
        radiation_full_principal_rate-radiation_normal_principal_rate : 0.0;
    Real radiation_residual = radiation_target-radiation_rate;
    tensor_correction[polarization] = -0.5*sqrt_chi*radiation_residual;
    Real corrected_radiation_rate =
        -2.0*(A_rhs_component+tensor_correction[polarization])/sqrt_chi +
        metric_rhs_derivative;
    AtomicDiagnosticMax(diag,CPBC_RADIATION_AMPLITUDE,radiation_amplitude,
                        collect_diagnostics);
    AtomicDiagnosticMax(diag,CPBC_MAX_CHARACTERISTIC,radiation_amplitude,
                        collect_diagnostics);
    AtomicDiagnosticMax(diag,CPBC_OUTGOING_RADIATION_AMPLITUDE,
                        outgoing_radiation_amplitude,collect_diagnostics);
    AtomicDiagnosticMax(diag,CPBC_MAX_OUTGOING_CHARACTERISTIC,
                        outgoing_radiation_amplitude,collect_diagnostics);
    AtomicDiagnosticMax(diag,CPBC_MAX_CORRECTION,
                        tensor_correction[polarization],collect_diagnostics);
    AtomicDiagnosticMax(diag,CPBC_ENFORCEMENT_ERROR,
                        corrected_radiation_rate-radiation_target,
                        collect_diagnostics);
  }

  const Real maximum_shift_ratio =
      fmax(fabs(beta_full_normal)/c_light,
           fabs(beta_full_normal-beta_bg_normal)/
               fmin(sqrt(lapse_discriminant),
                    fmin(sqrt(shift_long_discriminant),
                         sqrt(shift_trans_discriminant))));
  AtomicDiagnosticMax(diag,CPBC_MAX_SHIFT_RATIO,
                      maximum_shift_ratio,collect_diagnostics);

  u_rhs(m,Z4c::I_Z4C_KHAT,k,j,i) = scalar_p_rhs[0];
  u_rhs(m,Z4c::I_Z4C_THETA,k,j,i) = scalar_p_rhs[1];
  for (int a = 0; a < 3; ++a) {
    Real gamma_correction =
        scalar_correction[3]*normal_u[a] +
        vector_correction_Gamma[0]*tangent1_u[a] +
        vector_correction_Gamma[1]*tangent2_u[a];
    u_rhs(m,Z4c::I_Z4C_GAMX+a,k,j,i) += gamma_correction;
  }

  // Reconstruct a trace-free Cartesian A_ij correction from the scalar,
  // two vector, and two tensor characteristic sectors.
  for (int a = 0; a < 3; ++a) {
    for (int b = a; b < 3; ++b) {
      Real correction =
          scalar_correction[2] *
              (normal_d[a]*normal_d[b] -
               0.5*(tangent1_d[a]*tangent1_d[b] +
                    tangent2_d[a]*tangent2_d[b])) +
          vector_correction_A[0] *
              (normal_d[a]*tangent1_d[b] +
               tangent1_d[a]*normal_d[b]) +
          vector_correction_A[1] *
              (normal_d[a]*tangent2_d[b] +
               tangent2_d[a]*normal_d[b]) +
          tensor_correction[0] *
              (tangent1_d[a]*tangent1_d[b] -
               tangent2_d[a]*tangent2_d[b]) +
          tensor_correction[1] *
              (tangent1_d[a]*tangent2_d[b] +
               tangent2_d[a]*tangent1_d[b]);
      u_rhs(m,Z4c::I_Z4C_AXX+SymmetricOffset(a,b),k,j,i) += correction;
    }
  }
  return CPBC_VALID;
}

// Legacy placeholder Sommerfeld treatment, retained as the default for
// backward compatibility.
KOKKOS_INLINE_FUNCTION
void Z4cSommerfeld(const Z4c::Z4c_vars& z4c, const Z4c::Z4c_vars& rhs,
    const RegionIndcs &indcs, const DualArray1D<RegionSize> &size,
    const int m, const int k, const int j, const int i) {
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dKhat_d;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dTheta_d;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> dGam_du;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> dA_ddd;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> s_u;
  Real idx[] = {1./size.d_view(m).dx1, 1./size.d_view(m).dx2,
                1./size.d_view(m).dx3};

  for (int a = 0; a < 3; a++) {
    dKhat_d(a) = Dx<2>(a, idx, z4c.vKhat, m, k, j, i);
    dTheta_d(a) = Dx<2>(a, idx, z4c.vTheta, m, k, j, i);
    for (int b = 0; b < 3; b++) {
      dGam_du(b,a) = Dx<2>(b, idx, z4c.vGam_u, m, a, k, j, i);
    }
  }
  for (int a = 0; a < 3; a++) {
    for (int b = a; b < 3; b++) {
      for (int c = 0; c < 3; c++) {
        dA_ddd(c,a,b) = Dx<2>(c, idx, z4c.vA_dd, m, a, b, k, j, i);
      }
    }
  }

  Real x1v = CellCenterX(i-indcs.is, indcs.nx1, size.d_view(m).x1min,
                         size.d_view(m).x1max);
  Real x2v = CellCenterX(j-indcs.js, indcs.nx2, size.d_view(m).x2min,
                         size.d_view(m).x2max);
  Real x3v = CellCenterX(k-indcs.ks, indcs.nx3, size.d_view(m).x3min,
                         size.d_view(m).x3max);
  Real r = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
  s_u(0) = x1v/r;
  s_u(1) = x2v/r;
  s_u(2) = x3v/r;

  rhs.vTheta(m,k,j,i) = -z4c.vTheta(m,k,j,i)/r;
  rhs.vKhat(m,k,j,i) = -sqrt(2.)*z4c.vKhat(m,k,j,i)/r;
  for (int a = 0; a < 3; a++) {
    rhs.vTheta(m,k,j,i) -= s_u(a)*dTheta_d(a);
    rhs.vKhat(m,k,j,i) -= sqrt(2.)*s_u(a)*dKhat_d(a);
  }
  for (int a = 0; a < 3; a++) {
    rhs.vGam_u(m,a,k,j,i) = -z4c.vGam_u(m,a,k,j,i)/r;
    for (int b = 0; b < 3; b++) {
      rhs.vGam_u(m,a,k,j,i) -= s_u(b)*dGam_du(b,a);
    }
  }
  for (int a = 0; a < 3; a++) {
    for (int b = a; b < 3; b++) {
      rhs.vA_dd(m,a,b,k,j,i) = -z4c.vA_dd(m,a,b,k,j,i)/r;
      for (int c = 0; c < 3; c++) {
        rhs.vA_dd(m,a,b,k,j,i) -= s_u(c)*dA_ddd(c,a,b);
      }
    }
  }
}

}  // namespace

TaskStatus Z4c::Z4cBoundaryRHS(Driver *pdriver, int stage) {
  auto &pm = pmy_pack->pmesh;
  auto &mb_bcs = pmy_pack->pmb->mb_bcs;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  const int nmb = pmy_pack->nmb_thispack;
  const int is = indcs.is;
  const int ie = indcs.ie;
  const int js = indcs.js;
  const int je = indcs.je;
  const int ks = indcs.ks;
  const int ke = indcs.ke;
  const bool user_sbc = opt.user_Sbc;

  if (opt.boundary_rhs_mode == boundary_rhs_sommerfeld) {
    auto &z4c_ = z4c;
    auto &rhs_ = rhs;
    if (pm->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::outflow ||
        pm->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::diode ||
        pm->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::user ||
        pm->mesh_bcs[BoundaryFace::outer_x1] == BoundaryFlag::outflow ||
        pm->mesh_bcs[BoundaryFace::outer_x1] == BoundaryFlag::diode ||
        pm->mesh_bcs[BoundaryFace::outer_x1] == BoundaryFlag::user) {
      par_for("z4crhs_bc_x1", DevExeSpace(), 0,nmb-1,ks,ke,js,je,
      KOKKOS_LAMBDA(int m, int k, int j) {
        if (UsesBoundaryRHS(mb_bcs.d_view(m,BoundaryFace::inner_x1),user_sbc)) {
          Z4cSommerfeld(z4c_,rhs_,indcs,size,m,k,j,is);
        }
        if (UsesBoundaryRHS(mb_bcs.d_view(m,BoundaryFace::outer_x1),user_sbc)) {
          Z4cSommerfeld(z4c_,rhs_,indcs,size,m,k,j,ie);
        }
      });
    }
    if (pm->mesh_bcs[BoundaryFace::inner_x2] == BoundaryFlag::outflow ||
        pm->mesh_bcs[BoundaryFace::inner_x2] == BoundaryFlag::diode ||
        pm->mesh_bcs[BoundaryFace::inner_x2] == BoundaryFlag::user ||
        pm->mesh_bcs[BoundaryFace::outer_x2] == BoundaryFlag::outflow ||
        pm->mesh_bcs[BoundaryFace::outer_x2] == BoundaryFlag::diode ||
        pm->mesh_bcs[BoundaryFace::outer_x2] == BoundaryFlag::user) {
      par_for("z4crhs_bc_x2", DevExeSpace(), 0,nmb-1,ks,ke,is,ie,
      KOKKOS_LAMBDA(int m, int k, int i) {
        if (UsesBoundaryRHS(mb_bcs.d_view(m,BoundaryFace::inner_x2),user_sbc)) {
          Z4cSommerfeld(z4c_,rhs_,indcs,size,m,k,js,i);
        }
        if (UsesBoundaryRHS(mb_bcs.d_view(m,BoundaryFace::outer_x2),user_sbc)) {
          Z4cSommerfeld(z4c_,rhs_,indcs,size,m,k,je,i);
        }
      });
    }
    if (pm->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::outflow ||
        pm->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::diode ||
        pm->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::user ||
        pm->mesh_bcs[BoundaryFace::outer_x3] == BoundaryFlag::outflow ||
        pm->mesh_bcs[BoundaryFace::outer_x3] == BoundaryFlag::diode ||
        pm->mesh_bcs[BoundaryFace::outer_x3] == BoundaryFlag::user) {
      par_for("z4crhs_bc_x3", DevExeSpace(), 0,nmb-1,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int j, int i) {
        if (UsesBoundaryRHS(mb_bcs.d_view(m,BoundaryFace::inner_x3),user_sbc)) {
          Z4cSommerfeld(z4c_,rhs_,indcs,size,m,ks,j,i);
        }
        if (UsesBoundaryRHS(mb_bcs.d_view(m,BoundaryFace::outer_x3),user_sbc)) {
          Z4cSommerfeld(z4c_,rhs_,indcs,size,m,ke,j,i);
        }
      });
    }
    return TaskStatus::complete;
  }

  if (!(use_analytic_background) || SetADMBackground == nullptr) {
    if (global_variable::my_rank == 0) {
      std::cout << "### FATAL ERROR: characteristic_cpbc requires an enrolled "
                << "analytic Z4c background callback." << std::endl;
    }
    std::exit(EXIT_FAILURE);
  }

  // Restrict each orientation launch to MeshBlocks that actually touch one of
  // its two physical faces.  This is a host-side O(nmb) scan over six flags;
  // the compact device list is copied only when AMR or load balancing changes
  // the layout.  A block may occur in more than one orientation list, while
  // the per-cell X1/X2/X3 ownership rules below remain disjoint.
  int updated_boundary_block_count[3] = {0,0,0};
  const BoundaryFace inner_face[3] = {
      BoundaryFace::inner_x1,BoundaryFace::inner_x2,BoundaryFace::inner_x3};
  const BoundaryFace outer_face[3] = {
      BoundaryFace::outer_x1,BoundaryFace::outer_x2,BoundaryFace::outer_x3};
  bool boundary_block_lists_changed = false;
  for (int dir = 0; dir < 3; ++dir) {
    for (int m = 0; m < nmb; ++m) {
      if (!UsesBoundaryRHS(mb_bcs.h_view(m,inner_face[dir]),user_sbc) &&
          !UsesBoundaryRHS(mb_bcs.h_view(m,outer_face[dir]),user_sbc)) {
        continue;
      }
      const int compact_m = updated_boundary_block_count[dir]++;
      if (compact_m >= characteristic_bc_boundary_block_count[dir] ||
          characteristic_bc_invalid.h_view(
              CPBC_NINVALID + 3*compact_m + dir) != m) {
        boundary_block_lists_changed = true;
      }
    }
    if (updated_boundary_block_count[dir] !=
        characteristic_bc_boundary_block_count[dir]) {
      boundary_block_lists_changed = true;
    }
  }
  for (int dir = 0; dir < 3; ++dir) {
    characteristic_bc_boundary_block_count[dir] =
        updated_boundary_block_count[dir];
  }
  if (boundary_block_lists_changed) {
    int compact_m[3] = {0,0,0};
    for (int dir = 0; dir < 3; ++dir) {
      for (int m = 0; m < nmb; ++m) {
        if (UsesBoundaryRHS(mb_bcs.h_view(m,inner_face[dir]),user_sbc) ||
            UsesBoundaryRHS(mb_bcs.h_view(m,outer_face[dir]),user_sbc)) {
          characteristic_bc_invalid.h_view(
              CPBC_NINVALID + 3*compact_m[dir]++ + dir) = m;
        }
      }
    }
    characteristic_bc_invalid.template modify<HostMemSpace>();
    characteristic_bc_invalid.template sync<DevExeSpace>();
  }

  const bool collect_diagnostics =
      !characteristic_bc_announced ||
      (opt.characteristic_bc_diagnostics &&
       stage == pdriver->nexp_stages &&
       pm->ncycle % opt.characteristic_bc_diagnostic_interval == 0);
  const bool measure_performance =
      opt.characteristic_bc_diagnostics &&
      opt.characteristic_bc_diagnostic_interval > 1 &&
      stage == pdriver->nexp_stages &&
      pm->ncycle % opt.characteristic_bc_diagnostic_interval ==
          opt.characteristic_bc_diagnostic_interval - 1;
  if (collect_diagnostics) {
    auto invalid_counts_ = Kokkos::subview(
        characteristic_bc_invalid.d_view,
        std::make_pair(0,static_cast<int>(CPBC_NINVALID)));
    Kokkos::deep_copy(DevExeSpace(),invalid_counts_,0);
    Kokkos::deep_copy(DevExeSpace(),characteristic_bc_diag,0.0);
    Kokkos::fence();
  } else if (measure_performance) {
    // Time a normal boundary step without the contended diagnostic atomics.
    Kokkos::fence();
  }
  Kokkos::Timer kernel_timer;

  auto u_ = u0;
  auto full_ = u_full;
  auto bg_ = u_bg;
  auto rhs_ = u_rhs;
  DvceArray5D<Real> matter_;
  const bool has_matter = pmy_pack->ptmunu != nullptr;
  if (has_matter) matter_ = pmy_pack->ptmunu->u_tmunu;
  auto diag_ = characteristic_bc_diag;
  auto invalid_ = characteristic_bc_invalid.d_view;
  const Options opt_ = opt;
  const Real time = pm->time;

  // Compile separate kernels for the validated zero-rate source and the
  // experimental tangential-principal source. This keeps the latter's larger
  // stencil and register footprint out of the default kernel.
  auto launch_characteristic_kernels = [&](auto source_tag) {
  constexpr bool tangential_principal = decltype(source_tag)::value;
  const char *x1_kernel = tangential_principal ?
      "z4c_cpbc_tangential_principal_x1" : "z4c_cpbc_zero_rate_x1";
  const char *x2_kernel = tangential_principal ?
      "z4c_cpbc_tangential_principal_x2" : "z4c_cpbc_zero_rate_x2";
  const char *x3_kernel = tangential_principal ?
      "z4c_cpbc_tangential_principal_x3" : "z4c_cpbc_zero_rate_x3";

  // The kernels have disjoint ownership. X1 owns every cell incident on an X1
  // physical face; X2 skips those cells; X3 skips both X1 and X2 incidents.
  // The normal itself includes every incident face, so edges/corners use a
  // symmetric composite normal without racing multiple device writes.
  if (characteristic_bc_boundary_block_count[0] > 0) {
    par_for(x1_kernel,DevExeSpace(),
    0,characteristic_bc_boundary_block_count[0]-1,ks,ke,js,je,
    KOKKOS_LAMBDA(int m, int k, int j) {
    m = invalid_(CPBC_NINVALID + 3*m);
    for (int outer = 0; outer < 2; ++outer) {
      BoundaryFace face = outer == 0 ? BoundaryFace::inner_x1 :
                                       BoundaryFace::outer_x1;
      if (!UsesBoundaryRHS(mb_bcs.d_view(m,face),user_sbc)) continue;
      int point_i = outer == 0 ? is : ie;
      int side[3] = {outer == 0 ? -1 : 1,0,0};
      if (j == js &&
          UsesBoundaryRHS(mb_bcs.d_view(m,BoundaryFace::inner_x2),user_sbc)) {
        side[1] = -1;
      } else if (j == je &&
                 UsesBoundaryRHS(mb_bcs.d_view(m,BoundaryFace::outer_x2),user_sbc)) {
        side[1] = 1;
      }
      if (k == ks &&
          UsesBoundaryRHS(mb_bcs.d_view(m,BoundaryFace::inner_x3),user_sbc)) {
        side[2] = -1;
      } else if (k == ke &&
                 UsesBoundaryRHS(mb_bcs.d_view(m,BoundaryFace::outer_x3),user_sbc)) {
        side[2] = 1;
      }
      Real idx[3] = {1.0/size.d_view(m).dx1,1.0/size.d_view(m).dx2,
                     1.0/size.d_view(m).dx3};
      int status =
          ApplyResidualCharacteristicBC<tangential_principal>(
          u_,full_,bg_,rhs_,matter_,has_matter,opt_,diag_,time,
          m,k,j,point_i,side,idx,collect_diagnostics);
      if (collect_diagnostics && status != CPBC_VALID) {
        Kokkos::atomic_add(&invalid_(status),1);
      }
    }
    });
  }

  if (characteristic_bc_boundary_block_count[1] > 0) {
    par_for(x2_kernel,DevExeSpace(),
    0,characteristic_bc_boundary_block_count[1]-1,ks,ke,is,ie,
    KOKKOS_LAMBDA(int m, int k, int i) {
    m = invalid_(CPBC_NINVALID + 3*m + 1);
    bool incident_x1 =
        (i == is && UsesBoundaryRHS(
             mb_bcs.d_view(m,BoundaryFace::inner_x1),user_sbc)) ||
        (i == ie && UsesBoundaryRHS(
             mb_bcs.d_view(m,BoundaryFace::outer_x1),user_sbc));
    if (incident_x1) return;
    for (int outer = 0; outer < 2; ++outer) {
      BoundaryFace face = outer == 0 ? BoundaryFace::inner_x2 :
                                       BoundaryFace::outer_x2;
      if (!UsesBoundaryRHS(mb_bcs.d_view(m,face),user_sbc)) continue;
      int point_j = outer == 0 ? js : je;
      int side[3] = {0,outer == 0 ? -1 : 1,0};
      if (k == ks &&
          UsesBoundaryRHS(mb_bcs.d_view(m,BoundaryFace::inner_x3),user_sbc)) {
        side[2] = -1;
      } else if (k == ke &&
                 UsesBoundaryRHS(mb_bcs.d_view(m,BoundaryFace::outer_x3),user_sbc)) {
        side[2] = 1;
      }
      Real idx[3] = {1.0/size.d_view(m).dx1,1.0/size.d_view(m).dx2,
                     1.0/size.d_view(m).dx3};
      int status =
          ApplyResidualCharacteristicBC<tangential_principal>(
          u_,full_,bg_,rhs_,matter_,has_matter,opt_,diag_,time,
          m,k,point_j,i,side,idx,collect_diagnostics);
      if (collect_diagnostics && status != CPBC_VALID) {
        Kokkos::atomic_add(&invalid_(status),1);
      }
    }
    });
  }

  if (characteristic_bc_boundary_block_count[2] > 0) {
    par_for(x3_kernel,DevExeSpace(),
    0,characteristic_bc_boundary_block_count[2]-1,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int j, int i) {
    m = invalid_(CPBC_NINVALID + 3*m + 2);
    bool incident_x1 =
        (i == is && UsesBoundaryRHS(
             mb_bcs.d_view(m,BoundaryFace::inner_x1),user_sbc)) ||
        (i == ie && UsesBoundaryRHS(
             mb_bcs.d_view(m,BoundaryFace::outer_x1),user_sbc));
    bool incident_x2 =
        (j == js && UsesBoundaryRHS(
             mb_bcs.d_view(m,BoundaryFace::inner_x2),user_sbc)) ||
        (j == je && UsesBoundaryRHS(
             mb_bcs.d_view(m,BoundaryFace::outer_x2),user_sbc));
    if (incident_x1 || incident_x2) return;
    for (int outer = 0; outer < 2; ++outer) {
      BoundaryFace face = outer == 0 ? BoundaryFace::inner_x3 :
                                       BoundaryFace::outer_x3;
      if (!UsesBoundaryRHS(mb_bcs.d_view(m,face),user_sbc)) continue;
      int point_k = outer == 0 ? ks : ke;
      int side[3] = {0,0,outer == 0 ? -1 : 1};
      Real idx[3] = {1.0/size.d_view(m).dx1,1.0/size.d_view(m).dx2,
                     1.0/size.d_view(m).dx3};
      int status =
          ApplyResidualCharacteristicBC<tangential_principal>(
          u_,full_,bg_,rhs_,matter_,has_matter,opt_,diag_,time,
          m,point_k,j,i,side,idx,collect_diagnostics);
      if (collect_diagnostics && status != CPBC_VALID) {
        Kokkos::atomic_add(&invalid_(status),1);
      }
    }
    });
  }
  };

  if (opt.characteristic_bc_source_mode ==
      characteristic_bc_source_tangential_principal) {
    launch_characteristic_kernels(CharacteristicSourceTag<true>{});
  } else {
    launch_characteristic_kernels(CharacteristicSourceTag<false>{});
  }

  if (measure_performance) {
    Kokkos::fence();
    Real max_kernel_seconds = kernel_timer.seconds();
    Real max_volume_rhs_seconds = characteristic_bc_volume_rhs_seconds;
    Real max_rank_fraction =
        characteristic_bc_volume_rhs_seconds > 0.0 ?
        max_kernel_seconds/characteristic_bc_volume_rhs_seconds : 0.0;
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE,&max_kernel_seconds,1,MPI_ATHENA_REAL,MPI_MAX,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&max_volume_rhs_seconds,1,MPI_ATHENA_REAL,
                  MPI_MAX,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&max_rank_fraction,1,MPI_ATHENA_REAL,MPI_MAX,
                  MPI_COMM_WORLD);
#endif
    characteristic_bc_kernel_seconds = max_kernel_seconds;
    characteristic_bc_volume_rhs_seconds = max_volume_rhs_seconds;
    characteristic_bc_max_rank_fraction = max_rank_fraction;
    characteristic_bc_performance_valid =
        max_kernel_seconds >= 0.0 && max_volume_rhs_seconds > 0.0 &&
        isfinite(max_kernel_seconds) && isfinite(max_volume_rhs_seconds) &&
        isfinite(max_rank_fraction);
  }

  if (collect_diagnostics) {
    Kokkos::fence();
    Real max_diagnostic_kernel_seconds = kernel_timer.seconds();
    auto invalid_h = Kokkos::create_mirror_view_and_copy(
        HostMemSpace(),characteristic_bc_invalid.d_view);
    int invalid_counts[CPBC_NINVALID];
    for (int n = 0; n < CPBC_NINVALID; ++n) {
      invalid_counts[n] = invalid_h(n);
    }
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE,invalid_counts,CPBC_NINVALID,MPI_INT,MPI_SUM,
                  MPI_COMM_WORLD);
#endif
    int invalid_count = 0;
    for (int n = 1; n < CPBC_NINVALID; ++n) {
      invalid_count += invalid_counts[n];
    }
    if (invalid_count != 0) {
      if (global_variable::my_rank == 0) {
        auto bg_h =
            Kokkos::create_mirror_view_and_copy(HostMemSpace(),u_bg);
        std::cout << "### FATAL ERROR: characteristic_cpbc found "
                  << invalid_count
                  << " boundary cells with an invalid metric, gauge driver, "
                  << "cone separation, characteristic speed, or excessive "
                  << "matter energy density."
                  << " metric_det="
                  << invalid_counts[CPBC_INVALID_METRIC_DETERMINANT]
                  << " metric_normal="
                  << invalid_counts[CPBC_INVALID_METRIC_NORMAL]
                  << " metric_tangent1="
                  << invalid_counts[CPBC_INVALID_METRIC_TANGENT1]
                  << " metric_tangent2="
                  << invalid_counts[CPBC_INVALID_METRIC_TANGENT2]
                  << " matter=" << invalid_counts[CPBC_INVALID_MATTER]
                  << " coefficient="
                  << invalid_counts[CPBC_INVALID_COEFFICIENT]
                  << " speed=" << invalid_counts[CPBC_INVALID_SPEED]
                  << " cones="
                  << invalid_counts[CPBC_INVALID_CONE_SEPARATION]
                  << " spacing=" << invalid_counts[CPBC_INVALID_SPACING]
                  << " scalar_map="
                  << invalid_counts[CPBC_INVALID_SCALAR_MAP]
                  << " sample_bg={chi:"
                  << bg_h(0,I_Z4C_CHI,ks,js,is)
                  << ",gxx:" << bg_h(0,I_Z4C_GXX,ks,js,is)
                  << ",gxy:" << bg_h(0,I_Z4C_GXY,ks,js,is)
                  << ",gxz:" << bg_h(0,I_Z4C_GXZ,ks,js,is)
                  << ",gyy:" << bg_h(0,I_Z4C_GYY,ks,js,is)
                  << ",gyz:" << bg_h(0,I_Z4C_GYZ,ks,js,is)
                  << ",gzz:" << bg_h(0,I_Z4C_GZZ,ks,js,is)
                  << ",alpha:" << bg_h(0,I_Z4C_ALPHA,ks,js,is)
                  << "}"
                  << std::endl;
      }
      std::exit(EXIT_FAILURE);
    }

    auto diag_h =
        Kokkos::create_mirror_view_and_copy(HostMemSpace(),characteristic_bc_diag);
    Real values[CPBC_NDIAG];
    for (int n = 0; n < CPBC_NDIAG; ++n) values[n] = diag_h(n);
    int max_boundary_block_count[3] = {
        characteristic_bc_boundary_block_count[0],
        characteristic_bc_boundary_block_count[1],
        characteristic_bc_boundary_block_count[2]};
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE,values,CPBC_NDIAG,MPI_ATHENA_REAL,MPI_MAX,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&max_diagnostic_kernel_seconds,1,
                  MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,max_boundary_block_count,3,MPI_INT,MPI_MAX,
                  MPI_COMM_WORLD);
#endif
    if (global_variable::my_rank == 0) {
      std::cout << "Z4C_CHARACTERISTIC_CPBC time=" << pm->time
                << " cycle=" << pm->ncycle
                << " source="
                << (opt.characteristic_bc_source_mode ==
                            characteristic_bc_source_tangential_principal ?
                        "tangential_principal" : "zero_rate")
                << " incoming_modes=10"
                << " boundary_blocks_max="
                << max_boundary_block_count[0] << ","
                << max_boundary_block_count[1] << ","
                << max_boundary_block_count[2]
                << " gauge=" << values[CPBC_GAUGE_AMPLITUDE]
                << " constraint=" << values[CPBC_CONSTRAINT_AMPLITUDE]
                << " radiation=" << values[CPBC_RADIATION_AMPLITUDE]
                << " outgoing_gauge="
                << values[CPBC_OUTGOING_GAUGE_AMPLITUDE]
                << " outgoing_constraint="
                << values[CPBC_OUTGOING_CONSTRAINT_AMPLITUDE]
                << " outgoing_radiation="
                << values[CPBC_OUTGOING_RADIATION_AMPLITUDE]
                << " enforcement=" << values[CPBC_ENFORCEMENT_ERROR]
                << " correction=" << values[CPBC_MAX_CORRECTION]
                << " shift_ratio=" << values[CPBC_MAX_SHIFT_RATIO]
                << " diagnostic_kernel_seconds="
                << max_diagnostic_kernel_seconds
                << " performance_valid="
                << (characteristic_bc_performance_valid ? 1 : 0)
                << " kernel_seconds=" << characteristic_bc_kernel_seconds
                << " volume_rhs_seconds="
                << characteristic_bc_volume_rhs_seconds
                << " kernel_fraction="
                << characteristic_bc_max_rank_fraction
                << std::endl;
    }
  }
  characteristic_bc_announced = true;
  return TaskStatus::complete;
}

}  // namespace z4c
