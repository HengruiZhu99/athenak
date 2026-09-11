#ifndef Z4C_BOUNDARY_RHS_HPP_
#define Z4C_BOUNDARY_RHS_HPP_
#include <limits>
#include <stdexcept>
#include "z4c/z4c.hpp"
#include "z4c/cartoon_derivatives.hpp"
#include "z4c/full_constraint_bjorhus.hpp"
#include "z4c/sommerfeld_derivatives.hpp"
#include "coordinates/cell_locations.hpp"
namespace z4c {
KOKKOS_INLINE_FUNCTION
static bool IsSommerfeldBoundaryFlag(const BoundaryFlag flag,
                                     const bool user_sbc) {
  return flag == BoundaryFlag::outflow || flag == BoundaryFlag::diode ||
         flag == BoundaryFlag::vacuum ||
         (flag == BoundaryFlag::user && user_sbc);
}

KOKKOS_INLINE_FUNCTION
static int SommerfeldBoundarySide(const DvceArray2D<BoundaryFlag> &bcs,
                                  const bool user_sbc, const int m,
                                  const int direction, const int index,
                                  const int lower, const int upper) {
  const auto inner = static_cast<BoundaryFace>(2 * direction);
  const auto outer = static_cast<BoundaryFace>(2 * direction + 1);
  if (index == lower && IsSommerfeldBoundaryFlag(bcs(m, inner), user_sbc)) {
    return -1;
  }
  if (index == upper && IsSommerfeldBoundaryFlag(bcs(m, outer), user_sbc)) {
    return 1;
  }
  return 0;
}

template <int NGHOST, typename ScalarField>
KOKKOS_INLINE_FUNCTION
static Real OneSidedScalarFirst(const int direction, const int side,
                                const Real inverse_spacing[3],
                                ScalarField &field, const int m, const int k,
                                const int j, const int i) {
  const int sk = direction == 2;
  const int sj = direction == 1;
  const int si = direction == 0;
  const int inward = -side;
  const auto value = [&](const int q) {
    return field(m, k + q * inward * sk, j + q * inward * sj,
                 i + q * inward * si);
  };
  return SommerfeldOneSidedFirst<NGHOST>(
      side, inverse_spacing[direction], value);
}

template <int NGHOST, typename VectorField>
KOKKOS_INLINE_FUNCTION
static Real OneSidedVectorFirst(const int direction, const int side,
                                const Real inverse_spacing[3],
                                VectorField &field, const int component,
                                const int m, const int k, const int j,
                                const int i) {
  const int sk = direction == 2;
  const int sj = direction == 1;
  const int si = direction == 0;
  const int inward = -side;
  const auto value = [&](const int q) {
    return field(m, component, k + q * inward * sk, j + q * inward * sj,
                 i + q * inward * si);
  };
  return SommerfeldOneSidedFirst<NGHOST>(
      side, inverse_spacing[direction], value);
}

template <int NGHOST, typename TensorField>
KOKKOS_INLINE_FUNCTION
static Real OneSidedTensorFirst(const int direction, const int side,
                                const Real inverse_spacing[3],
                                TensorField &field, const int component_a,
                                const int component_b, const int m, const int k,
                                const int j, const int i) {
  const int sk = direction == 2;
  const int sj = direction == 1;
  const int si = direction == 0;
  const int inward = -side;
  const auto value = [&](const int q) {
    return field(m, component_a, component_b, k + q * inward * sk,
                 j + q * inward * sj, i + q * inward * si);
  };
  return SommerfeldOneSidedFirst<NGHOST>(
      side, inverse_spacing[direction], value);
}

//! Differentiate a just-computed RHS without reading a stale RHS ghost.
//!
//! CalcRHS writes active points only and BoundaryRHS runs before the RK update and
//! subsequent communication.  Consequently a centered tangential stencil at a local
//! MeshBlock edge must not consume the preceding stage's RHS ghost.  Use the same
//! robust O2 closure toward the local active interior at those edges.  The collapsed
//! Cartoon direction remains a tensor-aware analytic derivative supplied by `provider`.
template <typename DerivativeProvider, typename ScalarField>
KOKKOS_INLINE_FUNCTION
static Real StageCurrentScalarFirst(
    const int direction, const int physical_side, const Real inverse_spacing[3],
    ScalarField &field, const DerivativeProvider &provider,
    const Z4cGridLayout &layout, const int m, const int k, const int j,
    const int i) {
  if (layout.nx3 == 1 && direction == 2) {
    return provider.ScalarFirst(direction, field);
  }
  const int index[3] = {i, j, k};
  const int lower[3] = {layout.is, layout.js, layout.ks};
  const int upper[3] = {layout.ie, layout.je, layout.ke};
  int closure_side = physical_side;
  if (closure_side == 0 && index[direction] == lower[direction]) closure_side = -1;
  if (closure_side == 0 && index[direction] == upper[direction]) closure_side = 1;
  if (closure_side != 0) {
    return OneSidedScalarFirst<2>(direction, closure_side, inverse_spacing,
                                  field, m, k, j, i);
  }
  const int sk = direction == 2;
  const int sj = direction == 1;
  const int si = direction == 0;
  const auto value = [&](const int q) {
    return field(m, k + q * sk, j + q * sj, i + q * si);
  };
  return BoundaryCenteredFirst(inverse_spacing[direction], value);
}

template <typename DerivativeProvider, typename TensorField>
KOKKOS_INLINE_FUNCTION
static Real StageCurrentTensorFirst(
    const int direction, const int physical_side, const Real inverse_spacing[3],
    TensorField &field, const int component_a, const int component_b,
    const DerivativeProvider &provider, const Z4cGridLayout &layout,
    const int m, const int k, const int j, const int i) {
  if (layout.nx3 == 1 && direction == 2) {
    return provider.template TensorFirst<TensorVariance::all_lower>(
        direction, component_a, component_b, field);
  }
  const int index[3] = {i, j, k};
  const int lower[3] = {layout.is, layout.js, layout.ks};
  const int upper[3] = {layout.ie, layout.je, layout.ke};
  int closure_side = physical_side;
  if (closure_side == 0 && index[direction] == lower[direction]) closure_side = -1;
  if (closure_side == 0 && index[direction] == upper[direction]) closure_side = 1;
  if (closure_side != 0) {
    return OneSidedTensorFirst<2>(direction, closure_side, inverse_spacing,
                                  field, component_a, component_b, m, k, j, i);
  }
  const int sk = direction == 2;
  const int sj = direction == 1;
  const int si = direction == 0;
  const auto value = [&](const int q) {
    return field(m, component_a, component_b, k + q * sk, j + q * sj,
                 i + q * si);
  };
  return BoundaryCenteredFirst(inverse_spacing[direction], value);
}

//----------------------------------------------------------------------------------------
//! \fn void Z4c::Z4cSommerfeld
//! \brief apply Sommerfeld BCs to the given set of points
template <typename Centering, typename Symmetry, int NGHOST>
KOKKOS_INLINE_FUNCTION
static void Z4cSommerfeld(const Z4c::Z4c_vars& z4c, const Z4c::Z4c_vars& rhs,
    const Z4cGridLayout &layout, const DualArray1D<RegionSize> &size,
    const DvceArray2D<BoundaryFlag> &bcs, const bool user_sbc,
    const int m, const int k, const int j, const int i) {
  // -------------------------------------------------------------------------------------
  // Scratch data
  //

  // First derivatives
  // Scalars
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dKhat_d;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dTheta_d;

  // Vectors
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> dGam_du;

  // Tensors
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> dA_ddd;


  // Psuedoradial vector
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> s_u;

  Real idx[] = {1./size.d_view(m).dx1, 1./size.d_view(m).dx2, 1./size.d_view(m).dx3};
  auto derivatives = MakeZ4cDerivativeProvider<Centering, Symmetry, NGHOST>(
      idx, size.d_view, layout.nx1, layout.is, m, k, j, i, layout.nx3 == 1);
  const int boundary_side[3] = {
      SommerfeldBoundarySide(bcs, user_sbc, m, 0, i, layout.is, layout.ie),
      SommerfeldBoundarySide(bcs, user_sbc, m, 1, j, layout.js, layout.je),
      SommerfeldBoundarySide(bcs, user_sbc, m, 2, k, layout.ks, layout.ke)};

  // -------------------------------------------------------------------------------------
  // First derivatives
  // Match the configured bulk stencil.  The physical ghost extrapolation order is
  // an independent input contract and must provide the corresponding boundary halo.
  for (int a = 0; a < 3; a++) {
    dKhat_d(a) = (boundary_side[a] == 0 ||
                  std::is_same_v<Centering, CellCenteredZ4c> ||
                  std::is_same_v<Symmetry, Cartesian3D>)
        ? derivatives.ScalarFirst(a, z4c.vKhat)
        : OneSidedScalarFirst<NGHOST>(
              a, boundary_side[a], idx, z4c.vKhat, m, k, j, i);
    dTheta_d(a) = (boundary_side[a] == 0 ||
                   std::is_same_v<Centering, CellCenteredZ4c> ||
                   std::is_same_v<Symmetry, Cartesian3D>)
        ? derivatives.ScalarFirst(a, z4c.vTheta)
        : OneSidedScalarFirst<NGHOST>(
              a, boundary_side[a], idx, z4c.vTheta, m, k, j, i);
  }
  for (int a = 0; a < 3; a++) {
    for (int b = 0; b < 3; b++) {
      dGam_du(b,a) = (boundary_side[b] == 0 ||
                      std::is_same_v<Centering, CellCenteredZ4c> ||
                      std::is_same_v<Symmetry, Cartesian3D>)
          ? derivatives.VectorFirst(b, a, z4c.vGam_u)
          : OneSidedVectorFirst<NGHOST>(
                b, boundary_side[b], idx, z4c.vGam_u, a, m, k, j, i);
    }
  }
  for (int a = 0; a < 3; a++) {
    for (int b = a; b < 3; b++) {
      for (int c = 0; c < 3; c++) {
        dA_ddd(c, a, b) = (boundary_side[c] == 0 ||
                            std::is_same_v<Centering, CellCenteredZ4c> ||
                            std::is_same_v<Symmetry, Cartesian3D>)
            ? derivatives.template TensorFirst<TensorVariance::all_lower>(
                  c, a, b, z4c.vA_dd)
            : OneSidedTensorFirst<NGHOST>(
                  c, boundary_side[c], idx, z4c.vA_dd, a, b, m, k, j, i);
      }
    }
  }

  // -------------------------------------------------------------------------------------
  // Compute psuedo-radial vector
  //
  Real &x1min = size.d_view(m).x1min;
  Real &x1max = size.d_view(m).x1max;
  Real &x2min = size.d_view(m).x2min;
  Real &x2max = size.d_view(m).x2max;
  Real &x3min = size.d_view(m).x3min;
  Real &x3max = size.d_view(m).x3max;

  Real x1v = Z4cPointX<Centering>(i-layout.is, layout.nx1, x1min, x1max);
  Real x2v = Z4cPointX<Centering>(j-layout.js, layout.nx2, x2min, x2max);
  Real x3v = 0.0;
  if constexpr (std::is_same_v<Symmetry, Cartesian3D>) {
    x3v = Z4cPointX<Centering>(k-layout.ks, layout.nx3, x3min, x3max);
  }

  Real r = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
  s_u(0) = x1v/r;
  s_u(1) = x2v/r;
  s_u(2) = x3v/r;

  // -------------------------------------------------------------------------------------
  // Boundary RHS for scalars
  //
  rhs.vTheta(m,k,j,i) = - z4c.vTheta(m,k,j,i)/r;
  rhs.vKhat(m,k,j,i) = - sqrt(2.) * z4c.vKhat(m,k,j,i)/r;
  for (int a = 0; a < 3; a++) {
    rhs.vTheta(m,k,j,i) -= s_u(a) * dTheta_d(a);
    rhs.vKhat(m,k,j,i) -= sqrt(2.) * s_u(a) * dKhat_d(a);
  }

  // -------------------------------------------------------------------------------------
  // Boundary RHS for Gamma
  //
  for (int a = 0; a < 3; a++) {
    rhs.vGam_u(m,a,k,j,i) = - z4c.vGam_u(m, a, k, j, i)/r;
    for (int b = 0; b < 3; b++) {
      rhs.vGam_u(m,a,k,j,i) -= s_u(b) * dGam_du(b,a);
    }
  }

  // -------------------------------------------------------------------------------------
  // Boundary RHS for A_ab
  //
  for (int a = 0; a < 3; a++) {
    for (int b = a; b < 3; b++) {
      rhs.vA_dd(m,a,b,k,j,i) = - z4c.vA_dd(m,a,b,k,j,i)/r;
      for (int c = 0; c < 3; c++) {
        rhs.vA_dd(m,a,b,k,j,i) -= s_u(c) * dA_ddd(c,a,b);
      }
    }
  }
}

//! Apply the four incoming full-field constraint compatibility equations at one point.
//!
//! The complete volume RHS is already present.  Physical-normal derivatives always use
//! the robust O2 inward stencil, independently of the bulk spatial order.  Nonincident
//! coordinate contributions (needed when the metric normal is oblique) use centered O2
//! only where both active neighbors are stage-current, with an inward O2 closure at
//! local block edges.  This deliberately avoids reading a stale RHS ghost.
template <typename Centering, typename Symmetry, int NGHOST>
KOKKOS_INLINE_FUNCTION
static void ApplyFullConstraintBjorhusAtPoint(
    const Z4c::Z4c_vars &z4c, const Z4c::Z4c_vars &rhs,
    const Z4cGridLayout &layout, const DualArray1D<RegionSize> &size,
    const DvceArray2D<BoundaryFlag> &bcs, const bool user_sbc,
    const int direction, const int m, const int k, const int j, const int i) {
  const int side[3] = {
      SommerfeldBoundarySide(bcs, user_sbc, m, 0, i, layout.is, layout.ie),
      SommerfeldBoundarySide(bcs, user_sbc, m, 1, j, layout.js, layout.je),
      SommerfeldBoundarySide(bcs, user_sbc, m, 2, k, layout.ks, layout.ke)};
  const bool cartoon_axis_point =
      std::is_same_v<Symmetry, CartoonSO2> && i == layout.is &&
      bcs(m, BoundaryFace::inner_x1) == BoundaryFlag::axis;
  if (!FullConstraintBjorhusOwnsPoint(direction, side, cartoon_axis_point)) return;

  Real metric_dd[3][3];
  for (int a = 0; a < 3; ++a) {
    for (int b = a; b < 3; ++b) {
      metric_dd[a][b] = metric_dd[b][a] = z4c.g_dd(m, a, b, k, j, i);
    }
  }
  FullConstraintBjorhusFrame frame;
  if (MakeFullConstraintBjorhusFrame(metric_dd, side, &frame) !=
      FullConstraintBjorhusStatus::valid) {
    Kokkos::abort("full_constraint_bjorhus: invalid conformal boundary frame");
  }

  const Real chi = z4c.chi(m, k, j, i);
  const Real alpha = z4c.alpha(m, k, j, i);
  if (!(isfinite(chi) && isfinite(alpha)) || chi <= 0.0 || alpha <= 0.0) {
    Kokkos::abort("full_constraint_bjorhus: invalid lapse or conformal factor");
  }
  const Real sqrt_chi = sqrt(chi);
  Real beta_normal = 0.0;
  for (int a = 0; a < 3; ++a) {
    beta_normal += frame.normal_d[a] * z4c.beta_u(m, a, k, j, i);
  }
  const Real light_speed = alpha * sqrt_chi;
  const Real lambda_incoming = beta_normal + light_speed;
  const Real lambda_outgoing = beta_normal - light_speed;
  if (!(isfinite(lambda_incoming) && isfinite(lambda_outgoing)) ||
      lambda_incoming <= 0.0 || lambda_outgoing >= 0.0) {
    Kokkos::abort(
        "full_constraint_bjorhus: boundary does not have one incoming and one "
        "outgoing physical-speed member");
  }

  const Real inverse_spacing[3] = {
      1.0 / size.d_view(m).dx1, 1.0 / size.d_view(m).dx2,
      1.0 / size.d_view(m).dx3};
  auto derivatives = MakeZ4cDerivativeProvider<Centering, Symmetry, NGHOST>(
      inverse_spacing, size.d_view, layout.nx1, layout.is, m, k, j, i,
      layout.nx3 == 1);

  Real derivative_rhs_chi = 0.0;
  Real derivative_rhs_metric[3][3] = {};
  for (int normal_direction = 0; normal_direction < 3; ++normal_direction) {
    const Real normal_component = frame.normal_u[normal_direction];
    if (fabs(normal_component) <= 32.0 * std::numeric_limits<Real>::epsilon()) {
      continue;
    }
    const Real dchi = StageCurrentScalarFirst(
        normal_direction, side[normal_direction], inverse_spacing, rhs.chi,
        derivatives, layout, m, k, j, i);
    derivative_rhs_chi += normal_component * dchi;
    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        const Real derivative = StageCurrentTensorFirst(
            normal_direction, side[normal_direction], inverse_spacing,
            rhs.g_dd, a, b, derivatives, layout, m, k, j, i);
        derivative_rhs_metric[a][b] += normal_component * derivative;
        if (a != b) derivative_rhs_metric[b][a] = derivative_rhs_metric[a][b];
      }
    }
  }

  Real rhs_A[3][3];
  Real rhs_A_trace = 0.0;
  Real derivative_rhs_metric_trace = 0.0;
  Real gamma_rhs_normal = 0.0;
  for (int a = 0; a < 3; ++a) {
    gamma_rhs_normal += frame.normal_d[a] * rhs.vGam_u(m, a, k, j, i);
    for (int b = 0; b < 3; ++b) {
      rhs_A[a][b] = rhs.vA_dd(m, a, b, k, j, i);
      rhs_A_trace += frame.metric_uu[a][b] * rhs_A[a][b];
      derivative_rhs_metric_trace +=
          frame.metric_uu[a][b] * derivative_rhs_metric[a][b];
    }
  }
  Real rhs_A_ss = 0.0;
  Real derivative_rhs_metric_ss = 0.0;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      rhs_A_ss += frame.normal_u[a] * frame.normal_u[b] * rhs_A[a][b];
      derivative_rhs_metric_ss +=
          frame.normal_u[a] * frame.normal_u[b] * derivative_rhs_metric[a][b];
    }
  }
  rhs_A_ss -= rhs_A_trace / 3.0;
  derivative_rhs_metric_ss -= derivative_rhs_metric_trace / 3.0;

  FullConstraintBjorhusRates volume_rates;
  const Real theta_rhs = rhs.vTheta(m, k, j, i);
  volume_rates.theta = sqrt_chi * theta_rhs +
                       0.5 * chi * gamma_rhs_normal + derivative_rhs_chi;
  volume_rates.z_normal =
      (4.0 * rhs.vKhat(m, k, j, i) / 3.0 + 2.0 * theta_rhs / 3.0 -
       2.0 * rhs_A_ss) /
          sqrt_chi -
      gamma_rhs_normal + derivative_rhs_metric_ss;
  for (int a = 0; a < 3; ++a) {
    volume_rates.vector_covector[a] = 0.0;
    for (int b = 0; b < 3; ++b) {
      volume_rates.vector_covector[a] +=
          -2.0 * rhs_A[a][b] * frame.normal_u[b] / sqrt_chi -
          frame.metric_dd[a][b] * rhs.vGam_u(m, b, k, j, i) +
          derivative_rhs_metric[a][b] * frame.normal_u[b];
    }
  }

  FullConstraintBjorhusCorrection correction;
  if (SolveFullConstraintBjorhusCorrection(chi, frame, volume_rates,
                                           &correction) !=
      FullConstraintBjorhusStatus::valid) {
    Kokkos::abort("full_constraint_bjorhus: singular incoming constraint map");
  }

  // Assert the local algebra before mutating the production RHS.  The vector result is
  // tested only in the tangent plane; its normal covector component is not an independent
  // incoming row.
  const FullConstraintBjorhusRates corrected =
      ApplyFullConstraintBjorhusCorrectionToRates(chi, frame, volume_rates,
                                                   correction);
  Real corrected_vector_normal = 0.0;
  Real scale = fmax(1.0, fmax(fabs(volume_rates.theta),
                              fabs(volume_rates.z_normal)));
  for (int a = 0; a < 3; ++a) {
    corrected_vector_normal +=
        frame.normal_u[a] * corrected.vector_covector[a];
    scale = fmax(scale, fabs(volume_rates.vector_covector[a]));
  }
  Real tangent_error = 0.0;
  for (int a = 0; a < 3; ++a) {
    const Real component = corrected.vector_covector[a] -
                           frame.normal_d[a] * corrected_vector_normal;
    tangent_error = fmax(tangent_error, fabs(component));
  }
  const Real tolerance = 2048.0 * std::numeric_limits<Real>::epsilon() * scale;
  if (fabs(corrected.theta) > tolerance ||
      fabs(corrected.z_normal) > tolerance || tangent_error > tolerance) {
    Kokkos::printf(
        "FULL_CONSTRAINT_BJORHUS_RESIDUAL m=%d k=%d j=%d i=%d "
        "chi=%.17e theta=%.17e z_normal=%.17e tangent=%.17e "
        "tolerance=%.17e scale=%.17e rates=[%.17e,%.17e,%.17e,%.17e,%.17e] "
        "correction=[%.17e,%.17e,%.17e,%.17e] normal_u=[%.17e,%.17e,%.17e] "
        "normal_d=[%.17e,%.17e,%.17e]\n",
        m, k, j, i, static_cast<double>(chi),
        static_cast<double>(corrected.theta),
        static_cast<double>(corrected.z_normal),
        static_cast<double>(tangent_error), static_cast<double>(tolerance),
        static_cast<double>(scale), static_cast<double>(volume_rates.theta),
        static_cast<double>(volume_rates.z_normal),
        static_cast<double>(volume_rates.vector_covector[0]),
        static_cast<double>(volume_rates.vector_covector[1]),
        static_cast<double>(volume_rates.vector_covector[2]),
        static_cast<double>(correction.theta),
        static_cast<double>(correction.gamma_u[0]),
        static_cast<double>(correction.gamma_u[1]),
        static_cast<double>(correction.gamma_u[2]),
        static_cast<double>(frame.normal_u[0]),
        static_cast<double>(frame.normal_u[1]),
        static_cast<double>(frame.normal_u[2]),
        static_cast<double>(frame.normal_d[0]),
        static_cast<double>(frame.normal_d[1]),
        static_cast<double>(frame.normal_d[2]));
    Kokkos::abort("full_constraint_bjorhus: incoming compatibility residual");
  }

  rhs.vTheta(m, k, j, i) += correction.theta;
  for (int a = 0; a < 3; ++a) {
    rhs.vGam_u(m, a, k, j, i) += correction.gamma_u[a];
  }
}

template <typename Centering, typename Symmetry>
KOKKOS_INLINE_FUNCTION
static void ApplyFullConstraintBjorhusConfigured(
    const Z4c::Z4c_vars &z4c, const Z4c::Z4c_vars &rhs,
    const Z4cGridLayout &layout, const DualArray1D<RegionSize> &size,
    const DvceArray2D<BoundaryFlag> &bcs, const bool user_sbc,
    const int fd_stencil, const int direction, const int m, const int k,
    const int j, const int i) {
  switch (fd_stencil) {
    case 2:
      ApplyFullConstraintBjorhusAtPoint<Centering, Symmetry, 2>(
          z4c, rhs, layout, size, bcs, user_sbc, direction, m, k, j, i);
      break;
    case 3:
      ApplyFullConstraintBjorhusAtPoint<Centering, Symmetry, 3>(
          z4c, rhs, layout, size, bcs, user_sbc, direction, m, k, j, i);
      break;
    case 4:
      ApplyFullConstraintBjorhusAtPoint<Centering, Symmetry, 4>(
          z4c, rhs, layout, size, bcs, user_sbc, direction, m, k, j, i);
      break;
    default:
      Kokkos::abort("full_constraint_bjorhus: invalid derivative stencil");
  }
}

template <typename Centering, typename Symmetry>
KOKKOS_INLINE_FUNCTION
static void Z4cSommerfeldConfigured(
    const Z4c::Z4c_vars &z4c, const Z4c::Z4c_vars &rhs,
    const Z4cGridLayout &layout, const DualArray1D<RegionSize> &size,
    const DvceArray2D<BoundaryFlag> &bcs, const bool user_sbc,
    const int fd_stencil, const int m, const int k, const int j, const int i) {
  // Preserve the established cell-centered and Cartesian closures exactly.  The
  // matched configured stencil and one-sided physical-normal derivative are a
  // native-VC Cartoon repair and must not silently alter legacy fingerprints.
  if constexpr (std::is_same_v<Centering, CellCenteredZ4c> ||
                std::is_same_v<Symmetry, Cartesian3D>) {
    Z4cSommerfeld<Centering, Symmetry, 2>(
        z4c, rhs, layout, size, bcs, user_sbc, m, k, j, i);
    return;
  }
  switch (fd_stencil) {
    case 2:
      Z4cSommerfeld<Centering, Symmetry, 2>(
          z4c, rhs, layout, size, bcs, user_sbc, m, k, j, i);
      break;
    case 3:
      Z4cSommerfeld<Centering, Symmetry, 3>(
          z4c, rhs, layout, size, bcs, user_sbc, m, k, j, i);
      break;
    case 4:
      Z4cSommerfeld<Centering, Symmetry, 4>(
          z4c, rhs, layout, size, bcs, user_sbc, m, k, j, i);
      break;
    default:
      Kokkos::abort("invalid Z4c Sommerfeld derivative stencil");
  }
}


// Explicit-state boundary sweep for auxiliary VC Cartoon blocks. The supplied
// fields already contain the complete stage RHS, including dissipation.
inline void ApplyLocalCartoonBoundaryRHS(const Z4cGridLayout &layout,
    const DualArray1D<RegionSize> &size,const DualArray2D<BoundaryFlag> &mb_bcs,
    const subcycling::BlockBatches &blocks,const Z4c::Z4c_vars &state,
    const Z4c::Z4c_vars &rhs,Z4cBoundaryRHSMode mode,bool user_sbc,int fd_stencil) {
  if(layout.centering!=Z4cGridCentering::vertex || layout.nx3!=1 ||
      fd_stencil<2 || fd_stencil>4 ||
      (mode!=Z4cBoundaryRHSMode::sommerfeld && mode!=Z4cBoundaryRHSMode::full_constraint_bjorhus))
    throw std::invalid_argument("unsupported local Cartoon boundary configuration");
  const auto bcs=mb_bcs.d_view;
  for(int direction=0;direction<2;++direction) for(int side=0;side<2;++side) {
    const int is=direction==0 ? (side==0 ? layout.is : layout.ie) : layout.is;
    const int ie=direction==0 ? is : layout.ie;
    const int js=direction==1 ? (side==0 ? layout.js : layout.je) : layout.js;
    const int je=direction==1 ? js : layout.je;
    blocks.For4("local Cartoon physical RHS",layout.ks,layout.ke,js,je,is,ie,
        KOKKOS_LAMBDA(int m,int k,int j,int i) {
      if(mode==Z4cBoundaryRHSMode::full_constraint_bjorhus) {
        // The existing ownership rule applies each composite corner normal once.
        ApplyFullConstraintBjorhusConfigured<VertexCenteredZ4c,CartoonSO2>(
            state,rhs,layout,size,bcs,user_sbc,fd_stencil,direction,m,k,j,i);
      } else if(IsSommerfeldBoundaryFlag(bcs(m,2*direction+side),user_sbc)) {
        Z4cSommerfeldConfigured<VertexCenteredZ4c,CartoonSO2>(
            state,rhs,layout,size,bcs,user_sbc,fd_stencil,m,k,j,i);
      }
    });
  }
}
}  // namespace z4c
#endif  // Z4C_BOUNDARY_RHS_HPP_
