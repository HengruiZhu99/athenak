//========================================================================================
//! \file reference_analytic_radial_q.hpp
//! \brief Compact analytic coefficients for the isotropic q-controlled trumpet.
//========================================================================================
#ifndef REF_GH_REFERENCE_ANALYTIC_RADIAL_Q_HPP_
#define REF_GH_REFERENCE_ANALYTIC_RADIAL_Q_HPP_

#include "athena.hpp"
#include "ref_gh/reference_trumpet_schwarzschild.hpp"

namespace ref_gh {

// These are the only cellwise quantities that depend on the stationary trumpet
// interpolation table.  Derivatives are with respect to physical radius r.
enum AnalyticRadialQStaticComponent : int {
  kAnalyticAlpha = 0,
  kAnalyticAlphaR,
  kAnalyticAlphaRR,
  kAnalyticTrumpetL,
  kAnalyticTrumpetLR,
  kAnalyticTrumpetLRR,
  kAnalyticShiftB,
  kAnalyticShiftBR,
  kAnalyticShiftBRR,
  kAnalyticU,
  kAnalyticUR,
  kAnalyticURR,
  kAnalyticRadialQStaticSize
};

// The q-dependent spatial scale and precisely the derivatives needed by the
// frame two-jet and the mixed-time gauge-reference baseline.  No trumpet-table
// interpolation occurs while this view is built.
enum AnalyticRadialQStageComponent : int {
  kAnalyticL = 0,
  kAnalyticLT,
  kAnalyticLR,
  kAnalyticLTT,
  kAnalyticLTR,
  kAnalyticLRR,
  kAnalyticLTTR,
  kAnalyticLTRR,
  kAnalyticRadialQStageSize
};

static_assert(kAnalyticRadialQStaticSize == 12,
              "analytic radial-q static layout changed");
static_assert(kAnalyticRadialQStageSize == 8,
              "analytic radial-q stage layout changed");
static_assert(kAnalyticRadialQStaticSize <= 16,
              "analytic radial-q static view exceeds its contract");
static_assert(kAnalyticRadialQStageSize <= 16,
              "analytic radial-q stage view exceeds its contract");

// A radial scalar carries eight independent derivatives rather than a generic
// 33-Real Cartesian ReferenceJet.  Cartesian components are reconstructed only
// when requested.  The two mixed third derivatives are the closed subset used
// by d_t theta_A for the moving reference frame.
struct AnalyticRadialScalar {
  Real value;
  Real dt;
  Real dr;
  Real dtt;
  Real dtr;
  Real drr;
  Real dttr;
  Real dtrr;

  KOKKOS_INLINE_FUNCTION
  Real D(const Real displacement[3], const Real radius, const int p) const {
    if (p == 0) return dt;
    return dr*displacement[p - 1]/radius;
  }

  KOKKOS_INLINE_FUNCTION
  Real DD(const Real displacement[3], const Real radius, const int p,
          const int q) const {
    if (p == 0 && q == 0) return dtt;
    if (p == 0 || q == 0) {
      const int spatial = (p == 0 ? q : p) - 1;
      return dtr*displacement[spatial]/radius;
    }
    const int i = p - 1;
    const int j = q - 1;
    const Real ni = displacement[i]/radius;
    const Real nj = displacement[j]/radius;
    return drr*ni*nj
           + (dr/radius)*(((i == j) ? 1.0 : 0.0) - ni*nj);
  }

  // Return partial_t partial_{i+1} partial_q f.  This is exactly the mixed
  // third-derivative slice carried by the generic oracle ReferenceJet.
  KOKKOS_INLINE_FUNCTION
  Real DtDD(const Real displacement[3], const Real radius, const int i,
            const int q) const {
    const Real ni = displacement[i]/radius;
    if (q == 0) return dttr*ni;
    const int j = q - 1;
    const Real nj = displacement[j]/radius;
    return dtrr*ni*nj
           + (dtr/radius)*(((i == j) ? 1.0 : 0.0) - ni*nj);
  }
};

KOKKOS_INLINE_FUNCTION
AnalyticRadialScalar ConstantAnalyticRadialScalar(const Real value) {
  return {value, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
}

KOKKOS_INLINE_FUNCTION
AnalyticRadialScalar operator+(const AnalyticRadialScalar &left,
                               const AnalyticRadialScalar &right) {
  return {left.value + right.value,
          left.dt + right.dt,
          left.dr + right.dr,
          left.dtt + right.dtt,
          left.dtr + right.dtr,
          left.drr + right.drr,
          left.dttr + right.dttr,
          left.dtrr + right.dtrr};
}

KOKKOS_INLINE_FUNCTION
AnalyticRadialScalar operator-(const AnalyticRadialScalar &value) {
  return {-value.value, -value.dt, -value.dr, -value.dtt, -value.dtr,
          -value.drr, -value.dttr, -value.dtrr};
}

KOKKOS_INLINE_FUNCTION
AnalyticRadialScalar operator-(const AnalyticRadialScalar &left,
                               const AnalyticRadialScalar &right) {
  return left + (-right);
}

KOKKOS_INLINE_FUNCTION
AnalyticRadialScalar operator*(const AnalyticRadialScalar &left,
                               const AnalyticRadialScalar &right) {
  return {
      left.value*right.value,
      left.dt*right.value + left.value*right.dt,
      left.dr*right.value + left.value*right.dr,
      left.dtt*right.value + 2.0*left.dt*right.dt
          + left.value*right.dtt,
      left.dtr*right.value + left.dt*right.dr + left.dr*right.dt
          + left.value*right.dtr,
      left.drr*right.value + 2.0*left.dr*right.dr
          + left.value*right.drr,
      left.dttr*right.value + left.dtt*right.dr
          + 2.0*(left.dtr*right.dt + left.dt*right.dtr)
          + left.dr*right.dtt + left.value*right.dttr,
      left.dtrr*right.value + left.drr*right.dt
          + 2.0*(left.dtr*right.dr + left.dr*right.dtr)
          + left.dt*right.drr + left.value*right.dtrr};
}

KOKKOS_INLINE_FUNCTION
AnalyticRadialScalar AnalyticRadialReciprocal(
    const AnalyticRadialScalar &input) {
  const Real inverse = 1.0/input.value;
  const Real first = -inverse*inverse;
  const Real second = 2.0*inverse*inverse*inverse;
  const Real third = -6.0*inverse*inverse*inverse*inverse;
  return {
      inverse,
      first*input.dt,
      first*input.dr,
      first*input.dtt + second*input.dt*input.dt,
      first*input.dtr + second*input.dt*input.dr,
      first*input.drr + second*input.dr*input.dr,
      first*input.dttr
          + second*(input.dr*input.dtt + 2.0*input.dt*input.dtr)
          + third*input.dt*input.dt*input.dr,
      first*input.dtrr
          + second*(input.dt*input.drr + 2.0*input.dr*input.dtr)
          + third*input.dt*input.dr*input.dr};
}

KOKKOS_INLINE_FUNCTION
AnalyticRadialScalar operator/(const AnalyticRadialScalar &left,
                               const AnalyticRadialScalar &right) {
  return left*AnalyticRadialReciprocal(right);
}

KOKKOS_INLINE_FUNCTION
void EvaluateAnalyticRadialQStatic(
    const DvceArray2D<Real> &table, const Real mass,
    const Real gaussian_width, const Real x, const Real y, const Real z,
    const Real center_x, const Real center_y, const Real center_z,
    Real coefficients[kAnalyticRadialQStaticSize]) {
  const Real dx = x - center_x;
  const Real dy = y - center_y;
  const Real dz = z - center_z;
  const Real radius = Kokkos::sqrt(dx*dx + dy*dy + dz*dz);
  const Real rho = radius/mass;
  const Real inverse_mass = 1.0/mass;

  const RadialProfile alpha =
      InterpolateTrumpetProfile(table, kCoeffAlpha, rho);
  coefficients[kAnalyticAlpha] = alpha.value;
  coefficients[kAnalyticAlphaR] = alpha.d1*inverse_mass;
  coefficients[kAnalyticAlphaRR] = alpha.d2*inverse_mass*inverse_mass;

  const RadialProfile trumpet_l = ArealRadiusToPsi2(
      InterpolateTrumpetProfile(table, kCoeffArealRadius, rho), rho);
  coefficients[kAnalyticTrumpetL] = trumpet_l.value;
  coefficients[kAnalyticTrumpetLR] = trumpet_l.d1*inverse_mass;
  coefficients[kAnalyticTrumpetLRR] =
      trumpet_l.d2*inverse_mass*inverse_mass;

  const RadialProfile shift =
      InterpolateTrumpetProfile(table, kCoeffShiftQ, rho);
  coefficients[kAnalyticShiftB] = shift.value*inverse_mass;
  coefficients[kAnalyticShiftBR] = shift.d1*inverse_mass*inverse_mass;
  coefficients[kAnalyticShiftBRR] =
      shift.d2*inverse_mass*inverse_mass*inverse_mass;

  const Real radial_scale = gaussian_width*mass;
  const Real inverse_scale2 = 1.0/(radial_scale*radial_scale);
  const Real log_rho = Kokkos::log(rho);
  const Real window = Kokkos::exp(-radius*radius*inverse_scale2);
  coefficients[kAnalyticU] = window*log_rho;
  coefficients[kAnalyticUR] = window*(1.0/radius
      - 2.0*radius*inverse_scale2*log_rho);
  coefficients[kAnalyticURR] = window*(
      -1.0/(radius*radius) - 4.0*inverse_scale2
      - 2.0*inverse_scale2*log_rho
      + 4.0*radius*radius*inverse_scale2*inverse_scale2*log_rho);
}

KOKKOS_INLINE_FUNCTION
void EvaluateAnalyticRadialQStage(
    const Real coefficients[kAnalyticRadialQStaticSize], const Real q,
    const Real q_dot, const Real q_ddot,
    Real stage[kAnalyticRadialQStageSize]) {
  const Real trumpet_l = coefficients[kAnalyticTrumpetL];
  const Real trumpet_lr = coefficients[kAnalyticTrumpetLR];
  const Real trumpet_lrr = coefficients[kAnalyticTrumpetLRR];
  const Real u = coefficients[kAnalyticU];
  const Real ur = coefficients[kAnalyticUR];
  const Real urr = coefficients[kAnalyticURR];
  const Real delta_q = q - 1.0;

  const Real lambda_t = -q_dot*u;
  const Real lambda_r = trumpet_lr/trumpet_l - delta_q*ur;
  const Real lambda_tt = -q_ddot*u;
  const Real lambda_tr = -q_dot*ur;
  const Real lambda_rr = trumpet_lrr/trumpet_l
      - (trumpet_lr/trumpet_l)*(trumpet_lr/trumpet_l) - delta_q*urr;
  const Real lambda_ttr = -q_ddot*ur;
  const Real lambda_trr = -q_dot*urr;
  const Real l = trumpet_l*Kokkos::exp(-delta_q*u);

  stage[kAnalyticL] = l;
  stage[kAnalyticLT] = l*lambda_t;
  stage[kAnalyticLR] = l*lambda_r;
  stage[kAnalyticLTT] = l*(lambda_tt + lambda_t*lambda_t);
  stage[kAnalyticLTR] = l*(lambda_tr + lambda_t*lambda_r);
  stage[kAnalyticLRR] = l*(lambda_rr + lambda_r*lambda_r);
  stage[kAnalyticLTTR] = l*(lambda_ttr + 2.0*lambda_t*lambda_tr
      + lambda_r*(lambda_tt + lambda_t*lambda_t));
  stage[kAnalyticLTRR] = l*(lambda_trr + 2.0*lambda_r*lambda_tr
      + lambda_t*(lambda_rr + lambda_r*lambda_r));
}

KOKKOS_INLINE_FUNCTION
AnalyticRadialScalar LoadAnalyticAlpha(
    const DvceArray5D<Real> &reference_static, const int m, const int k,
    const int j, const int i) {
  return {reference_static(m, kAnalyticAlpha, k, j, i), 0.0,
          reference_static(m, kAnalyticAlphaR, k, j, i), 0.0, 0.0,
          reference_static(m, kAnalyticAlphaRR, k, j, i), 0.0, 0.0};
}

KOKKOS_INLINE_FUNCTION
AnalyticRadialScalar LoadAnalyticL(
    const DvceArray5D<Real> &reference_stage, const int m, const int k,
    const int j, const int i) {
  return {reference_stage(m, kAnalyticL, k, j, i),
          reference_stage(m, kAnalyticLT, k, j, i),
          reference_stage(m, kAnalyticLR, k, j, i),
          reference_stage(m, kAnalyticLTT, k, j, i),
          reference_stage(m, kAnalyticLTR, k, j, i),
          reference_stage(m, kAnalyticLRR, k, j, i),
          reference_stage(m, kAnalyticLTTR, k, j, i),
          reference_stage(m, kAnalyticLTRR, k, j, i)};
}

KOKKOS_INLINE_FUNCTION
AnalyticRadialScalar LoadAnalyticShiftB(
    const DvceArray5D<Real> &reference_static, const int m, const int k,
    const int j, const int i) {
  return {reference_static(m, kAnalyticShiftB, k, j, i), 0.0,
          reference_static(m, kAnalyticShiftBR, k, j, i), 0.0, 0.0,
          reference_static(m, kAnalyticShiftBRR, k, j, i), 0.0, 0.0};
}

struct AnalyticRadialQPoint {
  AnalyticRadialScalar alpha;
  AnalyticRadialScalar l;
  AnalyticRadialScalar b;
  Real displacement[3];  // NOLINT(runtime/arrays)
  Real radius;
};

// Oracle adapter only: materialize the generic 33-Real jet at one point so the
// compact coefficient algebra and the independent generic geometry builder can
// be tested as two separate stages.  Production must never call this adapter.
KOKKOS_INLINE_FUNCTION
ReferenceJet AnalyticRadialScalarOracleJet(
    const AnalyticRadialQPoint &point,
    const AnalyticRadialScalar &radial) {
  ReferenceJet jet;
  jet.value = radial.value;
  for (int p = 0; p < 4; ++p) {
    jet.d[p] = radial.D(point.displacement, point.radius, p);
    for (int q = 0; q < 4; ++q) {
      jet.dd[p][q] = radial.DD(point.displacement, point.radius, p, q);
    }
  }
  for (int i = 0; i < 3; ++i) {
    for (int q = 0; q < 4; ++q) {
      jet.dt_dd[i][q] =
          radial.DtDD(point.displacement, point.radius, i, q);
    }
  }
  return jet;
}

KOKKOS_INLINE_FUNCTION
AnalyticRadialQPoint MakeAnalyticRadialQPoint(
    const DvceArray5D<Real> &reference_static,
    const DvceArray5D<Real> &reference_stage, const int m, const int k,
    const int j, const int i, const Real x, const Real y, const Real z,
    const Real center_x, const Real center_y, const Real center_z) {
  AnalyticRadialQPoint point;
  point.alpha = LoadAnalyticAlpha(reference_static, m, k, j, i);
  point.l = LoadAnalyticL(reference_stage, m, k, j, i);
  point.b = LoadAnalyticShiftB(reference_static, m, k, j, i);
  point.displacement[0] = x - center_x;
  point.displacement[1] = y - center_y;
  point.displacement[2] = z - center_z;
  point.radius = Kokkos::sqrt(
      point.displacement[0]*point.displacement[0]
      + point.displacement[1]*point.displacement[1]
      + point.displacement[2]*point.displacement[2]);
  return point;
}

KOKKOS_INLINE_FUNCTION
Real AnalyticCoordinateValue(const AnalyticRadialQPoint &point,
                             const AnalyticRadialScalar &radial,
                             const int coordinate) {
  return radial.value*point.displacement[coordinate];
}

KOKKOS_INLINE_FUNCTION
Real AnalyticCoordinateD(const AnalyticRadialQPoint &point,
                         const AnalyticRadialScalar &radial,
                         const int coordinate, const int p) {
  return radial.D(point.displacement, point.radius, p)
             *point.displacement[coordinate]
         + ((p == coordinate + 1) ? radial.value : 0.0);
}

KOKKOS_INLINE_FUNCTION
Real AnalyticCoordinateDD(const AnalyticRadialQPoint &point,
                          const AnalyticRadialScalar &radial,
                          const int coordinate, const int p, const int q) {
  return radial.DD(point.displacement, point.radius, p, q)
             *point.displacement[coordinate]
         + ((q == coordinate + 1)
                ? radial.D(point.displacement, point.radius, p) : 0.0)
         + ((p == coordinate + 1)
                ? radial.D(point.displacement, point.radius, q) : 0.0);
}

KOKKOS_INLINE_FUNCTION
Real AnalyticCoordinateProductValue(const AnalyticRadialQPoint &point,
                                    const AnalyticRadialScalar &radial,
                                    const int first, const int second) {
  return radial.value*point.displacement[first]*point.displacement[second];
}

KOKKOS_INLINE_FUNCTION
Real AnalyticCoordinateProductD(const AnalyticRadialQPoint &point,
                                const AnalyticRadialScalar &radial,
                                const int first, const int second,
                                const int p) {
  const Real x_first = point.displacement[first];
  const Real x_second = point.displacement[second];
  return radial.D(point.displacement, point.radius, p)*x_first*x_second
      + ((p == first + 1) ? radial.value*x_second : 0.0)
      + ((p == second + 1) ? radial.value*x_first : 0.0);
}

KOKKOS_INLINE_FUNCTION
Real AnalyticCoordinateProductDD(const AnalyticRadialQPoint &point,
                                 const AnalyticRadialScalar &radial,
                                 const int first, const int second,
                                 const int p, const int q) {
  const Real x_first = point.displacement[first];
  const Real x_second = point.displacement[second];
  const Real d_p = radial.D(point.displacement, point.radius, p);
  const Real d_q = radial.D(point.displacement, point.radius, q);
  Real value = radial.DD(point.displacement, point.radius, p, q)
               *x_first*x_second;
  if (p == first + 1) value += d_q*x_second;
  if (p == second + 1) value += d_q*x_first;
  if (q == first + 1) value += d_p*x_second;
  if (q == second + 1) value += d_p*x_first;
  if ((p == first + 1 && q == second + 1)
      || (p == second + 1 && q == first + 1)) value += radial.value;
  return value;
}

KOKKOS_INLINE_FUNCTION
Real AnalyticCoordinateDtDD(const AnalyticRadialQPoint &point,
                            const AnalyticRadialScalar &radial,
                            const int spatial, const int coordinate,
                            const int q) {
  const int p = spatial + 1;
  Real value = radial.DtDD(
      point.displacement, point.radius, spatial, q)
      *point.displacement[coordinate];
  if (q == coordinate + 1) {
    value += radial.DD(point.displacement, point.radius, 0, p);
  }
  if (p == coordinate + 1) {
    value += radial.DD(point.displacement, point.radius, 0, q);
  }
  return value;
}

KOKKOS_INLINE_FUNCTION
Real AnalyticCoordinateProductDtDD(
    const AnalyticRadialQPoint &point,
    const AnalyticRadialScalar &radial, const int spatial,
    const int first, const int second, const int q) {
  const int p = spatial + 1;
  const Real x_first = point.displacement[first];
  const Real x_second = point.displacement[second];
  const Real dt_p = radial.DD(point.displacement, point.radius, 0, p);
  const Real dt_q = radial.DD(point.displacement, point.radius, 0, q);
  Real value = radial.DtDD(
      point.displacement, point.radius, spatial, q)*x_first*x_second;
  if (p == first + 1) value += dt_q*x_second;
  if (p == second + 1) value += dt_q*x_first;
  if (q == first + 1) value += dt_p*x_second;
  if (q == second + 1) value += dt_p*x_first;
  if ((p == first + 1 && q == second + 1)
      || (p == second + 1 && q == first + 1)) value += radial.dt;
  return value;
}

KOKKOS_INLINE_FUNCTION
Real AnalyticMetric(const AnalyticRadialQPoint &point,
                    const int a, const int b) {
  if (a == 0 && b == 0) {
    const AnalyticRadialScalar lb = point.l*point.b;
    const AnalyticRadialScalar radial_radius2{
        point.radius*point.radius, 0.0, 2.0*point.radius, 0.0, 0.0,
        2.0, 0.0, 0.0};
    return (-(point.alpha*point.alpha) + lb*lb*radial_radius2).value;
  }
  if (a == 0 || b == 0) {
    const int spatial = (a == 0 ? b : a) - 1;
    return AnalyticCoordinateValue(point, point.l*point.l*point.b, spatial);
  }
  return (a == b) ? (point.l*point.l).value : 0.0;
}

KOKKOS_INLINE_FUNCTION
Real AnalyticDMetric(const AnalyticRadialQPoint &point, const int p,
                     const int a, const int b) {
  if (a == 0 && b == 0) {
    const AnalyticRadialScalar lb = point.l*point.b;
    const AnalyticRadialScalar radial_radius2{
        point.radius*point.radius, 0.0, 2.0*point.radius, 0.0, 0.0,
        2.0, 0.0, 0.0};
    return (-(point.alpha*point.alpha) + lb*lb*radial_radius2)
        .D(point.displacement, point.radius, p);
  }
  if (a == 0 || b == 0) {
    const int spatial = (a == 0 ? b : a) - 1;
    return AnalyticCoordinateD(
        point, point.l*point.l*point.b, spatial, p);
  }
  return (a == b) ? (point.l*point.l).D(
      point.displacement, point.radius, p) : 0.0;
}

KOKKOS_INLINE_FUNCTION
Real AnalyticDDMetric(const AnalyticRadialQPoint &point, const int p,
                      const int q, const int a, const int b) {
  if (a == 0 && b == 0) {
    const AnalyticRadialScalar lb = point.l*point.b;
    const AnalyticRadialScalar radial_radius2{
        point.radius*point.radius, 0.0, 2.0*point.radius, 0.0, 0.0,
        2.0, 0.0, 0.0};
    return (-(point.alpha*point.alpha) + lb*lb*radial_radius2)
        .DD(point.displacement, point.radius, p, q);
  }
  if (a == 0 || b == 0) {
    const int spatial = (a == 0 ? b : a) - 1;
    return AnalyticCoordinateDD(
        point, point.l*point.l*point.b, spatial, p, q);
  }
  return (a == b) ? (point.l*point.l).DD(
      point.displacement, point.radius, p, q) : 0.0;
}

KOKKOS_INLINE_FUNCTION
Real AnalyticDtDDMetric(const AnalyticRadialQPoint &point,
                        const int spatial, const int q,
                        const int a, const int b) {
  if (a == 0 && b == 0) {
    const AnalyticRadialScalar lb = point.l*point.b;
    const AnalyticRadialScalar radial_radius2{
        point.radius*point.radius, 0.0, 2.0*point.radius, 0.0, 0.0,
        2.0, 0.0, 0.0};
    return (-(point.alpha*point.alpha) + lb*lb*radial_radius2)
        .DtDD(point.displacement, point.radius, spatial, q);
  }
  if (a == 0 || b == 0) {
    const int coordinate = (a == 0 ? b : a) - 1;
    return AnalyticCoordinateDtDD(
        point, point.l*point.l*point.b, spatial, coordinate, q);
  }
  return (a == b) ? (point.l*point.l).DtDD(
      point.displacement, point.radius, spatial, q) : 0.0;
}

KOKKOS_INLINE_FUNCTION
Real AnalyticInverseMetric(const AnalyticRadialQPoint &point,
                           const int a, const int b) {
  const AnalyticRadialScalar inverse_alpha =
      AnalyticRadialReciprocal(point.alpha);
  const AnalyticRadialScalar inverse_l = AnalyticRadialReciprocal(point.l);
  if (a == 0 && b == 0) return -(inverse_alpha*inverse_alpha).value;
  if (a == 0 || b == 0) {
    const int spatial = (a == 0 ? b : a) - 1;
    return AnalyticCoordinateValue(
        point, point.b*inverse_alpha*inverse_alpha, spatial);
  }
  Real value = (a == b) ? (inverse_l*inverse_l).value : 0.0;
  value -= AnalyticCoordinateProductValue(
      point, point.b*point.b*inverse_alpha*inverse_alpha, a - 1, b - 1);
  return value;
}

KOKKOS_INLINE_FUNCTION
Real AnalyticDInverseMetric(const AnalyticRadialQPoint &point, const int p,
                            const int a, const int b) {
  const AnalyticRadialScalar inverse_alpha =
      AnalyticRadialReciprocal(point.alpha);
  const AnalyticRadialScalar inverse_l = AnalyticRadialReciprocal(point.l);
  if (a == 0 && b == 0) {
    return -(inverse_alpha*inverse_alpha)
        .D(point.displacement, point.radius, p);
  }
  if (a == 0 || b == 0) {
    const int spatial = (a == 0 ? b : a) - 1;
    return AnalyticCoordinateD(
        point, point.b*inverse_alpha*inverse_alpha, spatial, p);
  }
  Real value = (a == b) ? (inverse_l*inverse_l).D(
      point.displacement, point.radius, p) : 0.0;
  value -= AnalyticCoordinateProductD(
      point, point.b*point.b*inverse_alpha*inverse_alpha,
      a - 1, b - 1, p);
  return value;
}

KOKKOS_INLINE_FUNCTION
Real AnalyticDtDInverseMetric(const AnalyticRadialQPoint &point,
                              const int spatial,
                              const int a, const int b) {
  const AnalyticRadialScalar inverse_alpha =
      AnalyticRadialReciprocal(point.alpha);
  const AnalyticRadialScalar inverse_l = AnalyticRadialReciprocal(point.l);
  if (a == 0 && b == 0) {
    return -(inverse_alpha*inverse_alpha).DD(
        point.displacement, point.radius, 0, spatial + 1);
  }
  if (a == 0 || b == 0) {
    const int coordinate = (a == 0 ? b : a) - 1;
    return AnalyticCoordinateDtDD(
        point, point.b*inverse_alpha*inverse_alpha,
        spatial, coordinate, 0);
  }
  Real value = (a == b) ? (inverse_l*inverse_l).DD(
      point.displacement, point.radius, 0, spatial + 1) : 0.0;
  value -= AnalyticCoordinateProductDtDD(
      point, point.b*point.b*inverse_alpha*inverse_alpha,
      spatial, a - 1, b - 1, 0);
  return value;
}

KOKKOS_INLINE_FUNCTION
Real ReferenceCoframe(const AnalyticRadialQPoint &point,
                      const int A, const int a) {
  if (A == 0 && a == 0) return point.alpha.value;
  if (A > 0 && a == 0) {
    return AnalyticCoordinateValue(point, point.l*point.b, A - 1);
  }
  return (A > 0 && a == A) ? point.l.value : 0.0;
}

KOKKOS_INLINE_FUNCTION
Real AnalyticDCoframe(const AnalyticRadialQPoint &point, const int p,
                      const int A, const int a) {
  if (A == 0 && a == 0) {
    return point.alpha.D(point.displacement, point.radius, p);
  }
  if (A > 0 && a == 0) {
    return AnalyticCoordinateD(point, point.l*point.b, A - 1, p);
  }
  return (A > 0 && a == A)
      ? point.l.D(point.displacement, point.radius, p) : 0.0;
}

KOKKOS_INLINE_FUNCTION
Real ReferenceFrame(const AnalyticRadialQPoint &point,
                    const int A, const int a) {
  const AnalyticRadialScalar inverse_alpha =
      AnalyticRadialReciprocal(point.alpha);
  if (A == 0 && a == 0) return inverse_alpha.value;
  if (A == 0 && a > 0) {
    return -AnalyticCoordinateValue(point, point.b*inverse_alpha, a - 1);
  }
  return (A > 0 && a == A)
      ? AnalyticRadialReciprocal(point.l).value : 0.0;
}

KOKKOS_INLINE_FUNCTION
Real ReferenceDFrame(const AnalyticRadialQPoint &point, const int p,
                     const int A, const int a) {
  const AnalyticRadialScalar inverse_alpha =
      AnalyticRadialReciprocal(point.alpha);
  if (A == 0 && a == 0) {
    return inverse_alpha.D(point.displacement, point.radius, p);
  }
  if (A == 0 && a > 0) {
    return -AnalyticCoordinateD(point, point.b*inverse_alpha, a - 1, p);
  }
  return (A > 0 && a == A)
      ? AnalyticRadialReciprocal(point.l).D(
            point.displacement, point.radius, p) : 0.0;
}

KOKKOS_INLINE_FUNCTION
Real ReferenceDDFrame(const AnalyticRadialQPoint &point, const int p,
                      const int q, const int A, const int a) {
  const AnalyticRadialScalar inverse_alpha =
      AnalyticRadialReciprocal(point.alpha);
  if (A == 0 && a == 0) {
    return inverse_alpha.DD(point.displacement, point.radius, p, q);
  }
  if (A == 0 && a > 0) {
    return -AnalyticCoordinateDD(
        point, point.b*inverse_alpha, a - 1, p, q);
  }
  return (A > 0 && a == A)
      ? AnalyticRadialReciprocal(point.l).DD(
            point.displacement, point.radius, p, q) : 0.0;
}

KOKKOS_INLINE_FUNCTION
Real ReferenceChristoffel(const AnalyticRadialQPoint &point,
                          const int a, const int b, const int c) {
  Real value = 0.0;
  for (int ell = 0; ell < 4; ++ell) {
    value += 0.5*AnalyticInverseMetric(point, a, ell)*(
        AnalyticDMetric(point, b, ell, c)
        + AnalyticDMetric(point, c, ell, b)
        - AnalyticDMetric(point, ell, b, c));
  }
  return value;
}

KOKKOS_INLINE_FUNCTION
Real ReferenceDChristoffel(const AnalyticRadialQPoint &point, const int p,
                           const int a, const int b, const int c) {
  Real value = 0.0;
  for (int ell = 0; ell < 4; ++ell) {
    const Real first = 0.5*(
        AnalyticDMetric(point, b, ell, c)
        + AnalyticDMetric(point, c, ell, b)
        - AnalyticDMetric(point, ell, b, c));
    const Real d_first = 0.5*(
        AnalyticDDMetric(point, p, b, ell, c)
        + AnalyticDDMetric(point, p, c, ell, b)
        - AnalyticDDMetric(point, p, ell, b, c));
    value += AnalyticDInverseMetric(point, p, a, ell)*first
             + AnalyticInverseMetric(point, a, ell)*d_first;
  }
  return value;
}

KOKKOS_INLINE_FUNCTION
Real ReferenceSpatialFrame(const AnalyticRadialQPoint &point,
                           const int I, const int i) {
  return (I == i) ? AnalyticRadialReciprocal(point.l).value : 0.0;
}

KOKKOS_INLINE_FUNCTION
Real ReferenceSpatialCoframe(const AnalyticRadialQPoint &point,
                             const int I, const int i) {
  return (I == i) ? point.l.value : 0.0;
}

KOKKOS_INLINE_FUNCTION
Real ReferenceDtSpatialFrame(const AnalyticRadialQPoint &point,
                             const int I, const int i) {
  return (I == i) ? AnalyticRadialReciprocal(point.l).dt : 0.0;
}

KOKKOS_INLINE_FUNCTION
Real ReferenceStructure(const AnalyticRadialQPoint &point,
                        const int I, const int J, const int K) {
  const AnalyticRadialScalar inverse_l = AnalyticRadialReciprocal(point.l);
  return ((J == K) ? inverse_l.D(
              point.displacement, point.radius, I + 1) : 0.0)
         - ((I == K) ? inverse_l.D(
              point.displacement, point.radius, J + 1) : 0.0);
}

KOKKOS_INLINE_FUNCTION
Real AnalyticRawSpin(const AnalyticRadialQPoint &point,
                     const int A, const int B, const int C) {
  Real value = 0.0;
  for (int a = 0; a < 4; ++a) {
    for (int c = 0; c < 4; ++c) {
      Real derivative = ReferenceDFrame(point, c, B, a);
      for (int d = 0; d < 4; ++d) {
        derivative += ReferenceChristoffel(point, a, c, d)
                      *ReferenceFrame(point, B, d);
      }
      value += ReferenceCoframe(point, A, a)
               *ReferenceFrame(point, C, c)*derivative;
    }
  }
  return value;
}

KOKKOS_INLINE_FUNCTION
Real ReferenceSpin(const AnalyticRadialQPoint &point,
                   const int A, const int B, const int C) {
  if (A == B) return 0.0;
  const Real eta_A = (A == 0) ? -1.0 : 1.0;
  const Real eta_B = (B == 0) ? -1.0 : 1.0;
  return 0.5*(AnalyticRawSpin(point, A, B, C)
              - eta_A*eta_B*AnalyticRawSpin(point, B, A, C));
}

KOKKOS_INLINE_FUNCTION
Real AnalyticRawSpinCoordinateDerivative(
    const AnalyticRadialQPoint &point, const int p, const int A,
    const int B, const int C) {
  Real value = 0.0;
  for (int a = 0; a < 4; ++a) {
    const Real d_coframe = AnalyticDCoframe(point, p, A, a);
    for (int c = 0; c < 4; ++c) {
      Real derivative = ReferenceDFrame(point, c, B, a);
      Real d_derivative = ReferenceDDFrame(point, p, c, B, a);
      for (int d = 0; d < 4; ++d) {
        derivative += ReferenceChristoffel(point, a, c, d)
                      *ReferenceFrame(point, B, d);
        d_derivative += ReferenceDChristoffel(point, p, a, c, d)
                          *ReferenceFrame(point, B, d)
                        + ReferenceChristoffel(point, a, c, d)
                          *ReferenceDFrame(point, p, B, d);
      }
      value += (d_coframe*ReferenceFrame(point, C, c)
                + ReferenceCoframe(point, A, a)
                  *ReferenceDFrame(point, p, C, c))*derivative
               + ReferenceCoframe(point, A, a)
                 *ReferenceFrame(point, C, c)*d_derivative;
    }
  }
  return value;
}

KOKKOS_INLINE_FUNCTION
Real ReferenceSpinDerivative(const AnalyticRadialQPoint &point,
                             const int D, const int A, const int B,
                             const int C) {
  if (A == B) return 0.0;
  const Real eta_A = (A == 0) ? -1.0 : 1.0;
  const Real eta_B = (B == 0) ? -1.0 : 1.0;
  Real value = 0.0;
  for (int p = 0; p < 4; ++p) {
    value += 0.5*ReferenceFrame(point, D, p)*(
        AnalyticRawSpinCoordinateDerivative(point, p, A, B, C)
        - eta_A*eta_B
          *AnalyticRawSpinCoordinateDerivative(point, p, B, A, C));
  }
  return value;
}

KOKKOS_INLINE_FUNCTION
Real AnalyticStructure4(const AnalyticRadialQPoint &point,
                        const int E, const int C, const int D) {
  Real value = 0.0;
  for (int a = 0; a < 4; ++a) {
    for (int p = 0; p < 4; ++p) {
      value += ReferenceCoframe(point, E, a)*(
          ReferenceFrame(point, C, p)*ReferenceDFrame(point, p, D, a)
          - ReferenceFrame(point, D, p)*ReferenceDFrame(point, p, C, a));
    }
  }
  return value;
}

KOKKOS_INLINE_FUNCTION
Real ReferenceRiemann(const AnalyticRadialQPoint &point,
                      const int A, const int B, const int C, const int D) {
  Real value = ReferenceSpinDerivative(point, C, A, B, D)
               - ReferenceSpinDerivative(point, D, A, B, C);
  for (int E = 0; E < 4; ++E) {
    value += ReferenceSpin(point, A, E, C)*ReferenceSpin(point, E, B, D)
             - ReferenceSpin(point, A, E, D)*ReferenceSpin(point, E, B, C)
             - AnalyticStructure4(point, E, C, D)
               *ReferenceSpin(point, A, B, E);
  }
  return value;
}

KOKKOS_INLINE_FUNCTION
Real ReferenceRicci(const AnalyticRadialQPoint &point,
                    const int A, const int B) {
  Real value = 0.0;
  for (int C = 0; C < 4; ++C) {
    value += ReferenceRiemann(point, C, A, C, B);
  }
  return value;
}

}  // namespace ref_gh

#endif  // REF_GH_REFERENCE_ANALYTIC_RADIAL_Q_HPP_
