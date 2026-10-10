// Uncompiled CPU-private complete rational RWM gauge prototype.
// No production include, native binding, connection reconstruction or adoption.
#ifndef RESEARCH_EXACT_GAUGE_ROWS_HPP_
#define RESEARCH_EXACT_GAUGE_ROWS_HPP_
#include "../exact-rational-backend-WIP001/exact_dyadic_ratio.hpp"
#include <array>

namespace exact_gauge {
using exact_ratio::Dyadic;
using exact_ratio::Rounded;
using exact_ratio::Status;

struct Atom { std::uint64_t value = 0, direction = 0; };
using Vector = std::array<Atom, 3>;
using Matrix = std::array<Vector, 3>;
struct Fields {
  Atom alpha, chi, physical_P;
  Vector beta, alpha_d, chi_d, lambda;
  Matrix beta_d, metric;  // beta_d[j][i] = partial_j beta^i; all9 metric entries.
};
struct Input {
  Fields live, reference;  // reference physical_P binds actual p.k_physical.
  std::array<Matrix, 4> scaled_connection;
  Vector domega;
  Atom omega;
};  // Exactly106 value/direction pairs; all106 are validated, even for parts.

struct Dual {
  Dyadic value, direction;
  static Dual atom(const Atom& a) { return {Dyadic::atom(a.value), Dyadic::atom(a.direction)}; }
  static Dual integer(std::int64_t n) { return {Dyadic::integer(n), Dyadic()}; }
};
inline Dual operator+(const Dual& a, const Dual& b) {
  return {exact_ratio::add(a.value, b.value), exact_ratio::add(a.direction, b.direction)};
}
inline Dual operator-(const Dual& a, const Dual& b) {
  return {exact_ratio::subtract(a.value, b.value), exact_ratio::subtract(a.direction, b.direction)};
}
inline Dual operator-(const Dual& a) {
  return {exact_ratio::negate(a.value), exact_ratio::negate(a.direction)};
}
inline Dual operator*(const Dual& a, const Dual& b) {
  return {exact_ratio::multiply(a.value, b.value),
          exact_ratio::add(exact_ratio::multiply(a.direction, b.value),
                           exact_ratio::multiply(a.value, b.direction))};
}
using DVector = std::array<Dual, 3>;
using DMatrix = std::array<DVector, 3>;
struct DFields {
  Dual alpha, chi, physical_P;
  DVector beta, alpha_d, chi_d, lambda;
  DMatrix beta_d, metric;
};
inline DVector decode(const Vector& v) {
  DVector result;
  for (int i = 0; i < 3; ++i) result[i] = Dual::atom(v[i]);
  return result;
}
inline DMatrix decode(const Matrix& m) {
  DMatrix result;
  for (int i = 0; i < 3; ++i) result[i] = decode(m[i]);
  return result;
}
inline DFields decode(const Fields& f) {
  return {Dual::atom(f.alpha), Dual::atom(f.chi), Dual::atom(f.physical_P),
          decode(f.beta), decode(f.alpha_d), decode(f.chi_d), decode(f.lambda),
          decode(f.beta_d), decode(f.metric)};
}

struct Cofactors { DMatrix adjugate; Dual determinant; };
inline Cofactors cofactors(const DMatrix& m) {
  Cofactors result;
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
    int row[2], col[2], r = 0, c = 0;
    // adjugate_ij = cofactor_ji; no symmetric or trace-free assumption.
    for (int k = 0; k < 3; ++k) {
      if (k != j) row[r++] = k;
      if (k != i) col[c++] = k;
    }
    Dual minor = m[row[0]][col[0]] * m[row[1]][col[1]] -
                 m[row[0]][col[1]] * m[row[1]][col[0]];
    result.adjugate[i][j] = ((i + j) & 1) ? -minor : minor;
  }
  for (int j = 0; j < 3; ++j) result.determinant = result.determinant + m[0][j] * result.adjugate[j][0];
  return result;
}

struct RoundingAudit {
  Rounded rounded{Status::finite, 0};
  bool exact = false;
  int direction = 0;  // sign(rounded - exact); zero for exact or overflow.
  std::int64_t grid_exponent = -1074;  // exact-value grid, including normal carry.
};
inline RoundingAudit round(const Dyadic& numerator, const Dyadic& denominator) {
  RoundingAudit result;
  result.rounded = exact_ratio::round_ratio(numerator, denominator);
  if (result.rounded.status != Status::finite) return result;
  const Dyadic delta = exact_ratio::subtract(
      exact_ratio::multiply(Dyadic::atom(result.rounded.bits), denominator), numerator);
  result.exact = delta.sign == 0;
  result.direction = delta.sign * denominator.sign;
  if (numerator.sign) {
    const std::int64_t shift = numerator.exponent - denominator.exponent;
    std::int64_t exponent = static_cast<std::int64_t>(numerator.magnitude.bits()) -
                            static_cast<std::int64_t>(denominator.magnitude.bits()) + shift;
    if (exact_ratio::compare_scaled(numerator.magnitude, shift, denominator.magnitude, exponent) < 0) --exponent;
    result.grid_exponent = exponent >= -1022 ? exponent - 52 : -1074;
  }
  return result;
}
struct Row { RoundingAudit value, direction; };
inline Row round(const Dual& numerator, const Dual& denominator) {
  const Dyadic direction_numerator = exact_ratio::subtract(
      exact_ratio::multiply(numerator.direction, denominator.value),
      exact_ratio::multiply(numerator.value, denominator.direction));
  const Dyadic direction_denominator = exact_ratio::multiply(denominator.value, denominator.value);
  return {round(numerator.value, denominator.value), round(direction_numerator, direction_denominator)};
}

enum class EngineStatus { evaluated, invalid_atom, nonpositive_scalar, singular_metric, capacity_failure };
enum RowIndex { regular_alpha, regular_beta_x, regular_beta_y, regular_beta_z,
                pole_alpha, pole_beta_x, pole_beta_y, pole_beta_z,
                rhs_alpha, rhs_beta_x, rhs_beta_y, rhs_beta_z };
struct Result {
  EngineStatus status = EngineStatus::invalid_atom;
  std::array<Row, 12> rows;
  bool all_parts_finite = false, all_rhs_finite = false;
};

inline Result evaluate(const Input& input) {
  Result result;
  try {
    const DFields u = decode(input.live), h = decode(input.reference);
    std::array<DMatrix, 4> gamma;
    for (int a = 0; a < 4; ++a) gamma[a] = decode(input.scaled_connection[a]);
    const DVector omega_d = decode(input.domega);
    const Dual omega = Dual::atom(input.omega);
    // Complete validation precedes scalar/singularity/zero shortcuts.
    if (u.alpha.value.sign <= 0 || u.chi.value.sign <= 0 ||
        h.alpha.value.sign <= 0 || h.chi.value.sign <= 0 || omega.value.sign <= 0) {
      result.status = EngineStatus::nonpositive_scalar;
      return result;
    }
    const Cofactors live = cofactors(u.metric), ref = cofactors(h.metric);
    if (!live.determinant.value.sign || !ref.determinant.value.sign) {
      result.status = EngineStatus::singular_metric;
      return result;
    }
    const Dual Q = live.determinant * ref.determinant;
    const Dual denominator = h.alpha * Q;
    const Dual a2 = u.alpha * u.alpha, h2 = h.alpha * h.alpha;
    const Dual two = Dual::integer(2);
    const Dual half{Dyadic::atom(UINT64_C(0x3fe0000000000000)), Dyadic()};
    DMatrix live_Vn, ref_Vn, N, K, Kh, M;
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
      live_Vn[i][j] = a2 * u.chi * live.adjugate[i][j] * ref.determinant;
      ref_Vn[i][j] = h2 * h.chi * ref.adjugate[i][j] * live.determinant;
      N[i][j] = live_Vn[i][j] - ref_Vn[i][j];
      K[i][j] = live_Vn[i][j] - u.beta[i] * u.beta[j] * Q;
      Kh[i][j] = ref_Vn[i][j] - h.beta[i] * h.beta[j] * Q;
      M[i][j] = K[i][j] - Kh[i][j];
    }
    std::array<Dual, 8> numerator;
    Dual Ua, Ta = -a2 * u.physical_P + u.alpha * h.alpha * h.physical_P;
    Dual lapse_connection;
    for (int j = 0; j < 3; ++j) {
      Ua = Ua + h.alpha * u.beta[j] * u.alpha_d[j] - u.alpha * h.beta[j] * h.alpha_d[j];
      Ta = Ta - u.alpha * (u.beta[j] - h.beta[j]) * omega_d[j];
      for (int k = 0; k < 3; ++k) lapse_connection = lapse_connection + M[j][k] * gamma[0][j][k];
    }
    numerator[regular_alpha] = Q * Ua;
    numerator[pole_alpha] = h.alpha * (Q * Ta - u.alpha * lapse_connection);
    for (int i = 0; i < 3; ++i) {
      Dual ordinary = a2 * u.chi * u.lambda[i] - h2 * h.chi * h.lambda[i];
      Dual gradient, pole;
      for (int j = 0; j < 3; ++j) {
        ordinary = ordinary + u.beta[j] * u.beta_d[j][i] - h.beta[j] * h.beta_d[j][i];
        gradient = gradient + live.adjugate[i][j] * ref.determinant *
            (half * a2 * u.chi_d[j] - u.alpha * u.chi * u.alpha_d[j]) -
            ref.adjugate[i][j] * live.determinant *
            (half * h2 * h.chi_d[j] - h.alpha * h.chi * h.alpha_d[j]);
        pole = pole + two * N[i][j] * omega_d[j];
        for (int k = 0; k < 3; ++k) {
          pole = pole - M[j][k] * gamma[i+1][j][k] -
              (K[j][k] * u.beta[i] - Kh[j][k] * h.beta[i]) * gamma[0][j][k];
        }
      }
      numerator[regular_beta_x + i] = h.alpha * (Q * ordinary + gradient);
      numerator[pole_beta_x + i] = h.alpha * pole;
    }
    // Each part and each assembled output has an independent exact rational.
    // No rounded part, inverse entry, source or identity seed is reused.
    result.all_parts_finite = true;
    result.all_rhs_finite = true;
    for (int i = 0; i < 8; ++i) {
      result.rows[i] = round(numerator[i], denominator);
      const Row& row = result.rows[i];
      result.all_parts_finite = result.all_parts_finite &&
          row.value.rounded.status == Status::finite && row.direction.rounded.status == Status::finite;
    }
    const Dual assembled_denominator = omega * denominator;
    for (int i = 0; i < 4; ++i) {
      const Dual assembled_numerator = omega * numerator[i] + numerator[pole_alpha + i];
      result.rows[rhs_alpha + i] = round(assembled_numerator, assembled_denominator);
      const Row& row = result.rows[rhs_alpha + i];
      result.all_rhs_finite = result.all_rhs_finite &&
          row.value.rounded.status == Status::finite && row.direction.rounded.status == Status::finite;
    }
    result.status = EngineStatus::evaluated;
    return result;
  } catch (const std::length_error&) {
    result.status = EngineStatus::capacity_failure;
    result.all_parts_finite = result.all_rhs_finite = false;
    return result;
  } catch (const std::invalid_argument&) {
    result.status = EngineStatus::invalid_atom;
    result.all_parts_finite = result.all_rhs_finite = false;
    return result;
  }
}

}  // namespace exact_gauge
#endif
