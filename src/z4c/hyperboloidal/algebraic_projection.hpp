// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_ALGEBRAIC_PROJECTION_HPP_
#define Z4C_HYPERBOLOIDAL_ALGEBRAIC_PROJECTION_HPP_

#include <Kokkos_Core.hpp>
#include <limits>

namespace z4c::hyperboloidal {

enum class AlgebraicProjectionStatus {
  success, nonfinite_input, nonsymmetric_input, nonpositive_metric,
  nonfinite_arithmetic, invalid_projected_metric
};

template<class T> struct AlgebraicTensorConstraints {
  bool valid = false;
  T determinant = std::numeric_limits<T>::quiet_NaN();
  T trace = std::numeric_limits<T>::quiet_NaN();
};

template<class T> struct AlgebraicProjectionResult {
  bool valid = false;
  AlgebraicProjectionStatus status = AlgebraicProjectionStatus::nonfinite_input;
  AlgebraicTensorConstraints<T> before{}, after{};
  T scale = std::numeric_limits<T>::quiet_NaN();
};

// Point values only: derivatives must be rebuilt from the projected field.
// Explicit symmetry and Sylvester checks reject invalid data without floors.
template<class T> KOKKOS_INLINE_FUNCTION
AlgebraicProjectionStatus MeasureAlgebraicTensors(const T g[3][3],
    const T a[3][3], AlgebraicTensorConstraints<T> &out) {
  out = {};
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
    if (!Kokkos::isfinite(g[i][j]) || !Kokkos::isfinite(a[i][j]))
      return AlgebraicProjectionStatus::nonfinite_input;
  }
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
    if (g[i][j] != g[j][i] || a[i][j] != a[j][i])
      return AlgebraicProjectionStatus::nonsymmetric_input;
  }
  const T minor = g[0][0]*g[1][1]-g[0][1]*g[0][1];
  const T c00 = g[1][1]*g[2][2]-g[1][2]*g[1][2];
  const T c01 = g[0][2]*g[1][2]-g[0][1]*g[2][2];
  const T c02 = g[0][1]*g[1][2]-g[0][2]*g[1][1];
  const T det = g[0][0]*c00+g[0][1]*c01+g[0][2]*c02;
  out.determinant = det;
  if (!Kokkos::isfinite(minor) || !Kokkos::isfinite(det))
    return AlgebraicProjectionStatus::nonfinite_arithmetic;
  if (!(g[0][0] > 0 && minor > 0 && det > 0))
    return AlgebraicProjectionStatus::nonpositive_metric;
  const T c11 = g[0][0]*g[2][2]-g[0][2]*g[0][2];
  const T c12 = g[0][1]*g[0][2]-g[0][0]*g[1][2];
  // Divide cofactors first, as Geometry() does, rather than overflow a
  // contraction numerator which could be finite after division by det.
  const T inverse[6] = {c00/det,c01/det,c02/det,c11/det,c12/det,minor/det};
  for (int i = 0; i < 6; ++i) if (!Kokkos::isfinite(inverse[i]))
    return AlgebraicProjectionStatus::nonfinite_arithmetic;
  out.trace = inverse[0]*a[0][0]+inverse[3]*a[1][1]+inverse[5]*a[2][2]
      +2*(inverse[1]*a[0][1]+inverse[2]*a[0][2]+inverse[4]*a[1][2]);
  if (!Kokkos::isfinite(out.trace))
    return AlgebraicProjectionStatus::nonfinite_arithmetic;
  out.valid = true;
  return AlgebraicProjectionStatus::success;
}

// Same algebraic repair as CartesianPatch::ProjectAlgebraic / Z4c::AlgConstr:
// g' = det(g)^(-1/3) g; A' = A - g tr_g(A)/3, using the ORIGINAL g in A'.
// This is a constraint repair, not a physical-geometry-preserving resplitting.
// Both tensors remain bitwise unchanged on any failure, including overflow in
// the candidate. The full symmetric tensors must be distinct storage.
template<class T> KOKKOS_INLINE_FUNCTION
AlgebraicProjectionResult<T> ProjectAlgebraicTensors(T g[3][3], T a[3][3]) {
  AlgebraicProjectionResult<T> result{};
  result.status = MeasureAlgebraicTensors(g,a,result.before);
  if (result.status != AlgebraicProjectionStatus::success) return result;
  result.scale = T(1)/Kokkos::cbrt(result.before.determinant);
  if (!(result.scale > 0) || !Kokkos::isfinite(result.scale)) {
    result.status = AlgebraicProjectionStatus::nonfinite_arithmetic;
    return result;
  }
  T projected_g[3][3], projected_a[3][3];
  for (int i = 0; i < 3; ++i) for (int j = i; j < 3; ++j) {
    projected_g[i][j] = projected_g[j][i] = result.scale*g[i][j];
    projected_a[i][j] = projected_a[j][i] = a[i][j]-g[i][j]*(result.before.trace/3);
  }
  const auto status = MeasureAlgebraicTensors(projected_g,projected_a,result.after);
  if (status != AlgebraicProjectionStatus::success) {
    result.status = status == AlgebraicProjectionStatus::nonpositive_metric
        ? AlgebraicProjectionStatus::invalid_projected_metric
        : AlgebraicProjectionStatus::nonfinite_arithmetic;
    return result;
  }
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
    g[i][j] = projected_g[i][j]; a[i][j] = projected_a[i][j];
  }
  result.valid = true;
  return result;
}

// Native HyperbolicComposite point layout: chi, g-I[xx,xy,xz,yy,yz,zz],
// A[xx,xy,xz,yy,yz,zz], then P+3, Theta, independent Lambda, and gauge fields.
// Only indices 1..12 are read/written; this does NOT admit the full state or
// alter chi, P, Theta, Lambda, or any gauge data. Values/optional compensation
// must each contain at least 13 entries and must not alias. Compensation is
// cleared at indices 1..12 only AFTER a successful repair. Scri callers must
// subsequently rebuild null values, geometric rates, dependent D and ghosts.
template<class T> KOKKOS_INLINE_FUNCTION
AlgebraicProjectionResult<T> ProjectAlgebraicNativeState(T *values,
                                                        T *compensation = nullptr) {
  constexpr int ti[6] = {0,0,0,1,1,2}, tj[6] = {0,1,2,1,2,2};
  T g[3][3], a[3][3];
  for (int t = 0; t < 6; ++t) {
    const int i = ti[t], j = tj[t];
    g[i][j] = g[j][i] = values[1+t]+T(i == j);
    a[i][j] = a[j][i] = values[7+t];
  }
  auto result = ProjectAlgebraicTensors(g,a);
  if (!result.valid) return result;
  T stored_g[6];
  for (int t = 0; t < 6; ++t) {
    const int i = ti[t], j = tj[t];
    stored_g[t] = g[i][j]-T(i == j);
    g[i][j] = g[j][i] = stored_g[t]+T(i == j);
  }
  // The deviation representation can lose a very small positive diagonal
  // when subtracting unity. Validate what the caller will actually recover.
  const auto status = MeasureAlgebraicTensors(g,a,result.after);
  if (status != AlgebraicProjectionStatus::success) {
    result.valid = false;
    result.status = status == AlgebraicProjectionStatus::nonpositive_metric
        ? AlgebraicProjectionStatus::invalid_projected_metric
        : AlgebraicProjectionStatus::nonfinite_arithmetic;
    return result;
  }
  for (int t = 0; t < 6; ++t) {
    const int i = ti[t], j = tj[t];
    values[1+t] = stored_g[t]; values[7+t] = a[i][j];
  }
  if (compensation) for (int f = 1; f <= 12; ++f) compensation[f] = 0;
  return result;
}

}  // namespace z4c::hyperboloidal
#endif  // Z4C_HYPERBOLOIDAL_ALGEBRAIC_PROJECTION_HPP_
