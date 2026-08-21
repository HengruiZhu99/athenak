//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cartoon_vertex_axis.hpp
//! \brief Exact active-axis regularity projection for native VC Cartoon Z4c state.

#ifndef Z4C_CARTOON_VERTEX_AXIS_HPP_
#define Z4C_CARTOON_VERTEX_AXIS_HPP_

#include <Kokkos_Macros.hpp>
#include <math.h>

#include "z4c/cartoon_axis_parity.hpp"

namespace z4c {

struct VertexAxisCorrection {
  Real max_abs = 0.0;
  Real max_rel = 0.0;
  int component = -1;
  int nonfinite = 0;
};

//! Project one local packed Z4c vector into the exact SO(2) axis subspace.
//! This is used when a component-wise operator (notably meridional KO) is
//! assembled in registers before it is written to the evolved axis vertex.
template <int N>
KOKKOS_INLINE_FUNCTION void ProjectVertexAxisZ4cValues(Real (&values)[N]) {
  static_assert(N >= static_cast<int>(Z4cStateComponent::count));
  const int metric_rr = static_cast<int>(Z4cStateComponent::g_rhorho);
  const int metric_yy = static_cast<int>(Z4cStateComponent::g_yy);
  const int atilde_rr = static_cast<int>(Z4cStateComponent::a_rhorho);
  const int atilde_yy = static_cast<int>(Z4cStateComponent::a_yy);
  const Real metric_average = 0.5 * (values[metric_rr] + values[metric_yy]);
  const Real atilde_average = 0.5 * (values[atilde_rr] + values[atilde_yy]);
  values[metric_rr] = metric_average;
  values[metric_yy] = metric_average;
  values[atilde_rr] = atilde_average;
  values[atilde_yy] = atilde_average;
  constexpr Z4cStateComponent zero_components[] = {
      Z4cStateComponent::g_rhoz, Z4cStateComponent::g_rhoy,
      Z4cStateComponent::g_zy, Z4cStateComponent::a_rhoz,
      Z4cStateComponent::a_rhoy, Z4cStateComponent::a_zy,
      Z4cStateComponent::gamma_rho, Z4cStateComponent::gamma_y,
      Z4cStateComponent::beta_rho, Z4cStateComponent::beta_y,
      Z4cStateComponent::b_rho, Z4cStateComponent::b_y};
  for (const auto component : zero_components) {
    values[static_cast<int>(component)] = 0.0;
  }
}

template <typename Array5D>
KOKKOS_INLINE_FUNCTION void SetVertexAxisComponent(
    const Array5D &state, const int meshblock, const int k, const int j,
    const int axis_index, const int component, const Real replacement,
    VertexAxisCorrection *correction) {
  const Real original = state(meshblock, component, k, j, axis_index);
  const Real absolute = fabs(replacement - original);
  const Real scale = fmax(1.0, fmax(fabs(original), fabs(replacement)));
  const Real relative = absolute / scale;
  if (!isfinite(original) || !isfinite(replacement) || !isfinite(absolute) ||
      !isfinite(relative)) {
    correction->nonfinite = 1;
  }
  if (absolute > correction->max_abs ||
      (absolute == correction->max_abs && component < correction->component)) {
    correction->max_abs = absolute;
    correction->component = component;
  }
  correction->max_rel = fmax(correction->max_rel, relative);
  state(meshblock, component, k, j, axis_index) = replacement;
}

//! Apply the exact SO(2) state identities at one evolved rho=0 vertex.
template <typename Array5D>
KOKKOS_INLINE_FUNCTION VertexAxisCorrection EnforceVertexAxisZ4cPoint(
    const Array5D &state, const int meshblock, const int k, const int j,
    const int axis_index) {
  VertexAxisCorrection correction;
  const int metric_rr = static_cast<int>(Z4cStateComponent::g_rhorho);
  const int metric_yy = static_cast<int>(Z4cStateComponent::g_yy);
  const int atilde_rr = static_cast<int>(Z4cStateComponent::a_rhorho);
  const int atilde_yy = static_cast<int>(Z4cStateComponent::a_yy);
  const Real metric_average =
      0.5 * (state(meshblock, metric_rr, k, j, axis_index) +
             state(meshblock, metric_yy, k, j, axis_index));
  const Real atilde_average =
      0.5 * (state(meshblock, atilde_rr, k, j, axis_index) +
             state(meshblock, atilde_yy, k, j, axis_index));
  SetVertexAxisComponent(state, meshblock, k, j, axis_index, metric_rr,
                         metric_average, &correction);
  SetVertexAxisComponent(state, meshblock, k, j, axis_index, metric_yy,
                         metric_average, &correction);
  SetVertexAxisComponent(state, meshblock, k, j, axis_index, atilde_rr,
                         atilde_average, &correction);
  SetVertexAxisComponent(state, meshblock, k, j, axis_index, atilde_yy,
                         atilde_average, &correction);

  constexpr Z4cStateComponent zero_components[] = {
      Z4cStateComponent::g_rhoz, Z4cStateComponent::g_rhoy,
      Z4cStateComponent::g_zy, Z4cStateComponent::a_rhoz,
      Z4cStateComponent::a_rhoy, Z4cStateComponent::a_zy,
      Z4cStateComponent::gamma_rho, Z4cStateComponent::gamma_y,
      Z4cStateComponent::beta_rho, Z4cStateComponent::beta_y,
      Z4cStateComponent::b_rho, Z4cStateComponent::b_y};
  for (const auto component : zero_components) {
    SetVertexAxisComponent(state, meshblock, k, j, axis_index,
                           static_cast<int>(component), 0.0, &correction);
  }
  return correction;
}

}  // namespace z4c

#endif  // Z4C_CARTOON_VERTEX_AXIS_HPP_
