//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cartoon_axis_parity_test.cpp
//! \brief Exhaustive packed-layout and parity checks for half-plane SO(2).

#include <array>
#include <iostream>

#include "coordinates/adm.hpp"
#include "z4c/cartoon_axis_parity.hpp"
#include "z4c/z4c.hpp"

namespace {

using z4c::AdmStateComponent;
using z4c::AxisDirection;
using z4c::AxisParity;
using z4c::ConstraintComponent;
using z4c::Z4cStateComponent;

constexpr int Value(const Z4cStateComponent component) {
  return static_cast<int>(component);
}

constexpr int Value(const AdmStateComponent component) {
  return static_cast<int>(component);
}

constexpr int Value(const ConstraintComponent component) {
  return static_cast<int>(component);
}

static_assert(Value(Z4cStateComponent::chi) == z4c::Z4c::I_Z4C_CHI);
static_assert(Value(Z4cStateComponent::g_rhorho) == z4c::Z4c::I_Z4C_GXX);
static_assert(Value(Z4cStateComponent::g_rhoz) == z4c::Z4c::I_Z4C_GXY);
static_assert(Value(Z4cStateComponent::g_rhoy) == z4c::Z4c::I_Z4C_GXZ);
static_assert(Value(Z4cStateComponent::g_zz) == z4c::Z4c::I_Z4C_GYY);
static_assert(Value(Z4cStateComponent::g_zy) == z4c::Z4c::I_Z4C_GYZ);
static_assert(Value(Z4cStateComponent::g_yy) == z4c::Z4c::I_Z4C_GZZ);
static_assert(Value(Z4cStateComponent::khat) == z4c::Z4c::I_Z4C_KHAT);
static_assert(Value(Z4cStateComponent::a_rhorho) == z4c::Z4c::I_Z4C_AXX);
static_assert(Value(Z4cStateComponent::a_rhoz) == z4c::Z4c::I_Z4C_AXY);
static_assert(Value(Z4cStateComponent::a_rhoy) == z4c::Z4c::I_Z4C_AXZ);
static_assert(Value(Z4cStateComponent::a_zz) == z4c::Z4c::I_Z4C_AYY);
static_assert(Value(Z4cStateComponent::a_zy) == z4c::Z4c::I_Z4C_AYZ);
static_assert(Value(Z4cStateComponent::a_yy) == z4c::Z4c::I_Z4C_AZZ);
static_assert(Value(Z4cStateComponent::gamma_rho) == z4c::Z4c::I_Z4C_GAMX);
static_assert(Value(Z4cStateComponent::gamma_z) == z4c::Z4c::I_Z4C_GAMY);
static_assert(Value(Z4cStateComponent::gamma_y) == z4c::Z4c::I_Z4C_GAMZ);
static_assert(Value(Z4cStateComponent::theta) == z4c::Z4c::I_Z4C_THETA);
static_assert(Value(Z4cStateComponent::alpha) == z4c::Z4c::I_Z4C_ALPHA);
static_assert(Value(Z4cStateComponent::beta_rho) == z4c::Z4c::I_Z4C_BETAX);
static_assert(Value(Z4cStateComponent::beta_z) == z4c::Z4c::I_Z4C_BETAY);
static_assert(Value(Z4cStateComponent::beta_y) == z4c::Z4c::I_Z4C_BETAZ);
static_assert(Value(Z4cStateComponent::b_rho) == z4c::Z4c::I_Z4C_BX);
static_assert(Value(Z4cStateComponent::b_z) == z4c::Z4c::I_Z4C_BY);
static_assert(Value(Z4cStateComponent::b_y) == z4c::Z4c::I_Z4C_BZ);
static_assert(Value(Z4cStateComponent::count) == z4c::Z4c::nz4c);

static_assert(Value(AdmStateComponent::g_rhorho) == adm::ADM::I_ADM_GXX);
static_assert(Value(AdmStateComponent::g_rhoz) == adm::ADM::I_ADM_GXY);
static_assert(Value(AdmStateComponent::g_rhoy) == adm::ADM::I_ADM_GXZ);
static_assert(Value(AdmStateComponent::g_zz) == adm::ADM::I_ADM_GYY);
static_assert(Value(AdmStateComponent::g_zy) == adm::ADM::I_ADM_GYZ);
static_assert(Value(AdmStateComponent::g_yy) == adm::ADM::I_ADM_GZZ);
static_assert(Value(AdmStateComponent::k_rhorho) == adm::ADM::I_ADM_KXX);
static_assert(Value(AdmStateComponent::k_rhoz) == adm::ADM::I_ADM_KXY);
static_assert(Value(AdmStateComponent::k_rhoy) == adm::ADM::I_ADM_KXZ);
static_assert(Value(AdmStateComponent::k_zz) == adm::ADM::I_ADM_KYY);
static_assert(Value(AdmStateComponent::k_zy) == adm::ADM::I_ADM_KYZ);
static_assert(Value(AdmStateComponent::k_yy) == adm::ADM::I_ADM_KZZ);
static_assert(Value(AdmStateComponent::psi4) == adm::ADM::I_ADM_PSI4);
static_assert(Value(AdmStateComponent::alpha) == adm::ADM::I_ADM_ALPHA);
static_assert(Value(AdmStateComponent::beta_rho) == adm::ADM::I_ADM_BETAX);
static_assert(Value(AdmStateComponent::beta_z) == adm::ADM::I_ADM_BETAY);
static_assert(Value(AdmStateComponent::beta_y) == adm::ADM::I_ADM_BETAZ);
static_assert(Value(AdmStateComponent::count) == adm::ADM::nadm);

static_assert(Value(ConstraintComponent::aggregate) == z4c::Z4c::I_CON_C);
static_assert(Value(ConstraintComponent::hamiltonian) == z4c::Z4c::I_CON_H);
static_assert(Value(ConstraintComponent::momentum_norm_squared) == z4c::Z4c::I_CON_M);
static_assert(Value(ConstraintComponent::z_norm_squared) == z4c::Z4c::I_CON_Z);
static_assert(Value(ConstraintComponent::momentum_rho) == z4c::Z4c::I_CON_MX);
static_assert(Value(ConstraintComponent::momentum_z) == z4c::Z4c::I_CON_MY);
static_assert(Value(ConstraintComponent::momentum_y) == z4c::Z4c::I_CON_MZ);
static_assert(Value(ConstraintComponent::count) == z4c::Z4c::ncon);

template <typename Component, std::size_t N, typename ParityFunction>
bool CheckTable(const std::array<AxisParity, N> &expected,
                ParityFunction parity) {
  for (std::size_t index = 0; index < N; ++index) {
    if (parity(static_cast<Component>(index)) != expected[index]) return false;
  }
  return true;
}

}  // namespace

int main() {
  constexpr std::array<AxisParity, z4c::Z4c::nz4c> z4c_expected = {
      AxisParity::even,
      AxisParity::even, AxisParity::odd, AxisParity::even,
      AxisParity::even, AxisParity::odd, AxisParity::even,
      AxisParity::even,
      AxisParity::even, AxisParity::odd, AxisParity::even,
      AxisParity::even, AxisParity::odd, AxisParity::even,
      AxisParity::odd, AxisParity::even, AxisParity::odd,
      AxisParity::even, AxisParity::even,
      AxisParity::odd, AxisParity::even, AxisParity::odd,
      AxisParity::odd, AxisParity::even, AxisParity::odd};
  constexpr std::array<AxisParity, adm::ADM::nadm> adm_expected = {
      AxisParity::even, AxisParity::odd, AxisParity::even,
      AxisParity::even, AxisParity::odd, AxisParity::even,
      AxisParity::even, AxisParity::odd, AxisParity::even,
      AxisParity::even, AxisParity::odd, AxisParity::even,
      AxisParity::even, AxisParity::even,
      AxisParity::odd, AxisParity::even, AxisParity::odd};
  constexpr std::array<AxisParity, z4c::Z4c::ncon> constraint_expected = {
      AxisParity::even, AxisParity::even, AxisParity::even, AxisParity::even,
      AxisParity::odd, AxisParity::even, AxisParity::odd};

  const bool directions =
      z4c::VectorAxisParity(AxisDirection::rho) == AxisParity::odd &&
      z4c::VectorAxisParity(AxisDirection::axial) == AxisParity::even &&
      z4c::VectorAxisParity(AxisDirection::suppressed) == AxisParity::odd;
  const bool tensors =
      z4c::SymmetricTensorAxisParity(AxisDirection::rho,
                                    AxisDirection::rho) == AxisParity::even &&
      z4c::SymmetricTensorAxisParity(AxisDirection::rho,
                                    AxisDirection::axial) == AxisParity::odd &&
      z4c::SymmetricTensorAxisParity(AxisDirection::rho,
                                    AxisDirection::suppressed) == AxisParity::even &&
      z4c::SymmetricTensorAxisParity(AxisDirection::axial,
                                    AxisDirection::suppressed) == AxisParity::odd;
  const bool states = CheckTable<Z4cStateComponent>(
      z4c_expected, z4c::Z4cStateAxisParity) &&
      CheckTable<AdmStateComponent>(adm_expected, z4c::AdmStateAxisParity) &&
      CheckTable<ConstraintComponent>(constraint_expected,
                                      z4c::ConstraintAxisParity);

  if (!directions || !tensors || !states) {
    std::cerr << "half-plane Cartoon axis parity contract failed\n";
    return 1;
  }
  std::cout << "half-plane Cartoon axis parity contract passed\n";
  return 0;
}
