#ifndef Z4C_STATE_VIEWS_HPP_
#define Z4C_STATE_VIEWS_HPP_
#include <stdexcept>
#include "z4c/z4c.hpp"
namespace z4c {
inline Z4c::Z4c_vars BindStateViews(const DvceArray5D<Real> &state) {
  if(state.extent_int(1)!=Z4c::nz4c) throw std::invalid_argument("invalid packed Z4c state");
  Z4c::Z4c_vars views;
  views.alpha.InitWithShallowSlice (state, Z4c::I_Z4C_ALPHA);
  views.beta_u.InitWithShallowSlice(state, Z4c::I_Z4C_BETAX, Z4c::I_Z4C_BETAZ);
  views.vB_d.InitWithShallowSlice(state, Z4c::I_Z4C_BX, Z4c::I_Z4C_BZ);
  views.chi.InitWithShallowSlice   (state, Z4c::I_Z4C_CHI);
  views.vKhat.InitWithShallowSlice  (state, Z4c::I_Z4C_KHAT);
  views.vTheta.InitWithShallowSlice (state, Z4c::I_Z4C_THETA);
  views.vGam_u.InitWithShallowSlice (state, Z4c::I_Z4C_GAMX, Z4c::I_Z4C_GAMZ);
  views.g_dd.InitWithShallowSlice  (state, Z4c::I_Z4C_GXX, Z4c::I_Z4C_GZZ);
  views.vA_dd.InitWithShallowSlice  (state, Z4c::I_Z4C_AXX, Z4c::I_Z4C_AZZ);

  return views;
}
}  // namespace z4c
#endif  // Z4C_STATE_VIEWS_HPP_
