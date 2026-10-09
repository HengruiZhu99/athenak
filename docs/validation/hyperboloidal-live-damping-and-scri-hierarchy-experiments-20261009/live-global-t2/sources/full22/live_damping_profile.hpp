#ifndef RESEARCH_LIVE_DAMPING_PROFILE_HPP_
#define RESEARCH_LIVE_DAMPING_PROFILE_HPP_
#include "z4c/hyperboloidal/layer_gauge.hpp"
namespace z4c { namespace hyperboloidal {
// Prescribed smooth turn-on, live value coefficient. No clamp, floor, field
// falloff, C1 term or new derivative is added to the evolution equations.
template<typename T>KOKKOS_INLINE_FUNCTION T ResearchLiveKappa2Raw(
 const Z4cJet<T>&u,const OmegaJet<T>&o,T kappa_input) {
 T raw=o.omega-T(1);
 for(int j=0;j<3;++j)raw+=T(2)*u.beta.value[j]*o.gradient[j]/kappa_input;
 return raw;
}
template<typename T>KOKKOS_INLINE_FUNCTION T ResearchLiveKappa2Profile(
 const Z4cJet<T>&u,const OmegaJet<T>&o,T radius,T kappa_input) {
 const auto V=SmoothCutoff(radius,T(.15),T(.3));
 if(V.value==T(0))return T(0);
 return V.value*ResearchLiveKappa2Raw(u,o,kappa_input);
}
template<typename T>struct ResearchDampingJet {T value{},d[3]{};};
// n is the Cartesian radial unit vector at this point (unused where V=0).
// Keep beta derivatives, Omega Hessian and the prescribed turn-on derivative.
template<typename T>KOKKOS_INLINE_FUNCTION ResearchDampingJet<T>
ResearchLiveKappa2SpatialJet(const Z4cJet<T>&u,const OmegaJet<T>&o,
                            T radius,const T n[3],T kappa_input) {
 ResearchDampingJet<T>out{};const auto V=SmoothCutoff(radius,T(.15),T(.3));
 if(V.value==T(0)&&V.d==T(0))return out;
 const T raw=ResearchLiveKappa2Raw(u,o,kappa_input);
 out.value=ResearchLiveKappa2Profile(u,o,radius,kappa_input);
 for(int i=0;i<3;++i){T draw=o.gradient[i];
  for(int j=0;j<3;++j)draw+=T(2)*(u.beta.d[i][j]*o.gradient[j]
                                      +u.beta.value[j]*o.hessian[i][j])/kappa_input;
  out.d[i]=V.d*n[i]*raw+V.value*draw;
 }
 return out;
}
}}
#endif
