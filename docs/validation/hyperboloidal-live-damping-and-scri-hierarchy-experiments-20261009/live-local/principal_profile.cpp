// The only geometric change is the prescribed kappa2 argument.
#include "z4c/hyperboloidal/conformal_rhs.hpp"
#include "live_damping_profile.hpp"
double live_radius=0;
namespace z4c { namespace hyperboloidal {
template<typename T>Z4cRHSParts<T> ProfilePrincipalRHS(const Z4cJet<T>&u,
 const OmegaJet<T>&o,T,T) {
 return ConformalRHS(u,o,T(10)/u.alpha.value,
                    ResearchLiveKappa2Profile(u,o,T(live_radius),T(10)));
}
}}
#define ConformalRHS ProfilePrincipalRHS
#include "kernel_symbol_copy.cpp"
