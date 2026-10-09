// The only geometric change is the prescribed kappa2 argument.
#include "z4c/hyperboloidal/conformal_rhs.hpp"
#include "damping_profile.hpp"
namespace z4c { namespace hyperboloidal {
template<typename T>Z4cRHSParts<T> ProfilePrincipalRHS(const Z4cJet<T>&u,
 const OmegaJet<T>&o,T,T) {
 return ConformalRHS(u,o,T(10)/u.alpha.value,
                    ResearchKappa2Profile(o.omega,T(1),T(.5),T(10)));
}
}}
#define ConformalRHS ProfilePrincipalRHS
#include "../../../../tst/hyperboloidal/kernel_symbol.cpp"
