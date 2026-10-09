#ifndef SCRATCH_GENERAL_BETA2_HPP_
#define SCRATCH_GENERAL_BETA2_HPP_
#include <cmath>
#include <initializer_list>
#include <stdexcept>
namespace outer_beta2 {
// Coefficient of Omega^2 added to the contravariant radial shift. This is an
// initial instantaneous compatibility calculation, not a gauge boundary closure.
inline double Coefficient(double S,double a,double M,double rho,
                          double nu=1.5,double eta_regular=1.) {
 for(double x:{S,a,M,rho,nu,eta_regular})if(!std::isfinite(x))
  throw std::invalid_argument("outer beta2 parameters must be finite");
 if(!(S>0&&a>=S/2&&M>=0&&rho>0&&nu>=0&&eta_regular>=0))
  throw std::invalid_argument("invalid outer beta2 parameters");
 if(rho==4)throw std::domain_error("rho4 has a second-jet obstruction or a nonunique coefficient");
 const double result=M*(M*(rho-8)+4*a*a*(2*eta_regular-nu)+8*a*(4-rho))/(4*S*a*(4-rho));
 if(!std::isfinite(result))throw std::overflow_error("outer beta2 coefficient is not finite");
 return result;
}
}
#endif
