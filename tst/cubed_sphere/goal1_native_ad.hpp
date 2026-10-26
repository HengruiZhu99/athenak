// Private scalar adapter. Include before any native hyperboloidal templates.
#pragma once
#include <cmath>
#include "mesh/cubed_sphere/map.hpp"
namespace cubed_sphere {
inline Jet operator-(const Jet&a){return Jet(0)-a;}
inline Jet&operator+=(Jet&a,const Jet&b){a=a+b;return a;}
inline Jet&operator-=(Jet&a,const Jet&b){a=a-b;return a;}
inline Jet&operator*=(Jet&a,const Jet&b){a=a*b;return a;}
inline Jet&operator/=(Jet&a,const Jet&b){a=a/b;return a;}
inline bool operator>(const Jet&a,const Jet&b){return a.f>b.f;}
inline bool operator<(const Jet&a,const Jet&b){return a.f<b.f;}
inline bool operator>=(const Jet&a,const Jet&b){return a.f>=b.f;}
inline bool operator<=(const Jet&a,const Jet&b){return a.f<=b.f;}
inline bool operator==(const Jet&a,const Jet&b){return a.f==b.f;}
inline bool operator!=(const Jet&a,const Jet&b){return a.f!=b.f;}
inline double Root(double a){return std::sqrt(a);}
inline Jet Root(const Jet&a){return Power(a,.5);}
inline bool JetFinite(const Jet&a){if(!std::isfinite(a.f))return false;for(int i=0;i<3;++i){if(!std::isfinite(a.d[i]))return false;for(int j=0;j<3;++j)if(!std::isfinite(a.h[i][j]))return false;}return true;}
}
namespace Kokkos {
// Qualified lookup in the native template requires this overload beforehand.
inline bool isfinite(const cubed_sphere::Jet&a){return cubed_sphere::JetFinite(a);}
}
