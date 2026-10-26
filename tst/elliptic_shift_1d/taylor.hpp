#ifndef ELLIPTIC_SHIFT_1D_TAYLOR_HPP_
#define ELLIPTIC_SHIFT_1D_TAYLOR_HPP_
#include <array>
#include <cmath>
#include <stdexcept>
#include <Kokkos_Core.hpp>
namespace scri_series {
constexpr int degree=16;
struct Taylor {
  std::array<long double,degree+1> c{};
  Taylor(long double x=0){c[0]=x;}
  Taylor& operator+=(const Taylor& b){for(int i=0;i<=degree;++i)c[i]+=b.c[i];return *this;}
  Taylor& operator-=(const Taylor& b){for(int i=0;i<=degree;++i)c[i]-=b.c[i];return *this;}
};
inline Taylor operator+(Taylor a,const Taylor&b){return a+=b;}
inline Taylor operator-(Taylor a,const Taylor&b){return a-=b;}
inline Taylor operator-(Taylor a){for(auto&v:a.c)v=-v;return a;}
inline Taylor operator*(const Taylor&a,const Taylor&b){Taylor o;for(int i=0;i<=degree;++i)for(int j=0;j<=i;++j)o.c[i]+=a.c[j]*b.c[i-j];return o;}
inline Taylor operator/(const Taylor&a,const Taylor&b){if(b.c[0]==0)throw std::runtime_error("Singular Taylor division");Taylor o;for(int i=0;i<=degree;++i){o.c[i]=a.c[i];for(int j=0;j<i;++j)o.c[i]-=o.c[j]*b.c[i-j];o.c[i]/=b.c[0];}return o;}
inline Taylor& operator*=(Taylor&a,const Taylor&b){a=a*b;return a;}
inline Taylor& operator/=(Taylor&a,const Taylor&b){a=a/b;return a;}
inline bool operator>(const Taylor&a,const Taylor&b){return a.c[0]>b.c[0];}
inline bool operator<(const Taylor&a,const Taylor&b){return a.c[0]<b.c[0];}
inline Taylor Dx(const Taylor&a){Taylor o;for(int i=0;i<degree;++i)o.c[i]=(i+1)*a.c[i+1];return o;}
inline Taylor DivX(const Taylor&a){Taylor o;for(int i=0;i<degree;++i)o.c[i]=a.c[i+1];return o;}
inline Taylor Sqrt(const Taylor&a){if(a.c[0]<=0)throw std::runtime_error("Nonpositive Taylor square root");Taylor o;o.c[0]=std::sqrt(a.c[0]);for(int i=1;i<=degree;++i){o.c[i]=a.c[i];for(int j=1;j<i;++j)o.c[i]-=o.c[j]*o.c[i-j];o.c[i]/=2*o.c[0];}return o;}
inline Taylor Integral(const Taylor&a,long double constant=0){Taylor o(constant);for(int i=1;i<=degree;++i)o.c[i]=a.c[i-1]/i;return o;}
}
namespace Kokkos {
inline scri_series::Taylor sqrt(const scri_series::Taylor&a){return scri_series::Sqrt(a);}
inline bool isfinite(const scri_series::Taylor&a){for(auto v:a.c)if(!std::isfinite(v))return false;return true;}
}
#endif
