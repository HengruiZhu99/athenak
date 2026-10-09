#ifndef SCRATCH_CUBIC_TAYLOR_HPP
#define SCRATCH_CUBIC_TAYLOR_HPP
#include <Kokkos_Core.hpp>
#include <array>
#include <cmath>
#include <stdexcept>
struct Poly {
  static constexpr int size=20;
  std::array<long double,size> c{};
  Poly()=default; Poly(long double x){c[0]=x;}
  static const std::array<std::array<int,3>,size>& Exponents(){
    static const auto e=[](){std::array<std::array<int,3>,size>a{};int k=0;
      for(int d=0;d<=3;++d)for(int i=d;i>=0;--i)for(int j=d-i;j>=0;--j)a[k++]={i,j,d-i-j};return a;}();return e;
  }
  static int Index(int a,int b,int c){if(a<0||b<0||c<0||a+b+c>3)return -1;const auto&e=Exponents();for(int i=0;i<size;++i)if(e[i]==std::array<int,3>{a,b,c})return i;return -1;}
  static Poly Variable(int i){Poly a;a.c[Index(i==0,i==1,i==2)]=1;return a;}
  Poly&operator+=(const Poly&b){for(int i=0;i<size;++i)c[i]+=b.c[i];return *this;}
  Poly&operator-=(const Poly&b){for(int i=0;i<size;++i)c[i]-=b.c[i];return *this;}
};
inline Poly operator+(Poly a,const Poly&b){return a+=b;}inline Poly operator-(Poly a,const Poly&b){return a-=b;}
inline Poly operator-(Poly a){for(auto&x:a.c)x=-x;return a;}
inline Poly operator*(const Poly&a,const Poly&b){Poly out;const auto&e=Poly::Exponents();
  for(int i=0;i<20;++i)if(a.c[i]!=0)for(int j=0;j<20;++j)if(b.c[j]!=0){int k=Poly::Index(e[i][0]+e[j][0],e[i][1]+e[j][1],e[i][2]+e[j][2]);if(k>=0)out.c[k]+=a.c[i]*b.c[j];}return out;}
inline Poly operator*(Poly a,long double s){for(auto&x:a.c)x*=s;return a;}
inline Poly operator*(long double s,Poly a){return a*s;}
inline Poly Power(Poly a,int n){Poly out(1);for(int i=0;i<n;++i)out=out*a;return out;}
inline Poly Inverse(const Poly&a){if(a.c[0]==0)throw std::runtime_error("Taylor inverse has zero constant");Poly q=a*(1/a.c[0])-Poly(1);return (Poly(1)-q+q*q-q*q*q)*(1/a.c[0]);}
inline Poly operator/(const Poly&a,const Poly&b){return a*Inverse(b);}
inline Poly Derivative(const Poly&a,int v){Poly out;const auto&e=Poly::Exponents();for(int i=0;i<20;++i)if(e[i][v]){auto k=e[i];const int f=k[v]--;out.c[Poly::Index(k[0],k[1],k[2])]+=f*a.c[i];}return out;}
inline Poly Compose(const Poly&a,const std::array<Poly,3>&x){Poly out;const auto&e=Poly::Exponents();for(int i=0;i<20;++i)if(a.c[i])out+=a.c[i]*Power(x[0],e[i][0])*Power(x[1],e[i][1])*Power(x[2],e[i][2]);return out;}
inline long double Evaluate(const Poly&a,const std::array<long double,3>&x){long double out=0;const auto&e=Poly::Exponents();for(int i=0;i<20;++i){long double z=a.c[i];for(int d=0;d<3;++d)for(int k=0;k<e[i][d];++k)z*=x[d];out+=z;}return out;}
inline Poly Log(const Poly&a){if(!(a.c[0]>0))throw std::runtime_error("Taylor log");const Poly q=a*(1/a.c[0])-Poly(1);return Poly(std::log(a.c[0]))+q-q*q*.5L+q*q*q/3.L;}
inline Poly Exp(const Poly&a){Poly q=a-Poly(a.c[0]);return (Poly(1)+q+q*q*.5L+q*q*q/6.L)*std::exp(a.c[0]);}
inline Poly Sqrt(const Poly&a){if(!(a.c[0]>0))throw std::runtime_error("Taylor sqrt");const Poly q=a*(1/a.c[0])-Poly(1);return (Poly(1)+q*.5L-q*q*.125L+q*q*q*.0625L)*std::sqrt(a.c[0]);}
struct D {
  Poly v{},d{};D()=default;D(long double x):v(x){}D(const Poly&x):v(x){}D(const Poly&x,const Poly&y):v(x),d(y){}
  D&operator+=(const D&x){v+=x.v;d+=x.d;return *this;}D&operator-=(const D&x){v-=x.v;d-=x.d;return *this;}
  D&operator*=(const D&x){d=d*x.v+v*x.d;v=v*x.v;return *this;}D&operator/=(const D&x){d=(d*x.v-v*x.d)/(x.v*x.v);v=v/x.v;return *this;}
};
inline D operator+(D a,const D&b){return a+=b;}inline D operator-(D a,const D&b){return a-=b;}inline D operator-(D a){return {-a.v,-a.d};}
inline D operator*(D a,const D&b){return a*=b;}inline D operator/(D a,const D&b){return a/=b;}
inline bool operator>(const D&a,const D&b){return a.v.c[0]>b.v.c[0];}inline bool operator<(const D&a,const D&b){return a.v.c[0]<b.v.c[0];}
inline bool operator<=(const D&a,const D&b){return a.v.c[0]<=b.v.c[0];}inline bool operator>=(const D&a,const D&b){return a.v.c[0]>=b.v.c[0];}inline bool operator==(const D&a,const D&b){return a.v.c[0]==b.v.c[0];}
inline D Derivative(const D&a,int v){return {Derivative(a.v,v),Derivative(a.d,v)};}
inline D Compose(const D&a,const std::array<Poly,3>&x){return {Compose(a.v,x),Compose(a.d,x)};}
namespace Kokkos {
inline bool isfinite(const D&a){for(int i=0;i<20;++i)if(!std::isfinite(a.v.c[i])||!std::isfinite(a.d.c[i]))return false;return true;}
inline D abs(const D&a){return a.v.c[0]>=0?a:-a;}
inline D exp(const D&a){const auto v=Exp(a.v);return {v,v*a.d};}
inline D log(const D&a){return {Log(a.v),a.d/a.v};}inline D log1p(const D&a){return log(D(1)+a);}
inline D sqrt(const D&a){const auto v=Sqrt(a.v);return {v,a.d/(2.L*v)};}
inline D pow(const D&a,const D&b){return exp(b*log(a));}
inline D hypot(const D&a,const D&b){return sqrt(a*a+b*b);}
}
#endif
