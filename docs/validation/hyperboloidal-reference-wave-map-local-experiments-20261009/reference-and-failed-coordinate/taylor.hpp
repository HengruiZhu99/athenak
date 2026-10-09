#ifndef PRIVATE_COORDINATE_TAYLOR_HPP_
#define PRIVATE_COORDINATE_TAYLOR_HPP_
#include <array>
#include <cmath>
#include <stdexcept>
// Univariate Taylor coefficients, coefficient[n]=ordinary derivative/n!.
template<int N> struct Taylor {
  std::array<double,N+1> c{};
  Taylor()=default;Taylor(double x){c[0]=x;}
  double&operator[](int k){return c[k];}double operator[](int k)const{return c[k];}
  Taylor&operator+=(const Taylor&b){for(int k=0;k<=N;++k)c[k]+=b[k];return *this;}
  Taylor&operator-=(const Taylor&b){for(int k=0;k<=N;++k)c[k]-=b[k];return *this;}
  Taylor&operator*=(const Taylor&b){auto a=*this;c.fill(0);for(int k=0;k<=N;++k)for(int j=0;j<=k;++j)c[k]+=a[j]*b[k-j];return *this;}
  friend Taylor operator+(Taylor a,const Taylor&b){return a+=b;}friend Taylor operator-(Taylor a,const Taylor&b){return a-=b;}
  friend Taylor operator*(Taylor a,const Taylor&b){return a*=b;}friend Taylor operator-(Taylor a){for(auto&x:a.c)x=-x;return a;}
};
template<int N> Taylor<N> Inverse(const Taylor<N>&a){if(a[0]==0)throw std::runtime_error("Taylor reciprocal zero");Taylor<N>b(1/a[0]);for(int k=1;k<=N;++k){for(int j=1;j<=k;++j)b[k]-=a[j]*b[k-j];b[k]/=a[0];}return b;}
template<int N> Taylor<N> operator/(const Taylor<N>&a,const Taylor<N>&b){return a*Inverse(b);}
template<int N> Taylor<N> Exp(const Taylor<N>&a){
  // Bell recurrence before exp: preserve derivative tails even when exp(a0)=0.
  Taylor<N>p(1),b;for(int k=1;k<=N;++k){for(int j=1;j<=k;++j)p[k]+=j*a[j]*p[k-j];p[k]/=k;}
  for(int k=0;k<=N;++k)b[k]=p[k]==0?0:std::copysign(std::exp(a[0]+std::log(std::abs(p[k]))),p[k]);return b;
}
template<int N> Taylor<N> Log(const Taylor<N>&a){if(!(a[0]>0))throw std::runtime_error("Taylor logarithm nonpositive");const auto inv=Inverse(a);Taylor<N>b(std::log(a[0]));for(int k=1;k<=N;++k)for(int j=1;j<=k;++j)b[k]+=j*a[j]*inv[k-j]/k;return b;}
template<int N> Taylor<N> Power(const Taylor<N>&a,double p){return Exp(Taylor<N>(p)*Log(a));}
template<int N> Taylor<N> Derivative(const Taylor<N+1>&a){Taylor<N>b;for(int k=0;k<=N;++k)b[k]=(k+1)*a[k+1];return b;}
template<int M,int N> Taylor<M> Truncate(const Taylor<N>&a){static_assert(M<=N,"Cannot invent Taylor coefficients");Taylor<M>b;for(int k=0;k<=M;++k)b[k]=a[k];return b;}
template<int N> Taylor<N> Variable(double r){Taylor<N>a(r);if(N>0)a[1]=1;return a;}
inline double Factorial(int n){double f=1;for(int k=2;k<=n;++k)f*=k;return f;}
#endif
