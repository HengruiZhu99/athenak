#ifndef PRIVATE_COORDINATE_CARTESIAN_TAYLOR_HPP_
#define PRIVATE_COORDINATE_CARTESIAN_TAYLOR_HPP_
#include "../taylor.hpp"
#include <array>
#include <algorithm>
// Cartesian multi-index Taylor coefficients; the only stored active entries
// have total degree<=N. Differentiation returns a lower-order type.
template<int N> struct Poly {
  static_assert(N>=0&&N<=3,"Cartesian degree budget");
  std::array<double,64>c{};
  Poly()=default;Poly(double x){c[0]=x;}
  static int at(int i,int j,int k){return 16*i+4*j+k;}
  double&operator()(int i,int j,int k){return c[at(i,j,k)];}double operator()(int i,int j,int k)const{return c[at(i,j,k)];}
  double value()const{return c[0];}
  Poly&operator+=(const Poly&b){for(int i=0;i<=N;++i)for(int j=0;j<=N-i;++j)for(int k=0;k<=N-i-j;++k)(*this)(i,j,k)+=b(i,j,k);return *this;}
  Poly&operator-=(const Poly&b){for(int i=0;i<=N;++i)for(int j=0;j<=N-i;++j)for(int k=0;k<=N-i-j;++k)(*this)(i,j,k)-=b(i,j,k);return *this;}
  Poly&operator*=(const Poly&b){const auto a=*this;c.fill(0);for(int i=0;i<=N;++i)for(int j=0;j<=N-i;++j)for(int k=0;k<=N-i-j;++k)for(int p=0;p<=i;++p)for(int q=0;q<=j;++q)for(int s=0;s<=k;++s)(*this)(i,j,k)+=a(p,q,s)*b(i-p,j-q,k-s);return *this;}
  friend Poly operator+(Poly a,const Poly&b){return a+=b;}friend Poly operator-(Poly a,const Poly&b){return a-=b;}friend Poly operator*(Poly a,const Poly&b){return a*=b;}
  friend Poly operator-(Poly a){for(auto&x:a.c)x=-x;return a;}
};
template<int N> Poly<N> Reciprocal(const Poly<N>&a){if(a.value()==0)throw std::runtime_error("Cartesian reciprocal zero");const auto z=(a-Poly<N>(a.value()))*Poly<N>(-1/a.value());Poly<N>s(1),term(1);for(int k=1;k<=N;++k){term*=z;s+=term;}return s*Poly<N>(1/a.value());}
template<int N> Poly<N> operator/(const Poly<N>&a,const Poly<N>&b){return a*Reciprocal(b);}
template<int M,int N> Poly<M> Cut(const Poly<N>&a){static_assert(M<=N,"No invented Cartesian jets");Poly<M>b;for(int i=0;i<=M;++i)for(int j=0;j<=M-i;++j)for(int k=0;k<=M-i-j;++k)b(i,j,k)=a(i,j,k);return b;}
template<int N> Poly<N-1> Partial(const Poly<N>&a,int direction){static_assert(N>0,"Derivative budget exhausted");Poly<N-1>b;for(int i=0;i<N;++i)for(int j=0;j<N-i;++j)for(int k=0;k<N-i-j;++k){int e[3]={i,j,k};++e[direction];b(i,j,k)=e[direction]*a(e[0],e[1],e[2]);}return b;}
template<int N> Poly<N> CartesianVariable(double x,int direction){Poly<N>a(x);int e[3]={0,0,0};e[direction]=1;if(N>0)a(e[0],e[1],e[2])=1;return a;}
template<int N> Poly<N> CartesianPower(const Poly<N>&a,double exponent){
 if(!(a.value()>0))throw std::runtime_error("Cartesian real power nonpositive");
 const auto z=(a-Poly<N>(a.value()))/Poly<N>(a.value());Poly<N>s(1),term(1);double binomial=1;
 for(int k=1;k<=N;++k){term*=z;binomial*=((exponent-k+1)/k);s+=Poly<N>(binomial)*term;}return Poly<N>(std::pow(a.value(),exponent))*s;
}
template<int N,int M> Poly<N> ComposeRadial(const Taylor<M>&a,const Poly<N>&r,double radius){static_assert(N<=M,"Radial derivative budget exhausted");const auto z=r-Poly<N>(radius);Poly<N>s(a[0]),term(1);for(int k=1;k<=N;++k){term*=z;s+=Poly<N>(a[k])*term;}return s;}
template<int N> using PVector=std::array<Poly<N>,3>;
template<int N> using PMatrix=std::array<std::array<Poly<N>,3>,3>;
template<int N> PMatrix<N> MatrixInv(const PMatrix<N>&g){PMatrix<N>co{},out{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)co[i][j]=g[(i+1)%3][(j+1)%3]*g[(i+2)%3][(j+2)%3]-g[(i+1)%3][(j+2)%3]*g[(i+2)%3][(j+1)%3];Poly<N>det;for(int j=0;j<3;++j)det+=g[0][j]*co[0][j];for(int i=0;i<3;++i)for(int j=0;j<3;++j)out[i][j]=co[j][i]/det;return out;}
template<int M,int N> PMatrix<M> CutMatrix(const PMatrix<N>&a){PMatrix<M>b;for(int i=0;i<3;++i)for(int j=0;j<3;++j)b[i][j]=Cut<M>(a[i][j]);return b;}
template<int N> double Ordinary(const Poly<N>&a,const std::array<int,3>&e){return a(e[0],e[1],e[2])*Factorial(e[0])*Factorial(e[1])*Factorial(e[2]);}
#endif
