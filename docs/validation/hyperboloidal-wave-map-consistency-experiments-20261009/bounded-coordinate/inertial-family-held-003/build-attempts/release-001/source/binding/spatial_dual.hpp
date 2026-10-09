// Derived read-only copy of the frozen continuum dual helpers; no native evolution.
// Exploratory continuum subsidiary audit. The dual derivative is exact;
// fourth-order finite differences differentiate only smooth operator coefficients.
#include <Kokkos_Core.hpp>
#include <array>
#include <complex>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
struct D {double v=0,d=0;D()=default;D(double x):v(x){}D(double x,double y):v(x),d(y){};
 D&operator+=(D x){v+=x.v;d+=x.d;return *this;}D&operator-=(D x){v-=x.v;d-=x.d;return *this;}
 D&operator*=(D x){d=d*x.v+v*x.d;v*=x.v;return *this;}D&operator/=(D x){d=(d*x.v-v*x.d)/(x.v*x.v);v/=x.v;return *this;}};
D operator+(D a,D b){return a+=b;}D operator-(D a,D b){return a-=b;}D operator*(D a,D b){return a*=b;}D operator/(D a,D b){return a/=b;}D operator-(D a){return {-a.v,-a.d};}
bool operator>(D a,D b){return a.v>b.v;}bool operator<(D a,D b){return a.v<b.v;}
namespace Kokkos {inline bool isfinite(D x){return std::isfinite(x.v)&&std::isfinite(x.d);}}
bool operator<=(D a,D b){return a.v<=b.v;}bool operator>=(D a,D b){return a.v>=b.v;}bool operator==(D a,D b){return a.v==b.v;}
namespace Kokkos {
inline D abs(D x){return x.v>=0?x:-x;}inline D exp(D x){const double y=std::exp(x.v);return {y,y*x.d};}
inline D log(D x){return {std::log(x.v),x.d/x.v};}inline D log1p(D x){return {std::log1p(x.v),x.d/(1+x.v)};}
inline D sqrt(D x){const double y=std::sqrt(x.v);return {y,x.d/(2*y)};}
inline D pow(D x,D y){const double z=std::pow(x.v,y.v);return {z,z*(y.d*std::log(x.v)+y.v*x.d/x.v)};}
}

// Independent first spatial jet over the exact perturbation dual D.
// Only first-order configuration rows consume this type. Missing third jets
// are not inserted into a full second-order source action.
template<class T> struct S1 {
 T v{};std::array<T,3>d{};
 S1()=default;template<class X>S1(X a):v(a){}
 S1&operator+=(const S1&a){v+=a.v;for(int k=0;k<3;++k)d[k]+=a.d[k];return *this;}
 S1&operator-=(const S1&a){v-=a.v;for(int k=0;k<3;++k)d[k]-=a.d[k];return *this;}
 S1&operator*=(const S1&a){const auto old=v;for(int k=0;k<3;++k)d[k]=d[k]*a.v+old*a.d[k];v*=a.v;return *this;}
 S1&operator/=(const S1&a){const auto old=v;for(int k=0;k<3;++k)d[k]=(d[k]*a.v-old*a.d[k])/(a.v*a.v);v/=a.v;return *this;}
 friend S1 operator+(S1 a,const S1&b){return a+=b;}
 friend S1 operator-(S1 a,const S1&b){return a-=b;}
 friend S1 operator*(S1 a,const S1&b){return a*=b;}
 friend S1 operator/(S1 a,const S1&b){return a/=b;}
 friend S1 operator-(S1 a){a.v=-a.v;for(auto&x:a.d)x=-x;return a;}
 friend bool operator>(const S1&a,const S1&b){return a.v>b.v;}
 friend bool operator<(const S1&a,const S1&b){return a.v<b.v;}
 friend bool operator>=(const S1&a,const S1&b){return a.v>=b.v;}
 friend bool operator<=(const S1&a,const S1&b){return a.v<=b.v;}
 friend bool operator==(const S1&a,const S1&b){return a.v==b.v;}
};
namespace Kokkos {
 template<class T>inline bool isfinite(const S1<T>&x){bool ok=Kokkos::isfinite(x.v);for(const auto&d:x.d)ok=ok&&Kokkos::isfinite(d);return ok;}
 template<class T>inline S1<T> abs(S1<T>x){return x.v>=T(0)?x:-x;}
 template<class T>inline S1<T> exp(const S1<T>&x){S1<T>y(Kokkos::exp(x.v));for(int k=0;k<3;++k)y.d[k]=y.v*x.d[k];return y;}
 template<class T>inline S1<T> log(const S1<T>&x){S1<T>y(Kokkos::log(x.v));for(int k=0;k<3;++k)y.d[k]=x.d[k]/x.v;return y;}
 template<class T>inline S1<T> log1p(const S1<T>&x){S1<T>y(Kokkos::log1p(x.v));for(int k=0;k<3;++k)y.d[k]=x.d[k]/(T(1)+x.v);return y;}
 template<class T>inline S1<T> sqrt(const S1<T>&x){S1<T>y(Kokkos::sqrt(x.v));for(int k=0;k<3;++k)y.d[k]=x.d[k]/(T(2)*y.v);return y;}
 template<class T>inline S1<T> pow(const S1<T>&x,const S1<T>&a){return Kokkos::exp(a*Kokkos::log(x));}
}

#include "z4c/hyperboloidal/layer_reference.hpp"
namespace hyp=z4c::hyperboloidal;
using Jet=hyp::Z4cJet<D>;using C=std::complex<double>;const C I(0,1);
using M=std::array<std::array<C,20>,20>;
struct Sample{M f{},q{};};
struct J {D v{},d[3]{},dd[3][3]{};J()=default;J(D x):v(x){}};
J operator+(const J&a,const J&b){J c(a.v+b.v);for(int i=0;i<3;++i){c.d[i]=a.d[i]+b.d[i];for(int j=0;j<3;++j)c.dd[i][j]=a.dd[i][j]+b.dd[i][j];}return c;}
J operator-(const J&a,const J&b){J c(a.v-b.v);for(int i=0;i<3;++i){c.d[i]=a.d[i]-b.d[i];for(int j=0;j<3;++j)c.dd[i][j]=a.dd[i][j]-b.dd[i][j];}return c;}
J operator*(const J&a,const J&b){J c(a.v*b.v);for(int i=0;i<3;++i){c.d[i]=a.d[i]*b.v+a.v*b.d[i];for(int j=0;j<3;++j)c.dd[i][j]=a.dd[i][j]*b.v+a.v*b.dd[i][j]+a.d[i]*b.d[j]+a.d[j]*b.d[i];}return c;}
J inv(const J&a){J b(D(1)/a.v);for(int i=0;i<3;++i){b.d[i]=-a.d[i]/(a.v*a.v);for(int j=0;j<3;++j)b.dd[i][j]=D(2)*a.d[i]*a.d[j]/(a.v*a.v*a.v)-a.dd[i][j]/(a.v*a.v);}return b;}
J operator/(const J&a,const J&b){return a*inv(b);}
J Metric(const Jet&u,int i,int j){J x(u.metric.g[i][j]);for(int d=0;d<3;++d){x.d[d]=u.metric.dg[d][i][j];for(int e=0;e<3;++e)x.dd[d][e]=u.metric.ddg[d][e][i][j];}return x;}
J A(const Jet&u,int i,int j){J x(u.a.k[i][j]);for(int d=0;d<3;++d)x.d[d]=u.a.dk[d][i][j];return x;}
void SetMetric(Jet&u,int i,int j,const J&x){u.metric.g[i][j]=u.metric.g[j][i]=x.v;for(int d=0;d<3;++d){u.metric.dg[d][i][j]=u.metric.dg[d][j][i]=x.d[d];for(int e=0;e<3;++e)u.metric.ddg[d][e][i][j]=u.metric.ddg[d][e][j][i]=x.dd[d][e];}}
void SetA(Jet&u,int i,int j,const J&x){u.a.k[i][j]=u.a.k[j][i]=x.v;for(int d=0;d<3;++d)u.a.dk[d][i][j]=u.a.dk[d][j][i]=x.d[d];}
void Consistent(Jet&u){const J x=Metric(u,0,0),y=Metric(u,0,1),z=Metric(u,0,2),w=Metric(u,1,1),v=Metric(u,1,2);
 SetMetric(u,2,2,(J(D(1))-J(D(2))*y*z*v+x*v*v+w*z*z)/(x*w-y*y));const J t=Metric(u,2,2);
 const J ix=w*t-v*v,iy=z*v-y*t,iz=y*v-z*w,iw=x*t-z*z,iv=y*z-x*v,it=x*w-y*y;
 SetA(u,2,2,(J(D(0))-ix*A(u,0,0)-J(D(2))*iy*A(u,0,1)-J(D(2))*iz*A(u,0,2)-iw*A(u,1,1)-J(D(2))*iv*A(u,1,2))/it);}
void Scalar(hyp::ScalarJet<D>&u,const J&x){u.value+=x.v;for(int d=0;d<3;++d){u.d[d]+=x.d[d];for(int e=0;e<3;++e)u.dd[d][e]+=x.dd[d][e];}}
void Vector(hyp::VectorJet<D>&u,int i,const J&x){u.value[i]+=x.v;for(int d=0;d<3;++d){u.d[d][i]+=x.d[d];for(int e=0;e<3;++e)u.dd[d][e][i]+=x.dd[d][e];}}
void Seed(Jet&u,int c,const J&x){const int ti[5]={0,0,0,1,1},tj[5]={0,1,2,1,2};
 if(c==0)Scalar(u.alpha,x);if(c==1)Scalar(u.chi,x);if(c==2)Scalar(u.trace,x);if(c==3)Scalar(u.theta,x);
 if(c>=4&&c<7)Vector(u.beta,c-4,x);if(c>=7&&c<12){const int i=ti[c-7],j=tj[c-7];SetMetric(u,i,j,Metric(u,i,j)+x);}
 if(c>=12&&c<17){const int i=ti[c-12],j=tj[c-12];SetA(u,i,j,A(u,i,j)+x);}if(c>=17)Vector(u.lambda,c-17,x);}
Jet Lift(const hyp::Z4cJet<double>&s){Jet u{};auto scalar=[](auto&a,const auto&b){a.value=b.value;for(int i=0;i<3;++i){a.d[i]=b.d[i];for(int j=0;j<3;++j)a.dd[i][j]=b.dd[i][j];}};
 scalar(u.alpha,s.alpha);scalar(u.chi,s.chi);scalar(u.trace,s.trace);scalar(u.theta,s.theta);
 for(int i=0;i<3;++i){u.beta.value[i]=s.beta.value[i];u.lambda.value[i]=s.lambda.value[i];for(int j=0;j<3;++j){u.beta.d[j][i]=s.beta.d[j][i];u.lambda.d[j][i]=s.lambda.d[j][i];u.metric.g[i][j]=s.metric.g[i][j];u.a.k[i][j]=s.a.k[i][j];for(int k=0;k<3;++k){u.beta.dd[k][j][i]=s.beta.dd[k][j][i];u.lambda.dd[k][j][i]=s.lambda.dd[k][j][i];u.metric.dg[k][i][j]=s.metric.dg[k][i][j];u.a.dk[k][i][j]=s.a.dk[k][i][j];for(int l=0;l<3;++l)u.metric.ddg[k][l][i][j]=s.metric.ddg[k][l][i][j];}}}return u;}
hyp::OmegaJet<D> Omega(const Jet&u,const hyp::LayerPoint<double>&p){hyp::OmegaJet<D> o{};o.omega=p.omega;for(int i=0;i<3;++i){o.gradient[i]=p.domega[i];for(int j=0;j<3;++j)o.hessian[i][j]=p.omega_hessian[i][j];}hyp::SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,o);return o;}
std::array<double,8> Constraints(const Jet&u,const hyp::LayerPoint<double>&p){auto c=hyp::EvolvedConstraints(u,Omega(u,p));if(!c.valid)throw std::runtime_error("constraints invalid");return {c.hamiltonian.d,c.momentum[0].d,c.momentum[1].d,c.momentum[2].d,c.z4.z_covector[0].d,c.z4.z_covector[1].d,c.z4.z_covector[2].d,c.z4.theta_physical.d};}
Sample Point(const hyp::LayerReference<double>&ref,const std::array<double,3>&x,double k,const std::array<double,3>&n,double kappa){const auto p=ref.At(x[0],x[1],x[2]);Sample s;
 for(int col=0;col<20;++col)for(int phase=0;phase<2;++phase){auto u=Lift(p.state);J seed(D(0,phase?0:1));for(int i=0;i<3;++i){seed.d[i]=D(0,phase?k*n[i]:0);for(int j=0;j<3;++j)seed.dd[i][j]=D(0,phase?0:-k*k*n[i]*n[j]);}Seed(u,col,seed);Consistent(u);
  auto o=Omega(u,p);hyp::Z4cRHS<D> f;if(!hyp::AssembleInterior(hyp::ConformalRHS(u,o,D(kappa)/u.alpha.value,D(0)),o.omega,f))throw std::runtime_error("rhs invalid");
  double v[20]{};v[1]=f.chi.d;v[2]=f.trace.d;v[3]=f.theta.d;const int ti[5]={0,0,0,1,1},tj[5]={0,1,2,1,2};for(int j=0;j<3;++j)v[17+j]=f.lambda[j].d;for(int j=0;j<5;++j){v[7+j]=f.metric[ti[j]][tj[j]].d;v[12+j]=f.a[ti[j]][tj[j]].d;}
  const C factor=phase?I:C(1,0);for(int row=0;row<20;++row)s.f[row][col]+=factor*v[row];auto q=Constraints(u,p);for(int row=0;row<8;++row)s.q[row][col]+=factor*q[row];
 }return s;}
