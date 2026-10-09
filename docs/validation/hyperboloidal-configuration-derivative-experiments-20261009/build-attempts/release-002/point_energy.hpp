#ifndef PRIVATE_TOTAL_J_POINT_ENERGY_HPP_
#define PRIVATE_TOTAL_J_POINT_ENERGY_HPP_
// Source-only preparation. No radial operator is formed by this header.
// Point inputs must come from the independently gated actual continuum action.
#include <array>
#include <cstddef>

namespace finite_rb {
using Vector20 = std::array<double, 20>;
using Vector10 = std::array<double, 10>;
using Matrix20 = std::array<Vector20, 20>;
using Tensor3 = std::array<std::array<double, 3>, 3>;
using TensorChart = std::array<double, 5>;

// U=(alpha,chi,hnn,hnT,hnU,hplus,hcross,beta_n,beta_T,beta_U).
// V=(P/Omega,Theta/Omega,Ann,AnT,AnU,Aplus,Across,lambda_n,lambda_T,lambda_U).
constexpr std::array<std::size_t, 10> q_indices{0,1,2,8,12,16,18,7,11,15};
constexpr std::array<std::size_t, 10> v_indices{3,4,5,9,13,17,19,6,10,14};

inline TensorChart ToKernelChart(const Tensor3 &t) {
  return {t[0][0],t[0][1],t[0][2],(t[1][1]-t[2][2])/2,t[1][2]};
}
inline Tensor3 FromKernelChart(const TensorChart &v) {
  Tensor3 t{};
  t[0][0]=v[0];t[0][1]=t[1][0]=v[1];t[0][2]=t[2][0]=v[2];
  t[1][1]=-v[0]/2+v[3];t[2][2]=-v[0]/2-v[3];t[1][2]=t[2][1]=v[4];
  return t;
}
inline Vector20 PackPrincipal(const Vector10 &q,const Vector10 &v) {
  Vector20 y{};
  for(std::size_t k=0;k<10;++k){y[q_indices[k]]=q[k];y[v_indices[k]]=v[k];}
  return y;
}
inline Vector20 Multiply(const Matrix20 &a,const Vector20 &x) {
  Vector20 y{};
  for(std::size_t i=0;i<20;++i)for(std::size_t j=0;j<20;++j)y[i]+=a[i][j]*x[j];
  return y;
}
inline Matrix20 Multiply(const Matrix20 &a,const Matrix20 &b) {
  Matrix20 c{};
  for(std::size_t i=0;i<20;++i)for(std::size_t j=0;j<20;++j)
    for(std::size_t k=0;k<20;++k)c[i][j]+=a[i][k]*b[k][j];
  return c;
}
template<std::size_t N>
inline double Dot(const std::array<double,N> &x,const std::array<double,N> &y) {
  double s=0;for(std::size_t k=0;k<N;++k)s+=x[k]*y[k];return s;
}
inline double Bilinear(const Vector20 &x,const Matrix20 &a,const Vector20 &y) {
  return Dot(x,Multiply(a,y));
}
struct Column {
  Vector10 u{},ut{},v_t{};
  Vector20 y{},y_t{},ds_y{};
};
struct Coefficients {
  Matrix20 h{},kn{},ds_h{},ds_kn{};
  double div_s=0,epsilon_over_s2=1;
};
inline Matrix20 VolumeDivergence(const Coefficients &c) {
  const auto hkn=Multiply(c.h,c.kn),first=Multiply(c.ds_h,c.kn),second=Multiply(c.h,c.ds_kn);
  Matrix20 gamma{};
  for(std::size_t i=0;i<20;++i)for(std::size_t j=0;j<20;++j)
    gamma[i][j]=first[i][j]+second[i][j]+c.div_s*hkn[i][j];
  return gamma;
}
inline Vector20 PointRemainder(const Column &column,const Coefficients &c) {
  auto r=column.y_t;const auto principal=Multiply(c.kn,column.ds_y);
  for(std::size_t k=0;k<20;++k)r[k]-=principal[k];
  return r;
}
inline double EnergyDensity(const Column &i,const Column &j,const Coefficients &c) {
  return Bilinear(i.y,c.h,j.y)+c.epsilon_over_s2*Dot(i.u,j.u);
}
inline double StrongSourceDensity(const Column &i,const Column &j,const Coefficients &c) {
  return Bilinear(i.y,c.h,j.y_t)+c.epsilon_over_s2*Dot(i.u,j.ut);
}
// This is not constructed from a K+K^T residual.
inline double SymmetricVolumeDensity(const Column &i,const Column &j,const Coefficients &c) {
  const auto ri=PointRemainder(i,c),rj=PointRemainder(j,c);
  const auto gamma=VolumeDivergence(c);
  return Bilinear(i.y,c.h,rj)+Bilinear(ri,c.h,j.y)-Bilinear(i.y,gamma,j.y)
    +c.epsilon_over_s2*(Dot(i.u,j.ut)+Dot(i.ut,j.u));
}
// ds_hy is independently differentiated from the full reference/field jets.
inline double WeakSourceDensity(const Column &i,const Column &j,const Coefficients &c,
                                const Vector20 &ds_hy) {
  const auto hy=Multiply(c.h,i.y);
  double density=c.epsilon_over_s2*Dot(i.u,j.ut);
  for(std::size_t k=0;k<10;++k)
    density+=-(ds_hy[q_indices[k]]+c.div_s*hy[q_indices[k]])*j.ut[k]
             +hy[v_indices[k]]*j.v_t[k];
  return density;
}
inline double WeakBoundaryDensity(const Column &i,const Column &j,const Coefficients &c) {
  const auto hy=Multiply(c.h,i.y);double density=0;
  for(std::size_t k=0;k<10;++k)density+=hy[q_indices[k]]*j.ut[k];
  return density;
}
inline double PrincipalBoundaryDensity(const Column &i,const Column &j,const Coefficients &c) {
  return Bilinear(i.y,Multiply(c.h,c.kn),j.y);
}
// dSigma=rb^2 dOmega is applied by the trace assembler, not silently here.
inline double IncomingPenaltyWork(const Vector20 &y,const Matrix20 &h,
                                  const Matrix20 &p_plus,double k_in) {
  return -k_in*Bilinear(y,h,Multiply(p_plus,y));
}
} // namespace finite_rb
#endif
