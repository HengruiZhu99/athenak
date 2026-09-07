// Complete intrinsic configuration sources. No primary derivative is replaced
// by an independent auxiliary when differentiating these composite functions.
#ifndef PC_GH_INTRINSIC_SOURCES_HPP_
#define PC_GH_INTRINSIC_SOURCES_HPP_
#include "pc_gh/intrinsic_jet.hpp"
#include "pc_gh/intrinsic_geometry.hpp"
namespace pc_gh::intrinsic {
template<typename T>
KOKKOS_INLINE_FUNCTION T GaugePlateau(T z) {
  using std::exp;
  if (ScalarValue(z)<=0.1) return T(0);
  if (ScalarValue(z)>=0.5) return T(1);
  T u=(z-0.1)/0.4;
  T log_ratio=1/(1-u)-1/u;
  // Logistic form avoids overflow and divides by quantities >=1.
  if (ScalarValue(log_ratio)>=0) return 1/(1+exp(-log_ratio));
  T ratio=exp(log_ratio);return ratio/(1+ratio);
}

template<typename T>
KOKKOS_INLINE_FUNCTION
void ConfigurationSources(const T u[nvar], T eta, T f[10], T ell[3][3]) {
  BaseGeometry<T> geo; BuildBaseGeometry(u,geo);
  T alpha=u[W]*u[RHO], theta=u[B]+u[B+4]+u[B+8];
  T fg[3][3]={}, wave[3][3]={}, td[3][3]={};
  for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
    fg[i][j]=-2*alpha*geo.curvature[i][j]-(2.0/3.0)*theta*geo.metric[i][j];
    for (int k=0;k<3;++k)
      fg[i][j]+=u[B+3*i+k]*geo.metric[k][j]+geo.metric[i][k]*u[B+3*j+k];
  }
  for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
    for (int r=0;r<3;++r) for (int s=0;s<3;++s)
      wave[i][j]+=geo.inverse_tri[i][r]*fg[r][s]*geo.inverse_tri[j][s];
    ell[i][j]=i<j ? T(0) : (i==j ? wave[i][j]/2 : wave[i][j]);
  }
  for (int i=0;i<3;++i) for (int j=0;j<3;++j)
    for (int k=0;k<3;++k) td[i][j]+=geo.tri[i][k]*ell[k][j];
  f[0]=u[W]*(alpha*u[K]-theta)/3;
  f[1]=-2*alpha*u[K];
  f[2]=ell[0][0]; f[3]=ell[1][1];
  f[4]=td[1][0]; f[5]=td[2][0]; f[6]=td[2][1];
  T sigma=GaugePlateau(alpha*u[W]*u[W]);
  for (int i=0;i<3;++i) {
    T gamma=0, v=0;
    for (int j=0;j<3;++j) {
      v+=geo.inverse_metric[i][j]*(alpha*alpha*u[W]*u[P+j]
          -alpha*u[W]*u[W]*u[LAPSE_GRADIENT+j]);
      for (int k=0;k<3;++k) for (int d=0;d<3;++d)
        gamma+=geo.inverse_metric[j][k]*geo.inverse_metric[i][d]
            *(geo.gradient[j][d][k]+geo.gradient[k][d][j]-geo.gradient[d][j][k])/2;
    }
    f[7+i]=gamma-u[Z+i]-eta*u[BETA+i]+sigma*v;
  }
}

KOKKOS_INLINE_FUNCTION
void DifferentiateConfiguration(const double u[nvar], const double du[nvar],
    double eta, double df[10], double dell[3][3]) {
  Jet state[nvar], f[10], ell[3][3];
  for (int n=0;n<nvar;++n) state[n]={u[n],du[n]};
  ConfigurationSources(state,Jet(eta),f,ell);
  for (int n=0;n<10;++n) df[n]=f[n].derivative;
  for (int i=0;i<3;++i) for (int j=0;j<3;++j) dell[i][j]=ell[i][j].derivative;
}
}  // namespace pc_gh::intrinsic
#endif
