//========================================================================================
// AthenaK astrophysical plasma code, 3-clause BSD License (LICENSE)
// Intrinsic chart geometry for the explicitly separate 50-field candidate.
//========================================================================================
#ifndef PC_GH_INTRINSIC_GEOMETRY_HPP_
#define PC_GH_INTRINSIC_GEOMETRY_HPP_
#include <cmath>
#include <Kokkos_Core.hpp>

namespace pc_gh::intrinsic {
// This ABI is not a 55-field PC-GH restart. Evolution integration is separate.
constexpr int layout_version = 1;
constexpr int nvar = 50;
enum Index : int {
  W=0, RHO=1, CHART=2, BETA=7, K=10, AHAT=11, Z=16, C=19,
  P=20, LAPSE_GRADIENT=23, S=26, B=41
};
static_assert(B+9 == nvar);

template<typename T>
struct BaseGeometry {
  T tri[3][3], inverse_tri[3][3], metric[3][3], inverse_metric[3][3];
  T ahat[3][3], curvature[3][3], gradient[3][3][3];
};

template<typename T>
struct Geometry : BaseGeometry<T> {
  T jacobian[5][3][3], hessian[5][5][3][3];
};

template<typename T>
KOKKOS_INLINE_FUNCTION
void SymmetricTraceFree(const T *v, T a[3][3]) {
  a[0][0]=v[0]; a[0][1]=a[1][0]=v[1]; a[0][2]=a[2][0]=v[2];
  a[1][1]=v[3]; a[1][2]=a[2][1]=v[4]; a[2][2]=-v[0]-v[3];
}

// Geometry needed by the complete RHS, without materializing J/Hessian arrays.
// Q is the directional differential of T*T^T at independent S. Applying Jet
// to this function retains both true chart derivatives and independent dS.
template<typename T>
KOKKOS_INLINE_FUNCTION
void BuildBaseGeometry(const T u[nvar], BaseGeometry<T> &o) {
  using std::exp;
  const T a=u[CHART], c=u[CHART+1], b=u[CHART+2];
  const T d=u[CHART+3], e=u[CHART+4];
  for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
    o.tri[i][j]=0; o.inverse_tri[i][j]=0;
  }
  o.tri[0][0]=exp(a); o.tri[1][1]=exp(c); o.tri[2][2]=exp(-a-c);
  o.tri[1][0]=b; o.tri[2][0]=d; o.tri[2][1]=e;
  auto &it=o.inverse_tri;
  it[0][0]=exp(-a); it[1][1]=exp(-c); it[2][2]=exp(a+c);
  it[1][0]=-b*it[0][0]*it[1][1]; it[2][1]=-e*it[1][1]*it[2][2];
  it[2][0]=(b*e*it[1][1]-d)*it[0][0]*it[2][2];
  SymmetricTraceFree(u+AHAT,o.ahat);
  for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
    o.metric[i][j]=o.inverse_metric[i][j]=o.curvature[i][j]=0;
    for (int r=0;r<3;++r) {
      o.metric[i][j]+=o.tri[i][r]*o.tri[j][r];
      o.inverse_metric[i][j]+=it[r][i]*it[r][j];
      for (int t=0;t<3;++t) o.curvature[i][j]+=o.tri[i][r]*o.ahat[r][t]*o.tri[j][t];
    }
  }
  for (int k=0;k<3;++k) {
    T dt[3][3]={};
    dt[0][0]=o.tri[0][0]*u[S+5*k]; dt[1][1]=o.tri[1][1]*u[S+5*k+1];
    dt[2][2]=-o.tri[2][2]*(u[S+5*k]+u[S+5*k+1]);
    dt[1][0]=u[S+5*k+2]; dt[2][0]=u[S+5*k+3]; dt[2][1]=u[S+5*k+4];
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
      o.gradient[k][i][j]=0;
      for (int r=0;r<3;++r) o.gradient[k][i][j]+=dt[i][r]*o.tri[j][r]+o.tri[i][r]*dt[j][r];
    }
  }
}

template<typename T>
KOKKOS_INLINE_FUNCTION
void BuildGeometry(const T u[nvar], Geometry<T> &o) {
  using std::exp;
  const T a=u[CHART], c=u[CHART+1], b=u[CHART+2];
  const T d=u[CHART+3], e=u[CHART+4];
  T dt[5][3][3]={}, ddt[5][5][3][3]={};
  for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
    o.tri[i][j]=0; o.inverse_tri[i][j]=0;
  }
  o.tri[0][0]=exp(a); o.tri[1][1]=exp(c); o.tri[2][2]=exp(-a-c);
  o.tri[1][0]=b; o.tri[2][0]=d; o.tri[2][1]=e;
  auto &it=o.inverse_tri;
  it[0][0]=exp(-a); it[1][1]=exp(-c); it[2][2]=exp(a+c);
  it[1][0]=-b*it[0][0]*it[1][1];
  it[2][1]=-e*it[1][1]*it[2][2];
  it[2][0]=(b*e*it[1][1]-d)*it[0][0]*it[2][2];
  dt[0][0][0]=o.tri[0][0]; dt[0][2][2]=-o.tri[2][2];
  dt[1][1][1]=o.tri[1][1]; dt[1][2][2]=-o.tri[2][2];
  dt[2][1][0]=1; dt[3][2][0]=1; dt[4][2][1]=1;
  ddt[0][0][0][0]=o.tri[0][0]; ddt[1][1][1][1]=o.tri[1][1];
  ddt[0][0][2][2]=ddt[0][1][2][2]=ddt[1][0][2][2]
      =ddt[1][1][2][2]=o.tri[2][2];
  SymmetricTraceFree(u+AHAT,o.ahat);
  for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
    o.metric[i][j]=o.inverse_metric[i][j]=o.curvature[i][j]=0;
    for (int r=0;r<3;++r) {
      o.metric[i][j]+=o.tri[i][r]*o.tri[j][r];
      o.inverse_metric[i][j]+=it[r][i]*it[r][j];
      for (int s=0;s<3;++s)
        o.curvature[i][j]+=o.tri[i][r]*o.ahat[r][s]*o.tri[j][s];
    }
    for (int q=0;q<5;++q) {
      o.jacobian[q][i][j]=0;
      for (int r=0;r<3;++r)
        o.jacobian[q][i][j]+=dt[q][i][r]*o.tri[j][r]+o.tri[i][r]*dt[q][j][r];
      for (int t=0;t<5;++t) {
        o.hessian[q][t][i][j]=0;
        for (int r=0;r<3;++r)
          o.hessian[q][t][i][j]+=ddt[q][t][i][r]*o.tri[j][r]
              +o.tri[i][r]*ddt[q][t][j][r]
              +dt[q][i][r]*dt[t][j][r]+dt[t][i][r]*dt[q][j][r];
      }
    }
    for (int k=0;k<3;++k) {
      o.gradient[k][i][j]=0;
      for (int q=0;q<5;++q) o.gradient[k][i][j]+=o.jacobian[q][i][j]*u[S+5*k+q];
    }
  }
}

// Invert the differential of the chart: X=J*v -> v. X must lie in the
// determinant-one tangent space. Validation is the caller's responsibility.
template<typename T>
KOKKOS_INLINE_FUNCTION
void ChartTangentInverse(const Geometry<T> &g, const T x[3][3], T v[5]) {
  T w[3][3]={}, ell[3][3]={}, td[3][3]={};
  for (int i=0;i<3;++i) for (int j=0;j<3;++j)
    for (int r=0;r<3;++r) for (int s=0;s<3;++s)
      w[i][j]+=g.inverse_tri[i][r]*x[r][s]*g.inverse_tri[j][s];
  for (int i=0;i<3;++i) for (int j=0;j<=i;++j)
    ell[i][j]=(i==j ? T(0.5) : T(1))*w[i][j];
  for (int i=0;i<3;++i) for (int j=0;j<3;++j)
    for (int r=0;r<3;++r) td[i][j]+=g.tri[i][r]*ell[r][j];
  v[0]=ell[0][0]; v[1]=ell[1][1];
  v[2]=td[1][0]; v[3]=td[2][0]; v[4]=td[2][1];
}
}  // namespace pc_gh::intrinsic
#endif
