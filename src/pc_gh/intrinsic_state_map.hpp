// Finite-radius state conversion. No restart reader invokes this helper.
#ifndef PC_GH_INTRINSIC_STATE_MAP_HPP_
#define PC_GH_INTRINSIC_STATE_MAP_HPP_
#include "pc_gh/intrinsic_geometry.hpp"
namespace pc_gh::intrinsic {
KOKKOS_INLINE_FUNCTION constexpr int TensorI(int n) { return n<3 ? 0 : (n<5 ? 1 : 2); }
KOKKOS_INLINE_FUNCTION constexpr int TensorJ(int n) { return n==0 ? 0 : (n==1 || n==3 ? 1 : 2); }

// The legacy tensor ABI is fixed here explicitly; the compiled fixture asserts
// these offsets against PcGh::Index. Old L is twice the intrinsic lapse gradient.
template<typename T>
KOKKOS_INLINE_FUNCTION
void ToLegacy(const T u[nvar], T old[55]) {
  Geometry<T> g; BuildGeometry(u,g);
  old[0]=u[W]; old[7]=u[K]; old[17]=u[C]; old[18]=u[RHO];
  for (int n=0;n<6;++n) {
    const int i=TensorI(n),j=TensorJ(n);
    old[1+n]=g.metric[i][j]; old[8+n]=g.curvature[i][j];
    for (int k=0;k<3;++k) old[25+6*k+n]=g.gradient[k][i][j];
  }
  for (int i=0;i<3;++i) {
    old[14+i]=u[Z+i]; old[19+i]=u[BETA+i]; old[22+i]=u[P+i];
    old[43+i]=2*u[LAPSE_GRADIENT+i];
  }
  for (int n=0;n<9;++n) old[46+n]=u[B+n];
}

template<typename T>
KOKKOS_INLINE_FUNCTION
bool FromLegacy(const T old[55], T u[nvar], T tolerance) {
  using std::abs; using std::sqrt; using std::log; using std::isfinite;
  for (int n=0;n<55;++n) if (!isfinite(old[n])) return false;
  if (!(old[0]>0 && old[18]>0)) return false;
  T metric[3][3]={}, a[3][3]={}, q[3][3][3]={};
  for (int n=0;n<6;++n) {
    const int i=TensorI(n),j=TensorJ(n);
    metric[i][j]=metric[j][i]=old[1+n]; a[i][j]=a[j][i]=old[8+n];
    for (int k=0;k<3;++k) q[k][i][j]=q[k][j][i]=old[25+6*k+n];
  }
  if (!(metric[0][0]>0)) return false;
  const T t00=sqrt(metric[0][0]);
  const T t10=metric[1][0]/t00, t20=metric[2][0]/t00;
  const T minor=metric[1][1]-t10*t10;
  if (!(minor>0)) return false;
  const T t11=sqrt(minor), t21=(metric[2][1]-t20*t10)/t11;
  const T last=metric[2][2]-t20*t20-t21*t21;
  if (!(last>0) || abs(metric[0][0]*minor*last-1)>tolerance) return false;
  T candidate[nvar]={};
  candidate[W]=old[0]; candidate[RHO]=old[18]; candidate[K]=old[7]; candidate[C]=old[17];
  candidate[CHART]=log(t00); candidate[CHART+1]=log(t11);
  candidate[CHART+2]=t10; candidate[CHART+3]=t20; candidate[CHART+4]=t21;
  Geometry<T> g; BuildGeometry(candidate,g);
  T ah[3][3]={};
  for (int i=0;i<3;++i) for (int j=0;j<3;++j)
    for (int r=0;r<3;++r) for (int s=0;s<3;++s)
      ah[i][j]+=g.inverse_tri[i][r]*a[r][s]*g.inverse_tri[j][s];
  T scale=1;
  for (int i=0;i<3;++i) for (int j=0;j<3;++j) scale+=abs(ah[i][j]);
  if (abs(ah[0][0]+ah[1][1]+ah[2][2])>tolerance*scale) return false;
  for (int n=0;n<5;++n) candidate[AHAT+n]=ah[TensorI(n)][TensorJ(n)];
  for (int k=0;k<3;++k) {
    T trace=0, trace_scale=1;
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
      T term=g.inverse_metric[i][j]*q[k][i][j]; trace+=term; trace_scale+=abs(term);
    }
    if (abs(trace)>tolerance*trace_scale) return false;
    ChartTangentInverse(g,q[k],candidate+S+5*k);
    candidate[Z+k]=old[14+k]; candidate[BETA+k]=old[19+k]; candidate[P+k]=old[22+k];
    candidate[LAPSE_GRADIENT+k]=old[43+k]/2;
  }
  for (int n=0;n<9;++n) candidate[B+n]=old[46+n];
  for (int n=0;n<nvar;++n) if (!isfinite(candidate[n])) return false;
  for (int n=0;n<nvar;++n) u[n]=candidate[n];
  return true;
}
}  // namespace pc_gh::intrinsic
#endif
