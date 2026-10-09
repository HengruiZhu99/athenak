// Mathematical tangent conversion only. No production/reference/PDE includes.
#ifndef RESEARCH_TOTAL_J_REFERENCE_CONVERSION_HPP_
#define RESEARCH_TOTAL_J_REFERENCE_CONVERSION_HPP_
#include "total_j_basis.hpp"
namespace totalj {
template<class T> Jet<T> Constant(T a) { Jet<T> j; j.value=a; return j; }
template<class T> Jet<T> Add(const Jet<T>&a,const Jet<T>&b) {
  Jet<T> c; c.value=a.value+b.value;
  for(int i=0;i<3;++i) { c.d[i]=a.d[i]+b.d[i];
    for(int k=0;k<3;++k) c.dd[i][k]=a.dd[i][k]+b.dd[i][k]; }
  return c;
}
template<class T> Jet<T> Scale(const Jet<T>&a,T b) {
  Jet<T> c; c.value=a.value*b;
  for(int i=0;i<3;++i) { c.d[i]=a.d[i]*b;
    for(int k=0;k<3;++k) c.dd[i][k]=a.dd[i][k]*b; }
  return c;
}
template<class T> Jet<T> Sub(const Jet<T>&a,const Jet<T>&b) { return Add(a,Scale(b,T(-1))); }
template<class T> Jet<T> Multiply(const Jet<T>&a,const Jet<T>&b) {
  Jet<T> c; c.value=a.value*b.value;
  for(int i=0;i<3;++i) { c.d[i]=a.d[i]*b.value+a.value*b.d[i];
    for(int k=0;k<3;++k) c.dd[i][k]=a.dd[i][k]*b.value+a.d[i]*b.d[k]
      +a.d[k]*b.d[i]+a.value*b.dd[i][k]; }
  return c;
}
template<class T> Jet<T> Inverse(const Jet<T>&a) {
  Jet<T> c; const T a2=a.value*a.value,a3=a2*a.value; c.value=T(1)/a.value;
  for(int i=0;i<3;++i) { c.d[i]=-a.d[i]/a2;
    for(int k=0;k<3;++k) c.dd[i][k]=T(2)*a.d[i]*a.d[k]/a3-a.dd[i][k]/a2; }
  return c;
}
template<class T> using MatrixJet=std::array<std::array<Jet<T>,3>,3>;
template<class T> MatrixJet<T> MatrixInverse(const MatrixJet<T>&a) {
  MatrixJet<T> cofactor{},out{};
  for(int i=0;i<3;++i) for(int k=0;k<3;++k) {
    const int r0=(i+1)%3,r1=(i+2)%3,c0=(k+1)%3,c1=(k+2)%3;
    cofactor[i][k]=Sub(Multiply(a[r0][c0],a[r1][c1]),Multiply(a[r0][c1],a[r1][c0]));
  }
  Jet<T> det{}; for(int k=0;k<3;++k) det=Add(det,Multiply(a[0][k],cofactor[0][k]));
  const auto invdet=Inverse(det);
  for(int i=0;i<3;++i) for(int k=0;k<3;++k) out[i][k]=Multiply(cofactor[k][i],invdet);
  return out;
}
template<class T> Jet<T> Contract(const MatrixJet<T>&a,const MatrixJet<T>&b) {
  Jet<T> out; for(int i=0;i<3;++i) for(int j=0;j<3;++j) out=Add(out,Multiply(a[i][j],b[i][j]));
  return out;
}
template<class T> struct MetricTangent {
  Jet<T> chi; MatrixJet<T> metric,reference_metric,reference_inverse;
};
// bar_gamma is the Penrose spatial metric, gtilde=chi*bar_gamma.
// All spatial coefficient derivatives follow through the jet products/inverse.
template<class T> MetricTangent<T> ConvertMetric(const MatrixJet<T>&bar_gamma,
                                               const Jet<T>&chi,
                                               const MatrixJet<T>&delta_bar_gamma) {
  MetricTangent<T> out;
  out.chi=Scale(Multiply(chi,Contract(MatrixInverse(bar_gamma),delta_bar_gamma)),T(-1)/T(3));
  for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
    out.reference_metric[i][j]=Multiply(chi,bar_gamma[i][j]);
    out.metric[i][j]=Add(Multiply(chi,delta_bar_gamma[i][j]),Multiply(bar_gamma[i][j],out.chi));
  }
  out.reference_inverse=MatrixInverse(out.reference_metric);
  return out;
}
// Aref is covariant Atilde. independent_A may be a Euclidean STF harmonic.
// Its reference-TF projection is supplemented by the nonzero linearized trace
// required by varying the inverse metric. This function propagates supplied
// coefficient jets; consumers needing only first A derivatives need not use dd.
template<class T> MatrixJet<T> ConvertA(const MetricTangent<T>&metric,
                                      const MatrixJet<T>&Aref,
                                      const MatrixJet<T>&independent_A) {
  MatrixJet<T> raised{},out{};
  for(int i=0;i<3;++i) for(int j=0;j<3;++j)
    for(int k=0;k<3;++k) for(int l=0;l<3;++l)
      raised[i][j]=Add(raised[i][j],Multiply(Multiply(metric.reference_inverse[i][k],
        metric.reference_inverse[j][l]),Aref[k][l]));
  const auto trace=Contract(metric.reference_inverse,independent_A);
  const auto required=Contract(raised,metric.metric);
  const auto isotropic=Scale(Sub(required,trace),T(1)/T(3));
  for(int i=0;i<3;++i) for(int j=0;j<3;++j)
    out[i][j]=Add(independent_A[i][j],Multiply(metric.reference_metric[i][j],isotropic));
  return out;
}
} // namespace totalj
#endif
