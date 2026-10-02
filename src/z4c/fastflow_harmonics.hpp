// Stable scalar harmonics and angular derivatives for FastFlow's GL nodes.
// Normalized associated-Legendre recurrence avoids factorial-sum cancellation.
#ifndef Z4C_FASTFLOW_HARMONICS_HPP_
#define Z4C_FASTFLOW_HARMONICS_HPP_
#include <cmath>

struct FastFlowHarmonic {
  double real, imag, th_real, th_imag, ph_real, ph_imag;
  double th2_real, th2_imag, ph2_real, ph2_imag, thph_real, thph_imag;
};
KOKKOS_INLINE_FUNCTION
FastFlowHarmonic StableFastFlowHarmonic(int l,int m,double theta,double phi) {
  const double st=std::sin(theta),ct=std::cos(theta);
  double p=1/std::sqrt(4*3.141592653589793238462643383279502884);
  for (int q=1;q<=m;++q) p*=-std::sqrt((2.0*q+1)/(2.0*q))*st;
  double prev=0;
  if (l>m) {
    prev=p;p=std::sqrt(2.0*m+3)*ct*p;
    for (int q=m+2;q<=l;++q) {
      const double den=double(q)*q-double(m)*m;
      const double a=std::sqrt((4.0*q*q-1)/den);
      const double b=std::sqrt((2.0*q+1)*((q-1.0)*(q-1.0)-double(m)*m)/((2.0*q-3)*den));
      const double next=a*ct*p-b*prev;prev=p;p=next;
    }
  }
  const double lower=l>m?std::sqrt((2.0*l+1)/(2.0*l-1)*(double(l)*l-double(m)*m))*prev:0;
  const double dp=(l*ct*p-lower)/st;
  const double d2p=-ct/st*dp-(double(l)*(l+1)-double(m)*m/(st*st))*p;
  const double c=std::cos(m*phi),s=std::sin(m*phi);
  return {p*c,p*s,dp*c,dp*s,-m*p*s,m*p*c,
          d2p*c,d2p*s,-m*m*p*c,-m*m*p*s,-m*dp*s,m*dp*c};
}
#endif  // Z4C_FASTFLOW_HARMONICS_HPP_
