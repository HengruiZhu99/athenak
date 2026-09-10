#include <cmath>
#include <iostream>
#include "z4c/chi_truncation_sensor.hpp"
int main() {
  bool pass = true;
  Real u[7];
  for (int degree=0; degree<=4; ++degree) {
    for (int q=0;q<7;++q) u[q]=std::pow(Real(q-3),degree);
    pass &= z4c::ChiDerivativeTruncationError(u,1,1)<1e-12;
  }
  // Exact leading D1 error on x^5 at the origin: 4 h^4.
  for (int q=0;q<7;++q) u[q]=std::pow(Real(q-3),5);
  pass &= std::abs(z4c::ChiDerivativeTruncationError(u,1,1)-4)<1e-10;
  // Exact leading D2 error on x^6: 8 h^4. D1 proxy vanishes by parity.
  for (int q=0;q<7;++q) u[q]=std::pow(Real(q-3),6);
  pass &= std::abs(z4c::ChiDerivativeTruncationError(u,1,1)-8)<1e-10;
  // Nyquist is invisible to centered odd derivatives but not D2 error.
  for (int q=0;q<7;++q) u[q]=(q%2 ? -1 : 1);
  pass &= z4c::ChiDerivativeTruncationError(u,1,1)>.7;
  Real previous=0;
  for (int n: {32,64,128}) {
    const Real h=1.0/n;
    for (int q=0;q<7;++q) u[q]=std::sin(.7+(q-3)*h*3);
    const Real error=z4c::ChiDerivativeTruncationError(u,h,1);
    if (previous) pass &= previous/error>15.5 && previous/error<16.5;
    previous=error;
  }
  pass &= std::abs(z4c::ResolutionScaledChiErrorThreshold(1,64,128)-1.0/16)<1e-15;
  pass &= std::abs(z4c::ResolutionScaledChiErrorThreshold(1,64,256)-1.0/256)<1e-15;
  if (!pass) {std::cerr << "chi truncation sensor FAILED\n";return 1;}
  std::cout << "chi truncation sensor PASS\n";
}
