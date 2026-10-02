// Standalone: c++ -std=c++17 -O2 test_fastflow_harmonics.cpp -o test_harmonics
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#define KOKKOS_INLINE_FUNCTION inline
#include "../../../src/z4c/fastflow_harmonics.hpp"

int main() {
  constexpr double pi=3.14159265358979323846;
  double addition=0,first=0,second=0;
  for (double th:{.13,.71,1.23,2.1,3.01}) for (int l:{0,1,2,8,16,32,48,64}) {
    double sum=0;
    for (int m=0;m<=l;++m) {
      auto y=StableFastFlowHarmonic(l,m,th,.37);
      sum+=(m?2:1)*(y.real*y.real+y.imag*y.imag);
    }
    addition=std::max(addition,std::abs(sum/((2*l+1)/(4*pi))-1));
  }
  for (double th:{.71,1.23,2.1}) for (int l:{1,2,8,16,32,48}) for (int m:{0,1,l}) {
    const double step=.0002;
    auto y=StableFastFlowHarmonic(l,m,th,.37);
    auto mm=StableFastFlowHarmonic(l,m,th-2*step,.37),m1=StableFastFlowHarmonic(l,m,th-step,.37);
    auto p1=StableFastFlowHarmonic(l,m,th+step,.37),pp=StableFastFlowHarmonic(l,m,th+2*step,.37);
    double d=(mm.real-8*m1.real+8*p1.real-pp.real)/(12*step);
    double dd=(-pp.real+16*p1.real-30*y.real+16*m1.real-mm.real)/(12*step*step);
    first=std::max(first,std::abs(d-y.th_real)/std::max(1.0,std::abs(y.th_real)));
    second=std::max(second,std::abs(dd-y.th2_real)/std::max(1.0,std::abs(y.th2_real)));
  }
  // Low-degree analytic values independently fix phase and normalization.
  auto y10=StableFastFlowHarmonic(1,0,.71,.37),y11=StableFastFlowHarmonic(1,1,.71,.37);
  if (std::abs(y10.real-std::sqrt(3/(4*pi))*std::cos(.71))>1e-14 ||
      std::abs(y11.real+std::sqrt(3/(8*pi))*std::sin(.71)*std::cos(.37))>1e-14)
    throw std::runtime_error("Harmonic phase/normalization failed");
  std::cout << "addition theorem " << addition << " first FD " << first << " second FD " << second << '\n';
  if (addition>2e-12 || first>2e-8 || second>2e-7) throw std::runtime_error("Stable harmonic control failed");
}
