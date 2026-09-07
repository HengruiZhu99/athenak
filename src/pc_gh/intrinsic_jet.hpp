// First directional automatic differentiation for complete composite sources.
#ifndef PC_GH_INTRINSIC_JET_HPP_
#define PC_GH_INTRINSIC_JET_HPP_
#include <cmath>
#include <Kokkos_Core.hpp>
namespace pc_gh::intrinsic {
struct Jet {
  double value, derivative;
  KOKKOS_INLINE_FUNCTION Jet(double v=0,double d=0):value(v),derivative(d) {}
  KOKKOS_INLINE_FUNCTION Jet &operator+=(Jet b) {
    value+=b.value; derivative+=b.derivative; return *this;
  }
};
KOKKOS_INLINE_FUNCTION Jet operator+(Jet a,Jet b) {
  return {a.value+b.value,a.derivative+b.derivative};
}
KOKKOS_INLINE_FUNCTION Jet operator-(Jet a,Jet b) {
  return {a.value-b.value,a.derivative-b.derivative};
}
KOKKOS_INLINE_FUNCTION Jet operator-(Jet a) {return {-a.value,-a.derivative};}
KOKKOS_INLINE_FUNCTION Jet operator*(Jet a,Jet b) {
  return {a.value*b.value,a.derivative*b.value+a.value*b.derivative};
}
KOKKOS_INLINE_FUNCTION Jet operator/(Jet a,Jet b) {
  double v=a.value/b.value;return {v,(a.derivative-v*b.derivative)/b.value};
}
KOKKOS_INLINE_FUNCTION Jet exp(Jet a) {
  double v=std::exp(a.value);return {v,v*a.derivative};
}
KOKKOS_INLINE_FUNCTION double ScalarValue(double a) {return a;}
KOKKOS_INLINE_FUNCTION double ScalarValue(Jet a) {return a.value;}
}  // namespace pc_gh::intrinsic
#endif
