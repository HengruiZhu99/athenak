#include <iostream>
#include <iomanip>
#define KOKKOS_INLINE_FUNCTION inline
#include "z4c/covariant_sources.hpp"
int main() {
  int n;
  std::cin >> n;
  std::cout << std::setprecision(17);
  for (int p=0; p<n; ++p) {
    double alpha,chi,K,Theta,g[3][3],gi[3][3],A[3][3];
    double Q[3],da[3],dc[3],dg[3][3][3],db[3][3];
    std::cin >> alpha >> chi >> K >> Theta;
    for (auto &a:g) for (auto &b:a) std::cin >> b;
    for (auto &a:gi) for (auto &b:a) std::cin >> b;
    for (auto &a:A) for (auto &b:a) std::cin >> b;
    for (auto &a:Q) std::cin >> a;
    for (auto &a:da) std::cin >> a;
    for (auto &a:dc) std::cin >> a;
    for (auto &a:dg) for (auto &b:a) for (auto &c:b) std::cin >> c;
    for (auto &a:db) for (auto &b:a) std::cin >> b;
    if (!std::cin) return 1;
    const auto out=z4c::BuildCovariantConstraintSource(alpha,chi,K,Theta,
        g,gi,A,Q,da,dc,dg,db);
    std::cout << out.khat << ' ' << out.theta;
    for (const auto a:out.gamma) std::cout << ' ' << a;
    for (const auto &a:out.A) for (const auto b:a) std::cout << ' ' << b;
    std::cout << '\n';
  }
}
