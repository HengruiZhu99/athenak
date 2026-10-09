// Scratch only: repeat the actual versioned principal extraction with C1 parts.
#include "z4c/hyperboloidal/layer_gauge.hpp"
#include "c1_additions.hpp"
namespace z4c { namespace hyperboloidal {
template <typename T>
Z4cRHSParts<T> C1PrincipalRHS(const Z4cJet<T>&u,const OmegaJet<T>&o,T k1,T k2) {
  auto base=ConformalRHS(u,o,k1,k2);
  const auto add=TensorC1Additions(u,o,T(1),true);
  // Interior-only temporary representation for this positive-Omega symbol test.
  // The scratch physical audit keeps regular/pole/double-pole separately.
  base.regular.trace+=add.regular.trace;
  base.regular.theta+=add.regular.theta;base.pole.theta+=add.pole.theta;
  for(int i=0;i<3;++i){base.regular.lambda[i]+=add.regular.lambda[i];
    base.pole.lambda[i]+=add.pole.lambda[i]+add.double_pole.lambda[i]/o.omega;
    for(int j=0;j<3;++j)base.pole.a[i][j]+=add.pole.a[i][j];}
  return base;
}
}}
#define ConformalRHS C1PrincipalRHS
#include "../../../tst/hyperboloidal/kernel_symbol.cpp"
