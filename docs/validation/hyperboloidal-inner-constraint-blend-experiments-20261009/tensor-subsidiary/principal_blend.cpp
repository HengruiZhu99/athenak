// Actual symbol extraction at the same prescribed radius as each kernel case.
#include "bulk_c1_additions.hpp"
double bulk_radius=0;
namespace z4c { namespace hyperboloidal {
template <typename T>
Z4cRHSParts<T> BulkPrincipalRHS(const Z4cJet<T>&u,const OmegaJet<T>&o,T k1,T k2) {
 auto base=ConformalRHS(u,o,k1,k2);
 const auto add=BulkC1Additions(u,o,T(bulk_radius));
 base.regular.trace+=add.regular.trace;
 base.regular.theta+=add.regular.theta;base.pole.theta+=add.pole.theta;
 for(int i=0;i<3;++i){base.regular.lambda[i]+=add.regular.lambda[i];
  // Positive-Omega symbol extractor only, never a native assembly convention.
  base.pole.lambda[i]+=add.pole.lambda[i]+add.double_pole.lambda[i]/o.omega;
  for(int j=0;j<3;++j)base.pole.a[i][j]+=add.pole.a[i][j];}
 return base;
}
}}
#define ConformalRHS BulkPrincipalRHS
#include "kernel_symbol_copy.cpp"
