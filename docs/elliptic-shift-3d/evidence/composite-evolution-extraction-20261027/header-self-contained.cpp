#include "z4c/hyperboloidal/composite_evolution.hpp"
#include <type_traits>
using E=z4c::hyperboloidal::CompositeTensorEvolution;
static_assert(E::fields==22,"field layout");
static_assert(std::is_same<decltype(std::declval<E&>().RHS(std::declval<const E::V&>(),Real{})),E::V>::value,"native-owner RHS interface");
static_assert(std::is_same<decltype(std::declval<const E&>().Boundary(std::declval<const E::V&>())),z4c::hyperboloidal::CompositeShiftStage::Boundary>::value,"native-owner boundary interface");
int main(){return E::tensor_i[0]+E::tensor_j[0];}
