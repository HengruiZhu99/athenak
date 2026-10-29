// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_POSITIVE_VARIABLES_HPP_
#define Z4C_HYPERBOLOIDAL_POSITIVE_VARIABLES_HPP_
#include "athena.hpp"
namespace z4c::hyperboloidal {
// Same fixed AF reference factors as the 1D puncture chart. These are numerical
// variable definitions, not an analytic solution subtracted from the PDE.
KOKKOS_INLINE_FUNCTION Real PositiveReference(Real r2,int field,bool puncture){
  const Real A=(1+r2)/2;
  if(!puncture)return field==0?1:A;
  const Real omega=(1-r2)/2,o4=omega*omega*omega*omega;
  return field==0?r2*r2/(r2*r2+o4):A*r2/(r2+o4);
}
KOKKOS_INLINE_FUNCTION Real PositiveLog(Real value,Real reference){return log(value/reference);}
// Physical-lapse floor, as in the donor: conformal alpha >= floor*Omega.
// Never rescue nonfinite logarithms. Chi has no floor.
KOKKOS_INLINE_FUNCTION Real MaterializePositive(Real logarithm,Real A,int field,
                                               Real physical_floor,bool &floored){
  floored=false;
  if(field==18&&physical_floor>0&&A<1&&Kokkos::isfinite(logarithm)){
    const Real lower=log(physical_floor*(1-A));
    if(logarithm<lower){logarithm=lower;floored=true;}
  }
  return exp(logarithm)-(field==18?A:0);
}
}
#endif
