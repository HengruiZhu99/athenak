#ifndef SCRATCH_NATIVE_Q_NULL_INJECTION_HPP_
#define SCRATCH_NATIVE_Q_NULL_INJECTION_HPP_
#include "q_null_feedback.hpp"
namespace z4c {namespace hyperboloidal {
template<class T> GaugeRHSParts<T> ResearchNativeQGauge(
 const LayerPoint<T>&p,const Z4cJet<T>&u,const LayerGaugeParameters&g){
 return qnf::Gauge(p,u,g,{.85,.95,5,true});
}
template<class T> bool ResearchNativeQAssemble(
 const GaugeRHSParts<T>&parts,T omega,GaugeRHS<T>&rhs){
 return qnf::Assemble(parts,omega,rhs);
}
}}
#define InteriorLayerGauge ResearchNativeQGauge
#define AssembleGaugeInterior ResearchNativeQAssemble
#endif
