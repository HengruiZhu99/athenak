#include "inner_conformal_trace.hpp"
#include "inner_lapse_advection.hpp"
namespace z4c {namespace hyperboloidal {
template<typename T> GaugeRHSParts<T> CandidatePrincipalGauge(
 const LayerPoint<T>&p,const Z4cJet<T>&u,const LayerGaugeParameters&g) {
 auto out=InteriorLayerGauge(p,u,g);
 out=ResearchInnerTraceGauge(p,u,g,out,COMBINED_ADVECTION);
 return out;
}
}}
#define InteriorLayerGauge CandidatePrincipalGauge
#include "kernel_symbol_copy.cpp"
