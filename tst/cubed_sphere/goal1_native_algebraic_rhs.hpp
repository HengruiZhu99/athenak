// Native algebraic-definition consistency only. No physical constraint reset.
#pragma once
#include "athena.hpp"
#include "z4c/hyperboloidal/conformal_rhs.hpp"
namespace cubed_sphere {namespace native_algebraic_rhs {
namespace h=z4c::hyperboloidal;
struct Diagnostics {bool valid=false;Real metric_before=0,A_before=0,metric_after=0,A_after=0,metric_change=0,A_change=0,raw_metric_A_rhs=0;};
KOKKOS_INLINE_FUNCTION Diagnostics Tangent(const h::MetricJet<Real>&metric,const h::CurvatureJet<Real>&A,h::Z4cRHS<Real>&rhs){
 Diagnostics out;auto g=h::Geometry(metric);if(!g.valid)return out;Real up[3][3]{};
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){for(int k=0;k<3;++k)for(int l=0;l<3;++l)up[i][j]+=g.inverse[i][k]*A.k[k][l]*g.inverse[j][l];out.metric_before+=g.inverse[i][j]*rhs.metric[i][j];out.A_before+=g.inverse[i][j]*rhs.a[i][j]-up[i][j]*rhs.metric[i][j];out.raw_metric_A_rhs=Kokkos::fmax(out.raw_metric_A_rhs,Kokkos::fmax(fabs(rhs.metric[i][j]),fabs(rhs.a[i][j])));}
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){Real change=metric.g[i][j]*out.metric_before/3;rhs.metric[i][j]-=change;out.metric_change=Kokkos::fmax(out.metric_change,fabs(change));}
 Real rate=0;for(int i=0;i<3;++i)for(int j=0;j<3;++j)rate+=g.inverse[i][j]*rhs.a[i][j]-up[i][j]*rhs.metric[i][j];
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){Real change=metric.g[i][j]*rate/3;rhs.a[i][j]-=change;out.A_change=Kokkos::fmax(out.A_change,fabs(change));out.metric_after+=g.inverse[i][j]*rhs.metric[i][j];out.A_after+=g.inverse[i][j]*rhs.a[i][j]-up[i][j]*rhs.metric[i][j];}
 out.valid=Kokkos::isfinite(out.metric_after)&&Kokkos::isfinite(out.A_after);return out;
}
}}
