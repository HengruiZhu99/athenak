// Algebraically equivalent outer native RWM equations for identity analysis.
// This is not a gauge candidate or a numerical source adoption.
#pragma once
#include "goal1_native_endpoint.hpp"
namespace cubed_sphere {namespace native_outer_gauge_identity {
namespace h=z4c::hyperboloidal;
template<class T>h::GaugeRHSParts<T> Compact(const h::Z4cJet<T>&u,const h::LayerPoint<T>&p,const T x[3]){
 h::GaugeRHSParts<T>out{};auto g=h::Geometry(u.metric),b=h::Geometry(h::PenroseMetric(u.metric,u.chi));if(!g.valid||!b.valid)return out;
 T a=u.alpha.value,C[3][3]{},trL=0,bx=0;for(int i=0;i<3;++i){bx+=u.beta.value[i]*x[i];for(int j=0;j<3;++j){C[i][j]=u.chi.value*g.inverse[i][j];if(i==j)trL+=a*a*C[i][j]-u.beta.value[i]*u.beta.value[j];}}
 T betaNu=0;for(int i=0;i<3;++i){betaNu+=u.beta.value[i]*p.domega[i];out.regular.alpha+=u.beta.value[i]*u.alpha.d[i];}
 out.pole.alpha=-a*(a*u.trace.value+betaNu+T(2)*trL/p.L);
 for(int i=0;i<3;++i){out.regular.beta[i]=a*a*b.contracted[i]+a*a*u.chi.value*(u.lambda.value[i]-g.contracted[i]);out.pole.beta[i]=T(4)*u.beta.value[i]*bx-T(2)*(p.omega*x[i]+u.beta.value[i])*trL/p.L;for(int j=0;j<3;++j){out.regular.beta[i]+=u.beta.value[j]*u.beta.d[j][i]-a*C[i][j]*u.alpha.d[j];out.pole.beta[i]-=T(8)*a*a*C[i][j]*x[j];}}
 out.valid=true;return out;
}
}}
