// Same nonlinear map in scalar-generic form, with an SPD ambient AD extension.
#pragma once
#include "goal1_native_ad.hpp"
#include <array>
#include "z4c/hyperboloidal/conformal_rhs.hpp"
#include "z4c/hyperboloidal/layer_gauge.hpp"
namespace cubed_sphere {namespace typed_momentum {
namespace h=z4c::hyperboloidal;
template<class T>bool MetricVelocity(const h::Z4cJet<T>&u,const T V[3][3],T v[5]){
 if(!(u.metric.g[0][0]>0))return false;const auto l00=Root(u.metric.g[0][0]),l10=u.metric.g[1][0]/l00,l20=u.metric.g[2][0]/l00;const auto s=u.metric.g[1][1]-l10*l10;if(!(s>0))return false;const auto l11=Root(s),l21=(u.metric.g[2][1]-l10*l20)/l11;
 const auto a=V[0][0]/(2*l00);v[0]=a/l00;v[2]=(V[1][0]-l10*a)/l00;v[3]=(V[2][0]-l20*a)/l00;const auto b=(V[1][1]-2*l10*v[2])/(2*l11);v[1]=b/l11;v[4]=(V[2][1]-v[2]*l20-l10*v[3]-l21*b)/l11;return true;
}
template<class T>bool MetricFromVelocity(const h::Z4cJet<T>&u,const T v[5],T V[3][3]){
 const auto gi=h::Geometry(u.metric);if(!gi.valid)return false;const auto l00=Root(u.metric.g[0][0]),l10=u.metric.g[1][0]/l00,l20=u.metric.g[2][0]/l00,l11=Root(u.metric.g[1][1]-l10*l10),l21=(u.metric.g[2][1]-l10*l20)/l11;const auto a=l00*v[0],b=l11*v[1];
 V[0][0]=2*l00*a;V[0][1]=V[1][0]=a*l10+l00*v[2];V[0][2]=V[2][0]=a*l20+l00*v[3];V[1][1]=2*l10*v[2]+2*l11*b;V[1][2]=V[2][1]=v[2]*l20+l10*v[3]+b*l21+l11*v[4];
 V[2][2]=-(gi.inverse[0][0]*V[0][0]+2*gi.inverse[0][1]*V[0][1]+2*gi.inverse[0][2]*V[0][2]+gi.inverse[1][1]*V[1][1]+2*gi.inverse[1][2]*V[1][2])/gi.inverse[2][2];return true;
}
template<class T>bool Velocity(const h::Z4cJet<T>&u,const h::Z4cRHS<T>&f,const h::GaugeRHS<T>&g,std::array<T,10>&q){
 T m[5];if(!MetricVelocity(u,f.metric,m))return false;q[0]=f.chi;q[6]=g.alpha;for(int i=0;i<5;++i)q[1+i]=m[i];for(int i=0;i<3;++i)q[7+i]=g.beta[i];return true;
}
template<class T>bool Momentum(const h::Z4cJet<T>&u,const h::Z4cJet<T>&ref,T omega,std::array<T,10>&pi){
 if(!(omega>0)||!(u.alpha.value>0)||!(u.chi.value>0))return false;const auto g=h::Geometry(u.metric);if(!g.valid)return false;T div=0;for(int i=0;i<3;++i)div+=u.beta.d[i][i]-ref.beta.d[i][i];const auto a=u.alpha.value,chi=u.chi.value;
 pi[0]=-T(2./3.)*chi*div+T(2./3.)*a*chi*((u.trace.value-ref.trace.value)+2*(u.theta.value-ref.theta.value))/omega;
 T m[3][3]{},v[5];for(int i=0;i<3;++i)for(int j=0;j<3;++j){m[i][j]=-2*a*(u.a.k[i][j]-ref.a.k[i][j])-T(2./3.)*u.metric.g[i][j]*div;for(int k=0;k<3;++k)m[i][j]+=u.metric.g[k][i]*(u.beta.d[j][k]-ref.beta.d[j][k])+u.metric.g[k][j]*(u.beta.d[i][k]-ref.beta.d[i][k]);}
 if(!MetricVelocity(u,m,v))return false;for(int i=0;i<5;++i)pi[1+i]=v[i];pi[6]=-a*a*(u.trace.value-ref.trace.value)/omega;
 for(int i=0;i<3;++i){pi[7+i]=a*a*chi*(u.lambda.value[i]-ref.lambda.value[i]);for(int j=0;j<3;++j)pi[7+i]+=g.inverse[i][j]*(a*a*(u.chi.d[j]-ref.chi.d[j])/2-a*chi*(u.alpha.d[j]-ref.alpha.d[j]));}return true;
}
template<class T>bool Reconstruct(h::Z4cJet<T>&u,const h::Z4cJet<T>&ref,T omega,const std::array<T,10>&pi){
 if(!(omega>0)||!(u.alpha.value>0)||!(u.chi.value>0))return false;const auto g=h::Geometry(u.metric);if(!g.valid)return false;const auto a=u.alpha.value,chi=u.chi.value;T div=0;for(int i=0;i<3;++i)div+=u.beta.d[i][i]-ref.beta.d[i][i];const auto dp=-omega*pi[6]/(a*a);u.trace.value=ref.trace.value+dp;u.theta.value=ref.theta.value+(3*omega*(pi[0]+T(2./3.)*chi*div)/(2*a*chi)-dp)/2;
 T source[3][3]{},s[5],v[5],A[3][3];for(int i=0;i<3;++i)for(int j=0;j<3;++j){source[i][j]=2*a*ref.a.k[i][j]-T(2./3.)*u.metric.g[i][j]*div;for(int k=0;k<3;++k)source[i][j]+=u.metric.g[k][i]*(u.beta.d[j][k]-ref.beta.d[j][k])+u.metric.g[k][j]*(u.beta.d[i][k]-ref.beta.d[i][k]);}
 if(!MetricVelocity(u,source,s))return false;for(int i=0;i<5;++i)v[i]=(s[i]-pi[1+i])/(2*a);if(!MetricFromVelocity(u,v,A))return false;for(int i=0;i<3;++i)for(int j=0;j<3;++j)u.a.k[i][j]=A[i][j];
 for(int i=0;i<3;++i){T gradient=0;for(int j=0;j<3;++j)gradient+=g.inverse[i][j]*(a*a*(u.chi.d[j]-ref.chi.d[j])/2-a*chi*(u.alpha.d[j]-ref.alpha.d[j]));u.lambda.value[i]=ref.lambda.value[i]+(pi[7+i]-gradient)/(a*a*chi);}return true;
}
}}
