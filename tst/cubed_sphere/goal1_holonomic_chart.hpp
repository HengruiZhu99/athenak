// Holonomic positive det-one metric chart for an isolated Z4c reduction.
#pragma once
#include <array>
#include <cmath>
#include "mesh/cubed_sphere/map.hpp"
#include "z4c/hyperboloidal/conformal_rhs.hpp"
#include "z4c/hyperboloidal/layer_gauge.hpp"
namespace cubed_sphere {namespace holonomic_chart {
namespace h=z4c::hyperboloidal;
using Config=std::array<Jet,10>;
inline Jet Exp(const Jet&a){Jet e(std::exp(a.f));for(int i=0;i<3;++i){e.d[i]=e.f*a.d[i];for(int j=0;j<3;++j)e.h[i][j]=e.f*(a.h[i][j]+a.d[i]*a.d[j]);}return e;}
inline Jet Scalar(const h::ScalarJet<Real>&s){Jet a(s.value);for(int i=0;i<3;++i){a.d[i]=s.d[i];for(int j=0;j<3;++j)a.h[i][j]=s.dd[i][j];}return a;}
inline h::ScalarJet<Real> Scalar(const Jet&s){h::ScalarJet<Real>a{};a.value=s.f;for(int i=0;i<3;++i){a.d[i]=s.d[i];for(int j=0;j<3;++j)a.dd[i][j]=s.h[i][j];}return a;}
inline Jet Metric(const h::MetricJet<Real>&m,int i,int j){Jet a(m.g[i][j]);for(int d=0;d<3;++d){a.d[d]=m.dg[d][i][j];for(int e=0;e<3;++e)a.h[d][e]=m.ddg[d][e][i][j];}return a;}
inline Jet Determinant(const h::MetricJet<Real>&m){Jet g[3][3];for(int i=0;i<3;++i)for(int j=0;j<3;++j)g[i][j]=Metric(m,i,j);return g[0][0]*(g[1][1]*g[2][2]-g[1][2]*g[2][1])-g[0][1]*(g[1][0]*g[2][2]-g[1][2]*g[2][0])+g[0][2]*(g[1][0]*g[2][1]-g[1][1]*g[2][0]);}
inline bool Finite(const Jet&a){bool valid=std::isfinite(a.f);for(int i=0;i<3;++i){valid=valid&&std::isfinite(a.d[i]);for(int j=0;j<3;++j)valid=valid&&std::isfinite(a.h[i][j]);}return valid;}
// Only value-level algebraic admission here. Differential determinant defects
// must be reported by the caller; extraction never resets the input jet.
inline bool Encode(const h::Z4cJet<Real>&u,Config&q){
 if(!(u.chi.value>0)||!(u.alpha.value>0))return false;
 auto det=Determinant(u.metric);if(!(u.metric.g[0][0]>0)||!std::isfinite(det.f)||std::abs(det.f-1)>1e-10)return false;
 auto l00=Power(Metric(u.metric,0,0),.5),l10=Metric(u.metric,1,0)/l00,l20=Metric(u.metric,2,0)/l00;
 auto a11=Metric(u.metric,1,1)-l10*l10;if(!(a11.f>0))return false;
 auto l11=Power(a11,.5),l21=(Metric(u.metric,2,1)-l10*l20)/l11;
 q[0]=Scalar(u.chi);q[1]=Log(l00);q[2]=Log(l11);q[3]=l10;q[4]=l20;q[5]=l21;q[6]=Scalar(u.alpha);
 for(int i=0;i<3;++i){q[7+i]=Jet(u.beta.value[i]);for(int d=0;d<3;++d){q[7+i].d[d]=u.beta.d[d][i];for(int e=0;e<3;++e)q[7+i].h[d][e]=u.beta.dd[d][e][i];}}
 for(const auto&a:q)if(!Finite(a))return false;return true;
}
// Reconstruct configuration fields only. Native momenta are left untouched.
inline bool Decode(const Config&q,h::Z4cJet<Real>&u){
 if(!(q[0].f>0)||!(q[6].f>0))return false;for(const auto&a:q)if(!Finite(a))return false;
 Jet L[3][3];L[0][0]=Exp(q[1]);L[1][1]=Exp(q[2]);L[2][2]=Exp(Jet(0)-q[1]-q[2]);L[1][0]=q[3];L[2][0]=q[4];L[2][1]=q[5];
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){Jet m;for(int k=0;k<3;++k)m=m+L[i][k]*L[j][k];if(!Finite(m))return false;u.metric.g[i][j]=m.f;for(int d=0;d<3;++d){u.metric.dg[d][i][j]=m.d[d];for(int e=0;e<3;++e)u.metric.ddg[d][e][i][j]=m.h[d][e];}}
 u.chi=Scalar(q[0]);u.alpha=Scalar(q[6]);for(int i=0;i<3;++i){u.beta.value[i]=q[7+i].f;for(int d=0;d<3;++d){u.beta.d[d][i]=q[7+i].d[d];for(int e=0;e<3;++e)u.beta.dd[d][e][i]=q[7+i].h[d][e];}}return true;
}
inline bool Velocity(const h::Z4cJet<Real>&u,const h::Z4cRHS<Real>&f,const h::GaugeRHS<Real>&g,std::array<Real,10>&qt){
 auto time=u;for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int d=0;d<3;++d){time.metric.dg[d][i][j]=d==0?f.metric[i][j]:0;for(int e=0;e<3;++e)time.metric.ddg[d][e][i][j]=0;}
 Config q;if(!Encode(time,q))return false;qt[0]=f.chi;qt[6]=g.alpha;for(int i=1;i<=5;++i)qt[i]=q[i].d[0];for(int i=0;i<3;++i)qt[7+i]=g.beta[i];return true;
}
}}
