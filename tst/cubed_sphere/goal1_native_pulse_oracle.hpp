// Exact jets of the existing native lapse/shift initial profile.
#pragma once
#include "goal1_typed_support.hpp"
#include "goal1_operator_oracle.hpp"
namespace cubed_sphere {namespace native_pulse_oracle {
namespace h=z4c::hyperboloidal;
inline h::Z4cJet<Real> Deviation(const Real x[3],Real amplitude,Real width){
 Jet y[3];Real z=0;for(int d=0;d<3;++d){y[d]=Jet(x[d]);y[d].d[d]=1;z+=x[d]*x[d];}Real w2=width*width,f=exp(-z/w2),b=1-z;Jet pulse(pow(b,4)*f);Real fp=f*(-4*pow(b,3)-pow(b,4)/w2),fpp=f*(12*b*b+8*pow(b,3)/w2+pow(b,4)/(w2*w2));for(int d=0;d<3;++d){pulse.d[d]=2*x[d]*fp;for(int e=0;e<3;++e)pulse.h[d][e]=4*x[d]*x[e]*fpp+(d==e?2*fp:0);}
 Jet values[4]={amplitude*pulse*(Jet(1)+.2*y[0]+.3*y[1]*y[2]),amplitude/2*pulse*(Jet(1)+.3*y[1]*y[2]),amplitude*.1*pulse*y[0],amplitude*.05*pulse*y[0]*y[1]};h::Z4cJet<Real>u{};
 auto scalar=[](const Jet&q){h::ScalarJet<Real>s{};s.value=q.f;for(int d=0;d<3;++d){s.d[d]=q.d[d];for(int e=0;e<3;++e)s.dd[d][e]=q.h[d][e];}return s;};u.alpha=scalar(values[0]);for(int n=0;n<3;++n){auto v=scalar(values[1+n]);u.beta.value[n]=v.value;for(int d=0;d<3;++d){u.beta.d[d][n]=v.d[d];for(int e=0;e<3;++e)u.beta.dd[d][e][n]=v.dd[d][e];}}return u;
}
}}
