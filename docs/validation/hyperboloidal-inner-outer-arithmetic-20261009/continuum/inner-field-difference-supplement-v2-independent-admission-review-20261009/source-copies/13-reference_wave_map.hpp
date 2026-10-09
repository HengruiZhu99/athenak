// Private finite-Omega physical reference wave-map gauge. No production option.
#ifndef RESEARCH_REFERENCE_WAVE_MAP_HPP_
#define RESEARCH_REFERENCE_WAVE_MAP_HPP_
#include "z4c/hyperboloidal/layer_gauge.hpp"
namespace rwm {
namespace hyp=z4c::hyperboloidal;
template<class T> struct Connection { T scaled[4][3][3]{}; bool valid=false; };
// Omega*Gamma[physical reference]; stationary inertial time implies Gamma^a_0b=0.
// b'= -L*Kbar-2b/r follows from the actual reference extrinsic curvature.
// No live fields, height integration, Omega division or absent higher jets enter.
template<class T> KOKKOS_INLINE_FUNCTION Connection<T> ReferenceConnection(
    const hyp::LayerPoint<T>&p,const T x[3]) {
  Connection<T> c{};
  if (!(p.omega>0) || !(p.alpha>0) || !(p.L>0)) return c;
  if (p.radius==0) { for(int i=0;i<3;++i)c.scaled[0][i][i]=-p.k_bar/3; c.valid=true; return c; }
  const T r=p.radius,h=p.alpha,b=p.b;
  T n[3],op=0; for(int i=0;i<3;++i){n[i]=x[i]/r;op+=n[i]*p.domega[i];}
  const T bp=-p.L*p.k_bar-2*b/r;
  const T ar=p.L*(p.omega*bp-b*op)/(h*h*h),at=b/(h*r);
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){
    c.scaled[0][i][j]=ar*n[i]*n[j]+at*((i==j?T(1):T(0))-n[i]*n[j]);
    for(int k=0;k<3;++k)c.scaled[k+1][i][j]=
      -(k==i?p.domega[j]:T(0))-(k==j?p.domega[i]:T(0))
      -x[k]*p.omega*p.omega_hessian[i][j]/p.L;
  }
  c.valid=true;for(int a=0;a<4;++a)for(int i=0;i<3;++i)for(int j=0;j<3;++j)
    c.valid=c.valid&&Kokkos::isfinite(c.scaled[a][i][j]);
  return c;
}
// Returns Omega*Fbar; source checks use finite positive alpha and Omega.
// Fbar=bar(g)^bc Gamma[physical reference]^a_bc-2 bar(g)^ai Omega_i/Omega.
template<class T> KOKKOS_INLINE_FUNCTION void ScaledSource(
    const hyp::LayerPoint<T>&p,const hyp::Z4cJet<T>&u,const Connection<T>&c,T f[4]) {
  const auto gi=hyp::Geometry(u.metric);const T a2=u.alpha.value*u.alpha.value;
  for(int a=0;a<4;++a){f[a]=0;for(int i=0;i<3;++i)for(int j=0;j<3;++j)
    f[a]+=(u.chi.value*gi.inverse[i][j]-u.beta.value[i]*u.beta.value[j]/a2)*c.scaled[a][i][j];
    for(int i=0;i<3;++i){const T ai=a==0?u.beta.value[i]/a2:
      u.chi.value*gi.inverse[a-1][i]-u.beta.value[a-1]*u.beta.value[i]/a2;
      f[a]-=2*ai*p.domega[i];}
  }
}
// Exact polynomial deviation expansion of the physical-P wave-map rows.
// Rhat+Shat/Omega=0 is the stationary Minkowski identity, not a numerical RHS
// counterterm. Reference deviations make both returned parts exactly zero.
// P remains Kphysical-2Theta physical; Lambda includes 2*gtildeInv*Zcov.
// The assembled rows contain no division by the LIVE lapse.
template<class T> KOKKOS_INLINE_FUNCTION hyp::GaugeRHSParts<T> Gauge(
    const hyp::LayerPoint<T>&p,const hyp::Z4cJet<T>&u,const Connection<T>&c) {
  hyp::GaugeRHSParts<T> q{};const T a=u.alpha.value,h=p.alpha,chi=u.chi.value,ch=p.state.chi.value;
  if(!c.valid || !(a>0) || !(chi>0) || !Kokkos::isfinite(a) || !Kokkos::isfinite(chi))return q;
  const auto gi=hyp::Geometry(u.metric),gh=hyp::Geometry(p.state.metric);
  if(!gi.valid||!gh.valid)return q;
  const T da=a-h,da2=(a+h)*da;
  T db[3],dgi[3][3]{},C[3][3]{},Ch[3][3]{},dC[3][3]{},dV[3][3]{},Lh[3][3]{},dL[3][3]{};
  T dB=0;
  for(int i=0;i<3;++i){db[i]=u.beta.value[i]-p.beta[i];dB+=db[i]*p.domega[i];}
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){
    // inverse(g)-inverse(ghat)=-inverse(g)*(g-ghat)*inverse(ghat).
    for(int k=0;k<3;++k)for(int l=0;l<3;++l)dgi[i][j]-=gi.inverse[i][k]*(u.metric.g[k][l]-p.state.metric.g[k][l])*gh.inverse[l][j];
    C[i][j]=chi*gi.inverse[i][j];Ch[i][j]=ch*gh.inverse[i][j];
    dC[i][j]=(chi-ch)*gi.inverse[i][j]+ch*dgi[i][j];
    dV[i][j]=da2*C[i][j]+h*h*dC[i][j];
    Lh[i][j]=h*h*Ch[i][j]-p.beta[i]*p.beta[j];
    dL[i][j]=dV[i][j]-db[i]*u.beta.value[j]-p.beta[i]*db[j];
  }
  q.pole.alpha=-a*(a*(u.trace.value-p.k_physical)+da*p.k_physical+dB);
  T ref_advection=0;for(int i=0;i<3;++i)ref_advection+=p.beta[i]*p.dalpha[i];
  q.regular.alpha=-da*ref_advection/h;
  for(int i=0;i<3;++i){q.regular.alpha+=u.beta.value[i]*(u.alpha.d[i]-p.dalpha[i])+db[i]*p.dalpha[i];
    for(int j=0;j<3;++j)q.pole.alpha-=a*dL[i][j]*c.scaled[0][i][j];}
  for(int i=0;i<3;++i){
    q.regular.beta[i]+=a*a*chi*(u.lambda.value[i]-p.state.lambda.value[i])+(da2*chi+h*h*(chi-ch))*p.state.lambda.value[i];
    for(int j=0;j<3;++j){
      q.regular.beta[i]+=u.beta.value[j]*(u.beta.d[j][i]-p.state.beta.d[j][i])+db[j]*p.state.beta.d[j][i];
      q.regular.beta[i]+=T(.5)*(a*a*gi.inverse[i][j]*(u.chi.d[j]-p.state.chi.d[j])+(da2*gi.inverse[i][j]+h*h*dgi[i][j])*p.state.chi.d[j]);
      q.regular.beta[i]-=a*C[i][j]*(u.alpha.d[j]-p.dalpha[j])+(da*C[i][j]+h*dC[i][j])*p.dalpha[j];
      q.pole.beta[i]+=2*dV[i][j]*p.domega[j];
      for(int k=0;k<3;++k)q.pole.beta[i]-=dL[j][k]*(c.scaled[i+1][j][k]+u.beta.value[i]*c.scaled[0][j][k])+Lh[j][k]*db[i]*c.scaled[0][j][k];
    }
  }
  q.valid=Kokkos::isfinite(q.regular.alpha)&&Kokkos::isfinite(q.pole.alpha);
  for(int i=0;i<3;++i)q.valid=q.valid&&Kokkos::isfinite(q.regular.beta[i])&&Kokkos::isfinite(q.pole.beta[i]);
  return q;
}
template<class T> KOKKOS_INLINE_FUNCTION bool Assemble(const hyp::GaugeRHSParts<T>&q,T omega,hyp::GaugeRHS<T>&f){
  if(!q.valid||!(omega>0)||!Kokkos::isfinite(omega))return false;
  f.alpha=q.regular.alpha+q.pole.alpha/omega;
  for(int i=0;i<3;++i)f.beta[i]=q.regular.beta[i]+q.pole.beta[i]/omega;
  bool valid=Kokkos::isfinite(f.alpha);for(int i=0;i<3;++i)valid=valid&&Kokkos::isfinite(f.beta[i]);return valid;
}
}
#endif
