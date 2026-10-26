// Replacement of two native fields only: geometric Penrose trace and Theta/Omega.
// No wave momenta or independent configuration gradients. Source analysis only.
#pragma once
#include "goal1_native_endpoint.hpp"
namespace cubed_sphere {namespace native_trace_chart {
namespace h=z4c::hyperboloidal;
template<class T>struct Parts {bool valid=false;std::array<T,22>regular{},numerator{};T nullness{},rw{},sw0{};};
template<class T>Parts<T> Evaluate(const h::Z4cJet<T>&u,const h::LayerPoint<T>&p,const rwm::Connection<T>&connection,T Q,T tau,const T dQ[3],const T dtau[3]){
 Parts<T>out;h::OmegaJet<T>o{};o.omega=p.omega;for(int i=0;i<3;++i){o.gradient[i]=p.domega[i];for(int j=0;j<3;++j)o.hessian[i][j]=p.omega_hessian[i][j];}h::SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,o);T alpha=u.alpha.value,chi=u.chi.value,w=o.normal,kappa=T(10)/alpha;
 auto gt=h::Geometry(u.metric),gb=h::Geometry(h::PenroseMetric(u.metric,u.chi));if(!gt.valid||!gb.valid)return out;auto f=h::ConformalRHS(u,o,kappa,T(0));auto gauge=rwm::Gauge(p,u,connection);auto zero=u;zero.trace.value=T(3)*w;auto gauge0=rwm::Gauge(p,zero,connection);if(!f.valid||!gauge.valid||!gauge0.valid)return out;
 T zup[3]{},zcov[3]{},dz[3][3]{},aup[3][3]{},H[3][3]{},Ha[3][3]{},lap=0,lapa=0,cross=0,grad=0,norma=0,divz=0,trH=0,divbeta=0,betaW=0,betaQ=0,betaTau=0;
 for(int i=0;i<3;++i){zup[i]=(u.lambda.value[i]-gt.contracted[i])/T(2);divbeta+=u.beta.d[i][i];betaW+=u.beta.value[i]*o.dnormal[i];betaQ+=u.beta.value[i]*dQ[i];betaTau+=u.beta.value[i]*dtau[i];for(int j=0;j<3;++j){zcov[i]+=u.metric.g[i][j]*(u.lambda.value[j]-gt.contracted[j])/T(2);for(int d=0;d<3;++d)dz[d][i]+=(u.metric.dg[d][i][j]*(u.lambda.value[j]-gt.contracted[j])+u.metric.g[i][j]*(u.lambda.d[d][j]-gt.dcontracted[d][j]))/T(2);}}
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){H[i][j]=o.hessian[i][j];Ha[i][j]=u.alpha.dd[i][j];T dZ=dz[i][j];for(int k=0;k<3;++k){H[i][j]-=gb.connection[k][i][j]*o.gradient[k];Ha[i][j]-=gb.connection[k][i][j]*u.alpha.d[k];dZ-=gt.connection[k][i][j]*zcov[k];for(int l=0;l<3;++l)aup[i][j]+=gt.inverse[i][k]*gt.inverse[j][l]*u.a.k[k][l];}lap+=gb.inverse[i][j]*H[i][j];lapa+=gb.inverse[i][j]*Ha[i][j];cross+=gb.inverse[i][j]*u.alpha.d[i]*o.gradient[j];grad+=gb.inverse[i][j]*o.gradient[i]*o.gradient[j];divz+=gt.inverse[i][j]*dZ;trH+=gt.inverse[i][j]*(H[i][j]+zcov[i]*o.gradient[j]+zcov[j]*o.gradient[i]);}
 for(int i=0;i<3;++i)for(int j=0;j<3;++j)norma+=aup[i][j]*u.a.k[i][j];T N=grad-w*w,Rw=0,Sw0=0;for(int i=0;i<3;++i){Rw-=o.gradient[i]*gauge.regular.beta[i]/alpha;Sw0-=o.gradient[i]*gauge0.pole.beta[i]/alpha;}Rw-=w*gauge.regular.alpha/alpha;Sw0-=w*gauge0.pole.alpha/alpha;out.nullness=N;out.rw=Rw;out.sw0=Sw0;
 out.regular=native_endpoint::Rows(f.regular,gauge.regular);out.numerator=native_endpoint::Rows(f.pole,gauge.pole);out.regular[0]=-T(2./3.)*chi*divbeta+T(2./3.)*alpha*chi*Q;for(int i=0;i<3;++i)out.regular[0]+=u.beta.value[i]*u.chi.d[i];out.numerator[0]=T(0);
 out.regular[7]=betaQ-lapa+alpha*(gb.scalar+T(2)*chi*divz+Q*Q);out.numerator[7]=o.omega*(T(3)*betaW+T(3)*cross+T(5)*alpha*lap+T(2)*alpha*w*Q-T(3)*alpha*kappa*tau-T(3)*Rw)-T(9)*alpha*N-T(3)*Sw0;
 out.regular[17]=betaTau+alpha*(gb.scalar-norma+T(2)*chi*divz)/T(2)+alpha*Q*Q/T(3);out.numerator[17]=alpha*(o.omega*(T(2)*lap+T(2)*w*Q-(T(4)*w+T(2)*kappa)*tau)-T(3)*N);
 out.regular[18]=gauge.regular.alpha-alpha*alpha*(Q-T(2)*tau);out.numerator[18]=gauge0.pole.alpha;
 for(int n=0;n<6;++n){int i=native_endpoint::ma[n],j=native_endpoint::mb[n];out.regular[8+n]=f.regular.a[i][j]+alpha*u.a.k[i][j]*Q;out.numerator[8+n]=T(2)*alpha*(chi*(H[i][j]+zcov[i]*o.gradient[j]+zcov[j]*o.gradient[i]-u.metric.g[i][j]*trH/T(3))+w*u.a.k[i][j]);}
 for(int i=0;i<3;++i){out.regular[14+i]=f.regular.lambda[i]-T(4./3.)*alpha*Q*zup[i];out.numerator[14+i]=-alpha*(T(4)*w+T(2)*kappa)*zup[i];for(int j=0;j<3;++j){out.regular[14+i]-=T(2./3.)*alpha*gt.inverse[i][j]*(T(2)*dQ[j]-T(3)*dtau[j]);out.numerator[14+i]-=T(2./3.)*alpha*gt.inverse[i][j]*(o.gradient[j]*(T(2)*Q-T(3)*tau)+T(6)*o.dnormal[j])+T(4)*alpha*aup[i][j]*o.gradient[j];}}
 out.valid=true;return out;
}
}}
