// Source-matched global Minkowski clock oracle, native22 fields and exact jets.
// Private local CPU helper. Analytic series are target evaluation data, not PDE state.
#pragma once
#include "goal1_clock_taylor.hpp"
#include "z4c/hyperboloidal/cartesian_patch.hpp"
namespace cubed_sphere {namespace clock_oracle {
namespace h=z4c::hyperboloidal;
struct Geometry{Series omega,L,A,b;};
inline Geometry Reference(const Series&r){
 Series o,b;
 if(r.value()<=.05){o=Series(1);b=Series(0);}
 else if(r.value()>=.95){o=Series(1)-r*r;b=Series(2)*r;}
 else{Series s=(r-Series(.05))/Series(.9),g=-Inverse(s)+Inverse(Series(1)-s);Series e=Exp(g.value()<=0?g:-g);Series w=g.value()<=0?e/(Series(1)+e):Inverse(Series(1)+e);o=Series(1)-w*r*r;b=Series(2)*r*w;}
 return {o,o-r*Dr(o),Power(o*o+b*b,.5),b};
}
inline double Integrand(double r){auto g=Reference(Series::R(r));return -g.L.value()/(g.A.value()*(g.A.value()+g.b.value()));}
struct Height {
 int order=48;std::array<double,7>panels{.05,.15,.3,.5,.7,.85,.95};std::array<double,7>prefix{};std::vector<double>nodes,weights;double constant=0;
 explicit Height(int n=48):order(n),nodes(n),weights(n){
  for(int i=0;i<n;++i){double x=std::cos(3.14159265358979323846*(i+.75)/(n+.5)),dp=0;
   for(int k=0;k<16;++k){double p0=1,p1=x;for(int j=2;j<=n;++j){double z=((2*j-1)*x*p1-(j-1)*p0)/j;p0=p1;p1=z;}dp=n*(x*p1-p0)/(x*x-1);double step=p1/dp;x-=step;if(std::abs(step)<2e-16)break;}
   double p0=1,p1=x;for(int j=2;j<=n;++j){double z=((2*j-1)*x*p1-(j-1)*p0)/j;p0=p1;p1=z;}dp=n*(x*p1-p0)/(x*x-1);nodes[i]=x;weights[i]=2/((1-x*x)*dp*dp);
  }
  prefix[0]=-.05;for(int i=1;i<7;++i)prefix[i]=prefix[i-1]+Integrate(panels[i-1],panels[i]);constant=prefix[6]-(1-.95)/(2*(1+.95));
 }
 double Integrate(double lo,double hi)const{double z=0;for(int i=0;i<order;++i)z+=weights[i]*Integrand((lo+hi)/2+(hi-lo)*nodes[i]/2);return z*(hi-lo)/2;}
 double Value(double r)const{if(r<=.05)return -r;if(r>=.95)return constant+(1-r)/(2*(1+r));int p=0;while(p+1<7&&r>panels[p+1])++p;return prefix[p]+Integrate(panels[p],r);}
 Series Jet(double radius)const{Series r=Series::R(radius),u(Value(radius));auto g=Reference(r);auto f=-g.L/(g.A*(g.A+g.b));for(int k=1;k<=3;++k)u.a[k][0]=f.a[k-1][0]/k;return u;}
};
inline Series Gaussian(const Series&x,double sigma){return Exp(-x*x/Series(sigma*sigma));}
struct Wave{Series H,HT,HR;};
inline Wave ClockWave(const Series&T,const Series&R,double sigma){
 const Series s2(sigma*sigma),z=Series(2)*T*R/s2;
 if(std::abs(z.value())<.5){Series C(1),Cp,even(1);double denom=1;
  for(int k=1;k<=13;++k){Series odd=even*z;even=odd*z;denom*=double(2*k)*double(2*k+1);C=C+Scale(even,1/denom);Cp=Cp+Scale(odd,2.*k/denom);}
  Series E=Exp(-(T*T+R*R)/s2);
  return {Series(4)*T/s2*E*C,Series(4)/s2*E*((Series(1)-Series(2)*T*T/s2)*C+z*Cp) ,Series(4)*T/s2*E*(-Series(2)*R/s2*C+Series(2)*T/s2*Cp)};
 }
 Series fu=Gaussian(T-R,sigma),fv=Gaussian(T+R,sigma),fpu=-Series(2)*(T-R)/s2*fu,fpv=-Series(2)*(T+R)/s2*fv;
 return {(fu-fv)/R,(fpu-fpv)/R,-(fpu+fpv)/R-(fu-fv)/(R*R)};
}
struct Radial {std::array<Series,10>f;double uref=0;};
inline Radial Evaluate(double radius,double time,double epsilon,const Height&height,double sigma=.2){
 Series r=Series::R(radius),t=Series::Time(time),e(epsilon);auto g=Reference(r);Series u0=t+height.Jet(radius),q,v,G,kr,kt,shear;
 if(radius>=.95){Series z;
  for(int k=0;k<8;++k){Series U=u0+g.omega*z,fu=Gaussian(U,sigma),fpu=-Series(2)*U/Series(sigma*sigma)*fu,fv,fpv;
   if(radius!=1){Series V=U+Series(2)*r/g.omega;fv=Gaussian(V,sigma);fpv=-Series(2)*V/Series(sigma*sigma)*fv;}
   z=z-(z+e/r*(fu-fv))/(Series(1)+e*g.omega/r*(fpu-fpv));
  }
  Series U=u0+g.omega*z,fu=Gaussian(U,sigma),fpu=-Series(2)*U/Series(sigma*sigma)*fu,fv,fpv,advanced;
  if(radius!=1){Series V=U+Series(2)*r/g.omega;fv=Gaussian(V,sigma);fpv=-Series(2)*V/Series(sigma*sigma)*fv;advanced=fpv/(g.omega*r);}
  Series I=Series(1)+e*g.omega/r*(fpu-fpv),D=(Inverse(g.A*(g.A+g.b))-e*((fu-fv)/(r*r)+Series(2)*advanced))/I;
  q=Inverse(I);v=Series(1)-g.omega*g.omega*D;G=g.L*g.L*D*(Series(2)-g.omega*g.omega*D);
  kr=g.L*g.L*(Series(2)*Dr(g.omega)*D+g.omega*Dr(D))/Power(G,1.5);kt=-v*g.L/(r*Power(G,.5));
  shear=g.L*g.L*(Dr(D)+Series(2)*D/r-Series(3)*g.L*g.omega*D*D/r+g.L*g.omega*g.omega*g.omega*D*D*D/r)/Power(G,1.5);
 }else{
  Series R=r/g.omega,T0=u0+R,delta;
  for(int k=0;k<8;++k){auto w=ClockWave(T0+delta,R,sigma);delta=delta-(delta+e*w.H)/(Series(1)+e*w.HT);}
  auto w=ClockWave(T0+delta,R,sigma);q=Inverse(Series(1)+e*w.HT);v=(g.b/g.A-e*w.HR)*q;Series s=Series(1)-v*v;G=g.L*g.L/(g.omega*g.omega)*s;
  kr=-g.omega*g.omega*Dr(v)/(g.L*Power(s,1.5));kt=-(radius==0?DivideRAtOrigin(v):v/r)*g.omega/Power(s,.5);shear=(kr-kt)/g.omega;
 }
 Series chi=Power(G,-1./3),lambda=Series(2./3)*Power(G,-5./3)*Dr(G)+Series(2)*Power(G,-2./3)*(radius==0?DivideRAtOrigin(G-Series(1)):(G-Series(1))/r);
 return {{chi,q*g.L/Power(G,.5),kr+Series(2)*kt,Series(0),-q*v*g.L/G,lambda,chi*G,chi,Series(2./3)*chi*shear*G,-Series(1./3)*chi*shear},height.Value(radius)};
}
inline h::Z4cJet<Real> NativeJet(const Radial&rad,double radius,const Real n[3]){
 h::Z4cJet<Real> axis{};auto scalar=[&](int k){h::ScalarJet<Real>s{};s.value=rad.f[k].value();s.d[0]=rad.f[k].a[1][0];s.dd[0][0]=2*rad.f[k].a[2][0];return s;};axis.chi=scalar(0);axis.alpha=scalar(1);axis.trace=scalar(2);axis.theta=scalar(3);
 axis.beta.value[0]=rad.f[4].value();axis.beta.d[0][0]=rad.f[4].a[1][0];axis.beta.dd[0][0][0]=2*rad.f[4].a[2][0];axis.lambda.value[0]=rad.f[5].value();axis.lambda.d[0][0]=rad.f[5].a[1][0];
 for(int i=0;i<3;++i){int g=i?7:6,A=i?9:8;axis.metric.g[i][i]=rad.f[g].value();axis.metric.dg[0][i][i]=rad.f[g].a[1][0];axis.metric.ddg[0][0][i][i]=2*rad.f[g].a[2][0];axis.a.k[i][i]=rad.f[A].value();axis.a.dk[0][i][i]=rad.f[A].a[1][0];}
 h::Z4cJet<Real>out{};
 if(radius>0)out=h::CartesianRadialJet(axis,Real(radius),n);
 else{
  out.chi.value=axis.chi.value;out.alpha.value=axis.alpha.value;out.trace.value=axis.trace.value;
  for(int d=0;d<3;++d){out.chi.dd[d][d]=axis.chi.dd[0][0];out.alpha.dd[d][d]=axis.alpha.dd[0][0];out.trace.dd[d][d]=axis.trace.dd[0][0];out.beta.d[d][d]=axis.beta.d[0][0];out.lambda.d[d][d]=axis.lambda.d[0][0];}
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){out.metric.g[i][j]=axis.metric.g[1][1]*(i==j);for(int d=0;d<3;++d)for(int e=0;e<3;++e)out.metric.ddg[d][e][i][j]=axis.metric.ddg[0][0][1][1]*(d==e)*(i==j)+(axis.metric.ddg[0][0][0][0]-axis.metric.ddg[0][0][1][1])/2*((d==i)*(e==j)+(d==j)*(e==i));}
 }
 for(int d=0;d<3;++d)for(int e=0;e<3;++e)for(int i=0;i<3;++i)out.lambda.dd[d][e][i]=0; // not consumed/provided
 return out;
}
}}
