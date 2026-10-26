// Native linear TT oracle evaluation only. Analytic coefficient series are not
// evolved fields. Independent MP physical/ADM jets validate this helper.
#pragma once
#include "goal1_native_ad.hpp"
#include "goal1_clock_runtime.hpp"
namespace cubed_sphere {namespace tensor_oracle {
namespace h=z4c::hyperboloidal;
struct S{static constexpr int N=4;double a[5][5]{};S()=default;S(double x){a[0][0]=x;}double value()const{return a[0][0];}static S R(double x){S z(x);z.a[1][0]=1;return z;}static S T(double x){S z(x);z.a[0][1]=1;return z;}};
inline S operator+(const S&x,const S&y){S z;for(int i=0;i<=4;++i)for(int j=0;j<=4-i;++j)z.a[i][j]=x.a[i][j]+y.a[i][j];return z;}
inline S operator-(const S&x,const S&y){S z;for(int i=0;i<=4;++i)for(int j=0;j<=4-i;++j)z.a[i][j]=x.a[i][j]-y.a[i][j];return z;}
inline S operator-(const S&x){return S(0)-x;}
inline S operator*(const S&x,const S&y){S z;for(int i=0;i<=4;++i)for(int j=0;j<=4-i;++j)for(int k=0;k<=i;++k)for(int l=0;l<=j;++l)z.a[i][j]+=x.a[k][l]*y.a[i-k][j-l];return z;}
inline S Scale(const S&x,double v){S z;for(int i=0;i<=4;++i)for(int j=0;j<=4-i;++j)z.a[i][j]=v*x.a[i][j];return z;}
inline S Pow(const S&x,double p){if(!(x.value()>0))throw std::runtime_error("TT positive analytic power domain");S d=Scale(x-S(x.value()),1/x.value()),term(1),z(1);double c=1;for(int k=1;k<=4;++k){term=term*d;c*=(p-k+1)/k;z=z+Scale(term,c);}return Scale(z,std::pow(x.value(),p));}
inline S Inv(const S&x){if(x.value()==0)throw std::runtime_error("TT analytic zero denominator");S d=Scale(x-S(x.value()),1/x.value()),term(1),z(1);for(int k=1;k<=4;++k){term=term*d;z=z+Scale(term,k%2?-1:1);}return Scale(z,1/x.value());}
inline S operator/(const S&x,const S&y){return x*Inv(y);}
inline S Exp(const S&x){S d=x-S(x.value()),term(1),z(1);double c=1;for(int k=1;k<=4;++k){term=term*d;c/=k;z=z+Scale(term,c);}return Scale(z,std::exp(x.value()));}
inline S Dr(const S&x){S z;for(int i=0;i<4;++i)for(int j=0;j<4-i;++j)z.a[i][j]=(i+1)*x.a[i+1][j];return z;}
inline S Dt(const S&x){S z;for(int i=0;i<4;++i)for(int j=0;j<4-i;++j)z.a[i][j]=(j+1)*x.a[i][j+1];return z;}
inline S IntegerPower(const S&x,int n){S z(1),b=x;while(n){if(n%2)z=z*b;b=b*b;n/=2;}return z;}
struct Ref{S o,L,A,b,G,B,chi;};
inline Ref Reference(const S&r){S o,b;if(r.value()<=.05){o=S(1);}else if(r.value()>=.95){o=S(1)-r*r;b=S(2)*r;}else{S s=(r-S(.05))/S(.9),g=-Inv(s)+Inv(S(1)-s),e=Exp(g.value()<=0?g:-g),w=g.value()<=0?e/(S(1)+e):Inv(S(1)+e);o=S(1)-w*r*r;b=S(2)*r*w;}S L=o-r*Dr(o),A=Pow(o*o+b*b,.5),G=L*L/(A*A),B=-b*A/L;return{o,L,A,b,G,B,Pow(G,-1./3)};}
inline S HeightJet(double radius,const clock_oracle::Height&height){S r=S::R(radius),u(height.Value(radius));auto g=Reference(r);S f=-g.L/(g.A*(g.A+g.b));for(int k=1;k<=4;++k)u.a[k][0]=f.a[k-1][0]/k;return u;}
inline S Profile(const S&u,int order,double width=.15){S s=(u+S(.35))/S(width);if(std::abs(s.value())>=1)return S(0);
 // Exact derivative factorization: f^(n)=(1-s²)^(16-n) Q_n(s).
 // Polynomial degree n<=4, avoiding cancellation in expanded degree32 f.
 double q[5]={1,0,0,0,0};int degree=0;
 for(int n=0;n<order;++n){double next[5]{};for(int k=0;k<=degree;++k){if(k)next[k-1]+=k*q[k];next[k+1]-=(k+2*(16-n))*q[k];}++degree;for(int k=0;k<=4;++k)q[k]=next[k];}
 S polynomial;for(int k=degree;k>=0;--k)polynomial=polynomial*s+S(q[k]);return S(1e-8/std::pow(width,order))*IntegerPower(S(1)-s*s,16-order)*polynomial;
}
struct Coeff{std::array<S,15>c;};
inline Coeff Evaluate(double radius,double time,const clock_oracle::Height&height,double width=.15){Coeff out{};if(radius==0)return out;S r=S::R(radius),t=S::T(time);auto g=Reference(r);S z=g.o/r,u=t+HeightJet(radius,height),f[5];for(int n=0;n<=4;++n)f[n]=Profile(u,n,width);
 S p=z*(f[4]+z*(S(2)*f[3]+z*(S(3)*f[2]+z*(S(3)*f[1]+S(3)*z*f[0])))),s=(p-S(6)*z*z*z*(f[2]+S(3)*z*f[1]+S(3)*z*z*f[0]))/S(2);
 S b=-S(2)*g.L*g.o/(r*r)*(f[3]+S(3)*z*f[2]+S(6)*z*z*f[1]+S(6)*z*z*z*f[0]),a=S(6)*g.L*g.L*g.o/(r*r*r)*(f[2]+S(3)*z*f[1]+S(3)*z*z*f[0]);S da=-g.B*g.B*a/(S(2)*g.A),dx=-g.chi*(-p+S(2)*s+a/g.G)/S(3),ba=-g.B*b,bc=-g.B*a/g.G;
 S ge=g.chi*p,gp=g.chi*s+dx,gb=g.chi*b,ga=g.chi*a+dx*g.G;S Gr=Dr(g.G),Br=Dr(g.B),kr=(g.B*Gr+S(2)*g.G*Br)/(S(2)*g.A),kt=g.B/(g.A*r),K=kr/g.G+S(2)*kt;
 S ke=(-Dt(p)+g.B*Dr(p)+S(2)*g.B*p/r+S(2)*ba/r)/(S(2)*g.A),kp=(-Dt(s)+g.B*Dr(s)+S(2)*g.B*s/r+S(2)*(bc-ba)/r)/(S(2)*g.A)-da*kt/g.A,kb=(-Dt(b)+g.B*Dr(b)+(Br+g.B/r)*b+Dr(ba)+(S(2)*g.G*bc-ba)/r)/(S(2)*g.A),ka=(-Dt(a)+g.B*Dr(a)+S(2)*Br*a+bc*Gr+S(2)*g.G*Dr(bc))/(S(2)*g.A)-da*kr/g.A;
 S dK=-ke+S(2)*kp+ka/g.G-((-p+S(2)*s)*kt+a*kr/(g.G*g.G)),w=-g.B*Dr(g.o)/g.A,dw=-bc*Dr(g.o)/g.A-w*da/g.A,dP=g.o*dK+S(3)*dw;
 S ie=-ge/(g.chi*g.chi),ip=-gp/(g.chi*g.chi),ib=-gb/(g.chi*g.chi*g.G),ia=-ga/(g.chi*g.chi*g.G*g.G),lv=-((-S(3)*ie+S(2)*ip+S(3)*ib)/r+Dr(ib)),ln=-((ie-S(2)*ip-S(3)*ib+S(2)*ia)/r+Dr(ia));
 out.c={dx,da,dP,ba,bc,lv,ln,ge,gp,gb,ga,g.chi*(ke-p*K/S(3)),dx*(kt-K/S(3))+g.chi*(kp-s*K/S(3)-dK/S(3)),g.chi*(kb-b*K/S(3)),dx*(kr-g.G*K/S(3))+g.chi*(ka-a*K/S(3)-g.G*dK/S(3))};return out;
}
inline h::Z4cJet<Real> NativeJet(const Coeff&rad,double radius,const Real n[3],int sector){h::Z4cJet<Real>u{};if(radius==0)return u;Jet x[3],r2;for(int i=0;i<3;++i){x[i]=Jet::Variable(radius*n[i],i);r2+=x[i]*x[i];}Jet r=Power(r2,.5),dr=r-Jet(radius),N[3],P[3][3],NN[3][3],E[3][3]{},Ep[3][3]{},vp[3],e;
 if(sector==0){E[0][0]=Jet(1);E[1][1]=Jet(-1);}else if(sector==1){E[0][1]=E[1][0]=Jet(1);}else throw std::runtime_error("invalid TT polarization");
 for(int i=0;i<3;++i)N[i]=x[i]/r;
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){NN[i][j]=N[i]*N[j];P[i][j]=Jet(i==j)-NN[i][j];e+=E[i][j]*NN[i][j];}
 for(int i=0;i<3;++i){for(int j=0;j<3;++j)vp[i]+=E[i][j]*N[j];vp[i]-=e*N[i];for(int j=0;j<3;++j)for(int k=0;k<3;++k)for(int l=0;l<3;++l)Ep[i][j]+=P[i][k]*E[k][l]*P[l][j];}
 Jet c[15];for(int k=0;k<15;++k){c[k]=Jet(rad.c[k].value())+Jet(rad.c[k].a[1][0])*dr;if(k!=5&&k!=6&&k<11)c[k]+=Jet(rad.c[k].a[2][0])*dr*dr;}
 auto scalar=[](const Jet&q){h::ScalarJet<Real>s{};s.value=q.f;for(int d=0;d<3;++d){s.d[d]=q.d[d];for(int e=0;e<3;++e)s.dd[d][e]=q.h[d][e];}return s;};u.chi=scalar(c[0]*e);u.alpha=scalar(c[1]*e);u.trace=scalar(c[2]*e);
 for(int i=0;i<3;++i){auto b=scalar(c[3]*vp[i]+c[4]*e*N[i]),l=scalar(c[5]*vp[i]+c[6]*e*N[i]);u.beta.value[i]=b.value;u.lambda.value[i]=l.value;for(int d=0;d<3;++d){u.beta.d[d][i]=b.d[d];u.lambda.d[d][i]=l.d[d];for(int j=0;j<3;++j)u.beta.dd[d][j][i]=b.dd[d][j];}
  for(int j=0;j<3;++j){Jet cross=vp[i]*N[j]+N[i]*vp[j],metric=c[7]*Ep[i][j]+c[8]*e*P[i][j]+c[9]*cross+c[10]*e*NN[i][j],A=c[11]*Ep[i][j]+c[12]*e*P[i][j]+c[13]*cross+c[14]*e*NN[i][j];u.metric.g[i][j]=metric.f;u.a.k[i][j]=A.f;for(int d=0;d<3;++d){u.metric.dg[d][i][j]=metric.d[d];u.a.dk[d][i][j]=A.d[d];for(int k=0;k<3;++k)u.metric.ddg[d][k][i][j]=metric.h[d][k];}}
 }return u;}
}}
