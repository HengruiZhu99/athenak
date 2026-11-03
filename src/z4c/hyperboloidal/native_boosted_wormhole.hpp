// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_NATIVE_BOOSTED_WORMHOLE_HPP_
#define Z4C_HYPERBOLOIDAL_NATIVE_BOOSTED_WORMHOLE_HPP_
#include "native_wormhole.hpp"
namespace z4c::hyperboloidal {
#ifndef HYPERBOLIC_BOOSTED_PUNCTURE
#define HYPERBOLIC_BOOSTED_PUNCTURE 0
#endif
inline constexpr bool boosted_puncture=HYPERBOLIC_BOOSTED_PUNCTURE;
#ifndef HYPERBOLIC_BOOST_FREE_INITIAL_GAUGE
#define HYPERBOLIC_BOOST_FREE_INITIAL_GAUGE 0
#endif
inline constexpr bool boost_free_initial_gauge=HYPERBOLIC_BOOST_FREE_INITIAL_GAUGE;
static_assert(!boost_free_initial_gauge||boosted_puncture);
namespace boost_detail {
// First spatial derivatives, propagated analytically through the embedding.
struct D {double v=0;std::array<double,3>d{};D()=default;D(double x):v(x){};};
inline D operator+(D a,D b){D c(a.v+b.v);for(int k=0;k<3;++k)c.d[k]=a.d[k]+b.d[k];return c;}
inline D operator-(D a,D b){D c(a.v-b.v);for(int k=0;k<3;++k)c.d[k]=a.d[k]-b.d[k];return c;}
inline D operator-(D a){return D(0)-a;}
inline D operator*(D a,D b){D c(a.v*b.v);for(int k=0;k<3;++k)c.d[k]=a.d[k]*b.v+a.v*b.d[k];return c;}
inline D operator/(D a,D b){D c(a.v/b.v);for(int k=0;k<3;++k)c.d[k]=(a.d[k]-c.v*b.d[k])/b.v;return c;}
inline D Sqrt(D a){D c(std::sqrt(a.v));for(int k=0;k<3;++k)c.d[k]=a.d[k]/(2*c.v);return c;}
inline D InvCbrt(D a){D c(1/std::cbrt(a.v));for(int k=0;k<3;++k)c.d[k]=-c.v*a.d[k]/(3*a.v);return c;}
template<class T>using Mat=std::array<std::array<T,3>,3>;
template<class T>T Det(const Mat<T>&a){return a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])-a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])+a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]);}
template<class T>Mat<T>Inv(const Mat<T>&a){Mat<T>b{};const T d=Det(a);for(int i=0;i<3;++i)for(int j=0;j<3;++j)b[i][j]=(a[(j+1)%3][(i+1)%3]*a[(j+2)%3][(i+2)%3]-a[(j+1)%3][(i+2)%3]*a[(j+2)%3][(i+1)%3])/d;return b;}
template<class T>Mat<T>Mul(const Mat<T>&a,const Mat<T>&b){Mat<T>c{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int k=0;k<3;++k)c[i][j]=c[i][j]+a[i][k]*b[k][j];return c;}
}
// Exact vacuum Schwarzschild embedding. Native quadrature/inversion only;
// no field table, finite-difference Lambda, or constraint-solving approximation.
// Derivation and admissibility controls: reproduction/boosted-puncture-v1.
class NativeBoostedWormhole {
  using D=boost_detail::D;template<class T>using Mat=boost_detail::Mat<T>;
  NativeWormhole base_;double m_,v_,lorentz_,rho0_,constant_;bool free_gauge_;
  struct Rad {double a,b,ap,bp,h1,h2,u0,u1,u2,B,Bp,Bpp,Q0,Q0p,S,sig,sig1,sig2;};
  static std::array<double,3>Poly(double z){
    if(z<=0)return {0,0,0};if(z>=1)return {1,0,0};
    const double t=std::min(z,1-z),p=std::pow(t,7)*(1716+t*(-9009+t*(20020+t*(-24024+t*(16380+t*(-6006+924*t))))));
    return {z<=.5?p:1-p,12012*std::pow(z*(1-z),6),72072*std::pow(z*(1-z),5)*(1-2*z)};
  }
  double Rstar(double R)const{return R+2*m_*std::log(R/(2*m_)-1);}
  double Tail(double R)const{
    const double y=1/R,split=std::min(y,1/(12*m_));
    const auto f=[&](double z){const double S=z>0?base_.Switch(1/z)[0]:1,A=std::sqrt(z*z-2*m_*z*z*z+S*S);return 1/(A*(A+S));};
    return base_.Integrate(f,0,split,base_.tol_)+base_.Integrate(f,split,y,base_.tol_);
  }
  Rad Radial(double rho)const{
    const double c=m_/(2*rho),psi=1+c,R=rho*psi*psi,Rp=1-c*c,Rpp=2*c*c/rho;
    const double width=.75*rho0_;const auto P=Poly((rho-.25*rho0_)/width);
    Rad q{};q.a=(1-c)/psi;q.b=psi*psi;q.ap=m_/(rho*rho*psi*psi);q.bp=-m_*psi/(rho*rho);
    q.B=1-P[0];q.Bp=-P[1]/width;q.Bpp=-P[2]/(width*width);q.Q0=1;q.sig=rho;q.sig1=1;
    if(rho<=rho0_)return q;
    const auto sw=Poly((R-4*m_)/(8*m_));const double S=sw[0],SR=sw[1]/(8*m_),SRR=sw[2]/(64*m_*m_),Sp=SR*Rp,Spp=SRR*Rp*Rp+SR*Rpp;
    const double f=q.a*q.a,fr=2*m_/(R*R),A=std::sqrt(f+R*R*S*S),AR=(fr+2*R*S*S+2*R*R*S*SR)/(2*A);
    const double hR=R*S/(f*A),hRR=(S+R*SR)/(f*A)-R*S*(fr*A+f*AR)/std::pow(f*A,2);
    q.h1=hR*Rp;q.h2=hRR*Rp*Rp+hR*Rpp;q.S=S;q.Q0=f/(A*A);q.Q0p=(fr/(A*A)-2*f*AR/(A*A*A))*Rp;
    const double g=1/(A*(A+R*S)),gR=-g*(AR/A+(AR+S+R*SR)/(A+R*S));
    const double C=Rstar(R)+constant_,Cp=Rp/f,Cpp=Rpp/f-fr*Rp*Rp/(f*f);
    q.sig=rho+S*(R-rho);q.sig1=1+Sp*(R-rho)+S*(Rp-1);q.sig2=Spp*(R-rho)+2*Sp*(Rp-1)+S*Rpp;
    q.u0=Tail(R)+(1-S)*C+S*q.sig;
    q.u1=-g*Rp+(1-S)*Cp+S*q.sig1+Sp*(q.sig-C);
    q.u2=-gR*Rp*Rp-g*Rpp+(1-S)*Cpp+2*Sp*(q.sig1-Cp)+Spp*(q.sig-C)+S*q.sig2;
    return q;
  }
 public:
  explicit NativeBoostedWormhole(double v=.1,double mass=1./3.,bool free_gauge=boost_free_initial_gauge):base_(mass,2e-13,4,12,NativeWormhole::TransitionProfile::polynomial_c6),m_(mass),v_(v),lorentz_(1/std::sqrt(1-v*v)),free_gauge_(free_gauge){
    if(!std::isfinite(v)||std::abs(v)>=.3)throw std::invalid_argument("boost outside audited mild range");
    rho0_=base_.InnerRadius()/base_.IsotropicScale();constant_=-Rstar(4*m_)-Tail(4*m_);
  }
  size_t CachedRadii()const{return base_.CachedRadii();}
  double ReferenceCoordinateVelocity()const{return v_*base_.IsotropicScale();}
  double InitialTrackerVelocity()const{return free_gauge_?0:ReferenceCoordinateVelocity();}
  static double InitialShiftWeight(double r){
    const double t=(r-.2)/.36;if(t<=0)return 0;if(t>=1)return 1;
    const double x=std::min(t,1-t),v=std::pow(x,5)*(126+x*(-420+x*(540+x*(-315+70*x))));
    return t<=.5?v:1-v;
  }
  std::array<double,31>Fields(const double x[3]){
    using namespace boost_detail;
    const double r=std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);if(!(r>0&&r<1))throw std::invalid_argument("boosted bulk point outside open compact ball");
    const auto bg=base_.Point(r);double rho,rp,rpp;
    if(r<=base_.InnerRadius()){rho=r/base_.IsotropicScale();rp=1/base_.IsotropicScale();rpp=0;}
    else{const double R=bg.areal;rho=(R-m_+std::sqrt(R*(R-2*m_)))/2;const double c=m_/(2*rho);const auto s=base_.Switch(R);
      const double A=std::sqrt(1-2*m_/R+R*R*s[0]*s[0]),AR=(2*m_/(R*R)+2*R*s[0]*s[0]+2*R*R*s[0]*s[1])/(2*A),dR=R*A/r;
      rp=dR/(1-c*c);rpp=(dR*(A+R*AR-1)/r-2*c*c/rho*rp*rp)/(1-c*c);}
    const auto lab=Radial(rho);double X[3],Y[3];for(int i=0;i<3;++i)X[i]=lab.sig*x[i]/r,Y[i]=X[i];Y[0]/=lorentz_;
    Mat<double>J{};Rad q;double rr,nn[3];bool converged=false;
    for(int it=0;it<40;++it){rr=std::sqrt(Y[0]*Y[0]+Y[1]*Y[1]+Y[2]*Y[2]);q=Radial(rr);for(int i=0;i<3;++i)nn[i]=Y[i]/rr;
      double residual[3],maximum=0;for(int i=0;i<3;++i){residual[i]=q.sig*nn[i];if(i==0)residual[i]=lorentz_*(residual[i]+v_*(q.u0-v_*Y[0]*q.B));residual[i]-=X[i];maximum=std::max(maximum,std::abs(residual[i]));
        for(int j=0;j<3;++j){J[i][j]=(i==j?q.sig/rr:0)+(q.sig1-q.sig/rr)*nn[i]*nn[j];if(i==0)J[i][j]=lorentz_*(J[i][j]+v_*(q.u1*nn[j]-v_*((j==0?q.B:0)+Y[0]*q.Bp*nn[j])));}}
      if(!(Det(J)>0))throw std::runtime_error("boost coordinate Jacobian not positive");
      if(maximum<2e-13*std::max(1.,lab.sig)){converged=true;break;}const auto inv=Inv(J);for(int i=0;i<3;++i)for(int j=0;j<3;++j)Y[i]-=inv[i][j]*residual[j];
    }if(!converged)throw std::runtime_error("boost map inversion failed");
    Mat<double>P{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)P[i][j]=(i==j?lab.sig/r:0)+(lab.sig1*rp-lab.sig/r)*x[i]*x[j]/(r*r);
    const auto T=Mul(Inv(J),P);
    D yd[3],xd[3],rad(r),rest(rr);for(int i=0;i<3;++i){yd[i]=D(Y[i]);yd[i].d=T[i];xd[i]=D(x[i]);xd[i].d[i]=1;rad.d[i]=x[i]/r;for(int a=0;a<3;++a)rest.d[i]+=nn[a]*T[a][i];}
    const auto lift=[](double value,double prime,const D&r){D d(value);for(int i=0;i<3;++i)d.d[i]=prime*r.d[i];return d;};
    D nd[3],nl[3];for(int i=0;i<3;++i)nd[i]=yd[i]/rest,nl[i]=xd[i]/rad;
    const D a=lift(q.a,q.ap,rest),b=lift(q.b,q.bp,rest),h1=lift(q.h1,q.h2,rest),B=lift(q.B,q.Bp,rest),Bp=lift(q.Bp,q.Bpp,rest),u1=lift(q.u1,q.u2,rest),sig=lift(q.sig,q.sig1,rest),sig1=lift(q.sig1,q.sig2,rest),Q0=lift(q.Q0,q.Q0p,rest);
    const D ls=lift(lab.sig,lab.sig1*rp,rad),lsp=lift(lab.sig1*rp,lab.sig2*rp*rp+lab.sig1*rpp,rad);
    D w[3],U[3];Mat<D>jd{},pd{};for(int i=0;i<3;++i){w[i]=-v_*((i==0?B:D(0))+yd[0]*Bp*nd[i]);U[i]=u1*nd[i]+w[i];}
    for(int i=0;i<3;++i)for(int j=0;j<3;++j){jd[i][j]=(i==j?sig/rest:D(0))+(sig1-sig/rest)*nd[i]*nd[j];if(i==0)jd[i][j]=lorentz_*(jd[i][j]+v_*U[j]);pd[i][j]=(i==j?ls/rad:D(0))+(lsp-ls/rad)*nl[i]*nl[j];}
    const auto td=Mul(Inv(jd),pd);D normal[3]{},wt[3]{};for(int i=0;i<3;++i)for(int k=0;k<3;++k){normal[i]=normal[i]+nd[k]*td[k][i];wt[i]=wt[i]+w[k]*td[k][i];}
    Mat<D>metric{};for(int i=0;i<3;++i)for(int j=0;j<3;++j){D tangent;for(int k=0;k<3;++k)tangent=tangent+(td[k][i]-nd[k]*normal[i])*(td[k][j]-nd[k]*normal[j]);metric[i][j]=b*b*(tangent+Q0*normal[i]*normal[j])-a*a*(h1*(normal[i]*wt[j]+wt[i]*normal[j])+wt[i]*wt[j]);}
    double dot=0,ww=0;for(int i=0;i<3;++i)dot+=nn[i]*w[i].v,ww+=w[i].v*w[i].v;
    const double Q=q.Q0-q.a*q.a/(q.b*q.b)*(2*q.h1*dot+ww);if(!(Q>0))throw std::runtime_error("boost slice is not spacelike");
    Mat<double>K{},KY{};double F[3],apF=0,bpF=0;for(int i=0;i<3;++i)F[i]=q.h1*nn[i]+w[i].v,apF+=q.ap*nn[i]*F[i],bpF+=q.bp*nn[i]*F[i];
    for(int i=0;i<3;++i)for(int j=0;j<3;++j){const double delta=i==j?1:0;
      const double F2=q.h2*nn[i]*nn[j]+q.h1/rr*(delta-nn[i]*nn[j])-v_*(q.Bp*((i==0?nn[j]:0)+(j==0?nn[i]:0))+Y[0]*(q.Bpp*nn[i]*nn[j]+q.Bp/rr*(delta-nn[i]*nn[j])));
      const double connection=(F[i]*q.bp*nn[j]+q.bp*nn[i]*F[j]-delta*bpF)/q.b;
      KY[i][j]=-(q.a*F2+q.ap*(nn[i]*F[j]+F[i]*nn[j])-q.a*connection-q.a*q.a/(q.b*q.b)*apF*F[i]*F[j])/std::sqrt(Q);
      if(q.S==1)K[i][j]=-metric[i][j].v;
      // Pullback below after every KY component has been calculated.
    }
    if(q.S!=1)for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int k=0;k<3;++k)for(int l=0;l<3;++l)K[i][j]+=T[k][i]*KY[k][l]*T[l][j];
    const auto inverse=Inv(metric);double trace=0;for(int i=0;i<3;++i)for(int j=0;j<3;++j)trace+=inverse[i][j].v*K[i][j];if(q.S==1)trace=-3;
    const D om=(D(1)-rad*rad)/2;Mat<D>bar{},gt{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)bar[i][j]=om*om*metric[i][j];const D chi=InvCbrt(Det(bar));
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)gt[i][j]=chi*bar[i][j];const auto gi=Inv(gt);
    auto fields=base_.Fields(x);fields[0]=chi.v;fields[13]=trace+3;int f=1;for(int i=0;i<3;++i)for(int j=i;j<3;++j)fields[f++]=gt[i][j].v-(i==j?1:0);
    f=7;for(int i=0;i<3;++i)for(int j=i;j<3;++j)fields[f++]=chi.v*om.v*(K[i][j]-metric[i][j].v*trace/3);
    for(int i=0;i<3;++i){fields[15+i]=0;for(int j=0;j<3;++j)fields[15+i]-=gi[i][j].d[j];}
    if(free_gauge_){
      // In the inner lab Cartesian chart this is exactly det(gamma_lab)^(-1/6).
      // The baseline hyperboloidal lapse supplies the regular unit scri limit;
      // the determinant ratio cancels the spatial compactification Jacobian.
      fields[18]=bg.lapse*std::sqrt(chi.v/bg.chi)-(1+r*r)/2;
      const double W=InitialShiftWeight(r);
      for(int i=0;i<3;++i)fields[22+i]=(1-W)*x[i]; // beta=-W*x; beta=0 inside.
    }else{
      // Historical BX1 gauge retained for exact reproduction; not the new default
      // in BG1. It imposed a localized initial compact coordinate drift.
      const auto pinv=Inv(P);for(int i=0;i<3;++i)fields[22+i]-=v_*lab.B*pinv[i][0];
    }
    for(auto z:fields)if(!std::isfinite(z))throw std::runtime_error("nonfinite boosted native initial field");return fields;
  }
};
}
#endif
