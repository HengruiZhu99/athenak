// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_NATIVE_WORMHOLE_HPP_
#define Z4C_HYPERBOLOIDAL_NATIVE_WORMHOLE_HPP_
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

namespace z4c::hyperboloidal {
// Host initialization only. Quadrature nodes integrate each requested radius;
// there is no sampled radial field table or interpolation. Exact duplicate
// radii may reuse an already evaluated point (common on spherical shells).
class NativeWormhole {
  friend class NativeBoostedWormhole;
 public:
  struct Radial {double chi,ar,trace,lapse,shift_ratio,areal;};
  enum class TransitionProfile {bump, polynomial_c6};
  // Optional areal transition radii in mass units support initial-slice controls.
  // Defaults preserve the frozen 4M--8M initializer; no evolution gauge changes.
  explicit NativeWormhole(double mass=1./3.,double tolerance=2e-13,
                          double transition_start_M=4,double transition_end_M=8,
                          TransitionProfile profile=TransitionProfile::bump,double compact_end=0)
      : mass_(mass),tol_(tolerance),r0_phys_(transition_start_M*mass),r1_phys_(transition_end_M*mass),
        profile_(profile),g8_(Rule(8)),g16_(Rule(16)) {
    if(!(mass>0)||!std::isfinite(mass)||!(tolerance>0)||tolerance>1e-5
        ||!std::isfinite(transition_start_M)||!std::isfinite(transition_end_M)
        ||!(transition_start_M>2)||!(transition_end_M>transition_start_M)
        ||(profile!=TransitionProfile::bump&&profile!=TransitionProfile::polynomial_c6))
      throw std::invalid_argument("invalid native wormhole parameters");
    j0_=Integral(1/r0_phys_);r0_=std::exp(-j0_);
    const double rho0=(r0_phys_-mass+std::sqrt(r0_phys_*(r0_phys_-2*mass)))/2;
    scale_=r0_/rho0;inner_shift_=ShiftIntegral(r0_phys_);
    if(compact_end!=0){
      if(!std::isfinite(compact_end)||profile!=TransitionProfile::polynomial_c6 ||
          !(compact_end>OuterTransitionRadius()&&compact_end<1))
        throw std::invalid_argument("compact slicing needs C6 anchor and an extended outer endpoint");
      // Above the anchor's outer join the map is the same exact CMC solution.
      compact_y_end_=Inverse(compact_end);compact_end_=compact_end;
      double lo=7,hi=150;const double target=1/r0_phys_;
      if(!(CompactIntegrate(r0_,lo)[0]>=target && CompactIntegrate(r0_,hi)[0]<=target))
        throw std::invalid_argument("compact slicing shape cannot preserve the central scale");
      for(int i=0;i<52;++i){const double mid=(lo+hi)/2;
        if(CompactIntegrate(r0_,mid)[0]>target)lo=mid;else hi=mid;}
      compact_shape_=(lo+hi)/2;const auto join=CompactIntegrate(r0_,compact_shape_);
      compact_join_residual_=join[0]/target-1;inner_shift_=join[1];
      if(std::abs(compact_join_residual_)>50*tol_)throw std::runtime_error("compact central matching failed");
    }
  }
  double InnerRadius()const{return r0_;}
  double OuterTransitionRadius()const{return compact_end_>0?compact_end_:std::exp(-Integral(1/r1_phys_));}
  double CompactShape()const{return compact_shape_;}
  double CompactEnd()const{return compact_end_;}
  double CompactJoinResidual()const{return compact_join_residual_;}
  double TransitionEndAreal()const{return compact_end_>0?1/compact_y_end_:r1_phys_;}
  double IsotropicScale()const{return scale_;}
  double Mass()const{return mass_;}
  double Tolerance()const{return tol_;}
  size_t CachedRadii()const{return cache_.size();}
  Radial Point(double r){
    if(!(r>0&&r<=1))throw std::invalid_argument("native wormhole excludes puncture, includes scri");
    if(r==1)return {1,0,-3,1,0,std::numeric_limits<double>::infinity()};
    auto found=cache_.find(r);if(found!=cache_.end())return found->second;
    const double o=(1-r)*(1+r)/2;
    Radial v{};
    if(r<=r0_){
      const double rho=r/scale_,psi=1+mass_/(2*rho);
      v.chi=scale_*scale_/(o*o*std::pow(psi,4));v.ar=0;v.trace=0;
      v.lapse=o/(psi*psi);v.shift_ratio=inner_shift_;v.areal=rho*psi*psi;
    }else if(compact_end_>0 && r<compact_end_){
      const auto mapped=CompactIntegrate(r,compact_shape_);const double y=mapped[0],R=1/y;
      const auto sw=CompactSwitch(r,compact_shape_);
      const double a=std::sqrt(y*y-2*mass_*y*y*y+sw[0]*sw[0])/y;
      v.chi=std::pow(r*y/o,2);v.ar=-2*r*sw[1]/(3*o*a);
      v.trace=-3*sw[0]-r*sw[1]/a;v.lapse=o*PhysicalLapse(R,sw[0]);
      v.shift_ratio=mapped[1];v.areal=R;
    }else{
      const double y=Inverse(r),R=1/y;const auto s=Switch(R);
      v.chi=std::pow(r*y/o,2);v.ar=-2*R*s[1]/(3*o);v.trace=-3*s[0]-R*s[1];
      v.lapse=o*PhysicalLapse(R,s[0]);v.shift_ratio=ShiftIntegral(R);v.areal=R;
    }
    if(!(v.chi>0&&v.lapse>0)||!std::isfinite(v.shift_ratio)||!std::isfinite(v.ar)||!std::isfinite(v.trace))
      throw std::runtime_error("nonfinite native initial data");
    cache_.emplace(r,v);return v;
  }
  // Derivatives in the exact S=1 outer plateau, in the native Omega=(1-r²)/2
  // frame. Differentiate y=1/R through y'=-sqrt(1+y²-2 M y³)/r, not through
  // the quadrature/inversion tolerance. Order: chi, Ar, K, conformal lapse,
  // shift_ratio. At scri chi=1+O((1-r)^3), alpha=(1+r²)/2+O((1-r)^3).
  struct OuterDerivatives {std::array<double,5> first{},second{};};
  OuterDerivatives OuterRadialDerivatives(double r) {
    if(!(r>=OuterTransitionRadius()&&r<=1))
      throw std::invalid_argument("outer reference derivatives require S=1");
    OuterDerivatives out{};
    if(r==1){out.first[3]=1;out.second[3]=1;return out;}
    const auto v=Point(r);
    const long double x=r,y=1.L/v.areal,m=mass_,o=(1-x*x)/2;
    const long double F=std::sqrt(1+y*y-2*m*y*y*y);
    const long double Fy=(y-3*m*y*y)/F;
    const long double Fyy=(1-6*m*y)/F-Fy*Fy/F;
    const long double yp=-F/x,ypp=F*(1+Fy)/(x*x);
    const long double log_o=-x/o,log_opp=-1/o-log_o*log_o;
    const long double c1=2*(1/x+yp/y-log_o);
    const long double c2=2*(-1/(x*x)+ypp/y-yp*yp/(y*y)-log_opp);
    const long double a1=log_o+(Fy/F-1/y)*yp;
    const long double a2=log_opp+(Fyy/F-Fy*Fy/(F*F)+1/(y*y))*yp*yp+(Fy/F-1/y)*ypp;
    out.first[0]=v.chi*c1;out.second[0]=v.chi*(c1*c1+c2);
    out.first[3]=v.lapse*a1;out.second[3]=v.lapse*(a1*a1+a2);
    return out;
  }
  std::array<double,31> Fields(const double x[3]){
    const double r=std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
    const auto v=Point(r);std::array<double,31> q{};
    q[0]=v.chi;q[13]=v.trace+3;q[18]=v.lapse-(1+r*r)/2;
    int field=7;
    for(int i=0;i<3;++i)for(int j=i;j<3;++j)
      q[field++]=1.5*v.ar*(x[i]*x[j]/(r*r)-(i==j?1./3.:0.));
    for(int i=0;i<3;++i)q[22+i]=x[i]*v.shift_ratio;
    return q;
  }
  // Actual scri data are limits, never singular evaluations at rounded radius.
  static std::array<double,31> ScriFields(){std::array<double,31>q{};q[0]=1;return q;}
 private:
  struct Gauss {std::vector<double>x,w;};
  double mass_,tol_,r0_phys_,r1_phys_,j0_,r0_,scale_,inner_shift_;
  TransitionProfile profile_;
  double compact_end_=0,compact_shape_=0,compact_y_end_=0,compact_join_residual_=0;
  // Host-only coordinate construction. No interpolated field table or evolution
  // constraint solve: every requested radius is independently integrated.
  std::array<double,2> CompactSwitch(double r,double p)const{
    if(r<=r0_)return {0,0};if(r>=compact_end_)return {1,0};
    const double z=(r-r0_)/(compact_end_-r0_),t=1-z;
    double term=1,sum=1,norm=p;
    for(int j=1;j<=6;++j){term*=((p+j-1)/j)*t;sum+=term;norm*=(p+j)/j;}
    const double value=std::pow(z,p)*sum;
    const double derivative=norm*std::pow(z,p-1)*std::pow(t,6)/(compact_end_-r0_);
    return {value,derivative};
  }
  std::array<double,2> CompactIntegrate(double target,double shape)const{
    if(!(target>=r0_&&target<=compact_end_))throw std::invalid_argument("invalid compact map radius");
    using V=std::array<double,2>;
    const auto rhs=[&](double r,const V&u){const auto s=CompactSwitch(r,shape);const double y=u[0];
      const double F=std::sqrt(y*y-2*mass_*y*y*y+s[0]*s[0]);
      if(!(F>0 && y>0))throw std::runtime_error("nonmonotone compact map");
      return V{-F/r,-PhysicalLapse(1/y,s[0])*s[1]*y/F};};
    const auto rk=[&](double r,const V&u,double h){const auto a=rhs(r,u);
      V v{};for(int j=0;j<2;++j)v[j]=u[j]+h*a[j]/2;const auto b=rhs(r+h/2,v);
      for(int j=0;j<2;++j)v[j]=u[j]+h*b[j]/2;const auto c=rhs(r+h/2,v);
      for(int j=0;j<2;++j)v[j]=u[j]+h*c[j];const auto d=rhs(r+h,v);
      for(int j=0;j<2;++j)v[j]=u[j]+h*(a[j]+2*b[j]+2*c[j]+d[j])/6;return v;};
    double r=compact_end_,h=-(compact_end_-r0_)/32;V u{compact_y_end_,0};
    int attempts=0;
    while(r>target){
      if(r-target<8*std::numeric_limits<double>::epsilon()){u=rk(r,u,target-r);break;}
      if(++attempts>200000)throw std::runtime_error("compact map integration did not converge");
      h=-std::min(-h,r-target);const auto whole=rk(r,u,h),half=rk(r,u,h/2),fine=rk(r+h/2,half,h/2);
      double error=0;for(int j=0;j<2;++j)error=std::max(error,std::abs(fine[j]-whole[j])/15);
      // Local relative/absolute step tolerance; Richardson extrapolation removes
      // the estimated leading RK4 error. Do not shrink the budget with h again:
      // that can ask for a difference below roundoff and cause step collapse.
      const double budget=(tol_+2*std::numeric_limits<double>::epsilon())*
          std::max({.01,std::abs(fine[0]),std::abs(fine[1])});
      if(error<=budget){for(int j=0;j<2;++j)u[j]=fine[j]+(fine[j]-whole[j])/15;r+=h;}
      const double factor=error>0?std::clamp(.8*std::pow(budget/error,.2),.2,2.):2.;h*=factor;
      if(-h<4*std::numeric_limits<double>::epsilon() && r>target+1e-14)
        throw std::runtime_error("compact map tolerance exhausted floating precision");
    }return u;
  }
  Gauss g8_,g16_;
  std::map<double,Radial>cache_;
  static Gauss Rule(int n){
    Gauss g;g.x.resize(n);g.w.resize(n);
    for(int i=0;i<(n+1)/2;++i){
      double z=std::cos(std::acos(-1.)*(i+.75)/(n+.5)),derivative=0;
      for(int k=0;k<40;++k){
        double p=1,previous=0;
        for(int j=1;j<=n;++j){const double old=previous;previous=p;p=((2*j-1)*z*previous-(j-1)*old)/j;}
        derivative=n*(z*p-previous)/(z*z-1);const double next=z-p/derivative;
        if(std::abs(next-z)<2e-16){z=next;break;}z=next;
      }
      // Reevaluate derivative at the final root for the quadrature weight.
      double p=1,previous=0;
      for(int j=1;j<=n;++j){const double old=previous;previous=p;p=((2*j-1)*z*previous-(j-1)*old)/j;}
      derivative=n*(z*p-previous)/(z*z-1);
      g.x[i]=-z;g.x[n-1-i]=z;g.w[i]=g.w[n-1-i]=2/((1-z*z)*derivative*derivative);
    }return g;
  }
  template<class F> static double Gaussian(const F&f,double lo,double hi,const Gauss&g){
    const double mid=(lo+hi)/2,half=(hi-lo)/2;long double total=0;
    for(size_t i=0;i<g.x.size();++i)total+=static_cast<long double>(g.w[i])*f(mid+half*g.x[i]);
    return half*total;
  }
  template<class F> double Integrate(const F&f,double lo,double hi,double atol,int depth=0)const{
    if(hi==lo)return 0;
    const double fine=Gaussian(f,lo,hi,g16_),coarse=Gaussian(f,lo,hi,g8_);
    if(std::abs(fine-coarse)<=atol+tol_*std::abs(fine))return fine;
    if(depth>=24)throw std::runtime_error("native initial quadrature did not converge");
    const double mid=(lo+hi)/2;
    return Integrate(f,lo,mid,atol/2,depth+1)+Integrate(f,mid,hi,atol/2,depth+1);
  }
  std::array<double,2> Switch(double R)const{
    if(R<=r0_phys_)return {0,0};if(R>=r1_phys_)return {1,0};
    const double z=(R-r0_phys_)/(r1_phys_-r0_phys_),E=1/z-1/(1-z);
    if(profile_==TransitionProfile::polynomial_c6){
      // Integral of 12012*z^6*(1-z)^6; symmetric evaluation avoids cancellation
      // at the outer plateau. C6 at both endpoints, not an analytic evolution switch.
      const double t=std::min(z,1-z);
      const double p=std::pow(t,7)*(1716+t*(-9009+t*(20020+t*(-24024+t*(16380+t*(-6006+924*t))))));
      return {z<=.5?p:1-p,12012*std::pow(z*(1-z),6)/(r1_phys_-r0_phys_)};
    }
    const double a=std::exp(-std::abs(E));const double s=E>=0?a/(1+a):1/(1+a);
    const double ds=a/((1+a)*(1+a))*(1/(z*z)+1/((1-z)*(1-z)))/(r1_phys_-r0_phys_);
    return {s,ds};
  }
  double Denominator(double y)const{
    const double s=y>0?Switch(1/y)[0]:1;
    return y*y-2*mass_*y*y*y+s*s;
  }
  double Integral(double y)const{
    const double split=std::min(y,1/r1_phys_);
    const auto f=[&](double z){return 1/std::sqrt(Denominator(z));};
    return Integrate(f,0,split,tol_*split)+Integrate(f,split,y,tol_*(y-split));
  }
  double Inverse(double r)const{
    const double target=-std::log(r);double lo=0,hi=1/r0_phys_,y=hi*target/j0_;
    for(int i=0;i<100;++i){
      const double error=Integral(y)-target;
      if(std::abs(error)<=std::max(tol_,8*std::numeric_limits<double>::epsilon())*target)return y;
      if(error<0)lo=y;else hi=y;
      const double trial=y-error*std::sqrt(Denominator(y));
      y=trial>lo&&trial<hi?trial:(lo+hi)/2;
    }throw std::runtime_error("native isotropic coordinate inversion did not converge");
  }
  double PhysicalLapse(double R,double s)const{
    const double rho=(R-mass_+std::sqrt(R*(R-2*mass_)))/2,psi=1+mass_/(2*rho);
    return (1-s)/(psi*psi)+s*std::sqrt(1-2*mass_/R+R*R);
  }
  double ShiftIntegral(double lower)const{
    if(lower>=r1_phys_)return 0;
    lower=std::max(lower,r0_phys_);
    const auto f=[&](double R){const auto s=Switch(R);
      return PhysicalLapse(R,s[0])*s[1]/std::sqrt(1-2*mass_/R+R*R*s[0]*s[0]);};
    return Integrate(f,lower,r1_phys_,tol_*(r1_phys_-lower));
  }
};
}
#endif
