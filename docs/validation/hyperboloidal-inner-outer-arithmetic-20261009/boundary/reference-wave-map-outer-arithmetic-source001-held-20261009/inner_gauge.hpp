// Private fixed-reference-position nonlinear source proposal. SOURCE ONLY.
#ifndef RESEARCH_INNER_GAUGE_HPP_
#define RESEARCH_INNER_GAUGE_HPP_
#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include "reference_wave_map.hpp"
namespace inner {
namespace hyp=z4c::hyperboloidal;
template<class T> T Coefficient(T alpha,T chi,double W,double G0,bool&valid,
                                Audit *audit=nullptr,bool value_only_control=false){
  valid=false;const double a=Number<T>::Value(alpha),ch=Number<T>::Value(chi);
  if(!(a>0)||!(ch>0)||!(G0>0)||!std::isfinite(a)||!std::isfinite(ch)||
     !std::isfinite(G0)||!std::isfinite(W)||W<0||W>1)return T(0);
  if(audit)++audit->coefficient_calls;
  if(W==1){valid=true;return T(.5);}
  int ea=0,ec=0,ex=0,eg=0;
  const double ma=std::frexp(a,&ea),mc=std::frexp(ch,&ec);
  const double mx=std::frexp(1-W,&ex),mg=std::frexp(G0,&eg);
  const double mA=ma*ma*mc,mX=mx*mg;
  const int eA=2*ea+ec,eX=ex+eg,e=std::max(eA,eX);
  const double u=std::scalbn(mA,eA-e),v=std::scalbn(mX,eX-e);
  if(audit) audit->coefficient_scaled_away+=(u==0)+(v==0);
  const double den=v+(1+W)*u,num=v+W*u;
  if(!(den>0)||!std::isfinite(den))return T(0);
  // When W=0 the numerator may be tiny although division by den would make
  // it representable. Divide its normal mantissa BEFORE final binary scaling.
  const double k=W==0?std::scalbn(mX/den,eX-e):num/den;
  // Fixed reference position/parameters: only live field derivatives enter.
  const double relative=2*(Number<T>::Derivative(alpha)/a)+
                           Number<T>::Derivative(chi)/ch;
  const double dk=value_only_control?0:
    -std::scalbn(ProductValue({u,mX,relative,1/den,1/den},audit),eX-e);
  valid=std::isfinite(k)&&std::isfinite(dk)&&k>=0&&k<=1;
  return Number<T>::Make(k,dk);
}

template<class T> hyp::GaugeRHSParts<T> Gauge(
    const hyp::LayerPoint<T>&p,const hyp::Z4cJet<T>&u,const T xyz[3],
    Parameters params={},Audit *audit=nullptr){
  hyp::GaugeRHSParts<T> q{};
  const T a=u.alpha.value,chi=u.chi.value,h=p.alpha,ch=p.state.chi.value;
  if(!(a>0)||!(chi>0)||!Kokkos::isfinite(a)||!Kokkos::isfinite(chi)||
     !(params.G0>0)||!std::isfinite(params.G0))return q;
  // Genericity is for live-field duals at a fixed external reference position.
  // Refuse a derivative of position rather than silently differentiating W as0.
  if(Number<T>::Derivative(p.radius)!=0)return q;
  const auto connection=rwm::ReferenceConnection(p,xyz);
  if(!connection.valid)return q;
  const double W=Number<T>::Value(hyp::SmoothCutoff(p.radius,T(.45),T(.85)).value);
  if(W==1)return rwm::Gauge(p,u,connection); // BEFORE every inner coefficient.
  bool coefficient_valid=false;
  const T k=Coefficient(a,chi,W,params.G0,coefficient_valid,audit);
  if(!coefficient_valid)return q;
  const auto g=hyp::Geometry(u.metric),gh=hyp::Geometry(p.state.metric);
  if(!g.valid||!gh.valid||!(h>0)||!(ch>0))return q;
  const T c=1-W,da=a-h;
  const T A0=Product<T>({a,a,chi},audit),Ah=Product<T>({h,h,ch},audit);
  const T B=Product<T>({c,T(params.G0)},audit)+Product<T>({T(W),A0},audit);
  const T dA=FieldSquareDifference(a,chi,h,ch,audit);
  if(!Kokkos::isfinite(A0)||!Kokkos::isfinite(B)||!Kokkos::isfinite(dA))return q;
  T db[3]{},dgi[3][3]{},dV[3][3]{},Lh[3][3]{},dL[3][3]{},dc[3]{},dal[3]{};
  T ref_adv=0;
  for(int i=0;i<3;++i){
    db[i]=u.beta.value[i]-p.beta[i];
    dc[i]=FieldLogGradientDifference(chi,u.chi.d[i],ch,p.state.chi.d[i],LogGradientField::Chi,audit);
    dal[i]=FieldLogGradientDifference(a,u.alpha.d[i],h,p.dalpha[i],LogGradientField::Alpha,audit);
    ref_adv+=Product<T>({p.beta[i],p.dalpha[i]},audit);
  }
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){
    for(int l=0;l<3;++l)for(int m=0;m<3;++m)
      dgi[i][j]-=Product<T>({g.inverse[i][l],u.metric.g[l][m]-p.state.metric.g[l][m],gh.inverse[m][j]},audit);
    dV[i][j]=Product<T>({dA,g.inverse[i][j]},audit)+Product<T>({Ah,dgi[i][j]},audit);
    Lh[i][j]=Product<T>({Ah,gh.inverse[i][j]},audit)-Product<T>({p.beta[i],p.beta[j]},audit);
    dL[i][j]=dV[i][j]-Product<T>({db[i],u.beta.value[j]},audit)-Product<T>({p.beta[i],db[j]},audit);
  }
  q.regular.alpha=-Product<T>({da,ref_adv,T(1)/h},audit);
  q.pole.alpha=-Product<T>({a,a+2*c,u.trace.value-p.k_physical},audit)
               -Product<T>({a,da,p.k_physical},audit);
  for(int i=0;i<3;++i){
    q.regular.alpha+=Product<T>({u.beta.value[i],u.alpha.d[i]-p.dalpha[i]},audit)
                     +Product<T>({db[i],p.dalpha[i]},audit);
    q.pole.alpha-=Product<T>({a,db[i],p.domega[i]},audit);
    q.regular.beta[i]=Product<T>({B,u.lambda.value[i]-p.state.lambda.value[i]},audit)
                     +Product<T>({dA,p.state.lambda.value[i]},audit);
    for(int j=0;j<3;++j){
      q.pole.alpha-=Product<T>({a,dL[i][j],connection.scaled[0][i][j]},audit);
      q.regular.beta[i]+=Product<T>({u.beta.value[j],u.beta.d[j][i]-p.state.beta.d[j][i]},audit)
                           +Product<T>({db[j],p.state.beta.d[j][i]},audit);
      q.regular.beta[i]+=Product<T>({T(2),a,a,k,k,g.inverse[i][j],dc[j]},audit)
        +Product<T>({T(.5),dA,T(1)/ch,g.inverse[i][j],p.state.chi.d[j]},audit)
        +Product<T>({T(.5),h,h,dgi[i][j],p.state.chi.d[j]},audit)
        -Product<T>({a,chi,g.inverse[i][j],dal[j]},audit)
        -Product<T>({dA,T(1)/h,g.inverse[i][j],p.dalpha[j]},audit)
        -Product<T>({h,ch,dgi[i][j],p.dalpha[j]},audit);
      q.pole.beta[i]+=Product<T>({T(2),dV[i][j],p.domega[j]},audit);
      for(int l=0;l<3;++l)q.pole.beta[i]-=
        Product<T>({dL[j][l],connection.scaled[i+1][j][l]},audit)
        +Product<T>({dL[j][l],u.beta.value[i],connection.scaled[0][j][l]},audit)
        +Product<T>({Lh[j][l],db[i],connection.scaled[0][j][l]},audit);
    }
  }
  q.valid=Kokkos::isfinite(q.regular.alpha)&&Kokkos::isfinite(q.pole.alpha);
  for(int i=0;i<3;++i)q.valid=q.valid&&Kokkos::isfinite(q.regular.beta[i])&&Kokkos::isfinite(q.pole.beta[i]);
  return q;
}
} // namespace inner
#endif
