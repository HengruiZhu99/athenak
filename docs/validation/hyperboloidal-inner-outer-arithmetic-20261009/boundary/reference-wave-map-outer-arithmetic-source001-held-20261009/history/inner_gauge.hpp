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
// A scalar adapter is explicit: field duals must register BOTH components.
// This CPU prototype is not an arbitrary AD or GPU implementation.
template<class T> struct Number;
template<> struct Number<double> {
  static double Value(double x){return x;}
  static double Derivative(double){return 0;}
  static double Make(double x,double){return x;}
};
struct Audit {
  unsigned long products=0, coefficient_calls=0, coefficient_scaled_away=0;
  int maximum_product_exponent=0;
  unsigned long dA_near=0,dA_far=0,dc_near=0,dc_far=0,dal_near=0,dal_far=0;
};
struct Parameters { double G0=.375; };

inline double ProductValue(std::initializer_list<double> xs,Audit *audit=nullptr){
  double mantissa=1;int exponent=0;bool negative=false,zero=false;
  for(double x:xs){
    if(!std::isfinite(x))return std::numeric_limits<double>::quiet_NaN();
    negative=negative!=std::signbit(x);
    if(x==0){zero=true;continue;}
    int e=0;mantissa*=std::frexp(std::abs(x),&e);exponent+=e;
  }
  if(audit){++audit->products;audit->maximum_product_exponent=
    std::max(audit->maximum_product_exponent,std::abs(exponent));}
  if(zero)return std::copysign(0.,negative?-1.:1.);
  return std::scalbn(negative?-mantissa:mantissa,exponent);
}
template<class T> T Product(std::initializer_list<T> xs,Audit *audit=nullptr){
  // Scale each primal product and each product-rule term separately. A zero
  // primal never discards its possibly nonzero dual tangent.
  double mantissa=1;int exponent=0;bool negative=false,zero=false;
  for(const T&x:xs){const double v=Number<T>::Value(x);
    if(!std::isfinite(v)||!std::isfinite(Number<T>::Derivative(x)))
      return Number<T>::Make(std::numeric_limits<double>::quiet_NaN(),
                            std::numeric_limits<double>::quiet_NaN());
    negative=negative!=std::signbit(v);if(v==0){zero=true;continue;}
    int e=0;mantissa*=std::frexp(std::abs(v),&e);exponent+=e;}
  if(audit){++audit->products;audit->maximum_product_exponent=
    std::max(audit->maximum_product_exponent,std::abs(exponent));}
  const double value=zero?std::copysign(0.,negative?-1.:1.):
    std::scalbn(negative?-mantissa:mantissa,exponent);
  double derivative=0;unsigned selected=0;
  for(const T&dx:xs){const double seed=Number<T>::Derivative(dx);
    if(seed!=0){double m=1;int e=0;bool neg=false,z=false;unsigned index=0;
      for(const T&x:xs){const double v=index++==selected?seed:Number<T>::Value(x);
        neg=neg!=std::signbit(v);if(v==0){z=true;continue;}
        int ei=0;m*=std::frexp(std::abs(v),&ei);e+=ei;}
      derivative+=z?std::copysign(0.,neg?-1.:1.):std::scalbn(neg?-m:m,e);}
    ++selected;}
  return Number<T>::Make(value,derivative);
}
// Exact [1/2,2] positive-PRIMAL closeness, without a ratio or half-product.
inline bool NearFieldValue(double x,double reference){
  if(!(x>0)||!(reference>0)||!std::isfinite(x)||!std::isfinite(reference))return false;
  int ex=0,er=0;const double mx=std::frexp(x,&ex),mr=std::frexp(reference,&er);
  const int difference=ex-er;
  if(difference==0)return true;
  if(difference==1)return mx<=mr;
  if(difference==-1)return mx>=mr;
  return false;
}
template<class T> T FieldSquareDifference(T a,T chi,T h,T reference_chi,
                                         Audit *audit=nullptr){
  const bool near=NearFieldValue(Number<T>::Value(a),Number<T>::Value(h))&&
    NearFieldValue(Number<T>::Value(chi),Number<T>::Value(reference_chi));
  if(audit){if(near)++audit->dA_near;else ++audit->dA_far;}
  // Preserve exact-reference deviation arithmetic only while BOTH fields are near.
  if(near)return Product<T>({a+h,a-h,chi},audit)+
                 Product<T>({h,h,chi-reference_chi},audit);
  return Product<T>({a,a,chi},audit)-Product<T>({h,h,reference_chi},audit);
}
enum class LogGradientField { Chi, Alpha };
template<class T> T FieldLogGradientDifference(T x,T gradient,T reference,
    T reference_gradient,LogGradientField field,Audit *audit=nullptr){
  const bool near=NearFieldValue(Number<T>::Value(x),Number<T>::Value(reference));
  if(audit){
    if(field==LogGradientField::Chi){if(near)++audit->dc_near;else ++audit->dc_far;}
    else {if(near)++audit->dal_near;else ++audit->dal_far;}
  }
  const T reference_log_gradient=reference_gradient/reference;
  if(near)return (gradient-reference_gradient)-
                 Product<T>({x-reference,reference_log_gradient},audit);
  return gradient-Product<T>({x,reference_log_gradient},audit);
}

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
