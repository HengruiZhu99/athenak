// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_BOUNDED_RHO_LOG_DERIVATIVE_HPP_
#define Z4C_HYPERBOLOIDAL_BOUNDED_RHO_LOG_DERIVATIVE_HPP_
#include "z4c/hyperboloidal/enriched_log_derivative.hpp"
#include "z4c/hyperboloidal/rho_log_derivative.hpp"

namespace z4c { namespace hyperboloidal {
// Separate HOST research variant. One geometry-only theta blends the complete
// base/rho scalar jet. Every value/gradient/Hessian row obeys the SAME16/32/64
// L1 caps; no physical-data selection, chart exponent or evolution adoption.
class BoundedRhoLogDerivative {
 public:
  using Point=EnrichedLogDerivative::Point;
  using Result=EnrichedLogDerivative::Result;
  struct Diagnostics {
    std::string status="unbuilt";int rank=0;
    Real theta=0,condition=0,base_condition=0,rho_condition=0,reproduction=0,maximum_weight_l1=0;
    Real value_weight_l1=0,scaled_gradient_weight_l1=0,scaled_hessian_weight_l1=0;
    Real base_value_l1=0,base_gradient_l1=0,base_hessian_l1=0;
    Real rho_value_l1=0,rho_gradient_l1=0,rho_hessian_l1=0;
  };
  const Diagnostics &diagnostics()const{return diagnostic_;}
  bool Build(const Point &query,const std::vector<Point> &points,Real spacing,const PuncturePosition &puncture){
    diagnostic_=Diagnostics{};weights_.clear();h_=spacing;stamp_=puncture.Capture();
    auto fail=[&](const std::string &reason){diagnostic_.status=reason;weights_.clear();return false;};
    EnrichedLogDerivative base;RhoLogDerivative rho;RhoLogDerivative::Options options;
    options.maximum_value_l1=options.maximum_gradient_l1=options.maximum_hessian_l1=1e9;
    if(!base.Build(query,points,spacing,puncture))return fail("base_"+base.diagnostics().status);
    if(!rho.Build(query,points,spacing,puncture,options))return fail("rho_"+rho.diagnostics().status);
    diagnostic_.base_condition=base.diagnostics().condition;diagnostic_.rho_condition=rho.diagnostics().condition;diagnostic_.condition=std::max(diagnostic_.base_condition,diagnostic_.rho_condition);diagnostic_.rank=58;
    const auto bw=Recover(base,points.size(),spacing,puncture),rw=Recover(rho,points.size(),spacing,puncture);const auto bn=Norms(bw),rn=Norms(rw);
    diagnostic_.base_value_l1=bn[0];diagnostic_.base_gradient_l1=bn[1];diagnostic_.base_hessian_l1=bn[2];diagnostic_.rho_value_l1=rn[0];diagnostic_.rho_gradient_l1=rn[1];diagnostic_.rho_hessian_l1=rn[2];
    if(!Safe(bn))return fail("base_functional_caps");
    Real theta=1;
    if(!Safe(rn)){Real lo=0,hi=1;for(int iteration=0;iteration<48;++iteration){const Real mid=(lo+hi)/2;if(Safe(Norms(Combine(bw,rw,mid))))lo=mid;else hi=mid;}theta=lo;}
    weights_=Combine(bw,rw,theta);auto actual=Norms(weights_);
    // Recover/Evaluate use anchor-normalized rows. Keep at most1e-12 distance
    // from the computed maximal feasible theta, never relax a cap for roundoff.
    for(int retry=0;!Safe(actual)&&retry<8;++retry){theta=std::max(Real(0),theta-1e-12);weights_=Combine(bw,rw,theta);actual=Norms(weights_);}
    if(!Safe(actual))return fail("combined_functional_caps");
    diagnostic_.theta=theta;diagnostic_.value_weight_l1=actual[0];diagnostic_.scaled_gradient_weight_l1=actual[1];diagnostic_.scaled_hessian_weight_l1=actual[2];
    for(const auto &row:weights_){Real norm=0;for(Real w:row)norm+=std::abs(w);diagnostic_.maximum_weight_l1=std::max(diagnostic_.maximum_weight_l1,norm);}
    std::vector<std::array<int,3>> powers;for(int degree=0;degree<=5;++degree)for(int a=0;a<=degree;++a)for(int b=0;b<=degree-a;++b)powers.push_back({a,b,degree-a-b});
    const Real r2=puncture.RadiusSquared(query.data());const auto y=puncture.Relative(query.data());
    for(int col=0;col<57;++col){std::vector<Real> values;for(const auto &x:points){Real value=1;
        if(col==56)value=.5*(std::log(puncture.RadiusSquared(x.data()))-std::log(r2));
        else for(int a=0;a<3;++a)value*=std::pow((x[a]-query[a])/spacing,powers[col][a]);values.push_back(value);}
      std::array<Real,11> target{};
      if(col==56){for(int a=0;a<3;++a)target[1+a]=spacing*y[a]/r2;int row=4;for(int a=0;a<3;++a)for(int b=a;b<3;++b,++row)target[row]=spacing*spacing*((a==b?1.:0)/r2-2*(y[a]/r2)*(y[b]/r2));target[10]=1;}
      else{const auto &p=powers[col];const int degree=p[0]+p[1]+p[2];target[0]=degree==0;
        if(degree==1)for(int a=0;a<3;++a)if(p[a]==1)target[1+a]=1;
        if(degree==2){int row=4;for(int a=0;a<3;++a)for(int b=a;b<3;++b,++row)if((a==b&&p[a]==2)||(a!=b&&p[a]==1&&p[b]==1))target[row]=a==b?2:1;}}
      const auto result=Apply(values);for(int f=0;f<11;++f)diagnostic_.reproduction=std::max(diagnostic_.reproduction,std::abs(result[f]-target[f]));
    }
    if(!std::isfinite(diagnostic_.reproduction)||diagnostic_.reproduction>1e-6)return fail("common_space_reproduction");
    diagnostic_.status="ready";return true;
  }
  bool Evaluate(const std::vector<Real>&values,const PuncturePosition&puncture,Result &result)const{
    if(weights_.empty()||values.size()!=weights_[0].size()||!puncture.Matches(stamp_))return false;for(Real v:values)if(!std::isfinite(v))return false;const auto r=Apply(values);Result out{};out.fitted_value=r[0];out.enrichment_coefficient=r[10];
    for(int a=0;a<3;++a)out.gradient[a]=r[1+a]/h_;int row=4;for(int a=0;a<3;++a)for(int b=a;b<3;++b,++row)out.hessian[a][b]=out.hessian[b][a]=r[row]/(h_*h_);
    if(!std::isfinite(out.fitted_value)||!std::isfinite(out.enrichment_coefficient))return false;for(int a=0;a<3;++a){if(!std::isfinite(out.gradient[a]))return false;for(int b=0;b<3;++b)if(!std::isfinite(out.hessian[a][b]))return false;}result=out;return true;
  }
 private:
  using Rows=std::vector<std::vector<Real>>;Rows weights_;Real h_=0;PuncturePosition::Stamp stamp_{};Diagnostics diagnostic_;
  template<class Fit>static Rows Recover(const Fit &fit,size_t count,Real spacing,const PuncturePosition&p){Rows w(11,std::vector<Real>(count));for(size_t i=0;i<count;++i){std::vector<Real> unit(count,0);unit[i]=1;typename Fit::Result r;if(!fit.Evaluate(unit,p,r))throw std::runtime_error("basis impulse rejected");w[0][i]=r.fitted_value;for(int a=0;a<3;++a)w[1+a][i]=spacing*r.gradient[a];int row=4;for(int a=0;a<3;++a)for(int b=a;b<3;++b,++row)w[row][i]=spacing*spacing*r.hessian[a][b];w[10][i]=r.enrichment_coefficient;}return w;}
  static Rows Combine(const Rows&a,const Rows&b,Real theta){Rows w=a;for(int f=0;f<11;++f){Real sum=0;for(size_t i=0;i<w[f].size();++i){w[f][i]=(1-theta)*a[f][i]+theta*b[f][i];sum+=w[f][i];}w[f][0]+=(f==0?1:0)-sum;}return w;}
  static std::array<Real,3>Norms(const Rows&w){std::array<Real,3> n{};for(int f=0;f<10;++f){Real sum=0,l1=0;for(Real v:w[f])sum+=v;l1=std::abs(w[f][0]+(f==0?1:0)-sum);for(size_t i=1;i<w[f].size();++i)l1+=std::abs(w[f][i]);if(f==0)n[0]=l1;else if(f<4)n[1]=std::max(n[1],l1);else n[2]=std::max(n[2],l1);}return n;}
  static bool Safe(const std::array<Real,3>&n){return std::isfinite(n[0]+n[1]+n[2])&&n[0]<=16&&n[1]<=32&&n[2]<=64;}
  std::array<Real,11>Apply(const std::vector<Real>&values)const{std::array<Real,11>r{};const Real anchor=values[0];r[0]=anchor;for(int f=0;f<11;++f)for(size_t i=0;i<values.size();++i)r[f]+=weights_[f][i]*(values[i]-anchor);return r;}
};
}}  // namespace z4c::hyperboloidal
#endif
