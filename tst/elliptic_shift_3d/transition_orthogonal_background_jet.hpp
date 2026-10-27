// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef TST_ELLIPTIC_SHIFT_3D_TRANSITION_ORTHOGONAL_BACKGROUND_JET_HPP_
#define TST_ELLIPTIC_SHIFT_3D_TRANSITION_ORTHOGONAL_BACKGROUND_JET_HPP_
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <vector>
#include "z4c/hyperboloidal/puncture_position.hpp"
#include "z4c/hyperboloidal/positive_field_chart.hpp"

namespace z4c { namespace hyperboloidal {
// Standalone orthogonal-background-residual derivative probe. Caller supplies
// the residual values and query jets after subtracting a P5 polynomial; the
// residual column is unshifted so its query value enters the value functional. Keeps all
// Cartesian P5 modes plus one caller-supplied analytic background column. Its
// coefficient is fitted freely per stencil/field, never prescribed or evolved.
// QR mechanics retain EnrichedLogDerivative attribution; no production adoption.
class TransitionOrthogonalBackgroundJet {
 public:
  using Point=std::array<Real,3>;
  static constexpr int polynomial_columns=56, columns=57, functionals=11;
  struct Options {
    Real gaussian_width=2, rank_tolerance=1.e-11;
    Real maximum_condition=1.e9, maximum_scaled_weight_l1=1.e9;
  };
  struct Result {
    Real fitted_value=0, enrichment_coefficient=0;
    Point gradient{};
    std::array<Point,3> hessian{};
  };
  struct Diagnostics {
    std::string status="unbuilt";
    int rank=0;
    Real condition=0, orthogonality=0, reproduction=0, maximum_weight_l1=0;
  };
  const Diagnostics &diagnostics() const {return diagnostic_;}
  bool Build(const Point &query,const std::vector<Point> &points,
             const std::vector<Real> &background,const ScalarJet<Real> &background_query,Real spacing,
             const PuncturePosition &puncture) {
    return Build(query,points,background,background_query,spacing,puncture,Options{});
  }
  bool Build(const Point &query,const std::vector<Point> &points,
             const std::vector<Real> &background,const ScalarJet<Real> &background_query,Real spacing,
             const PuncturePosition &puncture,const Options &options) {
    diagnostic_=Diagnostics{};weights_.clear();h_=spacing;stamp_=puncture.Capture();
    const int n=static_cast<int>(points.size());
    auto fail=[&](const char *s){diagnostic_.status=s;weights_.clear();return false;};
    if(n<columns||n>256)return fail("sample_count");
    if(!(h_>0)||!std::isfinite(h_)||!(h_*h_>0)||!std::isfinite(h_*h_)||
       !(options.gaussian_width>0)||!std::isfinite(options.gaussian_width)||
       !(options.rank_tolerance>0)||!std::isfinite(options.rank_tolerance)||
       !(options.maximum_condition>0)||!std::isfinite(options.maximum_condition)||
       !(options.maximum_scaled_weight_l1>0)||!std::isfinite(options.maximum_scaled_weight_l1))
      return fail("invalid_geometry");
    if(background.size()!=points.size()||!std::isfinite(background_query.value))return fail("invalid_background");
    for(int a=0;a<3;++a){if(!std::isfinite(background_query.d[a]))return fail("invalid_background");for(int b=0;b<3;++b)if(!std::isfinite(background_query.dd[a][b]))return fail("invalid_background");}
    auto powers=Monomials();
    std::vector<Real> phi(n*columns),sqrtw(n),norm(columns),v(n*columns),q(n*columns);
    for(int i=0;i<n;++i){
      if(!std::isfinite(background[i]))return fail("invalid_background");
      Point t;Real d2=0;
      for(int a=0;a<3;++a){t[a]=(points[i][a]-query[a])/h_;d2+=t[a]*t[a];}
      sqrtw[i]=std::exp(-d2/(4*options.gaussian_width*options.gaussian_width));
      if(!(sqrtw[i]>0)||!std::isfinite(sqrtw[i]))return fail("invalid_geometry");
      for(int j=0;j<polynomial_columns;++j){Real value=1;
        for(int a=0;a<3;++a)value*=std::pow(t[a],powers[j][a]);
        phi[i*columns+j]=value;}
      phi[i*columns+56]=background[i];
      for(int j=0;j<columns;++j){const Real value=sqrtw[i]*phi[i*columns+j];
        v[j*n+i]=value;norm[j]+=value*value;}
    }
    for(int j=0;j<columns;++j){norm[j]=std::sqrt(norm[j]);
      if(!(norm[j]>0)||!std::isfinite(norm[j]))return fail("rank_deficient");
      for(int i=0;i<n;++i)v[j*n+i]/=norm[j];}
    std::vector<Real> R(columns*columns,0);std::array<int,columns> permutation;
    std::iota(permutation.begin(),permutation.end(),0);
    for(int k=0;k<columns;++k){
      int pivot=k;Real best=-1;
      for(int j=k;j<columns;++j){Real s=0;for(int i=0;i<n;++i)s+=v[j*n+i]*v[j*n+i];
        if(s>best){best=s;pivot=j;}}
      if(pivot!=k){for(int i=0;i<n;++i)std::swap(v[k*n+i],v[pivot*n+i]);
        for(int j=0;j<k;++j)std::swap(R[j*columns+k],R[j*columns+pivot]);
        std::swap(permutation[k],permutation[pivot]);}
      // Two-pass MGS; update existing R entries for reorthogonalization.
      for(int pass=0;pass<2;++pass)for(int j=0;j<k;++j){
        Real s=0;for(int i=0;i<n;++i)s+=q[j*n+i]*v[k*n+i];
        R[j*columns+k]+=s;for(int i=0;i<n;++i)v[k*n+i]-=s*q[j*n+i];}
      Real d=0;for(int i=0;i<n;++i)d+=v[k*n+i]*v[k*n+i];d=std::sqrt(d);
      if(!(d>options.rank_tolerance)){diagnostic_.rank=k;return fail("rank_deficient");}
      R[k*columns+k]=d;diagnostic_.rank=k+1;
      for(int i=0;i<n;++i)q[k*n+i]=v[k*n+i]/d;
      for(int j=k+1;j<columns;++j)for(int pass=0;pass<2;++pass){
        Real s=0;for(int i=0;i<n;++i)s+=q[k*n+i]*v[j*n+i];
        R[k*columns+j]+=s;for(int i=0;i<n;++i)v[j*n+i]-=s*q[k*n+i];}
    }
    Real rnorm=0;std::array<Real,columns> inverse_rows{};
    for(int i=0;i<columns;++i){Real s=0;for(int j=i;j<columns;++j)s+=std::abs(R[i*columns+j]);rnorm=std::max(rnorm,s);}
    for(int col=0;col<columns;++col){std::array<Real,columns> x{};
      for(int i=columns-1;i>=0;--i){Real s=i==col?1:0;
        for(int j=i+1;j<columns;++j)s-=R[i*columns+j]*x[j];x[i]=s/R[i*columns+i];}
      for(int i=0;i<columns;++i)inverse_rows[i]+=std::abs(x[i]);}
    diagnostic_.condition=rnorm*(*std::max_element(inverse_rows.begin(),inverse_rows.end()));
    if(!std::isfinite(diagnostic_.condition)||diagnostic_.condition>options.maximum_condition)
      return fail("condition_rejected");
    for(int k=0;k<columns;++k)for(int j=0;j<=k;++j){Real s=0;
      for(int i=0;i<n;++i)s+=q[k*n+i]*q[j*n+i];
      diagnostic_.orthogonality=std::max(diagnostic_.orthogonality,std::abs(s-(j==k?1:0)));}
    std::vector<Real> L(functionals*columns,0);L[0]=1;
    for(int j=0;j<polynomial_columns;++j){const auto &p=powers[j];int degree=p[0]+p[1]+p[2];
      if(degree==1)for(int a=0;a<3;++a)if(p[a]==1)L[(1+a)*columns+j]=1;
      if(degree==2){int row=4;for(int a=0;a<3;++a)for(int b=a;b<3;++b,++row)
        if((a==b&&p[a]==2)||(a!=b&&p[a]==1&&p[b]==1))
          L[row*columns+j]=a==b?2:1;}}
    for(int a=0;a<3;++a)L[(1+a)*columns+56]=h_*background_query.d[a];
    int row=4;for(int a=0;a<3;++a)for(int b=a;b<3;++b,++row)
      L[row*columns+56]=h_*h_*background_query.dd[a][b];
    L[0*columns+56]=background_query.value;
    L[10*columns+56]=1;weights_.resize(functionals,std::vector<Real>(n));
    for(int f=0;f<functionals;++f){std::array<Real,columns> z{};
      for(int k=0;k<columns;++k){Real s=L[f*columns+permutation[k]]/norm[permutation[k]];
        for(int j=0;j<k;++j)s-=R[j*columns+k]*z[j];z[k]=s/R[k*columns+k];}
      Real l1=0;for(int i=0;i<n;++i){Real s=0;for(int k=0;k<columns;++k)s+=q[k*n+i]*z[k];
        weights_[f][i]=sqrtw[i]*s;l1+=std::abs(weights_[f][i]);}
      diagnostic_.maximum_weight_l1=std::max(diagnostic_.maximum_weight_l1,l1);
      for(int j=0;j<columns;++j){Real s=0;for(int i=0;i<n;++i)s+=weights_[f][i]*phi[i*columns+j];
        diagnostic_.reproduction=std::max(diagnostic_.reproduction,std::abs(s-L[f*columns+j]));}
    }
    if(!std::isfinite(diagnostic_.maximum_weight_l1)||diagnostic_.maximum_weight_l1>options.maximum_scaled_weight_l1)
      return fail("weight_amplification");
    if(!std::isfinite(diagnostic_.reproduction)||diagnostic_.reproduction>1.e-6)
      return fail("reproduction_failure");
    diagnostic_.status="ready";return true;
  }
  bool Evaluate(const std::vector<Real> &log_values,const PuncturePosition &puncture,
                Result &out) const {
    if(weights_.empty()||!puncture.Matches(stamp_)||log_values.size()!=weights_[0].size())return false;
    for(Real x:log_values)if(!std::isfinite(x))return false;
    std::array<Real,functionals> d{};const Real anchor=log_values[0];
    for(int f=0;f<functionals;++f)for(std::size_t i=0;i<log_values.size();++i)
      d[f]+=weights_[f][i]*(log_values[i]-anchor);
    for(Real value:d)if(!std::isfinite(value))return false;
    out.fitted_value=anchor+d[0];out.enrichment_coefficient=d[10];
    for(int a=0;a<3;++a)out.gradient[a]=d[1+a]/h_;
    int row=4;for(int a=0;a<3;++a)for(int b=a;b<3;++b,++row)
      out.hessian[a][b]=out.hessian[b][a]=d[row]/(h_*h_);
    if(!std::isfinite(out.fitted_value))return false;
    for(int a=0;a<3;++a){if(!std::isfinite(out.gradient[a]))return false;
      for(int b=0;b<3;++b)if(!std::isfinite(out.hessian[a][b]))return false;}
    return true;
  }
 private:
  static std::array<std::array<int,3>,polynomial_columns> Monomials(){
    std::array<std::array<int,3>,polynomial_columns> p{};int n=0;
    for(int degree=0;degree<=5;++degree)for(int a=0;a<=degree;++a)
      for(int b=0;b<=degree-a;++b)p[n++]={a,b,degree-a-b};return p;
  }
  Real h_=0;PuncturePosition::Stamp stamp_{};Diagnostics diagnostic_;
  std::vector<std::vector<Real>> weights_;
};
}}  // namespace z4c::hyperboloidal
#endif
