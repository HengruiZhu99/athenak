// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_AFFINE_SHIFT_REMAINDER_HPP_
#define Z4C_HYPERBOLOIDAL_AFFINE_SHIFT_REMAINDER_HPP_
#include <algorithm>
#include <array>
#include <utility>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include "z4c/hyperboloidal/conformal_rhs.hpp"
#include "z4c/hyperboloidal/puncture_position.hpp"
namespace z4c { namespace hyperboloidal {
using AffineShiftPoint=std::array<Real,3>;

// Global affine Cartesian field plus an independently stored remainder. This
// owns no elliptic solve, evolved field, puncture parity or singular weight.
struct AffineShiftRemainder {
  PuncturePosition puncture;
  AffineShiftPoint b{};
  std::array<AffineShiftPoint,3> B{};  // B[component][direction]
  std::vector<AffineShiftPoint> u;
  VectorJet<Real> AffineJet(const Real x[3])const{
    const auto y=puncture.Relative(x);VectorJet<Real> out{};
    for(int i=0;i<3;++i){out.value[i]=b[i];for(int a=0;a<3;++a){
      out.value[i]+=B[i][a]*y[a];out.d[a][i]=B[i][a];}}
    return out;
  }
  // Differentiate u first. Never construct nodal beta and differentiate that.
  VectorJet<Real> ComposeJet(const Real x[3],const VectorJet<Real>& remainder)const{
    auto out=AffineJet(x);for(int i=0;i<3;++i){out.value[i]+=remainder.value[i];
      for(int a=0;a<3;++a){out.d[a][i]+=remainder.d[a][i];
        for(int d=0;d<3;++d)out.dd[a][d][i]=remainder.dd[a][d][i];}}return out;
  }
  AffineShiftPoint BoundaryRemainder(const Real x[3],const Real full_beta[3])const{
    const auto affine=AffineJet(x);AffineShiftPoint remainder;
    for(int i=0;i<3;++i)remainder[i]=full_beta[i]-affine.value[i];return remainder;
  }
  void Recenter(const PuncturePosition &next){
    next.Validate();puncture.Validate();AffineShiftPoint next_b=b;
    for(int i=0;i<3;++i)for(int a=0;a<3;++a)
      next_b[i]+=B[i][a]*(next.position[a]-puncture.position[a]);
    for(Real x:next_b)if(!std::isfinite(x))throw std::overflow_error("affine shift recenter overflow");
    b=next_b;puncture=next;
    // The global affine field and u are unchanged. If the moment support moves,
    // its canonical Mu=0 constraint must subsequently be rebalanced separately.
  }
};

// Compact CONFORMAL-volume moments, supplied measure sqrt(det(g_tilde))*dVx.
// No chi^-3/2 or Omega^-3 factor appears. Compact support selects a coordinate
// decomposition, not a physical-domain exclusion; every u sample is retained.
class AffineShiftMoments {
 public:
  AffineShiftMoments(const std::vector<AffineShiftPoint> &points,
      const std::vector<Real> &conformal_volumes,const PuncturePosition &point,
      Real radius,Real maximum_condition=1.e8):points_(points),radius_(radius),stamp_(point.Capture()){
    if(points.empty()||points.size()!=conformal_volumes.size()||!(radius>0)
       ||!std::isfinite(radius)||!(maximum_condition>0)||!std::isfinite(maximum_condition))
      throw std::invalid_argument("invalid affine moment geometry");
    phi_.resize(points.size());weight_.resize(points.size());long double sum=0;
    for(size_t p=0;p<points.size();++p){
      if(!(conformal_volumes[p]>0)||!std::isfinite(conformal_volumes[p]))
        throw std::invalid_argument("invalid conformal cell volume");
      const auto y=point.Relative(points[p].data());long double r2=0;phi_[p][0]=1;
      for(int a=0;a<3;++a){phi_[p][a+1]=static_cast<long double>(y[a])/radius_;r2+=phi_[p][a+1]*phi_[p][a+1];}
      const long double taper=r2<1?std::pow(1-r2,4):0;
      weight_[p]=static_cast<long double>(conformal_volumes[p])*taper;
      if(weight_[p]>0)++support_points_;sum+=weight_[p];
    }
    if(!(sum>0)||!std::isfinite(sum))throw std::invalid_argument("empty affine moment support");
    for(auto &w:weight_)w/=sum;
    for(size_t p=0;p<points.size();++p)for(int i=0;i<4;++i)for(int j=0;j<4;++j)
      gram_[i][j]+=weight_[p]*phi_[p][i]*phi_[p][j];
    for(int i=0;i<4;++i)for(int j=0;j<=i;++j){long double s=gram_[i][j];
      for(int k=0;k<j;++k)s-=cholesky_[i][k]*cholesky_[j][k];
      if(i==j){if(!(s>1.e-12L))throw std::invalid_argument("affine moment Gram rank deficient");cholesky_[i][j]=std::sqrt(s);}
      else cholesky_[i][j]=s/cholesky_[j][j];}
    long double gnorm=0;std::array<long double,4> inverse_rows{};
    for(int i=0;i<4;++i){long double row=0;for(int j=0;j<4;++j)row+=std::abs(gram_[i][j]);gnorm=std::max(gnorm,row);}
    for(int j=0;j<4;++j){std::array<long double,4> rhs{};rhs[j]=1;const auto inverse=Solve(rhs);
      for(int i=0;i<4;++i)inverse_rows[i]+=std::abs(inverse[i]);}
    condition_=static_cast<Real>(gnorm*(*std::max_element(inverse_rows.begin(),inverse_rows.end())));
    if(!std::isfinite(condition_)||condition_>maximum_condition)
      throw std::invalid_argument("affine moment Gram ill conditioned");
  }
  Real Condition()const{return condition_;}
  size_t SupportPoints()const{return support_points_;}
  size_t Size()const{return points_.size();}
  // This cannot restore a remainder already rounded out of physical nodal beta.
  AffineShiftRemainder FromNodal(const std::vector<AffineShiftPoint>& beta,
                               const PuncturePosition &point)const{
    AffineShiftRemainder chart;chart.puncture=point;chart.u=beta;Rebalance(chart);return chart;
  }
  void Rebalance(AffineShiftRemainder &chart)const{
    if(!chart.puncture.Matches(stamp_)||chart.u.size()!=points_.size())
      throw std::invalid_argument("stale affine moment chart/shape");
    for(const auto &v:chart.u)for(Real x:v)if(!std::isfinite(x))throw std::invalid_argument("nonfinite affine remainder");
    size_t anchor=0;while(weight_[anchor]==0)++anchor;
    std::array<std::array<long double,4>,3> correction{};
    for(int component=0;component<3;++component){std::array<long double,4> rhs{};
      const long double constant=chart.u[anchor][component];
      // Anchoring exactly preserves constants; only small remainder differences
      // enter when the chart has been stored separately from the start.
      for(size_t p=0;p<points_.size();++p)for(int i=0;i<4;++i)
        rhs[i]+=weight_[p]*phi_[p][i]*(static_cast<long double>(chart.u[p][component])-constant);
      correction[component]=Solve(rhs);correction[component][0]+=constant;
    }
    auto next=chart;
    for(int i=0;i<3;++i){next.b[i]+=static_cast<Real>(correction[i][0]);
      for(int a=0;a<3;++a)next.B[i][a]+=static_cast<Real>(correction[i][a+1]/radius_);}
    for(size_t p=0;p<points_.size();++p)for(int i=0;i<3;++i){long double affine=0;
      for(int a=0;a<4;++a)affine+=correction[i][a]*phi_[p][a];
      next.u[p][i]=static_cast<Real>(static_cast<long double>(chart.u[p][i])-affine);
      if(!std::isfinite(next.u[p][i]))throw std::overflow_error("affine projection overflow");}
    for(Real x:next.b)if(!std::isfinite(x))throw std::overflow_error("affine coefficient overflow");
    for(const auto &row:next.B)for(Real x:row)if(!std::isfinite(x))throw std::overflow_error("affine gradient overflow");
    chart=std::move(next);
  }
  Real MaximumMoment(const AffineShiftRemainder &chart)const{
    if(!chart.puncture.Matches(stamp_)||chart.u.size()!=points_.size())
      throw std::invalid_argument("stale affine moment diagnostic");
    long double maximum=0;
    for(int c=0;c<3;++c)for(int a=0;a<4;++a){long double moment=0;
      for(size_t p=0;p<points_.size();++p)moment+=weight_[p]*phi_[p][a]*chart.u[p][c];
      maximum=std::max(maximum,std::abs(moment));}return static_cast<Real>(maximum);
  }
 private:
  std::array<long double,4> Solve(const std::array<long double,4>&rhs)const{
    std::array<long double,4> y{},x{};
    for(int i=0;i<4;++i){long double s=rhs[i];for(int j=0;j<i;++j)s-=cholesky_[i][j]*y[j];y[i]=s/cholesky_[i][i];}
    for(int i=3;i>=0;--i){long double s=y[i];for(int j=i+1;j<4;++j)s-=cholesky_[j][i]*x[j];x[i]=s/cholesky_[i][i];}return x;
  }
  std::vector<AffineShiftPoint> points_;
  Real radius_,condition_=0;
  PuncturePosition::Stamp stamp_;
  size_t support_points_=0;
  std::vector<std::array<long double,4>> phi_;
  std::vector<long double> weight_;
  std::array<std::array<long double,4>,4> gram_{},cholesky_{};
};
}}  // namespace z4c::hyperboloidal
#endif
