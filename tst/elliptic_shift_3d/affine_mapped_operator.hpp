// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef TST_ELLIPTIC_SHIFT_3D_AFFINE_MAPPED_OPERATOR_HPP_
#define TST_ELLIPTIC_SHIFT_3D_AFFINE_MAPPED_OPERATOR_HPP_
#include "z4c/hyperboloidal/composite_shift.hpp"
namespace affine_mapped_probe {
namespace h=z4c::hyperboloidal;
// A fixed, field-independent Cartesian-volume projection onto global affine
// fields. This is a discretization probe, not an enabled evolution chart.
class Operator {
 public:
  struct Affine {Real mean[3]{},slope[3][3]{};};
  explicit Operator(const h::CompositeShiftStage&s):stage(s){
    const auto&g=stage.grid;volume=0;for(Real w:g.volumes)volume+=w;
    if(!(volume>0))throw std::invalid_argument("affine projection volume");
    for(int d=0;d<3;++d){const Real anchor=g.geometry[0].x[d];center[d]=anchor;
      for(size_t p=0;p<g.active.size();++p)center[d]+=g.volumes[p]/volume*(g.geometry[p].x[d]-anchor);}
    h::MetricJet<Real>covariance{};
    for(size_t p=0;p<g.active.size();++p)for(int i=0;i<3;++i)for(int j=0;j<3;++j)
      covariance.g[i][j]+=g.volumes[p]/volume*(g.geometry[p].x[i]-center[i])*(g.geometry[p].x[j]-center[j]);
    const auto inverse=h::Geometry(covariance);if(!inverse.valid)throw std::invalid_argument("affine projection rank");
    Real norm=0,inorm=0;for(int i=0;i<3;++i){Real a=0,b=0;for(int j=0;j<3;++j){
      a+=std::abs(covariance.g[i][j]);b+=std::abs(inverse.inverse[i][j]);cov_inverse[i][j]=inverse.inverse[i][j];}
      norm=std::max(norm,a);inorm=std::max(inorm,b);}
    condition=norm*inorm;if(!std::isfinite(condition)||condition>1e4)throw std::invalid_argument("affine projection conditioning");
  }
  Affine Fit(const multigrid::Vector&beta)const{
    const auto&g=stage.grid;if(beta.size()!=3*g.active.size())throw std::invalid_argument("affine projection shape");Affine a;
    for(int v=0;v<3;++v){a.mean[v]=beta[v];for(size_t p=0;p<g.active.size();++p)
      a.mean[v]+=g.volumes[p]/volume*(beta[3*p+v]-beta[v]);}
    Real moments[3][3]{};
    for(size_t p=0;p<g.active.size();++p)for(int v=0;v<3;++v)for(int d=0;d<3;++d)
      moments[v][d]+=g.volumes[p]/volume*(g.geometry[p].x[d]-center[d])*(beta[3*p+v]-a.mean[v]);
    for(int v=0;v<3;++v)for(int d=0;d<3;++d)for(int j=0;j<3;++j)a.slope[v][d]+=moments[v][j]*cov_inverse[j][d];return a;
  }
  h::VectorJet<Real> At(const Affine&a,const Real*x)const{
    h::VectorJet<Real>q{};for(int v=0;v<3;++v){q.value[v]=a.mean[v];for(int d=0;d<3;++d){q.value[v]+=a.slope[v][d]*(x[d]-center[d]);q.d[d][v]=a.slope[v][d];}}return q;
  }
  void Apply(const multigrid::Vector&beta,multigrid::Vector&out,const h::CompositeShiftStage::Boundary&boundary={})const{
    const auto&g=stage.grid;const auto affine=Fit(beta);std::vector<Real>remainder(3*g.count,0);
    for(size_t p=0;p<g.active.size();++p){const auto a=At(affine,g.geometry[p].x);
      for(int v=0;v<3;++v)remainder[v*g.count+g.active[p]]=beta[3*p+v]-a.value[v];}
    g.Fill(remainder,3,[&](int v,const auto&x){const auto a=At(affine,x.data());return(boundary?boundary(x)[v]:0)-a.value[v];});
    out.resize(beta.size());for(size_t p=0;p<g.active.size();++p){auto jet=At(affine,g.geometry[p].x);
      for(int v=0;v<3;++v){jet.value[v]=beta[3*p+v];Real d[3],dd[3][3];g.Derivatives(remainder,v,p,d,dd);
        for(int i=0;i<3;++i){jet.d[i][v]+=d[i];for(int j=0;j<3;++j)jet.dd[i][j][v]+=dd[i][j];}}
      Real value[3];h::ApplyEllipticShift(stage.metric[p],stage.geometry[p],jet,value);
      for(int v=0;v<3;++v)out[3*p+v]=-value[v];}
  }
  Real condition=0;
 private:
  const h::CompositeShiftStage&stage;Real center[3]{},cov_inverse[3][3]{},volume=0;
};
}
#endif
