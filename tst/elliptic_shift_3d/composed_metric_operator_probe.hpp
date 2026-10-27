// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Standalone candidate only: Frechet derivative of the actual global metric
// difference operator, rather than a Leibniz-expanded second derivative.
#ifndef TST_ELLIPTIC_SHIFT_3D_COMPOSED_METRIC_OPERATOR_PROBE_HPP_
#define TST_ELLIPTIC_SHIFT_3D_COMPOSED_METRIC_OPERATOR_PROBE_HPP_
#include <map>
#include <optional>
#include "z4c/hyperboloidal/composite_shift.hpp"
#include "mesh/cubed_sphere/scri_surface.hpp"
#include "mesh/cubed_sphere/hermite_boundary_sampler.hpp"
namespace composed_metric_probe {
namespace h=z4c::hyperboloidal;namespace c=cubed_sphere;
using V=multigrid::Vector;using U=h::Z4cJet<Real>;
class Operator {
 public:
  const h::CompositeShiftStage&stage;const c::CompositeDifference&grid;const c::ScriSurface&surface;
  const std::vector<U> bulk,boundary_geometry;const size_t N,S;
  Operator(const h::CompositeShiftStage&s,const c::ScriSurface&b,std::vector<U>u,std::vector<U>v)
      :stage(s),grid(s.grid),surface(b),bulk(std::move(u)),boundary_geometry(std::move(v)),N(grid.active.size()),S(b.rows.size()){
    if(bulk.size()!=N||boundary_geometry.size()!=S||N>50000)throw std::invalid_argument("composed probe geometry shape/budget");
  }
  V Pad(const V&values,const V&trace,int components)const{
    if(values.size()!=N*components||trace.size()!=S*components)throw std::invalid_argument("composed probe pad shape");
    V out(components*grid.count,0);for(size_t p=0;p<N;++p)for(int f=0;f<components;++f)out[f*grid.count+grid.active[p]]=values[p*components+f];
    std::map<std::array<Real,3>,V>cache;grid.Fill(out,components,[&](int f,const auto&x){auto it=cache.find(x);if(it==cache.end())it=cache.emplace(x,surface.Interpolate(trace,components,x.data())).first;return it->second[f];});return out;
  }
  V Variation(const V&rates,const V&trace)const{
    const auto pad=Pad(rates,trace,6);V out(3*N);
    for(size_t p=0;p<N;++p){h::MetricRate<Real>r{};for(int t=0;t<6;++t){const int i=ti[t],j=tj[t];r.h[i][j]=r.h[j][i]=rates[6*p+t];Real d[3],dd[3][3];grid.Derivatives(pad,t,p,d,dd);for(int a=0;a<3;++a)r.dh[a][i][j]=r.dh[a][j][i]=d[a];}
      h::MetricConnectionVariation(bulk[p].metric,h::Geometry(bulk[p].metric),r,out.data()+3*p);}
    return out;
  }
  void Rates(const V&beta,V&rates,V&trace,const h::CompositeShiftStage::Boundary&bc={})const{
    if(beta.size()!=3*N)throw std::invalid_argument("composed beta shape");
    V pad,datum(3*S);std::optional<h::CartesianAffineDerivative::Field>affine;
    if(stage.affine_derivative){affine.emplace(stage.affine_derivative->Fit(beta));pad=stage.affine_derivative->PaddedRemainder(beta,*affine,bc);}
    else {pad.assign(3*grid.count,0);for(size_t p=0;p<N;++p)for(int v=0;v<3;++v)pad[v*grid.count+grid.active[p]]=beta[3*p+v];grid.Fill(pad,3,[&](int v,const auto&x){return bc?bc(x)[v]:0;});}
    for(size_t p=0;p<S;++p){const auto&x=surface.rows[p].geometry.x;const auto b=bc?bc({x[0],x[1],x[2]}):std::array<Real,3>{};const auto a=affine?stage.affine_derivative->At(*affine,x):h::VectorJet<Real>{};for(int v=0;v<3;++v)datum[3*p+v]=b[v]-a.value[v];}
    c::HermiteBoundarySampler sampler(surface);const auto prepared=sampler.Prepare(pad,datum,3,1);rates.resize(6*N);trace.resize(6*S);
    for(size_t p=0;p<N+S;++p){const bool edge=p>=N;const auto k=edge?p-N:p;h::VectorJet<Real>b{};
      if(edge){const auto sampled=sampler.Sample(k,prepared);const auto&x=surface.rows[k].geometry.x;const auto a=affine?stage.affine_derivative->At(*affine,x):h::VectorJet<Real>{};const auto v=bc?bc({x[0],x[1],x[2]}):std::array<Real,3>{};for(int i=0;i<3;++i){b.value[i]=v[i];for(int d=0;d<3;++d)b.d[d][i]=sampled.d[i][d]+a.d[d][i];}}
      else if(affine)b=stage.affine_derivative->Sample(beta,pad,*affine,k);
      else for(int i=0;i<3;++i){b.value[i]=beta[3*k+i];Real d[3],dd[3][3];grid.Derivatives(pad,i,k,d,dd);for(int a=0;a<3;++a)b.d[a][i]=d[a];}
      const auto r=h::ShiftMetricRate((edge?boundary_geometry[k]:bulk[k]).metric,b);for(int t=0;t<6;++t)(edge?trace:rates)[6*k+t]=r.h[ti[t]][tj[t]];
    }
  }
  void Apply(const V&beta,V&out,const h::CompositeShiftStage::Boundary&bc={})const{V active,trace;Rates(beta,active,trace,bc);out=Variation(active,trace);for(auto&x:out)x=-x;}
  // Actual frozen nonshift metric rate plus analytic reference beta=-x. No
  // physical RHS residual is subtracted and no field value is changed.
  V Source()const{V active(6*N),trace(6*S);for(size_t p=0;p<N+S;++p){bool edge=p>=N;size_t k=edge?p-N:p;const auto&u=edge?boundary_geometry[k]:bulk[k];const auto&x=edge?surface.rows[k].geometry.x:grid.geometry[k].x;h::VectorJet<Real>ref{};for(int v=0;v<3;++v){ref.value[v]=-x[v];ref.d[v][v]=-1;}const auto shift=h::ShiftMetricRate(u.metric,ref);for(int t=0;t<6;++t){const int i=ti[t],j=tj[t];(edge?trace:active)[6*k+t]=shift.h[i][j]-2*u.alpha.value*u.a.k[i][j];}}
    auto out=Variation(active,trace);for(auto&x:out)x=-x;return out;}
 private:
  static constexpr int ti[6]={0,0,0,1,1,2},tj[6]={0,1,2,1,2,2};
};
}
#endif
