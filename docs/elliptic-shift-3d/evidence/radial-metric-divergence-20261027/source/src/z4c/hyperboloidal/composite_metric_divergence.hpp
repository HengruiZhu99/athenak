// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_COMPOSITE_METRIC_DIVERGENCE_HPP_
#define Z4C_HYPERBOLOIDAL_COMPOSITE_METRIC_DIVERGENCE_HPP_

#include <array>
#include <functional>
#include <map>
#include "mesh/cubed_sphere/scri_surface.hpp"
#include "z4c/hyperboloidal/elliptic_shift.hpp"

namespace z4c { namespace hyperboloidal {

// Diagnostic prototype, NOT a production replacement for CompositeShiftStage.
// V_D(g) uses the sampled inverse metric and the indicated discrete first
// derivative of g. Variation computes its exact Frechet derivative DV_D[h],
// including inverse/determinant variations. BOTH operator H_g(beta) and source
// C=-2 alpha A pass through precisely this same DV_D; no product-rule source
// derivative or conformal-flatness approximation is made.
//
// Centered composition has an explicit bulk checkerboard null and is an error
// localization control. ForwardAdjoint uses H(D+) and DV(D-), with fourth-order
// one-sided local derivatives D-=-D+^T on a uniform periodic Cartesian grid.
// Its flat symbol is coercive: Re <b,Lb> <= -|d|^2 |b|^2/3. This is NOT a proof
// of coercivity for variable coefficients, mapped interfaces or face closures.
//
// Intermediate tensor rates receive a fresh active-only halo exchange, so the
// centered control needs ghost2, not an extended ghost4 stencil. The biased
// pair needs ghost4. Face H is computed from direct beta data and reconstructed
// derivatives, not independently prescribed as a second elliptic BC.
class CompositeMetricDivergence {
 public:
  using Vector=std::vector<Real>;
  using Tensor=std::array<Real,6>;
  using TensorBoundary=std::function<Tensor(const std::array<Real,3>&)>;
  using VectorBoundary=std::function<std::array<Real,3>(const std::array<Real,3>&)>;
  using VectorReference=std::function<VectorJet<Real>(const Real*)>;
  enum class Pair {Centered,ForwardAdjoint};
  struct TensorField {Vector active,surface;};  // point-major six symmetric Cartesian components
  const cubed_sphere::CompositeDifference &grid;
  const Pair pair;
  cubed_sphere::ScriSurface face;
  inline static constexpr int ti[6]={0,0,0,1,1,2},tj[6]={0,1,2,1,2,2};

  explicit CompositeMetricDivergence(const cubed_sphere::CompositeDifference &g,Pair p=Pair::Centered)
      :grid(g),pair(p),face(g,6,6) {
    if(grid.order!=4||grid.ng<(pair==Pair::Centered?2:4))
      throw std::invalid_argument("metric-divergence prototype requires order4, ghost2 centered or ghost4 biased");
    BuildFaceFirst();
  }
  TensorField SampleTensor(const TensorBoundary &data) const {
    if(!data)throw std::invalid_argument("missing tensor data");
    TensorField out;out.active.resize(6*grid.active.size());out.surface.resize(6*face.rows.size());
    for(size_t p=0;p<grid.active.size();++p){const auto &x=grid.geometry[p].x;const auto t=data({x[0],x[1],x[2]});for(int f=0;f<6;++f)out.active[6*p+f]=t[f];}
    for(size_t p=0;p<face.rows.size();++p){const auto &x=face.rows[p].geometry.x;const auto t=data({x[0],x[1],x[2]});for(int f=0;f<6;++f)out.surface[6*p+f]=t[f];}
    Validate(out);return out;
  }
  void SetGeometry(TensorField metric) {
    Validate(metric);metric_values_=std::move(metric);metric_padded_=Pad(metric_values_);
    metric_h_.assign(grid.active.size(),{});metric_dv_.assign(grid.active.size(),{});geometry_.resize(grid.active.size());
    for(size_t p=0;p<grid.active.size();++p){
      for(int f=0;f<6;++f){const int i=ti[f],j=tj[f];const Real value=metric_values_.active[6*p+f];
        metric_h_[p].g[i][j]=metric_h_[p].g[j][i]=metric_dv_[p].g[i][j]=metric_dv_[p].g[j][i]=value;
        const auto dh=Derivative(metric_padded_,f,p,false),dv=Derivative(metric_padded_,f,p,true);
        for(int a=0;a<3;++a){metric_h_[p].dg[a][i][j]=metric_h_[p].dg[a][j][i]=dh[a];metric_dv_[p].dg[a][i][j]=metric_dv_[p].dg[a][j][i]=dv[a];}}
      // Only inverse, connection and contracted connection enter this class.
      // Its zero second-jet Ricci scratch is NEVER exposed as a diagnostic.
      geometry_[p]=Geometry(metric_dv_[p]);if(!geometry_[p].valid)throw std::runtime_error("invalid discrete connection metric");
    }
  }
  Vector Connection() const {
    Ready();Vector out(3*grid.active.size());for(size_t p=0;p<grid.active.size();++p)for(int i=0;i<3;++i)out[3*p+i]=geometry_[p].contracted[i];return out;
  }
  // Exact derivative of the DISCRETE connection, with its linear tensor ghost
  // and face reconstruction held fixed. A centered finite metric perturbation
  // audit independently checks this identity for determinant-changing rates.
  void Variation(const TensorField &rate,Vector &out) const {
    Ready();Validate(rate);const auto padded=Pad(rate);out.resize(3*grid.active.size());
    for(size_t p=0;p<grid.active.size();++p){MetricRate<Real> h{};
      for(int f=0;f<6;++f){const int i=ti[f],j=tj[f];h.h[i][j]=h.h[j][i]=rate.active[6*p+f];const auto d=Derivative(padded,f,p,true);
        for(int a=0;a<3;++a)h.dh[a][i][j]=h.dh[a][j][i]=d[a];}
      Real value[3];MetricConnectionVariation(metric_dv_[p],geometry_[p],h,value);for(int i=0;i<3;++i)out[3*p+i]=value[i];
    }
  }
  TensorField ShiftRate(const Vector &beta,const VectorBoundary &boundary={},const VectorReference &reference={}) const {
    Ready();if(beta.size()!=3*grid.active.size())throw std::invalid_argument("metric-divergence beta shape");
    Vector padded(3*grid.count,0),surface(3*face.rows.size());
    for(size_t p=0;p<grid.active.size();++p)for(int i=0;i<3;++i)padded[i*grid.count+grid.active[p]]=beta[3*p+i];
    grid.Fill(padded,3,[&](int i,const auto &x){return boundary?boundary(x)[i]:Real(0);});
    for(size_t p=0;p<face.rows.size();++p){const auto &x=face.rows[p].geometry.x;const auto b=boundary?boundary({x[0],x[1],x[2]}):std::array<Real,3>{};
      for(int i=0;i<3;++i)surface[3*p+i]=b[i];}
    TensorField out;out.active.resize(6*grid.active.size());out.surface.resize(6*face.rows.size());
    for(size_t p=0;p<grid.active.size();++p){VectorJet<Real> b=reference?reference(grid.geometry[p].x):VectorJet<Real>{};
      for(int i=0;i<3;++i){b.value[i]+=beta[3*p+i];const auto d=Derivative(padded,i,p,false);for(int a=0;a<3;++a)b.d[a][i]+=d[a];}
      const auto h=ShiftMetricRate(metric_h_[p],b);for(int f=0;f<6;++f)out.active[6*p+f]=h.h[ti[f]][tj[f]];
    }
    for(size_t p=0;p<face.rows.size();++p){const auto &x=face.rows[p].geometry.x;MetricJet<Real> g{};
      VectorJet<Real> b=reference?reference(x):VectorJet<Real>{};
      for(int i=0;i<3;++i){b.value[i]+=surface[3*p+i];const auto d=FaceDerivative(p,padded,surface,3,i);for(int a=0;a<3;++a)b.d[a][i]+=d[a];}
      for(int f=0;f<6;++f){const int i=ti[f],j=tj[f];g.g[i][j]=g.g[j][i]=metric_values_.surface[6*p+f];const auto d=FaceDerivative(p,metric_padded_,metric_values_.surface,6,f);
        for(int a=0;a<3;++a)g.dg[a][i][j]=g.dg[a][j][i]=d[a];}
      const auto h=ShiftMetricRate(g,b);for(int f=0;f<6;++f)out.surface[6*p+f]=h.h[ti[f]][tj[f]];
    }
    Validate(out);return out;
  }
  TensorField CurvatureRate(const std::vector<Z4cJet<Real>> &u,const TensorBoundary &face_rate) const {
    if(u.size()!=grid.active.size()||!face_rate)throw std::invalid_argument("curvature metric-rate data");
    TensorField out;out.active.resize(6*u.size());out.surface.resize(6*face.rows.size());
    for(size_t p=0;p<u.size();++p)for(int f=0;f<6;++f)out.active[6*p+f]=-2*u[p].alpha.value*u[p].a.k[ti[f]][tj[f]];
    for(size_t p=0;p<face.rows.size();++p){const auto &x=face.rows[p].geometry.x;const auto c=face_rate({x[0],x[1],x[2]});for(int f=0;f<6;++f)out.surface[6*p+f]=c[f];}
    Validate(out);return out;
  }
  // Raw L beta (no minus sign). With nonzero direct boundary/reference values
  // this is affine; a Krylov caller must subtract Apply(0)'s known lift.
  void Apply(const Vector &beta,Vector &out,const VectorBoundary &boundary={},const VectorReference &reference={}) const {
    Variation(ShiftRate(beta,boundary,reference),out);
  }
  void Source(const TensorField &curvature,Real eta,Vector &out) const {
    if(!(eta>=0)||!std::isfinite(eta))throw std::invalid_argument("metric-connection damping");
    Variation(curvature,out);for(size_t p=0;p<grid.active.size();++p)for(int i=0;i<3;++i)out[3*p+i]=-out[3*p+i]-eta*geometry_[p].contracted[i];
  }
  // The optional analytic reference is only a derivative factorization (e.g.
  // beta=-x+delta); its FULL metric rate remains in Apply and the residual.
  // No reference RHS subtraction, potential residual or gauge change occurs.

 private:
  struct FaceTerm {int point;bool boundary;std::array<Real,3> d{};};
  std::vector<std::vector<FaceTerm>> face_first_;
  TensorField metric_values_;
  Vector metric_padded_;
  std::vector<MetricJet<Real>> metric_h_,metric_dv_;
  std::vector<SpatialGeometry<Real>> geometry_;
  void Ready() const {if(geometry_.size()!=grid.active.size())throw std::logic_error("frozen discrete metric missing");}
  void Validate(const TensorField &t) const {
    if(t.active.size()!=6*grid.active.size()||t.surface.size()!=6*face.rows.size())throw std::invalid_argument("metric-rate tensor shape");
    for(const auto *data:{&t.active,&t.surface})for(Real v:*data)if(!std::isfinite(v))throw std::invalid_argument("nonfinite metric-rate tensor");
  }
  Vector Pad(const TensorField &tensor) const {
    Vector padded(6*grid.count,0);for(size_t p=0;p<grid.active.size();++p)for(int f=0;f<6;++f)padded[f*grid.count+grid.active[p]]=tensor.active[6*p+f];
    std::map<std::array<Real,3>,Vector> cache;
    grid.Fill(padded,6,[&](int f,const auto &x){auto found=cache.find(x);if(found==cache.end())found=cache.emplace(x,face.Interpolate(tensor.surface,6,x.data())).first;return found->second[f];});return padded;
  }
  std::array<Real,3> Derivative(const Vector &padded,int f,size_t p,bool outer) const {
    std::array<Real,3> result{};const int point=f*grid.count+grid.active[p];
    for(int A=0;A<3;++A){Real d=0;
      if(pair==Pair::Centered)d=grid.n*(8*(padded[point+grid.Offset(A,1)]-padded[point+grid.Offset(A,-1)])
          -(padded[point+grid.Offset(A,2)]-padded[point+grid.Offset(A,-2)]))/12;
      else{constexpr Real w[5]={-25./12,4,-3,4./3,-.25};const int direction=outer?-1:1;
        for(int k=1;k<=4;++k)d+=direction*w[k]*(padded[point+grid.Offset(A,direction*k)]-padded[point]);d*=grid.n;}
      for(int a=0;a<3;++a)result[a]+=grid.geometry[p].J[A][a]*d;
    }
    return result;
  }
  std::array<Real,3> FaceDerivative(size_t p,const Vector &padded,const Vector &surface,int fields,int f) const {
    if(padded.size()!=size_t(fields*grid.count)||surface.size()!=size_t(fields*face.rows.size()))throw std::invalid_argument("face derivative field shape");
    const Real anchor=surface[p*fields+f];std::array<Real,3> result{};
    for(const auto &term:face_first_[p]){const Real value=term.boundary?surface[term.point*fields+f]:padded[f*grid.count+term.point];
      for(int a=0;a<3;++a)result[a]+=term.d[a]*(value-anchor);}
    return result;
  }
  void BuildFaceFirst() {
    // A degree5 active+face polynomial supplies boundary derivatives. Apply the
    // SAME D4 symbol to it, including its fifth-derivative truncation term; using
    // only its exact first derivative would mismatch volume H by O(h^4) and can
    // reduce the composed boundary operator to third order.
    const int n=grid.n;const Real correction=(pair==Pair::Centered?Real(-1./30):Real(-1./5))/std::pow(Real(n),4);
    face_first_.resize(face.rows.size());
    for(size_t p=0;p<face.rows.size();++p){const auto &row=face.rows[p];
      const int begin[3]={n-5,std::clamp(row.j-2,0,n-6),std::clamp(row.k-2,0,n-6)};
      const Real q[3]={1,(row.j+.5)/n,(row.k+.5)/n};cubed_sphere::Jet basis[3][6];Real fifth[3][6];
      for(int A=0;A<3;++A)for(int i=0;i<6;++i){const Real node=A==0&&i==5?1:Real(begin[A]+i+.5)/n;basis[A][i]=cubed_sphere::Jet(1);Real denom=1;
        for(int j=0;j<6;++j)if(i!=j){const Real other=A==0&&j==5?1:Real(begin[A]+j+.5)/n;
          basis[A][i]=basis[A][i]*(cubed_sphere::Jet::Variable(q[A],A)-cubed_sphere::Jet(other))/cubed_sphere::Jet(node-other);denom*=node-other;}
        fifth[A][i]=120/denom;}
      for(int k=0;k<6;++k)for(int j=0;j<6;++j)for(int i=0;i<6;++i){const int index[3]={i,j,k};FaceTerm term;term.boundary=i==5;
        term.point=term.boundary?face.base[row.block]+begin[1]+j+n*(begin[2]+k):grid.Index(row.block,begin[0]+i,begin[1]+j,begin[2]+k);
        for(int A=0;A<3;++A){Real weight=basis[A][index[A]].d[A]+correction*fifth[A][index[A]];
          for(int B=0;B<3;++B)if(A!=B)weight*=basis[B][index[B]].f;
          for(int a=0;a<3;++a)term.d[a]+=row.geometry.J[A][a]*weight;}
        face_first_[p].push_back(term);
      }
    }
  }
};
}}  // namespace z4c::hyperboloidal
#endif
