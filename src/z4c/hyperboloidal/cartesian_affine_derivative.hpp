// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_CARTESIAN_AFFINE_DERIVATIVE_HPP_
#define Z4C_HYPERBOLOIDAL_CARTESIAN_AFFINE_DERIVATIVE_HPP_
#include "mesh/cubed_sphere/composite_difference.hpp"
#include "z4c/hyperboloidal/elliptic_shift.hpp"
namespace z4c {namespace hyperboloidal {
// Host fine-discretization option, independent of the scalar preconditioner.
// P is a fixed Cartesian-volume linear projection onto affine vector fields.
// Dnew = Danalytic P + Dmapped(1−P). No physical field is projected/reset.
class CartesianAffineDerivative {
 public:
  using Vector=std::vector<double>;
  using Boundary=std::function<std::array<Real,3>(const std::array<Real,3>&)>;
  struct Field {Real mean[3]{},slope[3][3]{};const CartesianAffineDerivative*owner=nullptr;unsigned long revision=0;};
  explicit CartesianAffineDerivative(const cubed_sphere::CompositeDifference&g)
      :grid(g),descriptor(g.mesh),active_storage(g.active.data()),geometry_storage(g.geometry.data()),
       volume_storage(g.volumes.data()),n(g.n),ng(g.ng),padded(g.padded),count(g.count),order(g.order){
    if(std::numeric_limits<Real>::digits<50)throw std::invalid_argument("affine host derivative requires double precision");
    volume=0;for(Real w:g.volumes){if(!(w>0)||!std::isfinite(w))throw std::invalid_argument("affine projection volume");volume+=w;}
    if(g.active.empty()||g.geometry.size()!=g.active.size()||g.volumes.size()!=g.active.size()||!std::isfinite(volume))throw std::invalid_argument("affine projection geometry");
    for(int d=0;d<3;++d){const Real anchor=g.geometry[0].x[d];center[d]=anchor;
      for(size_t p=0;p<g.active.size();++p)center[d]+=g.volumes[p]/volume*(g.geometry[p].x[d]-anchor);}
    MetricJet<Real>covariance{};
    for(size_t p=0;p<g.active.size();++p)for(int i=0;i<3;++i)for(int j=0;j<3;++j)
      covariance.g[i][j]+=g.volumes[p]/volume*(g.geometry[p].x[i]-center[i])*(g.geometry[p].x[j]-center[j]);
    const auto inverse=Geometry(covariance);if(!inverse.valid)throw std::invalid_argument("affine projection rank");
    Real norm=0,inorm=0;for(int i=0;i<3;++i){Real a=0,b=0;for(int j=0;j<3;++j){a+=std::abs(covariance.g[i][j]);b+=std::abs(inverse.inverse[i][j]);cov_inverse[i][j]=inverse.inverse[i][j];}norm=std::max(norm,a);inorm=std::max(inorm,b);}
    condition=norm*inorm;if(!std::isfinite(condition)||condition>1e4)throw std::invalid_argument("affine projection conditioning");
  }
  CartesianAffineDerivative(cubed_sphere::CompositeDifference&&)=delete;
  CartesianAffineDerivative(const CartesianAffineDerivative&)=delete;
  CartesianAffineDerivative&operator=(const CartesianAffineDerivative&)=delete;
  Field Fit(const Vector&beta)const{
    Check();if(beta.size()!=3*grid.active.size())throw std::invalid_argument("affine projection shape");
    for(Real x:beta)if(!std::isfinite(x))throw std::invalid_argument("nonfinite affine projection data");
    Field a;a.owner=this;a.revision=grid.mesh.revision;
    for(int v=0;v<3;++v){a.mean[v]=beta[v];for(size_t p=0;p<grid.active.size();++p)a.mean[v]+=grid.volumes[p]/volume*(beta[3*p+v]-beta[v]);}
    Real moment[3][3]{};for(size_t p=0;p<grid.active.size();++p)for(int v=0;v<3;++v)for(int d=0;d<3;++d)
      moment[v][d]+=grid.volumes[p]/volume*(grid.geometry[p].x[d]-center[d])*(beta[3*p+v]-a.mean[v]);
    for(int v=0;v<3;++v)for(int d=0;d<3;++d)for(int j=0;j<3;++j)a.slope[v][d]+=moment[v][j]*cov_inverse[j][d];return a;
  }
  VectorJet<Real> At(const Field&a,const Real*x)const{
    if(a.owner!=this||a.revision!=grid.mesh.revision)throw std::logic_error("foreign/stale affine field");
    VectorJet<Real>q{};for(int v=0;v<3;++v){q.value[v]=a.mean[v];for(int d=0;d<3;++d){q.value[v]+=a.slope[v][d]*(x[d]-center[d]);q.d[d][v]=a.slope[v][d];}}return q;
  }
  std::vector<Real>PaddedRemainder(const Vector&beta,const Field&a,const Boundary&boundary={})const{
    if(beta.size()!=3*grid.active.size())throw std::invalid_argument("affine remainder shape");std::vector<Real>r(3*grid.count,0);
    for(size_t p=0;p<grid.active.size();++p){const auto q=At(a,grid.geometry[p].x);for(int v=0;v<3;++v)r[v*grid.count+grid.active[p]]=beta[3*p+v]-q.value[v];}
    grid.Fill(r,3,[&](int v,const auto&x){const auto q=At(a,x.data());return(boundary?boundary(x)[v]:0)-q.value[v];});return r;
  }
  VectorJet<Real> Sample(const Vector&beta,const std::vector<Real>&remainder,const Field&a,size_t p)const{
    if(p>=grid.active.size()||beta.size()!=3*grid.active.size()||remainder.size()!=size_t(3)*grid.count)throw std::invalid_argument("affine derivative sample shape");
    auto q=At(a,grid.geometry[p].x);for(int v=0;v<3;++v){q.value[v]=beta[3*p+v];Real d[3],dd[3][3];grid.Derivatives(remainder,v,p,d,dd);
      for(int i=0;i<3;++i){q.d[i][v]+=d[i];for(int j=0;j<3;++j)q.dd[i][j][v]+=dd[i][j];}}return q;
  }
  Real condition=0;
 private:
  void Check()const{
    if(grid.n!=n||grid.ng!=ng||grid.padded!=padded||grid.count!=count||grid.order!=order||grid.mesh.cells!=descriptor.cells||grid.mesh.ghosts!=descriptor.ghosts
       ||grid.mesh.revision!=descriptor.revision||grid.mesh.radii!=descriptor.radii||grid.mesh.face_subdivisions!=descriptor.face_subdivisions||grid.mesh.half_width!=descriptor.half_width
       ||grid.mesh.blocks.size()!=descriptor.blocks.size()||grid.active.data()!=active_storage||grid.geometry.data()!=geometry_storage||grid.volumes.data()!=volume_storage
       ||grid.geometry.size()!=grid.active.size()||grid.volumes.size()!=grid.active.size())throw std::logic_error("stale affine derivative geometry");
    for(size_t m=0;m<descriptor.blocks.size();++m){const auto&a=descriptor.blocks[m];const auto&b=grid.mesh.blocks[m];
      if(a.face!=b.face||a.layer!=b.layer||a.refinement!=b.refinement)throw std::logic_error("stale affine derivative topology");
      for(int d=0;d<3;++d)if(a.lo[d]!=b.lo[d]||a.hi[d]!=b.hi[d])throw std::logic_error("stale affine derivative map");}
  }
  const cubed_sphere::CompositeDifference&grid;const cubed_sphere::CompositeMesh descriptor;
  const int*active_storage;const cubed_sphere::Point*geometry_storage;const Real*volume_storage;
  int n,ng,padded,count,order;Real center[3]{},cov_inverse[3][3]{},volume=0;
};
}}
#endif
