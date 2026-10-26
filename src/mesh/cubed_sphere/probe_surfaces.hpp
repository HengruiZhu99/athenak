#ifndef MESH_CUBED_SPHERE_PROBE_SURFACES_HPP_
#define MESH_CUBED_SPHERE_PROBE_SURFACES_HPP_
#include "z4c/hyperboloidal/cartesian_hawking.hpp"
namespace cubed_sphere {
struct ProbeSurfacePoint {
  Real area=0,correction=0,plus=0,minus=0,lapse=0,proper=0;
};
KOKKOS_INLINE_FUNCTION ProbeSurfacePoint SurfacePoint(
    const z4c::hyperboloidal::Z4cJet<Real> &u,
    const z4c::hyperboloidal::OmegaJet<Real> &o,const Real x[3]) {
  namespace h=z4c::hyperboloidal;
  ProbeSurfacePoint p;
  const auto d=h::CoordinateSphereDensity(u,o,x);
  if (!d.valid) {p.area=-1;return p;}
  const auto g=h::Geometry(h::PenroseMetric(u.metric,u.chi));
  Real r2=0,norm2=0,s[3]{};
  for(int a=0;a<3;++a) {r2+=x[a]*x[a];
    for(int b=0;b<3;++b) {s[a]+=g.inverse[a][b]*x[b];norm2+=g.inverse[a][b]*x[a]*x[b];}}
  const Real norm=sqrt(norm2);for(int a=0;a<3;++a)s[a]/=norm;
  Real div=0,no=0,at=0,grr=0;
  for(int a=0;a<3;++a) {no+=s[a]*o.gradient[a];
    for(int b=0;b<3;++b) {
      Real Hess=(a==b ? 1:0);
      for(int c=0;c<3;++c)Hess-=g.connection[c][a][b]*x[c];
      div+=(g.inverse[a][b]-s[a]*s[b])*Hess/norm;
      at+=u.a.k[a][b]*(s[a]*s[b]-g.inverse[a][b]);
      grr+=u.metric.g[a][b]*x[a]*x[b]/(r2*u.chi.value*o.omega*o.omega);
    }}
  const Real curvature=o.omega*div-2*no;
  const Real transverse=o.omega*at/u.chi.value-Real(2)/3*(u.trace.value+2*u.theta.value);
  p.area=d.area;p.correction=d.correction;p.plus=transverse+curvature;
  p.minus=transverse-curvature;p.lapse=u.alpha.value/o.omega;p.proper=sqrt(grr);
  return p;
}
// Tensor cubic interpolation on a uniform Cartesian bulk. Near cube faces the
// four active nodes are one-sided; evaluations stay inside the physical cube.
// This is a diagnostic interpolation, never an evolution or boundary operator.
struct BulkSurfaceSampler {
  int n;Real L;std::vector<ProbeSurfacePoint> points;
  ProbeSurfacePoint operator()(const double x[3]) const {
    int first[3];Real w[3][4];const Real dx=2*L/n;
    for(int d=0;d<3;++d) {
      if(fabs(x[d])>L)throw std::runtime_error("surface leaves Cartesian bulk");
      const Real t=(x[d]+L)/dx-.5;
      first[d]=std::max(0,std::min(n-4,int(floor(t))-1));
      for(int a=0;a<4;++a) {w[d][a]=1;
        for(int b=0;b<4;++b)if(a!=b)w[d][a]*=(t-first[d]-b)/(a-b);}
    }
    ProbeSurfacePoint v;
    for(int k=0;k<4;++k)for(int j=0;j<4;++j)for(int i=0;i<4;++i) {
      const auto &p=points[((first[2]+k)*n+first[1]+j)*n+first[0]+i];
      if(!(p.area>0))throw std::runtime_error("invalid surface interpolation node");
      const Real weight=w[0][i]*w[1][j]*w[2][k];
      v.area+=weight*p.area;v.correction+=weight*p.correction;
      v.plus+=weight*p.plus;v.minus+=weight*p.minus;
      v.lapse+=weight*p.lapse;v.proper+=weight*p.proper;
    }
    return v;
  }
};
struct ProbeSphere {Real r,area,areal,mass,plus,lo,hi,lapse,proper;};
inline ProbeSphere MeasureProbeSphere(const BulkSurfaceSampler &sample,Real r,int nq=16) {
  Real mean=0,lo=1e100,hi=-1e100,lapse=0,proper=0;int count=0;
  // Equal solid-angle midpoint quadrature for expansion/geometry. Hawking
  // area/mass independently use Gauss-Legendre, with a doubled-order control.
  const Real pi=acos(-1.);
  for(int i=0;i<nq;++i)for(int j=0;j<2*nq;++j) {
    Real mu=-1+2*(i+.5)/nq,phi=pi*(j+.5)/nq;
    double x[3]={r*sqrt(1-mu*mu)*cos(phi),r*sqrt(1-mu*mu)*sin(phi),r*mu};
    auto p=sample(x);mean+=p.plus;lo=std::min(lo,p.plus);hi=std::max(hi,p.plus);
    lapse+=p.lapse;proper+=p.proper;++count;
  }
  auto mass=z4c::hyperboloidal::IntegrateHawkingMass(r,nq,[&](const double x[3]) {
    auto p=sample(x);z4c::hyperboloidal::HawkingDensity<double> d;
    d.area=p.area;d.correction=p.correction;d.valid=p.area>0 && std::isfinite(p.correction);return d;});
  return {r,mass.area,mass.areal_radius,mass.mass,mean/count,lo,hi,lapse/count,proper/count};
}
}
#endif
