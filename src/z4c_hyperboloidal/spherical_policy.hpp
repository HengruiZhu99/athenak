// Ported from scri-ko01 src/z4c/hyperboloidal/spherical_policy.hpp; mathematical
// expressions retained. Native integration has no legacy driver dependency.
// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_NATIVE_SPHERICAL_POLICY_HPP_
#define Z4C_HYPERBOLOIDAL_NATIVE_SPHERICAL_POLICY_HPP_
#include "z4c_hyperboloidal/cartesian_algebraic_chart.hpp"
#include "z4c_hyperboloidal/hyperbolic_shift.hpp"
#include "z4c_hyperboloidal/tensor_scri.hpp"
#include "z4c_hyperboloidal/stationary_gauge_reference.hpp"
namespace z4c_hyperboloidal {
inline constexpr bool spherical_policy=1;
// Stateless Cartesian point equations and scri limit, independent of ownership.
// Stencil construction and block-local endpoint lifting belong to native tasks.
struct SphericalPolicy {
  using J=mapped::Jet;
    KOKKOS_INLINE_FUNCTION static Real Radius(const Real*x){return sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);}
    KOKKOS_INLINE_FUNCTION static J Radius2(const Real*x){J r2;for(int a=0;a<3;++a){const J z=J::Variable(x[a],a);r2=r2+z*z;}return r2;}
    KOKKOS_INLINE_FUNCTION static ScalarJet<Real> Scalar(const J&q){ScalarJet<Real>r{};r.value=q.f;
      for(int a=0;a<3;++a){r.d[a]=q.d[a];for(int b=0;b<3;++b)r.dd[a][b]=q.h[a][b];}return r;}
    KOKKOS_INLINE_FUNCTION static VectorJet<Real> Vector(const CartesianChartJets&j,int f){VectorJet<Real>v{};
      for(int a=0;a<3;++a){v.value[a]=j[f+a].f;for(int b=0;b<3;++b){v.d[b][a]=j[f+a].d[b];for(int c=0;c<3;++c)v.dd[b][c][a]=j[f+a].h[b][c];}}return v;}
    KOKKOS_INLINE_FUNCTION static Z4cJet<Real> Physical(const Real*x,const CartesianChartJets&j){
      Z4cJet<Real>u{};u.chi=Scalar(j[0]);u.trace=Scalar(j[13]);u.theta=Scalar(j[14]);u.alpha=Scalar(j[18]);
      u.trace.value-=3;const Real r=Radius(x);u.alpha.value+=(1+r*r)/2;
      for(int a=0;a<3;++a){u.alpha.d[a]+=x[a];u.alpha.dd[a][a]+=1;}
      u.beta=Vector(j,22);u.lambda=Vector(j,15);
      for(int a=0;a<3;++a){u.beta.value[a]-=x[a];u.beta.d[a][a]-=1;}
      constexpr int ti[6]={0,0,0,1,1,2},tj[6]={0,1,2,1,2,2};
      for(int t=0;t<6;++t){const int a=ti[t],b=tj[t];u.metric.g[a][b]=u.metric.g[b][a]=j[1+t].f+(a==b);u.a.k[a][b]=u.a.k[b][a]=j[7+t].f;
        for(int c=0;c<3;++c){u.metric.dg[c][a][b]=u.metric.dg[c][b][a]=j[1+t].d[c];u.a.dk[c][a][b]=u.a.dk[c][b][a]=j[7+t].d[c];
          for(int d=0;d<3;++d)u.metric.ddg[c][d][a][b]=u.metric.ddg[c][d][b][a]=j[1+t].h[c][d];}}
      return u;
    }
    KOKKOS_INLINE_FUNCTION static Real CompatibleWeight(Real r,const HyperbolicShiftParameters<Real>&parameters){
      const Real a=parameters.inner_begin,b=parameters.inner_end,L=b-a;
      const Real z=fmax(0.,fmin(1.,fmin((r-(a-.5*L))/(.625*L),(b+.125*L-r)/(.25*L))));
      return z*z*z*(10-15*z+6*z*z);
    }
  KOKKOS_INLINE_FUNCTION static void StoreGeometry(const Z4cRHS<Real>&r,Real*out){
    out[0]=r.chi;out[13]=r.trace;out[14]=r.theta;
    constexpr int ti[6]={0,0,0,1,1,2},tj[6]={0,1,2,1,2,2};
    for(int t=0;t<6;++t){out[1+t]=r.metric[ti[t]][tj[t]];out[7+t]=r.a[ti[t]][tj[t]];}
    for(int a=0;a<3;++a)out[15+a]=r.lambda[a];
  }
  KOKKOS_INLINE_FUNCTION static bool Restore(const Real*x,CartesianChartJets&j,const CartesianChartJets&ref){
    const J r2=Radius2(x);j[0]=(j[0]+ref[0]/r2)*r2;
    for(int f=1;f<31;++f)j[f]=j[f]+ref[f];
    return TryExpandCartesianChart(j);
  }
  struct BoundaryResult {std::array<double,31> rate{};std::array<double,3> defect{};double clock=0;bool valid=true;};
  KOKKOS_INLINE_FUNCTION static BoundaryResult Boundary(const mapped::Point&map,Real spacing,const CartesianChartJets&j,const HyperbolicShiftParameters<Real>&parameters,const StationaryGaugeSource<Real>&boundary_source,Real penalty){
    const auto*x=map.x;const auto u=Physical(x,j);
    auto o=parameters.profile.At(x);o.omega=0;SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,o);
    auto parts=ConformalRHS(u,o,parameters.kappa1,parameters.kappa2);
    if(!HyperbolicDamping(x,u,o,parts,parameters)){BoundaryResult bad;bad.valid=false;return bad;};
    const auto pole=TensorPoleNormalJet(u,o,x,x,parameters.kappa1,parameters.kappa2,1.,parameters.outer_lapse_damping,true,true);
    auto rate=TensorScriLimit(parts.regular,pole,-1.);const auto trace=ScriRegularityTraces(u,o);const auto g=Geometry(u.metric);
    double Q=0,invn[3]{};for(int a=0;a<3;++a)for(int b=0;b<3;++b){Q+=g.inverse[a][b]*x[a]*x[b];invn[a]+=g.inverse[a][b]*x[b];}
    const double fac=penalty/spacing;
    rate.chi+=fac*(4*parts.pole.chi-8*(u.chi.value-1/Q));
    rate.trace+=fac*(4*parts.pole.trace-8*(u.trace.value+3));rate.theta+=fac*(4*parts.pole.theta-8*u.theta.value);
    for(int a=0;a<3;++a){rate.lambda[a]+=fac*(4*parts.pole.lambda[a]-8*trace.lambda[a]);
      for(int b=0;b<3;++b)rate.a[a][b]+=fac*(4*parts.pole.a[a][b]-8*trace.a[a][b]);}
    BoundaryResult out;StoreGeometry(rate,out.rate.data());
    const auto B=Vector(j,19),D=Vector(j,25);auto gauge=HyperbolicGauge(x,u,B,D.value,parameters);
    ApplyStationaryGaugeSource(x,u.alpha.value,boundary_source,gauge);
    out.clock=gauge.lapse_regular-pole.lapse.derivative;
    double Qdot=0;for(int a=0;a<3;++a)for(int b=0;b<3;++b)Qdot-=invn[a]*rate.metric[a][b]*invn[b];
    const double speed_t=(rate.chi*Q+u.chi.value*Qdot)/(2*std::sqrt(u.chi.value*Q));
    for(int a=0;a<3;++a){double restoring=0;for(int b=0;b<3;++b)restoring+=parameters.restoring*x[b]*j[22+a].d[b];
      out.rate[19+a]=gauge.flux_rate[a];out.rate[22+a]=-speed_t*x[a];
      out.defect[a]=gauge.shift_regular[a]+restoring-out.rate[22+a];}
    return out;
  }
};
}
#endif
