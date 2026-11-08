// Ported from scri-ko01 src/z4c/hyperboloidal/spherical_reference_3d.hpp; mathematical
// expressions retained. Native integration has no legacy driver dependency.
// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_NATIVE_SPHERICAL_REFERENCE_3D_HPP_
#define Z4C_HYPERBOLOIDAL_NATIVE_SPHERICAL_REFERENCE_3D_HPP_
#include "z4c_hyperboloidal/cartesian_algebraic_chart.hpp"
#include "z4c_hyperboloidal/stationary_gauge_reference.hpp"
namespace z4c_hyperboloidal {
// Setup only: spherical initial/reference data. Evolved residuals are arbitrary
// Cartesian scalar/vector/tensor fields and are never angularly averaged.
class SphericalReference3D {
 public:
  using J=mapped::Jet;
  HyperbolicShiftParameters<double> p;
  NativeWormhole native;
  StationaryGaugeReference source;
  std::map<double,std::array<std::array<double,12>,3>> cache;
  explicit SphericalReference3D(const HyperbolicShiftParameters<double>&parameters)
      :p(parameters),native(1./3.,2e-13,p.reference_start_M,p.reference_end_M,
        NativeWormhole::TransitionProfile::polynomial_c6,p.reference_compact_end),source(p) {}
  std::array<double,12> Radial(double r) {
    const double x[3]={r,0,0};auto f=native.Fields(x);p.profile.TransformFields(r,f);
    const auto ref=source.At(r);const double A=(1+r*r)/2;
    f[22]=(1-Smoothstep9((r-p.inner_begin)/(p.inner_end-p.inner_begin)))*r;
    if(r>p.profile.flat_radius){const double b=Smoothstep9((r-p.profile.flat_radius)/(p.inner_begin-p.profile.flat_radius));
      f[18]=(1-b)*(f[18]+A)+b*ref.alpha-A;f[22]=ref.shift_ratio*r;}
    return {f[0],1,1,f[7],f[10],f[13]-3,0,0,f[18]+A,f[22]-r,0,0};
  }
  const std::array<std::array<double,12>,3>& RadialJets(double r) {
    auto found=cache.find(r);if(found!=cache.end())return found->second;
    std::array<std::array<double,12>,3> out{};out[0]=Radial(r);
    const double h=r==1?.000125:std::min({.000125,r/8,(1-r)/8});
    if(!(h>0))throw std::invalid_argument("reference point outside punctured compact ball");
    const int begin=r==1?-6:-3;
    for(int i=0;i<7;++i){const double z=begin+i;J basis(1);
      for(int k=0;k<7;++k)if(k!=i)basis=basis*(J::Variable(0,0)-J(begin+k))/J(i-k);
      const auto q=Radial(r+h*z);
      for(int f=0;f<12;++f){out[1][f]+=basis.d[0]*(q[f]-out[0][f])/h;
        out[2][f]+=basis.h[0][0]*(q[f]-out[0][f])/(h*h);}
    }
    if(r==1){out[1].fill(0);out[2].fill(0);out[1][8]=out[2][8]=1;out[1][9]=-1;}
    return cache.emplace(r,out).first->second;
  }
  CartesianChartJets At(const double*x,bool boundary=false) {
    J xx[3],r2;for(int a=0;a<3;++a){xx[a]=J::Variable(x[a],a);r2=r2+xx[a]*xx[a];}
    J r=mapped::Power(r2,.5);if(boundary)r.f=1;
    const auto&radial=RadialJets(r.f);std::array<J,12> q{};
    for(int f=0;f<12;++f){q[f].f=radial[0][f];for(int a=0;a<3;++a){q[f].d[a]=radial[1][f]*r.d[a];
      for(int b=0;b<3;++b)q[f].h[a][b]=radial[2][f]*r.d[a]*r.d[b]+radial[1][f]*r.h[a][b];}}
    CartesianChartJets v{};v[0]=q[0];v[13]=q[5]+J(3);v[14]=q[6];v[18]=q[8]-(J(1)+r2)/J(2);
    const int ti[6]={0,0,0,1,1,2},tj[6]={0,1,2,1,2,2};
    for(int t=0;t<6;++t){const int a=ti[t],b=tj[t];const J nn=xx[a]*xx[b]/r2;
      v[1+t]=(q[1]-q[2])*nn+(a==b?q[2]-J(1):J(0));
      v[7+t]=(q[3]-q[4])*nn+(a==b?q[4]:J(0));}
    for(int a=0;a<3;++a){v[15+a]=q[7]*xx[a]/r;v[19+a]=q[10]*xx[a]/r;
      v[22+a]=q[9]*xx[a]/r+xx[a];v[25+a]=q[11]*xx[a]/r;}
    ExpandCartesianChart(v);return v;
  }
};
}
#endif
