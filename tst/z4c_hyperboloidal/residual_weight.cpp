// Regular-center and transition regression without an evolution driver.
#include <cassert>
#include <cmath>
#include <iostream>
#include "athena.hpp"
#include "z4c_hyperboloidal/spherical_policy.hpp"
#include "z4c_hyperboloidal/spherical_reference_3d.hpp"
#include "z4c_hyperboloidal/central_z4c_configuration.hpp"
using namespace z4c_hyperboloidal;
int main(int argc,char **argv) {
  Kokkos::initialize(argc,argv);
  {
    ResidualWeight weight{.30,.55};
    auto parameters=CentralZ4cConfiguration{20,2,.97}.Resolve().gauge;
    SphericalReference3D reference(parameters);
    const double zero[3]{};auto origin=reference.At(zero);
    for(const auto &v:origin){assert(std::isfinite(v.f));for(int a=0;a<3;++a){
      assert(std::isfinite(v.d[a]));for(int b=0;b<3;++b)assert(std::isfinite(v.h[a][b]));}}
    for(double h:{.01,.001,.0001,.00001}) {
      const double x[3]={h,h/2,-h/3};auto ref=reference.At(x);
      CartesianChartJets q{};q[0].f=.125/(weight.inner*weight.inner);
      assert(SphericalPolicy::Restore(x,q,ref,weight));
      for(int a=0;a<3;++a){assert(q[0].d[a]==ref[0].d[a]);
        for(int b=0;b<3;++b)assert(q[0].h[a][b]==ref[0].h[a][b]);}
    }
    double previous=1e100;
    for(double h:{.008,.004,.002,.001}) {
      // Constant physical perturbation, sampled through the transition weight.
      const double x[3]={.421,0,0};auto qval=[&](double r){double y[3]={r,0,0};return .125/weight.At(y).f;};
      mapped::Jet q(qval(x[0]));
      q.d[0]=(qval(x[0]-2*h)-8*qval(x[0]-h)+8*qval(x[0]+h)-qval(x[0]+2*h))/(12*h);
      q.h[0][0]=(-qval(x[0]-2*h)+16*qval(x[0]-h)-30*q.f+16*qval(x[0]+h)-qval(x[0]+2*h))/(12*h*h);
      auto physical=weight.At(x)*q;
      double error=std::abs(physical.h[0][0]);
      assert(error<previous/10);previous=error;
      std::cout << "transition h=" << h << " Hessian error=" << error << '\n';
    }
    // Displaced manufactured centers: a regular scalar through the global
    // origin and both component neighborhoods, including the C6 transition.
    for(double r:{0.,.202337,-.202337,.421,.8}) {
      const double x[3]={r,.003,-.002};auto ref=reference.At(x);
      mapped::Jet plus,minus;
      for(int a=0;a<3;++a){auto z=mapped::Jet::Variable(x[a],a);
        auto u=z-mapped::Jet(a==0?.202337:0),v=z+mapped::Jet(a==0?.202337:0);
        plus=plus+u*u;minus=minus+v*v;}
      const auto physical=ref[0]+plus*minus;
      CartesianChartJets q{};q[0]=(physical-ref[0])/weight.At(x);
      assert(SphericalPolicy::Restore(x,q,ref,weight));
      assert(std::abs(q[0].f-physical.f)<1e-12);
      for(int a=0;a<3;++a){assert(std::abs(q[0].d[a]-physical.d[a])<1e-11);
        for(int b=0;b<3;++b)assert(std::abs(q[0].h[a][b]-physical.h[a][b])<1e-10);}
    }
    const double outer[3]={.8,.1,.1};
    auto a=weight.At(outer),b=SphericalPolicy::Radius2(outer);
    assert(a.f==b.f);for(int i=0;i<3;++i){assert(a.d[i]==b.d[i]);
      for(int j=0;j<3;++j)assert(a.h[i][j]==b.h[i][j]);}
    std::cout << "regular weight and origin jets PASS\n";
  }
  Kokkos::finalize();
}
