// Read-only binary64 export of the native reference and mapped geometry.
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include "mesh/cubed_sphere/map.hpp"
#include "mesh/cubed_sphere/reference_wave_map.hpp"
namespace h=z4c::hyperboloidal;
int main(int argc,char **argv) {
  Kokkos::initialize(argc,argv);
  std::cout<<std::setprecision(17);
  std::string kind;
  while(std::cin>>kind) {
    if(kind=="layer") {
      double a,r0,r1,r;std::cin>>a>>r0>>r1>>r;
      h::LayerReference<double> ref(1.,a,{true,r0,r1});
      auto p=ref.At(r,0.,0.);auto w=h::SmoothCutoff(r,r0,r1);
      h::OmegaJet<double> o{};o.omega=p.omega;
      for(int i=0;i<3;++i){o.gradient[i]=p.domega[i];
        for(int j=0;j<3;++j)o.hessian[i][j]=p.omega_hessian[i][j];}
      h::SetStationaryOmegaNormal(p.alpha,p.beta,p.dalpha,p.state.beta.d,o);
      h::Z4cRHS<double> rhs{};h::GaugeRHS<double> gauge{};
      auto parts=h::ConformalRHS(p.state,o,10./p.alpha,0.);
      bool valid=h::AssembleInterior(parts,p.omega,rhs);
      double x[3]={r,0.,0.};
      bool gv=rwm::Assemble(rwm::Gauge(p,p.state,rwm::ReferenceConnection(p,x)),p.omega,gauge);
      double raw=0; if(valid) {
        raw=std::max({std::abs(rhs.chi),std::abs(rhs.trace),std::abs(rhs.theta)});
        for(int i=0;i<3;++i){raw=std::max(raw,std::abs(rhs.lambda[i]));
          for(int j=0;j<3;++j)raw=std::max({raw,std::abs(rhs.metric[i][j]),std::abs(rhs.a[i][j])});}
      }
      const auto &u=p.state;
      std::cout<<"layer "<<a<<" "<<r0<<" "<<r1<<" "<<r;
      for(double v:{w.value,w.d,w.dd,w.ddd,p.omega,p.domega[0],p.omega_hessian[0][0],
          p.alpha,p.dalpha[0],u.alpha.dd[0][0],p.L,p.b,p.k_physical,p.k_bar,
          u.chi.value,u.chi.d[0],u.chi.dd[0][0],u.metric.g[0][0],u.metric.dg[0][0][0],
          u.metric.ddg[0][0][0][0],u.metric.g[1][1],u.metric.dg[0][1][1],
          u.metric.ddg[0][0][1][1],u.beta.value[0],u.beta.d[0][0],u.beta.dd[0][0][0],
          u.a.k[0][0],u.a.k[1][1],p.outgoing,p.ingoing,raw})std::cout<<" "<<v;
      std::cout<<" "<<valid<<" "<<gv<<"\n";
    } else if(kind=="map") {
      double L,R,k,q[3];int patch;std::cin>>L>>R>>k>>patch>>q[0]>>q[1]>>q[2];
      cubed_sphere::Map map{L,R,k};auto p=map.Geometry(patch,q);
      std::cout<<"map "<<L<<" "<<R<<" "<<k<<" "<<patch;
      for(double v:q)std::cout<<" "<<v;
      for(double v:p.x)std::cout<<" "<<v;
      for(auto &row:p.J)for(double v:row)std::cout<<" "<<v;
      for(auto &mat:p.H)for(auto &row:mat)for(double v:row)std::cout<<" "<<v;
      std::cout<<"\n";
    } else return 2;
  }
  Kokkos::finalize();
}
