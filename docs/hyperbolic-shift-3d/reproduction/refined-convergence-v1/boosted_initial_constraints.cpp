// Independent Cartesian differentiation of the native exact-boost embedding.
// No mesh interpolation, KO, time integration, or equation changes.
#include <fstream>
#include <iomanip>
#include <iostream>
#include "refined_configuration.hpp"
#include "z4c/hyperboloidal/native_boosted_wormhole.hpp"
#include "z4c/hyperboloidal/hyperbolic_composite.hpp"
namespace h=z4c::hyperboloidal;
int main(int argc,char**argv){
 MPI_Init(&argc,&argv);Kokkos::initialize(argc,argv);int code=0;
 try{
  if(argc!=3)throw std::invalid_argument("config fresh.csv");
  hyperbolic_reproduction::RefinedConfiguration cfg;cfg.Load(argv[1],hyperbolic_reproduction::ContinuousCoreMesh(16));
  h::NativeBoostedWormhole native(cfg.boost_velocity,1./3.,true,cfg.initial_start_M,cfg.initial_end_M);
  std::ifstream exists(argv[2]);if(exists.good())throw std::runtime_error("existing output");
  std::ofstream out(argv[2]);out<<std::setprecision(17)<<"h,r,ray,x,y,z,H,M,Mr,Q,Q0,detJ\n";
  const double directions[][3]={{1,0,0},{-1,0,0},{0,1,0},{.6,.8,0},{-.6,.8,0},{.36,.48,.8}};
  for(Real eps:{.0008,.0004,.0002,.0001,.00005})
   for(Real r:{.3,.47,.62,.8,.875,.9,.915,.93,.945,.96,.98})for(int ray=0;ray<6;++ray){
    Real x[3];for(int a=0;a<3;++a)x[a]=r*directions[ray][a];
    auto fields=[&](const Real*y){auto f=native.Fields(y);cfg.InitialFields(y,f);return f;};
    const auto f=fields(x);std::vector<Real>v(f.begin(),f.end());
    std::vector<std::array<Real,3>>d(31);std::vector<std::array<std::array<Real,3>,3>>dd(31);
    const int offsets[4]={-2,-1,1,2};const Real weights[4]={1,-8,8,-1};
    for(int axis=0;axis<3;++axis){std::array<double,31>a[4];
     for(int k=0;k<4;++k){Real y[3]={x[0],x[1],x[2]};y[axis]+=eps*offsets[k];a[k]=fields(y);}
     for(int j=0;j<31;++j){d[j][axis]=(a[0][j]-8*a[1][j]+8*a[2][j]-a[3][j])/(12*eps);
      dd[j][axis][axis]=(-a[0][j]+16*a[1][j]-30*f[j]+16*a[2][j]-a[3][j])/(12*eps*eps);}
     for(int b=axis+1;b<3;++b){
      for(int k=0;k<4;++k)for(int l=0;l<4;++l){Real y[3]={x[0],x[1],x[2]};y[axis]+=eps*offsets[k];y[b]+=eps*offsets[l];auto z=fields(y);
       for(int j=0;j<31;++j)dd[j][axis][b]+=weights[k]*weights[l]*z[j]/(144*eps*eps);}
      for(int j=0;j<31;++j)dd[j][b][axis]=dd[j][axis][b];
     }
    }
    const auto u=h::HyperbolicComposite::Jet(x,v,d,dd);auto omega=cfg.gauge.profile.At(x);
    h::SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,omega);
    auto c=h::EvolvedConstraints(u,omega);if(!c.valid)throw std::runtime_error("invalid native constraint sample");
    h::NativeBoostedWormhole::InitialGeometryAudit audit;native.Fields(x,&audit);
    out<<eps<<','<<r<<','<<ray<<','<<x[0]<<','<<x[1]<<','<<x[2]<<','<<c.hamiltonian<<','<<std::sqrt(c.momentum_physical_norm2)<<','<<std::sqrt(c.momentum_conformal_norm2)<<','<<audit.spacelike_margin<<','<<audit.unboosted_margin<<','<<audit.boost_jacobian<<'\n';
   }
  out.close();if(!out)throw std::runtime_error("output failed");
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';code=2;}
 Kokkos::finalize();MPI_Finalize();return code;
}
