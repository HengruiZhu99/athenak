// Independent anisotropic shell-coordinate derivative control of exact boosted data.
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
  std::ofstream out(argv[2]);out<<std::setprecision(17)<<"Nr,Na,r,ray,x,y,z,H,M,Mr,Q,Q0,detJ\n";
  const double directions[][3]={{1,0,0},{-1,0,0},{0,1,0},{.3,.4,std::sqrt(.75)}};
  const int resolutions[][2]={{12,12},{24,12},{48,12},{96,12},{12,24},{12,48},{12,96},{24,24},{48,48},{96,96}};
  auto mesh=cfg.Mesh(12);
  for(const auto&res:resolutions)
   for(Real r:{.715,.8275,.8875,.903125,.909375,.915625,.921875,.928125,.934375,.940625,.946875,.975})for(int ray=0;ray<4;++ray){
    Real x[3];for(int a=0;a<3;++a)x[a]=r*directions[ray][a];
    const int block=mesh.Owner(x);Real q[3];mesh.Inverse(block,x,q);const auto geometry=mesh.Geometry(block,q);
    const Real spacing[3]={1./res[0],1./res[1],1./res[1]};
    auto fields=[&](const Real*local){Real y[3];mesh.Forward(block,local,y);auto f=native.Fields(y);cfg.InitialFields(y,f);return f;};
    const auto f=fields(q);std::vector<Real>v(f.begin(),f.end());
    std::vector<std::array<Real,3>>d(31);std::vector<std::array<std::array<Real,3>,3>>dd(31);
    const int offsets[4]={-2,-1,1,2};const Real weights[4]={1,-8,8,-1};
    for(int axis=0;axis<3;++axis){std::array<double,31>a[4];
     for(int k=0;k<4;++k){Real y[3]={q[0],q[1],q[2]};y[axis]+=spacing[axis]*offsets[k];a[k]=fields(y);}
     for(int j=0;j<31;++j){d[j][axis]=(a[0][j]-8*a[1][j]+8*a[2][j]-a[3][j])/(12*spacing[axis]);
      dd[j][axis][axis]=(-a[0][j]+16*a[1][j]-30*f[j]+16*a[2][j]-a[3][j])/(12*spacing[axis]*spacing[axis]);}
     for(int b=axis+1;b<3;++b){
      for(int k=0;k<4;++k)for(int l=0;l<4;++l){Real y[3]={q[0],q[1],q[2]};y[axis]+=spacing[axis]*offsets[k];y[b]+=spacing[b]*offsets[l];auto z=fields(y);
       for(int j=0;j<31;++j)dd[j][axis][b]+=weights[k]*weights[l]*z[j]/(144*spacing[axis]*spacing[b]);}
      for(int j=0;j<31;++j)dd[j][b][axis]=dd[j][axis][b];
     }
    }
    const auto logical_d=d;const auto logical_dd=dd;
    for(int f=0;f<31;++f)for(int a=0;a<3;++a){d[f][a]=0;
      for(int A=0;A<3;++A)d[f][a]+=geometry.J[A][a]*logical_d[f][A];
      for(int b=0;b<3;++b){dd[f][a][b]=0;
        for(int A=0;A<3;++A){dd[f][a][b]+=geometry.H[A][a][b]*logical_d[f][A];
          for(int B=0;B<3;++B)dd[f][a][b]+=geometry.J[A][a]*geometry.J[B][b]*logical_dd[f][A][B];}
      }
    }
    const auto u=h::HyperbolicComposite::Jet(x,v,d,dd);auto omega=cfg.gauge.profile.At(x);
    h::SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,omega);
    auto c=h::EvolvedConstraints(u,omega);if(!c.valid)throw std::runtime_error("invalid native constraint sample");
    h::NativeBoostedWormhole::InitialGeometryAudit audit;native.Fields(x,&audit);
    out<<res[0]<<','<<res[1]<<','<<r<<','<<ray<<','<<x[0]<<','<<x[1]<<','<<x[2]<<','<<c.hamiltonian<<','<<std::sqrt(c.momentum_physical_norm2)<<','<<std::sqrt(c.momentum_conformal_norm2)<<','<<audit.spacelike_margin<<','<<audit.unboosted_margin<<','<<audit.boost_jacobian<<'\n';
   }
  out.close();if(!out)throw std::runtime_error("output failed");
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';code=2;}
 Kokkos::finalize();MPI_Finalize();return code;
}
