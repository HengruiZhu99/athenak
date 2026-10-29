#include <iostream>
#include <iomanip>
#include <fstream>
#include "mesh/cubed_sphere/composite_difference.hpp"
#include "fixed_composite_mesh.hpp"
#include "z4c/hyperboloidal/hyperbolic_shift.hpp"
int main(int argc,char**argv){
 try{
  const int n=argc>1?std::stoi(argv[1]):12,o=argc>2?std::stoi(argv[2]):3;
#ifdef HYPERBOLIC_TEST_SPHERICAL_OVERLAP
  auto mesh=hyperbolic_reproduction::SphericalOverlapMesh(n);
#else
  cubed_sphere::CompositeMesh mesh(.125,{.45,.75,1},1,n,3);mesh.evolved_overlap=o;
#endif
  cubed_sphere::CompositeDifference g(mesh,4,true);g.SetInnerTransitionLogical(true);
  double inverse_error=0,jacobian_error=0,hessian_error=0,total_volume=0,max_core_frequency=0,max_shell_frequency=0;
  for(size_t p=0;p<g.active.size();++p){
    const auto&geo=g.geometry[p];const auto&cell=g.cells[p];const int m=cell.block;
    const Real q[3]={(cell.i+.5)/n,(cell.j+.5)/n,(cell.k+.5)/n};Real recovered[3];mesh.Inverse(m,geo.x,recovered);
    for(int a=0;a<3;++a)inverse_error=std::max(inverse_error,std::abs(q[a]-recovered[a]));
    total_volume+=g.DiagnosticVolume(p);
    const Real r=std::sqrt(geo.x[0]*geo.x[0]+geo.x[1]*geo.x[1]+geo.x[2]*geo.x[2]);
    const auto coeff=z4c::hyperboloidal::ShiftCoefficients(r,(1+r*r)/2,1.);Real frequency=0;
    for(int a=0;a<3;++a){Real Q=0,beta=0;for(int d=0;d<3;++d){Q+=geo.J[a][d]*geo.J[a][d];beta-=geo.J[a][d]*geo.x[d];}
      frequency+=std::abs(beta)+std::sqrt(std::max(4*coeff.mu*Q/3,(coeff.F+coeff.eta*.25*coeff.w*coeff.w)*Q));}
    auto&maximum=mesh.blocks[m].face<0?max_core_frequency:max_shell_frequency;maximum=std::max(maximum,n*frequency);
  }
  // Independently difference the inverse map and its analytic Jacobian in
  // physical Cartesian directions at interior and extended angular points.
  for(int m=0;m<int(mesh.blocks.size());++m)for(Real angle:{-.1,.3,.9,1.1}){
    const Real q[3]={.37,angle,.61};const auto geo=mesh.Geometry(m,q);const Real eps=1e-6;
    for(int d=0;d<3;++d){Real xp[3],xm[3],qp[3],qm[3];
      for(int a=0;a<3;++a){xp[a]=geo.x[a];xm[a]=geo.x[a];}xp[d]+=eps;xm[d]-=eps;
      mesh.Inverse(m,xp,qp);mesh.Inverse(m,xm,qm);const auto gp=mesh.Geometry(m,qp),gm=mesh.Geometry(m,qm);
      for(int A=0;A<3;++A){jacobian_error=std::max(jacobian_error,std::abs((qp[A]-qm[A])/(2*eps)-geo.J[A][d]));
        for(int a=0;a<3;++a)hessian_error=std::max(hessian_error,std::abs((gp.J[A][a]-gm.J[A][a])/(2*eps)-geo.H[A][a][d]));}}
  }
  if(inverse_error>2e-13||jacobian_error>2e-8||hessian_error>2e-6)throw std::runtime_error("map inverse/Jacobian/Hessian check failed");
  std::cout<<std::setprecision(17)<<"map_inverse_error="<<inverse_error<<" jacobian_error="<<jacobian_error<<" map_hessian_error="<<hessian_error
    <<" volume="<<total_volume<<" volume_error="<<std::abs(total_volume-4*std::acos(-1.)/3)
    <<" core_cfl_dt_00025="<<max_core_frequency*.00025<<" shell_cfl_dt_00025="<<max_shell_frequency*.00025<<'\n';
  if(argc>3){std::ofstream out(argv[3]);out<<std::setprecision(17)<<"x,y,z,r,volume\n";
    for(size_t p=0;p<g.active.size();++p){const auto*x=g.geometry[p].x;
      out<<x[0]<<','<<x[1]<<','<<x[2]<<','<<std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])<<','<<g.DiagnosticVolume(p)<<'\n';}
    if(!out)throw std::runtime_error("coordinate output failed");}
  double max_weight=0,max_constant=0,max_linear=0;size_t rows=0,extra=0;
  for(const auto&r:g.halos){
   const int block=r.target/(g.padded*g.padded*g.padded);
   int p=r.target%(g.padded*g.padded*g.padded),i=p%g.padded-g.storage_halo;
   int j=(p/g.padded)%g.padded-g.storage_halo,k=p/(g.padded*g.padded)-g.storage_halo;
   const Real q[3]={(i+.5)/n,(j+.5)/n,(k+.5)/n};Real x[3];g.mesh.Forward(block,q,x);
   if(std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])>=.45)continue;
   double sum=0,l1=0,v[3]{};
   for(const auto&e:r.entries){const int a=g.inverse_active.at(e.point);
    if(a<0)throw std::runtime_error("unevolved donor");
    extra+=size_t(a)>=g.nominal_count;sum+=e.weight;l1+=std::abs(e.weight);
    for(int d=0;d<3;++d)v[d]+=e.weight*g.geometry[a].x[d];
   }
   max_constant=std::max(max_constant,std::abs(sum-1));max_weight=std::max(max_weight,l1);
   for(int d=0;d<3;++d)max_linear=std::max(max_linear,std::abs(v[d]-x[d]));++rows;
  }
  if(o&&extra==0)throw std::runtime_error("no overlap donors used");
  if(max_constant>2e-12)throw std::runtime_error("constant reproduction failed");
  std::vector<Real> field(g.count,0);
  const Real wave[3]={1,2,-3};
  auto exact=[&](const Real*x){return std::sin(x[0]+2*x[1]-3*x[2]);};
  for(size_t p=0;p<g.active.size();++p)field[g.active[p]]=exact(g.geometry[p].x);
  g.Fill(field,1,[&](int,const std::array<Real,3>&x){return exact(x.data());});
  Real first2=0,second2=0,volume=0,firstmax=0,secondmax=0;
  for(size_t p=0;p<g.nominal_count;++p){if(g.mesh.blocks[g.cells[p].block].layer!=0)continue;
    Real d[3],dd[3][3];g.Derivatives(field,0,p,d,dd);
    const Real*x=g.geometry[p].x,phase=x[0]+2*x[1]-3*x[2];
    const Real v=g.DiagnosticVolume(p);volume+=v;
    for(int a=0;a<3;++a){const Real e=d[a]-wave[a]*std::cos(phase);
      first2+=v*e*e;firstmax=std::max(firstmax,std::abs(e));
      for(int b=0;b<3;++b){const Real e2=dd[a][b]+wave[a]*wave[b]*std::sin(phase);
        second2+=v*e2*e2;secondmax=std::max(secondmax,std::abs(e2));}}
  }
  if(!std::isfinite(first2+second2))throw std::runtime_error("nonfinite derivative");
  for(size_t p=g.nominal_count;p<g.active.size();++p)
    if(g.DiagnosticVolume(p)!=0)throw std::runtime_error("duplicate diagnostic volume");
  std::cout<<std::setprecision(17)<<"n="<<n<<" overlap="<<o<<" nominal="<<g.nominal_count<<" evolved="<<g.active.size()
   <<" inner_rows="<<rows<<" overlap_donor_terms="<<extra<<" max_weight_l1="<<max_weight
   <<" constant_error="<<max_constant<<" physical_linear_error="<<max_linear<<" gradient_l2="<<std::sqrt(first2/volume)<<" hessian_l2="<<std::sqrt(second2/volume)<<" gradient_max="<<firstmax<<" hessian_max="<<secondmax<<'\n';
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
