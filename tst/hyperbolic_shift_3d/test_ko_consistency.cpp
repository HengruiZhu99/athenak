#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include "z4c/hyperboloidal/hyperbolic_composite.hpp"
namespace h=z4c::hyperboloidal;
int main(int argc,char**argv){
 Kokkos::initialize(argc,argv);int status=0;
 try{
  const int n=argc>1?std::stoi(argv[1]):12;
  cubed_sphere::CompositeMesh mesh(.18,{.45,.75,1},1,n,3,2);
  mesh.evolved_overlap=3;mesh.spherical_inner_boundary=true;mesh.equiangular=true;
  mesh.inner_radius=.16;mesh.continuous_cartesian_core=true;
  h::HyperbolicComposite cpu(mesh,false);cpu.ko=.5;
  const int nf=cpu.nf;const auto&g=cpu.grid;
  std::vector<Real>u(cpu.state.size(),0),q(u.size(),0);
  for(size_t p=0;p<g.active.size();++p){
   const auto&cell=g.cells[p];const auto&block=g.mesh.blocks[cell.block];
   const auto*x=g.geometry[p].x;
   // Core coordinate continuous across its internal block boundaries, shell
   // local radial coordinate polynomial; all actual seam-edge nodes included.
   const Real t=block.face<0?x[0]/.36:(cell.i+.5)/n;
   u[nf*p]=std::pow(t,4);u[nf*p+1]=std::pow(t,5);
   u[nf*p+2]=std::sin(2.3*x[0]+.7*x[1]-.4*x[2]);
   u[nf*p+3]=std::pow(x[0],4);
   u[nf*p+4]=std::sin(.137*p);
  }
  cpu.Dissipate(u,q);
  // Separate core, overlap/inner shells, outer shells, near-scri bands. Edges
  // include radial and angular faces; no ghost values enter this operator.
  Real maxima[4][5]{},edge[4][5]{},energy=0,outer=0;
  for(size_t p=0;p<g.active.size();++p){
   const auto&c=g.cells[p];const auto&b=g.mesh.blocks[c.block];
   const Real r=cpu.Radius(g.geometry[p].x);
   int region=b.face<0?0:b.layer==0?1:r<.8?2:3;
   bool at_edge=false;for(int a=0;a<3;++a)at_edge|=g.ShiftActive(p,a,-1)<0||g.ShiftActive(p,a,1)<0;
   for(int f=0;f<5;++f){maxima[region][f]=std::max(maxima[region][f],std::abs(q[nf*p+f]));if(at_edge)edge[region][f]=std::max(edge[region][f],std::abs(q[nf*p+f]));}
   energy+=g.volumes[p]*u[nf*p+4]*q[nf*p+4];
   // Lines supported wholly beyond the taper must give exactly zero.
   if(r>.95)outer=std::max(outer,std::abs(q[nf*p+4]));
  }
  if(!(energy<0)||outer!=0)throw std::runtime_error("energy or outer taper failed");
  if constexpr(h::CompositeDissipation::consistent){
   for(int r=0;r<4;++r)if(maxima[r][0]>2e-9)throw std::runtime_error("quartic logical polynomial not annihilated");
  }
  std::cout<<std::setprecision(17)<<"n,order,region,logical4,logical5,smooth,x4,edge_logical4,edge_logical5,edge_smooth,edge_x4,energy\n";
  for(int r=0;r<4;++r)std::cout<<n<<','<<h::CompositeDissipation::order<<','<<r<<','<<maxima[r][0]<<','<<maxima[r][1]<<','<<maxima[r][2]<<','<<maxima[r][3]<<','<<edge[r][0]<<','<<edge[r][1]<<','<<edge[r][2]<<','<<edge[r][3]<<','<<energy<<'\n';
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';status=1;}
 Kokkos::finalize();return status;
}
