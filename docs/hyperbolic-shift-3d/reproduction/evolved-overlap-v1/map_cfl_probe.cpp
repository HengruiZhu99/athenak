#include <iostream>
#include <iomanip>
#include "mesh/cubed_sphere/composite.hpp"
#include "z4c/hyperboloidal/hyperbolic_shift.hpp"
using namespace cubed_sphere;
double frequency(const Point&p,int n){
 const auto*x=p.x;double r=std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
 const auto c=z4c::hyperboloidal::ShiftCoefficients(r,(1+r*r)/2,1.);
 double speed=0;
 for(int a=0;a<3;++a){double Q=0,beta=0;
  for(int d=0;d<3;++d){Q+=p.J[a][d]*p.J[a][d];beta-=p.J[a][d]*x[d];}
  speed+=std::abs(beta)+std::sqrt(std::max(4*c.mu*Q/3,(c.F+c.eta*.25*c.w*c.w)*Q));
 }return n*speed;
}
void test(int kind,double rin){
 const int n=16,o=kind?3:0;
 CompositeMesh mesh(.125,{.45,.75,1},1,n,3);
 if(kind>=2){mesh.radii.insert(mesh.radii.begin(),rin);for(auto&b:mesh.blocks)if(b.face>=0)++b.layer;}
 double maximum=0,rmax=0;int bm=-1,pm[3]{};
 for(int m=0;m<int(mesh.blocks.size());++m){const auto&b=mesh.blocks[m];
  if(kind>=2&&b.layer==0)continue; // Skip the unused cube-to-rin map.
  const bool inner=kind>=2?b.layer==1:b.layer==0;
  int low[3]{},high[3]={n,n,n};
  if(b.face<0||inner)for(int a=0;a<3;++a){low[a]=-o;high[a]=n+o;}
  // Physical Cartesian/spherical overlap replaces inward radial extension.
  if(kind>=2&&inner)low[0]=0;
  for(int k=low[2];k<high[2];++k)for(int j=low[1];j<high[1];++j)for(int i=low[0];i<high[0];++i){
   Real q[3]={(i+.5)/n,(j+.5)/n,(k+.5)/n};double factors[3]={1,1,1};
   if(kind==3&&b.face>=0)for(int a=1;a<3;++a){const double u=std::tan((2*q[a]-1)*std::acos(-1.)/4);q[a]=(u+1)/2;factors[a]=4/(std::acos(-1.)*(1+u*u));}
   auto p=mesh.Geometry(m,q);
   for(int a=1;a<3;++a)for(int d=0;d<3;++d)p.J[a][d]*=factors[a];
   const double f=frequency(p,n);
   if(f>maximum){maximum=f;bm=m;pm[0]=i;pm[1]=j;pm[2]=k;rmax=std::sqrt(p.x[0]*p.x[0]+p.x[1]*p.x[1]+p.x[2]*p.x[2]);}
  }
 }
 std::cout<<std::setprecision(17)<<"kind="<<kind<<" rin="<<rin<<" n="<<n<<" overlap="<<o<<" cfl_dt_00025="<<maximum*.00025<<" dt_at_cfl_04="<<.4/maximum<<" worst_block="<<bm<<" i="<<pm[0]<<" j="<<pm[1]<<" k="<<pm[2]<<" radius="<<rmax<<'\n';
}
int main(){test(0,0);test(1,0);for(double rin:{.075,.1,.125})test(2,rin);test(3,.125);}
