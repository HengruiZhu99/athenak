// Analytic mapped-function accuracy, including all faces and angular edge samples.
// Angular ghosts are analytic here: this does NOT test interpatch interpolation.
#include <cmath>
#include <iostream>
#include <iomanip>
#include "mesh/cubed_sphere/conformal_bridge.hpp"
#include "mesh/cubed_sphere/interior_radial.hpp"
using namespace cubed_sphere;
int main(int argc,char **argv) {
 Kokkos::initialize(argc,argv);std::cout<<std::setprecision(17)<<"stretch,n,gradient_max,hessian_max\n";
 const Real d1[5]={Real(1)/12,Real(-2)/3,0,Real(2)/3,Real(-1)/12};
 const Real d2[5]={Real(-1)/12,Real(4)/3,Real(-5)/2,Real(4)/3,Real(-1)/12};
 const Real wave[3]={.7,-.3,.2};
 for(Real stretch:{Real(0),Real(2.8592697712828787)})for(int n:{16,32,64,128}) {
  Map map{.25,1,stretch};Real graderr=0,hesserr=0;
  Real spacing[3]={Real(1)/(2*n),Real(2)/n,Real(2)/n};
  for(int face=0;face<6;++face)for(int depth=0;depth<3;++depth)for(int edge=0;edge<3;++edge) {
   Real q[3]={1-(depth+.5)*spacing[0],edge?1-1./n:.31,edge==2?-1+1./n:-.27};
   auto geo=map.Geometry(face,q);
   auto eval=[&](Real r,Real u,Real v){Real point[3]={r,u,v},x[3];map.Forward(face,point,x);return std::exp(wave[0]*x[0]+wave[1]*x[1]+wave[2]*x[2]);};
   auto radial=[&](int j){return eval(1-(5-j+.5)*spacing[0],q[1],q[2]);};
   Real d[3]{},dd[3][3]{};
   d[0]=InteriorRadialFirst(radial,depth,spacing[0]);dd[0][0]=InteriorRadialSecond(radial,depth,spacing[0]);
   for(int a=1;a<3;++a) {
    for(int j=0;j<5;++j){Real x[3]={q[0],q[1],q[2]};x[a]+=(j-2)*spacing[a];Real f=eval(x[0],x[1],x[2]);d[a]+=d1[j]*f/spacing[a];dd[a][a]+=d2[j]*f/(spacing[a]*spacing[a]);}
    auto cross=[&](int node){Real sum=0;for(int j=0;j<5;++j){Real x[3]={1-(5-node+.5)*spacing[0],q[1],q[2]};x[a]+=(j-2)*spacing[a];sum+=d1[j]*eval(x[0],x[1],x[2])/spacing[a];}return sum;};
    dd[0][a]=dd[a][0]=InteriorRadialFirst(cross,depth,spacing[0]);
   }
   for(int j=0;j<5;++j)for(int k=0;k<5;++k)dd[1][2]+=d1[j]*d1[k]*eval(q[0],q[1]+(j-2)*spacing[1],q[2]+(k-2)*spacing[2])/(spacing[1]*spacing[2]);
   dd[2][1]=dd[1][2];CartesianDerivatives(d,dd,geo);
   Real f=eval(q[0],q[1],q[2]);for(int a=0;a<3;++a){graderr=std::fmax(graderr,std::abs(d[a]-wave[a]*f));for(int b=0;b<3;++b)hesserr=std::fmax(hesserr,std::abs(dd[a][b]-wave[a]*wave[b]*f));}
  }
  std::cout<<stretch<<","<<n<<","<<graderr<<","<<hesserr<<"\n";
 }
 Kokkos::finalize();
}
