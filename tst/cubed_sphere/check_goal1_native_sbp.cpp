// Executes the exact research header to be compiled into the native candidate.
#include <iomanip>
#include <iostream>
#include <cmath>
#include "mesh/cubed_sphere/research_radial_sbp.hpp"
using namespace cubed_sphere;
struct Analytic {
  Map map;int patch,nodes;Real u,v,du,dv;bool constant=false;
  KOKKOS_INLINE_FUNCTION Real operator()(int i,int j,int k)const{
    if(constant)return 1;Real q[3]={(i+.5)/nodes,u+j*du,v+k*dv},x[3];map.Forward(patch,q,x);
    return exp(.2*x[0]+.3*x[1]-.1*x[2]);
  }
};
struct LineGeometry {Map map;int patch,nodes;Real u,v;
  KOKKOS_INLINE_FUNCTION Point operator()(int i)const{Real q[3]={(i+.5)/nodes,u,v};return map.Geometry(patch,q);}};
struct Basis {int column;KOKKOS_INLINE_FUNCTION Real operator()(int i,int,int)const{return i==column?1:0;}};
int main(int argc,char**argv){Kokkos::initialize(argc,argv);std::cout<<std::setprecision(17);
 for(int nodes:{24,48,96}){
  Real ds=1./nodes,du=4./nodes;auto op=MakeResearchRadialSBP(nodes,4,ds);
  auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),op.norm);
  auto d=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),op.d1),d2=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),op.d2);
  Real e1=0,e2=0,s1=1,s2=1;
  for(int i=0;i<nodes;++i)for(int j=0;j<nodes;++j){Real boundary=(i==j?(i==0?-1:(i==nodes-1?1:0)):0);Real a=ds*(h(i)*d(i,j)+h(j)*d(j,i))-boundary;e1=std::max(e1,fabs(a));
   Real product=0;for(int k=0;k<nodes;++k)product+=d(k,i)*h(k)*ds*d(k,j);
   Real expected=(i==0?-d(0,j):(i==nodes-1?d(nodes-1,j):0));Real b=h(i)*ds*d2(i,j)+product-expected;e2=std::max(e2,fabs(b));s2=std::max({s2,fabs(product),fabs(expected)});s1=std::max(s1,fabs(boundary));}
  std::cout<<"{\"type\":\"identity\",\"nodes\":"<<nodes<<",\"D1_scaled\":"<<e1/s1<<",\"D2_scaled\":"<<e2/s2<<",\"min_norm\":"<<h(0)<<"}\n";
  for(Real stretch:{0.,2.8592697712828787}){
   Real eg=0,eh=0,em=0,zero=0,negative=0;
   for(int patch=0;patch<6;++patch)for(int angular=0;angular<3;++angular){
    Real u=angular==0?.31:1.,v=angular==0?-.27:(angular==1?.4:1.);Map map{.25,.98,stretch};Analytic q{map,patch,nodes,u,v,du,du};auto constant=q;constant.constant=true;
    for(int i=0;i<nodes;++i){Real logical[3]={(i+.5)/nodes,u,v};auto point=map.Geometry(patch,logical);auto jet=ResearchSBPScalar(q,i,op,du,du);auto z=ResearchSBPScalar(constant,i,op,du,du);const Real f=q(i,0,0),wave[3]={.2,.3,-.1};
     Real noH[3][3]{};for(int a=0;a<3;++a)for(int b=0;b<3;++b)for(int A=0;A<3;++A)for(int B=0;B<3;++B)noH[a][b]+=point.J[A][a]*point.J[B][b]*jet.dd[A][B];
     CartesianDerivatives(jet.d,jet.dd,point);CartesianDerivatives(z.d,z.dd,point);
     for(int a=0;a<3;++a){eg=std::max(eg,fabs(jet.d[a]-wave[a]*f));zero=std::max(zero,fabs(z.d[a]));for(int b=0;b<3;++b){eh=std::max(eh,fabs(jet.dd[a][b]-wave[a]*wave[b]*f));zero=std::max(zero,fabs(z.dd[a][b]));if(a!=b)em=std::max(em,fabs(jet.dd[a][b]-wave[a]*wave[b]*f));negative=std::max(negative,fabs(noH[a][b]-wave[a]*wave[b]*f));}}
    }
   }
   std::cout<<"{\"type\":\"accuracy\",\"nodes\":"<<nodes<<",\"stretch\":"<<stretch<<",\"gradient\":"<<eg<<",\"hessian\":"<<eh<<",\"mixed\":"<<em<<",\"constant\":"<<zero<<",\"omitted_H\":"<<negative<<"}\n";
   if(nodes<=48){LineGeometry geo{{.25,.98,stretch},0,nodes,1.,.4};
    std::cout<<"{\"type\":\"KO\",\"nodes\":"<<nodes<<",\"stretch\":"<<stretch<<",\"norm_volume\":[";
    for(int i=0;i<nodes;++i)std::cout<<(i?",":"")<<h(i)*ResearchLogicalVolume(geo(i))*ds;
    std::cout<<"],\"matrix\":[";
    for(int i=0;i<nodes;++i){std::cout<<(i?",":"")<<"[";for(int j=0;j<nodes;++j)std::cout<<(j?",":"")<<ResearchRadialDissipation(Basis{j},geo,i,op,ds,.1);std::cout<<"]";}
    std::cout<<"],\"line_weight\":[";
    for(int l=0;l<nodes-3;++l){Real w=0;for(int j=0;j<4;++j){auto p=geo(l+j);Real n=0;for(int a=0;a<3;++a)n+=p.J[0][a]*p.J[0][a];w+=ResearchLogicalVolume(p)*sqrt(n)/4;}std::cout<<(l?",":"")<<w;}
    std::cout<<"]}\n";
   }
  }
 }
 Kokkos::finalize();
}
