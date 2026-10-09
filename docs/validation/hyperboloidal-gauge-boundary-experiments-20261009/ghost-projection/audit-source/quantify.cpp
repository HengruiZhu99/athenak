// Actual native arrays, strict-interior ghost plans and derivative identities.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "z4c/hyperboloidal/cartesian_patch.hpp"
namespace hyp=z4c::hyperboloidal;
struct Stats {int count=0;double sum=0,maximum=0,radius=0;void Add(double v,double r) {++count;sum+=v*v;if(std::abs(v)>maximum){maximum=std::abs(v);radius=r;}}};
int main(int argc,char**argv) {
 Kokkos::ScopeGuard guard(argc,argv);std::cout<<std::setprecision(17);
 const int degree=std::stoi(argv[1]);const std::string file=argv[2],prefix=argv[3];
 hyp::SphericalGhostGrid g;g.radius=1;for(int d=0;d<3;++d){g.n[d]=30;g.h[d]=2.1/24;g.first[d]=-.5*29*g.h[d];}
 hyp::LayerParameters lp;lp.enabled=true;lp.r0=.05;lp.r1=.95;
 hyp::LayerGaugeParameters lg;lg.physical_trace_lapse=true;lg.preferred_source=false;
 hyp::CartesianConformalPatch patch(g,.5,degree,lp,lg,true);
 const auto data=patch.Allocate("exact native payload");auto host=Kokkos::create_mirror_view(data);
 std::ifstream stream(file,std::ios::binary);stream.read(reinterpret_cast<char*>(host.data()),host.size()*sizeof(Real));
 if(!stream||stream.peek()!=EOF)throw std::runtime_error("unexpected exact native payload size");
 Kokkos::deep_copy(data,host);patch.Prepare(data);
 const auto full=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),data);
 const auto plans=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),patch.ghosts);
 Stats detghost,traceghost;int invalid=0;double detmin=1e99,ghost_change=0;
 std::ofstream points(prefix+"-ghosts.csv");points<<std::setprecision(17)<<"s,x,y,z,det,trace_A,valid,gxx,gxy,gxz,gyy,gyz,gzz,Axx,Axy,Axz,Ayy,Ayz,Azz\n";
 const int ti[6]={0,0,0,1,1,2},tj[6]={0,1,2,1,2,2};
 for(size_t point=0;point<plans.extent(0);++point){
  const int s=plans(point).target,i=s%30,j=s/30%30,k=s/900;
  const double x=g.first[0]+i*g.h[0],y=g.first[1]+j*g.h[1],z=g.first[2]+k*g.h[2],r=std::sqrt(x*x+y*y+z*z);
  hyp::MetricJet<Real> metric{};double A[3][3]{};
  for(int l=0;l<6;++l){metric.g[ti[l]][tj[l]]=metric.g[tj[l]][ti[l]]=full(0,z4c::Z4c::I_Z4C_GXX+l,k,j,i);A[ti[l]][tj[l]]=A[tj[l]][ti[l]]=full(0,z4c::Z4c::I_Z4C_AXX+l,k,j,i);}
  const auto geometry=hyp::Geometry(metric);detmin=std::min(detmin,geometry.determinant);detghost.Add(geometry.determinant-1,r);
  double trace=0;if(!geometry.valid)++invalid;else {for(int a=0;a<3;++a)for(int b=0;b<3;++b)trace+=geometry.inverse[a][b]*A[a][b];traceghost.Add(trace,r);}
  for(int f=0;f<z4c::Z4c::nz4c;++f)ghost_change=std::max(ghost_change,std::abs(full(0,f,k,j,i)-host(0,f,k,j,i)));
  points<<s<<","<<x<<","<<y<<","<<z<<","<<geometry.determinant<<","<<trace<<","<<geometry.valid;
  for(int l=0;l<6;++l)points<<","<<metric.g[ti[l]][tj[l]];
  for(int l=0;l<6;++l)points<<","<<A[ti[l]][tj[l]];
  points<<"\n";
 }
 const auto nodes=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),patch.active);
 const auto fields=hyp::BindCartesianFields(patch.deviations);const Real idx[3]={1/g.h[0],1/g.h[1],1/g.h[2]};
 Stats first[2],second[2],tf[2],pointdet[2],pointtrace[2];
 std::ofstream deriv(prefix+"-derivatives.csv");deriv<<std::setprecision(17)<<"s,r,max_det_first_identity,max_det_second_identity,max_trace_A_derivative_identity,point_det_residual,point_trace_A\n";
 for(size_t point=0;point<nodes.extent(0);++point){
  const int s=nodes(point),i=s%30,j=s/30%30,k=s/900;
  const double x=g.first[0]+i*g.h[0],y=g.first[1]+j*g.h[1],z=g.first[2]+k*g.h[2],r=std::sqrt(x*x+y*y+z*z);const int region=r>.9;
  auto u=hyp::LoadMeshJet<3>(fields,idx,0,k,j,i);hyp::AddReferenceJet(u,patch.reference.At(x,y,z),patch.reference);
  const auto geo=hyp::Geometry(u.metric);if(!geo.valid)throw std::runtime_error("invalid actual active geometry");
  double trace=0;for(int a=0;a<3;++a)for(int b=0;b<3;++b)trace+=geo.inverse[a][b]*u.a.k[a][b];
  double first_max=0,second_max=0,tf_max=0;
  for(int d=0;d<3;++d){
   double C=0,T=0;
   for(int a=0;a<3;++a)for(int b=0;b<3;++b){
    C+=geo.inverse[a][b]*u.metric.dg[d][a][b];T+=geo.inverse[a][b]*u.a.dk[d][a][b];
    for(int c=0;c<3;++c)for(int e=0;e<3;++e)T-=geo.inverse[a][c]*u.metric.dg[d][c][e]*geo.inverse[e][b]*u.a.k[a][b];
   }
   first_max=std::max(first_max,std::abs(C));tf_max=std::max(tf_max,std::abs(T));
   for(int e=0;e<3;++e){double C2=0;for(int a=0;a<3;++a)for(int b=0;b<3;++b){C2+=geo.inverse[a][b]*u.metric.ddg[d][e][a][b];for(int c=0;c<3;++c)for(int q=0;q<3;++q)C2-=geo.inverse[a][c]*u.metric.dg[e][c][q]*geo.inverse[q][b]*u.metric.dg[d][a][b];}second_max=std::max(second_max,std::abs(C2));}
  }
  first[region].Add(first_max,r);second[region].Add(second_max,r);tf[region].Add(tf_max,r);pointdet[region].Add(geo.determinant-1,r);pointtrace[region].Add(trace,r);
  deriv<<s<<","<<r<<","<<first_max<<","<<second_max<<","<<tf_max<<","<<geo.determinant-1<<","<<trace<<"\n";
 }
 auto print=[&](const char*name,const Stats&s){std::cout<<"\""<<name<<"\":{\"count\":"<<s.count<<",\"RMS\":"<<(s.count?std::sqrt(s.sum/s.count):0)<<",\"max_abs\":"<<s.maximum<<",\"max_radius\":"<<s.radius<<"}";};
 std::cout<<"{\"degree\":"<<degree<<",\"payload\":\""<<file<<"\",\"invalid_ghost_SPD_count\":"<<invalid<<",\"min_ghost_determinant\":"<<detmin<<",\"max_rebuilt_vs_saved_ghost_difference\":"<<ghost_change<<",";
 print("ghost_det_minus_one",detghost);std::cout<<",";print("ghost_trace_A",traceghost);
 for(int region=0;region<2;++region){std::cout<<",\""<<(region?"outer_r_gt_.9":"bulk_r_lt_.9")<<"\":{";print("det_first_identity",first[region]);std::cout<<",";print("det_second_identity",second[region]);std::cout<<",";print("trace_A_derivative_identity",tf[region]);std::cout<<",";print("active_det_minus_one",pointdet[region]);std::cout<<",";print("active_trace_A",pointtrace[region]);std::cout<<"}";}
 std::cout<<"}\n";
}
