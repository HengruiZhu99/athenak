// Scratch scalar isolate: exact production ghost planner and native derivative coefficients.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>
#include "athena.hpp"
#include "utils/finite_diff.hpp"
#include "z4c/hyperboloidal/spherical_ghosts.hpp"
#include "z4c/hyperboloidal/interior_dissipation.hpp"
namespace hyp=z4c::hyperboloidal;
struct Probe {int column,nx,ny;KOKKOS_INLINE_FUNCTION Real operator()(int,int k,int j,int i)const{return i+nx*(j+ny*k)==column?1:0;}KOKKOS_INLINE_FUNCTION Real operator()(int s)const{return s==column?1:0;}};
struct Velocity {double first,h;KOKKOS_INLINE_FUNCTION Real operator()(int,int a,int k,int j,int i)const{const int at[3]={i,j,k};return -(first+h*at[a]);}};
struct Mask {const std::vector<int>*map;bool operator()(int s)const{return s>=0&&static_cast<size_t>(s)<map->size()&&(*map)[s]>=0;}};
struct Scalar {Kokkos::View<double*> q;int nx,ny;KOKKOS_INLINE_FUNCTION Real operator()(int,int k,int j,int i)const{return q(i+nx*(j+ny*k));}KOKKOS_INLINE_FUNCTION Real operator()(int s)const{return q(s);}};
int main(int argc,char**argv){
 Kokkos::ScopeGuard guard(argc,argv);
 try{
  const int n=std::stoi(argv[1]);const double span=std::stod(argv[2]);const std::string closure=argv[3],derivative=argv[4],prefix=argv[6];const double ko=std::stod(argv[5]);
  if((closure!="ray"&&closure!="nearest"&&closure!="fallback"&&closure!="monotone")||(derivative!="upwind"&&derivative!="centered")||ko<0)throw std::runtime_error("invalid policy");
  hyp::SphericalGhostGrid grid;grid.radius=1;for(int d=0;d<3;++d){grid.n[d]=n+6;grid.h[d]=span/n;grid.first[d]=-.5*(n+5)*grid.h[d];}
  const int nx=grid.n[0],ny=grid.n[1],total=nx*ny*grid.n[2];const double h=grid.h[0],first=grid.first[0];const int stride[3]={1,nx,nx*ny};const Real idx[3]={1/h,1/h,1/h},spacing[3]={h,h,h};const Velocity beta{first,h};
  auto plan=hyp::PlanSymmetricSphericalGhosts(grid,3,2);std::vector<int> active,lookup(total,-1),ghostlookup(total,-1);
  for(int k=3;k<grid.n[2]-3;++k)for(int j=3;j<ny-3;++j)for(int i=3;i<nx-3;++i)if(grid.Interior(i,j,k)){lookup[grid.Index(i,j,k)]=active.size();active.push_back(grid.Index(i,j,k));}
  double plan_sum_error=0,max_l1=0;size_t references=0;for(size_t p=0;p<plan.size();++p){ghostlookup[plan[p].target]=p;double sum=0,l1=0;for(int b=0;b<plan[p].count;++b){const int donor=plan[p].donors[b];if(lookup[donor]<0)throw std::runtime_error("donor not active or recursive");sum+=plan[p].weights[b];l1+=std::abs(plan[p].weights[b]);++references;}plan_sum_error=std::max(plan_sum_error,std::abs(sum-1));max_l1=std::max(max_l1,l1);}
  if(closure=="nearest")for(auto &p:plan){double best=1e99;std::vector<int> candidates;const int at[3]={p.target%nx,p.target/nx%ny,p.target/(nx*ny)};for(int b=0;b<p.count;++b){int s=p.donors[b];const int bt[3]={s%nx,s/nx%ny,s/(nx*ny)};double dist=0;for(int d=0;d<3;++d)dist+=(at[d]-bt[d])*(at[d]-bt[d]);if(dist<best){best=dist;candidates.clear();candidates.push_back(s);}else if(dist==best)candidates.push_back(s);}std::sort(candidates.begin(),candidates.end());candidates.erase(std::unique(candidates.begin(),candidates.end()),candidates.end());p.count=candidates.size();for(int b=0;b<p.count;++b){p.donors[b]=candidates[b];p.weights[b]=1./p.count;}}
  std::ofstream points(prefix+"-points.csv");points<<std::setprecision(17)<<"column,index,x,y,z,r,boundary_distance\n";for(size_t a=0;a<active.size();++a){int s=active[a],i=s%nx,j=s/nx%ny,k=s/(nx*ny);double x=first+i*h,y=first+j*h,z=first+k*h,r=std::sqrt(x*x+y*y+z*z);points<<a<<','<<s<<','<<x<<','<<y<<','<<z<<','<<r<<','<<1-r<<'\n';}
  std::vector<std::map<int,double>> matrix(active.size());int fallback_axis_count=0;std::vector<std::array<bool,3>> fallback(active.size());
  for(size_t row=0;row<active.size();++row){const int s=active[row],i=s%nx,j=s/nx%ny,k=s/(nx*ny);const int index[3]={i,j,k};bool one_sided[3]{};
   for(int d=0;d<3;++d){const double v=first+h*index[d];bool touches=false;for(int shift=-3;shift<=3;++shift){bool used=derivative=="centered"?(std::abs(shift)<=2&&shift!=0):(v>0?(shift>=-3&&shift<=1):(shift>=-1&&shift<=3));if(used&&lookup[s+shift*stride[d]]<0)touches=true;}one_sided[d]=closure=="monotone"||(closure=="fallback"&&touches);if(one_sided[d])++fallback_axis_count;}
   for(int d=0;d<3;++d)fallback[row][d]=one_sided[d];
   std::set<int> raw{s};for(int d=0;d<3;++d)for(int offset=-3;offset<=3;++offset)raw.insert(s+offset*stride[d]);
   for(const int column:raw){Probe probe{column,nx,ny};double coefficient=0;
    for(int d=0;d<3;++d){const double v=first+h*index[d];if(one_sided[d]){const int inner=s+(v>0?-1:1)*stride[d];if(lookup[inner]<0)throw std::runtime_error("fallback inward donor not active");coefficient-=std::abs(v)*(probe(s)-probe(inner))/h;}else if(derivative=="centered")coefficient+=beta(0,d,k,j,i)*Dx<3>(d,idx,probe,0,k,j,i);else coefficient+=Lx<3>(d,idx,beta,probe,0,d,k,j,i);}
    coefficient+=ko*hyp::InteriorKOSixth(probe,Mask{&lookup},s,stride,spacing);if(coefficient==0)continue;
    if(lookup[column]>=0)matrix[row][lookup[column]]+=coefficient;else{const int gi=ghostlookup[column];if(gi<0)throw std::runtime_error("missing actual required ghost");const auto &p=plan[gi];for(int b=0;b<p.count;++b)matrix[row][lookup[p.donors[b]]]+=coefficient*p.weights[b];}
   }
  }
  // Independent actual FillSphericalGhosts execution and native stencil matvec.
  Kokkos::View<double*> full("actual scalar fill",total);Kokkos::View<hyp::SphericalGhostStencil*> uploaded("actual plans",plan.size());
  auto ph=Kokkos::create_mirror_view(uploaded);for(size_t p=0;p<plan.size();++p)ph(p)=plan[p];Kokkos::deep_copy(uploaded,ph);
  auto qh=Kokkos::create_mirror_view(full);for(int s=0;s<total;++s)qh(s)=lookup[s]>=0?std::sin(.1234*s)+std::cos(.877*s):std::numeric_limits<double>::quiet_NaN();Kokkos::deep_copy(full,qh);hyp::FillSphericalGhosts(full,uploaded);Kokkos::deep_copy(qh,full);
  Scalar scalar{full,nx,ny};double matvec_error=0;
  for(size_t row=0;row<active.size();++row){const int s=active[row],i=s%nx,j=s/nx%ny,k=s/(nx*ny),at[3]={i,j,k};double native=0,assembled=0;for(int d=0;d<3;++d){double v=first+h*at[d];if(fallback[row][d])native-=std::abs(v)*(scalar(s)-scalar(s+(v>0?-1:1)*stride[d]))/h;else if(derivative=="centered")native+=beta(0,d,k,j,i)*Dx<3>(d,idx,scalar,0,k,j,i);else native+=Lx<3>(d,idx,beta,scalar,0,d,k,j,i);}native+=ko*hyp::InteriorKOSixth(scalar,Mask{&lookup},s,stride,spacing);for(const auto &[column,value]:matrix[row])assembled+=value*qh(active[column]);matvec_error=std::max(matvec_error,std::abs(native-assembled));}
  if(matvec_error>1e-10)throw std::runtime_error("assembled operator disagrees with actual native fill/stencil execution");
  size_t entries=0;double constant=0;bool metzler=true;for(size_t row=0;row<matrix.size();++row){double sum=0;for(const auto &[column,value]:matrix[row]){if(value==0)continue;++entries;sum+=value;if(column!=static_cast<int>(row)&&value< -1e-13)metzler=false;}constant=std::max(constant,std::abs(sum));}
  std::ofstream out(prefix+".mtx");out<<"%%MatrixMarket matrix coordinate real general\n"<<matrix.size()<<' '<<matrix.size()<<' '<<entries<<'\n'<<std::setprecision(17);for(size_t row=0;row<matrix.size();++row)for(const auto &[column,value]:matrix[row])if(value!=0)out<<row+1<<' '<<column+1<<' '<<value<<'\n';
  std::cout<<std::setprecision(17)<<"{\"n\":"<<n<<",\"span\":"<<span<<",\"spacing\":"<<h<<",\"active\":"<<active.size()<<",\"ghosts\":"<<plan.size()<<",\"closure\":\""<<closure<<"\",\"derivative\":\""<<derivative<<"\",\"ko\":"<<ko<<",\"nonzeros\":"<<entries<<",\"constant_residual_linf\":"<<constant<<",\"native_fill_stencil_matvec_max_error\":"<<matvec_error<<",\"original_plan_constant_error\":"<<plan_sum_error<<",\"original_plan_max_weight_l1\":"<<max_l1<<",\"strict_inside_nonrecursive_donor_references\":"<<references<<",\"fallback_axis_count\":"<<fallback_axis_count<<",\"matrix_is_metzler\":"<<metzler<<"}\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
