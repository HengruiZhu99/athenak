// Bounded finer-grid evaluation of the actual dropped-line shell KO operator.
// No volume grid allocation or solution evolution; all six face maps, both
// non-overlapping shells, corner/edge neighborhoods and face-interior controls.
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include "athena.hpp"
#include "mesh/cubed_sphere/composite_difference.hpp"
#include "z4c/hyperboloidal/composite_dissipation.hpp"
#include "fixed_composite_mesh.hpp"
namespace h=z4c::hyperboloidal;
struct Node{Real volume,inverse[3],taper;long double value;};
int main(int argc,char**argv){
 Kokkos::initialize(argc,argv);int status=0;
 try{
  std::cout<<std::setprecision(17)<<"n,samples,max_abs,layer,face,i,j,k,scaled_n4\n";
  for(int n:{16,24,32,48,64,96,128,192,256}){
   auto mesh=hyperbolic_reproduction::ContinuousCoreMesh(n);std::set<int>indices;
   for(int a=0;a<6;++a){indices.insert(a);indices.insert(n-1-a);}for(int a:{1,2,3})indices.insert(a*n/4);
   long double maximum=0;int location[5]{};size_t samples=0;
   using S=h::CompositeDissipation;
   for(size_t m=0;m<mesh.blocks.size();++m){const auto&block=mesh.blocks[m];if(block.layer<1)continue;
    std::map<std::array<int,3>,Node> cache;
    auto node=[&](std::array<int,3>i)->const Node&{
      auto found=cache.find(i);if(found!=cache.end())return found->second;
      const Real q[3]={(i[0]+.5)/n,(i[1]+.5)/n,(i[2]+.5)/n};const auto g=mesh.Geometry(m,q);
      Node v{};v.volume=cubed_sphere::PhysicalCellVolume(g,n);Real r=0;for(int a=0;a<3;++a)r+=g.x[a]*g.x[a];r=std::sqrt(r);
      v.taper=std::pow(std::max(Real(0),1-r*r/.64),6);
      for(int a=0;a<3;++a){for(int b=0;b<3;++b)v.inverse[a]+=g.J[a][b]*g.J[a][b];v.inverse[a]=std::sqrt(v.inverse[a]);}
      // Evaluate the analytic mapped smooth field in long double. This may
      // equal double on some platforms; the refinement evidence, rather than
      // a precision assumption, determines the resolved truncation range.
      // Metric weights remain actual production map values.
      const long double u=std::tan(acosl(-1.L)*(-1+2*(i[1]+.5L)/n)/4),vangle=std::tan(acosl(-1.L)*(-1+2*(i[2]+.5L)/n)/4);
      const long double radius=mesh.radii[block.layer-1]+(i[0]+.5L)/n*(mesh.radii[block.layer]-mesh.radii[block.layer-1]);
      const long double scale=radius/std::sqrt(1+u*u+vangle*vangle);long double x[3]{};const int a=block.face/2,b=(a+1)%3,c=(a+2)%3;
      x[a]=(block.face%2?-scale:scale);x[b]=scale*u;x[c]=scale*vangle;
      v.value=std::sin(2.3L*x[0]+.7L*x[1]-.4L*x[2]);
      return cache.emplace(i,v).first->second;
    };
    for(int k:indices)for(int j:indices)for(int i:indices){std::array<int,3>p{i,j,k};const auto target=node(p);long double rhs=0;
      for(int axis=0;axis<3;++axis)for(int a=0;a<S::points;++a){auto start=p;start[axis]-=a;
        if(start[axis]<0||start[axis]+S::order>=n)continue;
        long double difference=0,weight=0;const long double anchor=node(start).value;
        for(int b=0;b<S::points;++b){auto q=start;q[axis]+=b;const auto&v=node(q);
          difference+=S::Coefficient(b)*(v.value-anchor);weight+=v.volume*v.inverse[axis]*v.taper/S::points;}
        rhs-=.5L*n*S::Coefficient(a)*weight*difference/(S::normalization*target.volume);
      }
      ++samples;if(std::abs(rhs)>maximum){maximum=std::abs(rhs);location[0]=block.layer;location[1]=block.face;location[2]=i;location[3]=j;location[4]=k;}
    }
   }
   std::cout<<n<<','<<samples<<','<<maximum;for(int q:location)std::cout<<','<<q;std::cout<<','<<maximum*n*n*n*n<<'\n'<<std::flush;
  }
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';status=1;}
 Kokkos::finalize();return status;
}
