// Cached, read-only fixed-location interpolation on an immutable composite grid.
#ifndef HYPERBOLIC_THIN_PROFILE_SAMPLES_HPP_
#define HYPERBOLIC_THIN_PROFILE_SAMPLES_HPP_
#include <iomanip>
#include <ostream>
#include "mesh/cubed_sphere/composite_difference.hpp"
namespace hyperbolic_reproduction {
struct Term {int point;Real value,radial;};
inline std::vector<Term> Stencil(const cubed_sphere::CompositeDifference&g,const Real*x,bool core){
 constexpr int width=6;int block=-1,first[3],lo[3],hi[3];Real t[3],jac[3][3]{},r=0;
 for(int a=0;a<3;++a)r+=x[a]*x[a];r=std::sqrt(r);
 if(core){
  for(int a=0;a<3;++a){lo[a]=-g.mesh.CoreOverlap();hi[a]=g.cartesian_n+g.mesh.CoreOverlap();
   t[a]=(x[a]+g.mesh.half_width)/g.cartesian_h-.5;jac[a][a]=1/g.cartesian_h;}
 }else{
  const int face=cubed_sphere::Map{}.Owner(x);
  for(int m=0;m<int(g.mesh.blocks.size());++m)if(g.mesh.blocks[m].layer==0&&g.mesh.blocks[m].face==face){block=m;break;}
  if(block<0)throw std::runtime_error("missing first-shell profile block");
  Real q[3];g.mesh.Inverse(block,x,q);const auto geo=g.mesh.Geometry(block,q);
  for(int a=0;a<3;++a){lo[a]=g.Lower(block,a);hi[a]=g.Upper(block,a);t[a]=g.n*q[a]-.5;
   for(int d=0;d<3;++d)jac[a][d]=g.n*geo.J[a][d];}
 }
 Real w[3][width]{},dw[3][width]{};
 for(int a=0;a<3;++a){
  if(!std::isfinite(t[a])||t[a]<lo[a]-1e-11||t[a]>hi[a]-1+1e-11)throw std::runtime_error("profile would extrapolate beyond evolved donor hull");
  first[a]=std::clamp(int(std::floor(t[a]))-2,lo[a],hi[a]-width);
  for(int i=0;i<width;++i){
   w[a][i]=1;for(int j=0;j<width;++j)if(j!=i)w[a][i]*=(t[a]-first[a]-j)/Real(i-j);
   for(int k=0;k<width;++k)if(k!=i){Real value=1/Real(i-k);
    for(int j=0;j<width;++j)if(j!=i&&j!=k)value*=(t[a]-first[a]-j)/Real(i-j);dw[a][i]+=value;}
  }
 }
 std::vector<Term> out;Real constant=0,derivative=0;
 for(int k=0;k<width;++k)for(int j=0;j<width;++j)for(int i=0;i<width;++i){
  const int point=core?g.CartesianPoint(first[0]+i,first[1]+j,first[2]+k):g.ActiveIndex(block,first[0]+i,first[1]+j,first[2]+k);
  if(point<0)throw std::runtime_error("profile reads a ghost instead of an evolved value");
  const int c[3]={i,j,k};Real dn=0;
  for(int a=0;a<3;++a){Real v=dw[a][c[a]];for(int b=0;b<3;++b)if(a!=b)v*=w[b][c[b]];
   for(int d=0;d<3;++d)dn+=x[d]/r*jac[a][d]*v;}
  const Real value=w[0][i]*w[1][j]*w[2][k];out.push_back({point,value,dn});constant+=value;derivative+=dn;
 }
 if(std::abs(constant-1)>1e-11||std::abs(derivative)>1e-9)throw std::runtime_error("profile constant/derivative consistency failed");
 return out;
}
// Rebuild after any mesh/geometry change. This research grid is immutable;
// neither construction nor application mutates evolution or boundary caches.
class ProfileSamples {
 struct Row {int domain,ray;Real r,x[3];std::vector<Term> terms;};
 std::vector<Row> rows_;
 size_t points_;
 public:
 explicit ProfileSamples(const cubed_sphere::CompositeDifference&grid):points_(grid.active.size()) {
  const Real rays[16][3]={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},
   {1,1,1},{1,1,-1},{1,-1,1},{1,-1,-1},{-1,1,1},{-1,1,-1},{-1,-1,1},{-1,-1,-1},{12,4,-3},{-12,-4,3}};
  for(int ray=0;ray<16;++ray)for(int ir=5;ir<=80;++ir){
   const Real r=ir/200.,length=std::sqrt(rays[ray][0]*rays[ray][0]+rays[ray][1]*rays[ray][1]+rays[ray][2]*rays[ray][2]);
   Real x[3];bool inside=true;
   for(int a=0;a<3;++a){x[a]=r*rays[ray][a]/length;inside=inside&&std::abs(x[a])<=.18+1e-14;}
   for(int domain=0;domain<2;++domain){
    if(domain==0?!inside:r<.175)continue;
    rows_.push_back({domain,ray,r,{x[0],x[1],x[2]},Stencil(grid,x,domain==0)});
   }
  }
 }
 int Size()const{return rows_.size();}
 static void Header(std::ostream&output){
  output<<std::setprecision(17)<<"t,domain,ray,r,x,y,z";
  for(int f=0;f<31;++f)output<<",v"<<f;
  for(int f=0;f<31;++f)output<<",dr_v"<<f;
  output<<'\n';
 }
 void Write(std::ostream&output,Real time,const std::vector<Real>&state)const{
  if(state.size()!=31*points_||!std::isfinite(time))throw std::runtime_error("profile state/clock mismatch");
  for(const auto&row:rows_){
   Real value[31]{},derivative[31]{};
   for(const auto&term:row.terms)for(int f=0;f<31;++f){
    value[f]+=term.value*state[31*term.point+f];derivative[f]+=term.radial*state[31*term.point+f];
   }
   for(int f=0;f<31;++f)if(!std::isfinite(value[f])||!std::isfinite(derivative[f]))
    throw std::runtime_error("nonfinite profile sample");
   output<<std::setprecision(17)<<time<<','<<row.domain<<','<<row.ray<<','<<row.r<<','<<row.x[0]<<','<<row.x[1]<<','<<row.x[2];
   for(Real v:value)output<<','<<v;
   for(Real v:derivative)output<<','<<v;
   output<<'\n';
  }
  if(!output)throw std::runtime_error("profile write failed");
 }
};
}  // namespace hyperbolic_reproduction
#endif
