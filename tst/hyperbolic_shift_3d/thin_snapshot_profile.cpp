// Read-only, common-location profiles; never advances or repairs a state.
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "mesh/cubed_sphere/composite_difference.hpp"
#include "fixed_composite_mesh.hpp"
namespace {
template<class T>T Load(const char*b,int offset){T v;std::memcpy(&v,b+offset,sizeof(T));return v;}
struct Term {int point;Real value,radial;};
std::vector<Term> Stencil(const cubed_sphere::CompositeDifference&g,const Real*x,bool core){
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
}
int main(int argc,char**argv){
 try{
  static_assert(sizeof(Real)==8,"V12 profiles require binary64");
  const uint16_t endian=1;if(*reinterpret_cast<const unsigned char*>(&endian)!=1)throw std::runtime_error("little-endian host required");
  if(argc!=3)throw std::runtime_error("usage: thin_snapshot_profile checkpoint output.csv");
  if(std::ifstream(argv[2]).good())throw std::runtime_error("refusing to overwrite profile");
  std::ifstream input(argv[1],std::ios::binary);char raw[184]{};input.read(raw,sizeof(raw));
  const char magic[16]="HYP3D_MPI_V12";
  if(!input||std::memcmp(raw,magic,16))throw std::runtime_error("not a complete V12 header");
  const int n=Load<int32_t>(raw,16),layers=Load<int32_t>(raw,156),shifted=Load<int32_t>(raw,160);
  if(n<12||n>256||n%2||layers<1||layers>3||shifted<0||shifted>1||Load<int32_t>(raw,20)!=31||Load<int32_t>(raw,28)!=1||Load<int32_t>(raw,24)<0||Load<int32_t>(raw,24)>3)
   throw std::runtime_error("invalid/unadmitted V12 dimensions or kind");
  const int offsets[]={80,84,88,92,96,100,104,108,112,116,144,148,152,164};
  const int expected[]={7,0,3,7,4,3,4,2,3,3,1,0,1,0};
  for(int i=0;i<14;++i)if(Load<int32_t>(raw,offsets[i])!=expected[i])throw std::runtime_error("V12 operator descriptor mismatch");
  if(Load<Real>(raw,120)!=.18||Load<Real>(raw,128)!=.16||Load<Real>(raw,136)!=1e-10)throw std::runtime_error("V12 geometry/positivity descriptor mismatch");
  for(int offset:{56,72,168,176})if(!std::isfinite(Load<Real>(raw,offset))||Load<Real>(raw,offset)<0)throw std::runtime_error("invalid V12 clock/coefficient");
  if(!std::isfinite(Load<Real>(raw,64))||Load<Real>(raw,64)<=0)throw std::runtime_error("invalid V12 timestep");
  const uint64_t cells=Load<uint64_t>(raw,32),surface=Load<uint64_t>(raw,40),N=n;
  const uint64_t expected_cells=12*N*N*N+8*(N+layers)*(N+layers)*(N+layers)+6*(N+3)*(N+6)*(N+6);
  input.seekg(0,std::ios::end);const auto bytes=input.tellg();input.seekg(sizeof(raw));
  if(cells!=expected_cells||surface!=31*6*N*N||bytes!=std::streamoff(184+16*(31*cells+surface)))
   throw std::runtime_error("V12 dimensions or exact file length mismatch");
  cubed_sphere::CompositeDifference grid(hyperbolic_reproduction::ThinCoreMesh(n,layers,shifted),4,true);
  if(cells!=grid.active.size()||surface!=31*6*uint64_t(n)*n)throw std::runtime_error("V12 point ordering/shape mismatch");
  std::vector<Real> state(31*cells);input.read(reinterpret_cast<char*>(state.data()),state.size()*sizeof(Real));
  if(!input)throw std::runtime_error("truncated V12 state");
  for(Real v:state)if(!std::isfinite(v))throw std::runtime_error("nonfinite V12 state");
  // Verify compensation and both surface arrays even though profiles use only volume values.
  uint64_t remaining=state.size()+2*surface;Real buffer[4096];
  while(remaining){const auto count=std::min<uint64_t>(remaining,4096);input.read(reinterpret_cast<char*>(buffer),count*sizeof(Real));
   if(!input)throw std::runtime_error("truncated V12 trailing arrays");
   for(uint64_t i=0;i<count;++i)if(!std::isfinite(buffer[i]))throw std::runtime_error("nonfinite V12 trailing array");remaining-=count;}
  char extra;if(input.get(extra))throw std::runtime_error("V12 checkpoint has trailing bytes");
  const Real rays[16][3]={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},
   {1,1,1},{1,1,-1},{1,-1,1},{1,-1,-1},{-1,1,1},{-1,1,-1},{-1,-1,1},{-1,-1,-1},{12,4,-3},{-12,-4,3}};
  std::ofstream output(argv[2]);output<<std::setprecision(17)<<"t,domain,ray,r,x,y,z";
  for(int f=0;f<31;++f)output<<",v"<<f;for(int f=0;f<31;++f)output<<",dr_v"<<f;output<<'\n';
  int rows=0;
  for(int ray=0;ray<16;++ray)for(int ir=5;ir<=80;++ir){
   const Real r=ir/200.,length=std::sqrt(rays[ray][0]*rays[ray][0]+rays[ray][1]*rays[ray][1]+rays[ray][2]*rays[ray][2]);
   Real x[3];bool inside=true;for(int a=0;a<3;++a){x[a]=r*rays[ray][a]/length;inside=inside&&std::abs(x[a])<=.18+1e-14;}
   for(int domain=0;domain<2;++domain){if(domain==0?!inside:r<.175)continue;
    const auto terms=Stencil(grid,x,domain==0);Real value[31]{},derivative[31]{};
    for(const auto&term:terms)for(int f=0;f<31;++f){value[f]+=term.value*state[31*term.point+f];derivative[f]+=term.radial*state[31*term.point+f];}
    output<<Load<Real>(raw,56)<<','<<domain<<','<<ray<<','<<r<<','<<x[0]<<','<<x[1]<<','<<x[2];
    for(Real v:value)output<<','<<v;for(Real v:derivative)output<<','<<v;output<<'\n';++rows;
   }
  }
  output.close();if(!output)throw std::runtime_error("profile write failed");
  std::cout<<"profile_rows="<<rows<<" n="<<n<<" layers="<<layers<<" shifted="<<shifted<<" time="<<Load<Real>(raw,56)<<'\n';
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
