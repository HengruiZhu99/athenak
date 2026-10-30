// Read-only, common-location profiles; never advances or repairs a state.
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "mesh/cubed_sphere/composite_difference.hpp"
#include "fixed_composite_mesh.hpp"
#include "thin_profile_samples.hpp"
namespace {
template<class T>T Load(const char*b,int offset){T v;std::memcpy(&v,b+offset,sizeof(T));return v;}

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
  const hyperbolic_reproduction::ProfileSamples samples(grid);
  std::ofstream output(argv[2]);samples.Header(output);
  samples.Write(output,Load<Real>(raw,56),state);
  const int rows=samples.Size();
  output.close();if(!output)throw std::runtime_error("profile write failed");
  std::cout<<"profile_rows="<<rows<<" n="<<n<<" layers="<<layers<<" shifted="<<shifted<<" time="<<Load<Real>(raw,56)<<'\n';
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
