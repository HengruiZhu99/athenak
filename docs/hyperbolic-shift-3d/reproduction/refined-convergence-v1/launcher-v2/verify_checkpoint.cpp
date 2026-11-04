// Read-only RF2 framing, finite-payload, metric and native FNV verification.
#include <fstream>
#include <iostream>
#include <string>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <iomanip>
#include <stdexcept>
struct Reader{
 std::ifstream in;uint64_t hash=14695981039346656037ull,bytes=0;
 explicit Reader(const char*p):in(p,std::ios::binary){if(!in)throw std::runtime_error("cannot open checkpoint");}
 void raw(char*p,size_t n){if(!in.read(p,n))throw std::runtime_error("truncated checkpoint");bytes+=n;for(size_t i=0;i<n;++i){hash^=static_cast<unsigned char>(p[i]);hash*=1099511628211ull;}}
 template<class T>T get(){T x;raw(reinterpret_cast<char*>(&x),sizeof(x));return x;}
 std::string zero(){std::string s;for(int i=0;i<4096;++i){char c=get<char>();if(!c)return s;s+=c;}throw std::runtime_error("oversized string");}
};
std::string quote(const std::string&s){std::string o="\"";for(char c:s){if(c=='\\'||c=='\"')o+='\\';if(static_cast<unsigned char>(c)<32)throw std::runtime_error("control character in identity");o+=c;}return o+'"';}
void require(bool q,const char*m){if(!q)throw std::runtime_error(m);}
int main(int argc,char**argv){try{
 require(argc==2,"usage: verify_checkpoint checkpoint");static_assert(sizeof(double)==8 && sizeof(int32_t)==4);
 Reader r(argv[1]);require(r.zero()=="HYP3D_AMR_RF2;checksum-fnv1a64;native31;configured-normalized-boost;","wrong RF2 schema");
 uint64_t length=r.get<uint64_t>();require(length>0&&length<16384,"bad identity size");std::string identity(length,' ');r.raw(identity.data(),length);require(identity.rfind("RF2;",0)==0,"wrong identity");
 const char*names[]={"chi","gxx-1","gxy","gxz","gyy-1","gyz","gzz-1","Axx","Axy","Axz","Ayy","Ayz","Azz","P+3","Theta","Lambda_x","Lambda_y","Lambda_z","alpha-(1+r2)/2","B_x","B_y","B_z","beta_x+x","beta_y+y","beta_z+z","D_x","D_y","D_z","J_x","J_y","J_z"};
 for(auto name:names)require(r.zero()==name,"wrong field layout");
 int32_t n=r.get<int32_t>(),kind=r.get<int32_t>(),sub=r.get<int32_t>();
 uint64_t steps=r.get<uint64_t>(),regrids=r.get<uint64_t>();
 auto finite=[&](){double x=r.get<double>();require(std::isfinite(x),"nonfinite header");return x;};
 double time=finite(),dt=finite(),seed=finite();std::array<double,3>tracker;for(auto&x:tracker)x=finite();
 for(int i=0;i<16;++i)finite();
 require(n>=12&&n%2==0&&kind>=0&&kind<=2&&(sub==2||sub==4||sub==8)&&time>=0&&dt>0&&seed>0&&seed<1,"bad configuration/time");
 uint64_t leaves=r.get<uint64_t>();require(leaves>=8&&leaves<=1024,"bad topology count");
 for(uint64_t i=0;i<leaves;++i){int32_t x=r.get<int32_t>(),y=r.get<int32_t>(),z=r.get<int32_t>(),l=r.get<int32_t>();require(l>=1&&l<20&&x>=0&&y>=0&&z>=0&&x<(1<<l)&&y<(1<<l)&&z<(1<<l),"bad logical leaf");}
 for(uint64_t i=0;i<leaves;++i)require(r.get<int32_t>()>=0,"negative leaf age");
 std::array<uint64_t,8>sizes{};double minchi=std::numeric_limits<double>::max();
 for(int a=0;a<8;++a){sizes[a]=r.get<uint64_t>();require(sizes[a]<=uint64_t(31)*1024*(n+8)*(n+8)*(n+8),"oversized array");
  std::array<double,31>cell{};
  for(uint64_t j=0;j<sizes[a];++j){double x=r.get<double>();require(std::isfinite(x),"nonfinite payload");if(a==0||a==2){cell[j%31]=x;if(j%31==30){double A=1+cell[1],B=cell[2],C=cell[3],D=1+cell[4],E=cell[5],F=1+cell[6];require(cell[0]>0&&A>0&&A*D>B*B&&A*(D*F-E*E)-B*(B*F-C*E)+C*(B*E-C*D)>0,"nonpositive chi or metric");minchi=std::min(minchi,cell[0]);}}}
 }
 require(sizes[0]>0&&sizes[0]%31==0&&sizes[1]==sizes[0]&&sizes[2]==uint64_t(31)*6*n*n&&sizes[3]==sizes[2]&&sizes[4]==sizes[0]/31&&sizes[5]==sizes[4]*16&&sizes[6]==sizes[2]/31*8&&sizes[7]%21==0,"array layout mismatch");
 require(r.get<uint64_t>()==0x414d5231434f4d50,"missing completion marker");uint64_t stored;require(bool(r.in.read(reinterpret_cast<char*>(&stored),8)),"missing checksum");require(stored==r.hash,"FNV checksum mismatch");char extra;require(!r.in.get(extra),"trailing bytes");
 std::cout<<std::setprecision(17)<<"{\"verified\":true,\"identity\":"<<quote(identity)<<",\"n\":"<<n<<",\"kind\":"<<kind<<",\"subdivisions\":"<<sub<<",\"steps\":"<<steps<<",\"regrids\":"<<regrids<<",\"time\":"<<time<<",\"dt\":"<<dt<<",\"leaves\":"<<leaves<<",\"bytes\":"<<r.bytes+8<<",\"min_chi\":"<<minchi<<",\"tracker\":["<<tracker[0]<<','<<tracker[1]<<','<<tracker[2]<<"],\"scope\":\"framing/finite/SPD/FNV; also require matching healthy native-worker-scheduler disposition and hashes\"}\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}return 0;}
