// Optional Kokkos tool: intrusive allocation census, separate from timing runs.
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <string>
#include <unistd.h>
extern "C" {struct SpaceHandle {char name[64];};}
namespace {
struct Use {uint64_t live=0,peak=0,allocations=0;};
std::mutex guard;
std::map<std::string,Use> labels,spaces;
void change(Use&v,uint64_t n,bool add){if(add){v.live+=n;v.peak=std::max(v.peak,v.live);++v.allocations;}else v.live-=n;}
void update(SpaceHandle s,const char*label,uint64_t n,bool add){
  std::lock_guard<std::mutex> lock(guard);change(labels[std::string(s.name)+":"+label],n,add);change(spaces[s.name],n,add);
}
void table(std::ostream&o,const std::map<std::string,Use>&rows){bool comma=false;o<<'{';for(auto&[name,v]:rows){if(comma)o<<',';comma=true;o<<std::quoted(name)<<":{\"live\":"<<v.live<<",\"peak\":"<<v.peak<<",\"allocations\":"<<v.allocations<<'}';}o<<'}';}
}
extern "C" void kokkosp_allocate_data(SpaceHandle s,const char*l,const void*,uint64_t n){update(s,l,n,true);}
extern "C" void kokkosp_deallocate_data(SpaceHandle s,const char*l,const void*,uint64_t n){update(s,l,n,false);}
extern "C" void kokkosp_finalize_library(){
  const char*prefix=std::getenv("HYPERBOLIC_ALLOCATION_PREFIX");if(!prefix)return;
  std::ofstream out(std::string(prefix)+".pid"+std::to_string(getpid())+".allocations.json");
  out<<"{\"pid\":"<<getpid()<<",\"spaces\":";table(out,spaces);out<<",\"labels\":";table(out,labels);out<<"}\n";
}
