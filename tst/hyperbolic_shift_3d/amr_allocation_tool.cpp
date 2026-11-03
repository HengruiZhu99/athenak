// Optional Kokkos tool: intrusive allocation census, separate from timing runs.
#include <cstdint>
#include <chrono>
#include "impl/Kokkos_Profiling_C_Interface.h"
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <string>
#include <unistd.h>
using SpaceHandle=Kokkos_Profiling_SpaceHandle;
namespace {
struct Use {uint64_t live=0,peak=0,allocations=0;};
std::mutex guard;
std::map<std::string,Use> labels,spaces;
using Clock=std::chrono::steady_clock;
struct KernelUse {double seconds=0;uint64_t calls=0;};
std::map<std::string,KernelUse> kernels;
std::map<uint64_t,std::pair<std::string,Clock::time_point>> running;
uint64_t next_id=0;
bool timings(){static const bool enabled=std::getenv("HYPERBOLIC_PROFILE_KERNELS")!=nullptr;return enabled;}
void begin(const char*label,uint64_t*id){if(!timings())return;std::lock_guard<std::mutex> lock(guard);*id=++next_id;running[*id]={label,Clock::now()};}
void end(uint64_t id){if(!timings())return;const auto now=Clock::now();std::lock_guard<std::mutex> lock(guard);auto it=running.find(id);if(it==running.end())return;auto&k=kernels[it->second.first];k.seconds+=std::chrono::duration<double>(now-it->second.second).count();++k.calls;running.erase(it);}

void change(Use&v,uint64_t n,bool add){if(add){v.live+=n;v.peak=std::max(v.peak,v.live);++v.allocations;}else v.live-=n;}
void update(SpaceHandle s,const char*label,uint64_t n,bool add){
  std::lock_guard<std::mutex> lock(guard);change(labels[std::string(s.name)+":"+label],n,add);change(spaces[s.name],n,add);
}
void table(std::ostream&o,const std::map<std::string,Use>&rows){bool comma=false;o<<'{';for(auto&[name,v]:rows){if(comma)o<<',';comma=true;o<<std::quoted(name)<<":{\"live\":"<<v.live<<",\"peak\":"<<v.peak<<",\"allocations\":"<<v.allocations<<'}';}o<<'}';}
}
extern "C" void kokkosp_request_tool_settings(uint32_t,Kokkos_Tools_ToolSettings*s){s->requires_global_fencing=timings();}
extern "C" void kokkosp_begin_parallel_for(const char*l,uint32_t,uint64_t*i){begin(l,i);}
extern "C" void kokkosp_end_parallel_for(uint64_t i){end(i);}
extern "C" void kokkosp_begin_parallel_reduce(const char*l,uint32_t,uint64_t*i){begin(l,i);}
extern "C" void kokkosp_end_parallel_reduce(uint64_t i){end(i);}
extern "C" void kokkosp_allocate_data(SpaceHandle s,const char*l,const void*,uint64_t n){update(s,l,n,true);}
extern "C" void kokkosp_deallocate_data(SpaceHandle s,const char*l,const void*,uint64_t n){update(s,l,n,false);}
extern "C" void kokkosp_finalize_library(){
  const char*prefix=std::getenv("HYPERBOLIC_ALLOCATION_PREFIX");if(!prefix)return;
  std::ofstream out(std::string(prefix)+".pid"+std::to_string(getpid())+".allocations.json");
  out<<"{\"pid\":"<<getpid()<<",\"spaces\":";table(out,spaces);out<<",\"labels\":";table(out,labels);out<<",\"fenced_kernel_timings\":"<<(timings()?"true":"false")<<",\"kernels\":{";
  bool comma=false;for(const auto&[name,k]:kernels){if(comma)out<<',';comma=true;out<<std::quoted(name)<<":{\"seconds\":"<<std::setprecision(17)<<k.seconds<<",\"calls\":"<<k.calls<<'}';}out<<"}}\n";
}
