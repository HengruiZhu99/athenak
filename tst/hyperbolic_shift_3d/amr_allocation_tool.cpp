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
struct Allocation {std::string label;uint64_t bytes;};
std::map<std::pair<std::string,uintptr_t>,Allocation> allocations;
uint64_t unmatched_frees=0,mismatched_labels=0,mismatched_sizes=0;
using Clock=std::chrono::steady_clock;
struct KernelUse {double seconds=0;uint64_t calls=0;};
std::map<std::string,KernelUse> kernels;
std::map<uint64_t,std::pair<std::string,Clock::time_point>> running;
uint64_t next_id=0;
bool timings(){static const bool enabled=std::getenv("HYPERBOLIC_PROFILE_KERNELS")!=nullptr;return enabled;}
void begin(const char*label,uint64_t*id){*id=0;if(!timings())return;std::lock_guard<std::mutex> lock(guard);*id=++next_id;running[*id]={label,Clock::now()};}
void end(uint64_t id){if(!timings())return;const auto now=Clock::now();std::lock_guard<std::mutex> lock(guard);auto it=running.find(id);if(it==running.end())return;auto&k=kernels[it->second.first];k.seconds+=std::chrono::duration<double>(now-it->second.second).count();++k.calls;running.erase(it);}

void change(Use&v,uint64_t n,bool add){if(add){v.live+=n;v.peak=std::max(v.peak,v.live);++v.allocations;}else v.live-=n;}
void update(SpaceHandle s,const char*label,const void*pointer,uint64_t n,bool add){
  std::lock_guard<std::mutex> lock(guard);
  const auto key=std::make_pair(std::string(s.name),reinterpret_cast<uintptr_t>(pointer));
  if(add){
    allocations.emplace(key,Allocation{label,n});
    change(labels[key.first+":"+label],n,true);change(spaces[key.first],n,true);
  }else{
    const auto it=allocations.find(key);
    if(it==allocations.end()){++unmatched_frees;return;}
    mismatched_labels+=it->second.label!=label;mismatched_sizes+=it->second.bytes!=n;
    // SYCL scratch can be freed with an "[unlabeled]" callback. Attribute
    // release to the allocation's pointer identity, not the free label.
    change(labels[key.first+":"+it->second.label],it->second.bytes,false);
    change(spaces[key.first],it->second.bytes,false);allocations.erase(it);
  }
}
void table(std::ostream&o,const std::map<std::string,Use>&rows){bool comma=false;o<<'{';for(auto&[name,v]:rows){if(comma)o<<',';comma=true;o<<std::quoted(name)<<":{\"live\":"<<v.live<<",\"peak\":"<<v.peak<<",\"allocations\":"<<v.allocations<<'}';}o<<'}';}
}
extern "C" void kokkosp_request_tool_settings(uint32_t,Kokkos_Tools_ToolSettings*s){s->requires_global_fencing=timings();}
extern "C" void kokkosp_begin_parallel_for(const char*l,uint32_t,uint64_t*i){begin(l,i);}
extern "C" void kokkosp_end_parallel_for(uint64_t i){end(i);}
extern "C" void kokkosp_begin_parallel_reduce(const char*l,uint32_t,uint64_t*i){begin(l,i);}
extern "C" void kokkosp_end_parallel_reduce(uint64_t i){end(i);}
extern "C" void kokkosp_allocate_data(SpaceHandle s,const char*l,const void*p,uint64_t n){update(s,l,p,n,true);}
extern "C" void kokkosp_deallocate_data(SpaceHandle s,const char*l,const void*p,uint64_t n){update(s,l,p,n,false);}
extern "C" void kokkosp_finalize_library(){
  const char*prefix=std::getenv("HYPERBOLIC_ALLOCATION_PREFIX");if(!prefix)return;
  std::ofstream out(std::string(prefix)+".pid"+std::to_string(getpid())+".allocations.json");
  const char*rank=std::getenv("PALS_RANKID");if(!rank)rank=std::getenv("PMI_RANK");if(!rank)rank=std::getenv("OMPI_COMM_WORLD_RANK");
  out<<"{\"rank\":"<<(rank?rank:"-1")<<",\"pid\":"<<getpid()<<",\"unmatched_frees\":"<<unmatched_frees<<",\"mismatched_free_labels\":"<<mismatched_labels<<",\"mismatched_free_sizes\":"<<mismatched_sizes<<",\"spaces\":";table(out,spaces);out<<",\"labels\":";table(out,labels);out<<",\"fenced_kernel_timings\":"<<(timings()?"true":"false")<<",\"kernels\":{";
  bool comma=false;for(const auto&[name,k]:kernels){if(comma)out<<',';comma=true;out<<std::quoted(name)<<":{\"seconds\":"<<std::setprecision(17)<<k.seconds<<",\"calls\":"<<k.calls<<'}';}out<<"}}\n";
}
