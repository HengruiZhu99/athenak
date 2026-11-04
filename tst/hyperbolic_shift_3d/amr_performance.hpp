// AMR-driver observability only. No fences, collectives or scientific state.
#ifndef HYPERBOLIC_AMR_PERFORMANCE_HPP_
#define HYPERBOLIC_AMR_PERFORMANCE_HPP_
#include <sys/resource.h>
#include <map>

struct AMRPerformance {
  struct Measurement {double seconds=0;uint64_t calls=0;};
  std::map<std::string,Measurement> phases;
  std::vector<std::pair<std::string,double>> samples;
  std::ofstream generations;
  std::string prefix;
  double started=MPI_Wtime(),tick=started;
  std::string phase="startup";
  int rank=global_variable::my_rank;
  explicit AMRPerformance(const std::string&output):prefix(output){
    generations.open(prefix+".rank"+std::to_string(rank)+".generations.jsonl");
    generations<<std::setprecision(17);
  }
  void Next(const std::string&name){
    const double now=MPI_Wtime();auto&m=phases[phase];m.seconds+=now-tick;++m.calls;
    samples.emplace_back(phase,now-tick);tick=now;phase=name;
  }
  static uint64_t RSS(){
    rusage u{};getrusage(RUSAGE_SELF,&u);
#ifdef __APPLE__
    return u.ru_maxrss;
#else
    return uint64_t(u.ru_maxrss)*1024;
#endif
  }
  void Generation(Evolution&e,const h::CartesianAMR&a,const char*event){
    std::map<std::string,size_t> work;std::set<int> blocks;
    size_t radial=0,overlap=0;
    const auto&g=e.Model().grid;
    for(int p=e.first;p<e.last;++p){const auto&c=g.cells[p];const auto&b=g.mesh.blocks[c.block];blocks.insert(c.block);
      ++work[b.face<0?"cartesian_level_"+std::to_string(b.refinement):"shell_"+std::to_string(b.layer)];
      overlap+=p>=int(g.mesh.blocks.size())*e.n*e.n*e.n;
      if(e.Model().radial_plan&&e.Model().radial_plan->Contains(p))++radial;
    }
    auto&out=generations;
    out<<"{\"schema_version\":2,\"event\":\""<<event<<"\",\"step\":"<<e.steps<<",\"leaves\":"<<std::count_if(g.mesh.blocks.begin(),g.mesh.blocks.end(),[](const auto&b){return b.face<0;})
       <<",\"active\":"<<e.local<<",\"overlap\":"<<overlap<<",\"radial_rows\":"<<radial
       <<",\"surface_rows\":"<<e.surface_local<<",\"sampled_normal_rows\":"<<e.sampled_normal()->LocalCount()<<",\"sampled_normal_points\":"<<e.sampled_normal()->LocalPoleSamples().extent(0)/31<<",\"selected_blocks\":"<<blocks.size()
       <<",\"block_span\":"<<e.last_block-e.first_block<<",\"work\":{";
    bool comma=false;for(auto&v:work){if(comma)out<<',';comma=true;out<<'"'<<v.first<<"\":"<<v.second;}
    out<<"},\"setup_seconds\":[";for(int i=0;i<8;++i){if(i)out<<',';out<<e.setup_seconds[i];}
    out<<"],\"nested_seconds\":{\"rhs\":"<<e.rhs_seconds<<",\"exchange\":"<<e.exchange_seconds
       <<",\"diagnostic_transfer\":"<<e.diagnostic_transfer_seconds<<",\"closure\":"<<e.closure_seconds<<",\"halo\":"<<e.halo_seconds
       <<",\"error_sync_overlapping\":"<<e.error_sync_seconds<<"},\"boundary_subphases\":[";
    for(int i=0;i<6;++i){if(i)out<<',';out<<e.boundary_phase_seconds[i];}
    out<<"],\"diagnostic_transfer_calls\":"<<e.diagnostic_transfer_calls<<",\"sparse_calls\":"<<e.sparse()->Calls()<<",\"sparse_send_bytes_per_call\":"<<31*sizeof(Real)*e.sparse()->SendPoints()
       <<",\"sparse_receive_bytes_per_call\":"<<31*sizeof(Real)*e.sparse()->ReceivePoints()
       <<",\"sparse_send_peers\":"<<e.sparse()->SendPeers()<<",\"sparse_receive_peers\":"<<e.sparse()->ReceivePeers()
       <<",\"halo_rows\":"<<e.exchange()->HaloRows()<<",\"normal_halo_rows\":"<<e.normal_exchange()->HaloRows()
       <<",\"memory_bytes\":{\"two_padded\":"<<(e.exchange()->padded.span()+e.normal_exchange()->padded.span())*sizeof(Real)
       <<",\"global_state_base\":"<<(e.state.span()+e.base.span())*sizeof(Real)
       <<",\"host_owned_padding\":"<<e.owned_padding.capacity()*sizeof(Real)
       <<",\"native_state_compensation\":"<<(a.state.span()+a.coarse.span()+a.compensation.span()+a.coarse_compensation.span())*sizeof(Real)
       <<",\"process_peak_rss\":"<<RSS()<<"}}\n";out.flush();
  }
  void Save(const char*status){
    Next("report_write");
    std::ofstream times(prefix+".rank"+std::to_string(rank)+".samples.jsonl");times<<std::setprecision(17);
    for(const auto&s:samples)times<<"{\"phase\":\""<<s.first<<"\",\"seconds\":"<<s.second<<"}\n";
    std::ofstream out(prefix+".rank"+std::to_string(rank)+".performance.json");out<<std::setprecision(17);
    out<<"{\"status\":\""<<status<<"\",\"rank\":"<<rank<<",\"wall_seconds\":"<<tick-started
       <<",\"peak_rss_bytes\":"<<RSS()<<",\"exclusive_phases\":{";
    bool comma=false;for(auto&v:phases){if(comma)out<<',';comma=true;
      out<<'"'<<v.first<<"\":{\"seconds\":"<<v.second.seconds<<",\"calls\":"<<v.second.calls<<'}';}
    out<<"},\"semantics\":\"host wall timers; no added fences; generation nested timers overlap driver phases; RSS includes host memory only\"}\n";
  }
};
#endif
