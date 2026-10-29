#include <chrono>
#include <iostream>
#include <filesystem>
#include "coupled_diagnostics.hpp"
namespace h=z4c::hyperboloidal;
int main(int argc,char**argv){
  Kokkos::initialize(argc,argv);int code=0;
  try {
    if(argc<3)throw std::invalid_argument("usage: coupled n output [initial|-] [t] [dt] [adaptive=1]");
    const int n=std::stoi(argv[1]);const std::string output=argv[2];
    const std::string initial=argc>3?argv[3]:"-";
    const Real end=argc>4?std::stod(argv[4]):0,dt=argc>5?std::stod(argv[5]):.0001;
    const bool adaptive=argc>6?std::stoi(argv[6]):0;
    const Real core=argc>7?std::stod(argv[7]):.125;
    const int workers=argc>8?std::stoi(argv[8]):1;
    if(workers<1||workers>8)throw std::invalid_argument("workers must be between one and eight");
    if(n%2||n<6||dt<=0||end<0)throw std::invalid_argument("even n>=6, dt>0, end>=0 required");
    if(std::filesystem::exists(output+".json"))throw std::runtime_error("refusing to overwrite existing result");
    const auto start=std::chrono::steady_clock::now();
    h::HyperbolicComposite e(cubed_sphere::CompositeMesh(core,{.45,.75,1},1,n,3),adaptive);e.workers=workers;
    std::ofstream coords(output+".coordinates.csv");coords<<std::setprecision(17)<<"x,y,z,r,volume\n";
    for(size_t p=0;p<e.grid.active.size();++p){const auto*x=e.grid.geometry[p].x;
      coords<<x[0]<<','<<x[1]<<','<<x[2]<<','<<e.Radius(x)<<','<<e.grid.volumes[p]<<'\n';}coords.close();
    if(initial=="pulse")for(size_t p=0;p<e.grid.active.size();++p){
      const Real r=e.Radius(e.grid.geometry[p].x),z=(r-.4)/.12;
      if(std::abs(z)<1)e.state[p*e.nf+18]=.01*std::exp(1-1/(1-z*z));
    }
    if(initial!="-"&&initial!="pulse"){std::ifstream in(initial);size_t count=0;in>>count;
      if(count!=e.state.size())throw std::runtime_error("initial field count mismatch");
      for(auto&v:e.state)if(!(in>>v)||!std::isfinite(v))throw std::runtime_error("invalid initial state");
      std::string trailing;if(in>>trailing)throw std::runtime_error("trailing initial data");}
    h::HyperbolicComposite::V boundary;
    Real initial_rhs=0;
    std::string status="completed",error;
    try{const auto rate=e.RHS(e.state,dt,boundary);
      for(Real x:rate)initial_rhs=std::max(initial_rhs,std::abs(x));
      coupled_diagnostics::Write(e,output+".initial");
    }catch(const std::exception&ex){status="failed";error=ex.what();code=2;}
    std::ofstream history(output+".history.csv");history<<std::setprecision(17)<<"t,null,compatibility,pole,clock,gap,cfl,interpolated_null,interpolated_null_before\n";
    auto record=[&]{history<<e.time<<','<<e.null_max<<','<<e.compatibility_max<<','<<e.pole_max<<','<<e.clock_max<<','<<e.gap_min<<','<<e.cfl_max<<','<<e.interpolated_null_max<<','<<e.interpolated_null_before_max<<'\n';history.flush();};record();
    try {while(status=="completed"&&e.time<end-1e-14){
      if(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()>400){status="bounded_stop";break;}
      e.Step(std::min(dt,end-e.time));if(e.steps%10==0)record();}
    }catch(const std::exception&ex){status="failed";error=ex.what();code=2;}
    if(status=="completed")try{e.RHS(e.state,dt,boundary);coupled_diagnostics::Write(e,output);}
      catch(const std::exception&ex){status="failed";error=ex.what();code=2;}
    record();std::ofstream state(output+".state.csv");state<<std::setprecision(17);
    for(size_t p=0;p<e.grid.active.size();++p){for(int f=0;f<e.nf;++f){if(f)state<<',';state<<e.state[p*e.nf+f];}state<<'\n';}
    std::ofstream edge(output+".boundary.csv");edge<<std::setprecision(17);
    for(size_t p=0;p<e.scri.rows.size();++p){for(int f=0;f<e.nf;++f){if(f)edge<<',';edge<<e.scri.values[p*e.nf+f];}edge<<'\n';}
    auto write_vector=[&](const std::string&suffix,const auto&values){std::ofstream f(output+suffix);f<<std::setprecision(17);
      for(size_t i=0;i<values.size();++i)f<<values[i]<<((i+1)%e.nf?',':'\n');};
    write_vector(".compensation.csv",e.compensation);write_vector(".boundary-compensation.csv",e.edge_compensation);
    if(!e.failed_state.empty()){write_vector(".failed-stage.csv",e.failed_state);write_vector(".failed-boundary.csv",e.failed_boundary);}
    std::ofstream report(output+".json");report<<std::setprecision(17)
      <<"{\"status\":\""<<status<<"\",\"error\":\""<<error<<"\",\"n\":"<<n
      <<",\"cells\":"<<e.grid.active.size()<<",\"t\":"<<e.time<<",\"steps\":"<<e.steps
      <<",\"failed_stage\":"<<e.failed_stage<<",\"workers\":"<<workers<<",\"core_half_width\":"<<core<<",\"dt\":"<<dt<<",\"ko\":"<<e.ko
      <<",\"adaptive\":"<<(adaptive?"true":"false")<<",\"initial_rhs_max\":"<<initial_rhs
      <<",\"interpolated_null_max\":"<<e.interpolated_null_max<<",\"interpolated_null_before_max\":"<<e.interpolated_null_before_max
      <<",\"null_max\":"<<e.null_max<<",\"compatibility_max\":"<<e.compatibility_max
      <<",\"pole_max\":"<<e.pole_max<<",\"seconds\":"
      <<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"}\n";
    std::cout<<"status="<<status<<" t="<<e.time<<" cells="<<e.grid.active.size()<<" initial_rhs="<<initial_rhs<<'\n';
    if(status=="bounded_stop")code=3;
  }catch(const std::exception&ex){std::cerr<<ex.what()<<'\n';code=1;}
  Kokkos::finalize();return code;
}
