// Bounded local Minkowski control; no checkpoint admission or production path.
#include <chrono>
#include <filesystem>
#include <iostream>
#include "z4c/hyperboloidal/hyperbolic_composite.hpp"
#include "z4c/hyperboloidal/algebraic_projection.hpp"
namespace h=z4c::hyperboloidal;
using Model=h::HyperbolicComposite;using V=Model::V;

// Same compensated RK4/final algebraic projection ordering as coupled_mpi.
// This separate CPU harness does not use the legacy unprojected Model::Step.
void Step(Model&e,Real dt){
  const V begin=e.state,edge=e.scri.values,old_comp=e.compensation,old_edge=e.edge_compensation;
  V rate[4],boundary[4];
  try{
    for(int s=0;s<4;++s){rate[s]=e.EvaluateRHS(e.state,dt,boundary[s]);
      if(s<3){const Real a=s<2?dt/2:dt;
        for(size_t i=0;i<begin.size();++i)e.state[i]=begin[i]+a*rate[s][i];
        for(size_t i=0;i<edge.size();++i)e.scri.values[i]=edge[i]+a*boundary[s][i];}}
    auto accept=[&](V&v,V&comp,const V&base,const V*r,bool surface){
      for(size_t i=0;i<v.size();++i){const Real inc=dt*(r[0][i]+2*r[1][i]+2*r[2][i]+r[3][i])/6-comp[i];
        v[i]=base[i]+inc;comp[i]=surface&&(i%31==18||i%31>=22)?0:(v[i]-base[i])-inc;}
      for(size_t p=0;p<v.size()/31;++p)if(!h::ProjectAlgebraicNativeState(&v[31*p],&comp[31*p]).valid)
        throw std::runtime_error("invalid accepted algebraic geometry");
    };
    accept(e.state,e.compensation,begin,rate,false);accept(e.scri.values,e.edge_compensation,edge,boundary,true);
    e.BoundaryValues();auto padded=e.ActivePadded(e.state);e.FillGhosts(padded);e.BoundaryRates(padded,dt);
    e.time+=dt;++e.steps;
  }catch(...){e.failed_state=e.state;e.failed_boundary=e.scri.values;
    e.state=begin;e.scri.values=edge;e.compensation=old_comp;e.edge_compensation=old_edge;throw;}
}

struct Diagnostic{Real state=0,H=0,M=0;};
Diagnostic Measure(Model&e,Real dt){
  V edge;e.EvaluateRHS(e.state,dt,edge);auto padded=e.ActivePadded(e.state);e.FillGhosts(padded);
  Diagnostic d;Real volume=0;V v(31);std::vector<std::array<Real,3>>first(31);
  std::vector<std::array<std::array<Real,3>,3>>second(31);
  for(const V*data:{&e.state,&e.scri.values})for(size_t i=0;i<data->size();++i)
    d.state=std::max(d.state,std::abs((*data)[i]-(i%31==0?1:0)));
  for(size_t p=0;p<e.grid.active.size();++p){
    for(int f=0;f<31;++f){v[f]=e.state[31*p+f];Real df[3],ddf[3][3];e.BulkDerivatives(padded,f,p,df,ddf);
      for(int a=0;a<3;++a){first[f][a]=df[a];for(int b=0;b<3;++b)second[f][a][b]=ddf[a][b];}}
    const auto*x=e.grid.geometry[p].x;const auto u=e.Jet(x,v,first,second);const auto c=h::EvolvedConstraints(u,e.Omega(x,u));
    const Real w=e.grid.volumes[p];volume+=w;d.H+=w*c.hamiltonian*c.hamiltonian;d.M+=w*c.momentum_conformal_norm2;
  }
  d.H=std::sqrt(d.H/volume);d.M=std::sqrt(std::max(Real(0),d.M/volume));return d;
}

int main(int argc,char**argv){try{
  if(argc!=5)throw std::invalid_argument("usage: scri_radial_duration variant n target output-prefix");
  const int variant=std::stoi(argv[1]),n=std::stoi(argv[2]);const Real end=std::stod(argv[3]),dt=.008/n;
  const std::string output=argv[4];
  if((variant!=1&&variant!=2)||n!=6||end<=0||end>2||std::filesystem::exists(output+".json"))
    throw std::invalid_argument("bounded control requires n6, variant1/2, target in (0,2], fresh output");
  const auto start=std::chrono::steady_clock::now();
  auto elapsed=[&](){return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();};
  Model e(cubed_sphere::CompositeMesh(.125,{.45,.75,1},1,n,3),false,true,2,
    Model::NormalDerivative::sampled_numerator,variant==2?Model::RadialClosure::donor_seven:Model::RadialClosure::ghost_lift);
  std::ofstream history(output+".history.csv");history<<std::setprecision(17)<<"t,steps,state,H_L2,M_L2,null_surface,null_ghost,wall_seconds\n";
  std::string status="completed",error;Diagnostic last;Real diagnostic_time=0,next=0;
  auto record=[&](){last=Measure(e,dt);diagnostic_time=e.time;
    history<<e.time<<','<<e.steps<<','<<last.state<<','<<last.H<<','<<last.M<<','<<e.null_max<<','<<e.interpolated_null_max<<','<<elapsed()<<'\n';history.flush();
    if(!std::isfinite(last.state)||!std::isfinite(last.H)||!std::isfinite(last.M)||last.state>1e-8||last.H>1e-7||last.M>1e-7||e.null_max>1e-12||e.interpolated_null_max>1e-12)
      status="science_failed";
  };
  try{
    record();next=.05;
    while(status=="completed"&&e.time<end-1e-14){
      if(elapsed()>325){status="bounded";break;}
      Step(e,std::min(dt,end-e.time));
      if(e.time>=next-1e-14){record();next+=.05;}
    }
    if(e.time!=diagnostic_time)record();
  }catch(const std::exception&ex){status="numerical_failure";error=ex.what();}
  auto save=[&](const std::string&name,const V&v){std::ofstream f(output+name,std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(Real));};
  save(".state.bin",e.state);save(".surface.bin",e.scri.values);
  if(!e.failed_state.empty()){save(".failed-state.bin",e.failed_state);save(".failed-surface.bin",e.failed_boundary);}
  std::ofstream report(output+".json");report<<std::setprecision(17)<<"{\"status\":\""<<status<<"\",\"error\":\""<<error
    <<"\",\"variant\":"<<variant<<",\"n\":"<<n<<",\"t\":"<<e.time<<",\"target\":"<<end<<",\"dt\":"<<dt
    <<",\"steps\":"<<e.steps<<",\"diagnostic_time\":"<<diagnostic_time<<",\"state\":"<<last.state
    <<",\"H_L2\":"<<last.H<<",\"M_L2\":"<<last.M<<",\"null_surface\":"<<e.null_max
    <<",\"null_ghost\":"<<e.interpolated_null_max<<",\"seconds\":"<<elapsed()<<",\"checkpoint_admitted\":false}\n";
  std::cout<<status<<" t="<<e.time<<" state="<<last.state<<'\n';return status=="completed"?0:2;
}catch(const std::exception&ex){std::cerr<<ex.what()<<'\n';return 1;}}
