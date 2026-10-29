#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <chrono>
#include "mesh/cubed_sphere/paired_trace_penalty.hpp"
#include "fixed_composite_mesh.hpp"
int main(int argc,char**argv){
  try{
    const int n=argc>1?std::stoi(argv[1]):12,layers=argc>2?std::stoi(argv[2]):1;
    const auto begin=std::chrono::steady_clock::now();
    const bool shifted=argc>3&&std::stoi(argv[3]);
    cubed_sphere::CompositeDifference g(hyperbolic_reproduction::ThinCoreMesh(n,layers,shifted),4,true);
    g.SetInnerTransitionLogical(true);
    if(argc>4){
      if(std::ifstream(argv[4]).good())throw std::runtime_error("refusing to overwrite coordinate output");
      std::ofstream coordinates(argv[4]);coordinates<<std::setprecision(17)<<"x,y,z,r,volume\n";
      for(size_t p=0;p<g.active.size();++p){const auto*x=g.geometry[p].x;
        coordinates<<x[0]<<','<<x[1]<<','<<x[2]<<','<<std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])<<','<<g.DiagnosticVolume(p)<<'\n';}
      if(!coordinates)throw std::runtime_error("coordinate write failed");
    }
    const auto built=std::chrono::steady_clock::now();cubed_sphere::PairedTracePenalty p(g);
    const auto planned=std::chrono::steady_clock::now();
    std::mt19937 gen(7921);std::uniform_real_distribution<Real> random(-1,1);
    std::vector<Real> q(g.active.size());for(auto&v:q)v=random(gen);
    Real loss=0;const auto rate=p.Apply(q,1,.25,&loss);Real energy=0;
    for(size_t i=0;i<q.size();++i)energy+=q[i]*g.volumes[i]*rate[i];
    const Real identity=std::abs(energy-loss)/std::max(Real(1),std::abs(loss));
    if(loss>0||identity>2e-12)throw std::runtime_error("paired energy identity failed");
    Real constant=0,linear=0,normal_linear=0,wave_jump=0,wave_normal=0;
    for(const auto&r:p.rows){Real c=0,l=0,w=0;
      for(const auto&t:r.terms){const auto*x=g.geometry[t.point].x;c+=t.coefficient;
        l+=t.coefficient*(x[0]+2*x[1]-3*x[2]);w+=t.coefficient*std::sin(7*x[0]+4*x[1]-3*x[2]);}
      constant=std::max(constant,std::abs(c));
      (r.normal?normal_linear:linear)=std::max(r.normal?normal_linear:linear,std::abs(l));
      (r.normal?wave_normal:wave_jump)=std::max(r.normal?wave_normal:wave_jump,std::abs(w));
    }
    if(constant>5e-12)throw std::runtime_error("trace does not preserve constants");
    Real min_radius=1;for(const auto&x:g.geometry)min_radius=std::min(min_radius,std::sqrt(x.x[0]*x.x[0]+x.x[1]*x.x[1]+x.x[2]*x.x[2]));
    if(!(min_radius>0))throw std::runtime_error("puncture node at origin");
    size_t terms=0;for(const auto&r:p.rows)terms+=r.terms.size();
    std::cout<<std::setprecision(17)<<"{\"status\":\"passed\",\"n\":"<<n<<",\"core_layers\":"<<layers
      <<",\"evolved\":"<<g.active.size()<<",\"core_traces\":"<<p.core_traces<<",\"shell_traces\":"<<p.shell_traces
      <<",\"trace_terms\":"<<terms<<",\"energy_identity_relative\":"<<identity<<",\"energy_rate\":"<<energy
      <<",\"constant_error\":"<<constant<<",\"linear_value_jump_max\":"<<linear<<",\"linear_normal_jump_max\":"<<normal_linear
      <<",\"oblique_wave_value_jump_max\":"<<wave_jump<<",\"oblique_wave_normal_jump_max\":"<<wave_normal
      <<",\"upper_value\":"<<p.upper_value<<",\"upper_normal\":"<<p.upper_normal<<",\"minimum_radius\":"<<min_radius
      <<",\"mesh_setup_seconds\":"<<std::chrono::duration<double>(built-begin).count()
      <<",\"trace_setup_seconds\":"<<std::chrono::duration<double>(planned-built).count()<<"}\n";
  }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
