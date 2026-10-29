// Read-only full-coupled candidate action. No evolution or equations are defined
// here: both selectors call HyperbolicComposite::EvaluateRHS after the same
// native retraction. Old directional adapters and their libraries are untouched.
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include "z4c/hyperboloidal/hyperbolic_composite.hpp"
#include "z4c/hyperboloidal/algebraic_projection.hpp"

namespace {
namespace h=z4c::hyperboloidal;
constexpr int ti[6]={0,0,0,1,1,2},tj[6]={0,1,2,1,2,2};
std::string error;
void Basis(int a,Real b[3][3]) {
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)b[i][j]=0;
  if(a==0){b[0][0]=1/std::sqrt(2.);b[1][1]=-b[0][0];}
  else if(a==1){b[0][0]=b[1][1]=1/std::sqrt(6.);b[2][2]=-2*b[0][0];}
  else{const int i=a==4?1:0,j=a==2?1:2;b[i][j]=b[j][i]=1/std::sqrt(2.);}
}
struct Probe {
  h::HyperbolicComposite model;
  Real dt;
  Probe(int n,Real epsilon,Real time_step,int variant):
      model(cubed_sphere::CompositeMesh(.125,{.45,.75,1},1,n,3),false,true,epsilon,
        variant?h::HyperbolicComposite::NormalDerivative::sampled_numerator:
                h::HyperbolicComposite::NormalDerivative::analytic_jets,
        variant==2?h::HyperbolicComposite::RadialClosure::donor_seven:
                   h::HyperbolicComposite::RadialClosure::ghost_lift),dt(time_step) {}
  int VolumeSize()const{return 26*model.grid.active.size();}
  int Size()const{return VolumeSize()+19*model.scri.rows.size();}
  static void Retract(const Real*input,std::vector<Real>&values,int fields,Real&det,Real&trace) {
    std::fill(values.begin(),values.end(),Real(0));
    for(size_t p=0;p<values.size()/31;++p){
      auto*v=&values[31*p];const auto*z=input+fields*p;
      for(int a=0;a<fields;++a)if(!std::isfinite(z[a]))throw std::runtime_error("nonfinite input coordinate");
      v[0]=1+z[0];
      for(int a=0;a<5;++a){Real b[3][3];Basis(a,b);
        for(int t=0;t<6;++t){v[1+t]+=z[1+a]*b[ti[t]][tj[t]];v[7+t]+=z[6+a]*b[ti[t]][tj[t]];}}
      v[13]=z[11];v[14]=z[12];
      for(int a=0;a<3;++a){v[15+a]=z[13+a];v[19+a]=z[16+a];}
      if(fields==26){v[18]=z[19];for(int a=0;a<3;++a){v[22+a]=z[20+a];v[25+a]=z[23+a];}}
      const auto projected=h::ProjectAlgebraicNativeState(v);
      if(!projected.valid)throw std::runtime_error("invalid diagnostic algebraic retraction");
      det=std::max(det,std::abs(projected.after.determinant-1));
      trace=std::max(trace,std::abs(projected.after.trace));
    }
  }
  static void Extract(const std::vector<Real>&rate,Real*output,int fields) {
    for(size_t p=0;p<rate.size()/31;++p){
      const auto*r=&rate[31*p];auto*out=output+fields*p;out[0]=r[0];
      for(int a=0;a<5;++a){Real b[3][3];Basis(a,b);out[1+a]=out[6+a]=0;
        for(int t=0;t<6;++t){const Real w=(ti[t]==tj[t]?1:2)*b[ti[t]][tj[t]];
          out[1+a]+=w*r[1+t];out[6+a]+=w*r[7+t];}}
      out[11]=r[13];out[12]=r[14];
      for(int a=0;a<3;++a){out[13+a]=r[15+a];out[16+a]=r[19+a];}
      if(fields==26){out[19]=r[18];for(int a=0;a<3;++a){out[20+a]=r[22+a];out[23+a]=r[25+a];}}
      for(int f=0;f<fields;++f)if(!std::isfinite(out[f]))throw std::runtime_error("nonfinite diagnostic RHS");
    }
  }
  void Evaluate(const Real*input,Real*output,Real*statistics) {
    Real det=0,trace=0;
    Retract(input,model.state,26,det,trace);Retract(input+VolumeSize(),model.scri.values,19,det,trace);
    model.null_max=model.compatibility_max=model.pole_max=model.clock_max=0;
    model.interpolated_null_before_max=model.interpolated_null_max=0;
    model.angular_cfl=model.angular_cfl_max=model.cfl_max=0;
    model.gap_min=std::numeric_limits<Real>::infinity();
    model.BoundaryValues();std::vector<Real> edge;
    // The selected native path finalizes geometry rates, reconstructs dependent
    // D, refreshes ghosts, and only then evaluates all coupled volume equations.
    const auto rate=model.EvaluateRHS(model.state,dt,edge);
    Extract(rate,output,26);Extract(edge,output+VolumeSize(),19);
    statistics[0]=model.null_max;statistics[1]=model.compatibility_max;
    statistics[2]=det;statistics[3]=trace;statistics[4]=model.pole_max;
    statistics[5]=model.clock_max;statistics[6]=model.angular_cfl_max;
    statistics[7]=model.interpolated_null_max;statistics[8]=model.interpolated_null_before_max;
    statistics[9]=model.gap_min;statistics[10]=model.cfl_max;
  }
};
}
extern "C" const char* scri_candidate_error(){return error.c_str();}
extern "C" void* scri_candidate_create(int n,double epsilon,double dt,int sampled){
  static_assert(sizeof(Real)==sizeof(double),"candidate diagnostic requires FP64");
  try{
    if((n!=6&&n!=16)||epsilon!=2||!std::isfinite(dt)||!(dt>0)||dt>.0005||
       (sampled<0||sampled>2))throw std::runtime_error("invalid bounded candidate configuration");
    return new Probe(n,epsilon,dt,sampled);
  }catch(const std::exception&e){error=e.what();return nullptr;}
}
extern "C" void scri_candidate_destroy(void*p){delete static_cast<Probe*>(p);}
extern "C" int scri_candidate_size(void*p){return static_cast<Probe*>(p)->Size();}
extern "C" int scri_candidate_volume_size(void*p){return static_cast<Probe*>(p)->VolumeSize();}
extern "C" int scri_candidate_evaluate(void*p,const double*input,double*output,double*statistics){
  try{static_cast<Probe*>(p)->Evaluate(input,output,statistics);return 0;}
  catch(const std::exception&e){error=e.what();return 1;}
}
