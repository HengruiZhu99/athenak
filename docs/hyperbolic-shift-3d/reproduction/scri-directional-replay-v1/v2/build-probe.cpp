// Read-only finite-difference adapter. All equations and closures are native.
// This separate oracle preserves scri_linear_probe.cpp and supports the actual
// n16 V4 diagnostic configuration without constructing a Krylov basis.
// The derivative is T0 RHS(P(U0+Ez)); it is neither the finite-dt accepted
// RK map nor the Jacobian at the saved nonlinear checkpoint.
#include <algorithm>
#include <cmath>
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
  Real configured_epsilon,dt;
  Probe(int n,Real epsilon,Real time_step):
      model(cubed_sphere::CompositeMesh(.125,{.45,.75,1},1,n,3),false,true,epsilon),
      configured_epsilon(epsilon),dt(time_step) {}
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
  void Evaluate(const Real*input,Real*output,Real*statistics,bool with_dissipation) {
    // Each call is independent, including after a failed or counterfactual call.
    model.angular_epsilon=with_dissipation?configured_epsilon:0;
    Real det=0,trace=0;
    Retract(input,model.state,26,det,trace);
    Retract(input+VolumeSize(),model.scri.values,19,det,trace);
    model.null_max=model.compatibility_max=model.pole_max=model.clock_max=0;
    model.interpolated_null_before_max=model.interpolated_null_max=0;
    model.angular_cfl=model.angular_cfl_max=0;
    model.BoundaryValues();std::vector<Real> edge;
    const auto rate=model.EvaluateRHS(model.state,dt,edge);
    Extract(rate,output,26);Extract(edge,output+VolumeSize(),19);
    statistics[0]=model.null_max;statistics[1]=model.compatibility_max;
    statistics[2]=det;statistics[3]=trace;statistics[4]=model.pole_max;
    statistics[5]=model.clock_max;statistics[6]=model.angular_cfl_max;
  }
};
}
extern "C" const char* scri_directional_error(){return error.c_str();}
extern "C" void* scri_directional_create(int n,double epsilon,double dt){
  static_assert(sizeof(Real)==sizeof(double),"directional diagnostic requires FP64");
  try{
    if((n!=6&&n!=8&&n!=12&&n!=16)||!std::isfinite(epsilon)||(epsilon!=0&&epsilon!=2)||
       !std::isfinite(dt)||!(dt>0)||dt>.0005)throw std::runtime_error("invalid bounded directional configuration");
    return new Probe(n,epsilon,dt);
  }catch(const std::exception&e){error=e.what();return nullptr;}
}
extern "C" void scri_directional_destroy(void*p){delete static_cast<Probe*>(p);}
extern "C" int scri_directional_size(void*p){return static_cast<Probe*>(p)->Size();}
extern "C" int scri_directional_volume_size(void*p){return static_cast<Probe*>(p)->VolumeSize();}
extern "C" int scri_directional_evaluate(void*p,const double*input,double*output,double*statistics,int enabled){
  try{
    if(enabled!=0&&enabled!=1)throw std::runtime_error("invalid dissipation switch");
    static_cast<Probe*>(p)->Evaluate(input,output,statistics,enabled!=0);return 0;
  }catch(const std::exception&e){error=e.what();return 1;}
}
extern "C" int scri_directional_retract(void*p,const double*input,double*volume,double*surface,double*statistics){
  try{
    auto&probe=*static_cast<Probe*>(p);Real det=0,trace=0;
    Probe::Retract(input,probe.model.state,26,det,trace);
    Probe::Retract(input+probe.VolumeSize(),probe.model.scri.values,19,det,trace);
    std::copy(probe.model.state.begin(),probe.model.state.end(),volume);
    std::copy(probe.model.scri.values.begin(),probe.model.scri.values.end(),surface);
    statistics[0]=det;statistics[1]=trace;return 0;
  }catch(const std::exception&e){error=e.what();return 1;}
}
extern "C" void scri_directional_volume_weights(void*p,double*data){
  const auto&weights=static_cast<Probe*>(p)->model.grid.volumes;
  std::copy(weights.begin(),weights.end(),data);
}
extern "C" void scri_directional_rows(void*p,double*data){
  const auto&model=static_cast<Probe*>(p)->model;
  for(size_t i=0;i<model.scri.rows.size();++i){const auto&r=model.scri.rows[i];
    for(int a=0;a<3;++a)data[6*i+a]=r.geometry.x[a];
    data[6*i+3]=model.grid.mesh.blocks[r.block].face;data[6*i+4]=r.j;data[6*i+5]=r.k;}
}
extern "C" void scri_directional_volume_rows(void*p,double*data){
  const auto&model=static_cast<Probe*>(p)->model;const int n=model.grid.n;
  for(size_t p=0;p<model.grid.active.size();++p){const int block=p/(n*n*n),local=p%(n*n*n);
    for(int a=0;a<3;++a)data[8*p+a]=model.grid.geometry[p].x[a];
    data[8*p+3]=model.grid.mesh.blocks[block].face;data[8*p+4]=model.grid.mesh.blocks[block].layer;
    data[8*p+5]=local%n;data[8*p+6]=(local/n)%n;data[8*p+7]=local/(n*n);}
}
