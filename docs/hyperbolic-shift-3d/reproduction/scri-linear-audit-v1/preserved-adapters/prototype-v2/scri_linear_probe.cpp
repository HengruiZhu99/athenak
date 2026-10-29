// Local diagnostic adapter only; no independent evolution equations.
#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include "z4c/hyperboloidal/hyperbolic_composite.hpp"
#include "z4c/hyperboloidal/algebraic_projection.hpp"

namespace {
namespace h=z4c::hyperboloidal;
// Frobenius-orthonormal symmetric tracefree basis at the Minkowski metric.
void Basis(int a, Real b[3][3]) {
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)b[i][j]=0;
  if(a==0){b[0][0]=1/std::sqrt(2.);b[1][1]=-b[0][0];}
  else if(a==1){b[0][0]=b[1][1]=1/std::sqrt(6.);b[2][2]=-2*b[0][0];}
  else{const int i=a==4?1:0,j=a==2?1:2;b[i][j]=b[j][i]=1/std::sqrt(2.);}
}
constexpr int fields=19;
constexpr int ti[6]={0,0,0,1,1,2},tj[6]={0,1,2,1,2,2};
std::string error;
struct Probe {
  h::HyperbolicComposite model;
  std::vector<Real> frozen;
  bool coupled;
  explicit Probe(int n,bool cross,bool full):model(cubed_sphere::CompositeMesh(.125,{.45,.75,1},1,n,3),false,cross),
      frozen(model.ActivePadded(model.state)),coupled(full) {}
  int VolumeSize()const{return coupled?26*model.grid.active.size():0;}
  int Size()const{return VolumeSize()+fields*model.scri.rows.size();}
  static void Retract(const Real*input,std::vector<Real>&values,int count,Real&det,Real&trace) {
    std::fill(values.begin(),values.end(),Real(0));
    for(size_t p=0;p<values.size()/31;++p){
      auto*v=&values[31*p];const auto*z=input+count*p;v[0]=1+z[0];
      for(int a=0;a<5;++a){Real b[3][3];Basis(a,b);
        for(int t=0;t<6;++t){v[1+t]+=z[1+a]*b[ti[t]][tj[t]];v[7+t]+=z[6+a]*b[ti[t]][tj[t]];}}
      v[13]=z[11];v[14]=z[12];
      for(int a=0;a<3;++a){v[15+a]=z[13+a];v[19+a]=z[16+a];}
      if(count==26){v[18]=z[19];for(int a=0;a<3;++a){v[22+a]=z[20+a];v[25+a]=z[23+a];}}
      const auto projected=h::ProjectAlgebraicNativeState(v);
      if(!projected.valid)throw std::runtime_error("invalid diagnostic surface retraction");
      det=std::max(det,std::abs(projected.after.determinant-1));
      trace=std::max(trace,std::abs(projected.after.trace));
    }
  }
  static void Extract(const std::vector<Real>&rate,Real*output,int count) {
    for(size_t p=0;p<rate.size()/31;++p){const auto*r=&rate[31*p];auto*out=output+count*p;
      out[0]=r[0];
      // Frobenius extraction equals the tangent of the standard final-stage
      // algebraic projection at Minkowski, where g=I and A=0.
      for(int a=0;a<5;++a){Real b[3][3];Basis(a,b);out[1+a]=out[6+a]=0;
        for(int t=0;t<6;++t){const Real w=(ti[t]==tj[t]?1:2)*b[ti[t]][tj[t]];
          out[1+a]+=w*r[1+t];out[6+a]+=w*r[7+t];}}
      out[11]=r[13];out[12]=r[14];
      for(int a=0;a<3;++a){out[13+a]=r[15+a];out[16+a]=r[19+a];}
      if(count==26){out[19]=r[18];for(int a=0;a<3;++a){out[20+a]=r[22+a];out[23+a]=r[25+a];}}
      for(int f=0;f<count;++f)if(!std::isfinite(out[f]))throw std::runtime_error("nonfinite diagnostic RHS");
    }
  }
  void Evaluate(const Real*input,Real*output,Real*statistics) {
    Real det=0,trace=0;
    if(coupled)Retract(input,model.state,26,det,trace);
    Retract(input+VolumeSize(),model.scri.values,fields,det,trace);
    model.null_max=model.compatibility_max=model.pole_max=model.clock_max=0;
    model.interpolated_null_before_max=model.interpolated_null_max=0;
    model.BoundaryValues();std::vector<Real> edge;
    if(coupled){
      // Exact existing closure -> dependent D -> ghost lift -> bulk/upwind/KO
      // pipeline, including feedback from volume donors to surface derivatives.
      const auto rate=model.EvaluateRHS(model.state,Real(.0001),edge);
      Extract(rate,output,26);
    }else{
      // Frozen active Minkowski donors; no ghost fill or volume RHS.
      edge=model.BoundaryRates(frozen,Real(.0001));
    }
    Extract(edge,output+VolumeSize(),fields);
    // No RK step or checkpoint admission occurs in either operator.
    statistics[0]=model.null_max;statistics[1]=model.compatibility_max;
    statistics[2]=det;statistics[3]=trace;statistics[4]=model.pole_max;statistics[5]=model.clock_max;
  }
};
}

extern "C" const char* scri_linear_error(){return error.c_str();}
extern "C" void* scri_linear_create(int n,int cross,int coupled){
  try{if(n<6||n>8||cross<0||cross>1||coupled<0||coupled>1)throw std::runtime_error("invalid bounded probe configuration");
    return new Probe(n,cross!=0,coupled!=0);}catch(const std::exception&e){error=e.what();return nullptr;}
}
extern "C" void scri_linear_destroy(void*p){delete static_cast<Probe*>(p);}
extern "C" int scri_linear_size(void*p){return static_cast<Probe*>(p)->Size();}
extern "C" int scri_linear_volume_size(void*p){return static_cast<Probe*>(p)->VolumeSize();}
extern "C" int scri_linear_evaluate(void*p,const double*input,double*output,double*statistics){
  static_assert(sizeof(Real)==sizeof(double),"diagnostic requires FP64");
  try{static_cast<Probe*>(p)->Evaluate(input,output,statistics);return 0;}
  catch(const std::exception&e){error=e.what();return 1;}
}
extern "C" void scri_linear_rows(void*p,double*data){
  const auto&model=static_cast<Probe*>(p)->model;
  for(size_t i=0;i<model.scri.rows.size();++i){const auto&r=model.scri.rows[i];
    for(int a=0;a<3;++a)data[6*i+a]=r.geometry.x[a];
    data[6*i+3]=model.grid.mesh.blocks[r.block].face;data[6*i+4]=r.j;data[6*i+5]=r.k;}
}
