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
  // Diagnostic surface assembly only. No term is removed from a native RHS.
  // Terms: regular R, -AD-normal N, 4N/h, -8C/h, Q (including B), B flux.
  void SurfaceParts(const Real*input,Real*full,Real*output,Real*statistics) {
    model.angular_epsilon=configured_epsilon;Real det=0,trace_error=0;
    Retract(input,model.state,26,det,trace_error);
    Retract(input+VolumeSize(),model.scri.values,19,det,trace_error);
    model.BoundaryValues();const auto original=model.scri.values;
    const auto padded=model.ActivePadded(model.state);
    const auto actual=model.BoundaryRates(padded,dt);
    model.scri.values=original;  // Materialize each sampled field at the same input.
    using V=std::vector<Real>;const int size=19*model.scri.rows.size();
    V terms[6];for(auto&v:terms)v.assign(original.size(),0);
    if(configured_epsilon>0)terms[4]=model.angular_plan->Apply(original,configured_epsilon);
    for(size_t p=0;p<model.scri.rows.size();++p){
      const auto&row=model.scri.rows[p];const auto*x=row.geometry.x;
      const auto sample=model.scri.Sample(p,padded,original,31);
      const auto u=model.Jet(x,sample.value,sample.d,sample.dd);
      const auto o=model.Omega(x,u,true);const auto g=h::Geometry(u.metric);
      auto parts=h::ConformalRHS(u,o,4.,-.99);
      if(!h::HyperbolicDamping(x,u,o,parts))throw std::runtime_error("invalid surface diagnostic damping");
      const auto pole=h::TensorPoleNormalJet(u,o,x,x,4.,-.99,1.,1.,true,true);
      const auto normal=h::TensorScriLimit(h::Z4cRHS<Real>{},pole,-1.);
      const auto trace=h::ScriRegularityTraces(u,o);const Real spacing=row.radial_spacing;
      h::Z4cRHS<Real> weak{},regularity{};Real q=0;
      for(int i=0;i<3;++i)for(int j=0;j<3;++j)q+=g.inverse[i][j]*x[i]*x[j];
      weak.chi=4/spacing*parts.pole.chi;weak.trace=4/spacing*parts.pole.trace;
      weak.theta=4/spacing*parts.pole.theta;
      regularity.chi=-8/spacing*(u.chi.value-1/q);
      regularity.trace=-8/spacing*(u.trace.value+3);regularity.theta=-8/spacing*u.theta.value;
      for(int i=0;i<3;++i){weak.lambda[i]=4/spacing*parts.pole.lambda[i];regularity.lambda[i]=-8/spacing*trace.lambda[i];
        for(int j=0;j<3;++j){weak.a[i][j]=4/spacing*parts.pole.a[i][j];regularity.a[i][j]=-8/spacing*trace.a[i][j];}}
      model.StoreGeometry(parts.regular,&terms[0][31*p]);model.StoreGeometry(normal,&terms[1][31*p]);
      model.StoreGeometry(weak,&terms[2][31*p]);model.StoreGeometry(regularity,&terms[3][31*p]);
      const auto B=model.Vector(19,sample.value,sample.d);
      const auto gauge=h::HyperbolicGauge(x,u,B,&sample.value[25]);
      for(int a=0;a<3;++a)terms[5][31*p+19+a]=gauge.flux_rate[a];
    }
    Extract(actual,full,19);for(int t=0;t<6;++t)Extract(terms[t],output+t*size,19);
    Real error_max=0,scale=0;
    for(int i=0;i<size;++i){Real sum=0;for(int t=0;t<6;++t)sum+=output[t*size+i];
      error_max=std::max(error_max,std::abs(sum-full[i]));scale=std::max(scale,std::abs(full[i]));}
    statistics[0]=error_max;statistics[1]=scale;statistics[2]=det;statistics[3]=trace_error;
  }
  // Numerator-factorization action only: five native half-cell radial donors
  // and the actual endpoint, with the layout and all penalties held fixed.
  void SampledNormal(const Real*input,Real*sampled,Real*automatic,Real*numerators,
                     Real*statistics,bool perturb_d) {
    model.angular_epsilon=configured_epsilon;Real det=0,trace_error=0;
    Retract(input,model.state,26,det,trace_error);
    Retract(input+VolumeSize(),model.scri.values,19,det,trace_error);model.BoundaryValues();
    auto padded=model.ActivePadded(model.state);model.FillGhosts(padded);Real ghost_error=0;
    if(perturb_d){
      for(size_t p=0;p<model.state.size()/31;++p)for(int a=0;a<3;++a)
        model.state[31*p+25+a]+=.001*(int((p+3*a)%17)-8);
      for(size_t p=0;p<model.scri.values.size()/31;++p)for(int a=0;a<3;++a)
        model.scri.values[31*p+25+a]+=.002*(int((p+5*a)%19)-9);
      auto changed=model.ActivePadded(model.state);model.FillGhosts(changed);
      for(int f=0;f<31;++f)if(f<=18||(f>=22&&f<25))for(int p=0;p<model.grid.count;++p)
        ghost_error=std::max(ghost_error,std::abs(changed[f*model.grid.count+p]-padded[f*model.grid.count+p]));
      padded=std::move(changed);
    }
    using V=std::vector<Real>;V nodes[6];for(auto&v:nodes)v.assign(model.scri.values.size(),0);
    V alternate(model.scri.values.size(),0),ad(alternate.size(),0);
    const int n=model.grid.n,size=19*model.scri.rows.size();
    const Real offsets[6]={-4.5,-3.5,-2.5,-1.5,-.5,0};Real weights[6]{};
    for(int i=0;i<6;++i){cubed_sphere::Jet basis(1);
      for(int j=0;j<6;++j)if(i!=j)
        basis=basis*(cubed_sphere::Jet::Variable(0,0)-cubed_sphere::Jet(offsets[j]))/cubed_sphere::Jet(offsets[i]-offsets[j]);
      weights[i]=basis.d[0];}
    Real moments=0;
    for(int degree=0;degree<=5;++degree){Real derivative=0;
      for(int i=0;i<6;++i)derivative+=weights[i]*std::pow(offsets[i],degree);
      moments=std::max(moments,std::abs(derivative-(degree==1?1:0)));}
    for(size_t p=0;p<model.scri.rows.size();++p){const auto&row=model.scri.rows[p];
      for(int s=0;s<5;++s){
        const size_t active=row.block*n*n*n+(row.k*n+row.j)*n+n-5+s;
        const auto*x=model.grid.geometry[active].x;V value(31);
        std::vector<std::array<Real,3>> d(31);std::vector<std::array<std::array<Real,3>,3>> dd(31);
        for(int f=0;f<31;++f){value[f]=model.state[31*active+f];Real df[3],ddf[3][3];
          model.grid.Derivatives(padded,f,active,df,ddf,f<=6||f==18||(f>=22&&f<25));
          for(int a=0;a<3;++a){d[f][a]=df[a];for(int b=0;b<3;++b)dd[f][a][b]=ddf[a][b];}}
        const auto u=model.Jet(x,value,d,dd);const auto o=model.Omega(x,u,false);
        auto parts=h::ConformalRHS(u,o,4.,-.99);
        if(!h::HyperbolicDamping(x,u,o,parts))throw std::runtime_error("invalid interior numerator diagnostic");
        model.StoreGeometry(parts.pole,&nodes[s][31*p]);
      }
      const auto*x=row.geometry.x;const auto endpoint=model.scri.Sample(p,padded,model.scri.values,31);
      const auto u=model.Jet(x,endpoint.value,endpoint.d,endpoint.dd);const auto o=model.Omega(x,u,true);
      auto parts=h::ConformalRHS(u,o,4.,-.99);
      if(!h::HyperbolicDamping(x,u,o,parts))throw std::runtime_error("invalid endpoint numerator diagnostic");
      model.StoreGeometry(parts.pole,&nodes[5][31*p]);
      const auto pole=h::TensorPoleNormalJet(u,o,x,x,4.,-.99,1.,1.,true,true);
      model.StoreGeometry(h::TensorScriLimit(h::Z4cRHS<Real>{},pole,-1.),&ad[31*p]);
      for(int f=0;f<31;++f)for(int s=0;s<6;++s)
        alternate[31*p+f]-=weights[s]*(nodes[s][31*p+f]-nodes[5][31*p+f])/row.radial_spacing;
    }
    Extract(alternate,sampled,19);Extract(ad,automatic,19);
    for(int s=0;s<6;++s)Extract(nodes[s],numerators+s*size,19);
    statistics[0]=moments;statistics[1]=ghost_error;
    statistics[2]=det;statistics[3]=trace_error;statistics[4]=weights[5];
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
extern "C" int scri_directional_surface_parts(void*p,const double*input,double*full,double*parts,double*statistics){
  try{static_cast<Probe*>(p)->SurfaceParts(input,full,parts,statistics);return 0;}
  catch(const std::exception&e){error=e.what();return 1;}
}
extern "C" int scri_directional_sampled_normal(void*p,const double*input,double*sampled,double*automatic,
                                              double*numerators,double*statistics,int perturb_d){
  try{
    if(perturb_d!=0&&perturb_d!=1)throw std::runtime_error("invalid D-only diagnostic switch");
    static_cast<Probe*>(p)->SampledNormal(input,sampled,automatic,numerators,statistics,perturb_d!=0);return 0;
  }catch(const std::exception&e){error=e.what();return 1;}
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
