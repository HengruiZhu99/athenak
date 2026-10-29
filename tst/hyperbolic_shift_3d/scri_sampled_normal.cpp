// CPU-only sampled-numerator closure checks. No timestep or evolution is run.
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include "z4c/hyperboloidal/hyperbolic_composite.hpp"
#include "z4c/hyperboloidal/algebraic_projection.hpp"

namespace h=z4c::hyperboloidal;
using Model=h::HyperbolicComposite;
using V=std::vector<Real>;
static constexpr Real dt=.0001;
static void Require(bool condition,const char*message){if(!condition)throw std::runtime_error(message);}
static bool Bits(const V&a,const V&b){return a.size()==b.size()&&std::memcmp(a.data(),b.data(),a.size()*sizeof(Real))==0;}
static Real Difference(const V&a,const V&b){Require(a.size()==b.size(),"comparison shape mismatch");Real error=0;
  for(size_t i=0;i<a.size();++i){Require(std::isfinite(a[i])&&std::isfinite(b[i]),"nonfinite fixture result");
    error=std::max(error,std::abs(a[i]-b[i]));}return error;}
static cubed_sphere::CompositeMesh Mesh(){return cubed_sphere::CompositeMesh(.125,{.45,.75,1},1,6,3);}
static void Seed(Model&model,bool perturb){
  auto fill=[&](const Real*x,Real*v){for(int f=0;f<31;++f)v[f]=(f==0?1.:0.)+
    (perturb?Real(.0001)*(1+f%4)*std::sin((2.1+.23*f)*x[0]+(.7+.11*f)*x[1]-(1.7+.17*f)*x[2]):0.);
    Require(h::ProjectAlgebraicNativeState(v).valid,"invalid seeded geometry");};
  for(size_t p=0;p<model.grid.active.size();++p)fill(model.grid.geometry[p].x,&model.state[31*p]);
  for(size_t p=0;p<model.scri.rows.size();++p)fill(model.scri.rows[p].geometry.x,&model.scri.values[31*p]);
  model.BoundaryValues();
}
static void Dump(std::ofstream&out,const V&values){
  const uint64_t size=values.size();out.write(reinterpret_cast<const char*>(&size),sizeof(size));
  out.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(Real));
  Require(bool(out),"failed default snapshot write");
}
// Compile this same fixture against the preserved V4 header with
// HYPERBOLIC_SAMPLED_DEFAULT_ONLY for an independent old-header bitwise oracle.
static void DefaultChecks(const char*filename){
  std::ofstream file;if(filename){file.open(filename,std::ios::binary);Require(bool(file),"cannot open default snapshot");}
  for(int configuration=0;configuration<3;++configuration){
    const bool cross=configuration!=0;const Real epsilon=configuration==2?2.:0.;
    Model implicit(Mesh(),false,cross,epsilon);
#ifndef HYPERBOLIC_SAMPLED_DEFAULT_ONLY
    Model explicit_ad(Mesh(),false,cross,epsilon,Model::NormalDerivative::analytic_jets);
#endif
    for(bool perturb:{false,true}){
      Seed(implicit,perturb);V edge;const auto before=implicit.state;
      const auto rhs=implicit.EvaluateRHS(before,dt,edge);
      Require(Bits(before,implicit.state),"RHS changed active independent state");
#ifndef HYPERBOLIC_SAMPLED_DEFAULT_ONLY
      Seed(explicit_ad,perturb);V explicit_edge;
      const auto explicit_rhs=explicit_ad.EvaluateRHS(explicit_ad.state,dt,explicit_edge);
      Require(Bits(rhs,explicit_rhs)&&Bits(edge,explicit_edge)&&Bits(implicit.scri.values,explicit_ad.scri.values),
              "explicit analytic selector changed default RHS/surface bits");
#endif
      if(filename){Dump(file,rhs);Dump(file,edge);Dump(file,implicit.scri.values);}
    }
  }
  std::cout<<"default_configurations=3 default_fixtures=6";
#ifndef HYPERBOLIC_SAMPLED_DEFAULT_ONLY
  std::cout<<" explicit_AD_bitwise=1\n";
#else
  std::cout<<" frozen_header_oracle=1\n";
#endif
}
#ifndef HYPERBOLIC_SAMPLED_DEFAULT_ONLY
static V Normals(const Model&model,const V&padded){
  V result(model.scri.values.size(),0);
  for(size_t p=0;p<model.scri.rows.size();++p)Model::StoreGeometry(model.SampledPoleNormalDerivative(p,padded),&result[31*p]);
  return result;
}
static Real NullDerivativeResponse(const Model&model,const V&values,const V&ad,const V&sampled,
                                  const V&ad_surface,const V&sampled_surface){
  Real error=0,change=0;
  for(size_t p=0;p<model.scri.rows.size();++p){const auto*x=model.scri.rows[p].geometry.x;const auto*v=&values[31*p];
    h::MetricJet<Real> metric{};for(int t=0;t<6;++t){const int i=Model::ti[t],j=Model::tj[t];
      metric.g[i][j]=metric.g[j][i]=v[1+t]+(i==j);}
    const auto geometry=h::Geometry(metric);Real inverse_normal[3]{},q=0,qdot=0;
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)inverse_normal[i]+=geometry.inverse[i][j]*x[j];
    for(int i=0;i<3;++i)q+=x[i]*inverse_normal[i];
    for(int t=0;t<6;++t){const int i=Model::ti[t],j=Model::tj[t];
      qdot-=(i==j?1:2)*inverse_normal[i]*(sampled[31*p+1+t]-ad[31*p+1+t])*inverse_normal[j];}
    const Real speed_t=((sampled[31*p]-ad[31*p])*q+v[0]*qdot)/(2*std::sqrt(v[0]*q));
    const Real mu=h::ShiftCoefficients(Real(1),Real(1),v[0]).mu;
    for(int a=0;a<3;++a){const Real expected=-x[a]*speed_t/mu;
      error=std::max(error,std::abs(sampled_surface[31*p+25+a]-ad_surface[31*p+25+a]-expected));
      change=std::max(change,std::abs(expected));}
  }
  Require(change>1e-8,"fixture does not exercise dependent-D response");return error;
}
// Native single-cell auxiliary equation, including the unchanged transport/KO
// additions. This checks that EvaluateRHS uses the FINAL dependent-D ghosts.
static std::array<Real,3> AuxiliaryAt(const Model&model,const V&values,const V&padded,const V&diss,size_t p){
  V value(31);std::vector<std::array<Real,3>> d(31);
  std::vector<std::array<std::array<Real,3>,3>> dd(31);
  for(int f=0;f<31;++f){value[f]=values[31*p+f];Real df[3],ddf[3][3];
    model.grid.Derivatives(padded,f,p,df,ddf,f<=6||f==18||(f>=22&&f<25));
    for(int a=0;a<3;++a){d[f][a]=df[a];for(int b=0;b<3;++b)dd[f][a][b]=ddf[a][b];}}
  const auto*x=model.grid.geometry[p].x;const auto u=Model::Jet(x,value,d,dd);const auto o=Model::Omega(x,u);
  auto parts=h::ConformalRHS(u,o,4.,-.99);Require(h::HyperbolicDamping(x,u,o,parts),"invalid auxiliary fixture source");
  h::Z4cRHS<Real> geometry{};Require(h::AssembleInterior(parts,o.omega,geometry),"invalid auxiliary fixture interior");
  const auto D=Model::Vector(25,value,d);std::array<Real,3> result{};
  Require(h::HyperbolicAuxiliaryRate(u,D,geometry.lambda,result.data()),"invalid auxiliary fixture rate");
  for(int a=0;a<3;++a)result[a]+=model.TransportCorrection(padded,25+a,p,u.beta.value)+diss[31*p+25+a]+diss[31*p+15+a];
  return result;
}
static void SampledChecks(){
  using Normal=Model::NormalDerivative;
  Model sampled(Mesh(),false,true,2,Normal::sampled_numerator),ad(Mesh(),false,true,2);
  Seed(sampled,false);V mink_edge;const auto mink_rhs=sampled.EvaluateRHS(sampled.state,dt,mink_edge);
  const Real mink_bulk=Difference(mink_rhs,V(mink_rhs.size(),0)),mink_surface=Difference(mink_edge,V(mink_edge.size(),0));
  Require(mink_bulk<2e-10&&mink_surface<2e-10,"sampled normal failed Minkowski identity");
  std::cout<<"minkowski_bulk="<<mink_bulk<<" minkowski_surface="<<mink_surface<<'\n';
  Seed(sampled,true);Seed(ad,true);const auto original=sampled.scri.values,values=sampled.state;
  auto padded=sampled.Padded(values);const auto normal=Normals(sampled,padded);
  // Native equation evaluation must use the passed RK-stage array, not the
  // model's unrelated persistent accepted-state storage.
  for(auto&x:sampled.state)x+=.01;
  Require(Bits(normal,Normals(sampled,padded)),"sampled normal read stale model.state");sampled.state=values;
  for(size_t p=0;p<sampled.grid.active.size();++p)for(int a=0;a<3;++a)sampled.state[31*p+25+a]+=.1*(a+1);
  for(size_t p=0;p<sampled.scri.rows.size();++p)for(int a=0;a<3;++a)sampled.scri.values[31*p+25+a]-=.2*(a+1);
  const auto d_padded=sampled.Padded(sampled.state);
  Require(Bits(normal,Normals(sampled,d_padded)),"D-only values contaminated geometric numerator");
  for(int f=0;f<31;++f)if(f<=18||(f>=22&&f<25))for(int p=0;p<sampled.grid.count;++p)
    Require(padded[f*sampled.grid.count+p]==d_padded[f*sampled.grid.count+p],"D-only perturbation changed geometric ghosts");
  sampled.state=values;sampled.scri.values=original;
  const auto ad_rate=ad.BoundaryRates(ad.ActivePadded(values),dt),ad_surface=ad.scri.values;
  const auto sampled_rate=sampled.BoundaryRates(padded,dt),final_surface=sampled.scri.values;
  Real replacement_error=0;
  for(size_t p=0;p<sampled.scri.rows.size();++p){const auto*x=sampled.scri.rows[p].geometry.x;
    const auto sample=sampled.scri.Sample(p,padded,original,31);const auto u=Model::Jet(x,sample.value,sample.d,sample.dd);
    const auto o=Model::Omega(x,u,true);auto parts=h::ConformalRHS(u,o,4.,-.99);
    Require(h::HyperbolicDamping(x,u,o,parts),"invalid endpoint fixture");
    const auto supplied=sampled.SampledPoleNormalDerivative(p,padded,&parts.pole);Real supplied_fields[31]{};
    Model::StoreGeometry(supplied,supplied_fields);
    const auto pole=h::TensorPoleNormalJet(u,o,x,x,4.,-.99,1.,1.,true,true);Real old_normal_rate[31]{};
    Model::StoreGeometry(h::TensorScriLimit(h::Z4cRHS<Real>{},pole,-1.),old_normal_rate);
    for(int f=0;f<31;++f){Require(supplied_fields[f]==normal[31*p+f],"provided endpoint changed sampled numerator");
      replacement_error=std::max(replacement_error,std::abs(sampled_rate[31*p+f]-ad_rate[31*p+f]+normal[31*p+f]+old_normal_rate[f]));
      if(f<25||f>27)Require(final_surface[31*p+f]==original[31*p+f],"closure reset independent surface field");}
  }
  Require(replacement_error<3e-12,"sampled mode changed more than the normal derivative");
  const Real dependent_error=NullDerivativeResponse(sampled,original,ad_rate,sampled_rate,ad_surface,final_surface);
  Require(dependent_error<3e-12,"sampled normal omitted dependent-D response");
  Require(sampled.compatibility_max<3e-12&&sampled.null_max<3e-14,"sampled closure lost null compatibility");
  auto final_padded=padded;sampled.FillGhosts(final_padded);
  size_t selected=0;Real stale_effect=0;V diss(values.size(),0);sampled.Dissipate(values,diss);
  std::array<Real,3> predicted{};
  // The last active radial layer uses outer ghost D; select a row where this
  // fixture distinguishes the provisional and final ghost arrays.
  for(const auto&row:sampled.scri.rows){const int n=sampled.grid.n;
    const size_t p=size_t(row.block)*n*n*n+(row.k*n+row.j)*n+n-1;
    const auto old=AuxiliaryAt(sampled,values,padded,diss,p),fresh=AuxiliaryAt(sampled,values,final_padded,diss,p);
    for(int a=0;a<3;++a)if(std::abs(old[a]-fresh[a])>stale_effect){stale_effect=std::abs(old[a]-fresh[a]);selected=p;predicted=fresh;}
  }
  Require(stale_effect>1e-7,"fixture cannot detect missing final dependent-D ghost refill");
  sampled.scri.values=original;V actual_edge;const auto actual_rhs=sampled.EvaluateRHS(values,dt,actual_edge);
  Require(Bits(actual_edge,sampled_rate)&&Bits(sampled.scri.values,final_surface),"automatic provisional ordering differs from manual closure");
  Real final_ghost_error=0;for(int a=0;a<3;++a)final_ghost_error=std::max(final_ghost_error,std::abs(actual_rhs[31*selected+25+a]-predicted[a]));
  Require(final_ghost_error<3e-12,"bulk auxiliary equation used stale dependent-D ghosts");
  Require(Bits(sampled.state,values),"sampled RHS changed active state");
  // Independently evolved Lambda and Theta remain distinct input data. Zeroing
  // either must change the geometric numerator; neither may be reset to a trace.
  Real independent_response[2]{};
  for(int which=0;which<2;++which){V changed=values;sampled.scri.values=original;
    for(size_t p=0;p<changed.size()/31;++p)for(int f=(which?14:15);f<=(which?14:17);++f)changed[31*p+f]=0;
    for(size_t p=0;p<sampled.scri.values.size()/31;++p)for(int f=(which?14:15);f<=(which?14:17);++f)sampled.scri.values[31*p+f]=0;
    independent_response[which]=Difference(normal,Normals(sampled,sampled.Padded(changed)));
    Require(independent_response[which]>1e-6,"fixture did not exercise independent Lambda/Theta");
  }
  bool rejected=false;try{sampled.SampledPoleNormalDerivative(0,V(1));}catch(const std::invalid_argument&){rejected=true;}
  Require(rejected,"sampled helper accepted wrong padded shape");
  sampled.scri.radial_nodes=7;rejected=false;
  try{sampled.SampledPoleNormalDerivative(0,padded);}catch(const std::invalid_argument&){rejected=true;}
  sampled.scri.radial_nodes=6;Require(rejected,"sampled helper silently changed radial layout");
  rejected=false;try{Model invalid(Mesh(),false,true,0,static_cast<Normal>(-1));}catch(const std::invalid_argument&){rejected=true;}
  Require(rejected,"invalid normal selector accepted");
  std::cout<<"D_only_bitwise=1 current_stage_only=1 independent_state_preserved=1 replacement_error="<<replacement_error
           <<" dependent_D_error="<<dependent_error<<" final_D_ghost_error="<<final_ghost_error
           <<" stale_D_counterfactual="<<stale_effect<<" Lambda_response="<<independent_response[0]
           <<" Theta_response="<<independent_response[1]<<" null="<<sampled.null_max
           <<" compatibility="<<sampled.compatibility_max<<" validation_checks=1\n";
}
#endif
int main(int argc,char**argv){
  Kokkos::initialize(argc,argv);int status=0;
  try{std::cout<<std::setprecision(17)<<"backend="<<Kokkos::DefaultExecutionSpace::name()<<'\n';
    const char*filename=nullptr;if(argc==3&&std::string(argv[1])=="--dump-default")filename=argv[2];
    else Require(argc==1,"usage: scri_sampled_normal [--dump-default PATH]");
    DefaultChecks(filename);
#ifndef HYPERBOLIC_SAMPLED_DEFAULT_ONLY
    SampledChecks();
#endif
    std::cout<<"CPU sampled-normal closure checks passed\n";
  }catch(const std::exception&e){std::cerr<<e.what()<<'\n';status=1;}
  Kokkos::finalize();return status;
}
