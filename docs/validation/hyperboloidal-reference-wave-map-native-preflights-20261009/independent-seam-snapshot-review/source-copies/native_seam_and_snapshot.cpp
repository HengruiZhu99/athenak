// Private read-only actual-array native seam and snapshot probe. No time stepping.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "z4c/hyperboloidal/cartesian_patch.hpp"
#include "native_wave_map.hpp"
namespace hyp=z4c::hyperboloidal;
using z4c::Z4c;
constexpr int ti[6]={0,0,0,1,1,2},tj[6]={0,1,2,1,2,2};
constexpr int selected[6][3]={{14,14,14},{18,15,14},{20,17,14},
                             {22,18,16},{24,17,16},{24,19,16}};
hyp::SphericalGhostGrid Grid(int n) {
  if(n!=16&&n!=24&&n!=32)throw std::invalid_argument("undeclared grid");
  hyp::SphericalGhostGrid g{};g.radius=1;
  for(int d=0;d<3;++d){g.n[d]=n+6;g.h[d]=2.2/n;g.first[d]=-1.1+(0.5-3)*g.h[d];}
  return g;
}
hyp::LayerGaugeParameters Parameters(){hyp::LayerGaugeParameters p;
  p.physical_trace_lapse=true;p.preferred_source=false;p.scri_lapse_damping=2;return p;}
double scaled(double x,double y){
  if(!std::isfinite(x)||!std::isfinite(y))throw std::runtime_error("nonfinite comparison");
  return std::abs(x-y)/std::max({1.,std::abs(x),std::abs(y)});
}
void pack(const hyp::Z4cRHS<Real>&r,const hyp::GaugeRHS<Real>&g,double out[22]){
  out[0]=r.chi;out[7]=r.trace;out[17]=r.theta;out[18]=g.alpha;
  for(int s=0;s<6;++s){out[1+s]=r.metric[ti[s]][tj[s]];out[8+s]=r.a[ti[s]][tj[s]];}
  for(int d=0;d<3;++d){out[14+d]=r.lambda[d];out[19+d]=g.beta[d];}
}
// This exact source-pinned formula constructs a standalone t=0 test array.
// It is not a call to Z4c::InitializeHyperboloidal. The later actual native
// t=0 restart is independently checked against the same production formula.
void Pulse(hyp::CartesianConformalPatch&p,DvceArray5D<Real>q,double a,double b){
  auto g=p.grid;auto nodes=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),p.active);
  for(size_t v=0;v<nodes.extent(0);++v){int s=nodes(v),i=s%g.n[0],j=s/g.n[0]%g.n[1],k=s/(g.n[0]*g.n[1]);
    double x=g.first[0]+i*g.h[0],y=g.first[1]+j*g.h[1],z=g.first[2]+k*g.h[2];
    double r2=x*x+y*y+z*z,shape=std::pow(1-r2,4)*std::exp(-r2/(.35*.35));
    q(0,Z4c::I_Z4C_ALPHA,k,j,i)+=a*shape*(1+.2*x+.3*y*z);
    const double v3[3]={1+.3*y*z,.2*x,.1*x*y};
    for(int d=0;d<3;++d)q(0,Z4c::I_Z4C_BETAX+d,k,j,i)+=b*shape*v3[d];
  }
}
int Seam(){
  auto g=Grid(24);hyp::CartesianConformalPatch p(g,.5,2,{true,.05,.95},Parameters(),true);
  p.kappa1=10;p.dissipation=.1;
  auto q=p.Allocate("seam t0 arrays"),out=p.Allocate("seam actual RHS");
  double maxerr=0,refmax=0,missingbeta=0,duplicatealpha=0;
  std::cout<<std::setprecision(17)<<"{\"grid_N\":24,\"span\":2.2,\"time\":0,\"rows\":[";
  bool comma=false;
  for(int state=0;state<3;++state){double a=state==0?0:state==1?.02:.2,b=state==0?0:state==1?.01:.1;
    p.InitializeReference(q);Pulse(p,q,a,b);p.RHS(q,out);
    auto dev=hyp::BindCartesianFields(p.deviations),full=hyp::BindCartesianFields(q);
    const double idx[3]={1/g.h[0],1/g.h[1],1/g.h[2]},spacing[3]={g.h[0],g.h[1],g.h[2]};
    const int stride[3]={1,g.n[0],g.n[0]*g.n[1]};
    for(auto&cell:selected){int i=cell[0],j=cell[1],k=cell[2],s=g.Index(i,j,k);
      if(!g.Interior(i,j,k))throw std::runtime_error("selected cell outside active sphere");
      const Real xyz[3]={g.first[0]+i*g.h[0],g.first[1]+j*g.h[1],g.first[2]+k*g.h[2]};
      auto pt=p.reference.At(xyz[0],xyz[1],xyz[2]);
      auto u=hyp::LoadMeshJet<3>(dev,idx,0,k,j,i);hyp::AddReferenceJet(u,pt,p.reference);
      hyp::Z4cJet<Real>background{};hyp::AddReferenceJet(background,pt,p.reference);
      auto conn=rwm::ReferenceConnection(pt,xyz);auto parts=rwm::Gauge(pt,u,conn);
      hyp::GaugeRHS<Real>gr{};hyp::Z4cRHS<Real>r{},r0{};
      if(!parts.valid||!hyp::AssembleInterior(hyp::ConformalRHS(u,hyp::CartesianOmega(u,pt),10/u.alpha.value,Real(0)),pt.omega,r)
          ||!hyp::AssembleInterior(hyp::ConformalRHS(background,hyp::CartesianOmega(background,pt),Real(10),Real(0)),pt.omega,r0))
        throw std::runtime_error("invalid manual centered seam");
      // Independent explicit assembly, including beta. Do not call the tested
      // ResearchNativeWaveMapGauge/rwm::Assemble in this comparator.
      gr.alpha=parts.regular.alpha+parts.pole.alpha/pt.omega;
      for(int d=0;d<3;++d)gr.beta[d]=parts.regular.beta[d]+parts.pole.beta[d]/pt.omega;
      r.chi-=r0.chi;r.trace-=r0.trace;r.theta-=r0.theta;
      for(int d=0;d<3;++d){r.lambda[d]-=r0.lambda[d];for(int e=0;e<3;++e){r.metric[d][e]-=r0.metric[d][e];r.a[d][e]-=r0.a[d][e];}}
      hyp::AddMeshUpwindAdvectionWithVelocity<3>(dev,full.beta_u,idx,0,k,j,i,r,gr);
      double expected[22];pack(r,gr,expected);
      double rowerr=0;
      for(int f=0;f<22;++f){expected[f]+=.1*hyp::InteriorKOSixth(hyp::CartesianComponent{p.deviations,f,g.n[0],g.n[1]},p.mask,s,stride,spacing);
        rowerr=std::max(rowerr,scaled(expected[f],out(0,f,k,j,i)));
        if(state==0)refmax=std::max(refmax,std::abs(out(0,f,k,j,i)));
      }
      maxerr=std::max(maxerr,rowerr);
      if(state){duplicatealpha=std::max(duplicatealpha,std::abs(parts.pole.alpha/pt.omega));
        for(int d=0;d<3;++d)missingbeta=std::max(missingbeta,std::abs(parts.pole.beta[d]/pt.omega));}
      if(comma)std::cout<<',';comma=true;
      std::cout<<"{\"state\":"<<state<<",\"ijk\":["<<i<<','<<j<<','<<k<<"],\"xyz\":["<<xyz[0]<<','<<xyz[1]<<','<<xyz[2]<<"],\"Omega\":"<<pt.omega
        <<",\"alpha\":"<<u.alpha.value<<",\"beta\":["<<u.beta.value[0]<<','<<u.beta.value[1]<<','<<u.beta.value[2]<<"],\"gauge_regular\":["<<parts.regular.alpha;
      for(int d=0;d<3;++d)std::cout<<','<<parts.regular.beta[d];
      std::cout<<"],\"gauge_pole\":["<<parts.pole.alpha;for(int d=0;d<3;++d)std::cout<<','<<parts.pole.beta[d];
      std::cout<<"],\"manual_full22\":[";for(int f=0;f<22;++f)std::cout<<(f?",":"")<<expected[f];
      std::cout<<"],\"native_full22\":[";for(int f=0;f<22;++f)std::cout<<(f?",":"")<<out(0,f,k,j,i);
      std::cout<<"],\"scaled_error\":"<<rowerr<<'}';
    }
  }
  std::cout<<"],\"max_scaled_error\":"<<maxerr<<",\"reference_rhs_abs\":"<<refmax
    <<",\"omit_beta_negative_control_abs\":"<<missingbeta<<",\"duplicate_alpha_negative_control_abs\":"<<duplicatealpha<<"}\n";
  if(!(maxerr<=2e-12&&refmax<=1e-10&&missingbeta>1e-8&&duplicatealpha>1e-8))throw std::runtime_error("native seam threshold failure");
  return 0;
}
int Snapshot(int n){
  auto g=Grid(n);hyp::CartesianConformalPatch p(g,.5,2,{true,.05,.95},Parameters(),true);
  p.kappa1=10;p.dissipation=.1;auto q=p.Allocate("saved native binary64 array");
  if(!std::cin.read(reinterpret_cast<char*>(q.data()),q.size()*sizeof(Real)))throw std::runtime_error("truncated native array");
  if(std::cin.peek()!=std::char_traits<char>::eof())throw std::runtime_error("extra native input bytes");
  p.Prepare(q);auto dev=hyp::BindCartesianFields(p.deviations);
  auto nodes=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),p.active);
  const Real idx[3]={1/g.h[0],1/g.h[1],1/g.h[2]};
  double sums[4]{},shells[4]{},normmax[4]{},gaugepole[4]{},devmax[25]{},devsum[25]{};
  double max_r[4]{},max_xyz[4][3]{},profile_error[3]{},bins[7][4]{};
  const double edges[8]={0,.25,.5,.75,.85,.9,.95,1};size_t bin_count[7]{};
  double maxdet=0,maxtrace=0,nullmax=0,amin=1e300,chimin=1e300;
  size_t shellcount=0;
  for(size_t v=0;v<nodes.extent(0);++v){int s=nodes(v),i=s%g.n[0],j=s/g.n[0]%g.n[1],k=s/(g.n[0]*g.n[1]);
    const Real xyz[3]={g.first[0]+i*g.h[0],g.first[1]+j*g.h[1],g.first[2]+k*g.h[2]};
    auto pt=p.reference.At(xyz[0],xyz[1],xyz[2]);auto u=hyp::LoadMeshJet<3>(dev,idx,0,k,j,i);hyp::AddReferenceJet(u,pt,p.reference);
    auto c=hyp::EvolvedConstraints(u,hyp::CartesianOmega(u,pt));
    if(!c.valid||!c.z4.valid||!(u.alpha.value>0)||!(u.chi.value>0))throw std::runtime_error("invalid saved active field/constraint");
    double squares[4]={c.hamiltonian*c.hamiltonian,c.momentum_conformal_norm2,c.z4.z_conformal_norm2,u.theta.value*u.theta.value};
    bool shell=pt.radius>=.9;if(shell)++shellcount;
    int bin=0;while(bin<6&&pt.radius>=edges[bin+1])++bin;++bin_count[bin];
    for(int f=0;f<4;++f){if(!std::isfinite(squares[f])||squares[f]<0)throw std::runtime_error("invalid constraint square");
      sums[f]+=squares[f];bins[bin][f]+=squares[f];if(shell)shells[f]+=squares[f];
      if(std::sqrt(squares[f])>normmax[f]){normmax[f]=std::sqrt(squares[f]);max_r[f]=pt.radius;for(int d=0;d<3;++d)max_xyz[f][d]=xyz[d];}}
    maxdet=std::max(maxdet,std::abs(c.z4.determinant_residual));maxtrace=std::max(maxtrace,std::abs(c.z4.tracefree_residual));
    nullmax=std::max(nullmax,std::abs(c.null_residual-pt.omega*pt.omega*pt.n_residue));
    amin=std::min(amin,u.alpha.value);chimin=std::min(chimin,u.chi.value);
    // Gauge pole numerators depend only on live values/reference. This is
    // reported as values-only; no full-source or derivative identity inferred.
    auto gp=rwm::Gauge(pt,u,rwm::ReferenceConnection(pt,xyz));if(!gp.valid)throw std::runtime_error("invalid gauge pole readback");
    if(shell){gaugepole[0]=std::max(gaugepole[0],std::abs(gp.pole.alpha));
      for(int d=0;d<3;++d)gaugepole[1+d]=std::max(gaugepole[1+d],std::abs(gp.pole.beta[d]));}
    const double r2=xyz[0]*xyz[0]+xyz[1]*xyz[1]+xyz[2]*xyz[2];
    const double shape=std::pow(1-r2,4)*std::exp(-r2/(.35*.35));
    const double pulse_vector[3]={1+.3*xyz[1]*xyz[2],.2*xyz[0],.1*xyz[0]*xyz[1]};
    for(int f=0;f<25;++f){double value=q(0,f,k,j,i),reference=hyp::ReferenceComponent(f,pt),delta=value-reference;
      if(!std::isfinite(value)||!std::isfinite(delta))throw std::runtime_error("nonfinite saved active component");
      devmax[f]=std::max(devmax[f],std::abs(delta));devsum[f]+=delta*delta;
      for(int state=0;state<3;++state){double a=state==0?0:state==1?.02:.2,b=state==0?0:state==1?.01:.1,expected=reference;
        if(f==18)expected+=a*shape*(1+.2*xyz[0]+.3*xyz[1]*xyz[2]);
        if(f>=19&&f<=21)expected+=b*shape*pulse_vector[f-19];
        profile_error[state]=std::max(profile_error[state],std::abs(value-expected));}}
  }
  std::cout<<std::setprecision(17)<<"{\"active_count\":"<<nodes.extent(0)<<",\"shell_r_ge_09_count\":"<<shellcount<<",\"rms_H_Mcon_Zcon_Theta\":[";
  for(int f=0;f<4;++f)std::cout<<(f?",":"")<<std::sqrt(sums[f]/nodes.extent(0));
  std::cout<<"],\"shell_rms_H_Mcon_Zcon_Theta\":[";for(int f=0;f<4;++f)std::cout<<(f?",":"")<<(shellcount?std::sqrt(shells[f]/shellcount):0);
  std::cout<<"],\"max_H_Mcon_Zcon_Theta\":[";for(int f=0;f<4;++f)std::cout<<(f?",":"")<<normmax[f];
  std::cout<<"],\"constraint_max_radius4\":[";for(int f=0;f<4;++f)std::cout<<(f?",":"")<<max_r[f];
  std::cout<<"],\"constraint_max_xyz4x3\":[";for(int f=0;f<4;++f)std::cout<<(f?",":"")<<'['<<max_xyz[f][0]<<','<<max_xyz[f][1]<<','<<max_xyz[f][2]<<']';
  std::cout<<"],\"reference_deviation_max25\":[";for(int f=0;f<25;++f)std::cout<<(f?",":"")<<devmax[f];
  std::cout<<"],\"reference_deviation_rms25\":[";for(int f=0;f<25;++f)std::cout<<(f?",":"")<<std::sqrt(devsum[f]/nodes.extent(0));
  std::cout<<"],\"shell_values_only_wave_gauge_pole_max4\":[";for(int f=0;f<4;++f)std::cout<<(f?",":"")<<gaugepole[f];
  std::cout<<"],\"initial_profile_max_error_reference_small_large\":["<<profile_error[0]<<','<<profile_error[1]<<','<<profile_error[2]<<"],\"radial_bins\":[";
  for(int b=0;b<7;++b){std::cout<<(b?",":"")<<"{\"rlo\":"<<edges[b]<<",\"rhi\":"<<edges[b+1]<<",\"count\":"<<bin_count[b]<<",\"rms4\":[";
    for(int f=0;f<4;++f)std::cout<<(f?",":"")<<(bin_count[b]?std::sqrt(bins[b][f]/bin_count[b]):0);
    std::cout<<"],\"squared_fraction4\":[";for(int f=0;f<4;++f)std::cout<<(f?",":"")<<(sums[f]?bins[b][f]/sums[f]:0);std::cout<<"]}";}
  std::cout<<"],\"det_max\":"<<maxdet<<",\"trace_max\":"<<maxtrace<<",\"alpha_min\":"<<amin<<",\"chi_min\":"<<chimin<<",\"null_deviation_max\":"<<nullmax
    <<",\"retained_max_gauge_speed\":"<<p.MaxGaugeSpeed(q)<<",\"Omega_min\":"<<p.min_omega<<"}\n";
  return 0;
}
int main(int argc,char**argv){Kokkos::ScopeGuard guard(argc,argv);try{
  static_assert(sizeof(Real)==8,"binary64 only");
  if(argc==2&&std::string(argv[1])=="--seam")return Seam();
  if(argc==3&&std::string(argv[1])=="--snapshot")return Snapshot(std::stoi(argv[2]));
  throw std::invalid_argument("only --seam or --snapshot N is admitted");
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
