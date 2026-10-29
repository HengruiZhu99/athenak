// Optional sampled-normal closure: component/ownership tests, no evolution.
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include "z4c/hyperboloidal/hyperbolic_sampled_normal_device.hpp"
#include "z4c/hyperboloidal/hyperbolic_composite_exchange.hpp"
#include "z4c/hyperboloidal/hyperbolic_scri_device.hpp"
#include "z4c/hyperboloidal/algebraic_projection.hpp"
#include "frozen_d98_scri_device.hpp"

#ifndef HYPERBOLIC_TEST_DONOR_SEVEN
#define HYPERBOLIC_TEST_DONOR_SEVEN 0
#endif
#ifndef HYPERBOLIC_TEST_INNER_TRANSITION
#define HYPERBOLIC_TEST_INNER_TRANSITION 0
#endif
#ifndef HYPERBOLIC_TEST_REFINED_CORE
#define HYPERBOLIC_TEST_REFINED_CORE 0
#endif
#ifndef HYPERBOLIC_TEST_CORE2
#define HYPERBOLIC_TEST_CORE2 0
#endif
#ifndef HYPERBOLIC_TEST_OVERLAP
#define HYPERBOLIC_TEST_OVERLAP 0
#endif
constexpr bool inner_transition=HYPERBOLIC_TEST_INNER_TRANSITION;
constexpr bool donor_seven=HYPERBOLIC_TEST_DONOR_SEVEN;
namespace h=z4c::hyperboloidal;
using Model=h::HyperbolicComposite;
using Device=h::HyperbolicScriDevice;
using Sampled=h::HyperbolicSampledNormalDevice;
using Exchange=h::HyperbolicCompositeExchange;
using V=std::vector<Real>;
static void Require(bool value,const char*message){if(!value)throw std::runtime_error(message);}
static Device::Buffer Upload(const V&values){
  Device::Buffer result("sampled test upload",values.size());auto host=Kokkos::create_mirror_view(result);
  for(size_t i=0;i<values.size();++i)host(i)=values[i];Kokkos::deep_copy(result,host);return result;
}
static V Download(Device::ConstBuffer values){
  auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),values);V result(values.extent(0));
  for(size_t i=0;i<result.size();++i)result[i]=host(i);return result;
}
static bool Bits(const V&a,const V&b){
  return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(Real))==0);
}
static Real Compare(const V&expected,const V&actual,size_t offset=0){
  Require(offset+actual.size()<=expected.size(),"comparison shape mismatch");Real error=0;
  for(size_t i=0;i<actual.size();++i){Require(std::isfinite(actual[i])&&std::isfinite(expected[offset+i]),"nonfinite comparison");
    error=std::max(error,std::abs(expected[offset+i]-actual[i])/(1+std::abs(expected[offset+i])));}return error;
}
static void Seed(Model&model,bool perturb){
  auto fill=[&](const Real*x,Real*v){
    for(int f=0;f<31;++f)v[f]=(f==0?1.:0.)+(perturb?Real(.0001)*(1+f%4)*
      std::sin((2.1+.23*f)*x[0]+(.7+.11*f)*x[1]-(1.7+.17*f)*x[2]):0.);
    Require(h::ProjectAlgebraicNativeState(v).valid,"invalid seeded algebraic geometry");};
  for(size_t p=0;p<model.grid.active.size();++p)fill(model.grid.geometry[p].x,&model.state[31*p]);
  for(size_t p=0;p<model.scri.rows.size();++p)fill(model.scri.rows[p].geometry.x,&model.scri.values[31*p]);
  model.BoundaryValues();
}
// Reuse the native CPU derivative/jet/source helpers as an independent execution
// oracle. There is no alternate equation implementation in this fixture.
static V CPUPoles(const Model&model,const V&padded){
  const int donors=model.scri.radial_nodes-1;
  V result(model.scri.rows.size()*donors*31,0),value(31);
  std::vector<std::array<Real,3>> d(31);
  std::vector<std::array<std::array<Real,3>,3>> dd(31);
  const int n=model.grid.n;
  for(size_t row=0;row<model.scri.rows.size();++row)for(int s=0;s<donors;++s){
    const auto&r=model.scri.rows[row];const size_t p=size_t(r.block)*n*n*n+(r.k*n+r.j)*n+n-donors+s;
    const auto*x=model.grid.geometry[p].x;
    for(int f=0;f<31;++f){value[f]=padded[f*model.grid.count+model.grid.active[p]];Real df[3],ddf[3][3];
      model.BulkDerivatives(padded,f,p,df,ddf,f<=6||f==18||(f>=22&&f<25));
      for(int a=0;a<3;++a){d[f][a]=df[a];for(int b=0;b<3;++b)dd[f][a][b]=ddf[a][b];}}
    const auto u=Model::Jet(x,value,d,dd);const auto omega=Model::Omega(x,u,false);
    auto parts=h::ConformalRHS(u,omega,4.,-.99);
    Require(h::HyperbolicDamping(x,u,omega,parts),"invalid CPU numerator");
    Model::StoreGeometry(parts.pole,&result[31*(row*donors+s)]);
  }
  return result;
}
template<class F>static void Reject(F operation,const char*message){
  bool rejected=false;try{operation();}catch(const std::invalid_argument&){rejected=true;}Require(rejected,message);
}
static void Check(bool perturb,int refinement=0,int core_subdivisions=1){
  auto mesh=cubed_sphere::CompositeMesh(.125,{.45,.75,1},1,HYPERBOLIC_TEST_OVERLAP?12:6,3,core_subdivisions);
  mesh.evolved_overlap=HYPERBOLIC_TEST_OVERLAP;
  if(refinement>0)mesh.RefineCartesian(0);
  // Refine one child again to exercise a genuine coarse/fine Cartesian seam.
  // The shell geometry, cell count and outer-boundary operator stay fixed.
  if(refinement>1)mesh.RefineCartesian(18);
  int first_core=-1,first_outer=-1,end_outer=-1;
  for(size_t b=0;b<mesh.blocks.size();++b){
    if(mesh.blocks[b].face<0&&first_core<0)first_core=b;
    if(mesh.blocks[b].layer==2){if(first_outer<0)first_outer=b;end_outer=b+1;}
  }
  Require(first_core>=0&&end_outer-first_outer==6,"invalid test topology");
  Model model(mesh,false,true,2,Model::NormalDerivative::sampled_numerator,
    donor_seven?Model::RadialClosure::donor_seven:Model::RadialClosure::ghost_lift);
  if(inner_transition)model.grid.SetInnerTransitionLogical(true);
  const int donors=model.scri.radial_nodes-1;
  Seed(model,perturb);const int total=model.scri.rows.size();const Real dt=.0001;
  const auto projected=model.scri.values,padded=model.Padded(model.state);
  const auto cpu_poles=CPUPoles(model,padded),q=model.angular_plan->Apply(projected,2);
  const auto cpu_rate=model.BoundaryRates(padded,dt),cpu_trace=model.scri.values;
  model.scri.values=projected;
  const auto active=Upload(model.state),surface=Upload(projected),rates=Upload(q);
  Sampled samples(model,0,total);Device full(model,0,total);
  Require(samples.FirstBlock()==first_outer&&samples.EndBlock()==end_outer,"wrong surface-owned donor blocks");
  Exchange exchange(model,samples.FirstBlock(),samples.EndBlock());
  // The independent volume owner may cover only the Cartesian core. Do not use
  // its unrelated halos: sampled donors require the dedicated outer blocks.
  Exchange core_owner(model,first_core,first_core+1);core_owner.SetSurface(projected);
  Require(core_owner.Fill(active).invalid==0,"core-owner exchange failed");
  exchange.SetSurface(projected);const auto halo=exchange.Fill(active);
  Require(halo.invalid==0,"surface-owner exchange failed");
  Require(samples.Evaluate(exchange.padded,surface).bad_rows==0,"native device samples failed");
  const auto full_poles=Download(samples.LocalPoleSamples());
  const Real pole_error=Compare(cpu_poles,full_poles);
  const auto summary=full.Evaluate(active,surface,dt,rates,samples.LocalPoleSamples());
  Require(summary.bad_rows==0,"native sampled scri closure failed");
  const auto actual=Download(full.LocalRHS()),trace=Download(full.LocalSurface());
  const Real rate_error=Compare(cpu_rate,actual),trace_error=Compare(cpu_trace,trace);
  Require(pole_error<2e-10&&rate_error<2e-10&&trace_error<2e-10,"CPU/device sampled-normal parity failed");
  Real partition_error=0;
  auto partition=[&](int first,int count){
    Sampled local(model,first,count);Device boundary(model,first,count);
    Exchange halos(model,local.FirstBlock(),local.EndBlock());halos.SetSurface(projected);
    // Poison unfilled halos so accidental dependence on nonowned blocks cannot
    // be hidden by zero-initialized allocation. Fill writes all active values.
    Kokkos::deep_copy(halos.padded,std::numeric_limits<Real>::quiet_NaN());
    Require(halos.Fill(active).invalid==0&&local.Evaluate(halos.padded,surface).bad_rows==0,"partition samples failed");
    const V local_q(q.begin()+31*first,q.begin()+31*(first+count));
    Require(boundary.Evaluate(active,surface,dt,Upload(local_q),local.LocalPoleSamples()).bad_rows==0,"partition closure failed");
    partition_error=std::max({partition_error,Compare(full_poles,Download(local.LocalPoleSamples()),size_t(first)*donors*31),
      Compare(actual,Download(boundary.LocalRHS()),31*first),Compare(trace,Download(boundary.LocalSurface()),31*first)});
  };
  for(int rank=0;rank<12;++rank)partition(total*rank/12,total*(rank+1)/12-total*rank/12);
  partition(29,18);  // Explicitly crosses two face blocks at n=6.
  partition(total,0);
  Require(partition_error<2e-13,"surface ownership changes sampled closure");
  // Both volume D and manually stored dependent surface D change, with every
  // geometric field held fixed. Native numerators must remain bitwise equal.
  auto changed=model.state,changed_surface=projected;
  for(size_t p=0;p<changed.size()/31;++p)for(int a=0;a<3;++a)changed[31*p+25+a]+=.17*std::sin(.13*p+a);
  for(size_t p=0;p<changed_surface.size()/31;++p)for(int a=0;a<3;++a)changed_surface[31*p+25+a]-=.29*std::cos(.19*p+a);
  exchange.SetSurface(changed_surface);Require(exchange.Fill(Upload(changed)).invalid==0,"D-only halo fill failed");
  Require(samples.Evaluate(exchange.padded,Upload(changed_surface)).bad_rows==0,"finite D-only sample change rejected");
  const auto changed_poles=Download(samples.LocalPoleSamples());const Real d_error=Compare(full_poles,changed_poles);
  Require(Bits(full_poles,changed_poles),"geometric numerator depends on finite D storage");
  Require(full.Evaluate(Upload(changed),Upload(changed_surface),dt,rates,samples.LocalPoleSamples()).bad_rows==0,
          "D-only closure failed");
  Require(Bits(actual,Download(full.LocalRHS()))&&Bits(trace,Download(full.LocalSurface())),"old D leaks into final closure");
  // Same-backend historical kernel comparison, including signed zero bits.
  Model ad(mesh,false,true,2);ad.state=model.state;ad.scri.values=projected;
  Device current(ad,0,total);h::FrozenScriDevice frozen(ad,0,total);
  Require(current.Evaluate(active,surface,dt,rates).bad_rows==0&&frozen.Evaluate(active,surface,dt,rates).bad_rows==0,
          "default reference evaluation failed");
  Require(Bits(Download(current.LocalRHS()),Download(frozen.LocalRHS()))&&
          Bits(Download(current.LocalSurface()),Download(frozen.LocalSurface())),"default kernel changed historical bits");
  Reject([&]{full.Evaluate(active,surface,dt,rates);},"sampled mode accepted missing samples");
  Reject([&]{current.Evaluate(active,surface,dt,rates,samples.LocalPoleSamples());},"default accepted sample input");
  Reject([&]{samples.Evaluate(Upload(V(3)));},"sampled kernel accepted wrong padded shape");
  Reject([&]{Sampled invalid(model,-1,2);},"sampled kernel accepted negative ownership");
  auto invalid=full_poles;invalid[31*donors*3+1]=1;
  auto bad=full.Evaluate(active,surface,dt,rates,Upload(invalid));
  Require(bad.bad_rows==1&&bad.first_bad==3&&bad.reason==Device::invalid_input,"forbidden numerator channel accepted");
  invalid=full_poles;invalid[31*donors*2]=std::numeric_limits<Real>::quiet_NaN();
  bad=full.Evaluate(active,surface,dt,rates,Upload(invalid));
  Require(bad.bad_rows==1&&bad.first_bad==2&&bad.reason==Device::invalid_input,"nonfinite numerator accepted");
  // Nonfinite D is rejected by native jet validation even though finite D has
  // no effect on N. This distinguishes invariance from skipped input checks.
  auto invalid_padded=padded;const auto&row=model.scri.rows[0];
  const int n=model.grid.n;
  const size_t donor=size_t(row.block)*n*n*n+(row.k*n+row.j)*n+n-donors+1;
  invalid_padded[25*model.grid.count+model.grid.active[donor]]=std::numeric_limits<Real>::quiet_NaN();
  const auto bad_sample=samples.Evaluate(Upload(invalid_padded),surface);
  Require(bad_sample.bad_rows>0&&bad_sample.first_bad==0&&bad_sample.reason==Sampled::invalid_input,
          "nonfinite native donor accepted");
  const Real null=std::max({halo.after,summary.null_after,model.null_max});
  Require(null<3e-14&&summary.compatibility<2e-12,"sampled null/dependent D closure inaccurate");
  if(donor_seven){
    model.scri.values=projected;V cpu_edge;
    const auto cpu_bulk=model.EvaluateRHS(model.state,dt,cpu_edge);
    h::HyperbolicCompositeDevice bulk(model.grid,0,model.grid.active.size(),model.radial_plan.get());
    const auto final_surface=Upload(model.scri.values);auto final_padding=model.Padded(model.state);
    if(inner_transition){
      Exchange all_halos(model,0,model.grid.mesh.blocks.size());all_halos.SetSurface(model.scri.values);
      Require(all_halos.Fill(active).invalid==0,"inner transition device exchange failed");
      const Real halo_error=Compare(final_padding,Download(all_halos.padded));
      Require(bulk.Evaluate(all_halos.padded,dt,false,.5,final_surface).bad_cells==0,"inner transition exchanged bulk failed");
      const Real exchanged_bulk_error=Compare(cpu_bulk,bulk.DownloadRHS());
      Require(halo_error<2e-10&&exchanged_bulk_error<2e-10,"inner transition device halo/bulk parity failed");
      std::cout<<"inner_transition_exchange_error="<<halo_error<<" bulk_error="<<exchanged_bulk_error<<'\n';
    }
    // Every exterior radial ghost is unused by the seven-node bulk closure,
    // including its native transport correction and sampled numerator jets.
    const auto&g=model.grid;
    for(int m=first_outer;m<end_outer;++m)for(int k=-g.ng;k<g.n+g.ng;++k)for(int j=-g.ng;j<g.n+g.ng;++j)
      for(int i=g.n;i<g.n+g.ng;++i)for(int f=0;f<31;++f)
        final_padding[f*g.count+g.Index(m,i,j,k)]=std::numeric_limits<Real>::quiet_NaN();
    Require(bulk.Evaluate(Upload(final_padding),dt,false,.5,final_surface).bad_cells==0,"direct radial bulk read exterior ghosts");
    const Real bulk_error=Compare(cpu_bulk,bulk.DownloadRHS());
    Require(bulk_error<2e-10,"direct radial complete bulk RHS differs from CPU");
    const auto constraints=bulk.Diagnostics(Upload(final_padding),0,final_surface);
    Require(constraints.invalid_cells==0&&std::isfinite(constraints.total.h2)&&std::isfinite(constraints.total.m2),
            "direct radial device constraints failed");
    std::cout<<"donor_seven_bulk_error="<<bulk_error<<" exterior_radial_poison=1\n";
  }
  std::cout<<"sampled_case="<<(perturb?"nonspherical":"minkowski")<<" n="<<mesh.cells<<" overlap="<<mesh.evolved_overlap<<" epsilon=2 cpu_pole_error="<<pole_error
    <<" cpu_rate_error="<<rate_error<<" cpu_trace_error="<<trace_error<<" partition_error="<<partition_error
    <<" finite_D_error="<<d_error<<" null="<<null<<" compatibility="<<summary.compatibility
    <<" default_bitwise=1 partitions=12 rejection_checks=1";
  if(donor_seven)std::cout<<" radial_nodes=7 radial_bulk_rows=3";
  if(inner_transition)std::cout<<" interpatch_interpolation=logical_inner_transition";
  if(refinement)std::cout<<" core_refinement="<<refinement<<" blocks="<<mesh.blocks.size();
  if(core_subdivisions==2)std::cout<<" cartesian_core_subdivisions=2 mesh_blocks=26";
  std::cout<<'\n';
}
int main(int argc,char**argv){
  Kokkos::initialize(argc,argv);int result=0;
  try{std::cout<<std::setprecision(17)<<"backend="<<Kokkos::DefaultExecutionSpace::name()<<'\n';
    if constexpr(HYPERBOLIC_TEST_CORE2){
      Check(false,0,2);Check(true,0,2);
    }else if constexpr(HYPERBOLIC_TEST_REFINED_CORE){
      Check(false,1);Check(true,1);Check(false,2);Check(true,2);
    }else{Check(false);Check(true);}
    std::cout<<(donor_seven?"Donor-seven device, ownership, default and closure parity passed\n":
      "Sampled-normal device, ownership, default and closure parity passed\n");
  }catch(const std::exception&e){std::cerr<<e.what()<<'\n';result=1;}
  Kokkos::finalize();return result;
}
