// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Independent audit of PUBLIC original-equation acceptance; no Stage edits.
#include <iomanip>
#include <iostream>
#include "z4c/hyperboloidal/composite_shift.hpp"
namespace h=z4c::hyperboloidal;
using V=multigrid::Vector;
struct Metrics{Real physical=0,conformal=0,balanced=0,maximum=0;bool finite=true,accepted=false;};
Real ConformalDot(const h::CompositeShiftStage &s,const V &a,const V &b){long double sum=0,volume=0;
 for(size_t p=0;p<s.grid.active.size();++p){volume+=s.conformal_volume[p];for(int i=0;i<3;++i)sum+=static_cast<long double>(s.conformal_volume[p])*a[3*p+i]*b[3*p+i];}
 return Real(sum/volume);}
Metrics Original(const h::CompositeShiftStage &s,const V &source,const h::CompositeShiftStage::Boundary &boundary,const V &beta,Real target,Real mix,Real guard){
 Metrics out;for(Real x:source)out.finite=out.finite&&std::isfinite(x);for(Real x:beta)out.finite=out.finite&&std::isfinite(x);
 if(!out.finite)return out;V a,r(source.size());s.Apply(beta,a,boundary);
 for(size_t i=0;i<r.size();++i){r[i]=-source[i]-a[i];out.finite=out.finite&&std::isfinite(r[i]);out.maximum=std::max(out.maximum,std::abs(r[i]));}
 if(!out.finite)return out;out.physical=std::sqrt(s.Dot(r,r));out.conformal=std::sqrt(ConformalDot(s,r,r));out.balanced=std::sqrt((1-mix)*out.physical*out.physical+mix*out.conformal*out.conformal);
 out.finite=std::isfinite(out.physical)&&std::isfinite(out.conformal)&&std::isfinite(out.maximum)&&std::isfinite(out.balanced);
 out.accepted=out.finite&&out.physical<=target&&(mix==0||out.conformal<=target)&&(guard==0||out.maximum<=guard);return out;}
Real Lower(const h::CompositeShiftStage &s,Real mix){Real result=1,vc=0;for(Real w:s.conformal_volume)vc+=w;
 if(mix>0)for(size_t p=0;p<s.norm_volume.size();++p){const Real ratio=std::exp(std::min(Real(0),std::log(s.conformal_volume[p])-std::log(vc)-s.log_norm_volume[p]+s.log_physical_volume));result=std::min(result,(1-mix)+mix*ratio);}return result;}
multigrid::KrylovReport Classic(const h::CompositeShiftStage &s,const V &source,const h::CompositeShiftStage::Boundary &boundary,V &beta,Real target,Real mix,int budget){
 V zero(source.size(),0),affine,rhs(source.size());s.Apply(zero,affine,boundary);for(size_t i=0;i<rhs.size();++i)rhs[i]=-source[i]-affine[i];
 return multigrid::FlexibleGMRES([&](const V &x,V &y){s.Apply(x,y);},[&](const V &x,V &y){y.resize(x.size());for(int v=0;v<3;++v){V scalar(s.grid.active.size());for(size_t p=0;p<scalar.size();++p)scalar[p]=x[3*p+v];const auto correction=s.preconditioner.Cycle(scalar);for(size_t p=0;p<scalar.size();++p)y[3*p+v]=correction[p];}},
 [&](const V &a,const V &b){return (1-mix)*s.Dot(a,b)+mix*ConformalDot(s,a,b);},rhs,beta,target*std::sqrt(Lower(s,mix)),0.,budget,24);}
void Require(bool value,const char *why){if(!value)throw std::runtime_error(why);}
void Print(const Metrics &m){std::cout<<"{\"physical\":"<<m.physical<<",\"conformal\":"<<m.conformal<<",\"balanced\":"<<m.balanced<<",\"maximum\":"<<m.maximum<<",\"finite\":"<<m.finite<<",\"accepted\":"<<m.accepted<<'}';}
std::array<Real,3> Polynomial(const std::array<Real,3> &x){return {.2+.1*x[0]+.03*x[1]*x[2],-.13+.07*x[1]*x[1],.1-.04*x[0]*x[2]};}
int main(int argc,char **argv){Kokkos::initialize(argc,argv);int status=0;try{
 std::cout<<std::setprecision(17);h::CompositeShiftStage s(cubed_sphere::CompositeMesh(.25,{.65,1},1,4,2,2),4,true,true);
 const size_t points=s.grid.active.size();std::vector<h::MetricJet<Real>> metric(points);for(auto &g:metric)for(int i=0;i<3;++i)g.g[i][i]=1;
 std::vector<Real> chi(points,1);size_t core=0;Real nearest=1;for(size_t p=0;p<points;++p){Real r2=0;for(Real x:s.grid.geometry[p].x)r2+=x*x;if(r2<nearest){nearest=r2;core=p;}}
 chi[core]=.1;s.SetGeometry(metric,chi);const Real target=1e-10,mix=.5,guard=1e-7;
 auto boundary_for=[](Real amplitude){return h::CompositeShiftStage::Boundary([=](const std::array<Real,3> &x){auto q=Polynomial(x);for(auto &v:q)v*=amplitude*.02;return q;});};
 auto forcing=[&](Real amplitude,const h::CompositeShiftStage::Boundary &boundary){V zero(3*points,0),affine,source(3*points);s.Apply(zero,affine,boundary);for(size_t p=0;p<points;++p)for(int v=0;v<3;++v)source[3*p+v]=-affine[3*p+v]-amplitude*(std::sin(.31*(p+1)*(v+1))+.2*std::cos(.17*p));return source;};
 auto bc=boundary_for(1);auto source=forcing(1,bc);V solution(3*points,0);Classic(s,source,bc,solution,1e-30,mix,1);const auto calibration=Original(s,source,bc,solution,1,mix,guard);const Real scale=.95*target/std::max(calibration.physical,calibration.conformal);
 h::ShiftSolveControl control;control.conformal_norm_fraction=mix;control.maximum_residual_tolerance=guard;
 std::cout<<"{\"cells\":"<<points<<",\"forced_budget\":[";
 for(int fixture=0;fixture<2;++fixture){const Real amplitude=scale*(fixture?1.1/.95:1);bc=boundary_for(amplitude);source=forcing(amplitude,bc);solution.assign(3*points,0);const auto classic=Classic(s,source,bc,solution,target,mix,1);const auto independent=Original(s,source,bc,solution,target,mix,guard);
  V actual(3*points,0);const auto report=s.Solve(source,bc,actual,target,0,1,control);const auto final=Original(s,source,bc,actual,target,mix,guard);
  Require(fixture?!independent.accepted:independent.accepted,"calibrated public original criterion differs from intended pass/fail");
  if(!fixture)Require(!classic.converged,"fixture does not exercise stricter inner norm false negative");
  Require(report.converged==final.accepted,"Stage convergence differs from independently recomputed original criteria");
  Require(fixture?!report.converged:report.converged,"Stage did not preserve public-tolerance budget acceptance/rejection");
  if(fixture)std::cout<<',';std::cout<<"{\"expected_accepted\":"<<(!fixture)<<",\"classic_inner_converged\":"<<classic.converged<<",\"stage_converged\":"<<report.converged<<",\"stage_inner_converged\":"<<report.inner_norm_converged<<",\"stage_external_acceptance\":"<<report.external_acceptance<<",\"stage_iterations\":"<<report.iterations<<",\"target\":"<<target<<",\"inner_target\":"<<target*std::sqrt(Lower(s,mix))<<",\"classic_original\":";Print(independent);std::cout<<",\"stage_original\":";Print(final);std::cout<<'}';}
 // Collapsed-chi near-node measure: balanced residual ALONE can miss either
 // physical or conformal failure. The candidate source includes actual affineBC.
 chi.assign(points,1);chi[core]=1e-80;s.SetGeometry(metric,chi);V candidate(3*points);for(size_t p=0;p<points;++p){const std::array<Real,3>x{s.grid.geometry[p].x[0],s.grid.geometry[p].x[1],s.grid.geometry[p].x[2]};const auto q=Polynomial(x);for(int i=0;i<3;++i)candidate[3*p+i]=q[i];}
 const h::CompositeShiftStage::Boundary polynomial=Polynomial;V applied;s.Apply(candidate,applied,polynomial);
 Real vc=0;for(Real w:s.conformal_volume)vc+=w;const size_t outer=points-1;
 std::cout<<"],\"localized_counterexamples\":[";
 for(int kind=0;kind<3;++kind){source=applied;for(Real &v:source)v=-v;const Real selected_mix=kind==0?.9999:.5;const Real selected_target=kind==2?1e-5:target;Real selected_guard=guard;const size_t node=kind==0?core:outer;
  const Real amplitude=kind==0?1.1*target:(kind==1?1.1*target/std::sqrt(s.conformal_volume[node]/vc):2e-7);
  source[3*node]-=amplitude;
  const auto initial=Original(s,source,polynomial,candidate,selected_target,selected_mix,selected_guard);Require(!initial.accepted,"localized public-criterion counterexample was accepted");
  if(kind<2)Require(initial.balanced<target,"counterexample did not defeat naive balanced threshold");
  if(kind==0)Require(initial.physical>target,"near-node physical failure missing");if(kind==1)Require(initial.conformal>target,"outer conformal failure missing");if(kind==2)Require(initial.maximum>selected_guard,"point guard failure missing");
  control.conformal_norm_fraction=selected_mix;control.maximum_residual_tolerance=selected_guard;auto corrected=candidate;const auto report=s.Solve(source,polynomial,corrected,selected_target,0,1,control);const auto final=Original(s,source,polynomial,corrected,selected_target,selected_mix,selected_guard);
  Require(!(report.converged&&report.iterations==0),"bad initial original residual falsely accepted without iteration");Require(report.converged==final.accepted,"localized final acceptance bypassed original check");
  if(kind)std::cout<<',';std::cout<<"{\"kind\":\""<<(kind==0?"physical_near_node":kind==1?"conformal_outside_core":"point_maximum")<<"\",\"initial\":";Print(initial);std::cout<<",\"final\":";Print(final);std::cout<<",\"stage_iterations\":"<<report.iterations<<",\"stage_converged\":"<<report.converged<<'}';}
 // Affine-lift roundoff: source is chosen from A0(beta)+Ab(0), then compared
 // with fresh Ab(beta). No arithmetic-equivalence assumption is made.
 chi.assign(points,1);s.SetGeometry(metric,chi);bool found=false,zero_inner=false;Metrics round_original,round_final;Real inner_norm=0,round_target=0,chosen=0;bool round_stage=false;int round_iterations=-1;
 for(Real amplitude:{1.,1.13,3.7,1e4,1e8}){V beta(3*points);for(size_t p=0;p<points;++p){beta[3*p]=amplitude;beta[3*p+1]=amplitude*.5;beta[3*p+2]=-amplitude*.25;}
  const h::CompositeShiftStage::Boundary boundary=[=](const auto &){return std::array<Real,3>{amplitude,amplitude*.5,-amplitude*.25};};V zero(3*points,0),affine,homogeneous;s.Apply(zero,affine,boundary);s.Apply(beta,homogeneous);source.resize(beta.size());V inner_residual(beta.size());
  for(size_t i=0;i<beta.size();++i){source[i]=-(homogeneous[i]+affine[i]);inner_residual[i]=-source[i]-affine[i]-homogeneous[i];}
  inner_norm=std::sqrt(s.Dot(inner_residual,inner_residual));const auto before=Original(s,source,boundary,beta,1,.5,guard);
  if(!(before.physical>std::max(Real(0),inner_norm)*1.5))continue;
  found=true;zero_inner=inner_norm==0;round_target=(before.physical+inner_norm)/2;round_original=Original(s,source,boundary,beta,round_target,.5,guard);Require(!round_original.accepted&&inner_norm<round_target,"roundoff gap fixture invalid");
  control.conformal_norm_fraction=.5;control.maximum_residual_tolerance=guard;const auto report=s.Solve(source,boundary,beta,round_target,0,1,control);round_final=Original(s,source,boundary,beta,round_target,.5,guard);round_stage=report.converged;round_iterations=report.iterations;chosen=amplitude;
  Require(report.converged==round_final.accepted,"affine roundoff caused false original-equation acceptance");if(zero_inner)Require(!report.converged&&report.iterations==0,"zero inner residual/external-false guard missing");break;
 }
 Require(found,"no reproducible affine roundoff residual gap found");
 // Explicitly reject every nonfinite residual; max(std::abs(NaN)) alone is unsafe.
 source.assign(3*points,0);source[0]=std::numeric_limits<Real>::quiet_NaN();V zero(3*points,0);const auto nonfinite=Original(s,source,{},zero,target,.5,guard);Require(!nonfinite.finite&&!nonfinite.accepted,"nonfinite original residual accepted");
 bool threw=false;try{s.Solve(source,{},zero,target,0,1,control);}catch(const std::exception&){threw=true;}Require(threw,"Stage failed to reject nonfinite source/residual");
 // A zero inner residual with failed external acceptance must return safely,
 // without normalizing an Arnoldi vector by zero or declaring success.
 V identity_zero(2,0),identity_x(2,0);int predicate_calls=0;
 const auto identity=[](const V &x,V &y){y=x;};
 const auto identity_dot=[](const V &x,const V &y){return x[0]*y[0]+x[1]*y[1];};
 const auto refused=multigrid::FlexibleGMRES(identity,identity,identity_dot,identity_zero,identity_x,1e-12,0,3,2,[&](const V &,const V &){++predicate_calls;return false;});
 Require(!refused.converged&&refused.inner_norm_converged&&!refused.external_acceptance&&refused.iterations==0&&predicate_calls==1,"zero-inner/external-false FGMRES contract failed");
 std::cout<<"],\"affine_roundoff\":{\"amplitude\":"<<chosen<<",\"inner_norm\":"<<inner_norm<<",\"inner_zero\":"<<zero_inner<<",\"target\":"<<round_target<<",\"initial_original\":";Print(round_original);std::cout<<",\"final_original\":";Print(round_final);std::cout<<",\"stage_converged\":"<<round_stage<<",\"stage_iterations\":"<<round_iterations<<"},\"zero_inner_external_false_rejected\":true,\"nonfinite_rejected\":true,\"scope\":\"Independent algebraic acceptance audit only; source and affine boundary reapplied in every original check; no gauge/evolution claim\"}\n";
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';status=1;}Kokkos::finalize();return status;}
