// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <iostream>
#include "z4c/hyperboloidal/composite_evolution.hpp"
#include "affine_mapped_operator.hpp"
#include "manufactured.hpp"
namespace h=z4c::hyperboloidal;
void Require(bool yes,const char*reason){if(!yes)throw std::runtime_error(reason);}
template<class F>void Reject(F f){bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}Require(rejected,"invalid affine state accepted");}
int main(int argc,char**argv){Kokkos::initialize(argc,argv);int code=0;try{
  auto mesh=cubed_sphere::CompositeMesh(.25,{.65,1},1,6,2,2);
  h::CompositeShiftStage stage(mesh,4,true,true,true,true);const auto&g=stage.grid;
  std::vector<h::MetricJet<Real>>metric;multigrid::Vector beta;
  for(const auto&p:g.geometry){const auto u=manufactured::State(p.x,true);metric.push_back(u.metric);for(int v=0;v<3;++v)beta.push_back(u.beta.value[v]);}
  stage.SetGeometry(metric);affine_mapped_probe::Operator prior(stage);
  const auto boundary=[](const std::array<Real,3>&x){const auto u=manufactured::State(x.data(),true);return std::array<Real,3>{u.beta.value[0],u.beta.value[1],u.beta.value[2]};};
  multigrid::Vector original,production;prior.Apply(beta,original,boundary);stage.Apply(beta,production,boundary);
  Require(original==production,"production affine arithmetic differs from qualified prototype");
  auto second=beta,sum=beta;for(size_t i=0;i<beta.size();++i){second[i]=.02*std::sin(.1*i);sum[i]=1.7*beta[i]-.3*second[i];}
  multigrid::Vector a,b,c;stage.Apply(beta,a);stage.Apply(second,b);stage.Apply(sum,c);Real linearity=0;
  for(size_t i=0;i<a.size();++i)linearity=std::max(linearity,std::abs(c[i]-(1.7*a[i]-.3*b[i])));
  Require(linearity<1e-8,"fixed-geometry affine operator not linear");
  const auto field=stage.affine_derivative->Fit(beta);auto bad=beta;bad[0]=std::numeric_limits<Real>::quiet_NaN();Reject([&]{stage.affine_derivative->Fit(bad);});bad.pop_back();Reject([&]{stage.affine_derivative->Fit(bad);});
  h::CartesianAffineDerivative foreign(g);Reject([&]{foreign.At(field,g.geometry[0].x);});
  ++stage.grid.mesh.revision;Reject([&]{stage.affine_derivative->Fit(beta);});--stage.grid.mesh.revision;
  h::CompositeTensorEvolution engine(mesh,true,1e-11,4,true,false,0,7,true,0,.24,false,true,true,false,.12,0,1,true);
  engine.Initialize(0,false);engine.RHS(engine.state,.0001);Real equilibrium=0;for(Real v:engine.boundary_rhs)equilibrium=std::max(equilibrium,std::abs(v));
  Require(equilibrium<1e-10,"affine boundary differentiation damages Minkowski equilibrium");
  engine.Initialize(.001,true);engine.Step(.0001);engine.RHS(engine.state,.0001);
  Require(engine.shift_affine&&engine.shift.affine_derivative,"affine engine lost stage representation");
  // At active cells the cached beta pad and the new fine operator have exactly
  // the same remainder. Surface values are restored to actual vector Dirichlet
  // data while their derivative directions use remainder+analytic affine jets.
  Real beta_consistency=0;for(size_t p=0;p<engine.jets.size();++p){const auto q=engine.shift.affine_derivative->Sample(engine.beta_delta,engine.beta_padded,*engine.shift_affine,p);
    for(int v=0;v<3;++v)for(int d=0;d<3;++d)beta_consistency=std::max(beta_consistency,std::abs(engine.jets[p].beta.d[d][v]-q.d[d][v]+(v==d?1.:0.)));}
  Require(beta_consistency<1e-13&&engine.max_shift_residual<1e-10,"evolution/fine shift derivatives inconsistent");
  // Pure stencil fixture with manufactured vector data (not scri gauge data):
  // the metric-rate response to b+Bx is exactly B+B^T−2I tr(B)/3. This checks
  // nonzero affine subtraction/restoration through the actual BoundaryRates.
  h::CompositeTensorEvolution stencil(mesh,true,1e-11,4,true,false,0,7,true,0,.24,false,true,true,false,.12,0,1,true);
  stencil.Initialize(0,false);const auto padded=stencil.Padded(stencil.state);stencil.GeometryJets(stencil.state,padded);
  const Real B[3][3]={{.12,.03,-.02},{-.04,.08,.01},{.02,-.03,-.05}};const Real tr=B[0][0]+B[1][1]+B[2][2];
  const auto datum=[&](const std::array<Real,3>&x){std::array<Real,3>v{.03,-.02,.01};for(int i=0;i<3;++i)for(int j=0;j<3;++j)v[i]+=B[i][j]*x[j];return v;};
  for(size_t p=0;p<stencil.jets.size();++p){const auto&x=stencil.shift.grid.geometry[p].x;const auto v=datum({x[0],x[1],x[2]});for(int i=0;i<3;++i)stencil.beta_delta[3*p+i]=v[i];}
  stencil.shift_affine.emplace(stencil.shift.affine_derivative->Fit(stencil.beta_delta));
  stencil.beta_padded=stencil.shift.affine_derivative->PaddedRemainder(stencil.beta_delta,*stencil.shift_affine,datum);
  for(size_t p=0;p<stencil.scri->rows.size();++p){const auto&x=stencil.scri->rows[p].geometry.x;const auto v=datum({x[0],x[1],x[2]});for(int i=0;i<3;++i)stencil.scri->beta[3*p+i]=v[i];}
  stencil.BoundaryRates(padded,.0001);Real surface_consistency=0;
  for(size_t p=0;p<stencil.scri->rows.size();++p)for(int t=0;t<6;++t){const int i=h::CompositeTensorEvolution::tensor_i[t],j=h::CompositeTensorEvolution::tensor_j[t];
    surface_consistency=std::max(surface_consistency,std::abs(stencil.boundary_rhs[22*p+1+t]-(B[i][j]+B[j][i]-(i==j?2*tr/3:0))));}
  Require(surface_consistency<1e-10,"surface/evolution affine derivative directions inconsistent");
  std::cout<<"{\"prototype_bitwise\":true,\"linearity_roundoff_max\":"<<linearity<<",\"minkowski_surface_rate_max\":"<<equilibrium
    <<",\"body_beta_derivative_consistency_max\":"<<beta_consistency<<",\"surface_affine_metric_rate_error_max\":"<<surface_consistency<<",\"original_vector_rms_max\":"<<engine.max_shift_residual
    <<",\"surface_pole\":"<<engine.boundary_pole<<",\"status\":\"passed\"}\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';code=1;}Kokkos::finalize();return code;}
