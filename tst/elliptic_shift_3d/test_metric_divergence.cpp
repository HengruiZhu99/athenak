// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <complex>
#include <iomanip>
#include <iostream>
#include "z4c/hyperboloidal/composite_metric_divergence.hpp"
#include "manufactured.hpp"
namespace h=z4c::hyperboloidal;
using Prototype=h::CompositeMetricDivergence;
using Vector=Prototype::Vector;
using Tensor=Prototype::Tensor;
using Field=Prototype::TensorField;
void Check(bool condition,const char *message){if(!condition)throw std::runtime_error(message);}
Real Maximum(const Vector &q){Real m=0;for(Real x:q){Check(std::isfinite(x),"nonfinite prototype result");m=std::max(m,std::abs(x));}return m;}

// Deliberately nonunit determinant for the curved case, without making the
// metric conformally flat. It remains a gauge MMS, not Einstein initial data.
h::Z4cJet<Real> State(const Real x[3],bool curved){auto u=manufactured::State(x,curved);
 if(curved){using J=cubed_sphere::Jet;J y[3];for(int a=0;a<3;++a)y[a]=J::Variable(x[a],a);
  const auto scale=manufactured::Exp(J(.08)*y[0]+J(.03)*y[1]*y[2]);const auto old=u.metric;
  u.chi.value=scale.f;for(int a=0;a<3;++a){u.chi.d[a]=scale.d[a];for(int b=0;b<3;++b)u.chi.dd[a][b]=scale.h[a][b];}
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){u.metric.g[i][j]=scale.f*old.g[i][j];
   for(int a=0;a<3;++a){u.metric.dg[a][i][j]=scale.d[a]*old.g[i][j]+scale.f*old.dg[a][i][j];
    for(int b=0;b<3;++b)u.metric.ddg[a][b][i][j]=scale.h[a][b]*old.g[i][j]+scale.d[a]*old.dg[b][i][j]+scale.d[b]*old.dg[a][i][j]+scale.f*old.ddg[a][b][i][j];}}
  const auto rate=h::ShiftMetricRate(u.metric,u.beta);for(int i=0;i<3;++i)for(int j=0;j<3;++j){u.a.k[i][j]=rate.h[i][j]/2;
   for(int a=0;a<3;++a)u.a.dk[a][i][j]=rate.dh[a][i][j]/2;}
 }
 return u;
}
Tensor Metric(const Real x[3],bool curved){const auto u=State(x,curved);Tensor out;for(int f=0;f<6;++f)out[f]=u.metric.g[Prototype::ti[f]][Prototype::tj[f]];return out;}
h::VectorJet<Real> Background(const Real *x){h::VectorJet<Real> out{};for(int a=0;a<3;++a){out.value[a]=-x[a];out.d[a][a]=-1;}return out;}
struct Audit {int n=0;bool curved=false;std::string pair;Real continuum_rms=0,residual_rms=0,residual_max=0,bulk_max=0,interface_max=0,scri_max=0,balanced_max=0,frechet_max=0,constant_max=0,determinant_error=0;};
Audit MMS(const cubed_sphere::CompositeDifference &grid,bool curved,Prototype::Pair pair){
 Prototype op(grid,pair);Audit result;result.n=grid.n;result.curved=curved;result.pair=pair==Prototype::Pair::Centered?"centered_control":"forward_adjoint_candidate";
 const auto metric=op.SampleTensor([&](const auto &x){return Metric(x.data(),curved);});op.SetGeometry(metric);
 std::vector<h::Z4cJet<Real>> u;Vector beta(3*grid.active.size());
 for(size_t p=0;p<grid.active.size();++p){u.push_back(State(grid.geometry[p].x,curved));const auto &v=u.back();
  result.determinant_error=std::max(result.determinant_error,std::abs(h::Geometry(v.metric).determinant-1));
  for(int i=0;i<3;++i)beta[3*p+i]=v.beta.value[i]+grid.geometry[p].x[i];}
 const auto boundary=[](const std::array<Real,3>&){return std::array<Real,3>{};};
 const auto shift_rate=op.ShiftRate(beta,boundary,Background);
 const auto c=op.CurvatureRate(u,[&](const auto &x){const auto v=State(x.data(),curved);Tensor out;for(int f=0;f<6;++f)out[f]=-2*v.alpha.value*v.a.k[Prototype::ti[f]][Prototype::tj[f]];return out;});
 Vector applied,source;op.Variation(shift_rate,applied);op.Source(c,0,source);Real volume=0,e2=0,r2=0;
 for(size_t p=0;p<u.size();++p){Real exact[3];h::ApplyEllipticShift(u[p].metric,h::Geometry(u[p].metric),u[p].beta,exact);
  const Real w=grid.volumes[p]*std::sqrt(h::Geometry(u[p].metric).determinant)/std::pow(u[p].chi.value,1.5);volume+=w;
  const int n=grid.n,m=p/(n*n*n),i=p%n,j=p/n%n,k=p/(n*n)%n;const bool seam=i<4||j<4||k<4||i>=n-4||j>=n-4||k>=n-4;
  for(int v=0;v<3;++v){const Real error=applied[3*p+v]-exact[v],r=applied[3*p+v]-source[3*p+v];e2+=w*error*error;r2+=w*r*r;
   result.residual_max=std::max(result.residual_max,std::abs(r));if(grid.mesh.blocks[m].face<0)result.bulk_max=std::max(result.bulk_max,std::abs(r));
   if(seam)result.interface_max=std::max(result.interface_max,std::abs(r));
   if(grid.mesh.blocks[m].layer==int(grid.mesh.radii.size())-1&&i>=n-4)result.scri_max=std::max(result.scri_max,std::abs(r));}}
 result.continuum_rms=std::sqrt(e2/volume);result.residual_rms=std::sqrt(r2/volume);
 // Discrete balance is a diagnostic manufactured control, not an independent
 // continuum accuracy claim: C=-H at both active cells and actual face points.
 Field balanced=shift_rate;for(Real &x:balanced.active)x=-x;for(Real &x:balanced.surface)x=-x;Vector balanced_source;op.Source(balanced,0,balanced_source);
 for(size_t i=0;i<applied.size();++i)result.balanced_max=std::max(result.balanced_max,std::abs(applied[i]-balanced_source[i]));
 // Independent finite metric perturbation differentiates the DISCRETE V_D,
 // including its coefficient/halo interpolation and determinant changes.
 const auto rate=op.SampleTensor([](const auto &x){return Tensor{.03+.01*x[0],.004*x[1],.006*x[2],-.02+.008*x[2],.007*x[0],.01+.004*x[1]};});
 Vector variation;op.Variation(rate,variation);const Real step=2e-5;Field plus=metric,minus=metric;
 for(size_t i=0;i<metric.active.size();++i){plus.active[i]+=step*rate.active[i];minus.active[i]-=step*rate.active[i];}
 for(size_t i=0;i<metric.surface.size();++i){plus.surface[i]+=step*rate.surface[i];minus.surface[i]-=step*rate.surface[i];}
 op.SetGeometry(plus);const auto vp=op.Connection();op.SetGeometry(minus);const auto vm=op.Connection();
 for(size_t i=0;i<vp.size();++i)result.frechet_max=std::max(result.frechet_max,std::abs((vp[i]-vm[i])/(2*step)-variation[i]));
 op.SetGeometry(op.SampleTensor([](const auto&){return Tensor{1,0,0,1,0,1};}));Vector constant(3*grid.active.size());
 for(size_t p=0;p<grid.active.size();++p){constant[3*p]=.13;constant[3*p+1]=-.27;constant[3*p+2]=.41;}
 Vector constant_l;op.Apply(constant,constant_l,[](const auto&){return std::array<Real,3>{.13,-.27,.41};});result.constant_max=Maximum(constant_l);
 Check(result.frechet_max<3e-7,"discrete metric Frechet identity failed");Check(result.balanced_max<3e-10,"operator/source do not share the same discrete variation");Check(result.constant_max<1e-12,"constant shift was not preserved");return result;
}
struct Checker {Real centered=0,biased=0,minimum_symbol_coercivity=1e30;size_t interior_cells=0;};
Checker Checkerboard(const cubed_sphere::CompositeDifference &grid){Checker out;Check(grid.n>=12,"checker audit needs four-cell interior stencil");Vector beta(3*grid.active.size(),0);
 for(size_t p=0;p<grid.active.size();++p)beta[3*p]=p%grid.n%2?-1:1;
 for(auto pair:{Prototype::Pair::Centered,Prototype::Pair::ForwardAdjoint}){Prototype op(grid,pair);op.SetGeometry(op.SampleTensor([](const auto&){return Tensor{1,0,0,1,0,1};}));Vector result;op.Apply(beta,result);
  for(size_t p=0;p<grid.active.size();++p){const int n=grid.n,m=p/(n*n*n),i=p%n,j=p/n%n,k=p/(n*n)%n;
   if(grid.mesh.blocks[m].face>=0||i<4||j<4||k<4||i>=n-4||j>=n-4||k>=n-4)continue;
   if(pair==Prototype::Pair::Centered){++out.interior_cells;out.centered=std::max(out.centered,std::abs(result[3*p]));}
   else out.biased=std::max(out.biased,std::abs(result[3*p]));}}
 using C=std::complex<Real>;const Real pi=std::acos(Real(-1));const Real weights[5]={-25./12,4,-3,4./3,-.25};
 for(const auto &theta:{std::array<Real,3>{pi,0,0},std::array<Real,3>{pi,pi,pi},std::array<Real,3>{.23,1.1,2.4}}){C d[3];Real norm=0;
  for(int a=0;a<3;++a){for(int k=1;k<=4;++k)d[a]+=weights[k]*(std::exp(C(0,k*theta[a]))-C(1));norm+=std::norm(d[a]);}
  for(int seed=0;seed<4;++seed){C b[3];Real bnorm=0;C dagger=0,transpose=0;
   for(int a=0;a<3;++a){b[a]=C(.3*(a+1)+.17*seed,.13*(seed+1)*(a-1));bnorm+=std::norm(b[a]);dagger+=std::conj(d[a])*b[a];transpose+=d[a]*b[a];}
   C energy=0;for(int a=0;a<3;++a){const C l=-norm*b[a]-d[a]*dagger+Real(2./3)*std::conj(d[a])*transpose;energy+=std::conj(b[a])*l;}
   out.minimum_symbol_coercivity=std::min(out.minimum_symbol_coercivity,-energy.real()/(norm*bnorm));}}
 Check(out.interior_cells>0&&out.centered<1e-10,"centered checkerboard null was not exposed");Check(out.biased>100,"biased pair did not remove flat checkerboard null");
 Check(out.minimum_symbol_coercivity>=Real(1./3)-1e-12,"biased flat symbol violates coercive bound");return out;
}
void Print(const Audit &q){std::cout<<"{\"n\":"<<q.n<<",\"curved_nonunit_determinant\":"<<q.curved<<",\"pair\":\""<<q.pair<<"\",\"continuum_operator_rms\":"<<q.continuum_rms
 <<",\"continuum_MMS_residual_rms\":"<<q.residual_rms<<",\"residual_max_all\":"<<q.residual_max<<",\"bulk_max\":"<<q.bulk_max<<",\"interface_max\":"<<q.interface_max<<",\"scri_max\":"<<q.scri_max
 <<",\"discrete_balanced_residual_max\":"<<q.balanced_max<<",\"finite_metric_Frechet_error\":"<<q.frechet_max<<",\"constant_shift_error\":"<<q.constant_max<<",\"determinant_departure\":"<<q.determinant_error<<"}";}
int main(int argc,char **argv){Kokkos::initialize(argc,argv);int status=0;try{
 std::vector<Audit> results;Checker checker;
 for(int n:{8,12}){cubed_sphere::CompositeDifference grid(cubed_sphere::CompositeMesh(.25,{1},1,n,4));
  for(bool curved:{false,true})for(auto pair:{Prototype::Pair::Centered,Prototype::Pair::ForwardAdjoint})results.push_back(MMS(grid,curved,pair));
  if(n==12)checker=Checkerboard(grid);}
 std::cout<<std::setprecision(17)<<"{\"prototype_not_admitted_for_evolution\":true,\"ghosts\":4,\"cases\":[";
 for(size_t i=0;i<results.size();++i){if(i)std::cout<<",";Print(results[i]);}
 std::cout<<"],\"observed_MMS_residual_orders\":[";
 for(size_t i=0;i<4;++i){if(i)std::cout<<",";std::cout<<std::log(results[i].residual_rms/results[4+i].residual_rms)/std::log(Real(12.)/8);}
 std::cout<<"],\"checkerboard\":{\"interior_cells\":"<<checker.interior_cells<<",\"centered_response\":"<<checker.centered<<",\"biased_response\":"<<checker.biased
  <<",\"biased_flat_symbol_min_coercivity\":"<<checker.minimum_symbol_coercivity<<"},\"scope\":\"same full discrete DV for source/operator; continuum error retained; no native integration or coupled success claim\"}\n";
}catch(const std::exception &e){std::cerr<<e.what()<<"\n";status=1;}Kokkos::finalize();return status;}
