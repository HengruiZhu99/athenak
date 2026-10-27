// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <iomanip>
#include <iostream>
#include "composed_metric_operator_probe.hpp"
#include "manufactured.hpp"
namespace h=z4c::hyperboloidal;namespace p=composed_metric_probe;
h::Z4cJet<Real> Data(const Real*x,int mode){auto u=manufactured::State(x,true);if(mode==2)return u;u.beta={};const Real b[3]={.03,-.02,.01},B[3][3]={{.12,.03,-.02},{-.04,.08,.01},{.02,-.03,-.05}};for(int i=0;i<3;++i){u.beta.value[i]=b[i];for(int j=0;j<3;++j){u.beta.value[i]+=B[i][j]*x[j];u.beta.d[j][i]=B[i][j];}}return u;}
int main(int argc,char**argv){Kokkos::initialize(argc,argv);int code=0;try{
 const int n=argc>1?std::stoi(argv[1]):6;const bool affine=argc>2?std::stoi(argv[2]):false;if(n<6||n>10)throw std::invalid_argument("composed probe budget");
 h::CompositeShiftStage stage(cubed_sphere::CompositeMesh(.25,{.65,1},1,n,2,2),4,true,true,true,affine);const auto&g=stage.grid;cubed_sphere::ScriSurface surface(g,6,7);std::vector<p::U>bulk,edge;std::vector<h::MetricJet<Real>>metrics;std::vector<Real>chi;
 for(const auto&q:g.geometry){auto u=Data(q.x,2);bulk.push_back(u);metrics.push_back(u.metric);chi.push_back(u.chi.value);}for(const auto&q:surface.rows)edge.push_back(Data(q.geometry.x,2));stage.SetGeometry(metrics,chi);p::Operator candidate(stage,surface,bulk,edge);
 p::V checker(3*bulk.size(),0),expanded,composed;for(size_t k=0;k<bulk.size();++k){const int i=k%n,j=k/n%n,l=k/(n*n)%n;checker[3*k]=((i+j+l)%2?1:-1);}stage.Apply(checker,expanded);candidate.Apply(checker,composed);
 std::cout<<std::setprecision(17)<<"{\"kind\":\"mapped_checkerboard\",\"n\":"<<n<<",\"affine\":"<<affine<<",\"expanded_norm\":"<<std::sqrt(stage.Dot(expanded,expanded))<<",\"composed_norm\":"<<std::sqrt(stage.Dot(composed,composed))<<",\"no_periodic_nullspace_exclusion_claim\":true}\n";
 for(int mode:{1,2}){p::V exact,source;for(const auto&q:g.geometry){const auto u=Data(q.x,mode);Real L[3];h::ApplyEllipticShift(u.metric,h::Geometry(u.metric),u.beta,L);for(int v=0;v<3;++v){exact.push_back(u.beta.value[v]);source.push_back(L[v]);}}
  const auto bc=[&](const std::array<Real,3>&x){const auto u=Data(x.data(),mode);return std::array<Real,3>{u.beta.value[0],u.beta.value[1],u.beta.value[2]};};
  p::V zero(exact.size(),0),lift,rhs(exact.size()),solution(exact.size(),0);candidate.Apply(zero,lift,bc);for(size_t k=0;k<rhs.size();++k)rhs[k]=-source[k]-lift[k];
  auto norm=[&](const auto&a,const auto&b){return stage.Dot(a,b);};auto pre=[&](const auto&a,auto&b){b.resize(a.size());for(int v=0;v<3;++v){p::V scalar(bulk.size());for(size_t k=0;k<bulk.size();++k)scalar[k]=a[3*k+v];const auto c=stage.preconditioner.Cycle(scalar);for(size_t k=0;k<bulk.size();++k)b[3*k+v]=c[k];}};
  auto certify=[&](const auto&a,const auto&){p::V y,r(a.size());candidate.Apply(a,y,bc);Real maximum=0;for(size_t k=0;k<a.size();++k){r[k]=-source[k]-y[k];maximum=std::max(maximum,std::abs(r[k]));}return std::sqrt(norm(r,r))<=1e-10&&maximum<=1e-7;};
  const auto report=multigrid::FlexibleGMRES([&](const auto&a,auto&b){candidate.Apply(a,b);},pre,norm,rhs,solution,1e-10,0,120,48,certify);
  p::V y,error(solution.size()),residual(solution.size());candidate.Apply(solution,y,bc);Real maxerr=0,maxres=0;for(size_t k=0;k<y.size();++k){error[k]=solution[k]-exact[k];residual[k]=-source[k]-y[k];maxerr=std::max(maxerr,std::abs(error[k]));maxres=std::max(maxres,std::abs(residual[k]));}
  std::cout<<"{\"kind\":\"continuum_mms\",\"n\":"<<n<<",\"affine\":"<<affine<<",\"mode\":"<<mode<<",\"iterations\":"<<report.iterations<<",\"converged\":"<<(report.converged&&certify(solution,residual))<<",\"original_declared_rms\":"<<std::sqrt(norm(residual,residual))<<",\"original_declared_point\":"<<maxres<<",\"solution_error_rms\":"<<std::sqrt(norm(error,error))<<",\"solution_error_max\":"<<maxerr<<",\"production_adopted\":false}\n";
  if(!report.converged||!certify(solution,residual))code=1;
 }
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';code=2;}Kokkos::finalize();return code;}
