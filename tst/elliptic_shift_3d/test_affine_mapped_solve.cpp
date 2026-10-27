// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <iostream>
#include <iomanip>
#include "affine_mapped_operator.hpp"
#include "manufactured.hpp"
namespace h=z4c::hyperboloidal;
h::Z4cJet<Real> Make(const Real*x,int mode){auto u=manufactured::State(x,true);if(mode==2)return u;u.beta={};
  const Real b[3]={.03,-.02,.01},B[3][3]={{.12,.03,-.02},{-.04,.08,.01},{.02,-.03,-.05}};
  for(int v=0;v<3;++v){u.beta.value[v]=b[v];for(int d=0;d<3;++d){u.beta.value[v]+=B[v][d]*x[d];u.beta.d[d][v]=B[v][d];}}return u;}
int main(int argc,char**argv){Kokkos::initialize(argc,argv);int code=0;try{
  const int n=argc>1?std::stoi(argv[1]):6,shells=argc>2?std::stoi(argv[2]):2;
  if(n<6||n>10||(shells!=1&&shells!=2))throw std::invalid_argument("bounded affine solve probe");
  h::CompositeShiftStage stage(cubed_sphere::CompositeMesh(.25,shells==1?std::vector<Real>{1}:std::vector<Real>{.65,1},1,n,2,2),4,true,true,true);
  const auto&g=stage.grid;std::vector<h::MetricJet<Real>>metric;for(const auto&p:g.geometry)metric.push_back(manufactured::State(p.x,true).metric);stage.SetGeometry(metric);
  affine_mapped_probe::Operator modified(stage);
  for(int mode:{1,2}){multigrid::Vector source,exact;
    for(const auto&p:g.geometry){const auto u=Make(p.x,mode);Real result[3];h::ApplyEllipticShift(u.metric,h::Geometry(u.metric),u.beta,result);
      for(int v=0;v<3;++v){source.push_back(result[v]);exact.push_back(u.beta.value[v]);}}
    const auto boundary=[&](const std::array<Real,3>&x){const auto u=Make(x.data(),mode);return std::array<Real,3>{u.beta.value[0],u.beta.value[1],u.beta.value[2]};};
    for(bool corrected:{false,true}){multigrid::Vector solution(source.size(),0),lift,rhs(source.size()),zero(source.size(),0);
      auto apply=[&](const auto&a,auto&b,const h::CompositeShiftStage::Boundary&bc){if(corrected)modified.Apply(a,b,bc);else stage.Apply(a,b,bc);};
      apply(zero,lift,boundary);for(size_t i=0;i<rhs.size();++i)rhs[i]=-source[i]-lift[i];
      const auto certify=[&](const auto&x,const auto&){multigrid::Vector y,residual(source.size());apply(x,y,boundary);Real maximum=0;
        for(size_t i=0;i<y.size();++i){residual[i]=-source[i]-y[i];maximum=std::max(maximum,std::abs(residual[i]));}
        return std::sqrt(stage.Dot(residual,residual))<=1e-10&&maximum<=1e-7;};
      auto precondition=[&](const auto&a,auto&b){b.resize(a.size());for(int v=0;v<3;++v){multigrid::Vector scalar(g.active.size());
        for(size_t p=0;p<scalar.size();++p)scalar[p]=a[3*p+v];const auto c=stage.preconditioner.Cycle(scalar);
        for(size_t p=0;p<scalar.size();++p)b[3*p+v]=c[p];}};
      const auto report=multigrid::FlexibleGMRES([&](const auto&a,auto&b){apply(a,b,{});},precondition,
        [&](const auto&a,const auto&b){return stage.Dot(a,b);},rhs,solution,1e-10,0,160,24,certify);
      multigrid::Vector value,residual(source.size()),error(source.size()),old_value,old_residual(source.size());apply(solution,value,boundary);stage.Apply(solution,old_value,boundary);
      Real maximum=0;for(size_t i=0;i<value.size();++i){residual[i]=-source[i]-value[i];old_residual[i]=-source[i]-old_value[i];error[i]=solution[i]-exact[i];maximum=std::max(maximum,std::abs(error[i]));}
      if(!report.converged||!certify(solution,residual))throw std::runtime_error("original corrected tensor equation did not converge");
      std::cout<<std::setprecision(17)<<"{\"n\":"<<n<<",\"shells\":"<<shells<<",\"mode\":"<<mode<<",\"corrected\":"<<corrected
        <<",\"iterations\":"<<report.iterations<<",\"original_declared_residual_rms\":"<<std::sqrt(stage.Dot(residual,residual))
        <<",\"old_discretization_residual_rms\":"<<std::sqrt(stage.Dot(old_residual,old_residual))
        <<",\"solution_rms_error\":"<<std::sqrt(stage.Dot(error,error))<<",\"solution_max_error\":"<<maximum
        <<",\"production_enabled\":false,\"coupled_validation\":false}\n";
    }
  }
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';code=1;}Kokkos::finalize();return code;}
