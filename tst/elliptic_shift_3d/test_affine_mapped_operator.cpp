// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <iostream>
#include <iomanip>
#include "affine_mapped_operator.hpp"
#include "manufactured.hpp"
namespace h=z4c::hyperboloidal;
h::Z4cJet<Real> State(const Real*x,bool curved,int mode){auto u=manufactured::State(x,curved);if(mode==2)return u;
  u.beta={};const Real b[3]={.03,-.02,.01},B[3][3]={{.12,.03,-.02},{-.04,.08,.01},{.02,-.03,-.05}};
  for(int v=0;v<3;++v){u.beta.value[v]=b[v];if(mode)for(int d=0;d<3;++d){u.beta.value[v]+=B[v][d]*x[d];u.beta.d[d][v]=B[v][d];}}return u;}
int main(int argc,char**argv){Kokkos::initialize(argc,argv);int code=0;try{
  const int n=argc>1?std::stoi(argv[1]):6,shells=argc>2?std::stoi(argv[2]):2;
  if(n<6||n>10||(shells!=1&&shells!=2))throw std::invalid_argument("bounded affine probe");
  for(bool curved:{false,true}){h::CompositeShiftStage stage(cubed_sphere::CompositeMesh(.25,shells==1?std::vector<Real>{1}:std::vector<Real>{.65,1},1,n,2,2),4,true,true,true);
    const auto&g=stage.grid;std::vector<h::MetricJet<Real>>metric;for(const auto&p:g.geometry)metric.push_back(manufactured::State(p.x,curved).metric);stage.SetGeometry(metric);
    affine_mapped_probe::Operator corrected(stage);
    for(int mode=0;mode<3;++mode){multigrid::Vector beta,truth,old,fresh;
      for(const auto&p:g.geometry){const auto u=State(p.x,curved,mode);Real result[3];h::ApplyEllipticShift(u.metric,h::Geometry(u.metric),u.beta,result);
        for(int v=0;v<3;++v){beta.push_back(u.beta.value[v]);truth.push_back(-result[v]);}}
      const auto boundary=[&](const std::array<Real,3>&x){const auto u=State(x.data(),curved,mode);return std::array<Real,3>{u.beta.value[0],u.beta.value[1],u.beta.value[2]};};
      stage.Apply(beta,old,boundary);corrected.Apply(beta,fresh,boundary);
      Real old_max=0,new_max=0,old2=0,new2=0,volume=0,source=0;
      for(size_t p=0;p<g.active.size();++p){volume+=3*g.volumes[p];for(int v=0;v<3;++v){const size_t i=3*p+v;const Real a=old[i]-truth[i],b=fresh[i]-truth[i];
        old_max=std::max(old_max,std::abs(a));new_max=std::max(new_max,std::abs(b));old2+=g.volumes[p]*a*a;new2+=g.volumes[p]*b*b;source=std::max(source,std::abs(truth[i]));}}
      if(mode<2&&new_max>1e-9)throw std::runtime_error("affine correction does not reproduce continuum operator");
      if(mode==1&&curved&&source<1e-4)throw std::runtime_error("curved affine source was incorrectly dropped");
      std::cout<<std::setprecision(17)<<"{\"n\":"<<n<<",\"shells\":"<<shells<<",\"curved\":"<<curved<<",\"mode\":"<<mode<<",\"cells\":"<<g.active.size()
        <<",\"old_max\":"<<old_max<<",\"new_max\":"<<new_max<<",\"old_rms\":"<<std::sqrt(old2/volume)<<",\"new_rms\":"<<std::sqrt(new2/volume)
        <<",\"continuum_source_max\":"<<source<<",\"projection_gram_condition\":"<<corrected.condition<<",\"production_enabled\":false}\n";
    }
  }
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';code=1;}Kokkos::finalize();return code;}
