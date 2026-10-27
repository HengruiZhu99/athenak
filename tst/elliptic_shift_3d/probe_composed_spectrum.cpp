// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Finite trial-space singular-value controls, not a full spectral-gap proof.
#include <iomanip>
#include <iostream>
#include "composed_metric_operator_probe.hpp"
namespace h=z4c::hyperboloidal;namespace c=cubed_sphere;namespace p=composed_metric_probe;
int main(int argc,char**argv){Kokkos::initialize(argc,argv);int status=0;try{
 int n=argc>1?std::stoi(argv[1]):6,shells=argc>2?std::stoi(argv[2]):7,refine=argc>3?std::stoi(argv[3]):0;if(n<6||n>10||(shells!=1&&shells!=2&&shells!=7)||refine<0||refine>1)throw std::invalid_argument("spectrum probe budget");
 c::CompositeMesh mesh(.25,shells==1?p::V{1}:shells==2?p::V{.65,1}:p::V{.45,.55,.60,.65,.70,.75,1},1,n,2,2);if(refine)mesh.RefineCartesian(0);h::CompositeShiftStage stage(mesh,4,true,true,true,false);const auto&g=stage.grid;c::ScriSurface scri(g,6,7);std::vector<p::U>bulk(g.active.size()),edge(scri.rows.size());std::vector<h::MetricJet<Real>>metrics;
 for(auto&u:bulk){u.chi.value=u.alpha.value=1;for(int i=0;i<3;++i)u.metric.g[i][i]=1;metrics.push_back(u.metric);}for(auto&u:edge){u.chi.value=u.alpha.value=1;for(int i=0;i<3;++i)u.metric.g[i][i]=1;}stage.SetGeometry(metrics);p::Operator op(stage,scri,bulk,edge);
 std::vector<p::V>basis,direct,composed,right_direct,right_composed;std::vector<std::string>names;Real orthogonality=0;
 auto add=[&](std::string name,int family,int mask,int component){p::V q(3*bulk.size(),0);for(size_t k=0;k<bulk.size();++k){const auto&x=g.geometry[k].x;Real r=0;for(int i=0;i<3;++i)r+=x[i]*x[i];r=std::sqrt(r);const int i=k%n,j=k/n%n,l=k/(n*n)%n;const bool cart=g.mesh.blocks[k/(n*n*n)].face<0;Real value=0,envelope=std::pow(std::max(Real(0),1-r*r),3);
   if(family==0)value=envelope*std::sin(3.141592653589793*x[0])*(1+.2*x[1]-.1*x[2]);
   if(family==1)value=envelope*std::exp(-r*r/.04);
   if(family==2)value=envelope*(((mask&1?i:0)+(mask&2?j:0)+(mask&4?l:0))%2?1:-1);
   if(family==3&&cart){value=1;for(int a=0;a<3;++a)value*=std::pow(std::cos(2*3.141592653589793*x[a]),6);value*=(((mask&1?i:0)+(mask&2?j:0)+(mask&4?l:0))%2?1:-1);}
   if(family==4&&r>.65&&r<.70)value=std::pow(std::sin(3.141592653589793*(r-.65)/.05),6)*(mask?(i%2?1:-1):1);
   q[3*k+component]=value;}
  const Real initial=std::sqrt(stage.Dot(q,q));if(initial<1e-14)return;
  for(int pass=0;pass<2;++pass)for(const auto&b:basis){Real projection=stage.Dot(q,b);for(size_t k=0;k<q.size();++k)q[k]-=projection*b[k];}const Real length=std::sqrt(stage.Dot(q,q));if(length<initial*1e-9)return;for(auto&x:q)x/=length;
  p::V a,b,pre(q.size()),pa,pb;stage.Apply(q,a);op.Apply(q,b);
  for(int v=0;v<3;++v){p::V scalar(bulk.size());for(size_t k=0;k<bulk.size();++k)scalar[k]=q[3*k+v];const auto correction=stage.preconditioner.Cycle(scalar);for(size_t k=0;k<bulk.size();++k)pre[3*k+v]=correction[k];}
  stage.Apply(pre,pa);op.Apply(pre,pb);right_direct.push_back(std::move(pa));right_composed.push_back(std::move(pb));names.push_back(name+"_"+std::to_string(component));basis.push_back(std::move(q));direct.push_back(std::move(a));composed.push_back(std::move(b));};
 for(int component=0;component<3;++component){add("smooth",0,0,component);add("gaussian",1,0,component);for(int mask=1;mask<8;++mask)add("global_parity"+std::to_string(mask),2,mask,component);for(int mask:{1,3,7})add("cart_parity"+std::to_string(mask),3,mask,component);for(int mask:{0,1})add("thin_shell"+std::to_string(mask),4,mask,component);}
 for(size_t i=0;i<basis.size();++i)for(size_t j=0;j<basis.size();++j)orthogonality=std::max(orthogonality,std::abs(stage.Dot(basis[i],basis[j])-(i==j?1:0)));
 std::cout<<std::setprecision(17)<<"{\"n\":"<<n<<",\"shells\":"<<shells<<",\"refined\":"<<refine<<",\"cells\":"<<bulk.size()<<",\"trial_modes\":"<<basis.size()<<",\"basis_orthogonality\":"<<orthogonality<<",\"names\":[";for(size_t i=0;i<names.size();++i)std::cout<<(i?",":"")<<'"'<<names[i]<<'"';std::cout<<"],\"normal_gram\":{";
 for(int which=0;which<4;++which){const auto&a=which==0?direct:which==1?composed:which==2?right_direct:right_composed;std::cout<<(which?",":"")<<'"'<<(which==0?"expanded":which==1?"composed":which==2?"expanded_rightMG":"composed_rightMG")<<"\":[";for(size_t i=0;i<a.size();++i){std::cout<<(i?",":"")<<'[';for(size_t j=0;j<a.size();++j)std::cout<<(j?",":"")<<stage.Dot(a[i],a[j]);std::cout<<']';}std::cout<<']';}
 std::cout<<"},\"restricted_only\":true,\"spectral_gap_proved\":false,\"production_adopted\":false}\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';status=2;}Kokkos::finalize();return status;}
