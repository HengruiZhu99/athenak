// Native field-slot binding for the isolated physical interface operator.
#pragma once
#include "goal1_physical_interface.hpp"
#include "mesh/cubed_sphere/conformal_bridge.hpp"
namespace cubed_sphere {namespace research_interface {
namespace hyp=z4c::hyperboloidal;
inline void StoreScalar(hyp::Z4cJet<Real>&u,int n,const hyp::ScalarJet<Real>&s) {
 if(n==0){u.chi=s;return;}if(n==7){u.trace=s;return;}if(n==17){u.theta=s;return;}if(n==18){u.alpha=s;return;}
 const int a[6]={0,0,0,1,1,2},b[6]={0,1,2,1,2,2};
 if(n>=1&&n<=6){int i=a[n-1],j=b[n-1];u.metric.g[i][j]=u.metric.g[j][i]=s.value;
  for(int d=0;d<3;++d){u.metric.dg[d][i][j]=u.metric.dg[d][j][i]=s.d[d];for(int e=0;e<3;++e)u.metric.ddg[d][e][i][j]=u.metric.ddg[d][e][j][i]=s.dd[d][e];}return;}
 if(n>=8&&n<=13){int i=a[n-8],j=b[n-8];u.a.k[i][j]=u.a.k[j][i]=s.value;for(int d=0;d<3;++d)u.a.dk[d][i][j]=u.a.dk[d][j][i]=s.d[d];return;}
 bool shift=n>=19;auto &v=shift?u.beta:u.lambda;int i=n-(shift?19:14);v.value[i]=s.value;
 for(int d=0;d<3;++d){v.d[d][i]=s.d[d];for(int e=0;e<3;++e)v.dd[d][e][i]=s.dd[d][e];}
}
template<class Pack>void RefreshNative(Pack (&packs)[2],const Physical&op) {
 const int N=op.points.size();std::vector<std::vector<double>> data(22,std::vector<double>(N));int first=0;
 for(auto &p:packs){auto dev=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),p.dev);
  for(size_t l=0;l<p.Cells().size();++l){const auto&c=p.Cells()[l];const int index=first+l;
   if(index>=N)throw std::runtime_error("physical native ordering overflow");
   for(int a=0;a<3;++a)if(std::abs(c.geometry.x[a]-op.points[index].x[a])>1e-12)throw std::runtime_error("physical native geometry ordering mismatch");
   for(int n=0;n<22;++n)data[n][index]=dev(c.m,n,c.k,c.j,c.i);
  }first+=p.Cells().size();
 }if(first!=N)throw std::runtime_error("physical native ordering incomplete");
 std::vector<hyp::Z4cJet<Real>> jets(N);
 for(int n=0;n<22;++n){std::array<std::vector<double>,3> df;std::array<std::vector<double>,9> dd;
  for(int d=0;d<3;++d)df[d]=op.D[d].Apply(data[n]);bool second=n<8||n>13;
  if(second)for(int a=0;a<3;++a)for(int b=0;b<3;++b)dd[3*a+b]=op.D[a].Apply(df[b]);
  for(int i=0;i<N;++i){hyp::ScalarJet<Real>s{};s.value=data[n][i];for(int a=0;a<3;++a){s.d[a]=df[a][i];if(second)for(int b=0;b<3;++b)s.dd[a][b]=.5*(dd[3*a+b][i]+dd[3*b+a][i]);}StoreScalar(jets[i],n,s);}
 }
 first=0;for(auto&p:packs){auto host=Kokkos::create_mirror_view(p.physical_jets);for(size_t i=0;i<p.Cells().size();++i)host(i)=jets[first+i];Kokkos::deep_copy(p.physical_jets,host);first+=p.Cells().size();}
}
}}
